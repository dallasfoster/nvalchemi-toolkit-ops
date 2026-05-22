# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch bindings for single-system cluster-pair tile neighbor list.

Mirrors the ``cell_list`` / ``naive`` wrapper pattern: exposes low-level
component ops (``build_tile_neighbor_list``, ``tile_to_matrix``,
``tile_to_coo``) that fill pre-allocated tensors, plus a high-level
convenience entry point ``tile_neighbor_list``.  All torch-side work
(Morton sort, allocation, ``wp.from_torch`` conversion) lives in this
module; the Warp layer at ``nvalchemiops.neighbors.tile_warp`` only
sees ``wp.array`` inputs.

Scope: single system, orthorhombic or triclinic PBC, float32, any
``N >= 0`` (the wrapper pads internally to a multiple of TILE_GROUP_SIZE).
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.neighbors.neighbor_utils import (
    fill_neighbor_matrix_tail as wp_fill_neighbor_matrix_tail,
)
from nvalchemiops.neighbors.tile_warp import (
    TILE_GROUP_SIZE,
)
from nvalchemiops.neighbors.tile_warp import (
    build_tile_neighbor_list as wp_build_tile_neighbor_list,
)
from nvalchemiops.neighbors.tile_warp import (
    tile_to_coo as wp_tile_to_coo,
)
from nvalchemiops.neighbors.tile_warp import (
    tile_to_matrix as wp_tile_to_matrix,
)
from nvalchemiops.torch.types import get_wp_dtype

__all__ = [
    "TILE_GROUP_SIZE",
    "estimate_tile_neighbor_list_sizes",
    "allocate_tile_neighbor_list",
    "build_tile_neighbor_list",
    "tile_to_matrix",
    "tile_to_coo",
    "tile_neighbor_list",
]


# =============================================================================
# Sizing + allocation helpers (torch-side, not ``custom_op``-wrapped)
# =============================================================================
def estimate_tile_neighbor_list_sizes(
    total_atoms: int,
    max_tiles_per_group: int = 256,
) -> tuple[int, int, int, int]:
    """Estimate allocation sizes for the tile neighbor list state.

    Any ``total_atoms >= 0`` is accepted; internally the state is sized
    at ``n_padded = ceil(total_atoms / TILE_GROUP_SIZE) * TILE_GROUP_SIZE`` so
    the kernels see a 32-aligned layout.  Padding slots receive a
    sentinel max Morton code (see ``_compute_morton_kernel``) so they
    sort to the end and are dropped by the convert/coo kernels'
    ``i_sorted < natom`` filter.

    Parameters
    ----------
    total_atoms : int
        Real atom count.
    max_tiles_per_group : int, default 256
        Upper bound on neighbor groups per row_group (dense-cutoff cap).

    Returns
    -------
    n_padded : int
        Padded atom count = ``ceil(total_atoms / TILE_GROUP_SIZE) * TILE_GROUP_SIZE``.
        Used to size positions / sorted SoA / sorted_atom_index scratch arrays.
    ngroup : int
        Number of 32-atom groups: ``n_padded // TILE_GROUP_SIZE``.
    ngroup_padded : int
        Group-array pad length for in-bounds ``wp.tile_load`` at any
        TILE-aligned offset.  Multiple of ``TILE_GROUP_SIZE``; at least one
        TILE slack over ``ngroup``.
    max_tiles : int
        Upper bound on the tile-pair list size.
    """
    if total_atoms < 0:
        raise ValueError(f"total_atoms must be >= 0; got {total_atoms}")
    n_padded = (
        (total_atoms + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    if n_padded == 0:
        n_padded = TILE_GROUP_SIZE  # always reserve at least one tile
    ngroup = n_padded // TILE_GROUP_SIZE
    ngroup_padded = (
        (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    if ngroup_padded == ngroup:
        ngroup_padded = ngroup + TILE_GROUP_SIZE
    max_tiles = ngroup * min(ngroup, max_tiles_per_group)
    return n_padded, ngroup, ngroup_padded, max_tiles


def allocate_tile_neighbor_list(
    total_atoms: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    max_tiles_per_group: int = 256,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Allocate all state tensors consumed by ``build_tile_neighbor_list``.

    Returns ``(sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y,
    sorted_pos_z, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x,
    group_ext_y, group_ext_z, num_tiles, tile_row_group, tile_col_group)``.
    """
    n_padded, ngroup, ngroup_padded, max_tiles = estimate_tile_neighbor_list_sizes(
        total_atoms,
        max_tiles_per_group=max_tiles_per_group,
    )
    # Scratch arrays sized at the padded layout so non-32-aligned
    # ``total_atoms`` is handled inside ``_build_tile_neighbor_list_op``.
    # Padding slots are populated with sentinel Morton codes (see
    # ``_compute_morton_kernel``) and dropped by the convert/coo
    # kernels' ``i_sorted < natom`` filter.
    sorted_atom_index = torch.empty(n_padded, dtype=torch.int32, device=device)
    morton_codes = torch.empty(n_padded, dtype=torch.int32, device=device)
    sorted_pos_x = torch.empty(n_padded, dtype=dtype, device=device)
    sorted_pos_y = torch.empty(n_padded, dtype=dtype, device=device)
    sorted_pos_z = torch.empty(n_padded, dtype=dtype, device=device)
    group_ctr_x = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ctr_y = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ctr_z = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_x = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_y = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_z = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    num_tiles = torch.zeros(1, dtype=torch.int32, device=device)
    tile_row_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    tile_col_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    return (
        sorted_atom_index,
        morton_codes,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        group_ctr_x,
        group_ctr_y,
        group_ctr_z,
        group_ext_x,
        group_ext_y,
        group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
    )


# =============================================================================
# Internal helpers
# =============================================================================
def _cell_invcell_from_cell(
    cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a ``(3, 3)`` or ``(1, 3, 3)`` cell to ``(cell_3x3,
    inv_cell_3x3)`` torch tensors for the Warp kernels.

    Accepts any non-degenerate cell (orthorhombic or triclinic).
    The downstream Warp kernels use ``_wrap_triclinic`` which is a
    strict superset of the old orthorhombic-only path.
    """
    if cell.ndim == 3:
        if cell.shape[0] != 1:
            raise ValueError(
                f"single-system tile_warp expects (1, 3, 3) cell; got {tuple(cell.shape)}"
            )
        cell_mat = cell[0]
    elif cell.ndim == 2:
        if cell.shape != (3, 3):
            raise ValueError(
                f"cell must be (3, 3) or (1, 3, 3); got {tuple(cell.shape)}"
            )
        cell_mat = cell
    else:
        raise ValueError(f"cell must be (3, 3) or (1, 3, 3); got {tuple(cell.shape)}")
    cell_mat = cell_mat.contiguous()
    inv_cell_mat = torch.linalg.inv(cell_mat).contiguous()
    return cell_mat, inv_cell_mat


def _mat33f_from_torch(mat: torch.Tensor):
    """Zero-copy view a ``(1, 3, 3)`` or ``(3, 3)`` torch tensor as a
    ``wp.array(dtype=wp.mat33f, shape=(1,))``.

    The tile_warp warp kernels read the cell / inv_cell as length-1
    ``wp.array(dtype=wp.mat33f)`` and dereference ``cell[0]`` inside the
    kernel body, matching the cell_list pattern.  This avoids a per-call
    host sync.
    """
    if mat.ndim == 2:
        mat = mat.unsqueeze(0)
    return wp.from_torch(
        mat.detach().contiguous().to(torch.float32),
        dtype=wp.mat33f,
        return_ctype=True,
    )


# =============================================================================
# Component ops (torch.library.custom_op wrappers)
# =============================================================================
@torch.library.custom_op(
    "nvalchemiops::_build_tile_neighbor_list",
    mutates_args=(
        "sorted_atom_index",
        "morton_codes",
        "sorted_pos_x",
        "sorted_pos_y",
        "sorted_pos_z",
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "num_tiles",
        "tile_row_group",
        "tile_col_group",
    ),
)
def _build_tile_neighbor_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    morton_codes: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
) -> None:
    """Compute Morton codes + argsort + SoA gather in torch, then run
    bbox reduction + tile enumeration on the warp side.

    Triclinic-safe via the (cell, inv_cell) pair; orthorhombic cells
    just have a diagonal inv_cell and produce the same result as the
    pre-triclinic implementation.

    See Also
    --------
    nvalchemiops.neighbors.tile_warp.build_tile_neighbor_list : warp launcher
    """
    N = positions.shape[0]
    if N == 0:
        return
    device = positions.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(positions.dtype)

    num_tiles.zero_()

    wp_cell = _mat33f_from_torch(cell)
    wp_inv_cell = _mat33f_from_torch(inv_cell)

    # ---- Steps 1-3: Morton codes + argsort + gather (torch-side) ----
    # Fractional coords via inv_cell; wrap into [0, 1) so the bucket
    # produces a deterministic 30-bit Morton code.  Orthorhombic cells
    # collapse this back to the cheaper diagonal multiply.
    #
    # Pad to ``n_padded = ceil(N / TILE_GROUP_SIZE) * TILE_GROUP_SIZE`` so the
    # kernels see a 32-aligned layout.  Padding slots get a sentinel
    # max 30-bit Morton code (0x3FFFFFFF) so radix sort places them at
    # the end of the sorted layout, where the convert/coo kernels'
    # ``i_sorted < natom`` filter naturally drops any pair involving
    # them.  Padding positions copy the last real atom (any in-cell
    # position is safe) so the SoA gather has well-defined reads.
    n_padded = int(sorted_atom_index.shape[0])
    if N < n_padded:
        padded_positions = torch.empty(
            (n_padded, 3),
            dtype=positions.dtype,
            device=positions.device,
        )
        padded_positions[:N].copy_(positions)
        padded_positions[N:].copy_(positions[-1:].expand(n_padded - N, 3))
    else:
        padded_positions = positions

    frac = padded_positions @ inv_cell.T
    frac = frac - torch.floor(frac)
    bucket = (frac * 1024.0).clamp(0, 1023).to(torch.int32)
    ix, iy, iz = bucket.unbind(dim=-1)

    def _spread(x: torch.Tensor) -> torch.Tensor:
        x = x & 0x3FF
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        x = (x | (x << 2)) & 0x09249249
        return x

    codes32 = (_spread(iz) << 2) | (_spread(iy) << 1) | _spread(ix)
    if N < n_padded:
        # Sentinel: one bit above any real 30-bit Morton code.  Real
        # codes saturate at 0x3FFFFFFF; 0x40000000 sorts after all of
        # them.  See ``_compute_morton_kernel`` for the matching
        # warp-side value.
        codes32[N:] = 0x40000000
    morton_codes.copy_(codes32)
    perm = torch.argsort(codes32)
    sorted_atom_index.copy_(perm.to(torch.int32))
    sorted_pos = padded_positions[perm].contiguous()
    sorted_pos_x.copy_(sorted_pos[:, 0].contiguous())
    sorted_pos_y.copy_(sorted_pos[:, 1].contiguous())
    sorted_pos_z.copy_(sorted_pos[:, 2].contiguous())

    # ---- Step 4: rank2group + group2tile (warp launcher) ----
    wp_sorted_pos_x = wp.from_torch(
        sorted_pos_x,
        dtype=wp_dtype,
        return_ctype=True,
    )
    wp_sorted_pos_y = wp.from_torch(
        sorted_pos_y,
        dtype=wp_dtype,
        return_ctype=True,
    )
    wp_sorted_pos_z = wp.from_torch(
        sorted_pos_z,
        dtype=wp_dtype,
        return_ctype=True,
    )
    wp_group_ctr_x = wp.from_torch(group_ctr_x, dtype=wp_dtype, return_ctype=True)
    wp_group_ctr_y = wp.from_torch(group_ctr_y, dtype=wp_dtype, return_ctype=True)
    wp_group_ctr_z = wp.from_torch(group_ctr_z, dtype=wp_dtype, return_ctype=True)
    wp_group_ext_x = wp.from_torch(group_ext_x, dtype=wp_dtype, return_ctype=True)
    wp_group_ext_y = wp.from_torch(group_ext_y, dtype=wp_dtype, return_ctype=True)
    wp_group_ext_z = wp.from_torch(group_ext_z, dtype=wp_dtype, return_ctype=True)
    wp_num_tiles = wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True)
    wp_tile_row_group = wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True)
    wp_tile_col_group = wp.from_torch(
        tile_col_group,
        dtype=wp.int32,
        return_ctype=True,
    )
    wp_build_tile_neighbor_list(
        sorted_pos_x=wp_sorted_pos_x,
        sorted_pos_y=wp_sorted_pos_y,
        sorted_pos_z=wp_sorted_pos_z,
        cell=wp_cell,
        inv_cell=wp_inv_cell,
        cutoff=float(cutoff),
        group_ctr_x=wp_group_ctr_x,
        group_ctr_y=wp_group_ctr_y,
        group_ctr_z=wp_group_ctr_z,
        group_ext_x=wp_group_ext_x,
        group_ext_y=wp_group_ext_y,
        group_ext_z=wp_group_ext_z,
        num_tiles=wp_num_tiles,
        tile_row_group=wp_tile_row_group,
        tile_col_group=wp_tile_col_group,
        wp_dtype=wp_dtype,
        device=wp_device,
    )


def build_tile_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    morton_codes: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
) -> None:
    """Build cluster-tile neighbor list state into pre-allocated tensors.

    Normalizes ``cell`` to a ``(3, 3)`` matrix + computes ``inv_cell``,
    then runs Morton sort (torch) + warp bbox reduction + warp
    tile-pair enumeration.  Triclinic cells supported.  All output
    tensors are filled in place.

    Parameters
    ----------
    positions : (N, 3) float
        Atomic coordinates, ``N % 32 == 0``, wrapped to the primary cell.
    cutoff : float
    cell : (1, 3, 3) or (3, 3) float
        Any non-degenerate cell (orthorhombic or triclinic).
    sorted_atom_index : (N,) int32 OUT
    morton_codes : (N,) int32 OUT (scratch used by the op)
    sorted_pos_x/y/z : (N,) float OUT
    group_ctr_x/y/z, group_ext_x/y/z : (ngroup_padded,) float OUT
    num_tiles : (1,) int32 OUT (atomic counter; reset internally)
    tile_row_group, tile_col_group : (max_tiles,) int32 OUT

    See Also
    --------
    nvalchemiops.neighbors.tile_warp.build_tile_neighbor_list : warp launcher
    """
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must be float32 or float64")
    cell_mat, inv_cell_mat = _cell_invcell_from_cell(cell)
    cell_mat = cell_mat.to(positions.dtype)
    inv_cell_mat = inv_cell_mat.to(positions.dtype)
    _build_tile_neighbor_list_op(
        positions,
        cutoff,
        cell_mat,
        inv_cell_mat,
        sorted_atom_index,
        morton_codes,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        group_ctr_x,
        group_ctr_y,
        group_ctr_z,
        group_ext_x,
        group_ext_y,
        group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
    )


@torch.library.custom_op(
    "nvalchemiops::_tile_to_matrix",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _tile_to_matrix_op(
    cutoff: float,
    natom: int,
    n_tiles: int,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
) -> None:
    if n_tiles <= 0:
        return
    device = sorted_pos_x.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_cell = _mat33f_from_torch(cell)
    wp_inv_cell = _mat33f_from_torch(inv_cell)
    wp_tile_to_matrix(
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, return_ctype=True),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(
            tile_col_group,
            dtype=wp.int32,
            return_ctype=True,
        ),
        cell=wp_cell,
        inv_cell=wp_inv_cell,
        cutoff=float(cutoff),
        natom=int(natom),
        n_tiles=int(n_tiles),
        neighbor_matrix=wp.from_torch(
            neighbor_matrix,
            dtype=wp.int32,
            return_ctype=True,
        ),
        num_neighbors=wp.from_torch(
            num_neighbors,
            dtype=wp.int32,
            return_ctype=True,
        ),
        neighbor_matrix_shifts=wp.from_torch(
            neighbor_matrix_shifts,
            dtype=wp.int32,
            return_ctype=True,
        ),
        wp_dtype=wp_dtype,
        device=wp_device,
    )


def tile_to_matrix(
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    natom: int,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
) -> None:
    """Convert the tile pair list to neighbor_matrix form in place."""
    cell_mat, inv_cell_mat = _cell_invcell_from_cell(cell)
    cell_mat = cell_mat.to(sorted_pos_x.dtype)
    inv_cell_mat = inv_cell_mat.to(sorted_pos_x.dtype)
    # Clamp ``num_tiles`` to the allocated buffer size: the build kernel
    # increments the counter unconditionally and only guards the write,
    # so on under-sized buffers ``num_tiles[0]`` can exceed
    # ``tile_row_group.shape[0]``.  Reading past that with the consumer
    # launcher would be an out-of-bounds GPU read.
    n_tiles = min(int(num_tiles.item()), int(tile_row_group.shape[0]))
    _tile_to_matrix_op(
        cutoff,
        natom,
        n_tiles,
        cell_mat,
        inv_cell_mat,
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
    )


@torch.library.custom_op(
    "nvalchemiops::_tile_to_coo",
    mutates_args=("pair_counter", "coo_list", "coo_shifts"),
)
def _tile_to_coo_op(
    cutoff: float,
    natom: int,
    n_tiles: int,
    max_pairs: int,
    cell: torch.Tensor,
    inv_cell: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
) -> None:
    if n_tiles <= 0:
        return
    device = sorted_pos_x.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_cell = _mat33f_from_torch(cell)
    wp_inv_cell = _mat33f_from_torch(inv_cell)
    wp_tile_to_coo(
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, return_ctype=True),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(
            tile_col_group,
            dtype=wp.int32,
            return_ctype=True,
        ),
        cell=wp_cell,
        inv_cell=wp_inv_cell,
        cutoff=float(cutoff),
        natom=int(natom),
        n_tiles=int(n_tiles),
        max_pairs=int(max_pairs),
        pair_counter=wp.from_torch(
            pair_counter,
            dtype=wp.int32,
            return_ctype=True,
        ),
        coo_list=wp.from_torch(coo_list, dtype=wp.int32, return_ctype=True),
        coo_shifts=wp.from_torch(coo_shifts, dtype=wp.int32, return_ctype=True),
        wp_dtype=wp_dtype,
        device=wp_device,
    )


def tile_to_coo(
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    natom: int,
    max_pairs: int,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
) -> None:
    """Convert the tile pair list to flat COO format in place."""
    cell_mat, inv_cell_mat = _cell_invcell_from_cell(cell)
    cell_mat = cell_mat.to(sorted_pos_x.dtype)
    inv_cell_mat = inv_cell_mat.to(sorted_pos_x.dtype)
    # See ``tile_to_matrix`` for the rationale on clamping num_tiles to
    # the allocated tile_row_group size.
    n_tiles = min(int(num_tiles.item()), int(tile_row_group.shape[0]))
    pair_counter.zero_()
    _tile_to_coo_op(
        cutoff,
        natom,
        n_tiles,
        max_pairs,
        cell_mat,
        inv_cell_mat,
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        pair_counter,
        coo_list,
        coo_shifts,
    )


# =============================================================================
# High-level convenience
# =============================================================================
def tile_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    max_neighbors: int | None = None,
    fill_value: int | None = None,
    format: str = "matrix",
    max_pairs: int | None = None,
    # matrix-format outputs
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    # coo-format outputs
    neighbor_list: torch.Tensor | None = None,
    neighbor_list_shifts: torch.Tensor | None = None,
    pair_counter: torch.Tensor | None = None,
    # scratch buffers
    sorted_atom_index: torch.Tensor | None = None,
    morton_codes: torch.Tensor | None = None,
    sorted_pos_x: torch.Tensor | None = None,
    sorted_pos_y: torch.Tensor | None = None,
    sorted_pos_z: torch.Tensor | None = None,
    group_ctr_x: torch.Tensor | None = None,
    group_ctr_y: torch.Tensor | None = None,
    group_ctr_z: torch.Tensor | None = None,
    group_ext_x: torch.Tensor | None = None,
    group_ext_y: torch.Tensor | None = None,
    group_ext_z: torch.Tensor | None = None,
    num_tiles: torch.Tensor | None = None,
    tile_row_group: torch.Tensor | None = None,
    tile_col_group: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Build a cluster-pair tile neighbor list (one-shot convenience).

    Runs Morton sort + warp bbox reduction + tile enumeration, then emits
    the result in one of three formats selected by ``format=``.  Supports
    orthorhombic and triclinic cells alike via ``_wrap_triclinic``.

    Parameters
    ----------
    positions : (N, 3) float
        Any ``N >= 0``.  Non-32-aligned ``N`` is supported via internal
        padding to ``ceil(N / TILE_GROUP_SIZE) * TILE_GROUP_SIZE``;
        padding slots use sentinel Morton codes and are filtered out by
        the convert/coo kernels.
    cutoff : float
    cell : (1, 3, 3) or (3, 3) float
        Any non-degenerate cell (orthorhombic or triclinic).
    max_neighbors : int, optional
        Falls back to ``estimate_max_neighbors(cutoff)``.  Matrix
        format only.
    fill_value : int, optional
        Matrix sentinel; defaults to ``N``.
    format : {"matrix", "coo", "tile"}, default "matrix"
        Output representation:

        - ``"matrix"``: returns
          ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)`` —
          the dense ``(N, max_neighbors)`` row-padded form used by
          ``cell_list`` and ``naive``.
        - ``"coo"``: returns
          ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)`` — flat
          pair list emitted directly by ``tile_to_coo`` (no matrix
          intermediate).  ``neighbor_ptr`` is reconstructed from
          ``bincount(neighbor_list[0])`` (cheap; requires a single CPU
          sync on ``pair_counter[0]`` that's needed for the trim
          anyway).
        - ``"tile"``: returns the native cluster-pair tile state as a
          7-tuple
          ``(num_tiles, tile_row_group, tile_col_group,
          sorted_atom_index, sorted_pos_x, sorted_pos_y, sorted_pos_z)``.
          No convert kernel is run.  Intended for downstream kernels
          that consume the tile-pair list directly with shared-memory
          tile loads.  Tile pairs are group-level half-fill: every
          emitted pair has ``tile_col_group[t] >= tile_row_group[t]``.
          The consumer chooses atom-level fill.
    max_pairs : int, optional
        Upper bound for COO output; defaults to ``N * max_neighbors``.
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts : optional
        Pre-allocated matrix-format outputs.  All-or-nothing only across
        the trio; supply all three or none.
    neighbor_list, neighbor_list_shifts, pair_counter : optional
        Pre-allocated COO-format outputs.  Shapes
        ``(2, max_pairs)``, ``(max_pairs, 3)``, ``(1,)`` int32.
        Same all-or-nothing semantics.
    sorted_atom_index, morton_codes, sorted_pos_{x,y,z},
    group_{ctr,ext}_{x,y,z}, num_tiles, tile_row_group, tile_col_group :
        Pre-allocated scratch buffers (shapes as returned by
        ``allocate_tile_neighbor_list``).  All-or-nothing: either
        provide every scratch buffer or none.  The trigger is
        ``sorted_atom_index``.  Reuse is safe — ``num_tiles`` is reset
        each call and every other scratch tensor is either fully
        overwritten or only read in regions the kernels just wrote.
    """
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must be float32 or float64")
    if format not in ("matrix", "coo", "tile"):
        raise ValueError(
            f"format must be 'matrix' | 'coo' | 'tile'; got {format!r}",
        )
    N = positions.shape[0]
    device = positions.device
    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)
    if fill_value is None:
        fill_value = N

    # Allocate scratch if caller didn't supply.  ``sorted_atom_index`` is the
    # all-or-nothing sentinel.
    if sorted_atom_index is None:
        (
            sorted_atom_index,
            morton_codes,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
        ) = allocate_tile_neighbor_list(
            N,
            device,
            dtype=positions.dtype,
        )

    build_tile_neighbor_list(
        positions,
        cutoff,
        cell,
        sorted_atom_index,
        morton_codes,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        group_ctr_x,
        group_ctr_y,
        group_ctr_z,
        group_ext_x,
        group_ext_y,
        group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
    )

    if format == "tile":
        return (
            num_tiles,
            tile_row_group,
            tile_col_group,
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
        )

    if format == "coo":
        if max_pairs is None:
            max_pairs = N * max_neighbors
        # ``tile_to_coo`` writes row-major (max_pairs, 2); we transpose to
        # package-canonical (2, num_pairs) on the way out.  Pre-allocation
        # kwargs accept the package layout (2, max_pairs); we view-as-flat
        # then reshape for the kernel.
        if neighbor_list is None:
            coo_buf = torch.empty(
                (max_pairs, 2),
                dtype=torch.int32,
                device=device,
            )
        else:
            # Caller passed (2, max_pairs).  Use a transposed view; the
            # kernel writes row-major into this buffer.
            coo_buf = neighbor_list.transpose(0, 1)
        if neighbor_list_shifts is None:
            neighbor_list_shifts = torch.empty(
                (max_pairs, 3),
                dtype=torch.int32,
                device=device,
            )
        if pair_counter is None:
            pair_counter = torch.zeros(1, dtype=torch.int32, device=device)
        tile_to_coo(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            cell,
            cutoff,
            N,
            int(max_pairs),
            pair_counter,
            coo_buf,
            neighbor_list_shifts,
        )
        # Trim to actual pair count and rebuild CSR neighbor_ptr.
        # The ``.item()`` is the only sync; it's needed for the slice
        # anyway, so the bincount is on a CPU-known-length tensor.
        npairs = int(pair_counter.item())
        nl = coo_buf[:npairs].transpose(0, 1).contiguous()  # (2, npairs)
        nls = neighbor_list_shifts[:npairs].contiguous()
        per_atom_counts = torch.bincount(nl[0].long(), minlength=N).to(
            torch.int32,
        )
        neighbor_ptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(per_atom_counts, dim=0).to(torch.int32),
            ],
        )
        return nl, neighbor_ptr, nls

    # ``format == "matrix"``: skip-prefill matrix path with tail fill.
    if neighbor_matrix is None:
        neighbor_matrix = torch.empty(
            (N, max_neighbors),
            dtype=torch.int32,
            device=device,
        )
    if num_neighbors is None:
        num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
    else:
        num_neighbors.zero_()
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = torch.empty(
            (N, max_neighbors, 3),
            dtype=torch.int32,
            device=device,
        )

    tile_to_matrix(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        cell,
        cutoff,
        N,
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
    )

    # Skip-prefill tail fill: write ``fill_value`` into the unused columns
    # of ``neighbor_matrix``.  Pairs with the always-write-shifts kernel
    # above to eliminate the per-step ``neighbor_matrix.fill_`` and
    # ``neighbor_matrix_shifts.zero_`` ops.
    if max_neighbors > 0:
        wp_fill_neighbor_matrix_tail(
            wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True),
            int(N),
            int(max_neighbors),
            int(fill_value),
            wp.from_torch(neighbor_matrix, dtype=wp.int32, return_ctype=True),
            str(device),
        )

    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
