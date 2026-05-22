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

"""PyTorch bindings for batched cluster-pair tile neighbor list.

All torch-side work (per-system Morton sort with system-major key,
per-system padding to a multiple of ``TILE_GROUP_SIZE``, SoA gather of
sorted positions, ``inv_cell_batch`` inversion, ``wp.from_torch``
conversion) lives in this module; the Warp layer at
``nvalchemiops.neighbors.tile_batch_warp`` only sees ``wp.array``
inputs.

Scope: float32, orthorhombic or triclinic PBC, one cell per system,
arbitrary natom per system (padded internally).
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.neighbors.neighbor_utils import (
    fill_neighbor_matrix_tail as wp_fill_neighbor_matrix_tail,
)
from nvalchemiops.neighbors.tile_batch_warp import (
    TILE_GROUP_SIZE,
)
from nvalchemiops.neighbors.tile_batch_warp import (
    batch_tile_to_coo as wp_batch_tile_to_coo,
)
from nvalchemiops.neighbors.tile_batch_warp import (
    batch_tile_to_matrix as wp_batch_tile_to_matrix,
)
from nvalchemiops.neighbors.tile_batch_warp import (
    build_batch_tile_neighbor_list as wp_build_batch_tile_neighbor_list,
)
from nvalchemiops.torch.types import get_wp_dtype

__all__ = [
    "TILE_GROUP_SIZE",
    "estimate_batch_tile_neighbor_list_sizes",
    "allocate_batch_tile_neighbor_list",
    "build_batch_tile_neighbor_list",
    "batch_tile_to_matrix",
    "batch_tile_to_coo",
    "batch_tile_neighbor_list",
]


# =============================================================================
# Sizing + allocation helpers
# =============================================================================
def estimate_batch_tile_neighbor_list_sizes(
    batch_ptr: torch.Tensor,
    max_tiles_per_group: int = 256,
) -> tuple[int, int, int, int, int]:
    """Estimate allocation sizes for the batched tile neighbor list state.

    Parameters
    ----------
    batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype=int32
        Cumulative atom counts defining per-system ranges.
    max_tiles_per_group : int, default 512
        Upper bound on neighbor groups per row_group (dense-cutoff cap).

    Returns
    -------
    n_padded : int
        Total padded atom count (sum of per-system ``ceil(natom/32)*32``).
    ngroup : int
        Number of 32-atom groups: ``n_padded // 32``.
    ngroup_padded : int
        Group-array pad length for in-bounds ``wp.tile_load`` at any
        TILE-aligned offset.
    max_tiles : int
        Upper bound on the tile pair list size.
    num_systems : int
    """
    num_systems = int(batch_ptr.shape[0]) - 1
    natom_per_system = (batch_ptr[1:] - batch_ptr[:-1]).to(torch.int64)
    natom_padded_per_system = (
        (natom_per_system + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    n_padded = int(natom_padded_per_system.sum().item())
    ngroup = n_padded // TILE_GROUP_SIZE
    ngroup_padded = (
        (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE + TILE_GROUP_SIZE
    max_tiles = ngroup * min(ngroup, max_tiles_per_group)
    return n_padded, ngroup, ngroup_padded, max_tiles, num_systems


def allocate_batch_tile_neighbor_list(
    batch_ptr: torch.Tensor,
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Allocate all state tensors consumed by ``build_batch_tile_neighbor_list``.

    Returns ``(sorted_atom_index, sort_inv_real, sorted_pos_x, sorted_pos_y,
    sorted_pos_z, batch_idx_sorted, batch_ptr_padded, group_system,
    group_ptr, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x,
    group_ext_y, group_ext_z, num_tiles, tile_row_group, tile_col_group,
    tile_system)``.
    """
    n_padded, ngroup, ngroup_padded, max_tiles, num_systems = (
        estimate_batch_tile_neighbor_list_sizes(
            batch_ptr,
            max_tiles_per_group=max_tiles_per_group,
        )
    )
    N = int(batch_ptr[-1].item())
    sorted_atom_index = torch.empty(n_padded, dtype=torch.int32, device=device)
    sort_inv = torch.empty(N, dtype=torch.int32, device=device)
    sorted_pos_x = torch.empty(n_padded, dtype=dtype, device=device)
    sorted_pos_y = torch.empty(n_padded, dtype=dtype, device=device)
    sorted_pos_z = torch.empty(n_padded, dtype=dtype, device=device)
    batch_idx_sorted = torch.empty(n_padded, dtype=torch.int32, device=device)
    batch_ptr_padded = torch.empty(num_systems + 1, dtype=torch.int32, device=device)
    group_system = torch.empty(ngroup, dtype=torch.int32, device=device)
    group_ptr = torch.empty(num_systems + 1, dtype=torch.int32, device=device)
    group_ctr_x = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ctr_y = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ctr_z = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_x = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_y = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    group_ext_z = torch.zeros(ngroup_padded, dtype=dtype, device=device)
    num_tiles = torch.zeros(1, dtype=torch.int32, device=device)
    tile_row_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    tile_col_group = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    tile_system = torch.zeros(max_tiles, dtype=torch.int32, device=device)
    return (
        sorted_atom_index,
        sort_inv,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        batch_idx_sorted,
        batch_ptr_padded,
        group_system,
        group_ptr,
        group_ctr_x,
        group_ctr_y,
        group_ctr_z,
        group_ext_x,
        group_ext_y,
        group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
    )


# =============================================================================
# Morton sort + padded scatter (torch side)
# =============================================================================
def _per_atom_morton_codes(
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    inv_cell_batch: torch.Tensor,
) -> torch.Tensor:
    """30-bit Morton codes using per-atom fractional coords (triclinic-safe)."""
    inv_per_atom = inv_cell_batch[batch_idx]
    frac = torch.einsum("ni,nij->nj", positions, inv_per_atom)
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

    return (_spread(iz) << 2) | (_spread(iy) << 1) | _spread(ix)


def _batched_morton_sort_padded(
    positions: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_ptr: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sort_inv: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    batch_idx_sorted: torch.Tensor,
    batch_ptr_padded: torch.Tensor,
) -> None:
    """Per-system Morton sort into the padded layout (torch ops only)."""
    N = positions.shape[0]
    num_systems = inv_cell_batch.shape[0]
    device = positions.device

    natom_per_system = batch_ptr[1:] - batch_ptr[:-1]
    natom_padded_per_system = (
        (natom_per_system + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE

    bpp = torch.zeros(num_systems + 1, dtype=torch.int32, device=device)
    torch.cumsum(natom_padded_per_system, dim=0, out=bpp[1:])
    batch_ptr_padded.copy_(bpp)

    codes = _per_atom_morton_codes(positions, batch_idx, inv_cell_batch)
    # System-major key: system index in high 32 bits, Morton code in low.
    system_major = codes.to(torch.int64) | (batch_idx.to(torch.int64) << 32)
    sorted_atom_index_real = torch.argsort(system_major).to(torch.int32)

    # Inverse permutation over real atoms.
    inv = torch.empty(N, dtype=torch.int32, device=device)
    inv[sorted_atom_index_real] = torch.arange(
        N,
        dtype=torch.int32,
        device=device,
    )
    sort_inv.copy_(inv)

    # Placement indices for each real atom in the padded layout.
    batch_idx_sorted_real = batch_idx[sorted_atom_index_real]
    within_system = (
        torch.arange(
            N,
            dtype=torch.int32,
            device=device,
        )
        - batch_ptr[batch_idx_sorted_real]
    )
    padded_slot = bpp[batch_idx_sorted_real] + within_system

    # sorted_atom_index[k] = N is the padding sentinel.
    sp = torch.full(
        (sorted_atom_index.shape[0],),
        N,
        dtype=torch.int32,
        device=device,
    )
    sp[padded_slot] = sorted_atom_index_real
    sorted_atom_index.copy_(sp)

    # System index for every padded slot.
    bis = (
        torch.repeat_interleave(
            torch.arange(num_systems, dtype=torch.int32, device=device),
            natom_padded_per_system,
        )
        .to(torch.int32)
        .contiguous()
    )
    batch_idx_sorted.copy_(bis)

    # Padding slots duplicate each system's first real atom, so
    # tile_min / tile_max over the group remain well-formed.  Duplicates
    # cannot cause a miss because they add a point already inside the
    # system's spatial region.
    first_atom_per_system = batch_ptr[:-1]
    position_index = torch.where(
        sp == N,
        first_atom_per_system[bis],
        sp,
    )
    sorted_pos = positions[position_index].contiguous()
    sorted_pos_x.copy_(sorted_pos[:, 0].contiguous())
    sorted_pos_y.copy_(sorted_pos[:, 1].contiguous())
    sorted_pos_z.copy_(sorted_pos[:, 2].contiguous())


# =============================================================================
# Component ops (torch wrappers around the warp launchers)
# =============================================================================
@torch.library.custom_op(
    "nvalchemiops::_build_batch_tile_neighbor_list",
    mutates_args=(
        "sorted_atom_index",
        "sort_inv",
        "sorted_pos_x",
        "sorted_pos_y",
        "sorted_pos_z",
        "batch_idx_sorted",
        "batch_ptr_padded",
        "group_system",
        "group_ptr",
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "num_tiles",
        "tile_row_group",
        "tile_col_group",
        "tile_system",
    ),
)
def _build_batch_tile_neighbor_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    batch_ptr: torch.Tensor,
    batch_idx: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sort_inv: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    batch_idx_sorted: torch.Tensor,
    batch_ptr_padded: torch.Tensor,
    group_system: torch.Tensor,
    group_ptr: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
) -> None:
    wp_device = str(positions.device)
    wp_dtype = get_wp_dtype(positions.dtype)

    _batched_morton_sort_padded(
        positions,
        batch_idx,
        batch_ptr,
        inv_cell_batch,
        sorted_atom_index,
        sort_inv,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        batch_idx_sorted,
        batch_ptr_padded,
    )

    group_ptr.copy_((batch_ptr_padded // TILE_GROUP_SIZE).to(torch.int32).contiguous())
    group_system.copy_(
        batch_idx_sorted[::TILE_GROUP_SIZE].to(torch.int32).contiguous(),
    )

    num_tiles.zero_()
    group_ctr_x.zero_()
    group_ctr_y.zero_()
    group_ctr_z.zero_()
    group_ext_x.zero_()
    group_ext_y.zero_()
    group_ext_z.zero_()

    wp_build_batch_tile_neighbor_list(
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, return_ctype=True),
        group_system=wp.from_torch(group_system, dtype=wp.int32, return_ctype=True),
        group_ptr=wp.from_torch(group_ptr, dtype=wp.int32, return_ctype=True),
        cell_batch=wp.from_torch(
            cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        inv_cell_batch=wp.from_torch(
            inv_cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        cutoff=float(cutoff),
        group_ctr_x=wp.from_torch(group_ctr_x, dtype=wp_dtype, return_ctype=True),
        group_ctr_y=wp.from_torch(group_ctr_y, dtype=wp_dtype, return_ctype=True),
        group_ctr_z=wp.from_torch(group_ctr_z, dtype=wp_dtype, return_ctype=True),
        group_ext_x=wp.from_torch(group_ext_x, dtype=wp_dtype, return_ctype=True),
        group_ext_y=wp.from_torch(group_ext_y, dtype=wp_dtype, return_ctype=True),
        group_ext_z=wp.from_torch(group_ext_z, dtype=wp_dtype, return_ctype=True),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(
            tile_col_group,
            dtype=wp.int32,
            return_ctype=True,
        ),
        tile_system=wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
        wp_dtype=wp_dtype,
        device=wp_device,
    )


def build_batch_tile_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell_batch: torch.Tensor,
    batch_ptr: torch.Tensor,
    sorted_atom_index: torch.Tensor,
    sort_inv: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    batch_idx_sorted: torch.Tensor,
    batch_ptr_padded: torch.Tensor,
    group_system: torch.Tensor,
    group_ptr: torch.Tensor,
    group_ctr_x: torch.Tensor,
    group_ctr_y: torch.Tensor,
    group_ctr_z: torch.Tensor,
    group_ext_x: torch.Tensor,
    group_ext_y: torch.Tensor,
    group_ext_z: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    inv_cell_batch: torch.Tensor | None = None,
) -> None:
    """Build batched tile neighbor list state into pre-allocated outputs.

    Runs the per-system Morton sort + padded SoA gather in torch, then
    the bbox reduction + tile-pair enumeration in warp.
    """
    if positions.dtype != torch.float32:
        raise TypeError("positions must be float32")
    if (
        cell_batch.dtype != torch.float32
        or cell_batch.ndim != 3
        or cell_batch.shape[1:] != (3, 3)
    ):
        raise ValueError(
            f"cell_batch must be (S, 3, 3) float32; got {tuple(cell_batch.shape)}",
        )
    if batch_ptr.dtype != torch.int32 or batch_ptr.ndim != 1:
        raise ValueError("batch_ptr must be 1D int32")

    num_systems = cell_batch.shape[0]
    if batch_ptr.shape[0] != num_systems + 1:
        raise ValueError(
            f"batch_ptr length {batch_ptr.shape[0]} != num_systems+1 = {num_systems + 1}",
        )
    N = positions.shape[0]
    if int(batch_ptr[-1].item()) != N:
        raise ValueError(f"batch_ptr[-1] ({int(batch_ptr[-1])}) != N ({N})")

    if inv_cell_batch is None:
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()

    batch_idx = torch.repeat_interleave(
        torch.arange(num_systems, dtype=torch.int32, device=positions.device),
        (batch_ptr[1:] - batch_ptr[:-1]).to(torch.int64),
    )

    _build_batch_tile_neighbor_list_op(
        positions,
        float(cutoff),
        cell_batch,
        inv_cell_batch,
        batch_ptr,
        batch_idx,
        sorted_atom_index,
        sort_inv,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        batch_idx_sorted,
        batch_ptr_padded,
        group_system,
        group_ptr,
        group_ctr_x,
        group_ctr_y,
        group_ctr_z,
        group_ext_x,
        group_ext_y,
        group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_tile_to_matrix",
    mutates_args=("neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"),
)
def _batch_tile_to_matrix_op(
    cutoff: float,
    natom: int,
    n_tiles: int,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
) -> None:
    if n_tiles <= 0:
        return
    wp_device = str(sorted_pos_x.device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_batch_tile_to_matrix(
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, return_ctype=True),
        cell_batch=wp.from_torch(
            cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        inv_cell_batch=wp.from_torch(
            inv_cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(
            tile_col_group,
            dtype=wp.int32,
            return_ctype=True,
        ),
        tile_system=wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
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


def batch_tile_to_matrix(
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    cutoff: float,
    natom: int,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    inv_cell_batch: torch.Tensor | None = None,
) -> None:
    """Convert the batched tile pair list to neighbor_matrix in place."""
    n_tiles = int(num_tiles.item())
    if n_tiles <= 0:
        return
    if inv_cell_batch is None:
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()
    _batch_tile_to_matrix_op(
        cutoff,
        natom,
        n_tiles,
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        cell_batch,
        inv_cell_batch,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_tile_to_coo",
    mutates_args=("pair_counter", "coo_list", "coo_shifts"),
)
def _batch_tile_to_coo_op(
    cutoff: float,
    natom: int,
    n_tiles: int,
    max_pairs: int,
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    inv_cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
) -> None:
    if n_tiles <= 0:
        return
    wp_device = str(sorted_pos_x.device)
    wp_dtype = get_wp_dtype(sorted_pos_x.dtype)
    wp_batch_tile_to_coo(
        sorted_atom_index=wp.from_torch(
            sorted_atom_index, dtype=wp.int32, return_ctype=True
        ),
        sorted_pos_x=wp.from_torch(sorted_pos_x, dtype=wp_dtype, return_ctype=True),
        sorted_pos_y=wp.from_torch(sorted_pos_y, dtype=wp_dtype, return_ctype=True),
        sorted_pos_z=wp.from_torch(sorted_pos_z, dtype=wp_dtype, return_ctype=True),
        cell_batch=wp.from_torch(
            cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        inv_cell_batch=wp.from_torch(
            inv_cell_batch.contiguous(),
            dtype=wp.mat33f,
            return_ctype=True,
        ),
        num_tiles=wp.from_torch(num_tiles, dtype=wp.int32, return_ctype=True),
        tile_row_group=wp.from_torch(tile_row_group, dtype=wp.int32, return_ctype=True),
        tile_col_group=wp.from_torch(
            tile_col_group,
            dtype=wp.int32,
            return_ctype=True,
        ),
        tile_system=wp.from_torch(tile_system, dtype=wp.int32, return_ctype=True),
        cutoff=float(cutoff),
        natom=int(natom),
        n_tiles=int(n_tiles),
        max_pairs=int(max_pairs),
        pair_counter=wp.from_torch(pair_counter, dtype=wp.int32, return_ctype=True),
        coo_list=wp.from_torch(coo_list, dtype=wp.int32, return_ctype=True),
        coo_shifts=wp.from_torch(coo_shifts, dtype=wp.int32, return_ctype=True),
        wp_dtype=wp_dtype,
        device=wp_device,
    )


def batch_tile_to_coo(
    sorted_atom_index: torch.Tensor,
    sorted_pos_x: torch.Tensor,
    sorted_pos_y: torch.Tensor,
    sorted_pos_z: torch.Tensor,
    cell_batch: torch.Tensor,
    num_tiles: torch.Tensor,
    tile_row_group: torch.Tensor,
    tile_col_group: torch.Tensor,
    tile_system: torch.Tensor,
    cutoff: float,
    natom: int,
    max_pairs: int,
    pair_counter: torch.Tensor,
    coo_list: torch.Tensor,
    coo_shifts: torch.Tensor,
    inv_cell_batch: torch.Tensor | None = None,
) -> None:
    """Convert the batched tile pair list to flat COO pair list in place."""
    n_tiles = int(num_tiles.item())
    if n_tiles <= 0:
        return
    if inv_cell_batch is None:
        inv_cell_batch = torch.linalg.inv(cell_batch).contiguous()
    pair_counter.zero_()
    _batch_tile_to_coo_op(
        cutoff,
        natom,
        n_tiles,
        max_pairs,
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        cell_batch,
        inv_cell_batch,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
        pair_counter,
        coo_list,
        coo_shifts,
    )


# =============================================================================
# High-level convenience
# =============================================================================
def batch_tile_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell_batch: torch.Tensor,
    batch_ptr: torch.Tensor,
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
    inv_cell_batch: torch.Tensor | None = None,
    # scratch buffers
    sorted_atom_index: torch.Tensor | None = None,
    sort_inv: torch.Tensor | None = None,
    sorted_pos_x: torch.Tensor | None = None,
    sorted_pos_y: torch.Tensor | None = None,
    sorted_pos_z: torch.Tensor | None = None,
    batch_idx_sorted: torch.Tensor | None = None,
    batch_ptr_padded: torch.Tensor | None = None,
    group_system: torch.Tensor | None = None,
    group_ptr: torch.Tensor | None = None,
    group_ctr_x: torch.Tensor | None = None,
    group_ctr_y: torch.Tensor | None = None,
    group_ctr_z: torch.Tensor | None = None,
    group_ext_x: torch.Tensor | None = None,
    group_ext_y: torch.Tensor | None = None,
    group_ext_z: torch.Tensor | None = None,
    num_tiles: torch.Tensor | None = None,
    tile_row_group: torch.Tensor | None = None,
    tile_col_group: torch.Tensor | None = None,
    tile_system: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Build a batched cluster-pair tile neighbor list (one-shot convenience).

    Supports triclinic ``cell_batch`` of shape ``(S, 3, 3)`` and
    arbitrary per-system atom counts (padded internally to a multiple
    of ``TILE_GROUP_SIZE``).  Output format selected by ``format=``:

    - ``"matrix"`` (default): returns
      ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``.
    - ``"coo"``: returns
      ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)`` via the
      direct ``batch_tile_to_coo`` path (no matrix intermediate).
      ``neighbor_ptr`` is reconstructed from
      ``bincount(neighbor_list[0])``.
    - ``"tile"``: returns the native cluster-pair tile state as an
      11-tuple
      ``(num_tiles, tile_row_group, tile_col_group, tile_system,
      sorted_atom_index, sorted_pos_x, sorted_pos_y, sorted_pos_z,
      batch_idx_sorted, batch_ptr_padded, group_ptr)`` — same shape as
      the SS tile output plus the per-tile/per-atom/per-system mapping
      arrays a batch tile consumer needs.

    Scratch buffers (``sorted_atom_index`` through ``tile_system``)
    follow the same all-or-nothing pre-allocation contract as
    ``tile_neighbor_list``: supply every scratch tensor (shapes as
    returned by ``allocate_batch_tile_neighbor_list``) or none.
    """
    if positions.dtype != torch.float32:
        raise TypeError("positions must be float32")
    if format not in ("matrix", "coo", "tile"):
        raise ValueError(
            f"format must be 'matrix' | 'coo' | 'tile'; got {format!r}",
        )
    device = positions.device
    N = int(batch_ptr[-1].item())

    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)
    if fill_value is None:
        fill_value = N

    if sorted_atom_index is None:
        (
            sorted_atom_index,
            sort_inv,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            batch_idx_sorted,
            batch_ptr_padded,
            group_system,
            group_ptr,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
        ) = allocate_batch_tile_neighbor_list(
            batch_ptr,
            device,
            dtype=positions.dtype,
        )

    build_batch_tile_neighbor_list(
        positions,
        cutoff,
        cell_batch,
        batch_ptr,
        sorted_atom_index,
        sort_inv,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        batch_idx_sorted,
        batch_ptr_padded,
        group_system,
        group_ptr,
        group_ctr_x,
        group_ctr_y,
        group_ctr_z,
        group_ext_x,
        group_ext_y,
        group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
        inv_cell_batch=inv_cell_batch,
    )

    if format == "tile":
        return (
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            batch_idx_sorted,
            batch_ptr_padded,
            group_ptr,
        )

    if format == "coo":
        if max_pairs is None:
            max_pairs = N * max_neighbors
        if neighbor_list is None:
            coo_buf = torch.empty(
                (max_pairs, 2),
                dtype=torch.int32,
                device=device,
            )
        else:
            coo_buf = neighbor_list.transpose(0, 1)
        if neighbor_list_shifts is None:
            neighbor_list_shifts = torch.empty(
                (max_pairs, 3),
                dtype=torch.int32,
                device=device,
            )
        if pair_counter is None:
            pair_counter = torch.zeros(1, dtype=torch.int32, device=device)
        batch_tile_to_coo(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            cell_batch,
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            cutoff,
            N,
            int(max_pairs),
            pair_counter,
            coo_buf,
            neighbor_list_shifts,
            inv_cell_batch=inv_cell_batch,
        )
        npairs = int(pair_counter.item())
        nl = coo_buf[:npairs].transpose(0, 1).contiguous()
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

    batch_tile_to_matrix(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        cell_batch,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
        cutoff,
        N,
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        inv_cell_batch=inv_cell_batch,
    )

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
