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

"""JAX bindings for the single-system cluster-pair tile neighbor list.

Mirrors :mod:`nvalchemiops.torch.neighbors.tile_warp`: the Morton sort +
SoA gather happens in JAX, then the bbox reduction + tile-pair enumeration
and the final conversion (matrix / COO) run on the Warp side via
``jax_callable`` callbacks.  Tile-format output returns the raw cluster
state for consumers that want to plug into shared-memory tile loads
directly.

Scope: single system, triclinic-safe, float32 only, ``N >= 0`` (any
non-32-aligned ``N`` is padded internally).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.neighbors.tile_warp import (
    TILE_GROUP_SIZE,
)
from nvalchemiops.neighbors.tile_warp import (
    build_tile_neighbor_list as _warp_build_tile_neighbor_list,
)
from nvalchemiops.neighbors.tile_warp import (
    tile_to_coo as _warp_tile_to_coo,
)
from nvalchemiops.neighbors.tile_warp import (
    tile_to_matrix as _warp_tile_to_matrix,
)

__all__ = [
    "TILE_GROUP_SIZE",
    "allocate_tile_neighbor_list",
    "build_tile_neighbor_list",
    "estimate_tile_neighbor_list_sizes",
    "tile_neighbor_list",
    "tile_to_coo",
    "tile_to_matrix",
]


# =============================================================================
# Sizing + allocation helpers (pure JAX, no Warp launches)
# =============================================================================
def estimate_tile_neighbor_list_sizes(
    total_atoms: int,
    max_tiles_per_group: int = 256,
) -> tuple[int, int, int, int]:
    """Estimate allocation sizes for the tile neighbor list state.

    Mirrors :func:`nvalchemiops.torch.neighbors.tile_warp.estimate_tile_neighbor_list_sizes`.

    Parameters
    ----------
    total_atoms : int
        Real atom count.  Any ``total_atoms >= 0`` is accepted.
    max_tiles_per_group : int, default 256
        Upper bound on neighbor groups per row_group.

    Returns
    -------
    n_padded : int
        Padded atom count = ``ceil(total_atoms / TILE_GROUP_SIZE) * TILE_GROUP_SIZE``.
    ngroup : int
        ``n_padded // TILE_GROUP_SIZE``.
    ngroup_padded : int
        Group-array pad length, multiple of TILE_GROUP_SIZE.
    max_tiles : int
        Upper bound on the tile-pair list size.
    """
    if total_atoms < 0:
        raise ValueError(f"total_atoms must be >= 0; got {total_atoms}")
    n_padded = (
        (total_atoms + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    if n_padded == 0:
        n_padded = TILE_GROUP_SIZE
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
    dtype: jnp.dtype = jnp.float32,
    max_tiles_per_group: int = 256,
) -> tuple[jax.Array, ...]:
    """Allocate the state tensors consumed by :func:`build_tile_neighbor_list`.

    Returns ``(sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y,
    sorted_pos_z, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x,
    group_ext_y, group_ext_z, num_tiles, tile_row_group, tile_col_group)``.
    """
    n_padded, ngroup, ngroup_padded, max_tiles = estimate_tile_neighbor_list_sizes(
        total_atoms,
        max_tiles_per_group=max_tiles_per_group,
    )
    return (
        jnp.zeros(n_padded, dtype=jnp.int32),  # sorted_atom_index
        jnp.zeros(n_padded, dtype=jnp.int32),  # morton_codes
        jnp.zeros(n_padded, dtype=dtype),  # sorted_pos_x
        jnp.zeros(n_padded, dtype=dtype),  # sorted_pos_y
        jnp.zeros(n_padded, dtype=dtype),  # sorted_pos_z
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ctr_x
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ctr_y
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ctr_z
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ext_x
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ext_y
        jnp.zeros(ngroup_padded, dtype=dtype),  # group_ext_z
        jnp.zeros(1, dtype=jnp.int32),  # num_tiles
        jnp.zeros(max_tiles, dtype=jnp.int32),  # tile_row_group
        jnp.zeros(max_tiles, dtype=jnp.int32),  # tile_col_group
    )


# =============================================================================
# Morton code helpers (JAX, mirror the torch path)
# =============================================================================
def _spread_10bit(x: jax.Array) -> jax.Array:
    """Spread the low 10 bits of ``x`` across a 30-bit Morton code."""
    x = x & 0x3FF
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


def _compute_morton_codes(
    positions_padded: jax.Array,
    inv_cell: jax.Array,
    natom: int,
    n_padded: int,
) -> jax.Array:
    """Compute 30-bit Morton codes for atoms, sentinel-pad the tail.

    Pads positions are assumed to copy a real atom (in-cell coordinate);
    after Morton coding their 30-bit code is OR-ed with ``0x40000000`` so
    they sort after every real-atom code.
    """
    inv_cell_2d = inv_cell[0] if inv_cell.ndim == 3 else inv_cell
    frac = positions_padded @ inv_cell_2d.T
    frac = frac - jnp.floor(frac)
    bucket = jnp.clip((frac * 1024.0).astype(jnp.int32), 0, 1023)
    ix, iy, iz = bucket[:, 0], bucket[:, 1], bucket[:, 2]
    codes = (_spread_10bit(iz) << 2) | (_spread_10bit(iy) << 1) | _spread_10bit(ix)
    if natom < n_padded:
        sentinel = jnp.full((n_padded - natom,), 0x40000000, dtype=jnp.int32)
        codes = jnp.concatenate([codes[:natom], sentinel])
    return codes


# =============================================================================
# jax_callable wrappers (Warp launchers behind GraphMode.WARP callbacks)
# =============================================================================
def _build_tile_neighbor_list_callback(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
) -> None:
    """jax_callable callback for the SS tile bbox + tile-pair enumeration."""
    _warp_build_tile_neighbor_list(
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell=cell,
        inv_cell=inv_cell,
        cutoff=float(cutoff),
        group_ctr_x=group_ctr_x,
        group_ctr_y=group_ctr_y,
        group_ctr_z=group_ctr_z,
        group_ext_x=group_ext_x,
        group_ext_y=group_ext_y,
        group_ext_z=group_ext_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
    )


def _tile_to_matrix_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
    cutoff: wp.float32,
    natom: wp.int32,
    n_tiles: wp.int32,
) -> None:
    """jax_callable callback for the tile-pair → matrix conversion kernel."""
    _warp_tile_to_matrix(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        cell=cell,
        inv_cell=inv_cell,
        cutoff=float(cutoff),
        natom=int(natom),
        n_tiles=int(n_tiles),
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
    )


def _tile_to_coo_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    pair_counter: wp.array(dtype=wp.int32),
    coo_list: wp.array(dtype=wp.int32, ndim=2),
    coo_shifts: wp.array(dtype=wp.int32, ndim=2),
    cutoff: wp.float32,
    natom: wp.int32,
    n_tiles: wp.int32,
    max_pairs: wp.int32,
) -> None:
    """jax_callable callback for the tile-pair → flat COO conversion kernel."""
    _warp_tile_to_coo(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        cell=cell,
        inv_cell=inv_cell,
        cutoff=float(cutoff),
        natom=int(natom),
        n_tiles=int(n_tiles),
        max_pairs=int(max_pairs),
        pair_counter=pair_counter,
        coo_list=coo_list,
        coo_shifts=coo_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
    )


_jax_build_tile_neighbor_list = jax_callable(
    _build_tile_neighbor_list_callback,
    num_outputs=9,
    in_out_argnames=[
        "group_ctr_x",
        "group_ctr_y",
        "group_ctr_z",
        "group_ext_x",
        "group_ext_y",
        "group_ext_z",
        "num_tiles",
        "tile_row_group",
        "tile_col_group",
    ],
    graph_mode=GraphMode.WARP,
)

_jax_tile_to_matrix = jax_callable(
    _tile_to_matrix_callback,
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"],
    graph_mode=GraphMode.WARP,
)

_jax_tile_to_coo = jax_callable(
    _tile_to_coo_callback,
    num_outputs=3,
    in_out_argnames=["pair_counter", "coo_list", "coo_shifts"],
    graph_mode=GraphMode.WARP,
)


# =============================================================================
# User-facing JAX entry points
# =============================================================================
def _normalize_cell(cell: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Coerce ``(3, 3)`` or ``(1, 3, 3)`` cell to ``(1, 3, 3)`` with given dtype."""
    if cell.ndim == 2:
        cell = cell[jnp.newaxis, :, :]
    if cell.shape != (1, 3, 3):
        raise ValueError(f"cell must have shape (3, 3) or (1, 3, 3); got {cell.shape}")
    return cell.astype(dtype)


def _morton_sort_and_gather(
    positions: jax.Array,
    cell: jax.Array,
    n_padded: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pad positions to ``n_padded``, compute Morton codes, argsort, gather SoA.

    Returns ``(sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y,
    sorted_pos_z)``.
    """
    N = positions.shape[0]
    inv_cell = jnp.linalg.inv(cell[0])
    if N < n_padded:
        # Pad with the last real atom (any in-cell point works).
        pad = jnp.broadcast_to(
            positions[-1:],
            (n_padded - N, 3),
        )
        padded_positions = jnp.concatenate([positions, pad], axis=0)
    else:
        padded_positions = positions
    morton_codes = _compute_morton_codes(padded_positions, inv_cell, N, n_padded)
    perm = jnp.argsort(morton_codes).astype(jnp.int32)
    sorted_pos = padded_positions[perm]
    return (
        perm,
        morton_codes,
        jnp.asarray(sorted_pos[:, 0]),
        jnp.asarray(sorted_pos[:, 1]),
        jnp.asarray(sorted_pos[:, 2]),
    )


def build_tile_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Build the tile neighbor list state in pre-allocated form.

    Runs Morton sort + SoA gather in JAX, then the bbox reduction + tile
    enumeration on the Warp side via a ``jax_callable`` callback.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3), dtype=float32
        Atomic coordinates.  Any ``N >= 0``; non-32-aligned ``N`` is padded
        internally.
    cutoff : float
        Cutoff distance for the bbox filter.
    cell : jax.Array, shape (3, 3) or (1, 3, 3), dtype=float32
        Unit cell matrix (orthorhombic or triclinic).

    Returns
    -------
    tuple of jax.Array
        ``(sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y,
        sorted_pos_z, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x,
        group_ext_y, group_ext_z, num_tiles, tile_row_group,
        tile_col_group)``.

    Notes
    -----
    Float32 only.  The cluster-pair tile kernels currently only support
    ``wp.float32``.
    """
    if positions.dtype != jnp.float32:
        raise TypeError(
            "positions must be float32 (tile_warp kernels are float32 only)"
        )
    N = positions.shape[0]
    n_padded, ngroup, ngroup_padded, max_tiles = estimate_tile_neighbor_list_sizes(N)

    cell_n = _normalize_cell(cell, jnp.float32)
    inv_cell_n = jnp.linalg.inv(cell_n[0])[jnp.newaxis, :, :]

    sorted_atom_index, morton_codes, sorted_pos_x, sorted_pos_y, sorted_pos_z = (
        _morton_sort_and_gather(positions, cell_n, n_padded)
    )

    # Allocate the bbox + tile output buffers.
    group_ctr_x = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ctr_y = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ctr_z = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ext_x = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ext_y = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ext_z = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    num_tiles = jnp.zeros(1, dtype=jnp.int32)
    tile_row_group = jnp.zeros(max_tiles, dtype=jnp.int32)
    tile_col_group = jnp.zeros(max_tiles, dtype=jnp.int32)

    (
        group_ctr_x,
        group_ctr_y,
        group_ctr_z,
        group_ext_x,
        group_ext_y,
        group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
    ) = _jax_build_tile_neighbor_list(
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        cell_n,
        inv_cell_n,
        group_ctr_x,
        group_ctr_y,
        group_ctr_z,
        group_ext_x,
        group_ext_y,
        group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        float(cutoff),
    )

    del ngroup  # implicit in group_*_x.shape[0]
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


def tile_to_matrix(
    sorted_atom_index: jax.Array,
    sorted_pos_x: jax.Array,
    sorted_pos_y: jax.Array,
    sorted_pos_z: jax.Array,
    num_tiles: jax.Array,
    tile_row_group: jax.Array,
    tile_col_group: jax.Array,
    cell: jax.Array,
    cutoff: float,
    natom: int,
    max_neighbors: int,
    *,
    fill_value: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert the tile pair list to dense neighbor-matrix form.

    Returns ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``.
    Skip-prefill: the warp kernel only writes the active entries, then a
    JAX-side ``jnp.where`` fills the unused tail columns with
    ``fill_value`` (defaults to ``natom``).
    """
    if fill_value is None:
        fill_value = natom
    cell_n = _normalize_cell(cell, jnp.float32)
    inv_cell_n = jnp.linalg.inv(cell_n[0])[jnp.newaxis, :, :]

    neighbor_matrix = jnp.zeros((natom, max_neighbors), dtype=jnp.int32)
    num_neighbors = jnp.zeros(natom, dtype=jnp.int32)
    neighbor_matrix_shifts = jnp.zeros((natom, max_neighbors, 3), dtype=jnp.int32)

    n_tiles_host = int(num_tiles[0])
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = _jax_tile_to_matrix(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        cell_n,
        inv_cell_n,
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        float(cutoff),
        int(natom),
        n_tiles_host,
    )

    # Tail fill: replace inactive columns with ``fill_value``.
    col_idx = jnp.arange(max_neighbors, dtype=jnp.int32)[jnp.newaxis, :]
    active = col_idx < num_neighbors[:, jnp.newaxis]
    neighbor_matrix = jnp.where(active, neighbor_matrix, jnp.int32(fill_value))
    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


def tile_to_coo(
    sorted_atom_index: jax.Array,
    sorted_pos_x: jax.Array,
    sorted_pos_y: jax.Array,
    sorted_pos_z: jax.Array,
    num_tiles: jax.Array,
    tile_row_group: jax.Array,
    tile_col_group: jax.Array,
    cell: jax.Array,
    cutoff: float,
    natom: int,
    max_pairs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert the tile pair list to flat COO form.

    Returns ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``.
    ``neighbor_list`` has shape ``(2, num_pairs)`` and ``neighbor_ptr`` is
    a CSR-style range over the per-atom outgoing pairs (reconstructed
    via ``jnp.bincount``).
    """
    cell_n = _normalize_cell(cell, jnp.float32)
    inv_cell_n = jnp.linalg.inv(cell_n[0])[jnp.newaxis, :, :]

    pair_counter = jnp.zeros(1, dtype=jnp.int32)
    coo_list = jnp.zeros((max_pairs, 2), dtype=jnp.int32)
    coo_shifts = jnp.zeros((max_pairs, 3), dtype=jnp.int32)

    n_tiles_host = int(num_tiles[0])
    pair_counter, coo_list, coo_shifts = _jax_tile_to_coo(
        sorted_atom_index,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        cell_n,
        inv_cell_n,
        pair_counter,
        coo_list,
        coo_shifts,
        float(cutoff),
        int(natom),
        n_tiles_host,
        int(max_pairs),
    )

    npairs = int(pair_counter[0])
    coo_list_trim = coo_list[:npairs]
    coo_shifts_trim = coo_shifts[:npairs]
    neighbor_list = coo_list_trim.T  # (2, npairs)

    # CSR-style neighbor_ptr from per-atom counts.
    per_atom = jnp.bincount(neighbor_list[0], length=natom).astype(jnp.int32)
    neighbor_ptr = jnp.concatenate(
        [
            jnp.zeros(1, dtype=jnp.int32),
            jnp.cumsum(per_atom, dtype=jnp.int32),
        ]
    )
    return neighbor_list, neighbor_ptr, coo_shifts_trim


def tile_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    cell: jax.Array,
    max_neighbors: int | None = None,
    fill_value: int | None = None,
    format: str = "matrix",
    max_pairs: int | None = None,
) -> tuple[jax.Array, ...]:
    """Build a cluster-pair tile neighbor list (one-shot convenience).

    Mirrors :func:`nvalchemiops.torch.neighbors.tile_warp.tile_neighbor_list`.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3), dtype=float32
        Any ``N >= 0``; non-32-aligned ``N`` is padded internally.
    cutoff : float
        Cutoff distance.
    cell : jax.Array, shape (3, 3) or (1, 3, 3), dtype=float32
        Unit cell matrix.
    max_neighbors : int, optional
        Max neighbors per atom (matrix format only).  Falls back to
        :func:`estimate_max_neighbors`.
    fill_value : int, optional
        Matrix sentinel; defaults to ``N``.
    format : {"matrix", "coo", "tile"}, default "matrix"
        Output representation.
    max_pairs : int, optional
        Upper bound for COO output; defaults to ``N * max_neighbors``.

    Returns
    -------
    For ``format == "matrix"``:
        ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``.
    For ``format == "coo"``:
        ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``.
    For ``format == "tile"``:
        ``(num_tiles, tile_row_group, tile_col_group, sorted_atom_index,
        sorted_pos_x, sorted_pos_y, sorted_pos_z)`` — the raw cluster
        state for downstream tile consumers.
    """
    if format not in ("matrix", "coo", "tile"):
        raise ValueError(
            f"format must be 'matrix' | 'coo' | 'tile'; got {format!r}",
        )
    N = positions.shape[0]
    if max_neighbors is None:
        max_neighbors = estimate_max_neighbors(cutoff)

    (
        sorted_atom_index,
        _morton_codes,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        _group_ctr_x,
        _group_ctr_y,
        _group_ctr_z,
        _group_ext_x,
        _group_ext_y,
        _group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
    ) = build_tile_neighbor_list(positions, cutoff, cell)

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
        return tile_to_coo(
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
        )

    # format == "matrix"
    return tile_to_matrix(
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
        int(max_neighbors),
        fill_value=fill_value,
    )
