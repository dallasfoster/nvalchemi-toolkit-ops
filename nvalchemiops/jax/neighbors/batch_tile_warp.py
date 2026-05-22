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

"""JAX bindings for the batched cluster-pair tile neighbor list.

Mirrors :mod:`nvalchemiops.torch.neighbors.batch_tile_warp`: the per-system
Morton sort + padded SoA gather happens in JAX, then the bbox reduction +
tile-pair enumeration and the final conversion (matrix / COO) run on the
Warp side via ``jax_callable`` callbacks.

Scope: triclinic-safe, float32, ``N >= 0``, variable per-system N.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import GraphMode, jax_callable

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.neighbors.tile_batch_warp import (
    TILE_GROUP_SIZE,
)
from nvalchemiops.neighbors.tile_batch_warp import (
    batch_tile_to_coo as _warp_batch_tile_to_coo,
)
from nvalchemiops.neighbors.tile_batch_warp import (
    batch_tile_to_matrix as _warp_batch_tile_to_matrix,
)
from nvalchemiops.neighbors.tile_batch_warp import (
    build_batch_tile_neighbor_list as _warp_build_batch_tile_neighbor_list,
)

__all__ = [
    "TILE_GROUP_SIZE",
    "batch_tile_neighbor_list",
    "batch_tile_to_coo",
    "batch_tile_to_matrix",
    "build_batch_tile_neighbor_list",
    "estimate_batch_tile_neighbor_list_sizes",
]


# =============================================================================
# Sizing helper (pure JAX, no Warp launches)
# =============================================================================
def estimate_batch_tile_neighbor_list_sizes(
    batch_ptr: jax.Array,
    max_tiles_per_group: int = 256,
) -> tuple[int, int, int, int, int]:
    """Estimate allocation sizes for the batched tile neighbor list state.

    Mirrors
    :func:`nvalchemiops.torch.neighbors.batch_tile_warp.estimate_batch_tile_neighbor_list_sizes`.

    Returns ``(n_padded, ngroup, ngroup_padded, max_tiles, num_systems)``.
    The ``.item()`` syncs are necessary to size buffers; cache the result
    if calling from a hot loop.
    """
    num_systems = int(batch_ptr.shape[0]) - 1
    natom_per_system = (batch_ptr[1:] - batch_ptr[:-1]).astype(jnp.int32)
    natom_padded_per_system = (
        (natom_per_system + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    n_padded = int(natom_padded_per_system.sum())
    ngroup = n_padded // TILE_GROUP_SIZE
    ngroup_padded = (
        (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE + TILE_GROUP_SIZE
    max_tiles = ngroup * min(ngroup, max_tiles_per_group)
    return n_padded, ngroup, ngroup_padded, max_tiles, num_systems


# =============================================================================
# Morton helpers (JAX, mirror the torch path)
# =============================================================================
def _spread_10bit(x: jax.Array) -> jax.Array:
    """Spread the low 10 bits of ``x`` across a 30-bit Morton code."""
    x = x & 0x3FF
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


def _per_atom_morton_codes(
    positions: jax.Array,
    batch_idx: jax.Array,
    inv_cell_batch: jax.Array,
) -> jax.Array:
    """Compute per-atom 30-bit Morton code using the per-system inverse cell."""
    inv_per_atom = inv_cell_batch[batch_idx]
    # frac = positions @ inv^T per atom
    frac = jnp.einsum("ij,ikj->ik", positions, inv_per_atom)
    frac = frac - jnp.floor(frac)
    bucket = jnp.clip((frac * 1024.0).astype(jnp.int32), 0, 1023)
    ix, iy, iz = bucket[:, 0], bucket[:, 1], bucket[:, 2]
    return (_spread_10bit(iz) << 2) | (_spread_10bit(iy) << 1) | _spread_10bit(ix)


def _batched_morton_sort_padded(
    positions: jax.Array,
    batch_idx: jax.Array,
    batch_ptr: jax.Array,
    inv_cell_batch: jax.Array,
    n_padded: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-system Morton sort into the padded SoA layout.

    Returns ``(sorted_atom_index, sort_inv, sorted_pos_x, sorted_pos_y,
    sorted_pos_z, batch_idx_sorted)`` plus the running padded prefix
    ``batch_ptr_padded`` (returned last in a 7-tuple actually — see code).
    """
    N = positions.shape[0]
    num_systems = inv_cell_batch.shape[0]

    natom_per_system = batch_ptr[1:] - batch_ptr[:-1]
    natom_padded_per_system = (
        (natom_per_system + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    batch_ptr_padded = jnp.concatenate(
        [
            jnp.zeros(1, dtype=jnp.int32),
            jnp.cumsum(natom_padded_per_system, dtype=jnp.int32),
        ]
    )

    codes = _per_atom_morton_codes(positions, batch_idx, inv_cell_batch)
    # Lexicographic (batch_idx, codes) sort via two stable passes — JAX
    # disables int64 by default, so packing (batch_idx << 32 | codes) into
    # an int64 key would silently truncate and break the per-system
    # ordering for batches with multiple systems.
    perm_morton = jnp.argsort(codes, stable=True)
    bi_after_morton = batch_idx[perm_morton]
    perm_outer = jnp.argsort(bi_after_morton, stable=True)
    sorted_atom_index_real = perm_morton[perm_outer].astype(jnp.int32)

    # Inverse permutation over real atoms.
    sort_inv = jnp.zeros(N, dtype=jnp.int32)
    sort_inv = sort_inv.at[sorted_atom_index_real].set(
        jnp.arange(N, dtype=jnp.int32),
    )

    # Placement indices for each real atom in the padded layout.
    batch_idx_sorted_real = batch_idx[sorted_atom_index_real]
    within_system = jnp.arange(N, dtype=jnp.int32) - batch_ptr[batch_idx_sorted_real]
    padded_slot = batch_ptr_padded[batch_idx_sorted_real] + within_system

    # sorted_atom_index[k] = N is the padding sentinel.
    sp = jnp.full((n_padded,), N, dtype=jnp.int32)
    sp = sp.at[padded_slot].set(sorted_atom_index_real)

    # System index for every padded slot.
    batch_idx_sorted = jnp.repeat(
        jnp.arange(num_systems, dtype=jnp.int32),
        natom_padded_per_system.astype(jnp.int32),
        total_repeat_length=n_padded,
    )

    # Padding slots duplicate each system's first real atom.
    first_atom_per_system = batch_ptr[:-1]
    position_index = jnp.where(
        sp == N,
        first_atom_per_system[batch_idx_sorted],
        sp,
    )
    sorted_pos = positions[position_index]
    return (
        sp,
        sort_inv,
        jnp.asarray(sorted_pos[:, 0]),
        jnp.asarray(sorted_pos[:, 1]),
        jnp.asarray(sorted_pos[:, 2]),
        batch_idx_sorted,
        batch_ptr_padded,
    )


# =============================================================================
# jax_callable wrappers
# =============================================================================
def _build_batch_tile_neighbor_list_callback(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    group_system: wp.array(dtype=wp.int32),
    group_ptr: wp.array(dtype=wp.int32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    cutoff: wp.float32,
) -> None:
    _warp_build_batch_tile_neighbor_list(
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        group_system=group_system,
        group_ptr=group_ptr,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
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
        tile_system=tile_system,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
    )


def _batch_tile_to_matrix_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array(dtype=wp.int32, ndim=3),
    cutoff: wp.float32,
    natom: wp.int32,
    n_tiles: wp.int32,
) -> None:
    _warp_batch_tile_to_matrix(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
        cutoff=float(cutoff),
        natom=int(natom),
        n_tiles=int(n_tiles),
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        wp_dtype=wp.float32,
        device=str(sorted_pos_x.device),
    )


def _batch_tile_to_coo_callback(
    sorted_atom_index: wp.array(dtype=wp.int32),
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    pair_counter: wp.array(dtype=wp.int32),
    coo_list: wp.array(dtype=wp.int32, ndim=2),
    coo_shifts: wp.array(dtype=wp.int32, ndim=2),
    cutoff: wp.float32,
    natom: wp.int32,
    n_tiles: wp.int32,
    max_pairs: wp.int32,
) -> None:
    _warp_batch_tile_to_coo(
        sorted_atom_index=sorted_atom_index,
        sorted_pos_x=sorted_pos_x,
        sorted_pos_y=sorted_pos_y,
        sorted_pos_z=sorted_pos_z,
        cell_batch=cell_batch,
        inv_cell_batch=inv_cell_batch,
        num_tiles=num_tiles,
        tile_row_group=tile_row_group,
        tile_col_group=tile_col_group,
        tile_system=tile_system,
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


_jax_build_batch_tile_neighbor_list = jax_callable(
    _build_batch_tile_neighbor_list_callback,
    num_outputs=10,
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
        "tile_system",
    ],
    graph_mode=GraphMode.WARP,
)

_jax_batch_tile_to_matrix = jax_callable(
    _batch_tile_to_matrix_callback,
    num_outputs=3,
    in_out_argnames=["neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"],
    graph_mode=GraphMode.WARP,
)

_jax_batch_tile_to_coo = jax_callable(
    _batch_tile_to_coo_callback,
    num_outputs=3,
    in_out_argnames=["pair_counter", "coo_list", "coo_shifts"],
    graph_mode=GraphMode.WARP,
)


# =============================================================================
# User-facing entry points
# =============================================================================
def _make_batch_idx(batch_ptr: jax.Array) -> jax.Array:
    """Build per-atom system index from a batch_ptr (cumulative atom counts)."""
    num_systems = int(batch_ptr.shape[0]) - 1
    natom_per_system = (batch_ptr[1:] - batch_ptr[:-1]).astype(jnp.int32)
    return jnp.repeat(
        jnp.arange(num_systems, dtype=jnp.int32),
        natom_per_system,
        total_repeat_length=int(batch_ptr[-1]),
    )


def build_batch_tile_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    cell_batch: jax.Array,
    batch_ptr: jax.Array,
) -> tuple[jax.Array, ...]:
    """Build the batched tile neighbor list state.

    Runs per-system Morton sort + padded SoA gather in JAX, then the
    bbox reduction + tile-pair enumeration on the Warp side.

    Parameters
    ----------
    positions : jax.Array, shape (N, 3), dtype=float32
        Concatenated atomic coordinates for all systems.
    cutoff : float
        Bbox cutoff distance.
    cell_batch : jax.Array, shape (S, 3, 3), dtype=float32
        Per-system unit cell matrices.
    batch_ptr : jax.Array, shape (S + 1,), dtype=int32
        Cumulative atom counts.

    Returns
    -------
    tuple of jax.Array
        ``(sorted_atom_index, sort_inv, sorted_pos_x, sorted_pos_y,
        sorted_pos_z, batch_idx_sorted, batch_ptr_padded, group_system,
        group_ptr, group_ctr_x, group_ctr_y, group_ctr_z, group_ext_x,
        group_ext_y, group_ext_z, num_tiles, tile_row_group,
        tile_col_group, tile_system)``.
    """
    if positions.dtype != jnp.float32:
        raise TypeError(
            "positions must be float32 (batch_tile_warp kernels are float32 only)"
        )
    if cell_batch.dtype != jnp.float32:
        raise TypeError("cell_batch must be float32")
    if cell_batch.ndim != 3 or cell_batch.shape[1:] != (3, 3):
        raise ValueError(
            f"cell_batch must have shape (S, 3, 3); got {cell_batch.shape}"
        )
    if batch_ptr.dtype != jnp.int32:
        batch_ptr = batch_ptr.astype(jnp.int32)

    n_padded, ngroup, ngroup_padded, max_tiles, _num_systems = (
        estimate_batch_tile_neighbor_list_sizes(batch_ptr)
    )

    inv_cell_batch = jnp.linalg.inv(cell_batch)
    batch_idx = _make_batch_idx(batch_ptr)

    (
        sorted_atom_index,
        sort_inv,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        batch_idx_sorted,
        batch_ptr_padded,
    ) = _batched_morton_sort_padded(
        positions,
        batch_idx,
        batch_ptr,
        inv_cell_batch,
        n_padded,
    )

    # Per-tile derived structure.
    group_ptr = (batch_ptr_padded // TILE_GROUP_SIZE).astype(jnp.int32)
    group_system = batch_idx_sorted[::TILE_GROUP_SIZE]

    group_ctr_x = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ctr_y = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ctr_z = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ext_x = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ext_y = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    group_ext_z = jnp.zeros(ngroup_padded, dtype=jnp.float32)
    num_tiles = jnp.zeros(1, dtype=jnp.int32)
    tile_row_group = jnp.zeros(max_tiles, dtype=jnp.int32)
    tile_col_group = jnp.zeros(max_tiles, dtype=jnp.int32)
    tile_system = jnp.zeros(max_tiles, dtype=jnp.int32)

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
        tile_system,
    ) = _jax_build_batch_tile_neighbor_list(
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        group_system,
        group_ptr,
        cell_batch,
        inv_cell_batch,
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
        float(cutoff),
    )

    del ngroup  # implicit in group_system.shape[0]
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


def batch_tile_to_matrix(
    sorted_atom_index: jax.Array,
    sorted_pos_x: jax.Array,
    sorted_pos_y: jax.Array,
    sorted_pos_z: jax.Array,
    cell_batch: jax.Array,
    num_tiles: jax.Array,
    tile_row_group: jax.Array,
    tile_col_group: jax.Array,
    tile_system: jax.Array,
    cutoff: float,
    natom: int,
    max_neighbors: int,
    *,
    fill_value: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert the batched tile pair list to dense neighbor-matrix form."""
    if fill_value is None:
        fill_value = natom
    inv_cell_batch = jnp.linalg.inv(cell_batch)

    neighbor_matrix = jnp.zeros((natom, max_neighbors), dtype=jnp.int32)
    num_neighbors = jnp.zeros(natom, dtype=jnp.int32)
    neighbor_matrix_shifts = jnp.zeros((natom, max_neighbors, 3), dtype=jnp.int32)

    n_tiles_host = int(num_tiles[0])
    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = _jax_batch_tile_to_matrix(
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
        float(cutoff),
        int(natom),
        n_tiles_host,
    )

    col_idx = jnp.arange(max_neighbors, dtype=jnp.int32)[jnp.newaxis, :]
    active = col_idx < num_neighbors[:, jnp.newaxis]
    neighbor_matrix = jnp.where(active, neighbor_matrix, jnp.int32(fill_value))
    return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


def batch_tile_to_coo(
    sorted_atom_index: jax.Array,
    sorted_pos_x: jax.Array,
    sorted_pos_y: jax.Array,
    sorted_pos_z: jax.Array,
    cell_batch: jax.Array,
    num_tiles: jax.Array,
    tile_row_group: jax.Array,
    tile_col_group: jax.Array,
    tile_system: jax.Array,
    cutoff: float,
    natom: int,
    max_pairs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert the batched tile pair list to flat COO form."""
    inv_cell_batch = jnp.linalg.inv(cell_batch)

    pair_counter = jnp.zeros(1, dtype=jnp.int32)
    coo_list = jnp.zeros((max_pairs, 2), dtype=jnp.int32)
    coo_shifts = jnp.zeros((max_pairs, 3), dtype=jnp.int32)

    n_tiles_host = int(num_tiles[0])
    pair_counter, coo_list, coo_shifts = _jax_batch_tile_to_coo(
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
        float(cutoff),
        int(natom),
        n_tiles_host,
        int(max_pairs),
    )

    npairs = int(pair_counter[0])
    coo_list_trim = coo_list[:npairs]
    coo_shifts_trim = coo_shifts[:npairs]
    neighbor_list = coo_list_trim.T  # (2, npairs)
    per_atom = jnp.bincount(neighbor_list[0], length=natom).astype(jnp.int32)
    neighbor_ptr = jnp.concatenate(
        [
            jnp.zeros(1, dtype=jnp.int32),
            jnp.cumsum(per_atom, dtype=jnp.int32),
        ]
    )
    return neighbor_list, neighbor_ptr, coo_shifts_trim


def batch_tile_neighbor_list(
    positions: jax.Array,
    cutoff: float,
    cell_batch: jax.Array,
    batch_ptr: jax.Array,
    max_neighbors: int | None = None,
    fill_value: int | None = None,
    format: str = "matrix",
    max_pairs: int | None = None,
) -> tuple[jax.Array, ...]:
    """Build a batched cluster-pair tile neighbor list (one-shot convenience).

    Mirrors
    :func:`nvalchemiops.torch.neighbors.batch_tile_warp.batch_tile_neighbor_list`.
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
        _sort_inv,
        sorted_pos_x,
        sorted_pos_y,
        sorted_pos_z,
        _batch_idx_sorted,
        _batch_ptr_padded,
        _group_system,
        _group_ptr,
        _group_ctr_x,
        _group_ctr_y,
        _group_ctr_z,
        _group_ext_x,
        _group_ext_y,
        _group_ext_z,
        num_tiles,
        tile_row_group,
        tile_col_group,
        tile_system,
    ) = build_batch_tile_neighbor_list(positions, cutoff, cell_batch, batch_ptr)

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
        )

    if format == "coo":
        if max_pairs is None:
            max_pairs = N * max_neighbors
        return batch_tile_to_coo(
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
        )

    return batch_tile_to_matrix(
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
        int(max_neighbors),
        fill_value=fill_value,
    )
