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

"""Core warp kernels and launchers for batched naive neighbor list construction.

This module contains warp kernels for batched O(N²) neighbor list computation.
See `nvalchemiops.torch.neighbors` for PyTorch bindings.
"""

from typing import Any

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    _decode_shift_index,
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
    compute_inv_cells,
    selective_zero_num_neighbors,
    wrap_positions_batch,
)

__all__ = [
    "batch_naive_neighbor_matrix",
    "batch_naive_neighbor_matrix_pbc",
]

###########################################################################################
########################### Batch Naive Neighbor List Kernels ############################
###########################################################################################


@wp.func
def _batch_naive_neighbor_body(
    tid: int,
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
):
    isys = batch_idx[tid]
    j_end = batch_ptr[isys + 1]
    positions_i = positions[tid]
    max_neighbors = neighbor_matrix.shape[1]
    for j in range(tid + 1, j_end):
        diff = positions_i - positions[j]
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            _update_neighbor_matrix(
                tid, j, neighbor_matrix, num_neighbors, max_neighbors, half_fill
            )


@wp.func
def _batch_naive_neighbor_pbc_body(
    shift: wp.vec3i,
    iatom_global: int,
    isys: int,
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
):
    jatom_start = batch_ptr[isys]
    jatom_end = batch_ptr[isys + 1]
    maxnb = neighbor_matrix.shape[1]
    _cell = cell[isys]
    _pos_i = positions[iatom_global]
    _int_i = per_atom_cell_offsets[iatom_global]
    positions_shifted = type(_cell[0])(shift) * _cell + _pos_i
    _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
    if _zero_shift:
        jatom_end = iatom_global
    for jatom in range(jatom_start, jatom_end):
        _pos_j = positions[jatom]
        diff = positions_shifted - _pos_j
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            # Correct the stored shift so that dist = pos_i - pos_j - shift*cell
            # holds for the original (potentially unwrapped) positions.
            _int_j = per_atom_cell_offsets[jatom]
            _corrected_shift = wp.vec3i(
                shift[0] - _int_i[0] + _int_j[0],
                shift[1] - _int_i[1] + _int_j[1],
                shift[2] - _int_i[2] + _int_j[2],
            )
            _update_neighbor_matrix_pbc(
                jatom,
                iatom_global,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                _corrected_shift,
                maxnb,
                half_fill,
            )


@wp.func
def _batch_naive_neighbor_pbc_body_prewrapped(
    shift: wp.vec3i,
    iatom_global: int,
    isys: int,
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
):
    jatom_start = batch_ptr[isys]
    jatom_end = batch_ptr[isys + 1]
    maxnb = neighbor_matrix.shape[1]
    _cell = cell[isys]
    _pos_i = positions[iatom_global]
    positions_shifted = type(_cell[0])(shift) * _cell + _pos_i
    _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
    if _zero_shift:
        jatom_end = iatom_global
    for jatom in range(jatom_start, jatom_end):
        _pos_j = positions[jatom]
        diff = positions_shifted - _pos_j
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            _update_neighbor_matrix_pbc(
                jatom,
                iatom_global,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                shift,
                maxnb,
                half_fill,
            )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate batch neighbor matrix using naive O(N^2) algorithm.

    Computes pairwise distances between atoms within each system in a batch
    and identifies neighbors within the specified cutoff distance. Atoms from
    different systems do not interact. No periodic boundary conditions are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
        Each row represents one atom's (x, y, z) position.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection in Cartesian units.
        Atoms within this distance are considered neighbors.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom. Atoms with the same index belong to
        the same system and can be neighbors.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
        System i contains atoms from batch_ptr[i] to batch_ptr[i+1]-1.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Entries are filled with atom indices, remaining entries stay as initialized.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
        Updated in-place with actual neighbor counts.
    half_fill : wp.bool
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - neighbor_matrix : Filled with neighbor atom indices
        - num_neighbors : Updated with neighbor counts per atom

    See Also
    --------
    _fill_naive_neighbor_matrix : Single system version
    _fill_batch_naive_neighbor_matrix_pbc : Version with periodic boundary conditions
    """
    tid = wp.tid()
    _batch_naive_neighbor_body(
        tid,
        positions,
        cutoff_sq,
        batch_idx,
        batch_ptr,
        neighbor_matrix,
        num_neighbors,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_pbc(
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate batch neighbor matrix with PBC using naive O(N^2) algorithm.

    Computes neighbor relationships between atoms across periodic boundaries by
    considering all periodic images within the cutoff distance. Processes multiple
    systems in a batch, where each system can have different periodic cells.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Assumed to be wrapped into the primary cell.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Integer cell offsets for each atom.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices for each system.
    cutoff_sq : float
        Squared cutoff distance.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Shift range per dimension per system.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Number of shifts per system (for bounds checking).
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Shift vectors for each neighbor.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors per atom.
    half_fill : wp.bool
        If True, only store half of the neighbor relationships.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - neighbor_matrix : Filled with neighbor atom indices
        - neighbor_matrix_shifts : Filled with corresponding shift vectors
        - num_neighbors : Updated with neighbor counts per atom

    Notes
    -----
    - Thread launch: 3D (num_systems, max_shifts_per_system, max_atoms_per_system)

    See Also
    --------
    _fill_batch_naive_neighbor_matrix : Version without periodic boundary conditions
    _fill_naive_neighbor_matrix_pbc : Single system version
    """
    isys, ishift_local, iatom = wp.tid()

    if ishift_local >= num_shifts_arr[isys]:
        return

    _natom = batch_ptr[isys + 1] - batch_ptr[isys]

    if iatom >= _natom:
        return

    iatom_global = iatom + batch_ptr[isys]
    shift = _decode_shift_index(ishift_local, shift_range[isys])
    _batch_naive_neighbor_pbc_body(
        shift,
        iatom_global,
        isys,
        positions,
        per_atom_cell_offsets,
        cell,
        cutoff_sq,
        batch_ptr,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_pbc_prewrapped(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Batch PBC neighbor matrix for pre-wrapped positions (no cell-offset correction).

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Positions already wrapped into the primary cell.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices for each system.
    cutoff_sq : float
        Squared cutoff distance.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Shift range per dimension per system.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Number of shifts per system.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Shift vectors for each neighbor.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors per atom.
    half_fill : wp.bool
        If True, only store half of the neighbor relationships.

    Notes
    -----
    - Thread launch: 3D (num_systems, max_shifts_per_system, max_atoms_per_system)
    """
    isys, ishift_local, iatom = wp.tid()

    if ishift_local >= num_shifts_arr[isys]:
        return

    _natom = batch_ptr[isys + 1] - batch_ptr[isys]

    if iatom >= _natom:
        return

    iatom_global = iatom + batch_ptr[isys]
    shift = _decode_shift_index(ishift_local, shift_range[isys])
    _batch_naive_neighbor_pbc_body_prewrapped(
        shift,
        iatom_global,
        isys,
        positions,
        cell,
        cutoff_sq,
        batch_ptr,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )


T = [wp.float32, wp.float64, wp.float16]
V = [wp.vec3f, wp.vec3d, wp.vec3h]
M = [wp.mat33f, wp.mat33d, wp.mat33h]
_fill_batch_naive_neighbor_matrix_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_prewrapped_overload = {}
for t, v, m in zip(T, V, M):
    _fill_batch_naive_neighbor_matrix_overload[t] = wp.overload(
        _fill_batch_naive_neighbor_matrix,
        [
            wp.array(dtype=v),
            t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_batch_naive_neighbor_matrix_pbc_overload[t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_pbc,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=m),
            t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.vec3i, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_overload[t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_pbc_prewrapped,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.vec3i, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )

###########################################################################################
########################### Selective Skip Kernels #######################################
###########################################################################################


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_selective(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective batch naive neighbor matrix kernel - skips non-rebuilt systems.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    half_fill : wp.bool
        If True, only store relationships where i < j.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system flags. Only systems with True are processed.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Atoms in systems where rebuild_flags[isys] is False are skipped
    """
    tid = wp.tid()
    isys = batch_idx[tid]
    if not rebuild_flags[isys]:
        return
    _batch_naive_neighbor_body(
        tid,
        positions,
        cutoff_sq,
        batch_idx,
        batch_ptr,
        neighbor_matrix,
        num_neighbors,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_pbc_selective(
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective batch naive PBC neighbor matrix kernel - skips non-rebuilt systems.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Assumed to be wrapped into the primary cell.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Integer cell offsets for each atom.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices for each system.
    cutoff_sq : float
        Squared cutoff distance.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Shift range per dimension per system.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Number of shifts per system.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Shift vectors for each neighbor.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors per atom.
    half_fill : wp.bool
        If True, only store half of the neighbor relationships.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system flags. Only systems with True are processed.

    Notes
    -----
    - Thread launch: 3D (num_systems, max_shifts_per_system, max_atoms_per_system)
    """
    isys, ishift_local, iatom = wp.tid()

    if not rebuild_flags[isys]:
        return

    if ishift_local >= num_shifts_arr[isys]:
        return

    _natom = batch_ptr[isys + 1] - batch_ptr[isys]

    if iatom >= _natom:
        return

    iatom_global = iatom + batch_ptr[isys]
    shift = _decode_shift_index(ishift_local, shift_range[isys])
    _batch_naive_neighbor_pbc_body(
        shift,
        iatom_global,
        isys,
        positions,
        per_atom_cell_offsets,
        cell,
        cutoff_sq,
        batch_ptr,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )


@wp.kernel(enable_backward=False)
def _fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_ptr: wp.array(dtype=wp.int32),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective batch PBC kernel for pre-wrapped positions - skips non-rebuilt systems.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Positions already wrapped into the primary cell.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices for each system.
    cutoff_sq : float
        Squared cutoff distance.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Shift range per dimension per system.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Number of shifts per system.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Shift vectors for each neighbor.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors per atom.
    half_fill : wp.bool
        If True, only store half of the neighbor relationships.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system flags. Only systems with True are processed.

    Notes
    -----
    - Thread launch: 3D (num_systems, max_shifts_per_system, max_atoms_per_system)
    """
    isys, ishift_local, iatom = wp.tid()

    if not rebuild_flags[isys]:
        return

    if ishift_local >= num_shifts_arr[isys]:
        return

    _natom = batch_ptr[isys + 1] - batch_ptr[isys]

    if iatom >= _natom:
        return

    iatom_global = iatom + batch_ptr[isys]
    shift = _decode_shift_index(ishift_local, shift_range[isys])
    _batch_naive_neighbor_pbc_body_prewrapped(
        shift,
        iatom_global,
        isys,
        positions,
        cell,
        cutoff_sq,
        batch_ptr,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )


_fill_batch_naive_neighbor_matrix_selective_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_selective_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective_overload = {}
for t, v, m in zip(T, V, M):
    _fill_batch_naive_neighbor_matrix_selective_overload[t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_selective,
        [
            wp.array(dtype=v),
            t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.bool),
        ],
    )
    _fill_batch_naive_neighbor_matrix_pbc_selective_overload[t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_pbc_selective,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=m),
            t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.vec3i, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.bool),
        ],
    )
    _fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective_overload[t] = (
        wp.overload(
            _fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective,
            [
                wp.array(dtype=v),
                wp.array(dtype=m),
                t,
                wp.array(dtype=wp.int32),
                wp.array(dtype=wp.vec3i),
                wp.array(dtype=wp.int32),
                wp.array(dtype=wp.int32, ndim=2),
                wp.array(dtype=wp.vec3i, ndim=2),
                wp.array(dtype=wp.int32),
                wp.bool,
                wp.array(dtype=wp.bool),
            ],
        )
    )

###########################################################################################
########################### Warp Launchers ###############################################
###########################################################################################


def batch_naive_neighbor_matrix(
    positions: wp.array,
    cutoff: float,
    batch_idx: wp.array,
    batch_ptr: wp.array,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
) -> None:
    """Core warp launcher for batched naive neighbor matrix construction (no PBC).

    Computes pairwise distances and fills the neighbor matrix for multiple systems
    in a batch using pure warp operations. No periodic boundary conditions are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j to avoid double counting.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool, optional
        Per-system rebuild flags. If provided, only systems where rebuild_flags[i]
        is True are processed; others are skipped on the GPU without CPU sync.
        Call selective_zero_num_neighbors before this launcher to reset counts.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.

    See Also
    --------
    batch_naive_neighbor_matrix_pbc : Version with periodic boundary conditions
    _fill_batch_naive_neighbor_matrix : Scalar (CPU) kernel
    _fill_batch_naive_neighbor_matrix_tiled : Tile-cooperative (CUDA) kernel

    Notes
    -----
    Dispatches internally:

    - On CPU, always uses the scalar kernel (Warp forces ``block_dim=1`` on CPU).
    - On CUDA, uses the tile-cooperative kernel when the adaptive ``use_tiled``
      heuristic favors it (``total_atoms >= 256 * num_systems`` with a tighter
      threshold above 12 288 atoms), otherwise falls back to the scalar kernel.
    """
    total_atoms = positions.shape[0]
    cutoff_sq = wp_dtype(cutoff * cutoff)
    is_cpu = "cpu" in str(device).lower()

    if is_cpu or total_atoms < 2048:
        use_tiled = False
    else:
        num_systems = batch_ptr.shape[0] - 1
        use_tiled = total_atoms >= 256 * num_systems
        if use_tiled and total_atoms > 12288:
            use_tiled = total_atoms >= 512 * num_systems

    if rebuild_flags is not None:
        selective_zero_num_neighbors(num_neighbors, batch_idx, rebuild_flags, device)
        if use_tiled:
            wp.launch_tiled(
                kernel=_fill_batch_naive_neighbor_matrix_tiled_selective_overload[
                    wp_dtype
                ],
                dim=[total_atoms],
                inputs=[
                    positions,
                    cutoff_sq,
                    batch_idx,
                    batch_ptr,
                    neighbor_matrix,
                    num_neighbors,
                    half_fill,
                    rebuild_flags,
                ],
                block_dim=BLOCK_DIM,
                device=device,
            )
        else:
            wp.launch(
                kernel=_fill_batch_naive_neighbor_matrix_selective_overload[wp_dtype],
                dim=total_atoms,
                inputs=[
                    positions,
                    cutoff_sq,
                    batch_idx,
                    batch_ptr,
                    neighbor_matrix,
                    num_neighbors,
                    half_fill,
                    rebuild_flags,
                ],
                device=device,
            )
    elif use_tiled:
        wp.launch_tiled(
            kernel=_fill_batch_naive_neighbor_matrix_tiled_overload[wp_dtype],
            dim=[total_atoms],
            inputs=[
                positions,
                cutoff_sq,
                batch_idx,
                batch_ptr,
                neighbor_matrix,
                num_neighbors,
                half_fill,
            ],
            block_dim=BLOCK_DIM,
            device=device,
        )
    else:
        wp.launch(
            kernel=_fill_batch_naive_neighbor_matrix_overload[wp_dtype],
            dim=total_atoms,
            inputs=[
                positions,
                cutoff_sq,
                batch_idx,
                batch_ptr,
                neighbor_matrix,
                num_neighbors,
                half_fill,
            ],
            device=device,
        )


def batch_naive_neighbor_matrix_pbc(
    positions: wp.array,
    cell: wp.array,
    cutoff: float,
    batch_ptr: wp.array,
    batch_idx: wp.array,
    shift_range: wp.array,
    num_shifts_arr: wp.array,
    max_shifts_per_system: int,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    max_atoms_per_system: int,
    half_fill: bool = False,
    rebuild_flags: wp.array | None = None,
    wrap_positions: bool = True,
) -> None:
    """Core warp launcher for batched naive neighbor matrix construction with PBC.

    Computes neighbor relationships between atoms across periodic boundaries for
    multiple systems in a batch using pure warp operations.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices for each system.
    cutoff : float
        Cutoff distance for neighbor detection.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts defining system boundaries.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Shift range per dimension per system.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Number of shifts per system.
    max_shifts_per_system : int
        Maximum per-system shift count (launch dimension).
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Shift vectors for each neighbor.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors per atom.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    max_atoms_per_system : int
        Maximum number of atoms in any single system.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool, optional
        Per-system rebuild flags.
    wrap_positions : bool, default=True
        If True, wrap input positions into the primary cell.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - When ``wrap_positions`` is True, positions are wrapped into the primary cell in a
      preprocessing step before the neighbor search kernel.

    See Also
    --------
    batch_naive_neighbor_matrix : Version without periodic boundary conditions
    _fill_batch_naive_neighbor_matrix_pbc : Scalar (CPU) PBC kernel
    _fill_batch_naive_neighbor_matrix_pbc_tiled : Tile-cooperative (CUDA) PBC kernel
    wrap_positions_batch : Preprocessing step that wraps positions

    Notes
    -----
    Dispatches internally:

    - On CPU, always uses the scalar 3D-launch kernels (Warp forces
      ``block_dim=1`` on CPU).
    - On CUDA with ``wrap_positions=True``, uses the tile-cooperative kernel
      when the adaptive ``use_tiled`` heuristic favors it
      (``256 <= avg_atoms_per_system < 6144`` and either
      ``avg_atoms_per_system >= 2048`` or ``total_atoms <= 8192``), otherwise
      falls back to the scalar 3D-launch kernel.
    - When ``wrap_positions=False`` the prewrapped scalar kernels are used on
      both devices (no tiled prewrapped variant).
    """
    total_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    cutoff_sq = wp_dtype(cutoff * cutoff)
    is_cpu = "cpu" in str(device).lower()

    if wrap_positions:
        wp_mat_dtype = (
            wp.mat33f
            if wp_dtype == wp.float32
            else wp.mat33d
            if wp_dtype == wp.float64
            else wp.mat33h
            if wp_dtype == wp.float16
            else None
        )
        wp_vec_dtype = (
            wp.vec3f
            if wp_dtype == wp.float32
            else wp.vec3d
            if wp_dtype == wp.float64
            else wp.vec3h
            if wp_dtype == wp.float16
            else None
        )
        inv_cell = wp.empty((cell.shape[0],), dtype=wp_mat_dtype, device=device)
        compute_inv_cells(cell, inv_cell, wp_dtype, device)
        positions_wrapped = wp.empty((total_atoms,), dtype=wp_vec_dtype, device=device)
        per_atom_cell_offsets = wp.empty(total_atoms, dtype=wp.vec3i, device=device)
        wrap_positions_batch(
            positions,
            cell,
            inv_cell,
            batch_idx,
            positions_wrapped,
            per_atom_cell_offsets,
            wp_dtype,
            device,
        )

        if is_cpu:
            use_tiled = False
        else:
            avg_atoms_per_system = total_atoms // max(num_systems, 1)
            use_tiled = 256 <= avg_atoms_per_system < 6144 and (
                avg_atoms_per_system >= 2048 or total_atoms <= 8192
            )

        if rebuild_flags is not None:
            selective_zero_num_neighbors(
                num_neighbors, batch_idx, rebuild_flags, device
            )
            if use_tiled:
                wp.launch_tiled(
                    kernel=_fill_batch_naive_neighbor_matrix_pbc_tiled_selective_overload[
                        wp_dtype
                    ],
                    dim=[total_atoms],
                    inputs=[
                        positions_wrapped,
                        per_atom_cell_offsets,
                        cell,
                        cutoff_sq,
                        batch_idx,
                        batch_ptr,
                        shift_range,
                        num_shifts_arr,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        half_fill,
                        rebuild_flags,
                    ],
                    block_dim=BLOCK_DIM,
                    device=device,
                )
            else:
                wp.launch(
                    kernel=_fill_batch_naive_neighbor_matrix_pbc_selective_overload[
                        wp_dtype
                    ],
                    dim=(num_systems, max_shifts_per_system, max_atoms_per_system),
                    inputs=[
                        positions_wrapped,
                        per_atom_cell_offsets,
                        cell,
                        cutoff_sq,
                        batch_ptr,
                        shift_range,
                        num_shifts_arr,
                        neighbor_matrix,
                        neighbor_matrix_shifts,
                        num_neighbors,
                        half_fill,
                        rebuild_flags,
                    ],
                    device=device,
                )
        elif use_tiled:
            wp.launch_tiled(
                kernel=_fill_batch_naive_neighbor_matrix_pbc_tiled_overload[wp_dtype],
                dim=[total_atoms],
                inputs=[
                    positions_wrapped,
                    per_atom_cell_offsets,
                    cell,
                    cutoff_sq,
                    batch_idx,
                    batch_ptr,
                    shift_range,
                    num_shifts_arr,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    half_fill,
                ],
                block_dim=BLOCK_DIM,
                device=device,
            )
        else:
            wp.launch(
                kernel=_fill_batch_naive_neighbor_matrix_pbc_overload[wp_dtype],
                dim=(num_systems, max_shifts_per_system, max_atoms_per_system),
                inputs=[
                    positions_wrapped,
                    per_atom_cell_offsets,
                    cell,
                    cutoff_sq,
                    batch_ptr,
                    shift_range,
                    num_shifts_arr,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    half_fill,
                ],
                device=device,
            )
    else:
        if rebuild_flags is not None:
            selective_zero_num_neighbors(
                num_neighbors, batch_idx, rebuild_flags, device
            )
            wp.launch(
                kernel=_fill_batch_naive_neighbor_matrix_pbc_prewrapped_selective_overload[
                    wp_dtype
                ],
                dim=(num_systems, max_shifts_per_system, max_atoms_per_system),
                inputs=[
                    positions,
                    cell,
                    cutoff_sq,
                    batch_ptr,
                    shift_range,
                    num_shifts_arr,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    half_fill,
                    rebuild_flags,
                ],
                device=device,
            )
        else:
            wp.launch(
                kernel=_fill_batch_naive_neighbor_matrix_pbc_prewrapped_overload[
                    wp_dtype
                ],
                dim=(num_systems, max_shifts_per_system, max_atoms_per_system),
                inputs=[
                    positions,
                    cell,
                    cutoff_sq,
                    batch_ptr,
                    shift_range,
                    num_shifts_arr,
                    neighbor_matrix,
                    neighbor_matrix_shifts,
                    num_neighbors,
                    half_fill,
                ],
                device=device,
            )


# =============================================================================
# Tile-cooperative variants
# =============================================================================
# The tiled kernels below are CUDA-only -- they rely on ``wp.launch_tiled``
# with ``block_dim = BLOCK_DIM`` threads cooperatively sweeping the j-loop.
# On CPU, Warp forces ``block_dim = 1``, which would silently break the
# lane-cooperative work partitioning.  The public launchers above branch on
# ``device`` and pick the scalar kernel on CPU and the tiled kernel on CUDA.

BLOCK_DIM = 64  # threads per block; must be a power of 2.


@wp.kernel(enable_backward=False, module="batch_naive_tiled")
def _fill_batch_naive_neighbor_matrix_tiled(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Tiled batched naive neighbor matrix kernel (no PBC).

    One block per atom *i*; ``BLOCK_DIM`` threads cooperatively sweep the
    j-loop in chunks.  Each lane checks its own j-atom and emits via
    per-pair atomics; the j-side mirror atomic is skipped when ``half_fill``.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Concatenated Cartesian coordinates for all systems.
    cutoff_sq : float
        Squared cutoff distance.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    batch_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        Cumulative atom counts.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Per-atom neighbor counts.
    half_fill : wp.bool
        If True, only store pairs where i < j.
    """
    i = wp.tid()
    isys = batch_idx[i]
    j_end = batch_ptr[isys + 1]
    N = positions.shape[0]
    max_nb = neighbor_matrix.shape[1]
    pos_i = positions[i]

    lane_tile = wp.tile_arange(BLOCK_DIM, dtype=wp.int32)
    lane = wp.untile(lane_tile)

    for chunk_start in range(i + 1, j_end, BLOCK_DIM):
        j_idx = chunk_start + lane
        safe_idx = wp.min(j_idx, N - 1)
        pos_j = positions[safe_idx]
        diff = pos_i - pos_j
        dist_sq = wp.dot(diff, diff)

        if j_idx < j_end and dist_sq < cutoff_sq:
            write_pos = wp.atomic_add(num_neighbors, i, 1)
            if write_pos < max_nb:
                neighbor_matrix[i, write_pos] = j_idx

            if not half_fill:
                rev_pos = wp.atomic_add(num_neighbors, j_idx, 1)
                if rev_pos < max_nb:
                    neighbor_matrix[j_idx, rev_pos] = i


@wp.kernel(enable_backward=False, module="batch_naive_tiled")
def _fill_batch_naive_neighbor_matrix_tiled_selective(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective tiled batched naive neighbor matrix kernel (no PBC).

    Identical to ``_fill_batch_naive_neighbor_matrix_tiled`` but skips
    atoms whose system has ``rebuild_flags[isys] == False``.

    Parameters
    ----------
    positions, cutoff_sq, batch_idx, batch_ptr, neighbor_matrix,
    num_neighbors, half_fill :
        Same as ``_fill_batch_naive_neighbor_matrix_tiled``.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system rebuild flags.
    """
    i = wp.tid()
    isys = batch_idx[i]
    if not rebuild_flags[isys]:
        return

    j_end = batch_ptr[isys + 1]
    N = positions.shape[0]
    max_nb = neighbor_matrix.shape[1]
    pos_i = positions[i]

    lane_tile = wp.tile_arange(BLOCK_DIM, dtype=wp.int32)
    lane = wp.untile(lane_tile)

    for chunk_start in range(i + 1, j_end, BLOCK_DIM):
        j_idx = chunk_start + lane
        safe_idx = wp.min(j_idx, N - 1)
        pos_j = positions[safe_idx]
        diff = pos_i - pos_j
        dist_sq = wp.dot(diff, diff)

        if j_idx < j_end and dist_sq < cutoff_sq:
            write_pos = wp.atomic_add(num_neighbors, i, 1)
            if write_pos < max_nb:
                neighbor_matrix[i, write_pos] = j_idx

            if not half_fill:
                rev_pos = wp.atomic_add(num_neighbors, j_idx, 1)
                if rev_pos < max_nb:
                    neighbor_matrix[j_idx, rev_pos] = i


@wp.kernel(enable_backward=False, module="batch_naive_tiled")
def _fill_batch_naive_neighbor_matrix_pbc_tiled(
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Tiled batched naive neighbor matrix kernel with PBC.

    One block per atom *i*; ``BLOCK_DIM`` threads cooperatively sweep the
    j-loop for each periodic shift.  Zero-shift restricts j < i to avoid
    self-pairing; nonzero shifts sweep the full system range.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Wrapped Cartesian coordinates.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Integer cell offsets from wrapping.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Per-system cell matrices.
    cutoff_sq : float
        Squared cutoff distance.
    batch_idx, batch_ptr :
        System index and atom-range arrays.
    shift_range : wp.array, shape (num_systems,), dtype=wp.vec3i
        Per-system shift ranges.
    num_shifts_arr : wp.array, shape (num_systems,), dtype=wp.int32
        Per-system shift counts.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Shift vector for each neighbor.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Per-atom neighbor counts.
    half_fill : wp.bool
        If True, only emit i→j (skip mirror j→i).
    """
    iatom = wp.tid()
    isys = batch_idx[iatom]
    j_start = batch_ptr[isys]
    j_end = batch_ptr[isys + 1]
    N = positions.shape[0]
    max_nb = neighbor_matrix.shape[1]

    _cell = cell[isys]
    pos_i = positions[iatom]
    int_i = per_atom_cell_offsets[iatom]
    n_shifts = num_shifts_arr[isys]
    _shift_range = shift_range[isys]

    lane_tile = wp.tile_arange(BLOCK_DIM, dtype=wp.int32)
    lane = wp.untile(lane_tile)

    for ishift in range(n_shifts):
        shift = _decode_shift_index(ishift, _shift_range)

        pos_i_shifted = type(_cell[0])(shift) * _cell + pos_i
        _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
        _j_end = iatom if _zero_shift else j_end

        for chunk_start in range(j_start, _j_end, BLOCK_DIM):
            j_idx = chunk_start + lane
            safe_j = wp.min(j_idx, N - 1)
            pos_j = positions[safe_j]
            diff = pos_i_shifted - pos_j
            dist_sq = wp.dot(diff, diff)

            if j_idx < _j_end and dist_sq < cutoff_sq:
                int_j = per_atom_cell_offsets[j_idx]
                corrected_shift = wp.vec3i(
                    shift[0] - int_i[0] + int_j[0],
                    shift[1] - int_i[1] + int_j[1],
                    shift[2] - int_i[2] + int_j[2],
                )
                write_pos = wp.atomic_add(num_neighbors, iatom, 1)
                if write_pos < max_nb:
                    neighbor_matrix[iatom, write_pos] = j_idx
                    neighbor_matrix_shifts[iatom, write_pos] = corrected_shift
                if not half_fill:
                    rev_pos = wp.atomic_add(num_neighbors, j_idx, 1)
                    if rev_pos < max_nb:
                        neighbor_matrix[j_idx, rev_pos] = iatom
                        neighbor_matrix_shifts[j_idx, rev_pos] = -corrected_shift


@wp.kernel(enable_backward=False, module="batch_naive_tiled")
def _fill_batch_naive_neighbor_matrix_pbc_tiled_selective(
    positions: wp.array(dtype=Any),
    per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    cutoff_sq: Any,
    batch_idx: wp.array(dtype=wp.int32),
    batch_ptr: wp.array(dtype=wp.int32),
    shift_range: wp.array(dtype=wp.vec3i),
    num_shifts_arr: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective tiled batched naive neighbor matrix kernel with PBC.

    Identical to ``_fill_batch_naive_neighbor_matrix_pbc_tiled`` but skips
    atoms whose system has ``rebuild_flags[isys] == False``.

    Parameters
    ----------
    See ``_fill_batch_naive_neighbor_matrix_pbc_tiled``.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system rebuild flags.
    """
    iatom = wp.tid()
    isys = batch_idx[iatom]
    if not rebuild_flags[isys]:
        return

    j_start = batch_ptr[isys]
    j_end = batch_ptr[isys + 1]
    N = positions.shape[0]
    max_nb = neighbor_matrix.shape[1]

    _cell = cell[isys]
    pos_i = positions[iatom]
    int_i = per_atom_cell_offsets[iatom]
    n_shifts = num_shifts_arr[isys]
    _shift_range = shift_range[isys]

    lane_tile = wp.tile_arange(BLOCK_DIM, dtype=wp.int32)
    lane = wp.untile(lane_tile)

    for ishift in range(n_shifts):
        shift = _decode_shift_index(ishift, _shift_range)

        pos_i_shifted = type(_cell[0])(shift) * _cell + pos_i
        _zero_shift = shift[0] == 0 and shift[1] == 0 and shift[2] == 0
        _j_end = iatom if _zero_shift else j_end

        base_shift = wp.vec3i(
            shift[0] - int_i[0],
            shift[1] - int_i[1],
            shift[2] - int_i[2],
        )

        for chunk_start in range(j_start, _j_end, BLOCK_DIM):
            j_idx = chunk_start + lane
            safe_j = wp.min(j_idx, N - 1)
            pos_j = positions[safe_j]
            diff = pos_i_shifted - pos_j
            dist_sq = wp.dot(diff, diff)

            if j_idx < _j_end and dist_sq < cutoff_sq:
                int_j = per_atom_cell_offsets[j_idx]
                corrected_shift = wp.vec3i(
                    base_shift[0] + int_j[0],
                    base_shift[1] + int_j[1],
                    base_shift[2] + int_j[2],
                )
                write_pos = wp.atomic_add(num_neighbors, iatom, 1)
                if write_pos < max_nb:
                    neighbor_matrix[iatom, write_pos] = j_idx
                    neighbor_matrix_shifts[iatom, write_pos] = corrected_shift
                if not half_fill:
                    rev_pos = wp.atomic_add(num_neighbors, j_idx, 1)
                    if rev_pos < max_nb:
                        neighbor_matrix[j_idx, rev_pos] = iatom
                        neighbor_matrix_shifts[j_idx, rev_pos] = -corrected_shift


# Tiled kernel overload tables
_T = [wp.float32, wp.float64, wp.float16]
_V = [wp.vec3f, wp.vec3d, wp.vec3h]
_M = [wp.mat33f, wp.mat33d, wp.mat33h]

_fill_batch_naive_neighbor_matrix_tiled_overload = {}
_fill_batch_naive_neighbor_matrix_tiled_selective_overload = {}

for _t, _v in zip(_T, _V):
    _fill_batch_naive_neighbor_matrix_tiled_overload[_t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_tiled,
        [
            wp.array(dtype=_v),
            _t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_batch_naive_neighbor_matrix_tiled_selective_overload[_t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_tiled_selective,
        [
            wp.array(dtype=_v),
            _t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.bool),
        ],
    )

_fill_batch_naive_neighbor_matrix_pbc_tiled_overload = {}
_fill_batch_naive_neighbor_matrix_pbc_tiled_selective_overload = {}

for _t, _v, _m in zip(_T, _V, _M):
    _fill_batch_naive_neighbor_matrix_pbc_tiled_overload[_t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_pbc_tiled,
        [
            wp.array(dtype=_v),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=_m),
            _t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.vec3i, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_batch_naive_neighbor_matrix_pbc_tiled_selective_overload[_t] = wp.overload(
        _fill_batch_naive_neighbor_matrix_pbc_tiled_selective,
        [
            wp.array(dtype=_v),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=_m),
            _t,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32, ndim=2),
            wp.array(dtype=wp.vec3i, ndim=2),
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.bool),
        ],
    )
