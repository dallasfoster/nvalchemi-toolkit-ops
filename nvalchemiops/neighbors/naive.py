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

"""Core warp kernels and launchers for naive neighbor list construction.

This module contains warp kernels for O(N²) neighbor list computation.
See `nvalchemiops.torch.neighbors` for PyTorch bindings.
"""

from typing import Any

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    _update_neighbor_matrix,
    _update_neighbor_matrix_pbc,
)


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate neighbor matrix using naive O(N^2) algorithm.

    Computes pairwise distances between all atoms and identifies neighbors
    within the specified cutoff distance. No periodic boundary conditions
    are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
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
    _fill_naive_neighbor_matrix_pbc : Version with periodic boundary conditions
    _fill_batch_naive_neighbor_matrix : Batch version for multiple systems
    """
    tid = wp.tid()
    j_end = positions.shape[0]

    positions_i = positions[tid]
    max_neighbors = neighbor_matrix.shape[1]
    for j in range(tid + 1, j_end):
        diff = positions_i - positions[j]
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            _update_neighbor_matrix(
                tid, j, neighbor_matrix, num_neighbors, max_neighbors, half_fill
            )


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_pbc(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    cell: wp.array(dtype=Any),
    shifts: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
) -> None:
    """Calculate neighbor matrix with periodic boundary conditions using naive O(N^2) algorithm.

    Computes neighbor relationships between atoms across periodic boundaries by
    considering all periodic images within the cutoff distance. Uses a 2D launch
    pattern to parallelize over both atoms and periodic shifts.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Each 3x3 matrix represents one system's periodic cell.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Integer shift vectors for periodic images. Each row represents
        (nx, ny, nz) multiples of the cell vectors.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Entries are filled with atom indices, remaining entries stay as initialized.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
        Each entry corresponds to the shift used for the neighbor in neighbor_matrix.
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
        - neighbor_matrix_shifts : Filled with corresponding shift vectors
        - num_neighbors : Updated with neighbor counts per atom

    See Also
    --------
    _fill_naive_neighbor_matrix : Version without periodic boundary conditions
    _fill_batch_naive_neighbor_matrix_pbc : Batch version for multiple systems
    """
    ishift, iatom = wp.tid()

    jatom_start = 0
    jatom_end = positions.shape[0]

    maxnb = neighbor_matrix.shape[1]
    _positions = positions[iatom]
    _shift = shifts[ishift]
    _cell = cell[0]

    # Wrap atom i to the primary cell to support unwrapped coordinates.
    # Fractional coordinates: frac = pos @ inv_cell (row-vector convention).
    _inv_cell = wp.inverse(_cell)
    _frac_i = _positions * _inv_cell
    _int_i = wp.vec3i(
        wp.int32(wp.floor(_frac_i[0])),
        wp.int32(wp.floor(_frac_i[1])),
        wp.int32(wp.floor(_frac_i[2])),
    )
    _pos_i_wrapped = _positions - type(_positions)(_int_i) * _cell
    positions_shifted = type(_cell[0])(_shift) * _cell + _pos_i_wrapped

    _zero_shift = _shift[0] == 0 and _shift[1] == 0 and _shift[2] == 0

    if _zero_shift:
        jatom_end = iatom

    for jatom in range(jatom_start, jatom_end):
        # Wrap atom j to the primary cell.
        _pos_j = positions[jatom]
        _frac_j = _pos_j * _inv_cell
        _int_j = wp.vec3i(
            wp.int32(wp.floor(_frac_j[0])),
            wp.int32(wp.floor(_frac_j[1])),
            wp.int32(wp.floor(_frac_j[2])),
        )
        _pos_j_wrapped = _pos_j - type(_positions)(_int_j) * _cell
        diff = positions_shifted - _pos_j_wrapped
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            # Correct the stored shift so that dist = pos_i - pos_j - shift*cell
            # holds for the original (potentially unwrapped) positions.
            _corrected_shift = wp.vec3i(
                _shift[0] - _int_i[0] + _int_j[0],
                _shift[1] - _int_i[1] + _int_j[1],
                _shift[2] - _int_i[2] + _int_j[2],
            )
            _update_neighbor_matrix_pbc(
                jatom,
                iatom,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                _corrected_shift,
                maxnb,
                half_fill,
            )


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_selective(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective single-system naive neighbor matrix kernel — skips when not rebuilding.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    half_fill : wp.bool
        If True, only store relationships where i < j.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool
        When False the kernel returns immediately — no recomputation.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - GPU-side conditional: no CPU-GPU synchronization occurs
    """
    tid = wp.tid()
    if not rebuild_flags[0]:
        return
    j_end = positions.shape[0]
    positions_i = positions[tid]
    max_neighbors = neighbor_matrix.shape[1]
    for j in range(tid + 1, j_end):
        diff = positions_i - positions[j]
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            _update_neighbor_matrix(
                tid, j, neighbor_matrix, num_neighbors, max_neighbors, half_fill
            )


@wp.kernel(enable_backward=False)
def _fill_naive_neighbor_matrix_pbc_selective(
    positions: wp.array(dtype=Any),
    cutoff_sq: Any,
    cell: wp.array(dtype=Any),
    shifts: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array2d(dtype=wp.vec3i),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Selective single-system PBC naive neighbor matrix kernel — skips when not rebuilding.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff_sq : float
        Squared cutoff distance for neighbor detection.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Cell matrix defining lattice vectors.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Integer shift vectors for periodic images.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    half_fill : wp.bool
        If True, only store relationships where i < j.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool
        When False the kernel returns immediately — no recomputation.

    Notes
    -----
    - Thread launch: 2D (total_shifts, total_atoms)
    - GPU-side conditional: no CPU-GPU synchronization occurs
    """
    ishift, iatom = wp.tid()
    if not rebuild_flags[0]:
        return

    jatom_start = 0
    jatom_end = positions.shape[0]

    maxnb = neighbor_matrix.shape[1]
    _positions = positions[iatom]
    _shift = shifts[ishift]
    _cell = cell[0]

    _inv_cell = wp.inverse(_cell)
    _frac_i = _positions * _inv_cell
    _int_i = wp.vec3i(
        wp.int32(wp.floor(_frac_i[0])),
        wp.int32(wp.floor(_frac_i[1])),
        wp.int32(wp.floor(_frac_i[2])),
    )
    _pos_i_wrapped = _positions - type(_positions)(_int_i) * _cell
    positions_shifted = type(_cell[0])(_shift) * _cell + _pos_i_wrapped

    _zero_shift = _shift[0] == 0 and _shift[1] == 0 and _shift[2] == 0

    if _zero_shift:
        jatom_end = iatom

    for jatom in range(jatom_start, jatom_end):
        _pos_j = positions[jatom]
        _frac_j = _pos_j * _inv_cell
        _int_j = wp.vec3i(
            wp.int32(wp.floor(_frac_j[0])),
            wp.int32(wp.floor(_frac_j[1])),
            wp.int32(wp.floor(_frac_j[2])),
        )
        _pos_j_wrapped = _pos_j - type(_positions)(_int_j) * _cell
        diff = positions_shifted - _pos_j_wrapped
        dist_sq = wp.length_sq(diff)
        if dist_sq < cutoff_sq:
            _corrected_shift = wp.vec3i(
                _shift[0] - _int_i[0] + _int_j[0],
                _shift[1] - _int_i[1] + _int_j[1],
                _shift[2] - _int_i[2] + _int_j[2],
            )
            _update_neighbor_matrix_pbc(
                jatom,
                iatom,
                neighbor_matrix,
                neighbor_matrix_shifts,
                num_neighbors,
                _corrected_shift,
                maxnb,
                half_fill,
            )


## Generate overloads for all kernels
T = [wp.float32, wp.float64, wp.float16]
V = [wp.vec3f, wp.vec3d, wp.vec3h]
M = [wp.mat33f, wp.mat33d, wp.mat33h]
_fill_naive_neighbor_matrix_overload = {}
_fill_naive_neighbor_matrix_pbc_overload = {}
_fill_naive_neighbor_matrix_selective_overload = {}
_fill_naive_neighbor_matrix_pbc_selective_overload = {}
for t, v, m in zip(T, V, M):
    _fill_naive_neighbor_matrix_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix,
        [
            wp.array(dtype=v),
            t,
            wp.array2d(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_naive_neighbor_matrix_pbc_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix_pbc,
        [
            wp.array(dtype=v),
            t,
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.bool,
        ],
    )
    _fill_naive_neighbor_matrix_selective_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix_selective,
        [
            wp.array(dtype=v),
            t,
            wp.array2d(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.bool),
        ],
    )
    _fill_naive_neighbor_matrix_pbc_selective_overload[t] = wp.overload(
        _fill_naive_neighbor_matrix_pbc_selective,
        [
            wp.array(dtype=v),
            t,
            wp.array(dtype=m),
            wp.array(dtype=wp.vec3i),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.bool,
            wp.array(dtype=wp.bool),
        ],
    )

###########################################################################################
########################### Warp Launchers ###############################################
###########################################################################################
#
# Selective variants: GPU-side rebuild_flags[0] check — no CPU-GPU sync.


def naive_neighbor_matrix_selective(
    positions: wp.array,
    cutoff: float,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
) -> None:
    """Selective warp launcher for naive neighbor matrix (no PBC).

    Equivalent to ``naive_neighbor_matrix`` but the kernel checks
    ``rebuild_flags[0]`` on the GPU and exits immediately when False.
    No CPU-GPU synchronisation occurs.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix (pre-allocated; untouched when not rebuilding).
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts (pre-zeroed by caller when rebuilding).
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool
        GPU-resident flag; False → kernel returns immediately.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str
        Warp device string.
    half_fill : bool, default=False
        If True, only store relationships where i < j.

    See Also
    --------
    naive_neighbor_matrix : Non-selective variant
    _fill_naive_neighbor_matrix_selective : Underlying kernel
    """
    total_atoms = positions.shape[0]
    wp.launch(
        kernel=_fill_naive_neighbor_matrix_selective_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            wp_dtype(cutoff * cutoff),
            neighbor_matrix,
            num_neighbors,
            half_fill,
            rebuild_flags,
        ],
        device=device,
    )


def naive_neighbor_matrix_pbc_selective(
    positions: wp.array,
    cutoff: float,
    cell: wp.array,
    shifts: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
) -> None:
    """Selective warp launcher for naive neighbor matrix with PBC.

    Equivalent to ``naive_neighbor_matrix_pbc`` but the kernel checks
    ``rebuild_flags[0]`` on the GPU and exits immediately when False.
    No CPU-GPU synchronisation occurs.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Cell matrix defining lattice vectors.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Pre-computed integer shift vectors.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix (pre-allocated; untouched when not rebuilding).
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: Shift vectors (pre-allocated; untouched when not rebuilding).
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Neighbor counts (pre-zeroed by caller when rebuilding).
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool
        GPU-resident flag; False → kernel returns immediately.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str
        Warp device string.
    half_fill : bool, default=False
        If True, only store relationships where i < j.

    See Also
    --------
    naive_neighbor_matrix_pbc : Non-selective variant
    _fill_naive_neighbor_matrix_pbc_selective : Underlying kernel
    """
    total_atoms = positions.shape[0]
    total_shifts = shifts.shape[0]
    wp.launch(
        kernel=_fill_naive_neighbor_matrix_pbc_selective_overload[wp_dtype],
        dim=(total_shifts, total_atoms),
        inputs=[
            positions,
            wp_dtype(cutoff * cutoff),
            cell,
            shifts,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            half_fill,
            rebuild_flags,
        ],
        device=device,
    )


def naive_neighbor_matrix(
    positions: wp.array,
    cutoff: float,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
) -> None:
    """Core warp launcher for naive neighbor matrix construction (no PBC).

    Computes pairwise distances and fills the neighbor matrix with atom indices
    within the cutoff distance using pure warp operations. No periodic boundary
    conditions are applied.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.

    See Also
    --------
    naive_neighbor_matrix_pbc : Version with periodic boundary conditions
    _fill_naive_neighbor_matrix : Kernel that performs the computation
    """
    total_atoms = positions.shape[0]

    wp.launch(
        kernel=_fill_naive_neighbor_matrix_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            wp_dtype(cutoff * cutoff),
            neighbor_matrix,
            num_neighbors,
            half_fill,
        ],
        device=device,
    )


def naive_neighbor_matrix_pbc(
    positions: wp.array,
    cutoff: float,
    cell: wp.array,
    shifts: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
) -> None:
    """Core warp launcher for naive neighbor matrix construction with PBC.

    Computes neighbor relationships between atoms across periodic boundaries using
    pure warp operations. Assumes shift vectors have been pre-computed.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Cell matrix defining lattice vectors in Cartesian coordinates.
    shifts : wp.array, shape (total_shifts, 3), dtype=wp.vec3i
        Integer shift vectors for periodic images. Each row represents
        (nx, ny, nz) multiples of the cell vectors.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
        Must be pre-allocated.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.
    - Shift vectors must be pre-computed using compute_naive_num_shifts and _expand_naive_shifts.

    See Also
    --------
    naive_neighbor_matrix : Version without periodic boundary conditions
    _fill_naive_neighbor_matrix_pbc : Kernel that performs the computation
    compute_naive_num_shifts : Computes shift ranges
    _expand_naive_shifts : Expands shift ranges into explicit vectors
    """
    total_atoms = positions.shape[0]
    total_shifts = shifts.shape[0]

    wp.launch(
        kernel=_fill_naive_neighbor_matrix_pbc_overload[wp_dtype],
        dim=(total_shifts, total_atoms),
        inputs=[
            positions,
            wp_dtype(cutoff * cutoff),
            cell,
            shifts,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            half_fill,
        ],
        device=device,
    )
