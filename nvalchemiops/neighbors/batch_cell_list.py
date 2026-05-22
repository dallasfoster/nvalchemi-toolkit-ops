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

"""Core warp kernels and launchers for batched cell list neighbor construction.

Batched O(N) cell-based neighbor list across heterogeneous systems with
per-system cell geometry and PBC.  Both atom-centric and pair-centric
queries are available; algorithm selection lives at the torch wrapper
layer.  See :mod:`nvalchemiops.torch.neighbors` for PyTorch bindings.

Output contract: ``neighbor_matrix_shifts[i, n]`` is written
unconditionally at every active slot; callers may allocate via
``torch.empty(...)`` and skip pre-zeroing.
"""

import os
from typing import Any

import warp as wp

from nvalchemiops.math import wpdivmod
from nvalchemiops.neighbors.neighbor_utils import (
    _decode_full_shift_index,
    _decode_shift_index,
    selective_zero_num_neighbors,
)

__all__ = [
    "batch_build_cell_list",
    "batch_query_cell_list",
    "batch_query_cell_list_pair_centric",
    "compute_batch_pair_centric_n_outer",
]


# Empirically calibrated dispatch thresholds for the batch path.
# Three-clause rule (see ``_should_dispatch_batch_pair_centric``):
#   1. cutoff >= 8 — pair-centric dominates almost everywhere here.
#   2. cutoff >= 6 AND total_atoms <= 65_536 AND avg_aps >= 4096 —
#      small-/medium-N MLIP regime with reasonably-sized systems.  The
#      avg_aps floor avoids picking pair-centric for many-tiny-systems
#      shapes where its setup overhead dominates the kernel speedup.
#   3. cutoff >= 6 AND num_systems <= 8 — few-large-systems regime,
#      where cell-level parallelism dominates atomic contention.
_BATCH_PAIR_DISPATCH_DEFAULTS = {
    "NVALCHEMI_NEIGHLIST_BATCH_PAIR_CUTOFF_FLOOR": 8.0,
    "NVALCHEMI_NEIGHLIST_BATCH_PAIR_TOTAL_CAP": 65_536,
    "NVALCHEMI_NEIGHLIST_BATCH_PAIR_AVG_APS_FLOOR": 4096,
    "NVALCHEMI_NEIGHLIST_BATCH_PAIR_NSYS_CAP": 8,
}


def _should_dispatch_batch_pair_centric(
    total_atoms: int,
    num_systems: int,
    cutoff: float,
) -> bool:
    """Sync-free dispatch decision for the batch path.

    Selects pair-centric when *any* of the three clauses holds:

    1. ``cutoff >= NVALCHEMI_NEIGHLIST_BATCH_PAIR_CUTOFF_FLOOR`` (default 8.0).
    2. ``cutoff >= 6`` AND
       ``total_atoms <= NVALCHEMI_NEIGHLIST_BATCH_PAIR_TOTAL_CAP`` (default 65_536)
       AND ``avg_atoms_per_system >= NVALCHEMI_NEIGHLIST_BATCH_PAIR_AVG_APS_FLOOR``
       (default 4096).
    3. ``cutoff >= 6`` AND
       ``num_systems <= NVALCHEMI_NEIGHLIST_BATCH_PAIR_NSYS_CAP`` (default 8)
       AND ``total_atoms > TOTAL_CAP`` (few-large-systems regime).

    Calibrated on Blackwell GB10.  The thresholds are env-tunable; the
    cost of picking wrong is bounded (≤ ~20 % mean wallclock penalty on
    cross-GPU measurements).  Recalibrate by sweeping
    ``benchmark_neighborlist.py --methods batch_cell_list_atom_centric
    batch_cell_list_pair_centric`` and resetting the env vars above.
    """

    def _i(name: str) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return int(_BATCH_PAIR_DISPATCH_DEFAULTS[name])
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return int(_BATCH_PAIR_DISPATCH_DEFAULTS[name])

    def _f(name: str) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return float(_BATCH_PAIR_DISPATCH_DEFAULTS[name])
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(_BATCH_PAIR_DISPATCH_DEFAULTS[name])

    cutoff_floor = _f("NVALCHEMI_NEIGHLIST_BATCH_PAIR_CUTOFF_FLOOR")
    total_cap = _i("NVALCHEMI_NEIGHLIST_BATCH_PAIR_TOTAL_CAP")
    avg_aps_floor = _i("NVALCHEMI_NEIGHLIST_BATCH_PAIR_AVG_APS_FLOOR")
    nsys_cap = _i("NVALCHEMI_NEIGHLIST_BATCH_PAIR_NSYS_CAP")

    if cutoff >= cutoff_floor:
        return True
    avg_aps = total_atoms // max(num_systems, 1)
    if cutoff >= 6.0 and total_atoms <= total_cap and avg_aps >= avg_aps_floor:
        return True
    # Few-large-systems clause: require total_atoms above the same cap so
    # the per-call setup overhead (gather + cell_to_system map + R_max
    # .item()) doesn't dominate for tiny batches.
    if cutoff >= 6.0 and num_systems <= nsys_cap and total_atoms > total_cap:
        return True
    return False


###########################################################################################
########################### Batch Cell List Construction ##################################
###########################################################################################


@wp.kernel(enable_backward=False)
def _batch_estimate_cell_list_sizes(
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    cell_size: Any,
    max_nbins: Any,
    number_of_cells: wp.array(dtype=Any),
    neighbor_search_radius: wp.array(dtype=Any),
) -> None:
    """
    Estimate the number of cells and neighbor search radius for a batch of systems.

    Parameters
    ----------
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    cell_size : Any
        Size of the cells, usually the neighbor cutoff distance in the simulation box.
    max_nbins : Any
        Maximum number of cells to allocate.
    number_of_cells : wp.array, shape (num_systems,), dtype=wp.int32
        OUTPUT: Number of cells in each system.
    neighbor_search_radius : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Radius of neighboring cells to search in each dimension for each system.

    Notes
    -----
    - Thread launch: One thread per system (dim=num_systems)
    - Modifies: number_of_cells, neighbor_search_radius
    - Each thread processes one complete system independently
    - For non-periodic directions with only 1 cell, search radius is set to 0
    """
    system_idx = wp.tid()
    system_cell_matrix = cell[system_idx]
    inverse_cell_transpose = wp.transpose(wp.inverse(system_cell_matrix))

    cells_per_dimension = wp.vec3i(0, 0, 0)
    # Calculate optimal number of cells in each dimension.  The search
    # radius is recomputed AFTER any adaptive promotion below so that R
    # tracks the actual cell width.
    for i in range(3):
        # Distance between parallel faces in reciprocal space
        face_distance = type(cell_size)(1.0) / wp.length(inverse_cell_transpose[i])
        cells_per_dimension[i] = max(wp.int32(face_distance / cell_size), 1)

    # Adaptive promotion: when the natural cell grid is small, the
    # atom-centric query kernel suffers from low cell-level parallelism
    # and high atoms-per-cell — at ``nx = 1`` it degenerates to O(N²)
    # in a single cell.  Promote each axis up to ``ADAPTIVE_MIN_CELLS``
    # so each cell holds fewer atoms; the search radius below grows
    # to compensate.  Skipped on non-PBC axes that genuinely have a
    # single cell.
    ADAPTIVE_MIN_CELLS = wp.int32(4)
    for i in range(3):
        if pbc[system_idx, i] or cells_per_dimension[i] > 1:
            while cells_per_dimension[i] < ADAPTIVE_MIN_CELLS:
                cells_per_dimension[i] = cells_per_dimension[i] * 2

    # Now that cells_per_dimension is final, compute the search radius
    # consistent with the (possibly promoted) cell width.
    for i in range(3):
        face_distance = type(cell_size)(1.0) / wp.length(inverse_cell_transpose[i])
        if cells_per_dimension[i] == 1 and not pbc[system_idx, i]:
            neighbor_search_radius[system_idx][i] = 0
        else:
            neighbor_search_radius[system_idx][i] = wp.int32(
                wp.ceil(
                    cell_size * type(cell_size)(cells_per_dimension[i]) / face_distance
                )
            )

    total_cells_this_system = int(
        cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
    )

    while total_cells_this_system > max_nbins:
        for dim in range(3):
            cells_per_dimension[dim] = max(cells_per_dimension[dim] // 2, 1)
        total_cells_this_system = int(
            cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
        )
    number_of_cells[system_idx] = total_cells_this_system


@wp.kernel(enable_backward=False)
def _batch_cell_list_construct_bin_size(
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    target_cell_size: Any,
    max_total_cells: Any,
) -> None:
    """Determine optimal spatial decomposition parameters for batch cell list construction.

    This kernel processes multiple systems simultaneously, calculating
    the optimal number of cells and neighbor search radii for each system based
    on their individual cell geometries and target cell sizes.

    The algorithm for each system:
    1. Computes optimal cell count per dimension based on cell geometry
    2. Reduces cell count if total exceeds maximum allowed per system
    3. Calculates neighbor search radius to ensure neighbor completeness

    Parameters
    ----------
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrix defining the simulation box.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Number of cells to create in x, y, z directions for each system.
    target_cell_size : float
        Desired cell size for each system, typically the neighbor cutoff distance.
    max_total_cells : int
        Maximum total cells allowed (nx * ny * nz ≤ max_total_cells // num_systems).

    Notes
    -----
    - Thread launch: One thread per system (dim=num_systems)
    - Modifies: cells_per_dimension, batch_neighbor_search_radius
    - Each thread processes one complete system independently
    - For non-periodic directions with only 1 cell, search radius is set to 0
    """
    # Thread ID corresponds to system index in the batch
    system_idx = wp.tid()

    # Get cell matrix and target size for this system
    num_systems = cell.shape[0]
    s_cell_matrix = cell[system_idx]
    inverse_cell_transpose = wp.transpose(wp.inverse(s_cell_matrix))

    # Compute optimal number of cells in each dimension for this system
    for dim in range(3):
        # Distance between parallel faces in reciprocal space
        face_distance = type(target_cell_size)(1.0) / wp.length(
            inverse_cell_transpose[dim]
        )
        cells_per_dimension[system_idx][dim] = max(
            wp.int32(face_distance / target_cell_size), 1
        )

    # Adaptive promotion (mirror of the logic in
    # ``_batch_estimate_cell_list_sizes``): ensure ≥ ADAPTIVE_MIN_CELLS
    # cells per PBC axis so the atom-centric query doesn't degenerate to
    # O(N²) at small box / large cutoff.
    ADAPTIVE_MIN_CELLS = wp.int32(4)
    for dim in range(3):
        if pbc[system_idx, dim] or cells_per_dimension[system_idx][dim] > 1:
            while cells_per_dimension[system_idx][dim] < ADAPTIVE_MIN_CELLS:
                cells_per_dimension[system_idx][dim] = (
                    cells_per_dimension[system_idx][dim] * 2
                )

    # Check if total cell count exceeds maximum allowed for this system
    total_cells_this_system = int(
        cells_per_dimension[system_idx][0]
        * cells_per_dimension[system_idx][1]
        * cells_per_dimension[system_idx][2]
    )

    # Reduce cell count if necessary by halving dimensions iteratively
    while (total_cells_this_system * num_systems) > max_total_cells:
        for dim in range(3):
            cells_per_dimension[system_idx][dim] = max(
                cells_per_dimension[system_idx][dim] // 2, 1
            )
        total_cells_this_system = int(
            cells_per_dimension[system_idx][0]
            * cells_per_dimension[system_idx][1]
            * cells_per_dimension[system_idx][2]
        )


@wp.kernel(enable_backward=False)
def _batch_cell_list_count_atoms_per_bin(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    batch_idx: wp.array(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    atoms_per_cell_count: wp.array(dtype=Any),
    atom_periodic_shifts: wp.array(dtype=Any),
) -> None:
    """Count atoms in each spatial cell across batch systems and compute periodic shifts.

    This is the first pass of the two-pass batch cell list construction algorithm.
    Each thread processes one atom, determines which system and cell it belongs to,
    handles periodic boundary conditions, and atomically increments the atom count
    for that cell. Supports heterogeneous batches with different system sizes.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting index in global cell arrays for each system (exclusive scan of cell counts).
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: Number of atoms assigned to each cell across all systems (modified atomically).
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: Periodic boundary crossings for each atom.

    Notes
    -----
    - Thread launch: One thread per atom across all systems (dim=total_atoms)
    - Modifies: batch_atoms_per_cell_count, batch_atom_periodic_shifts
    - Uses atomic operations for thread-safe counting across batch
    - Each thread first determines which system it belongs to, then processes normally
    """
    atom_idx = wp.tid()

    # Find which system this atom belongs to using binary-like search
    system_idx = batch_idx[atom_idx]

    # Get system-specific parameters
    s_cell_matrix = cell[system_idx]
    s_cells_per_dimension = cells_per_dimension[system_idx]
    s_cell_offset = cell_offsets[system_idx]

    # Transform to fractional coordinates for this system
    inverse_cell = wp.inverse(s_cell_matrix)
    fractional_position = positions[atom_idx] * inverse_cell

    # Determine which cell this atom belongs to within its system
    cell_coords = wp.vec3i(0, 0, 0)
    for dim in range(3):
        cell_coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim]
                * type(fractional_position[dim])(s_cells_per_dimension[dim])
            )
        )

        # Handle periodic boundary conditions for this system
        if pbc[system_idx, dim]:
            cell_before_wrap = cell_coords[dim]
            num_cells_this_dim = s_cells_per_dimension[dim]
            quotient, remainder = wpdivmod(cell_before_wrap, num_cells_this_dim)
            atom_periodic_shifts[atom_idx][dim] = quotient
            cell_coords[dim] = remainder
        else:
            # Clamp to valid cell range for non-periodic dimensions
            atom_periodic_shifts[atom_idx][dim] = 0
            cell_coords[dim] = wp.clamp(
                cell_coords[dim], 0, s_cells_per_dimension[dim] - 1
            )

    # Compute linear cell index with system offset for global cell indexing
    global_linear_cell_index = (
        s_cell_offset
        + cell_coords[0]
        + s_cells_per_dimension[0]
        * (cell_coords[1] + s_cells_per_dimension[1] * cell_coords[2])
    )

    # Atomically increment the count for this cell across the entire batch
    wp.atomic_add(atoms_per_cell_count, global_linear_cell_index, 1)


@wp.kernel(enable_backward=False)
def _batch_cell_list_bin_atoms(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    cells_per_dimension: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    atom_to_cell_mapping: wp.array(dtype=Any),
    atoms_per_cell_count: wp.array(dtype=Any),
    cell_atom_start_indices: wp.array(dtype=Any),
    cell_atom_list: wp.array(dtype=Any),
) -> None:
    """Assign atoms to cells and build cell-to-atom mapping for batch systems.

    This is the second pass of the two-pass batch cell list construction algorithm.
    Each thread processes one atom, determines which system and cell it belongs to,
    and adds it to that cell's atom list using atomic operations for thread safety.
    Supports heterogeneous batches with different system sizes.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems,3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array2d, shape (num_systems, 3), dtype=bool
        Periodic boundary condition flags for each system and dimension.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        Index of the system for each atom.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting index in global cell arrays for each system (exclusive scan of cell counts).
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        MODIFIED: Running count of atoms added to each cell (reset before use).
    cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell's atoms.
    cell_atom_list : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: Flattened list of atom indices organized by cell across all systems.

    Notes
    -----
    - Thread launch: One thread per atom across all systems (dim=total_atoms)
    - Modifies: atom_to_cell_mapping, atoms_per_cell_count, cell_atom_list
    - atoms_per_cell_count must be zeroed before calling this kernel
    - Uses atomic operations for thread-safe list building across batch
    """
    atom_idx = wp.tid()

    # Find which system this atom belongs to
    system_idx = batch_idx[atom_idx]

    # Get system-specific parameters
    s_cell_matrix = cell[system_idx]
    s_cells_per_dimension = cells_per_dimension[system_idx]
    s_cell_offset = cell_offsets[system_idx]

    # Transform to fractional coordinates
    inverse_cell = wp.inverse(s_cell_matrix)
    fractional_position = positions[atom_idx] * inverse_cell

    # Determine which cell this atom belongs to within its system
    cell_coords = wp.vec3i(0, 0, 0)
    for dim in range(3):
        cell_coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim]
                * type(fractional_position[dim])(s_cells_per_dimension[dim])
            )
        )

        # Handle periodic boundary conditions
        if pbc[system_idx, dim]:
            cell_before_wrap = cell_coords[dim]
            num_cells_this_dim = s_cells_per_dimension[dim]
            _, remainder = wpdivmod(cell_before_wrap, num_cells_this_dim)
            cell_coords[dim] = remainder
        else:
            # Clamp to valid cell range for non-periodic dimensions
            cell_coords[dim] = wp.clamp(
                cell_coords[dim], 0, s_cells_per_dimension[dim] - 1
            )

    # Store the cell assignment for this atom
    atom_to_cell_mapping[atom_idx] = cell_coords

    # Compute global linear cell index with system offset
    global_linear_cell_index = (
        s_cell_offset
        + cell_coords[0]
        + s_cells_per_dimension[0]
        * (cell_coords[1] + s_cells_per_dimension[1] * cell_coords[2])
    )

    # Atomically get position in this cell's atom list
    position_in_cell = wp.atomic_add(atoms_per_cell_count, global_linear_cell_index, 1)
    final_list_index = (
        cell_atom_start_indices[global_linear_cell_index] + position_in_cell
    )

    # Store this atom's index in the cell's atom list
    cell_atom_list[final_list_index] = atom_idx


@wp.kernel(enable_backward=False)
def _compute_cells_per_system(
    cells_per_dimension: wp.array(dtype=wp.vec3i),
    cells_per_system: wp.array(dtype=wp.int32),
) -> None:
    """Compute total cells per system from cell dimension vectors.

    For each system, computes the product of cells in x, y, z dimensions.
    Used as input to exclusive scan for computing cell offsets.

    Parameters
    ----------
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    cells_per_system : wp.array, shape (num_systems,), dtype=wp.int32
        OUTPUT: Total number of cells for each system.

    Notes
    -----
    - Thread launch: One thread per system (dim=num_systems)
    - Modifies: cells_per_system
    """
    system_idx = wp.tid()
    dims = cells_per_dimension[system_idx]
    cells_per_system[system_idx] = dims[0] * dims[1] * dims[2]


@wp.kernel(enable_backward=False)
def _batch_cell_list_build_neighbor_matrix_local_count_sorted(
    sorted_positions: wp.array(dtype=Any),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=wp.bool),
    batch_idx: wp.array(dtype=wp.int32),
    cutoff: Any,
    cells_per_dimension: wp.array(dtype=wp.vec3i),
    neighbor_search_radius: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Sorted-reads atom-centric batched cell-list neighbor matrix kernel.

    Batched analogue of
    :func:`nvalchemiops.neighbors.cell_list._cell_list_build_neighbor_matrix_local_count_sorted`.
    Each thread iterates over the *sorted* atom slot ``s`` (so consecutive
    threads in a warp work on consecutive atoms within a cell, giving
    coalesced inner-loop reads of ``sorted_positions[cell_start + j]`` and
    ``sorted_atom_periodic_shifts[cell_start + j]``).  The home cell lookup
    (``atom_to_cell_mapping[atom_idx]``) and per-system metadata
    (``cell[system_idx]``, etc.) are single per-thread reads.

    Output writes go to ``neighbor_matrix[atom_idx, n]`` where
    ``atom_idx = cell_atom_list[s]`` is the original (pre-sort) atom
    index, preserving the public output ordering.

    Selective rebuild: ``rebuild_flags`` is a per-system ``wp.bool`` array
    (shape ``(num_systems,)``); ``False`` for the home system makes the
    thread return immediately.  Non-selective callers pass an always-True
    array.

    Parameters
    ----------
    sorted_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Per-cell-contiguous positions (output of ``_gather_positions_by_cell``).
    sorted_atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Per-cell-contiguous PBC shift vectors.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Per-system unit cell matrices.
    pbc : wp.array2d, shape (num_systems, 3), dtype=wp.bool
        Per-system periodicity flags.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cutoff : scalar (float32 or float64)
        Neighbor search cutoff distance.
    cells_per_dimension : wp.array, shape (num_systems,), dtype=wp.vec3i
        Cells per axis for each system.
    neighbor_search_radius : wp.array, shape (num_systems,), dtype=wp.vec3i
        Per-system search radius.
    atom_to_cell_mapping : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Local cell coordinates for each atom.
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        Atoms per global cell.
    cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
        Per-cell start offsets into ``cell_atom_list`` / ``sorted_positions``.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Cell-contiguous original atom indices (the permutation).
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Per-system starting global cell index.
    neighbor_matrix : wp.array2d, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: neighbor indices indexed by original ``atom_idx``.
    neighbor_matrix_shifts : wp.array2d, shape (total_atoms, max_neighbors), dtype=wp.vec3i
        OUTPUT: per-pair PBC shift vectors.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: per-atom neighbor count.
    half_fill : wp.bool
        ``True`` → half-shell + ``j > i`` self-cell filter;
        ``False`` → full shell + ``j != i`` self-cell filter.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system rebuild flags.

    Returns
    -------
    None
        Writes to ``neighbor_matrix``, ``neighbor_matrix_shifts``, and
        ``num_neighbors``.

    Notes
    -----
    - Thread launch: one thread per cell-list slot (dim=total_atoms);
      slots whose ``cell_atom_list[s]`` is out-of-range are skipped (for
      padded layouts).
    - Each thread is the unique writer to its own ``atom_idx`` row, so
      the per-row count ``n`` lives in a register and is committed to
      ``num_neighbors[atom_idx]`` at the end — no atomics on the count.
    - Always-write shifts contract: ``neighbor_matrix_shifts`` is filled
      in-place at the active slots; callers may skip the zero-prefill.
    """
    s = wp.tid()
    if s >= sorted_positions.shape[0]:
        return

    atom_idx = cell_atom_list[s]
    if atom_idx >= sorted_positions.shape[0]:
        return

    system_idx = batch_idx[atom_idx]
    if not rebuild_flags[system_idx]:
        return

    cutoff_distance_sq = cutoff * cutoff
    central_atom_position = sorted_positions[s]
    central_atom_shift = sorted_atom_periodic_shifts[s]
    central_atom_cell_coords = atom_to_cell_mapping[atom_idx]
    max_neighbors = neighbor_matrix.shape[1]

    s_cell = cell[system_idx]
    s_cell_transpose = wp.transpose(s_cell)
    s_cells_per_dimension = cells_per_dimension[system_idx]
    s_cell_offset = cell_offsets[system_idx]
    s_neighbor_search_radius = neighbor_search_radius[system_idx]
    s_pbc = pbc[system_idx]

    cpd_x = s_cells_per_dimension[0]
    cpd_y = s_cells_per_dimension[1]
    cpd_z = s_cells_per_dimension[2]

    pbc_x = s_pbc[0]
    pbc_y = s_pbc[1]
    pbc_z = s_pbc[2]

    dx_lo = wp.int32(0)
    if not half_fill:
        dx_lo = -s_neighbor_search_radius[0]

    n = wp.int32(0)

    for dx in range(dx_lo, s_neighbor_search_radius[0] + 1):
        for dy in range(-s_neighbor_search_radius[1], s_neighbor_search_radius[1] + 1):
            for dz in range(
                -s_neighbor_search_radius[2], s_neighbor_search_radius[2] + 1
            ):
                if half_fill:
                    if not (
                        dx > 0
                        or (dx == 0 and dy > 0)
                        or (dx == 0 and dy == 0 and dz >= 0)
                    ):
                        continue
                target_x = central_atom_cell_coords[0] + dx
                target_y = central_atom_cell_coords[1] + dy
                target_z = central_atom_cell_coords[2] + dz

                if not pbc_x and (target_x < 0 or target_x >= cpd_x):
                    continue
                if not pbc_y and (target_y < 0 or target_y >= cpd_y):
                    continue
                if not pbc_z and (target_z < 0 or target_z >= cpd_z):
                    continue

                cs_x, wc_x = wpdivmod(target_x, cpd_x)
                cs_y, wc_y = wpdivmod(target_y, cpd_y)
                cs_z, wc_z = wpdivmod(target_z, cpd_z)

                global_linear_cell_index = (
                    s_cell_offset + wc_x + cpd_x * (wc_y + cpd_y * wc_z)
                )

                cell_start_index = cell_atom_start_indices[global_linear_cell_index]
                num_atoms_in_cell = atoms_per_cell_count[global_linear_cell_index]

                for cell_atom_idx in range(num_atoms_in_cell):
                    j_slot = cell_start_index + cell_atom_idx
                    neighbor_atom_idx = cell_atom_list[j_slot]
                    neighbor_atom_shift = sorted_atom_periodic_shifts[j_slot]
                    neighbor_pos = sorted_positions[j_slot]

                    shift_x = cs_x
                    shift_y = cs_y
                    shift_z = cs_z

                    if pbc_x:
                        shift_x += central_atom_shift[0] - neighbor_atom_shift[0]
                    else:
                        shift_x = 0
                    if pbc_y:
                        shift_y += central_atom_shift[1] - neighbor_atom_shift[1]
                    else:
                        shift_y = 0
                    if pbc_z:
                        shift_z += central_atom_shift[2] - neighbor_atom_shift[2]
                    else:
                        shift_z = 0

                    if dx == 0 and dy == 0 and dz == 0:
                        if half_fill:
                            if neighbor_atom_idx <= atom_idx:
                                continue
                        else:
                            if neighbor_atom_idx == atom_idx:
                                continue

                    if shift_x == 0 and shift_y == 0 and shift_z == 0:
                        dr = neighbor_pos - central_atom_position
                    else:
                        fractional_shift = type(central_atom_position)(
                            type(cutoff)(shift_x),
                            type(cutoff)(shift_y),
                            type(cutoff)(shift_z),
                        )
                        cartesian_shift = s_cell_transpose * fractional_shift
                        dr = neighbor_pos - central_atom_position + cartesian_shift

                    distance_sq = wp.dot(dr, dr)

                    if distance_sq < cutoff_distance_sq:
                        if n < max_neighbors:
                            neighbor_matrix[atom_idx, n] = neighbor_atom_idx
                            neighbor_matrix_shifts[atom_idx, n] = wp.vec3i(
                                shift_x, shift_y, shift_z
                            )
                        n += 1

    num_neighbors[atom_idx] = n


T = [wp.float32, wp.float64]
V = [wp.vec3f, wp.vec3d]
M = [wp.mat33f, wp.mat33d]
_batch_estimate_cell_list_sizes_overload = {}
_batch_cell_list_construct_bin_size_overload = {}
_batch_cell_list_count_atoms_per_bin_overload = {}
_batch_cell_list_bin_atoms_overload = {}
_batch_cell_list_build_neighbor_matrix_local_count_sorted_overload = {}
for t, v, m in zip(T, V, M):
    _batch_estimate_cell_list_sizes_overload[t] = wp.overload(
        _batch_estimate_cell_list_sizes,
        [
            wp.array(dtype=m),
            wp.array2d(dtype=wp.bool),
            t,
            wp.int32,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
        ],
    )
    _batch_cell_list_construct_bin_size_overload[t] = wp.overload(
        _batch_cell_list_construct_bin_size,
        [
            wp.array(dtype=m),
            wp.array2d(dtype=wp.bool),
            wp.array(dtype=wp.vec3i),
            t,
            wp.int32,
        ],
    )
    _batch_cell_list_count_atoms_per_bin_overload[t] = wp.overload(
        _batch_cell_list_count_atoms_per_bin,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array2d(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
        ],
    )
    _batch_cell_list_bin_atoms_overload[t] = wp.overload(
        _batch_cell_list_bin_atoms,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array2d(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
        ],
    )
    _batch_cell_list_build_neighbor_matrix_local_count_sorted_overload[t] = wp.overload(
        _batch_cell_list_build_neighbor_matrix_local_count_sorted,
        [
            wp.array(dtype=v),  # sorted_positions
            wp.array(dtype=wp.vec3i),  # sorted_atom_periodic_shifts
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.bool),  # pbc
            wp.array(dtype=wp.int32),  # batch_idx
            t,  # cutoff
            wp.array(dtype=wp.vec3i),  # cells_per_dimension
            wp.array(dtype=wp.vec3i),  # neighbor_search_radius
            wp.array(dtype=wp.vec3i),  # atom_to_cell_mapping
            wp.array(dtype=wp.int32),  # atoms_per_cell_count
            wp.array(dtype=wp.int32),  # cell_atom_start_indices
            wp.array(dtype=wp.int32),  # cell_atom_list
            wp.array(dtype=wp.int32),  # cell_offsets
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.bool,  # half_fill
            wp.array(dtype=wp.bool),  # rebuild_flags
        ],
    )

###########################################################################################
########################### Warp Launchers ###############################################
###########################################################################################


def batch_build_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    batch_idx: wp.array,
    cells_per_dimension: wp.array,
    cell_offsets: wp.array,
    cells_per_system: wp.array,
    atom_periodic_shifts: wp.array,
    atom_to_cell_mapping: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for building batch spatial cell lists.

    Constructs spatial decomposition data structures for multiple systems using
    pure warp operations. This function launches warp kernels to organize atoms
    into spatial cells across all systems in the batch.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags for each system and dimension.
    cutoff : float
        Neighbor search cutoff distance.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Number of cells in x, y, z directions for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        OUTPUT: Starting index in global cell arrays for each system.
        Computed internally via exclusive scan of cells_per_dimension products.
    cells_per_system : wp.array, shape (num_systems,), dtype=wp.int32
        SCRATCH: Temporary buffer for total cells per system.
        Used as input to exclusive scan for computing cell_offsets.
        Must be pre-allocated by caller.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: Periodic boundary crossings for each atom.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Number of atoms in each cell. Must be zeroed by caller before first use.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Starting index in cell_atom_list for each cell's atoms.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Flattened list of atom indices organized by cell.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface. Caller must ensure atoms_per_cell_count is zeroed.
    - atoms_per_cell_count must be zeroed before calling this function.
    - cell_offsets is computed internally after cells_per_dimension is determined.
    - This function handles the internal cumsum for cell_atom_start_indices using wp.utils.array_scan.
    - For framework bindings, use the torch/jax wrappers instead.

    See Also
    --------
    batch_query_cell_list : Query cell list to build neighbor matrix (call after this)
    _batch_cell_list_construct_bin_size : Kernel for computing cell dimensions
    _batch_cell_list_count_atoms_per_bin : Kernel for counting atoms per cell
    _batch_cell_list_bin_atoms : Kernel for binning atoms into cells
    """
    total_atoms = positions.shape[0]
    num_systems = cell.shape[0]
    max_total_cells = atoms_per_cell_count.shape[0]
    wp_cutoff = wp_dtype(cutoff)

    # Construct cell dimensions
    wp.launch(
        _batch_cell_list_construct_bin_size_overload[wp_dtype],
        dim=num_systems,
        device=device,
        inputs=(
            cell,
            pbc,
            cells_per_dimension,
            wp_cutoff,
            max_total_cells,
        ),
    )

    # Compute cell_offsets from cells_per_dimension (exclusive scan of products)
    # This must happen after construct_bin_size fills cells_per_dimension
    wp.launch(
        _compute_cells_per_system,
        dim=num_systems,
        device=device,
        inputs=(cells_per_dimension, cells_per_system),
    )
    wp.utils.array_scan(cells_per_system, cell_offsets, inclusive=False)

    # Count atoms per bin (expects atoms_per_cell_count to be zeroed by caller)
    wp.launch(
        _batch_cell_list_count_atoms_per_bin_overload[wp_dtype],
        dim=total_atoms,
        inputs=(
            positions,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            cell_offsets,
            atoms_per_cell_count,
            atom_periodic_shifts,
        ),
        device=device,
    )

    # Compute exclusive scan to get starting indices for each cell
    # This converts atom counts [3, 5, 2, ...] -> starting indices [0, 3, 8, ...]
    wp.utils.array_scan(atoms_per_cell_count, cell_atom_start_indices, inclusive=False)

    # Zero counts before binning atoms (second pass needs fresh counts)
    atoms_per_cell_count.zero_()

    # Bin atoms (expects atoms_per_cell_count to be zeroed)
    wp.launch(
        _batch_cell_list_bin_atoms_overload[wp_dtype],
        dim=total_atoms,
        inputs=(
            positions,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            cell_offsets,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ),
        device=device,
    )


def batch_query_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    batch_idx: wp.array,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    cell_offsets: wp.array,
    atom_periodic_shifts: wp.array,
    atom_to_cell_mapping: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    sorted_positions: wp.array,
    sorted_atom_periodic_shifts: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = False,
    algorithm: str = "atom_centric",
    cells_per_system: wp.array | None = None,
    cell_to_system: wp.array | None = None,
    n_outer: int | None = None,
    R_max: tuple[int, int, int] | None = None,
    total_cells: int | None = None,
) -> None:
    """Core warp launcher for querying batch spatial cell lists to build neighbor matrices.

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance for multiple systems using pure warp operations.
    Mirrors the single-system :func:`nvalchemiops.neighbors.cell_list.query_cell_list`
    signature: ``algorithm`` selects which of the two batch query kernels to
    launch; pair-centric requires additional caller-allocated scratch +
    metadata that the atom-centric path doesn't need (mirrors the existing
    asymmetry where the atom-centric batch kernel is non-sorted scattered
    reads, while the pair-centric path needs gather scratch + a global-cell
    → system map).

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems in the batch.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system in the batch.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags for each system and dimension.
    cutoff : float
        Neighbor search cutoff distance.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Number of cells in x, y, z directions for each system.
    neighbor_search_radius : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Radius of neighboring cells to search for each system.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting index in global cell arrays for each system.
        Output from batch_build_cell_list.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Periodic boundary crossings for each atom. Output from batch_build_cell_list.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        3D cell coordinates for each atom. Output from batch_build_cell_list.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        Number of atoms in each cell. Output from batch_build_cell_list.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell. Output from batch_build_cell_list.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Flattened list of atom indices organized by cell. Output from batch_build_cell_list.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships (i < j).
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool, optional
        Per-system rebuild flags. If provided, only systems where rebuild_flags[i]
        is True are processed; others are skipped on the GPU without CPU sync.

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays must be pre-allocated by caller.

    See Also
    --------
    batch_build_cell_list : Build cell list data structures (call before this)
    _batch_cell_list_build_neighbor_matrix_local_count_sorted :
        Sorted-reads atom-centric kernel (CPU + CUDA).
    batch_query_cell_list_pair_centric : Pair-centric alternative (CUDA only).

    Notes
    -----
    Both atom-centric and pair-centric paths consume the per-cell-contiguous
    ``sorted_positions`` / ``sorted_atom_periodic_shifts`` scratch; the
    launcher runs :func:`_gather_positions_by_cell` (one thread per atom)
    to fill them before dispatching the chosen kernel.
    """
    total_atoms = positions.shape[0]

    if algorithm == "pair_centric":
        missing = [
            name
            for name, val in (
                ("cells_per_system", cells_per_system),
                ("cell_to_system", cell_to_system),
                ("n_outer", n_outer),
                ("R_max", R_max),
                ("total_cells", total_cells),
            )
            if val is None
        ]
        if missing:
            raise ValueError(
                f"algorithm='pair_centric' requires the following kwargs to "
                f"be provided (caller-allocated scratch + sync-computed "
                f"metadata): {missing}.  See compute_batch_pair_centric_n_outer "
                f"for n_outer.",
            )
    elif algorithm != "atom_centric":
        raise ValueError(
            f"algorithm must be 'atom_centric' | 'pair_centric', got {algorithm!r}",
        )

    # Both paths consume per-cell-contiguous sorted positions/shifts;
    # _gather_positions_by_cell fills the caller-provided scratch.
    wp.launch(
        _gather_positions_by_cell_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            atom_periodic_shifts,
            cell_atom_list,
            sorted_positions,
            sorted_atom_periodic_shifts,
        ],
        device=device,
    )

    if algorithm == "pair_centric":
        batch_query_cell_list_pair_centric(
            positions=positions,
            cell=cell,
            pbc=pbc,
            cutoff=cutoff,
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            cell_offsets=cell_offsets,
            cells_per_system=cells_per_system,
            atom_periodic_shifts=atom_periodic_shifts,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            sorted_positions=sorted_positions,
            sorted_atom_periodic_shifts=sorted_atom_periodic_shifts,
            cell_to_system=cell_to_system,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            wp_dtype=wp_dtype,
            device=device,
            total_cells=int(total_cells),
            n_outer=int(n_outer),
            R_max=R_max,
            half_fill=bool(half_fill),
        )
        return

    selective_zero_num_neighbors(num_neighbors, batch_idx, rebuild_flags, device)
    wp.launch(
        _batch_cell_list_build_neighbor_matrix_local_count_sorted_overload[wp_dtype],
        dim=total_atoms,
        inputs=(
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell,
            pbc,
            batch_idx,
            wp_dtype(cutoff),
            cells_per_dimension,
            neighbor_search_radius,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            cell_offsets,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            bool(half_fill),
            rebuild_flags,
        ),
        device=device,
    )


BLOCK_DIM = 64


def compute_batch_pair_centric_n_outer(
    R_max: tuple[int, int, int], half_fill: bool
) -> int:
    """Compute the number of non-self outer cell offsets at the given radius.

    Used to size the pair-centric launch grid.  Same closed form as
    :func:`nvalchemiops.neighbors.cell_list._compute_pair_centric_n_outer`,
    operating on the cross-system maximum ``R_max``; blocks targeting
    systems with smaller per-axis radii early-return when their decoded
    offset is out-of-range.

    Parameters
    ----------
    R_max : tuple[int, int, int]
        Cross-system maximum per-axis neighbor search radius.
    half_fill : bool
        If True, use the half-shell offset count; otherwise the full
        shell minus the self entry.

    Returns
    -------
    int
        Number of non-self outer cell offsets to enumerate.

    See Also
    --------
    batch_query_cell_list_pair_centric : Consumer that sizes its launch
        grid by ``(total_cells × (n_outer + 1))``.
    """
    Rx, Ry, Rz = int(R_max[0]), int(R_max[1]), int(R_max[2])
    if half_fill:
        return Rx * (2 * Ry + 1) * (2 * Rz + 1) + Ry * (2 * Rz + 1) + Rz
    return (2 * Rx + 1) * (2 * Ry + 1) * (2 * Rz + 1) - 1


@wp.func_native(snippet="return (int)threadIdx.x;")
def _pc_thread_idx() -> wp.int32: ...


@wp.func_native(snippet="return (int)blockIdx.x;")
def _pc_block_idx() -> wp.int32: ...


@wp.kernel(enable_backward=False, module="unique")
def _batch_pair_centric_outer(
    sorted_positions: wp.array(dtype=Any),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    cutoff: Any,
    cells_per_dimension: wp.array(dtype=Any),
    neighbor_search_radius: wp.array(dtype=Any),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_to_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=Any, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    block_dim_const: wp.int32,
    total_cells: wp.int32,
    n_offsets: wp.int32,
    R_max: wp.vec3i,
    half_fill: wp.bool,
) -> None:
    """Emit pairs for one ``(source_cell, offset_idx)`` per CUDA block.

    Block index encodes ``bid = source_cell * n_offsets + offset_idx``.
    Same kernel handles within-cell (``offset_idx == 0``) and outer-shell
    pairs.  Half-shell decoding via :func:`_decode_shift_index`;
    full-shell prepends the ``(0,0,0)`` self entry and dispatches the
    remaining indices to :func:`_decode_full_shift_index`.

    Blocks whose decoded offset exceeds the source system's per-axis
    radius early-return (heterogeneous-R handling).  Selective rebuild
    is unsupported here; selective callers go through the atom-centric
    kernel.
    """
    bid = _pc_block_idx()
    lane = _pc_thread_idx()
    source_cell = bid / n_offsets
    offset_idx = bid % n_offsets
    if source_cell >= total_cells:
        return

    src_count = atoms_per_cell_count[source_cell]
    if src_count == 0:
        return
    src_start = cell_atom_start_indices[source_cell]

    system_idx = cell_to_system[source_cell]
    s_cell_offset = cell_offsets[system_idx]
    s_cpd = cells_per_dimension[system_idx]
    s_nsr = neighbor_search_radius[system_idx]
    s_cell_mat = cell[system_idx]
    s_cell_transpose = wp.transpose(s_cell_mat)

    pbc_x = pbc[system_idx, 0]
    pbc_y = pbc[system_idx, 1]
    pbc_z = pbc[system_idx, 2]

    cpd_x = s_cpd[0]
    cpd_y = s_cpd[1]
    cpd_z = s_cpd[2]
    cpd_xy = cpd_x * cpd_y

    # Reconstruct (cax, cay, caz) within the system from the global cell index.
    local_cell = source_cell - s_cell_offset
    cax = local_cell % cpd_x
    cay = (local_cell / cpd_x) % cpd_y
    caz = local_cell / cpd_xy

    if half_fill:
        offset_vec = _decode_shift_index(offset_idx, R_max)
    else:
        if offset_idx == 0:
            offset_vec = wp.vec3i(0, 0, 0)
        else:
            offset_vec = _decode_full_shift_index(offset_idx - 1, R_max)
    dx_v = offset_vec[0]
    dy_v = offset_vec[1]
    dz_v = offset_vec[2]
    is_self = dx_v == 0 and dy_v == 0 and dz_v == 0

    # Skip blocks whose offset exceeds this system's per-axis search
    # radius.  When R is heterogeneous across systems, the launch grid
    # is sized for R_max and out-of-range blocks early-return.  The
    # self entry (0,0,0) is always within range.
    if dx_v > s_nsr[0] or dx_v < -s_nsr[0]:
        return
    if dy_v > s_nsr[1] or dy_v < -s_nsr[1]:
        return
    if dz_v > s_nsr[2] or dz_v < -s_nsr[2]:
        return

    target_x = cax + dx_v
    target_y = cay + dy_v
    target_z = caz + dz_v

    if not pbc_x and (target_x < 0 or target_x >= cpd_x):
        return
    if not pbc_y and (target_y < 0 or target_y >= cpd_y):
        return
    if not pbc_z and (target_z < 0 or target_z >= cpd_z):
        return

    cs_x_base, wc_x = wpdivmod(target_x, cpd_x)
    cs_y_base, wc_y = wpdivmod(target_y, cpd_y)
    cs_z_base, wc_z = wpdivmod(target_z, cpd_z)

    nbr_cell = s_cell_offset + wc_x + cpd_x * (wc_y + cpd_y * wc_z)
    nbr_count = atoms_per_cell_count[nbr_cell]
    if nbr_count == 0:
        return
    nbr_start = cell_atom_start_indices[nbr_cell]

    cutoff_distance_sq = cutoff * cutoff
    max_neighbors = neighbor_matrix.shape[1]

    # Stride loop: each lane owns a subset of source-cell atoms.
    slot = lane
    while slot < src_count:
        s = src_start + slot
        atom_idx = cell_atom_list[s]
        central_atom_position = sorted_positions[s]
        central_atom_shift = sorted_atom_periodic_shifts[s]

        for j_local in range(nbr_count):
            if is_self:
                if j_local == slot:
                    continue
            j_slot = nbr_start + j_local
            neighbor_atom_idx = cell_atom_list[j_slot]
            if is_self and half_fill:
                if neighbor_atom_idx <= atom_idx:
                    continue
            neighbor_atom_shift = sorted_atom_periodic_shifts[j_slot]
            neighbor_pos = sorted_positions[j_slot]

            shift_x = cs_x_base
            shift_y = cs_y_base
            shift_z = cs_z_base

            if pbc_x:
                shift_x += central_atom_shift[0] - neighbor_atom_shift[0]
            else:
                shift_x = 0

            if pbc_y:
                shift_y += central_atom_shift[1] - neighbor_atom_shift[1]
            else:
                shift_y = 0

            if pbc_z:
                shift_z += central_atom_shift[2] - neighbor_atom_shift[2]
            else:
                shift_z = 0

            fractional_shift = type(central_atom_position)(
                type(central_atom_position[0])(shift_x),
                type(central_atom_position[0])(shift_y),
                type(central_atom_position[0])(shift_z),
            )
            cartesian_shift = s_cell_transpose * fractional_shift

            dr = neighbor_pos - central_atom_position + cartesian_shift
            distance_sq = wp.dot(dr, dr)

            if distance_sq < cutoff_distance_sq:
                pos_emit = wp.atomic_add(num_neighbors, atom_idx, 1)
                if pos_emit < max_neighbors:
                    neighbor_matrix[atom_idx, pos_emit] = neighbor_atom_idx
                    # Always-write shifts.
                    neighbor_matrix_shifts[atom_idx, pos_emit] = wp.vec3i(
                        shift_x, shift_y, shift_z
                    )
        slot += block_dim_const


@wp.kernel(enable_backward=False)
def _gather_positions_by_cell(
    positions: wp.array(dtype=Any),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell_atom_list: wp.array(dtype=wp.int32),
    sorted_positions: wp.array(dtype=Any),
    sorted_shifts: wp.array(dtype=wp.vec3i),
) -> None:
    """Reorder per-atom positions and shifts into cell-contiguous layout.

    Writes ``sorted_positions[s] = positions[cell_atom_list[s]]`` and the
    corresponding ``sorted_shifts`` entry, indexed by the cell-list flat
    slot index ``s``.  Required so the pair-centric kernel's inner-loop
    reads coalesce within a cell's contiguous atom block.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Per-atom positions in original ordering.
    atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Per-atom integer PBC shift vectors in original ordering.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Cell-contiguous atom indices output by ``batch_build_cell_list``.
    sorted_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: positions gathered in cell-contiguous order.
    sorted_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
        OUTPUT: shifts gathered in cell-contiguous order.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - sorted_positions : Filled in cell-contiguous order
        - sorted_shifts : Filled in cell-contiguous order

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)

    See Also
    --------
    batch_query_cell_list_pair_centric : Consumer of the gathered layout
    """
    idx = wp.tid()
    atom_idx = cell_atom_list[idx]
    sorted_positions[idx] = positions[atom_idx]
    sorted_shifts[idx] = atom_periodic_shifts[atom_idx]


@wp.kernel(enable_backward=False)
def _build_cell_to_system_map(
    cell_offsets: wp.array(dtype=wp.int32),
    cells_per_system: wp.array(dtype=wp.int32),
    cell_to_system: wp.array(dtype=wp.int32),
) -> None:
    """Build a global-cell-index → system-index lookup table.

    Each thread fills ``cell_to_system[c] = s`` for every cell ``c`` in
    its system ``s``, where the cell range is
    ``[cell_offsets[s], cell_offsets[s] + cells_per_system[s])``.

    Parameters
    ----------
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting global cell index of each system.
    cells_per_system : wp.array, shape (num_systems,), dtype=wp.int32
        Number of cells in each system.
    cell_to_system : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: system index for each global cell.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - cell_to_system : Filled with per-cell system indices

    Notes
    -----
    - Thread launch: One thread per system (dim=num_systems)

    See Also
    --------
    batch_query_cell_list_pair_centric : Consumer of the lookup table
    """
    sys_idx = wp.tid()
    offset = cell_offsets[sys_idx]
    count = cells_per_system[sys_idx]
    for c in range(count):
        cell_to_system[offset + c] = sys_idx


T = [wp.float32, wp.float64]
V = [wp.vec3f, wp.vec3d]
M = [wp.mat33f, wp.mat33d]
_batch_pair_centric_outer_overload = {}
_gather_positions_by_cell_overload = {}
for t, v, m in zip(T, V, M):
    _gather_positions_by_cell_overload[t] = wp.overload(
        _gather_positions_by_cell,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=v),
            wp.array(dtype=wp.vec3i),
        ],
    )
    _batch_pair_centric_outer_overload[t] = wp.overload(
        _batch_pair_centric_outer,
        [
            wp.array(dtype=v),  # sorted_positions
            wp.array(dtype=wp.vec3i),  # sorted_atom_periodic_shifts
            wp.array(dtype=m),  # cell
            wp.array2d(dtype=wp.bool),  # pbc
            t,  # cutoff
            wp.array(dtype=wp.vec3i),  # cells_per_dimension
            wp.array(dtype=wp.vec3i),  # neighbor_search_radius
            wp.array(dtype=wp.int32),  # atoms_per_cell_count
            wp.array(dtype=wp.int32),  # cell_atom_start_indices
            wp.array(dtype=wp.int32),  # cell_atom_list
            wp.array(dtype=wp.int32),  # cell_offsets
            wp.array(dtype=wp.int32),  # cell_to_system
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.int32,  # block_dim_const
            wp.int32,  # total_cells
            wp.int32,  # n_offsets (= n_outer + 1; self lives at index 0)
            wp.vec3i,  # R_max
            wp.bool,  # half_fill
        ],
    )


def batch_query_cell_list_pair_centric(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    cell_offsets: wp.array,
    cells_per_system: wp.array,
    atom_periodic_shifts: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    sorted_positions: wp.array,
    sorted_atom_periodic_shifts: wp.array,
    cell_to_system: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    wp_dtype: type,
    device: str,
    total_cells: int,
    n_outer: int,
    R_max: tuple[int, int, int],
    half_fill: bool = True,
    block_dim: int = BLOCK_DIM,
) -> None:
    """Core warp launcher for the pair-centric batched cell-list query.

    Launches ``_batch_pair_centric_outer`` over ``total_cells ×
    (n_outer + 1)`` blocks of ``block_dim`` threads.  ``offset_idx == 0``
    is the self-cell entry (within-cell pairs, with the appropriate filter
    for ``half_fill``); ``offset_idx > 0`` are the outer-shell offsets
    decoded on-the-fly from ``R_max``.  Per-system out-of-range offsets
    early-return.

    Per-emit ``wp.atomic_add(num_neighbors, atom_i, 1)`` because multiple
    blocks may write to the same atom's row; caller must pre-zero
    ``num_neighbors``.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems.
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Unit cell matrices for each system.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags.
    cutoff : float
        Neighbor search cutoff distance.
    cells_per_dimension : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Cells per axis for each system.  From :func:`batch_build_cell_list`.
    neighbor_search_radius : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        Per-system search radius.  From :func:`batch_build_cell_list`.
    cell_offsets : wp.array, shape (num_systems,), dtype=wp.int32
        Starting global cell index per system.  From :func:`batch_build_cell_list`.
    cells_per_system : wp.array, shape (num_systems,), dtype=wp.int32
        Number of cells per system.  From :func:`batch_build_cell_list`.
    atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Per-atom PBC shift vectors.  From :func:`batch_build_cell_list`.
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        Atoms per cell.  From :func:`batch_build_cell_list`.
    cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
        Cell start indices into ``cell_atom_list``.  From :func:`batch_build_cell_list`.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Cell-organized atom indices.  From :func:`batch_build_cell_list`.
    sorted_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT (scratch): cell-contiguous reordered positions.  Caller
        allocates; ``_gather_positions_by_cell`` overwrites each call.
    sorted_atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
        OUTPUT (scratch): cell-contiguous reordered shifts.
    cell_to_system : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT (scratch): global-cell → system map.  Caller allocates;
        ``_build_cell_to_system_map`` overwrites each call.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: neighbor indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: per-pair PBC shift vectors.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT (atomic): per-atom neighbor counts.  Caller must zero
        before this call.
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string (e.g. ``"cuda:0"``).
    total_cells : int
        Sum of ``cells_per_system``.  Caller computes once per geometry change.
    n_outer : int
        Non-self outer cell offsets at ``R_max``.  From
        :func:`compute_batch_pair_centric_n_outer`.
    R_max : tuple[int, int, int]
        Cross-system maximum per-axis search radius.
    half_fill : bool, default True
        If True, half-shell + ``j > i`` self filter; if False, full shell
        + ``j != i`` self filter.
    block_dim : int, default BLOCK_DIM
        Threads per CUDA block.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - sorted_positions, sorted_atom_periodic_shifts : Scratch rewritten
        - cell_to_system : Scratch rewritten
        - neighbor_matrix : Filled with neighbor atom indices
        - neighbor_matrix_shifts : Filled with shift vectors
        - num_neighbors : Updated atomically with per-atom counts

    Notes
    -----
    - Pair-centric kernels use raw ``threadIdx`` / ``blockIdx`` via
      ``wp.func_native``; CUDA-only.

    See Also
    --------
    batch_query_cell_list : Atom-centric alternative
    compute_batch_pair_centric_n_outer : Computes ``n_outer``
    _batch_pair_centric_outer : Kernel that performs the enumeration
    """
    natom = int(positions.shape[0])
    num_systems = int(cell.shape[0])
    block_dim_int = int(block_dim)
    n_outer_int = int(n_outer)
    hf = bool(half_fill)
    R_max_vec = wp.vec3i(int(R_max[0]), int(R_max[1]), int(R_max[2]))

    # Build the cell-to-system map (cheap; num_systems threads).
    wp.launch(
        _build_cell_to_system_map,
        dim=num_systems,
        inputs=[cell_offsets, cells_per_system, cell_to_system],
        device=device,
    )

    # Gather positions / shifts into cell-contiguous order.
    wp.launch(
        _gather_positions_by_cell_overload[wp_dtype],
        dim=natom,
        inputs=[
            positions,
            atom_periodic_shifts,
            cell_atom_list,
            sorted_positions,
            sorted_atom_periodic_shifts,
        ],
        device=device,
    )

    # +1 for the self entry at offset_idx==0.
    n_offsets_int = n_outer_int + 1
    wp.launch(
        _batch_pair_centric_outer_overload[wp_dtype],
        dim=total_cells * n_offsets_int * block_dim_int,
        block_dim=block_dim_int,
        inputs=[
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell,
            pbc,
            wp_dtype(cutoff),
            cells_per_dimension,
            neighbor_search_radius,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            cell_offsets,
            cell_to_system,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            block_dim_int,
            total_cells,
            n_offsets_int,
            R_max_vec,
            hf,
        ],
        device=device,
    )
