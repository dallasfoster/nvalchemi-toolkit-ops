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

"""CSR cell-list neighbor-list build and query.

This is the production O(N) cell-list kernel.  Two query paths share the
same CSR build:

* **Atom-centric** (:func:`query_cell_list_local_count_sorted`) — one
  thread per atom; thread-local-counter optimisation.  Best at large
  N or when each cell holds few atoms.
* **Pair-centric** (:func:`query_cell_list_pair_centric_sorted`) — one
  CUDA block per ``(source_cell, outer_offset)`` pair plus a self-cell
  kernel.  Per-emit ``wp.atomic_add(num_neighbors, atom_i, 1)`` trades
  the thread-local-counter optimisation for ``ncell × n_outer``
  parallelism, which scales much better at small/medium N or large
  cutoff.  The torch wrapper auto-selects between the two using a
  sync-free atoms-per-cell heuristic.

Layout (CSR)
~~~~~~~~~~~~

* ``cells_per_dimension``      — grid shape ``(nx, ny, nz)`` per axis.
* ``atom_periodic_shifts[a]``  — integer wrap shift each atom picked up
  while being mapped into the primary box.
* ``atom_to_cell_mapping[a]``  — ``(ix, iy, iz)`` cell coordinate of atom ``a``.
* ``atoms_per_cell_count[c]``  — number of atoms in cell ``c``.
* ``cell_atom_start_indices[c]`` — cumulative offset into ``cell_atom_list``.
* ``cell_atom_list``           — atom indices, segmented per cell.

Always-write shift contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The query kernels write ``neighbor_matrix_shifts[i, n]`` unconditionally
at every active slot.  Callers may allocate ``neighbor_matrix_shifts``
via ``torch.empty(...)`` and skip ``neighbor_matrix_shifts.zero_()``
between steps; tail slots (``column >= num_neighbors[i]``) are left
uninitialized and consumers gate on ``neighbor_matrix != sentinel``.

See Also
--------
:mod:`nvalchemiops.torch.neighbors.cell_list` :
    PyTorch bindings + auto-select between atom-centric and pair-centric.
"""

import os
from typing import Any

import warp as wp

from nvalchemiops.math import wpdivmod
from nvalchemiops.neighbors.neighbor_utils import (
    _decode_full_shift_index,
    _decode_shift_index,
    gather_fused_overload,
)

# The warp-level cell_list API holds no hidden scratch — callers
# preallocate ``sorted_positions``, ``sorted_atom_periodic_shifts``, and
# ``rebuild_flags`` (a 1-element ``wp.bool`` array) and reuse them across
# calls.  This keeps the warp layer graph-capture-safe with no surprise
# allocations.  The torch wrapper manages per-process scratch lifetime on
# behalf of users.


__all__ = [
    "build_cell_list",
    "make_outer_neigh_offsets",
    "query_cell_list",
    "query_cell_list_local_count_sorted",
    "query_cell_list_pair_centric_sorted",
]


def make_outer_neigh_offsets(neighbor_search_radius: int) -> list[tuple[int, int, int]]:
    """Generate the half-shell ``(dx, dy, dz)`` offsets for the outer-cell loop.

    Returns the set of integer cell offsets reachable within
    ``neighbor_search_radius`` cells in any direction, restricted to one
    half-space so each unordered cell pair is enumerated exactly once.
    For ``radius = 1`` returns 13 offsets; for ``radius = 2`` returns 62.

    Parameters
    ----------
    neighbor_search_radius : int
        Maximum number of cells to step in any axis.  Must be ``>= 1``.

    Returns
    -------
    list[tuple[int, int, int]]
        Half-shell offset tuples ``(dx, dy, dz)`` excluding ``(0, 0, 0)``.

    Notes
    -----
    Half-space convention: ``(dx > 0)`` OR ``(dx == 0 AND dy > 0)`` OR
    ``(dx == 0 AND dy == 0 AND dz > 0)`` — picks one cell from each
    unordered ``(a, b)`` cell pair.  Used by the pair-centric query
    kernel together with an atom-index half-fill on same-cell pairs.
    """
    R = int(neighbor_search_radius)
    offsets = []
    for dx in range(0, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0):
                    offsets.append((dx, dy, dz))
    return offsets


###########################################################################################
########################### Cell List Construction ########################################
###########################################################################################


@wp.kernel(enable_backward=False)
def _estimate_cell_list_sizes(
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cell_size: Any,
    max_nbins: Any,
    number_of_cells: wp.array(dtype=Any),
    neighbor_search_radius: wp.array(dtype=Any),
) -> None:
    """Estimate allocation sizes for torch.compile-friendly cell list construction.

    Parameters
    ----------
    cell : wp.array(dtype=Any), shape (1, 3, 3)
        Unit cell matrix defining the simulation box.
    pbc : wp.array(dtype=Any), shape (3,), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
        True enables PBC, False disables it for that dimension.
    cell_size : Any
        Size of the cells in the simulation box.
    max_nbins : Any
        Maximum number of cells to allocate.
    number_of_cells : wp.array(dtype=Any), shape (1,)
        Output: Number of cells in the simulation box.
    neighbor_search_radius : wp.array(dtype=Any), shape (3,)
        Output: Radius of neighboring cells to search in each dimension.
    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: number_of_cells
    - Handles: periodic boundaries by wrapping and clamping
    """
    # Convert cell matrix to inverse transpose for coordinate transformations
    inverse_cell_transpose = wp.transpose(wp.inverse(cell[0]))

    cells_per_dimension = wp.vec3i(0, 0, 0)
    # Calculate optimal number of cells in each dimension.  The search
    # radius is computed AFTER any adaptive promotion below so that R
    # tracks the actual cell width (kernel formulation
    # ``R = ceil(cell_size * cells / face_distance) =
    # ceil(cell_size / actual_cell_width)`` keeps R consistent with
    # the cutoff).
    for i in range(3):
        # Distance between parallel faces in reciprocal space
        face_distance = type(cell_size)(1.0) / wp.length(inverse_cell_transpose[i])
        cells_per_dimension[i] = max(wp.int32(face_distance / cell_size), 1)

    # Adaptive promotion: when the natural cell grid is small, the
    # atom-centric query kernel suffers from low cell-level parallelism
    # and high atoms-per-cell — at ``nx = 1`` it degenerates to O(N²)
    # in a single cell.  Promote each axis up to a minimum cells/axis
    # floor so each cell holds fewer atoms; the search radius
    # downstream grows to compensate.  Skipped on non-PBC axes that
    # genuinely have a single cell (open boundaries handle this in the
    # query kernel).
    ADAPTIVE_MIN_CELLS = wp.int32(4)
    for i in range(3):
        if pbc[i] or cells_per_dimension[i] > 1:
            while cells_per_dimension[i] < ADAPTIVE_MIN_CELLS:
                cells_per_dimension[i] = cells_per_dimension[i] * 2

    # Now that cells_per_dimension is final, compute the search radius.
    # Recompute face_distance per axis (cheap; the transpose+inverse is
    # already cached above).
    for i in range(3):
        face_distance = type(cell_size)(1.0) / wp.length(inverse_cell_transpose[i])
        if cells_per_dimension[i] == 1 and not pbc[i]:
            neighbor_search_radius[i] = 0
        else:
            neighbor_search_radius[i] = wp.int32(
                wp.ceil(
                    cell_size * type(cell_size)(cells_per_dimension[i]) / face_distance
                )
            )

    # Check if total cell count exceeds maximum allowed
    total_cells = int(
        cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
    )

    # Reduce cell count if necessary by halving dimensions iteratively
    while total_cells > max_nbins:
        for i in range(3):
            cells_per_dimension[i] = max(cells_per_dimension[i] // 2, 1)
        total_cells = int(
            cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
        )

    number_of_cells[0] = total_cells


@wp.kernel(enable_backward=False)
def _cell_list_construct_bin_size(
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    target_cell_size: Any,
    max_cells_allowed: Any,
) -> None:
    """Determine optimal spatial decomposition parameters for cell list construction.

    This kernel calculates the number of cells needed in each spatial dimension
    and the neighbor search radius based on the simulation cell geometry and
    target cell size. Assumes a single system (not batched).

    The algorithm:
    1. Computes optimal cell count per dimension based on cell geometry
    2. Reduces cell count if total exceeds maximum allowed
    3. Calculates neighbor search radius to ensure completeness

    Parameters
    ----------
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix defining simulation box geometry.
    pbc : wp.array, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        OUTPUT: Number of cells to create in x, y, z directions.
    target_cell_size : float
        Desired cell size, typically the neighbor cutoff distance.
    max_cells_allowed : int
        Maximum total number of cells allowed (nx * ny * nz ≤ max_cells_allowed).

    Notes
    -----
    - Modifies: cells_per_dimension, neighbor_search_radius
    - Thread launch: Single thread (dim=1)
    - For non-periodic directions with only 1 cell, search radius is set to 0
    """

    # Convert cell matrix to inverse transpose for coordinate transformations
    inverse_cell_transpose = wp.transpose(wp.inverse(cell[0]))

    # Calculate optimal number of cells in each dimension
    for i in range(3):
        # Distance between parallel faces in reciprocal space
        face_distance = type(target_cell_size)(1.0) / wp.length(
            inverse_cell_transpose[i]
        )
        cells_per_dimension[i] = max(wp.int32(face_distance / target_cell_size), 1)

    # Adaptive promotion (mirror of the logic in ``_estimate_cell_list_sizes``):
    # ensure ≥ ADAPTIVE_MIN_CELLS cells per PBC axis so the atom-centric
    # query doesn't degenerate to O(N²) at small box / large cutoff.
    ADAPTIVE_MIN_CELLS = wp.int32(4)
    for i in range(3):
        if pbc[i] or cells_per_dimension[i] > 1:
            while cells_per_dimension[i] < ADAPTIVE_MIN_CELLS:
                cells_per_dimension[i] = cells_per_dimension[i] * 2

    # Check if total cell count exceeds maximum allowed
    total_cells = int(
        cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
    )

    # Reduce cell count if necessary by halving dimensions iteratively
    while total_cells > max_cells_allowed:
        for i in range(3):
            cells_per_dimension[i] = max(cells_per_dimension[i] // 2, 1)
        total_cells = int(
            cells_per_dimension[0] * cells_per_dimension[1] * cells_per_dimension[2]
        )


@wp.kernel(enable_backward=False)
def _cell_list_count_atoms_per_bin(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    atoms_per_cell_count: wp.array(dtype=Any),
    atom_periodic_shifts: wp.array(dtype=Any),
) -> None:
    """Count atoms in each spatial cell and compute periodic boundary shifts.

    This is the first pass of the two-pass cell list construction algorithm.
    Each thread processes one atom, determines which cell it belongs to,
    handles periodic boundary conditions, and atomically increments the
    atom count for that cell.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for coordinate transformations.
    pbc : wp.array, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of spatial cells in x, y, z directions.
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: Number of atoms assigned to each cell (modified atomically).
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: Periodic boundary crossings for each atom.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: atoms_per_cell_count, atom_periodic_shifts
    - Uses atomic operations for thread-safe counting
    - Handles periodic boundaries by wrapping coordinates and tracking shifts
    """
    atom_idx = wp.tid()

    # Transform to fractional coordinates
    inverse_cell_transpose = wp.transpose(wp.inverse(cell[0]))
    fractional_position = inverse_cell_transpose * positions[atom_idx]

    # Determine which cell this atom belongs to
    cell_coords = wp.vec3i(0, 0, 0)
    for dim in range(3):
        cell_coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim]
                * type(fractional_position[dim])(cells_per_dimension[dim])
            )
        )

        # Handle periodic boundary conditions
        if pbc[dim]:
            cell_before_wrap = cell_coords[dim]
            num_cells = cells_per_dimension[dim]
            quotient, remainder = wpdivmod(cell_before_wrap, num_cells)
            atom_periodic_shifts[atom_idx][dim] = quotient
            cell_coords[dim] = remainder
        else:
            # Clamp to valid cell range for non-periodic dimensions
            atom_periodic_shifts[atom_idx][dim] = 0
            cell_coords[dim] = wp.clamp(
                cell_coords[dim], 0, cells_per_dimension[dim] - 1
            )

    # Convert 3D cell coordinates to linear index
    linear_cell_index = cell_coords[0] + cells_per_dimension[0] * (
        cell_coords[1] + cells_per_dimension[1] * cell_coords[2]
    )

    # Atomically increment the count for this cell
    wp.atomic_add(atoms_per_cell_count, linear_cell_index, 1)


@wp.kernel(enable_backward=False)
def _cell_list_bin_atoms(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cells_per_dimension: wp.array(dtype=Any),
    atom_to_cell_mapping: wp.array(dtype=Any),
    atoms_per_cell_count: wp.array(dtype=Any),
    cell_atom_start_indices: wp.array(dtype=Any),
    cell_atom_list: wp.array(dtype=Any),
) -> None:
    """Assign atoms to their spatial cells and build cell-to-atom mapping.

    This is the second pass of the two-pass cell list construction algorithm.
    Each thread processes one atom, determines its cell assignment, and adds
    it to that cell's atom list using atomic operations for thread safety.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for coordinate transformations.
    pbc : wp.array, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of spatial cells in x, y, z directions.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: 3D cell coordinates for each atom.
    atoms_per_cell_count : wp.array, shape (total_cells,), dtype=wp.int32
        MODIFIED: Running count of atoms added to each cell (reset before use).
    cell_atom_start_indices : wp.array, shape (total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell's atoms.
    cell_atom_list : wp.array, shape (total_cells,), dtype=wp.int32
        OUTPUT: Flattened list of atom indices organized by cell.

    Notes
    -----
    - Thread launch: One thread per atom (dim=total_atoms)
    - Modifies: atom_to_cell_mapping, atoms_per_cell_count, cell_atom_list
    - atoms_per_cell_count must be zeroed before calling this kernel
    - Uses atomic operations for thread-safe list building
    """
    atom_idx = wp.tid()

    # Safety check for thread bounds
    if atom_idx >= positions.shape[0]:
        return

    # Transform to fractional coordinates
    inverse_cell_transpose = wp.transpose(wp.inverse(cell[0]))
    fractional_position = inverse_cell_transpose * positions[atom_idx]

    # Determine which cell this atom belongs to
    cell_coords = wp.vec3i(0, 0, 0)
    for dim in range(3):
        cell_coords[dim] = wp.int32(
            wp.floor(
                fractional_position[dim]
                * type(fractional_position[dim])(cells_per_dimension[dim])
            )
        )

        # Handle periodic boundary conditions
        if pbc[dim]:
            cell_before_wrap = cell_coords[dim]
            num_cells = cells_per_dimension[dim]
            _, remainder = wpdivmod(cell_before_wrap, num_cells)
            cell_coords[dim] = remainder
        else:
            # Clamp to valid cell range for non-periodic dimensions
            cell_coords[dim] = wp.clamp(
                cell_coords[dim], 0, cells_per_dimension[dim] - 1
            )

    # Store the cell assignment for this atom
    atom_to_cell_mapping[atom_idx] = cell_coords

    # Convert 3D cell coordinates to linear index
    linear_cell_index = cell_coords[0] + cells_per_dimension[0] * (
        cell_coords[1] + cells_per_dimension[1] * cell_coords[2]
    )

    # Atomically get position in this cell's atom list
    position_in_cell = wp.atomic_add(atoms_per_cell_count, linear_cell_index, 1)

    # Calculate final position in flattened atom list
    final_list_index = cell_atom_start_indices[linear_cell_index] + position_in_cell

    # Store this atom's index in the cell's atom list
    cell_atom_list[final_list_index] = atom_idx


@wp.kernel(enable_backward=False)
def _cell_list_build_neighbor_matrix_local_count_sorted(
    sorted_positions: wp.array(dtype=Any),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=bool),
    cutoff: Any,
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=Any, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Sorted-position variant of the local-counter cell-list query.

    Each thread iterates over the *sorted* atom slot ``s`` (so consecutive
    threads in a warp work on consecutive atoms within a cell, when the cell
    has at least warp-size atoms).  All inner-loop reads —
    ``sorted_positions[cell_start + j]``,
    ``sorted_atom_periodic_shifts[cell_start + j]``,
    ``cell_atom_list[cell_start + j]`` (used for the output write only) —
    advance sequentially within a neighbor cell, giving coalesced loads
    across a warp regardless of where the atoms originally lived in the
    flat ``positions`` array.

    The home atom's read (``sorted_positions[s]``) and the home cell lookup
    (``atom_to_cell_mapping[atom_idx]``) are still single per-thread
    reads — uncoalesced but only one each, so cheap.

    Output writes go to ``neighbor_matrix[atom_idx, n]`` where
    ``atom_idx = cell_atom_list[s]`` is the original atom index.  This
    keeps the public output API identical to the non-sorted kernel.

    Both half_fill modes supported:

    - ``half_fill=True``  — iterates the half-shell of cell offsets and
      filters ``j <= i`` in the self cell.  Each unordered pair is
      enumerated by the lower-index atom only.
    - ``half_fill=False`` — iterates the full shell and filters ``j == i``
      in the self cell.  Each pair is enumerated by BOTH endpoints (each
      thread writes only to its own row, so the local-counter trick is
      still valid).

    Selective rebuild: ``rebuild_flags`` is a 1-element ``wp.bool`` array.
    When ``False`` the kernel returns immediately (caller is responsible
    for preserving the prior ``neighbor_matrix`` contents).  Non-selective
    callers pass the module-level always-True sentinel (no special-case
    in the kernel body).
    """
    if not rebuild_flags[0]:
        return
    s = wp.tid()  # sorted slot index
    if s >= sorted_positions.shape[0]:
        return

    atom_idx = cell_atom_list[s]
    if atom_idx >= sorted_positions.shape[0]:
        # Sentinel for unused slot if the layout pads.
        return

    cutoff_distance_sq = cutoff * cutoff
    central_atom_position = sorted_positions[s]
    central_atom_shift = sorted_atom_periodic_shifts[s]
    central_atom_cell = atom_to_cell_mapping[atom_idx]
    max_neighbors = neighbor_matrix.shape[1]

    cell_mat = cell[0]
    cell_transpose = wp.transpose(cell_mat)

    cpd_x = cells_per_dimension[0]
    cpd_y = cells_per_dimension[1]
    cpd_z = cells_per_dimension[2]

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]

    # Outer-axis lower bound: 0 for half-shell, -R[0] for full shell.
    dx_lo = wp.int32(0)
    if not half_fill:
        dx_lo = -neighbor_search_radius[0]

    n = wp.int32(0)

    for dx in range(dx_lo, neighbor_search_radius[0] + 1):
        for dy in range(-neighbor_search_radius[1], neighbor_search_radius[1] + 1):
            for dz in range(-neighbor_search_radius[2], neighbor_search_radius[2] + 1):
                if half_fill:
                    # Half-space convention — pick one cell from each
                    # unordered cell pair; same-cell handled via inner
                    # j-filter below.
                    if not (
                        dx > 0
                        or (dx == 0 and dy > 0)
                        or (dx == 0 and dy == 0 and dz >= 0)
                    ):
                        continue
                target_x = central_atom_cell[0] + dx
                target_y = central_atom_cell[1] + dy
                target_z = central_atom_cell[2] + dz

                if not pbc_x and (target_x < 0 or target_x >= cpd_x):
                    continue
                if not pbc_y and (target_y < 0 or target_y >= cpd_y):
                    continue
                if not pbc_z and (target_z < 0 or target_z >= cpd_z):
                    continue

                cs_x, wc_x = wpdivmod(target_x, cpd_x)
                cs_y, wc_y = wpdivmod(target_y, cpd_y)
                cs_z, wc_z = wpdivmod(target_z, cpd_z)

                linear_cell_index = wc_x + cpd_x * (wc_y + cpd_y * wc_z)

                cell_start_index = cell_atom_start_indices[linear_cell_index]
                num_atoms_in_cell = atoms_per_cell_count[linear_cell_index]

                for cell_atom_idx in range(num_atoms_in_cell):
                    j_slot = cell_start_index + cell_atom_idx
                    # ALL reads in this loop body are sequential within the
                    # cell (j_slot increments by 1 each iter) → coalesced
                    # across a warp when threads share the same outer cell.
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

                    fractional_shift = type(central_atom_position)(
                        type(central_atom_position[0])(shift_x),
                        type(central_atom_position[0])(shift_y),
                        type(central_atom_position[0])(shift_z),
                    )
                    cartesian_shift = cell_transpose * fractional_shift

                    dr = neighbor_pos - central_atom_position + cartesian_shift
                    distance_sq = wp.dot(dr, dr)

                    if distance_sq < cutoff_distance_sq:
                        if n < max_neighbors:
                            neighbor_matrix[atom_idx, n] = neighbor_atom_idx
                            # Always-write shifts.
                            neighbor_matrix_shifts[atom_idx, n] = wp.vec3i(
                                shift_x, shift_y, shift_z
                            )
                        n += 1

    num_neighbors[atom_idx] = n


@wp.func_native(snippet="return (int)threadIdx.x;")
def _cell_centric_thread_idx() -> wp.int32: ...


@wp.func_native(snippet="return (int)blockIdx.x;")
def _cell_centric_block_idx() -> wp.int32: ...


@wp.kernel(enable_backward=False, module="unique")
def _cell_list_build_neighbor_matrix_pair_centric_outer(
    sorted_positions: wp.array(dtype=Any),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=bool),
    cutoff: Any,
    cells_per_dimension: wp.array(dtype=wp.int32),
    neighbor_search_radius: wp.array(dtype=wp.int32),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=Any, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    block_dim_const: wp.int32,
    total_cells: wp.int32,
    n_outer: wp.int32,
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Emit pairs for one (source_cell, outer_offset) pair per CUDA block.

    Block index encodes ``bid = source_cell * n_outer + offset_idx``.
    Each block decodes the outer offset on-the-fly via
    :func:`_decode_shift_index` (half-shell, ``half_fill=True``) or
    :func:`_decode_full_shift_index` (full shell, ``half_fill=False``),
    then lanes stride-loop over the source cell's atoms.  Per-pair
    distance check + atomic emit into the source atom's row.

    Self-cell pairs (``dx = dy = dz = 0``) are NOT handled here — they
    belong to the companion kernel
    :func:`_cell_list_build_neighbor_matrix_pair_centric_self`.

    Selective rebuild: ``rebuild_flags`` is a 1-element ``wp.bool``
    array; ``False`` makes every block return immediately.  Non-selective
    callers pass the module-level always-True sentinel.

    Always-write shift contract (caller skips
    ``neighbor_matrix_shifts.zero_()``).
    """
    if not rebuild_flags[0]:
        return
    bid = _cell_centric_block_idx()
    lane = _cell_centric_thread_idx()
    source_cell = bid / n_outer
    offset_idx = bid % n_outer
    if source_cell >= total_cells:
        return

    src_count = atoms_per_cell_count[source_cell]
    if src_count == 0:
        return
    src_start = cell_atom_start_indices[source_cell]

    cutoff_distance_sq = cutoff * cutoff
    max_neighbors = neighbor_matrix.shape[1]

    cell_mat = cell[0]
    cell_transpose = wp.transpose(cell_mat)

    cpd_x = cells_per_dimension[0]
    cpd_y = cells_per_dimension[1]
    cpd_z = cells_per_dimension[2]
    cpd_xy = cpd_x * cpd_y

    cax = source_cell % cpd_x
    cay = (source_cell / cpd_x) % cpd_y
    caz = source_cell / cpd_xy

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]

    # Decode the outer offset on-the-fly.  Half-shell uses idx+1 to skip
    # the (0, 0, 0) self entry at the start of _decode_shift_index's
    # enumeration; full shell uses _decode_full_shift_index which already
    # excludes self.
    R_vec = wp.vec3i(
        neighbor_search_radius[0],
        neighbor_search_radius[1],
        neighbor_search_radius[2],
    )
    if half_fill:
        offset_vec = _decode_shift_index(offset_idx + 1, R_vec)
    else:
        offset_vec = _decode_full_shift_index(offset_idx, R_vec)
    dx_v = offset_vec[0]
    dy_v = offset_vec[1]
    dz_v = offset_vec[2]

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

    nbr_cell = wc_x + cpd_x * (wc_y + cpd_y * wc_z)
    nbr_count = atoms_per_cell_count[nbr_cell]
    if nbr_count == 0:
        return
    nbr_start = cell_atom_start_indices[nbr_cell]

    # Stride loop: each lane owns a subset of source-cell atoms.
    slot = lane
    while slot < src_count:
        s = src_start + slot
        atom_idx = cell_atom_list[s]
        central_atom_position = sorted_positions[s]
        central_atom_shift = sorted_atom_periodic_shifts[s]

        for j_local in range(nbr_count):
            j_slot = nbr_start + j_local
            neighbor_atom_idx = cell_atom_list[j_slot]
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
            cartesian_shift = cell_transpose * fractional_shift

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


@wp.kernel(enable_backward=False, module="unique")
def _cell_list_build_neighbor_matrix_pair_centric_self(
    sorted_positions: wp.array(dtype=Any),
    sorted_atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=bool),
    cutoff: Any,
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=Any, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    block_dim_const: wp.int32,
    total_cells: wp.int32,
    half_fill: wp.bool,
    rebuild_flags: wp.array(dtype=wp.bool),
) -> None:
    """Emit same-cell pairs for one source cell per CUDA block.

    Self-cell filter chosen by ``half_fill``:

    - ``half_fill=True``  — only emit when ``neighbor_atom_idx > atom_idx``
      (atom-index half-fill).  Each unordered same-cell pair appears once.
    - ``half_fill=False`` — emit for every ``neighbor_atom_idx != atom_idx``.
      Each ordered same-cell pair appears once (both directions written).

    Distance check applies because ``cell_width = cutoff`` permits
    intra-cell distances up to ``cell_width * sqrt(3)`` which exceeds
    ``cutoff``.  Per-emit ``wp.atomic_add(num_neighbors, atom_idx, 1)``
    to share rows with the outer kernel's writes for the same ``atom_idx``.

    Selective rebuild via ``rebuild_flags`` (same semantics as the
    pair-centric outer kernel and the atom-centric sorted kernel).
    """
    if not rebuild_flags[0]:
        return
    cell_idx = _cell_centric_block_idx()
    lane = _cell_centric_thread_idx()
    if cell_idx >= total_cells:
        return

    src_count = atoms_per_cell_count[cell_idx]
    if src_count <= 1:
        return
    src_start = cell_atom_start_indices[cell_idx]

    cutoff_distance_sq = cutoff * cutoff
    max_neighbors = neighbor_matrix.shape[1]
    cell_mat = cell[0]
    cell_transpose = wp.transpose(cell_mat)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]

    slot = lane
    while slot < src_count:
        s = src_start + slot
        atom_idx = cell_atom_list[s]
        central_atom_position = sorted_positions[s]
        central_atom_shift = sorted_atom_periodic_shifts[s]

        for j_local in range(src_count):
            if j_local == slot:
                continue
            j_slot = src_start + j_local
            neighbor_atom_idx = cell_atom_list[j_slot]
            if half_fill:
                if neighbor_atom_idx <= atom_idx:
                    continue
            # half_fill=False: keep every ``neighbor_atom_idx != atom_idx``
            # (the j_local == slot guard already drops the self pair).

            neighbor_atom_shift = sorted_atom_periodic_shifts[j_slot]
            neighbor_pos = sorted_positions[j_slot]

            shift_x = wp.int32(0)
            shift_y = wp.int32(0)
            shift_z = wp.int32(0)
            if pbc_x:
                shift_x = central_atom_shift[0] - neighbor_atom_shift[0]
            if pbc_y:
                shift_y = central_atom_shift[1] - neighbor_atom_shift[1]
            if pbc_z:
                shift_z = central_atom_shift[2] - neighbor_atom_shift[2]

            fractional_shift = type(central_atom_position)(
                type(central_atom_position[0])(shift_x),
                type(central_atom_position[0])(shift_y),
                type(central_atom_position[0])(shift_z),
            )
            cartesian_shift = cell_transpose * fractional_shift

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


T = [wp.float32, wp.float64]
V = [wp.vec3f, wp.vec3d]
M = [wp.mat33f, wp.mat33d]
_estimate_cell_list_sizes_overload = {}
_cell_list_construct_bin_size_overload = {}
_cell_list_count_atoms_per_bin_overload = {}
_cell_list_bin_atoms_overload = {}
_cell_list_build_neighbor_matrix_local_count_sorted_overload = {}
_cell_list_build_neighbor_matrix_pair_centric_outer_overload = {}
_cell_list_build_neighbor_matrix_pair_centric_self_overload = {}
for t, v, m in zip(T, V, M):
    _estimate_cell_list_sizes_overload[t] = wp.overload(
        _estimate_cell_list_sizes,
        [
            wp.array(dtype=m),
            wp.array(dtype=wp.bool),
            t,
            wp.int32,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
        ],
    )
    _cell_list_construct_bin_size_overload[t] = wp.overload(
        _cell_list_construct_bin_size,
        [
            wp.array(dtype=m),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            t,
            wp.int32,
        ],
    )
    _cell_list_count_atoms_per_bin_overload[t] = wp.overload(
        _cell_list_count_atoms_per_bin,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
        ],
    )
    _cell_list_bin_atoms_overload[t] = wp.overload(
        _cell_list_bin_atoms,
        [
            wp.array(dtype=v),
            wp.array(dtype=m),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.vec3i),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
        ],
    )
    _cell_list_build_neighbor_matrix_local_count_sorted_overload[t] = wp.overload(
        _cell_list_build_neighbor_matrix_local_count_sorted,
        [
            wp.array(dtype=v),  # sorted_positions
            wp.array(dtype=wp.vec3i),  # sorted_atom_periodic_shifts
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.bool),  # pbc
            t,  # cutoff
            wp.array(dtype=wp.int32),  # cells_per_dimension
            wp.array(dtype=wp.int32),  # neighbor_search_radius
            wp.array(dtype=wp.int32),  # atoms_per_cell_count
            wp.array(dtype=wp.int32),  # cell_atom_start_indices
            wp.array(dtype=wp.int32),  # cell_atom_list
            wp.array(dtype=wp.vec3i),  # atom_to_cell_mapping
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.bool,  # half_fill
            wp.array(dtype=wp.bool),  # rebuild_flags
        ],
    )
    _cell_list_build_neighbor_matrix_pair_centric_outer_overload[t] = wp.overload(
        _cell_list_build_neighbor_matrix_pair_centric_outer,
        [
            wp.array(dtype=v),  # sorted_positions
            wp.array(dtype=wp.vec3i),  # sorted_atom_periodic_shifts
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.bool),  # pbc
            t,  # cutoff
            wp.array(dtype=wp.int32),  # cells_per_dimension
            wp.array(dtype=wp.int32),  # neighbor_search_radius
            wp.array(dtype=wp.int32),  # atoms_per_cell_count
            wp.array(dtype=wp.int32),  # cell_atom_start_indices
            wp.array(dtype=wp.int32),  # cell_atom_list
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.int32,  # block_dim_const
            wp.int32,  # total_cells
            wp.int32,  # n_outer
            wp.bool,  # half_fill
            wp.array(dtype=wp.bool),  # rebuild_flags
        ],
    )
    _cell_list_build_neighbor_matrix_pair_centric_self_overload[t] = wp.overload(
        _cell_list_build_neighbor_matrix_pair_centric_self,
        [
            wp.array(dtype=v),  # sorted_positions
            wp.array(dtype=wp.vec3i),  # sorted_atom_periodic_shifts
            wp.array(dtype=m),  # cell
            wp.array(dtype=wp.bool),  # pbc
            t,  # cutoff
            wp.array(dtype=wp.int32),  # atoms_per_cell_count
            wp.array(dtype=wp.int32),  # cell_atom_start_indices
            wp.array(dtype=wp.int32),  # cell_atom_list
            wp.array2d(dtype=wp.int32),  # neighbor_matrix
            wp.array2d(dtype=wp.vec3i),  # neighbor_matrix_shifts
            wp.array(dtype=wp.int32),  # num_neighbors
            wp.int32,  # block_dim_const
            wp.int32,  # total_cells
            wp.bool,  # half_fill
            wp.array(dtype=wp.bool),  # rebuild_flags
        ],
    )


###########################################################################################
################################ Core Warp Launchers #######################################
###########################################################################################


def build_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    atom_periodic_shifts: wp.array,
    atom_to_cell_mapping: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for building spatial cell list.

    Constructs a spatial decomposition data structure for efficient neighbor searching
    using pure warp operations. This function launches warp kernels to organize atoms
    into spatial cells.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix defining the simulation box.
    pbc : wp.array, shape (3,), dtype=wp.bool
        Periodic boundary condition flags for x, y, z directions.
    cutoff : float
        Maximum distance for neighbor search.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        OUTPUT: Number of cells created in x, y, z directions.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: Periodic boundary crossings for each atom.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Number of atoms in each cell. Must be zeroed by caller before first use.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        OUTPUT: Array for cell start offsets. Caller provides pre-allocated
        and zeroed array.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Flattened list of atom indices organized by cell.
    wp_dtype : type
        Warp dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level Warp interface. The caller must ensure
      ``atoms_per_cell_count`` is zeroed before calling.
    - This function handles the cumsum internally using ``wp.utils.array_scan``.
    - For framework bindings, use the torch/jax wrappers instead.

    See Also
    --------
    query_cell_list : Query cell list to build neighbor matrix (call after this)
    wp.utils.array_scan : Warp utility for computing prefix sums
    _cell_list_construct_bin_size : Kernel for computing cell dimensions
    _cell_list_count_atoms_per_bin : Kernel for counting atoms per cell
    _cell_list_bin_atoms : Kernel for binning atoms into cells
    """
    total_atoms = positions.shape[0]
    max_total_cells = atoms_per_cell_count.shape[0]
    wp_cutoff = wp_dtype(cutoff)

    # Construct cell dimensions
    wp.launch(
        _cell_list_construct_bin_size_overload[wp_dtype],
        dim=1,
        device=device,
        inputs=(
            cell,
            pbc,
            cells_per_dimension,
            wp_cutoff,
            max_total_cells,
        ),
    )

    # Count atoms per bin (expects atoms_per_cell_count to be zeroed by caller)
    wp.launch(
        _cell_list_count_atoms_per_bin_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            cell,
            pbc,
            cells_per_dimension,
            atoms_per_cell_count,
            atom_periodic_shifts,
        ],
        device=device,
    )

    # Compute exclusive scan to get starting indices for each cell
    # This converts [3, 5, 2, 0, 4, ...] -> [0, 3, 8, 10, 10, ...]
    wp.utils.array_scan(atoms_per_cell_count, cell_atom_start_indices, inclusive=False)

    # Zero counts before binning atoms (second pass needs fresh counts)
    atoms_per_cell_count.zero_()

    # Bin atoms (expects atoms_per_cell_count to be zeroed)
    wp.launch(
        _cell_list_bin_atoms_overload[wp_dtype],
        dim=total_atoms,
        inputs=[
            positions,
            cell,
            pbc,
            cells_per_dimension,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ],
        device=device,
    )


def query_cell_list_local_count_sorted(
    sorted_positions: wp.array,
    sorted_atom_periodic_shifts: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    atom_to_cell_mapping: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    half_fill: bool = True,
) -> None:
    """Atom-centric query with sorted-position reads.

    See ``_cell_list_build_neighbor_matrix_local_count_sorted`` for the
    rationale.  The caller is responsible for pre-gathering
    ``sorted_positions`` and ``sorted_atom_periodic_shifts`` via
    ``gather_fused``.  Output writes go to the
    public ``neighbor_matrix[atom_idx, n]`` indexed by the original atom
    index, so downstream code is unaware of the reordering.

    ``half_fill=True`` iterates the half-shell + ``j > i`` self filter;
    ``half_fill=False`` iterates the full shell + ``j != i`` self filter.
    The local-counter (per-thread register) is valid in both modes
    because each thread is the unique writer to its own row.

    ``rebuild_flags`` is a caller-allocated 1-element ``wp.bool`` array.
    Non-selective callers pass a permanent always-True array (allocated
    once and reused).  When ``False`` the kernel returns immediately.
    """
    total_atoms = sorted_positions.shape[0]
    wp.launch(
        _cell_list_build_neighbor_matrix_local_count_sorted_overload[wp_dtype],
        dim=total_atoms,
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
            atom_to_cell_mapping,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            bool(half_fill),
            rebuild_flags,
        ],
        device=device,
    )


def _compute_pair_centric_n_outer(R: tuple[int, int, int], half_fill: bool) -> int:
    """Number of non-self outer cell offsets at per-axis radius ``R``.

    Half-shell: ``Rx*(2*Ry+1)*(2*Rz+1) + Ry*(2*Rz+1) + Rz``.
    Full shell: ``(2*Rx+1)*(2*Ry+1)*(2*Rz+1) - 1``.
    """
    Rx, Ry, Rz = int(R[0]), int(R[1]), int(R[2])
    if half_fill:
        return Rx * (2 * Ry + 1) * (2 * Rz + 1) + Ry * (2 * Rz + 1) + Rz
    return (2 * Rx + 1) * (2 * Ry + 1) * (2 * Rz + 1) - 1


# Dispatch envar: "auto" (default), "atom_centric", or "pair_centric".
_ALGO_ENV = "NVALCHEMI_NEIGHLIST_ALGO"


def _dispatch_algorithm(natom: int, cutoff: float) -> str:
    """Pick ``"atom_centric"`` or ``"pair_centric"`` for the given (N, cutoff).

    Sync-free: takes Python ints / floats, no GPU reads.

    Pair-centric wins iff any of:
      1. ``cutoff >= 8  AND N <= 65536``
      2. ``cutoff >= 6  AND N <=  8192``
      3. ``cutoff >= 4  AND N <=  1024``

    Override via env var ``NVALCHEMI_NEIGHLIST_ALGO`` =
    ``"auto"`` (default) / ``"atom_centric"`` / ``"pair_centric"``.

    Calibrated on GB10 sm_121.  Cross-GPU sensitivity is real but
    bounded (≤ ~35 % wallclock penalty per cell, ≤ ~3 % mean) — recalibrate
    by sweeping ``benchmark_neighborlist.py --methods
    cell_list_atom_centric cell_list_pair_centric`` and editing the rule
    above, or pin a single variant via the env var.
    """
    override = os.environ.get(_ALGO_ENV, "auto").strip()
    if override in ("atom_centric", "pair_centric"):
        return override
    n = int(natom)
    c = float(cutoff)
    if (
        (c >= 8.0 and n <= 65536)
        or (c >= 6.0 and n <= 8192)
        or (c >= 4.0 and n <= 1024)
    ):
        return "pair_centric"
    return "atom_centric"


def query_cell_list_pair_centric_sorted(
    sorted_positions: wp.array,
    sorted_atom_periodic_shifts: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
    atoms_per_cell_count: wp.array,
    cell_atom_start_indices: wp.array,
    cell_atom_list: wp.array,
    neighbor_matrix: wp.array,
    neighbor_matrix_shifts: wp.array,
    num_neighbors: wp.array,
    rebuild_flags: wp.array,
    wp_dtype: type,
    device: str,
    n_outer: int,
    block_dim: int = 64,
    half_fill: bool = True,
) -> None:
    """Pair-centric query: one block per (source cell, outer offset).

    Two kernels:

    * Outer-offset kernel — handles all non-self cell-cell pairs.  Block
      index encodes ``(source_cell, offset_idx)`` so the launch grid
      scales with ``ncell × n_outer``.  The offset table is decoded
      on-the-fly from ``neighbor_search_radius`` + ``half_fill`` via
      :func:`_decode_shift_index` / :func:`_decode_full_shift_index`.
    * Self-cell kernel — handles same-cell pairs.  ``ncell`` blocks.
      ``half_fill=True`` filters ``j > i``; ``half_fill=False`` filters
      ``j != i`` (every ordered pair).

    Both kernels use per-emit ``wp.atomic_add(num_neighbors, atom_i, 1)``
    because multiple blocks (different outer offsets) may target the
    same source atom.

    Pair-set output is identical to the atom-centric kernel; per-row
    ordering inside ``neighbor_matrix`` is non-deterministic across
    builds.

    Parameters
    ----------
    sorted_positions, sorted_atom_periodic_shifts : wp.array
        Pre-gathered sorted-by-cell views (output of ``gather_fused``).
    neighbor_search_radius : wp.array(dtype=wp.int32), shape (3,)
        Per-axis radius ``(Rx, Ry, Rz)``.  The kernel decodes each block's
        outer-offset index into a ``(dx, dy, dz)`` vector on-the-fly via
        the shared shift-index decoders.
    n_outer : int
        Number of non-self outer cell offsets — must match
        ``_compute_pair_centric_n_outer(R, half_fill)``.  Caller computes
        this once per geometry change (the only host-side dependency on
        the per-axis radius).
    half_fill : bool, default True
        Half-shell + ``j > i`` self filter when True; full shell + ``j != i``
        self filter when False.  Same semantics as the atom-centric kernel.
    block_dim : int, default 64
        Threads per CUDA block.  Should bracket typical
        atoms-per-cell counts; cells with more atoms see lanes
        stride-loop.
    rebuild_flags : wp.array(dtype=wp.bool), shape (1,)
        Caller-allocated GPU-resident flag.  ``False`` makes every block
        return immediately and outputs are preserved.  ``True`` runs the
        full build — caller must zero ``num_neighbors`` first so the
        per-emit atomic_adds start from 0.  Non-selective callers
        allocate a permanent always-True array once and reuse it.
    """
    total_cells = int(atoms_per_cell_count.shape[0])
    block_dim_int = int(block_dim)
    n_outer_int = int(n_outer)
    hf = bool(half_fill)

    # Outer-offset kernel: one block per (source_cell, offset_idx).
    wp.launch(
        _cell_list_build_neighbor_matrix_pair_centric_outer_overload[wp_dtype],
        dim=total_cells * n_outer_int * block_dim_int,
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
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            block_dim_int,
            total_cells,
            n_outer_int,
            hf,
            rebuild_flags,
        ],
        device=device,
    )

    # Self-cell kernel: one block per source cell.
    wp.launch(
        _cell_list_build_neighbor_matrix_pair_centric_self_overload[wp_dtype],
        dim=total_cells * block_dim_int,
        block_dim=block_dim_int,
        inputs=[
            sorted_positions,
            sorted_atom_periodic_shifts,
            cell,
            pbc,
            wp_dtype(cutoff),
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            block_dim_int,
            total_cells,
            hf,
            rebuild_flags,
        ],
        device=device,
    )


def query_cell_list(
    positions: wp.array,
    cell: wp.array,
    pbc: wp.array,
    cutoff: float,
    cells_per_dimension: wp.array,
    neighbor_search_radius: wp.array,
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
    n_outer: int | None = None,
) -> None:
    """Core warp launcher for querying spatial cell list to build neighbor matrix.

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance using pure warp operations.  All
    scratch + output arrays (including ``sorted_positions``,
    ``sorted_atom_periodic_shifts`` for the per-cell-contiguous gather,
    and the ``rebuild_flags`` 1-element bool array) are caller-allocated
    — this layer holds no hidden state.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms, 3), dtype=wp.vec3*
        Atomic coordinates in Cartesian space.
    cell : wp.array, shape (1, 3, 3), dtype=wp.mat33*
        Unit cell matrix for periodic boundary coordinate shifts.
    pbc : wp.array, shape (3,), dtype=wp.bool
        Periodic boundary condition flags.
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    cells_per_dimension : wp.array, shape (3,), dtype=wp.int32
        Number of cells in x, y, z directions from build_cell_list_warp.
    neighbor_search_radius : wp.array, shape (3,), dtype=wp.int32
        Radius of neighboring cells to search in each dimension.
    atom_periodic_shifts : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        Periodic boundary crossings for each atom from build_cell_list_warp.
    atom_to_cell_mapping : wp.array, shape (total_atoms, 3), dtype=wp.vec3i
        3D cell coordinates for each atom from build_cell_list_warp.
    atoms_per_cell_count : wp.array, shape (max_total_cells,), dtype=wp.int32
        Number of atoms in each cell from build_cell_list_warp.
    cell_atom_start_indices : wp.array, shape (max_total_cells,), dtype=wp.int32
        Starting index in cell_atom_list for each cell from build_cell_list_warp.
    cell_atom_list : wp.array, shape (total_atoms,), dtype=wp.int32
        Flattened list of atom indices organized by cell from build_cell_list_warp.
    neighbor_matrix : wp.array, shape (total_atoms, max_neighbors), dtype=wp.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
    neighbor_matrix_shifts : wp.array, shape (total_atoms, max_neighbors, 3), dtype=wp.vec3i
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Number of neighbors found for each atom.
    wp_dtype : type
        Warp dtype (``wp.float32`` or ``wp.float64``).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').
    sorted_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Caller-allocated scratch.  ``gather_fused`` writes into it each
        call.
    sorted_atom_periodic_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
        Caller-allocated scratch.  ``gather_fused`` writes into it each
        call.
    rebuild_flags : wp.array, shape (1,), dtype=wp.bool
        Caller-allocated 1-element flag.  ``False`` makes the kernel
        return immediately.  Non-selective callers allocate an always-True
        array once and reuse it.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships (i < j).
    algorithm : {"atom_centric", "pair_centric"}, default "atom_centric"
        Selects which of the two sorted fast-path kernels to launch.
        Both produce identical pair sets for either ``half_fill`` value;
        per-row ordering inside ``neighbor_matrix`` differs.

        ``"pair_centric"`` requires ``n_outer`` (the host-side count of
        non-self outer cell offsets at the per-axis radius — caller
        precomputes via :func:`_compute_pair_centric_n_outer`).  Auto-
        dispatch (sync-free) lives at the torch-wrapper layer where the
        sync is already paid; direct-warp callers pick explicitly.
        Pair-centric is CUDA-only — CPU callers must use atom-centric.
    n_outer : int, optional
        Required when ``algorithm="pair_centric"``.  Number of non-self
        outer cell offsets at the per-axis search radius — see
        :func:`_compute_pair_centric_n_outer` for the closed form.

    Notes
    -----
    - All output AND scratch arrays must be pre-allocated by the caller;
      this layer holds no hidden state.  ``num_neighbors`` must be
      zeroed before each call (atomic_add semantics).  Shifts output
      uses the always-write contract (no prefill required).

    See Also
    --------
    build_cell_list                     : Build cell list (call before this)
    query_cell_list_local_count_sorted  : Atom-centric kernel (both half_fill modes)
    query_cell_list_pair_centric_sorted : Pair-centric alternative
    _dispatch_algorithm                 : Sync-free (N, cutoff) auto-dispatch rule
    _compute_pair_centric_n_outer       : Closed-form for ``n_outer``
    """
    total_atoms = positions.shape[0]

    cpu_only = "cpu" in str(device).lower()
    if algorithm == "atom_centric":
        chosen = "atom_centric"
    elif algorithm == "pair_centric":
        if cpu_only:
            raise ValueError(
                "algorithm='pair_centric' is not supported on CPU "
                "(kernels use raw blockIdx/threadIdx).  Pass "
                "'atom_centric' instead.",
            )
        if n_outer is None:
            raise ValueError(
                "algorithm='pair_centric' requires n_outer.  Compute via "
                "_compute_pair_centric_n_outer((Rx, Ry, Rz), half_fill).",
            )
        chosen = "pair_centric"
    else:
        raise ValueError(
            f"algorithm must be 'atom_centric' | 'pair_centric', got {algorithm!r}",
        )

    if os.environ.get("NVALCHEMI_NEIGHLIST_DISPATCH_LOG"):
        _v = (
            "pair_centric_sorted"
            if chosen == "pair_centric"
            else "atom_centric_local_count_sorted"
        )
        _sel = " (selective)" if rebuild_flags is not None else ""
        print(
            f"[neighlist-dispatch] (wp.launcher) natom={int(total_atoms)} "
            f"cutoff={float(cutoff):.3f} half_fill={bool(half_fill)} -> {_v}{_sel}",
            flush=True,
        )

    # Both fast paths consume per-cell-contiguous sorted positions/shifts;
    # gather_fused writes into caller-provided scratch.
    wp.launch(
        gather_fused_overload[wp_dtype],
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
    if chosen == "pair_centric":
        query_cell_list_pair_centric_sorted(
            sorted_positions=sorted_positions,
            sorted_atom_periodic_shifts=sorted_atom_periodic_shifts,
            cell=cell,
            pbc=pbc,
            cutoff=float(cutoff),
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            rebuild_flags=rebuild_flags,
            wp_dtype=wp_dtype,
            device=device,
            n_outer=int(n_outer),
            block_dim=64,
            half_fill=bool(half_fill),
        )
    else:
        query_cell_list_local_count_sorted(
            sorted_positions=sorted_positions,
            sorted_atom_periodic_shifts=sorted_atom_periodic_shifts,
            cell=cell,
            pbc=pbc,
            cutoff=float(cutoff),
            cells_per_dimension=cells_per_dimension,
            neighbor_search_radius=neighbor_search_radius,
            atoms_per_cell_count=atoms_per_cell_count,
            cell_atom_start_indices=cell_atom_start_indices,
            cell_atom_list=cell_atom_list,
            atom_to_cell_mapping=atom_to_cell_mapping,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            rebuild_flags=rebuild_flags,
            wp_dtype=wp_dtype,
            device=device,
            half_fill=bool(half_fill),
        )
