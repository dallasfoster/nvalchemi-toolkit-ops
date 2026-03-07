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

"""PyTorch bindings for rebuild detection.

This module provides PyTorch custom operators for detecting when cell lists and
neighbor lists need to be rebuilt.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.rebuild_detection import (
    check_batch_cell_list_rebuild,
    check_batch_neighbor_list_rebuild,
    check_cell_list_rebuild,
    check_neighbor_list_rebuild,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "cell_list_needs_rebuild",
    "neighbor_list_needs_rebuild",
    "check_cell_list_rebuild_needed",
    "check_neighbor_list_rebuild_needed",
    "batch_neighbor_list_needs_rebuild",
    "batch_cell_list_needs_rebuild",
]

###########################################################################################
########################### Cell List Rebuild Detection ###################################
###########################################################################################


@torch.library.custom_op("nvalchemiops::_cell_list_needs_rebuild", mutates_args=())
def _cell_list_needs_rebuild(
    current_positions: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Detect if spatial cell list requires rebuilding due to atomic motion.

    Parameters
    ----------
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates in Cartesian space.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from the existing cell list.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of spatial cells in x, y, z directions.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix for coordinate transformations.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.

    Returns
    -------
    rebuild_needed : torch.Tensor, shape (1,), dtype=bool
        True if any atom has moved to a different cell requiring rebuild.

    See Also
    --------
    nvalchemiops.neighborlist.rebuild_detection.wp_check_cell_list_rebuild : Core warp launcher
    cell_list_needs_rebuild : High-level wrapper function
    """
    total_atoms = current_positions.shape[0]
    device = current_positions.device
    pbc = pbc.squeeze(0)

    if total_atoms == 0:
        return torch.tensor([False], device=device, dtype=torch.bool)

    # Get warp data types for the input tensor precision
    wp_dtype = get_wp_dtype(current_positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(current_positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(current_positions.dtype)
    wp_device = str(device)

    # Convert PyTorch tensors to warp arrays
    wp_current_positions = wp.from_torch(
        current_positions, dtype=wp_vec_dtype, return_ctype=True
    )
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )
    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.int32, return_ctype=True
    )

    # Initialize rebuild flag (False = no rebuild needed)
    rebuild_needed = torch.tensor([False], device=device, dtype=torch.bool)
    wp_rebuild_flag = wp.from_torch(rebuild_needed, dtype=wp.bool, return_ctype=True)

    # Call core warp launcher
    check_cell_list_rebuild(
        current_positions=wp_current_positions,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        cells_per_dimension=wp_cells_per_dimension,
        cell=wp_cell,
        pbc=wp_pbc,
        rebuild_flag=wp_rebuild_flag,
        wp_dtype=wp_dtype,
        device=wp_device,
    )

    return rebuild_needed


@_cell_list_needs_rebuild.register_fake
def _cell_list_needs_rebuild_fake(
    current_positions: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation for torch.compile compatibility.

    Returns a conservative default (no rebuild needed) for compilation tracing.
    The actual implementation will be called during runtime execution.
    """
    return torch.tensor([False], device=current_positions.device, dtype=torch.bool)


def cell_list_needs_rebuild(
    current_positions: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Detect if spatial cell list requires rebuilding due to atomic motion.

    This torch.compile-compatible custom operator efficiently determines if any atoms
    have moved between spatial cells since the last cell list construction. Uses GPU
    acceleration with early termination for optimal performance.

    Parameters
    ----------
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates in Cartesian space.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from the existing cell list.
        Typically obtained from build_cell_list.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of spatial cells in x, y, z directions.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix for coordinate transformations.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Periodic boundary condition flags for x, y, z directions.

    Returns
    -------
    rebuild_needed : torch.Tensor, shape (1,), dtype=bool
        True if any atom has moved to a different cell requiring rebuild.

    Notes
    -----
    - Currently only supports single system.
    - torch.compile compatible custom operation
    - Uses GPU kernels for parallel cell assignment computation
    - Early termination optimization stops computation once rebuild is detected
    - Handles periodic boundary conditions correctly
    - Returns tensor (not Python bool) for compilation compatibility

    See Also
    --------
    nvalchemiops.neighborlist.rebuild_detection.wp_check_cell_list_rebuild : Core warp launcher
    check_cell_list_rebuild_needed : Convenience wrapper that returns Python bool
    """
    return _cell_list_needs_rebuild(
        current_positions,
        atom_to_cell_mapping,
        cells_per_dimension,
        cell,
        pbc,
    )


###########################################################################################
########################### Neighbor List Rebuild Detection ##############################
###########################################################################################


@torch.library.custom_op("nvalchemiops::_neighbor_list_needs_rebuild", mutates_args=())
def _neighbor_list_needs_rebuild(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    skin_distance_threshold: float,
) -> torch.Tensor:
    """Detect if neighbor list requires rebuilding due to excessive atomic motion.

    Parameters
    ----------
    reference_positions : torch.Tensor, shape (total_atoms, 3)
        Atomic positions when the neighbor list was last built.
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic positions to compare against reference.
    skin_distance_threshold : float
        Maximum allowed displacement before neighbor list becomes invalid.

    Returns
    -------
    rebuild_needed : torch.Tensor, shape (1,), dtype=bool
        True if any atom has moved beyond skin distance.

    See Also
    --------
    nvalchemiops.neighborlist.rebuild_detection.wp_check_neighbor_list_rebuild : Core warp launcher
    neighbor_list_needs_rebuild : High-level wrapper function
    """
    # Check for shape compatibility
    if reference_positions.shape != current_positions.shape:
        return torch.tensor([True], device=current_positions.device, dtype=torch.bool)

    total_atoms = reference_positions.shape[0]
    device = reference_positions.device

    if total_atoms == 0:
        return torch.tensor([False], device=device, dtype=torch.bool)

    # Get warp data types for the input tensor precision
    wp_dtype = get_wp_dtype(reference_positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(reference_positions.dtype)
    wp_device = str(device)

    # Convert PyTorch tensors to warp arrays
    wp_reference_positions = wp.from_torch(
        reference_positions, dtype=wp_vec_dtype, return_ctype=True
    )
    wp_current_positions = wp.from_torch(
        current_positions, dtype=wp_vec_dtype, return_ctype=True
    )

    # Initialize rebuild flag (False = no rebuild needed)
    rebuild_needed = torch.tensor([False], device=device, dtype=torch.bool)
    wp_rebuild_flag = wp.from_torch(rebuild_needed, dtype=wp.bool, return_ctype=True)

    # Call core warp launcher
    check_neighbor_list_rebuild(
        reference_positions=wp_reference_positions,
        current_positions=wp_current_positions,
        skin_distance_threshold=skin_distance_threshold,
        rebuild_flag=wp_rebuild_flag,
        wp_dtype=wp_dtype,
        device=wp_device,
    )

    return rebuild_needed


@_neighbor_list_needs_rebuild.register_fake
def _neighbor_list_needs_rebuild_fake(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    skin_distance_threshold: float,
) -> torch.Tensor:
    """Fake implementation for torch.compile compatibility.

    Returns a conservative default (no rebuild needed) for compilation tracing.
    The actual implementation will be called during runtime execution.
    """
    return torch.tensor([False], device=current_positions.device, dtype=torch.bool)


def neighbor_list_needs_rebuild(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    skin_distance_threshold: float,
) -> torch.Tensor:
    """Detect if neighbor list requires rebuilding due to excessive atomic motion.

    This torch.compile-compatible custom operator efficiently determines if any atoms
    have moved beyond the skin distance since the neighbor list was last built. Uses
    GPU acceleration with early termination for optimal performance in MD simulations.

    The skin distance approach allows neighbor lists to remain valid even when atoms
    move slightly, reducing the frequency of expensive neighbor list reconstructions.

    Parameters
    ----------
    reference_positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates when the neighbor list was last constructed.
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates to compare against reference.
    skin_distance_threshold : float
        Maximum allowed atomic displacement before neighbor list becomes invalid.
        Typically set to (cutoff_radius - cutoff) / 2 for safety.

    Returns
    -------
    rebuild_needed : torch.Tensor, shape (1,), dtype=bool
        True if any atom has moved beyond skin distance requiring rebuild.

    Notes
    -----
    - Currently only supports single system.
    - torch.compile compatible custom operation
    - Uses GPU kernels for parallel displacement computation
    - Early termination optimization stops computation once rebuild is detected
    - Displacement calculation uses Euclidean distance
    - Returns tensor (not Python bool) for compilation compatibility

    See Also
    --------
    nvalchemiops.neighborlist.rebuild_detection.wp_check_neighbor_list_rebuild : Core warp launcher
    check_neighbor_list_rebuild_needed : Convenience wrapper that returns Python bool
    """
    return _neighbor_list_needs_rebuild(
        reference_positions, current_positions, skin_distance_threshold
    )


###########################################################################################
########################### High-level API Functions ######################################
###########################################################################################


def check_cell_list_rebuild_needed(
    current_positions: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> bool:
    """Determine if spatial cell list requires rebuilding based on atomic motion.

    This high-level convenience function determines if a spatial cell list needs to be
    reconstructed due to atomic movement. It uses GPU acceleration to efficiently detect
    when atoms have moved between spatial cells.

    The function checks if any atoms have moved to different spatial cells since the
    cell list was last built by comparing current positions against the stored cell
    assignments from the existing cell list.

    This function is not torch.compile compatible (use cell_list_needs_rebuild for that).

    Parameters
    ----------
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates to check against existing cell assignments.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates assigned to each atom from existing cell list.
        This is the key tensor used for comparison with current positions.
        Typically obtained from build_cell_list.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of spatial cells in x, y, z directions from existing cell list.
    cell : torch.Tensor, shape (1, 3, 3)
        Current unit cell matrix for coordinate transformations.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Current periodic boundary condition flags for x, y, z directions.

    Returns
    -------
    needs_rebuild : bool
        True if any atom has moved to a different cell requiring cell list rebuild.

    Notes
    -----
    - Currently only supports single system.
    - Uses GPU kernels for efficient parallel computation
    - Primary check: atomic motion between spatial cells
    - Early termination optimization for performance
    - Returns Python bool (calls .item() on tensor result)

    See Also
    --------
    cell_list_needs_rebuild : Returns tensor instead of bool (torch.compile compatible)
    nvalchemiops.neighborlist.rebuild_detection.wp_check_cell_list_rebuild : Core warp launcher
    """
    rebuild_tensor = cell_list_needs_rebuild(
        current_positions,
        atom_to_cell_mapping,
        cells_per_dimension,
        cell,
        pbc,
    )

    return rebuild_tensor.item()


def check_neighbor_list_rebuild_needed(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    skin_distance_threshold: float,
) -> bool:
    """Determine if neighbor list requires rebuilding based on atomic motion.

    This high-level function provides a convenient interface to check if a neighbor
    list needs reconstruction due to excessive atomic movement. Uses the skin distance
    approach to minimize unnecessary neighbor list rebuilds during MD simulations.

    The skin distance technique allows atoms to move slightly without invalidating
    the neighbor list, reducing computational overhead. When any atom moves beyond
    the skin distance, the neighbor list must be rebuilt to maintain accuracy.

    This function is not torch.compile compatible.

    Parameters
    ----------
    reference_positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates when the neighbor list was last constructed.
        Used as the reference point for displacement calculations.
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates to compare against reference positions.
        Must have the same shape as reference_positions.
    skin_distance_threshold : float
        Maximum allowed atomic displacement before neighbor list becomes invalid.
        Typically set to (cutoff_radius - cutoff) / 2 for safety.
        Units should match the coordinate system.

    Returns
    -------
    needs_rebuild : bool
        True if any atom has moved beyond skin distance requiring neighbor list rebuild.

    Notes
    -----
    - Currently only supports single system.
    - Uses GPU acceleration for efficient displacement computation
    - Early termination optimization for performance
    - Essential for efficient molecular dynamics simulations

    See Also
    --------
    neighbor_list_needs_rebuild : Returns tensor instead of bool
    nvalchemiops.neighborlist.rebuild_detection.wp_check_neighbor_list_rebuild : Core warp launcher
    """
    rebuild_tensor = neighbor_list_needs_rebuild(
        reference_positions, current_positions, skin_distance_threshold
    )

    return rebuild_tensor.item()


###########################################################################################
########################### Batch Neighbor List Rebuild Detection ########################
###########################################################################################


@torch.library.custom_op(
    "nvalchemiops::_batch_neighbor_list_needs_rebuild", mutates_args=()
)
def _batch_neighbor_list_needs_rebuild(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    batch_idx: torch.Tensor,
    skin_distance_threshold: float,
) -> torch.Tensor:
    """Detect per-system if neighbor lists require rebuilding due to atomic motion.

    Parameters
    ----------
    reference_positions : torch.Tensor, shape (total_atoms, 3)
        Atomic positions when each system's neighbor list was last built.
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic positions to compare against reference.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        System index for each atom.
    skin_distance_threshold : float
        Maximum allowed displacement before neighbor list becomes invalid.

    Returns
    -------
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=bool
        Per-system flags: True if any atom in that system moved beyond skin distance.

    See Also
    --------
    batch_neighbor_list_needs_rebuild : High-level wrapper function
    """
    if reference_positions.shape != current_positions.shape:
        num_systems = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1
        return torch.ones(
            num_systems, device=current_positions.device, dtype=torch.bool
        )

    total_atoms = reference_positions.shape[0]
    device = reference_positions.device

    num_systems = int(batch_idx.max().item()) + 1 if total_atoms > 0 else 1
    rebuild_flags = torch.zeros(num_systems, device=device, dtype=torch.bool)

    if total_atoms == 0:
        return rebuild_flags

    wp_dtype = get_wp_dtype(reference_positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(reference_positions.dtype)
    wp_device = str(device)

    wp_reference = wp.from_torch(
        reference_positions, dtype=wp_vec_dtype, return_ctype=True
    )
    wp_current = wp.from_torch(current_positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_batch_idx = wp.from_torch(
        batch_idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
    )
    wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

    check_batch_neighbor_list_rebuild(
        reference_positions=wp_reference,
        current_positions=wp_current,
        batch_idx=wp_batch_idx,
        skin_distance_threshold=skin_distance_threshold,
        rebuild_flags=wp_rebuild_flags,
        wp_dtype=wp_dtype,
        device=wp_device,
    )

    return rebuild_flags


@_batch_neighbor_list_needs_rebuild.register_fake
def _batch_neighbor_list_needs_rebuild_fake(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    batch_idx: torch.Tensor,
    skin_distance_threshold: float,
) -> torch.Tensor:
    """Fake implementation for torch.compile compatibility."""
    num_systems = batch_idx.max() + 1 if batch_idx.numel() > 0 else 1
    return torch.zeros(num_systems, device=reference_positions.device, dtype=torch.bool)


def batch_neighbor_list_needs_rebuild(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    batch_idx: torch.Tensor,
    skin_distance_threshold: float,
) -> torch.Tensor:
    """Detect per-system if neighbor lists require rebuilding due to atomic motion.

    This torch.compile-compatible custom operator efficiently determines which systems
    in a batch need their neighbor list rebuilt based on atomic displacements. Uses
    GPU-side flagging with no CPU-GPU synchronization.

    Parameters
    ----------
    reference_positions : torch.Tensor, shape (total_atoms, 3)
        Atomic positions when each system's neighbor list was last built.
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates to compare against reference.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        System index for each atom.
    skin_distance_threshold : float
        Maximum allowed atomic displacement before neighbor list becomes invalid.
        Typically set to (cutoff_radius - cutoff) / 2 for safety.

    Returns
    -------
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=bool
        Per-system flags: True if any atom in that system moved beyond the skin distance.
        Can be passed directly to batch neighbor list functions for GPU-side selective rebuild.

    Notes
    -----
    - torch.compile compatible custom operation
    - No CPU-GPU synchronization required; all flag writes happen on GPU
    - ``num_systems`` is inferred as ``batch_idx.max() + 1``
    - Returns tensor (not Python bool) for compilation compatibility

    See Also
    --------
    neighbor_list_needs_rebuild : Single-system version
    check_batch_neighbor_list_rebuild_needed : Convenience wrapper returning Python list
    """
    return _batch_neighbor_list_needs_rebuild(
        reference_positions, current_positions, batch_idx, skin_distance_threshold
    )


###########################################################################################
########################### Batch Cell List Rebuild Detection ############################
###########################################################################################


@torch.library.custom_op(
    "nvalchemiops::_batch_cell_list_needs_rebuild", mutates_args=()
)
def _batch_cell_list_needs_rebuild(
    current_positions: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Detect per-system if spatial cell lists require rebuilding due to atomic motion.

    Parameters
    ----------
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates in Cartesian space.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from the existing cell lists.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        System index for each atom.
    cells_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32
        Number of spatial cells in x, y, z directions for each system.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Per-system unit cell matrices for coordinate transformations.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Per-system periodic boundary condition flags.

    Returns
    -------
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=bool
        Per-system flags: True if any atom in that system changed cells.

    See Also
    --------
    batch_cell_list_needs_rebuild : High-level wrapper function
    """
    total_atoms = current_positions.shape[0]
    num_systems = cell.shape[0]
    device = current_positions.device

    rebuild_flags = torch.zeros(num_systems, device=device, dtype=torch.bool)

    if total_atoms == 0:
        return rebuild_flags

    wp_dtype = get_wp_dtype(current_positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(current_positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(current_positions.dtype)
    wp_device = str(device)

    wp_current = wp.from_torch(current_positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )
    wp_batch_idx = wp.from_torch(
        batch_idx.to(dtype=torch.int32), dtype=wp.int32, return_ctype=True
    )
    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    # pbc shape (num_systems, 3) → 2D warp array of bool
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

    check_batch_cell_list_rebuild(
        current_positions=wp_current,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        batch_idx=wp_batch_idx,
        cells_per_dimension=wp_cells_per_dimension,
        cell=wp_cell,
        pbc=wp_pbc,
        rebuild_flags=wp_rebuild_flags,
        wp_dtype=wp_dtype,
        device=wp_device,
    )

    return rebuild_flags


@_batch_cell_list_needs_rebuild.register_fake
def _batch_cell_list_needs_rebuild_fake(
    current_positions: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation for torch.compile compatibility."""
    num_systems = cell.shape[0]
    return torch.zeros(num_systems, device=current_positions.device, dtype=torch.bool)


def batch_cell_list_needs_rebuild(
    current_positions: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Detect per-system if spatial cell lists require rebuilding due to atomic motion.

    This torch.compile-compatible custom operator efficiently determines which systems
    in a batch need their cell list rebuilt by checking if any atoms have moved between
    spatial cells. Uses GPU-side flagging with no CPU-GPU synchronization.

    Parameters
    ----------
    current_positions : torch.Tensor, shape (total_atoms, 3)
        Current atomic coordinates in Cartesian space.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from the existing cell lists.
        Typically obtained from batch_build_cell_list.
    batch_idx : torch.Tensor, shape (total_atoms,), dtype=int32
        System index for each atom.
    cells_per_dimension : torch.Tensor, shape (num_systems, 3), dtype=int32
        Number of spatial cells in x, y, z directions for each system.
    cell : torch.Tensor, shape (num_systems, 3, 3)
        Per-system unit cell matrices for coordinate transformations.
    pbc : torch.Tensor, shape (num_systems, 3), dtype=bool
        Per-system periodic boundary condition flags.

    Returns
    -------
    rebuild_flags : torch.Tensor, shape (num_systems,), dtype=bool
        Per-system flags: True if any atom in that system changed cells.

    Notes
    -----
    - torch.compile compatible custom operation
    - No CPU-GPU synchronization required; all flag writes happen on GPU
    - Returns tensor (not Python bool) for compilation compatibility

    See Also
    --------
    cell_list_needs_rebuild : Single-system version
    batch_neighbor_list_needs_rebuild : Skin-distance based alternative
    """
    return _batch_cell_list_needs_rebuild(
        current_positions,
        atom_to_cell_mapping,
        batch_idx,
        cells_per_dimension,
        cell,
        pbc,
    )
