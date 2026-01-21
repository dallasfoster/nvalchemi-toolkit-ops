# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Dynamics Utilities
==================

Utility functions for molecular dynamics simulations with both mutating and
non-mutating APIs for gradient tracking compatibility.

Available Functions
-------------------
Thermostat Utilities
~~~~~~~~~~~~~~~~~~~~
compute_kinetic_energy
    Compute kinetic energy per system.

compute_temperature
    Compute instantaneous temperature from kinetic energy.

initialize_velocities / initialize_velocities_out
    Initialize velocities from Maxwell-Boltzmann distribution.

remove_com_motion / remove_com_motion_out
    Remove center of mass velocity.

Cell Utilities
~~~~~~~~~~~~~~
compute_cell_volume
    Compute cell volume V = |det(cell)|.

compute_cell_inverse
    Compute cell inverse for coordinate transformations.

compute_strain_tensor
    Compute strain tensor from current and reference cells.

apply_strain_to_cell
    Apply strain tensor to cell.

scale_positions_with_cell / scale_positions_with_cell_out
    Scale positions when cell changes, maintaining fractional coordinates.

remap_positions_to_cell / remap_positions_to_cell_out
    Alias for scale_positions_with_cell.

wrap_positions_to_cell / wrap_positions_to_cell_out
    Wrap positions into primary cell.

cartesian_to_fractional
    Convert Cartesian to fractional coordinates.

fractional_to_cartesian
    Convert fractional to Cartesian coordinates.

Cell Filter Utilities (Variable-Cell Optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
align_cell
    Align cell to upper-triangular form for stable optimization.

extend_batch_idx / extend_atom_ptr
    Extend batch_idx/atom_ptr arrays for cell DOFs.

pack_positions_with_cell / unpack_positions_with_cell
    Pack/unpack atomic positions and cell into extended arrays.

pack_velocities_with_cell / unpack_velocities_with_cell
    Pack/unpack atomic velocities and cell velocity into extended arrays.

pack_forces_with_cell
    Pack atomic forces and cell force into extended arrays.

pack_masses_with_cell
    Pack atomic masses and cell mass into extended arrays.

stress_to_cell_force
    Convert stress tensor to cell force for optimization.
"""

from .cell_filter import (
    # Cell alignment
    align_cell,
    # Batch index extension
    extend_atom_ptr,
    extend_batch_idx,
    # Pack utilities
    pack_forces_with_cell,
    pack_masses_with_cell,
    pack_positions_with_cell,
    pack_velocities_with_cell,
    # Stress conversion
    stress_to_cell_force,
    # Unpack utilities
    unpack_positions_with_cell,
    unpack_velocities_with_cell,
)
from .cell_utils import (
    apply_strain_to_cell,
    # Coordinate transformations
    cartesian_to_fractional,
    compute_cell_inverse,
    # Cell properties
    compute_cell_volume,
    # Strain operations
    compute_strain_tensor,
    fractional_to_cartesian,
    # Position operations (mutating)
    scale_positions_with_cell,
    # Position operations (non-mutating)
    scale_positions_with_cell_out,
    wrap_positions_to_cell,
    wrap_positions_to_cell_out,
)
from .constraints import (
    # RATTLE - Mutating
    rattle_constraints,
    # RATTLE - Non-mutating
    rattle_constraints_out,
    rattle_iteration,
    rattle_iteration_out,
    # SHAKE - Mutating
    shake_constraints,
    # SHAKE - Non-mutating
    shake_constraints_out,
    shake_iteration,
    shake_iteration_out,
)
from .thermostat_utils import (
    # Non-mutating (compute only)
    compute_kinetic_energy,
    compute_temperature,
    # Mutating
    initialize_velocities,
    # Non-mutating
    initialize_velocities_out,
    remove_com_motion,
    remove_com_motion_out,
)

__all__ = [
    # Thermostat utilities
    "compute_kinetic_energy",
    "compute_temperature",
    "initialize_velocities",
    "remove_com_motion",
    "initialize_velocities_out",
    "remove_com_motion_out",
    # Cell utilities
    "compute_cell_volume",
    "compute_cell_inverse",
    "compute_strain_tensor",
    "apply_strain_to_cell",
    "scale_positions_with_cell",
    "wrap_positions_to_cell",
    "scale_positions_with_cell_out",
    "wrap_positions_to_cell_out",
    "cartesian_to_fractional",
    "fractional_to_cartesian",
    # Cell filter utilities (variable-cell optimization)
    "align_cell",
    "extend_batch_idx",
    "extend_atom_ptr",
    "pack_positions_with_cell",
    "pack_velocities_with_cell",
    "pack_forces_with_cell",
    "pack_masses_with_cell",
    "unpack_positions_with_cell",
    "unpack_velocities_with_cell",
    "stress_to_cell_force",
    # Constraint utilities (SHAKE/RATTLE)
    "shake_constraints",
    "shake_iteration",
    "shake_constraints_out",
    "shake_iteration_out",
    "rattle_constraints",
    "rattle_iteration",
    "rattle_constraints_out",
    "rattle_iteration_out",
]
