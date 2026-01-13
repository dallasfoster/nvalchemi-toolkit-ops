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
MD Integrators
==============

GPU-accelerated molecular dynamics integrators with both mutating and
non-mutating APIs for gradient tracking compatibility.

Available Integrators
---------------------
velocity_verlet
    Symplectic velocity Verlet integrator for NVE ensemble.
    Time-reversible, second-order accurate.

langevin
    BAOAB Langevin integrator for NVT ensemble.
    Stochastic thermostat with optimal configurational sampling.

nose_hoover
    Nosé-Hoover chain thermostat for NVT ensemble.
    Deterministic, time-reversible extended Lagrangian dynamics.
    Based on Martyna-Tobias-Klein equations with Yoshida-Suzuki integration.

npt
    NPT (isothermal-isobaric) integrator using Nosé-Hoover chains.
    Coupled thermostat and barostat for constant pressure/temperature.
    Based on Martyna-Tobias-Klein equations.

nph
    NPH (isenthalpic-isobaric) integrator without thermostat.
    Constant enthalpy and pressure for adiabatic dynamics.
    Based on Martyna-Tobias-Klein equations.

API Patterns
------------
- Mutating APIs: Modify arrays in-place (e.g., `velocity_verlet_position_update`)
- Non-mutating APIs: Return new arrays (e.g., `velocity_verlet_position_update_out`)
"""

from .langevin import (
    langevin_baoab_finalize,
    langevin_baoab_finalize_out,
    # Mutating
    langevin_baoab_half_step,
    # Non-mutating
    langevin_baoab_half_step_out,
)
from .nose_hoover import (
    nhc_compute_chain_energy,
    # Utilities
    nhc_compute_masses,
    nhc_position_update,
    nhc_position_update_out,
    # Mutating
    nhc_thermostat_chain_update,
    # Non-mutating
    nhc_thermostat_chain_update_out,
    nhc_velocity_half_step,
    nhc_velocity_half_step_out,
)
from .npt import (
    # Barostat utilities
    compute_barostat_mass,
    compute_barostat_potential_energy,
    compute_cell_kinetic_energy,
    # Pressure calculations
    compute_pressure_tensor,
    compute_scalar_pressure,
    # NPH integration - Mutating
    nph_barostat_half_step,  # Unified: auto-dispatches based on target_pressures dtype
    nph_cell_update,
    nph_position_update,
    nph_position_update_out,
    nph_velocity_half_step,  # Unified: mode="isotropic"|"anisotropic"
    # NPH integration - Non-mutating
    nph_velocity_half_step_out,
    npt_barostat_half_step,  # Unified: auto-dispatches based on target_pressures dtype
    npt_cell_update,
    npt_cell_update_out,
    npt_position_update,
    npt_position_update_out,
    # NPT integration - Mutating
    npt_thermostat_half_step,
    npt_velocity_half_step,  # Unified: mode="isotropic"|"anisotropic"
    # NPT integration - Non-mutating
    npt_velocity_half_step_out,
    # High-level NPH
    run_nph_step,
    # High-level NPT
    run_npt_step,
    vec3d,
    vec3f,
    vec9d,
    # Tensor types for pressure/virial
    vec9f,
)
from .velocity_rescaling import (
    # Utility
    compute_rescale_factor,
    # Mutating
    velocity_rescale,
    # Non-mutating
    velocity_rescale_out,
)
from .velocity_verlet import (
    # Mutating
    velocity_verlet_position_update,
    # Non-mutating
    velocity_verlet_position_update_out,
    velocity_verlet_velocity_finalize,
    velocity_verlet_velocity_finalize_out,
)

__all__ = [
    # Velocity Verlet - Mutating
    "velocity_verlet_position_update",
    "velocity_verlet_velocity_finalize",
    # Velocity Verlet - Non-mutating
    "velocity_verlet_position_update_out",
    "velocity_verlet_velocity_finalize_out",
    # Langevin - Mutating
    "langevin_baoab_half_step",
    "langevin_baoab_finalize",
    # Langevin - Non-mutating
    "langevin_baoab_half_step_out",
    "langevin_baoab_finalize_out",
    # Velocity Rescaling - Mutating
    "velocity_rescale",
    # Velocity Rescaling - Non-mutating
    "velocity_rescale_out",
    # Velocity Rescaling - Utility
    "compute_rescale_factor",
    # Nosé-Hoover Chain - Mutating
    "nhc_thermostat_chain_update",
    "nhc_velocity_half_step",
    "nhc_position_update",
    "nhc_compute_chain_energy",
    # Nosé-Hoover Chain - Non-mutating
    "nhc_thermostat_chain_update_out",
    "nhc_velocity_half_step_out",
    "nhc_position_update_out",
    # Nosé-Hoover Chain - Utilities
    "nhc_compute_masses",
    # NPT/NPH - Tensor types
    "vec9f",
    "vec9d",
    "vec3f",
    "vec3d",
    # NPT/NPH - Pressure calculations
    "compute_pressure_tensor",
    "compute_scalar_pressure",
    # NPT/NPH - Barostat utilities
    "compute_barostat_mass",
    "compute_cell_kinetic_energy",
    "compute_barostat_potential_energy",
    # NPT - Mutating (unified with mode/dtype dispatch)
    "npt_thermostat_half_step",
    "npt_barostat_half_step",
    "npt_velocity_half_step",
    "npt_position_update",
    "npt_cell_update",
    # NPT - Non-mutating
    "npt_velocity_half_step_out",
    "npt_position_update_out",
    "npt_cell_update_out",
    # NPT - High-level
    "run_npt_step",
    # NPH - Mutating (unified with mode/dtype dispatch)
    "nph_barostat_half_step",
    "nph_velocity_half_step",
    "nph_position_update",
    "nph_cell_update",
    # NPH - Non-mutating
    "nph_velocity_half_step_out",
    "nph_position_update_out",
    # NPH - High-level
    "run_nph_step",
]
