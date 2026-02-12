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
Langevin Dynamics Kernels
=========================

GPU-accelerated Warp kernels for Langevin dynamics (NVT ensemble) using
the BAOAB splitting scheme for optimal configurational sampling.

This module provides both mutating (in-place) and non-mutating versions
of each kernel for gradient tracking compatibility.

MATHEMATICAL FORMULATION
========================

Langevin equation of motion:

.. math::

    m\\ddot{\\mathbf{r}} = \\mathbf{F} - \\gamma m \\mathbf{v}
                         + \\sqrt{2 \\gamma m k_B T} \\boldsymbol{\\eta}(t)

BAOAB SPLITTING SCHEME
======================

The BAOAB splitting provides optimal configurational sampling accuracy:

.. math::

    B: \\quad \\mathbf{v} \\leftarrow \\mathbf{v} + \\frac{\\Delta t}{2m}\\mathbf{F}

    A: \\quad \\mathbf{r} \\leftarrow \\mathbf{r} + \\frac{\\Delta t}{2}\\mathbf{v}

    O: \\quad \\mathbf{v} \\leftarrow c_1 \\mathbf{v} + c_2 \\boldsymbol{\\xi}

    A: \\quad \\mathbf{r} \\leftarrow \\mathbf{r} + \\frac{\\Delta t}{2}\\mathbf{v}

    B: \\quad \\mathbf{v} \\leftarrow \\mathbf{v} + \\frac{\\Delta t}{2m}\\mathbf{F}

where:
- :math:`c_1 = e^{-\\gamma \\Delta t}` (velocity damping factor)
- :math:`c_2 = \\sqrt{k_B T (1 - c_1^2)/m}` (noise amplitude)
- :math:`\\boldsymbol{\\xi} \\sim \\mathcal{N}(0, 1)` (standard normal)

BATCH MODE
==========

Supports three execution modes:

**Single System Mode**::

    dt = wp.array([0.001], dtype=wp.float64, device="cuda:0")
    temperature = wp.array([1.0], dtype=wp.float64, device="cuda:0")
    friction = wp.array([1.0], dtype=wp.float64, device="cuda:0")

    langevin_baoab_half_step(
        positions, velocities, forces, masses, dt, temperature, friction,
        random_seed=42
    )

**Batch Mode with batch_idx**::

    # Per-system parameters (different T, gamma, dt for each system)
    batch_idx = wp.array([0]*N0 + [1]*N1 + [2]*N2, dtype=wp.int32, device="cuda:0")
    dt = wp.array([dt0, dt1, dt2], dtype=wp.float64, device="cuda:0")
    temperature = wp.array([T0, T1, T2], dtype=wp.float64, device="cuda:0")
    friction = wp.array([gamma0, gamma1, gamma2], dtype=wp.float64, device="cuda:0")

    langevin_baoab_half_step(
        positions, velocities, forces, masses, dt, temperature, friction,
        random_seed=42, batch_idx=batch_idx
    )

**Batch Mode with atom_ptr**::

    atom_ptr = wp.array([0, N0, N0+N1, N0+N1+N2], dtype=wp.int32, device="cuda:0")
    # Same per-system parameters as batch_idx mode

    langevin_baoab_half_step(
        positions, velocities, forces, masses, dt, temperature, friction,
        random_seed=42, atom_ptr=atom_ptr
    )

REFERENCES
==========

- Leimkuhler & Matthews (2013). J. Chem. Phys. 138, 174102 (BAOAB integrator)
- Leimkuhler & Matthews (2015). Molecular Dynamics (textbook)
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ..utils.launch_helpers import (
    ExecutionMode,
    KernelFamily,
    launch_family,
    register_overloads,
    resolve_execution_mode,
    validate_out_array,
)

__all__ = [
    # Mutating (in-place) APIs
    "langevin_baoab_half_step",
    "langevin_baoab_finalize",
    # Non-mutating (output) APIs
    "langevin_baoab_half_step_out",
    "langevin_baoab_finalize_out",
]

# ------------------------------------------------------------------------------
# Langevin Integrator Functions
# ------------------------------------------------------------------------------


@wp.func
def langevin_noise_amplitude(
    kT: Any,  # wp.float32 or wp.float64
    c1: Any,  # wp.float32 or wp.float64
    mass: Any,  # wp.float32 or wp.float64
) -> Any:
    """Compute Langevin thermostat noise amplitude coefficient.

    Calculates the amplitude c₂ for Gaussian random noise in the BAOAB
    or OVRVO Langevin integrators. This ensures the Ornstein-Uhlenbeck
    process generates the correct equilibrium distribution.

    The formula is: c₂ = √(kT(1 - c₁²)/m)

    where c₁ = exp(-γΔt) is the velocity decay factor.

    Parameters
    ----------
    kT : wp.float32 or wp.float64
        Temperature in energy units (Boltzmann constant × temperature).
        Units must be consistent with force units.
    c1 : wp.float32 or wp.float64
        Velocity decay factor c₁ = exp(-γΔt), where γ is friction
        and Δt is timestep.
    mass : wp.float32 or wp.float64
        Particle mass.

    Returns
    -------
    c2 : same type as kT
        Noise amplitude coefficient.

    Examples
    --------
    In a Langevin BAOAB kernel (O step)::

        gamma = friction[system_id]
        dt_val = dt[system_id]
        kT_val = temperature[system_id]
        mass_val = masses[atom_idx]

        # Compute coefficients
        gamma_dt = gamma * dt_val
        c1 = wp.exp(-gamma_dt)
        c2 = langevin_noise_amplitude(kT_val, c1, mass_val)

        # Apply Ornstein-Uhlenbeck update
        rng = wp.rand_init(seed, atom_idx)
        noise_x = wp.randn(rng)
        noise_y = wp.randn(rng)
        noise_z = wp.randn(rng)

        vel_new = c1 * vel + c2 * wp.vec3(noise_x, noise_y, noise_z)

    Notes
    -----
    - **Statistical Mechanics**: The c₂ term ensures detailed balance
      and canonical sampling at temperature T
    - **Variance**: The noise has variance kT(1-c₁²)/m per component
    - **Limits**:
      - γΔt → 0: c₁ → 1, c₂ → √(γΔt·kT/m) (weak coupling)
      - γΔt → ∞: c₁ → 0, c₂ → √(kT/m) (strong coupling)

    Physical Interpretation:
        The (1 - c₁²) factor represents the variance reduction from
        exponential decay. The noise must compensate to maintain
        thermal equilibrium at temperature T.

    See Also
    --------
    velocity_half_step_from_acceleration : For B and A steps in BAOAB
    position_update_from_velocity : For A steps in BAOAB
    """
    inv_mass = type(mass)(1.0) / mass
    one = type(kT)(1.0)

    # Variance of noise: kT(1 - c1²)/m
    c2_squared = kT * (one - c1 * c1) * inv_mass

    # Amplitude (standard deviation)
    return wp.sqrt(c2_squared)


# ==============================================================================
# Mutating Kernels (in-place updates)
# ==============================================================================


@wp.kernel
def _langevin_baoab_half_step_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    friction: wp.array(dtype=Any),
    random_seed: wp.uint64,
):
    """BAOAB Langevin half-step: B-A-O-A sequence (in-place).

    Performs the first four operations of BAOAB:
    B: v += (dt/2m)*F
    A: r += (dt/2)*v
    O: v = c1*v + c2*xi (thermostat)
    A: r += (dt/2)*v

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,). MODIFIED in-place.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Temperature (kT). Shape (1,) for single, (B,) for batched.
    friction : wp.array(dtype=wp.float32 or wp.float64)
        Friction coefficient. Shape (1,) for single, (B,) for batched.
    random_seed : int
        Random seed for stochastic forces.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    dt_val = dt[0]
    kT = temperature[0]
    gamma = friction[0]

    half_dt = type(dt_val)(0.5) * dt_val

    # B step: v += (dt/2m)*F
    acc = compute_acceleration_from_force(force, mass)
    vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)

    # A step: r += (dt/2)*v
    pos_step = position_update_from_velocity(pos, vel_step, half_dt)

    # O step: Ornstein-Uhlenbeck thermostat
    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)
    inv_mass = type(mass)(1.0) / mass
    c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
    c2 = wp.sqrt(c2_sq)

    # Generate Gaussian random numbers using Box-Muller
    rng_state = wp.rand_init(int(random_seed), atom_idx)

    xi = type(vel)(
        type(kT)(wp.randn(rng_state)),
        type(kT)(wp.randn(rng_state)),
        type(kT)(wp.randn(rng_state)),
    )
    vel_step = c1 * vel_step + c2 * xi

    # A step: r += (dt/2)*v
    pos_step2 = position_update_from_velocity(pos_step, vel_step, half_dt)

    positions[atom_idx] = pos_step2
    velocities[atom_idx] = vel_step


@wp.kernel
def _langevin_baoab_finalize_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
):
    """BAOAB Langevin final B step (in-place).

    Completes the BAOAB sequence with the final velocity half-step.

    Performs the final velocity half-step:
    B: v += (dt/2m)*F

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    random_seed : int
        Random seed for stochastic forces.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]
    force = forces_new[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[0]

    acc = compute_acceleration_from_force(force, mass)
    vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)
    velocities[atom_idx] = vel_step


# ==============================================================================
# Non-Mutating Kernels (write to output arrays)
# ==============================================================================


@wp.kernel
def _langevin_baoab_half_step_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    friction: wp.array(dtype=Any),
    random_seed: wp.uint64,
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """BAOAB Langevin half-step: B-A-O-A sequence (non-mutating).

    Performs the first four operations of BAOAB:

    - **B**: :math:`\\mathbf{v} \\leftarrow \\mathbf{v} + \\frac{\\Delta t}{2m}\\mathbf{F}`
    - **A**: :math:`\\mathbf{r} \\leftarrow \\mathbf{r} + \\frac{\\Delta t}{2}\\mathbf{v}`
    - **O**: :math:`\\mathbf{v} \\leftarrow c_1 \\mathbf{v} + c_2 \\boldsymbol{\\xi}` (thermostat)
    - **A**: :math:`\\mathbf{r} \\leftarrow \\mathbf{r} + \\frac{\\Delta t}{2}\\mathbf{v}`

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Temperature (kT). Shape (1,) for single, (B,) for batched.
    friction : wp.array(dtype=wp.float32 or wp.float64)
        Friction coefficient. Shape (1,) for single, (B,) for batched.
    random_seed : int
        Random seed for stochastic forces.
    positions_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output atomic positions. Shape (N,).
    velocities_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output atomic velocities. Shape (N,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    dt_val = dt[0]
    kT = temperature[0]
    gamma = friction[0]

    half_dt = type(dt_val)(0.5) * dt_val

    # B step
    acc = compute_acceleration_from_force(force, mass)
    vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)

    # A step
    pos_step = position_update_from_velocity(pos, vel_step, half_dt)

    # O step
    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)
    inv_mass = type(mass)(1.0) / mass
    c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
    c2 = wp.sqrt(c2_sq)

    rng_state = wp.rand_init(int(random_seed), atom_idx)

    xi = type(vel)(
        type(kT)(wp.randn(rng_state)),
        type(kT)(wp.randn(rng_state)),
        type(kT)(wp.randn(rng_state)),
    )
    vel_step = c1 * vel_step + c2 * xi

    # A step
    pos_step2 = position_update_from_velocity(pos_step, vel_step, half_dt)

    positions_out[atom_idx] = pos_step2
    velocities_out[atom_idx] = vel_step


@wp.kernel
def _langevin_baoab_finalize_out_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """BAOAB Langevin final B step (non-mutating).

    Performs the final velocity half-step:
    B: v += (dt/2m)*F

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    velocities_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output atomic velocities. Shape (N,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]
    force = forces_new[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[0]

    acc = compute_acceleration_from_force(force, mass)
    vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)
    velocities_out[atom_idx] = vel_step


# ==============================================================================
# Batched Mutating Kernels
# ==============================================================================


@wp.kernel
def _batch_langevin_baoab_half_step_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    friction: wp.array(dtype=Any),
    random_seed: wp.uint64,
):
    """BAOAB Langevin half-step for batched systems (in-place).

    Performs the first four operations of BAOAB:

    - **B**: :math:`\\mathbf{v} \\leftarrow \\mathbf{v} + \\frac{\\Delta t}{2m}\\mathbf{F}`
    - **A**: :math:`\\mathbf{r} \\leftarrow \\mathbf{r} + \\frac{\\Delta t}{2}\\mathbf{v}`
    - **O**: :math:`\\mathbf{v} \\leftarrow c_1 \\mathbf{v} + c_2 \\boldsymbol{\\xi}` (thermostat)
    - **A**: :math:`\\mathbf{r} \\leftarrow \\mathbf{r} + \\frac{\\Delta t}{2}\\mathbf{v}`

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,). MODIFIED in-place.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    batch_idx : wp.array(dtype=wp.int32)
        System index for each atom. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Temperature (kT). Shape (1,) for single, (B,) for batched.
    friction : wp.array(dtype=wp.float32 or wp.float64)
        Friction coefficient. Shape (1,) for single, (B,) for batched.
    random_seed : int
        Random seed for stochastic forces.

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    dt_val = dt[system_id]
    kT = temperature[system_id]
    gamma = friction[system_id]

    half_dt = type(dt_val)(0.5) * dt_val

    # B step
    acc = compute_acceleration_from_force(force, mass)
    vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)

    # A step
    pos_step = position_update_from_velocity(pos, vel_step, half_dt)

    # O step
    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)
    inv_mass = type(mass)(1.0) / mass
    c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
    c2 = wp.sqrt(c2_sq)

    rng_state = wp.rand_init(int(random_seed), atom_idx)

    xi = type(vel)(
        type(kT)(wp.randn(rng_state)),
        type(kT)(wp.randn(rng_state)),
        type(kT)(wp.randn(rng_state)),
    )
    vel_step = c1 * vel_step + c2 * xi

    # A step
    pos_step2 = position_update_from_velocity(pos_step, vel_step, half_dt)

    positions[atom_idx] = pos_step2
    velocities[atom_idx] = vel_step


@wp.kernel
def _batch_langevin_baoab_finalize_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """BAOAB Langevin final B step for batched systems (in-place).

    Performs the final velocity half-step:
    B: v += (dt/2m)*F

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    batch_idx : wp.array(dtype=wp.int32)
        System index for each atom. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    vel = velocities[atom_idx]
    force = forces_new[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[system_id]

    acc = compute_acceleration_from_force(force, mass)
    vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)
    velocities[atom_idx] = vel_step


# ==============================================================================
# Batched Non-Mutating Kernels
# ==============================================================================


@wp.kernel
def _batch_langevin_baoab_half_step_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    friction: wp.array(dtype=Any),
    random_seed: wp.uint64,
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """BAOAB Langevin half-step for batched systems (non-mutating).

    Performs the first four operations of BAOAB:

    - **B**: :math:`\\mathbf{v} \\leftarrow \\mathbf{v} + \\frac{\\Delta t}{2m}\\mathbf{F}`
    - **A**: :math:`\\mathbf{r} \\leftarrow \\mathbf{r} + \\frac{\\Delta t}{2}\\mathbf{v}`
    - **O**: :math:`\\mathbf{v} \\leftarrow c_1 \\mathbf{v} + c_2 \\boldsymbol{\\xi}` (thermostat)
    - **A**: :math:`\\mathbf{r} \\leftarrow \\mathbf{r} + \\frac{\\Delta t}{2}\\mathbf{v}`

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    batch_idx : wp.array(dtype=wp.int32)
        System index for each atom. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Temperature (kT). Shape (1,) for single, (B,) for batched.
    friction : wp.array(dtype=wp.float32 or wp.float64)
        Friction coefficient. Shape (1,) for single, (B,) for batched.
    random_seed : int
        Random seed for stochastic forces.
    positions_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output atomic positions. Shape (N,).
    velocities_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output atomic velocities. Shape (N,).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    dt_val = dt[system_id]
    kT = temperature[system_id]
    gamma = friction[system_id]

    half_dt = type(dt_val)(0.5) * dt_val

    # B step
    acc = compute_acceleration_from_force(force, mass)
    vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)

    # A step
    pos_step = position_update_from_velocity(pos, vel_step, half_dt)

    # O step
    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)
    inv_mass = type(mass)(1.0) / mass
    c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
    c2 = wp.sqrt(c2_sq)

    rng_state = wp.rand_init(int(random_seed), atom_idx)

    xi = type(vel)(
        type(kT)(wp.randn(rng_state)),
        type(kT)(wp.randn(rng_state)),
        type(kT)(wp.randn(rng_state)),
    )
    vel_step = c1 * vel_step + c2 * xi

    pos_step2 = position_update_from_velocity(pos_step, vel_step, half_dt)

    positions_out[atom_idx] = pos_step2
    velocities_out[atom_idx] = vel_step


@wp.kernel
def _batch_langevin_baoab_finalize_out_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """BAOAB Langevin final B step for batched systems (non-mutating).

    Performs the final velocity half-step:
    B: v += (dt/2m)*F

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    velocities_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output atomic velocities. Shape (N,).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    vel = velocities[atom_idx]
    force = forces_new[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[system_id]

    acc = compute_acceleration_from_force(force, mass)
    vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)
    velocities_out[atom_idx] = vel_step


# ==============================================================================
# Pointer-Based (CSR) Mutating Kernels
# ==============================================================================


@wp.kernel
def _langevin_baoab_half_step_ptr_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    friction: wp.array(dtype=Any),
    random_seed: wp.uint64,
):
    """BAOAB Langevin half-step using atom_ptr (in-place).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (num_atoms_total,). MODIFIED in-place.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (num_atoms_total,). MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (num_atoms_total,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (num_atoms_total,).
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep per system. Shape (num_systems,).
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Temperature (kT) per system. Shape (num_systems,).
    friction : wp.array(dtype=wp.float32 or wp.float64)
        Friction coefficient per system. Shape (num_systems,).
    random_seed : int
        Random seed for stochastic forces.

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0, a1 = atom_ptr[sys_id], atom_ptr[sys_id + 1]

    dt_val = dt[sys_id]
    kT = temperature[sys_id]
    gamma = friction[sys_id]

    half_dt = type(dt_val)(0.5) * dt_val
    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)

    for i in range(a0, a1):
        pos = positions[i]
        vel = velocities[i]
        force = forces[i]
        mass = masses[i]

        # B step: v += (dt/2m)*F
        acc = compute_acceleration_from_force(force, mass)
        vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)

        # A step: r += (dt/2)*v
        pos_step = position_update_from_velocity(pos, vel_step, half_dt)

        # O step: Ornstein-Uhlenbeck thermostat
        inv_mass = type(mass)(1.0) / mass
        c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
        c2 = wp.sqrt(c2_sq)

        # Generate Gaussian random numbers
        rng_state = wp.rand_init(int(random_seed), i)
        xi = type(vel)(
            type(kT)(wp.randn(rng_state)),
            type(kT)(wp.randn(rng_state)),
            type(kT)(wp.randn(rng_state)),
        )
        vel_step = c1 * vel_step + c2 * xi

        # A step: r += (dt/2)*v
        pos_step2 = position_update_from_velocity(pos_step, vel_step, half_dt)

        positions[i] = pos_step2
        velocities[i] = vel_step


@wp.kernel
def _langevin_baoab_finalize_ptr_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """BAOAB Langevin final B step using atom_ptr (in-place).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (num_atoms_total,). MODIFIED in-place.
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions. Shape (num_atoms_total,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (num_atoms_total,).
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep per system. Shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0, a1 = atom_ptr[sys_id], atom_ptr[sys_id + 1]
    dt_val = dt[sys_id]

    for i in range(a0, a1):
        vel = velocities[i]
        force = forces_new[i]
        mass = masses[i]

        acc = compute_acceleration_from_force(force, mass)
        vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)
        velocities[i] = vel_step


# ==============================================================================
# Pointer-Based (CSR) Non-Mutating Kernels
# ==============================================================================


@wp.kernel
def _langevin_baoab_half_step_ptr_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    friction: wp.array(dtype=Any),
    random_seed: wp.uint64,
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """BAOAB Langevin half-step using atom_ptr (non-mutating).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (num_atoms_total,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (num_atoms_total,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (num_atoms_total,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (num_atoms_total,).
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep per system. Shape (num_systems,).
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Temperature (kT) per system. Shape (num_systems,).
    friction : wp.array(dtype=wp.float32 or wp.float64)
        Friction coefficient per system. Shape (num_systems,).
    random_seed : int
        Random seed for stochastic forces.
    positions_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output positions. Shape (num_atoms_total,).
    velocities_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output velocities. Shape (num_atoms_total,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0, a1 = atom_ptr[sys_id], atom_ptr[sys_id + 1]

    dt_val = dt[sys_id]
    kT = temperature[sys_id]
    gamma = friction[sys_id]

    half_dt = type(dt_val)(0.5) * dt_val
    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)

    for i in range(a0, a1):
        pos = positions[i]
        vel = velocities[i]
        force = forces[i]
        mass = masses[i]

        # B step
        acc = compute_acceleration_from_force(force, mass)
        vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)

        # A step
        pos_step = position_update_from_velocity(pos, vel_step, half_dt)

        # O step
        inv_mass = type(mass)(1.0) / mass
        c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
        c2 = wp.sqrt(c2_sq)

        rng_state = wp.rand_init(int(random_seed), i)
        xi = type(vel)(
            type(kT)(wp.randn(rng_state)),
            type(kT)(wp.randn(rng_state)),
            type(kT)(wp.randn(rng_state)),
        )
        vel_step = c1 * vel_step + c2 * xi

        # A step
        pos_step2 = position_update_from_velocity(pos_step, vel_step, half_dt)

        positions_out[i] = pos_step2
        velocities_out[i] = vel_step


@wp.kernel
def _langevin_baoab_finalize_ptr_out_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """BAOAB Langevin final B step using atom_ptr (non-mutating).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (num_atoms_total,).
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions. Shape (num_atoms_total,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (num_atoms_total,).
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep per system. Shape (num_systems,).
    velocities_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output velocities. Shape (num_atoms_total,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0, a1 = atom_ptr[sys_id], atom_ptr[sys_id + 1]
    dt_val = dt[sys_id]

    for i in range(a0, a1):
        vel = velocities[i]
        force = forces_new[i]
        mass = masses[i]

        acc = compute_acceleration_from_force(force, mass)
        vel_step = velocity_half_step_from_acceleration(vel, acc, dt_val)
        velocities_out[i] = vel_step


# ==============================================================================
# Kernel Overloads via KernelFamily
# ==============================================================================

# Half-step (inplace) -- keyed by vec_dtype
_half_step_families = {
    v: KernelFamily(
        single=register_overloads(
            _langevin_baoab_half_step_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t),
                           wp.array(dtype=t), wp.uint64],
            dtype_pairs=((v, t),),
        )[v],
        batch_idx=register_overloads(
            _batch_langevin_baoab_half_step_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t),
                           wp.array(dtype=t), wp.array(dtype=t), wp.uint64],
            dtype_pairs=((v, t),),
        )[v],
        atom_ptr=register_overloads(
            _langevin_baoab_half_step_ptr_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t),
                           wp.array(dtype=t), wp.array(dtype=t), wp.uint64],
            dtype_pairs=((v, t),),
        )[v],
    )
    for v, t in ((wp.vec3f, wp.float32), (wp.vec3d, wp.float64))
}

# Half-step (out) -- keyed by vec_dtype
_half_step_out_families = {
    v: KernelFamily(
        single=register_overloads(
            _langevin_baoab_half_step_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t),
                           wp.array(dtype=t), wp.uint64,
                           wp.array(dtype=v_), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
        batch_idx=register_overloads(
            _batch_langevin_baoab_half_step_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t),
                           wp.array(dtype=t), wp.array(dtype=t), wp.uint64,
                           wp.array(dtype=v_), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
        atom_ptr=register_overloads(
            _langevin_baoab_half_step_ptr_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t),
                           wp.array(dtype=t), wp.array(dtype=t), wp.uint64,
                           wp.array(dtype=v_), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
    )
    for v, t in ((wp.vec3f, wp.float32), (wp.vec3d, wp.float64))
}

# Finalize (inplace) -- keyed by vec_dtype
_finalize_families = {
    v: KernelFamily(
        single=register_overloads(
            _langevin_baoab_finalize_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=t)],
            dtype_pairs=((v, t),),
        )[v],
        batch_idx=register_overloads(
            _batch_langevin_baoab_finalize_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
            dtype_pairs=((v, t),),
        )[v],
        atom_ptr=register_overloads(
            _langevin_baoab_finalize_ptr_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
            dtype_pairs=((v, t),),
        )[v],
    )
    for v, t in ((wp.vec3f, wp.float32), (wp.vec3d, wp.float64))
}

# Finalize (out) -- keyed by vec_dtype
_finalize_out_families = {
    v: KernelFamily(
        single=register_overloads(
            _langevin_baoab_finalize_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
        batch_idx=register_overloads(
            _batch_langevin_baoab_finalize_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32),
                           wp.array(dtype=t), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
        atom_ptr=register_overloads(
            _langevin_baoab_finalize_ptr_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32),
                           wp.array(dtype=t), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
    )
    for v, t in ((wp.vec3f, wp.float32), (wp.vec3d, wp.float64))
}


# ==============================================================================
# Functional Interface - Mutating
# ==============================================================================


def langevin_baoab_half_step(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    temperature: wp.array,
    friction: wp.array,
    random_seed: int,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> None:
    """
    Perform BAOAB Langevin half-step (B-A-O-A sequence) in-place.

    This function performs the first four operations of the BAOAB splitting:
    B (velocity), A (position), O (thermostat), A (position).

    After calling this function, recalculate forces at the new positions,
    then call langevin_baoab_finalize() to complete the step.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,). MODIFIED in-place.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Temperature (kT). Shape (1,) for single, (B,) for batched.
    friction : wp.array(dtype=wp.float32 or wp.float64)
        Friction coefficient. Shape (1,) for single, (B,) for batched.
    random_seed : int
        Random seed for stochastic forces.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from positions.

    Example
    -------
    Single system NVT simulation::

        import warp as wp
        import numpy as np

        # Setup
        positions = wp.array(np.random.randn(100, 3), dtype=wp.vec3d, device="cuda:0")
        velocities = wp.array(np.random.randn(100, 3), dtype=wp.vec3d, device="cuda:0")
        forces = wp.array(np.random.randn(100, 3), dtype=wp.vec3d, device="cuda:0")
        masses = wp.array(np.ones(100), dtype=wp.float64, device="cuda:0")

        dt = wp.array([0.001], dtype=wp.float64, device="cuda:0")
        temperature = wp.array([1.0], dtype=wp.float64, device="cuda:0")  # kT in energy units
        friction = wp.array([1.0], dtype=wp.float64, device="cuda:0")

        # BAOAB half-step
        langevin_baoab_half_step(
            positions, velocities, forces, masses, dt, temperature, friction,
            random_seed=42
        )

    Complete BAOAB step::

        for step in range(num_steps):
            # Step 1: BAOAB half-step (B-A-O-A)
            langevin_baoab_half_step(
                positions, velocities, forces, masses, dt, temperature, friction,
                random_seed=step
            )

            # Step 2: Recalculate forces
            forces = compute_forces(positions)

            # Step 3: Final B step
            langevin_baoab_finalize(velocities, forces, masses, dt)

    Batched mode::

        # With batch_idx (3 systems)
        batch_idx = wp.array([0]*30 + [1]*40 + [2]*30, dtype=wp.int32, device="cuda:0")
        dt = wp.array([0.001, 0.002, 0.0015], dtype=wp.float64, device="cuda:0")
        temperature = wp.array([1.0, 1.5, 1.2], dtype=wp.float64, device="cuda:0")
        friction = wp.array([1.0, 1.0, 1.0], dtype=wp.float64, device="cuda:0")

        langevin_baoab_half_step(
            positions, velocities, forces, masses, dt, temperature, friction,
            random_seed=42, batch_idx=batch_idx
        )

    See Also
    --------
    langevin_baoab_finalize : Complete the BAOAB step
    """
    mode = resolve_execution_mode(batch_idx, atom_ptr)

    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]
    vec_dtype = positions.dtype
    family = _half_step_families[vec_dtype]
    seed = wp.uint64(random_seed)

    if mode is ExecutionMode.ATOM_PTR:
        dim = atom_ptr.shape[0] - 1  # num_systems
    else:
        dim = num_atoms

    launch_family(
        family,
        mode=mode,
        dim=dim,
        inputs_single=[positions, velocities, forces, masses, dt,
                        temperature, friction, seed],
        inputs_batch=[positions, velocities, forces, masses, batch_idx, dt,
                       temperature, friction, seed],
        inputs_ptr=[positions, velocities, forces, masses, atom_ptr, dt,
                     temperature, friction, seed],
        device=device,
    )


def langevin_baoab_finalize(
    velocities: wp.array,
    forces_new: wp.array,
    masses: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> None:
    """
    Finalize BAOAB Langevin step (final B step) in-place.

    Completes the BAOAB sequence with the final velocity half-step update
    using forces calculated at the new positions.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from velocities.
    """
    mode = resolve_execution_mode(batch_idx, atom_ptr)

    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    vec_dtype = velocities.dtype
    family = _finalize_families[vec_dtype]

    if mode is ExecutionMode.ATOM_PTR:
        dim = atom_ptr.shape[0] - 1  # num_systems
    else:
        dim = num_atoms

    launch_family(
        family,
        mode=mode,
        dim=dim,
        inputs_single=[velocities, forces_new, masses, dt],
        inputs_batch=[velocities, forces_new, masses, batch_idx, dt],
        inputs_ptr=[velocities, forces_new, masses, atom_ptr, dt],
        device=device,
    )


# ==============================================================================
# Functional Interface - Non-Mutating
# ==============================================================================


def langevin_baoab_half_step_out(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    temperature: wp.array,
    friction: wp.array,
    random_seed: int,
    positions_out: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> tuple[wp.array, wp.array]:
    """
    Perform BAOAB Langevin half-step (B-A-O-A sequence) non-mutating.

    Writes new positions and velocities to output arrays.
    Input arrays are NOT modified.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions at time t. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities at time t. Shape (N,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms at time t. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array
        Timestep(s). Shape (1,) for single, (B,) for batched.
    temperature : wp.array
        Temperature (kT). Shape (1,) for single, (B,) for batched.
    friction : wp.array
        Friction coefficient. Shape (1,) for single, (B,) for batched.
    random_seed : int
        Random seed for stochastic forces.
    positions_out : wp.array
        Pre-allocated output array for new positions.  Must match
        ``positions`` in shape, dtype, and device.
    velocities_out : wp.array
        Pre-allocated output array for new velocities.  Must match
        ``velocities`` in shape, dtype, and device.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    tuple[wp.array, wp.array]
        (positions_out, velocities_out) - New positions and velocities.
    """
    mode = resolve_execution_mode(batch_idx, atom_ptr)

    validate_out_array(positions_out, positions, "positions_out")
    validate_out_array(velocities_out, velocities, "velocities_out")

    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]
    vec_dtype = positions.dtype
    family = _half_step_out_families[vec_dtype]
    seed = wp.uint64(random_seed)

    if mode is ExecutionMode.ATOM_PTR:
        dim = atom_ptr.shape[0] - 1  # num_systems
    else:
        dim = num_atoms

    launch_family(
        family,
        mode=mode,
        dim=dim,
        inputs_single=[positions, velocities, forces, masses, dt,
                        temperature, friction, seed,
                        positions_out, velocities_out],
        inputs_batch=[positions, velocities, forces, masses, batch_idx, dt,
                       temperature, friction, seed,
                       positions_out, velocities_out],
        inputs_ptr=[positions, velocities, forces, masses, atom_ptr, dt,
                     temperature, friction, seed,
                     positions_out, velocities_out],
        device=device,
    )

    return positions_out, velocities_out


def langevin_baoab_finalize_out(
    velocities: wp.array,
    forces_new: wp.array,
    masses: wp.array,
    dt: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Finalize BAOAB Langevin step (final B step) non-mutating.

    Writes full-step velocities to output array.
    Input arrays are NOT modified.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Velocities after half-step. Shape (N,).
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array
        Timestep(s). Shape (1,) for single, (B,) for batched.
    velocities_out : wp.array
        Pre-allocated output array for final velocities.  Must match
        ``velocities`` in shape, dtype, and device.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from velocities.

    Returns
    -------
    wp.array
        Full-step velocities.
    """
    mode = resolve_execution_mode(batch_idx, atom_ptr)

    validate_out_array(velocities_out, velocities, "velocities_out")

    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    vec_dtype = velocities.dtype
    family = _finalize_out_families[vec_dtype]

    if mode is ExecutionMode.ATOM_PTR:
        dim = atom_ptr.shape[0] - 1  # num_systems
    else:
        dim = num_atoms

    launch_family(
        family,
        mode=mode,
        dim=dim,
        inputs_single=[velocities, forces_new, masses, dt, velocities_out],
        inputs_batch=[velocities, forces_new, masses, batch_idx, dt, velocities_out],
        inputs_ptr=[velocities, forces_new, masses, atom_ptr, dt, velocities_out],
        device=device,
    )

    return velocities_out
