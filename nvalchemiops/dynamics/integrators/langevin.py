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

REFERENCES
==========

- Leimkuhler & Matthews (2013). J. Chem. Phys. 138, 174102 (BAOAB integrator)
- Leimkuhler & Matthews (2015). Molecular Dynamics (textbook)
"""

from __future__ import annotations

from typing import Any

import warp as wp

__all__ = [
    # Mutating (in-place) APIs
    "langevin_baoab_half_step",
    "langevin_baoab_finalize",
    # Non-mutating (output) APIs
    "langevin_baoab_half_step_out",
    "langevin_baoab_finalize_out",
]


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

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    # B step: v += (dt/2m)*F
    vel_step = vel + half_dt * force * inv_mass

    # A step: r += (dt/2)*v
    pos_step = pos + half_dt * vel_step

    # O step: Ornstein-Uhlenbeck thermostat
    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)
    c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
    c2 = wp.sqrt(c2_sq)

    # Generate Gaussian random numbers using Box-Muller
    rng_state = wp.rand_init(int(random_seed), atom_idx)

    xi = type(vel)(
        type(kT)(wp.randn(rng_state)), 
        type(kT)(wp.randn(rng_state)), 
        type(kT)(wp.randn(rng_state))
    )
    vel_step = c1 * vel_step + c2 * xi

    # A step: r += (dt/2)*v
    pos_step2 = pos_step + half_dt * vel_step

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

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    vel_step = vel + half_dt * force * inv_mass
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
    B: v += (dt/2m)*F
    A: r += (dt/2)*v
    O: v = c1*v + c2*xi (thermostat)
    A: r += (dt/2)*v

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

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    # B step
    vel_step = vel + half_dt * force * inv_mass

    # A step
    pos_step = pos + half_dt * vel_step

    # O step
    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)
    c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
    c2 = wp.sqrt(c2_sq)

    rng_state = wp.rand_init(int(random_seed), atom_idx)

    xi = type(vel)(
        type(kT)(wp.randn(rng_state)), 
        type(kT)(wp.randn(rng_state)), 
        type(kT)(wp.randn(rng_state))
    )
    vel_step = c1 * vel_step + c2 * xi

    # A step
    pos_step2 = pos_step + half_dt * vel_step

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

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    vel_step = vel + half_dt * force * inv_mass
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

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    # B step
    vel_step = vel + half_dt * force * inv_mass

    # A step
    pos_step = pos + half_dt * vel_step

    # O step
    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)
    c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
    c2 = wp.sqrt(c2_sq)

    rng_state = wp.rand_init(int(random_seed), atom_idx)

    xi = type(vel)(
        type(kT)(wp.randn(rng_state)), 
        type(kT)(wp.randn(rng_state)), 
        type(kT)(wp.randn(rng_state))
    )
    vel_step = c1 * vel_step + c2 * xi

    
    # A step
    pos_step2 = pos_step + half_dt * vel_step

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

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    vel_step = vel + half_dt * force * inv_mass
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
    B: v += (dt/2m)*F
    A: r += (dt/2)*v
    O: v = c1*v + c2*xi (thermostat)
    A: r += (dt/2)*v

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

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    # B step
    vel_step = vel + half_dt * force * inv_mass

    # A step
    pos_step = pos + half_dt * vel_step

    # O step
    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)
    c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
    c2 = wp.sqrt(c2_sq)

    rng_state = wp.rand_init(int(random_seed), atom_idx)

    xi = type(vel)(
        type(kT)(wp.randn(rng_state)), 
        type(kT)(wp.randn(rng_state)), 
        type(kT)(wp.randn(rng_state))
    )
    vel_step = c1 * vel_step + c2 * xi

    pos_step2 = pos_step + half_dt * vel_step

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

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    vel_step = vel + half_dt * force * inv_mass
    velocities_out[atom_idx] = vel_step


# ==============================================================================
# Kernel Overloads for Explicit Typing
# ==============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]      # Vector types

# Half-step kernel overloads
_langevin_baoab_half_step_kernel_overload = {}
_batch_langevin_baoab_half_step_kernel_overload = {}
_langevin_baoab_half_step_out_kernel_overload = {}
_batch_langevin_baoab_half_step_out_kernel_overload = {}

# Finalize kernel overloads
_langevin_baoab_finalize_kernel_overload = {}
_batch_langevin_baoab_finalize_kernel_overload = {}
_langevin_baoab_finalize_out_kernel_overload = {}
_batch_langevin_baoab_finalize_out_kernel_overload = {}

for t, v in zip(_T, _V):
    # Half-step kernels
    _langevin_baoab_half_step_kernel_overload[v] = wp.overload(
        _langevin_baoab_half_step_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t),
         wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t), wp.uint64],
    )
    _batch_langevin_baoab_half_step_kernel_overload[v] = wp.overload(
        _batch_langevin_baoab_half_step_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t),
         wp.array(dtype=wp.int32), wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t), wp.uint64],
    )
    _langevin_baoab_half_step_out_kernel_overload[v] = wp.overload(
        _langevin_baoab_half_step_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t),
         wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t), wp.uint64,
         wp.array(dtype=v), wp.array(dtype=v)],
    )
    _batch_langevin_baoab_half_step_out_kernel_overload[v] = wp.overload(
        _batch_langevin_baoab_half_step_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t),
         wp.array(dtype=wp.int32), wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t), wp.uint64,
         wp.array(dtype=v), wp.array(dtype=v)],
    )

    # Finalize kernels
    _langevin_baoab_finalize_kernel_overload[v] = wp.overload(
        _langevin_baoab_finalize_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t)],
    )
    _batch_langevin_baoab_finalize_kernel_overload[v] = wp.overload(
        _batch_langevin_baoab_finalize_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )
    _langevin_baoab_finalize_out_kernel_overload[v] = wp.overload(
        _langevin_baoab_finalize_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=v)],
    )
    _batch_langevin_baoab_finalize_out_kernel_overload[v] = wp.overload(
        _batch_langevin_baoab_finalize_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t), wp.array(dtype=v)],
    )


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
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device. If None, inferred from positions.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    vec_dtype = positions.dtype
    if batch_idx is not None:
        wp.launch(
            _batch_langevin_baoab_half_step_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses, batch_idx,
                    dt, temperature, friction, wp.uint64(random_seed)],
            device=device,
        )
    else:
        wp.launch(
            _langevin_baoab_half_step_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses,
                    dt, temperature, friction, wp.uint64(random_seed)],
            device=device,
        )


def langevin_baoab_finalize(
    velocities: wp.array,
    forces_new: wp.array,
    masses: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
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
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device. If None, inferred from velocities.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]

    vec_dtype = velocities.dtype
    if batch_idx is not None:
        wp.launch(
            _batch_langevin_baoab_finalize_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, batch_idx, dt],
            device=device,
        )
    else:
        wp.launch(
            _langevin_baoab_finalize_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, dt],
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
    positions_out: wp.array = None,
    velocities_out: wp.array = None,
    batch_idx: wp.array = None,
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
    positions_out : wp.array, optional
        Output array for new positions. If None, allocated internally.
    velocities_out : wp.array, optional
        Output array for new velocities. If None, allocated internally.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    tuple[wp.array, wp.array]
        (positions_out, velocities_out) - New positions and velocities.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    if positions_out is None:
        positions_out = wp.empty_like(positions)
    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    vec_dtype = positions.dtype
    if batch_idx is not None:
        wp.launch(
            _batch_langevin_baoab_half_step_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses, batch_idx,
                    dt, temperature, friction, wp.uint64(random_seed),
                    positions_out, velocities_out],
            device=device,
        )
    else:
        wp.launch(
            _langevin_baoab_half_step_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses,
                    dt, temperature, friction, wp.uint64(random_seed),
                    positions_out, velocities_out],
            device=device,
        )

    return positions_out, velocities_out


def langevin_baoab_finalize_out(
    velocities: wp.array,
    forces_new: wp.array,
    masses: wp.array,
    dt: wp.array,
    velocities_out: wp.array = None,
    batch_idx: wp.array = None,
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
    velocities_out : wp.array, optional
        Output array for final velocities. If None, allocated internally.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device. If None, inferred from velocities.

    Returns
    -------
    wp.array
        Full-step velocities.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]

    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    vec_dtype = velocities.dtype
    if batch_idx is not None:
        wp.launch(
            _batch_langevin_baoab_finalize_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, batch_idx, dt, velocities_out],
            device=device,
        )
    else:
        wp.launch(
            _langevin_baoab_finalize_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, dt, velocities_out],
            device=device,
        )

    return velocities_out
