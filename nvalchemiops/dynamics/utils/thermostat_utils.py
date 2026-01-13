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
Thermostat Utility Kernels
==========================

GPU-accelerated Warp kernels for temperature-related computations in
molecular dynamics simulations.

This module provides both mutating (in-place) and non-mutating versions
of each kernel for gradient tracking compatibility.

MATHEMATICAL FORMULATION
========================

Kinetic Energy:

.. math::

    KE = \\frac{1}{2} \\sum_i m_i |\\mathbf{v}_i|^2

Temperature (from equipartition theorem):

.. math::

    T = \\frac{2 \\cdot KE}{N_{DOF} \\cdot k_B}

Maxwell-Boltzmann Distribution:

.. math::

    v_i \\sim \\mathcal{N}\\left(0, \\sqrt{\\frac{k_B T}{m_i}}\\right)
"""

from __future__ import annotations

from typing import Any

import warp as wp

__all__ = [
    # Mutating APIs
    "compute_kinetic_energy",
    "compute_temperature",
    "initialize_velocities",
    "remove_com_motion",
    # Non-mutating APIs
    "initialize_velocities_out",
    "remove_com_motion_out",
]


# ==============================================================================
# Kinetic Energy Kernels
# ==============================================================================


@wp.kernel
def _compute_kinetic_energy_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    kinetic_energy: wp.array(dtype=Any),
):
    """Compute kinetic energy contribution from each atom.

    Accumulates KE = 0.5 * sum_i(m_i * v_i · v_i) using atomic adds.

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    kinetic_energy : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Pre-allocated output array. 
        Shape (1,) for single system

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    v_sq = wp.dot(vel, vel)
    ke_contribution = type(mass)(0.5) * mass * v_sq

    wp.atomic_add(kinetic_energy, 0, ke_contribution)


@wp.kernel
def _batch_compute_kinetic_energy_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    kinetic_energies: wp.array(dtype=Any),
):
    """Compute per-system kinetic energy for batched systems.

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    batch_idx : wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).
    kinetic_energies : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Pre-allocated output array. 
        Shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    v_sq = wp.dot(vel, vel)
    ke_contribution = type(mass)(0.5) * mass * v_sq

    wp.atomic_add(kinetic_energies, system_id, ke_contribution)


# ==============================================================================
# COM Velocity Kernels
# ==============================================================================


@wp.kernel
def _compute_com_velocity_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
):
    """Compute center of mass momentum and total mass.

    COM velocity is computed after kernel as: v_COM = total_momentum / total_mass

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    total_momentum : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Total momentum. Shape (1,).
    total_mass : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Total mass. Shape (1,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    mom = mass * vel
    wp.atomic_add(total_momentum, 0, mom)
    wp.atomic_add(total_mass, 0, mass)

@wp.kernel
def _batch_compute_com_velocity_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
):
    """Compute center of mass momentum and total mass.

    COM velocity is computed after kernel as: v_COM = total_momentum / total_mass

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    total_momentum : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Total momentum. Shape (1,).
    total_mass : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Total mass. Shape (1,).
    batch_idx : wp.array(dtype=wp.int32), e.g., wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).
    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    vel = velocities[atom_idx]
    mass = masses[atom_idx]

    mom = mass * vel
    wp.atomic_add(total_momentum, system_id, mom)
    wp.atomic_add(total_mass, system_id, mass)


@wp.kernel
def _remove_com_motion_kernel(
    velocities: wp.array(dtype=Any),
    com_velocity: wp.array(dtype=Any),
):
    """Remove center of mass velocity from all atoms (in-place).

    v_i = v_i - v_COM

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity. Shape (1,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]

    velocities[atom_idx] = vel - com_velocity[0]

@wp.kernel
def _batch_remove_com_motion_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    com_velocity: wp.array(dtype=Any),
):
    """Remove center of mass velocity from all atoms (in-place).

    v_i = v_i - v_COM

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    batch_idx : wp.array(dtype=wp.int32), e.g., wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity. Shape (num_systems,).
    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    vel = velocities[atom_idx]

    velocities[atom_idx] = vel - com_velocity[system_id]


@wp.kernel
def _remove_com_motion_out_kernel(
    velocities: wp.array(dtype=Any),
    com_velocity: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Remove center of mass velocity from all atoms (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity. Shape (1,).
    velocities_out : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Output velocities. Shape (num_atoms,).
    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel = velocities[atom_idx]

    velocities_out[atom_idx] = vel - com_velocity[0]

@wp.kernel
def _batch_remove_com_motion_out_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    com_velocity: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any), 
):
    """Remove center of mass velocity from all atoms (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    batch_idx : wp.array(dtype=wp.int32), e.g., wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity. Shape (num_systems,).
    velocities_out : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Output velocities. Shape (num_atoms,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    vel = velocities[atom_idx]

    velocities_out[atom_idx] = vel - com_velocity[system_id]


# ==============================================================================
# Velocity Initialization Kernels
# ==============================================================================


@wp.kernel
def _initialize_velocities_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    random_seed: wp.uint64,
):
    """Initialize velocities from Maxwell-Boltzmann distribution (in-place).

    Each velocity component is drawn from N(0, sqrt(kT/m)).

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    temperature : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Temperature. Shape (1,).
    random_seed : wp.uint64
        Random seed.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    mass = masses[atom_idx]
    kT = temperature[0]

    # Standard deviation: sigma = sqrt(kT/m)
    sigma = wp.sqrt(type(mass)(kT) / mass)

    # Initialize RNG state for this atom
    rng_state = wp.rand_init(int(random_seed), atom_idx)

    # Generate Gaussian-distributed velocities using wp.randn (N(0,1))
    vx = sigma * type(mass)(wp.randn(rng_state))
    vy = sigma * type(mass)(wp.randn(rng_state))
    vz = sigma * type(mass)(wp.randn(rng_state))

    vel = velocities[atom_idx]
    velocities[atom_idx] = type(vel)(vx, vy, vz)

@wp.kernel
def _batch_initialize_velocities_kernel(
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    random_seed: wp.uint64,
    batch_idx: wp.array(dtype=wp.int32),
):
    """Initialize velocities from Maxwell-Boltzmann distribution (in-place, batched).

    Each velocity component is drawn from N(0, sqrt(kT/m)).

    Parameters
    ----------
    velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Atomic velocities. Shape (num_atoms,).
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    temperature : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Temperature. Shape (num_systems,).
    random_seed : wp.uint64
        Random seed.
    batch_idx : wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    mass = masses[atom_idx]
    kT = temperature[system_id]

    # Standard deviation: sigma = sqrt(kT/m)
    sigma = wp.sqrt(type(mass)(kT) / mass)

    # Initialize RNG state for this atom
    rng_state = wp.rand_init(int(random_seed), atom_idx)

    # Generate Gaussian-distributed velocities using wp.randn (N(0,1))
    vx = sigma * type(mass)(wp.randn(rng_state))
    vy = sigma * type(mass)(wp.randn(rng_state))
    vz = sigma * type(mass)(wp.randn(rng_state))

    vel = velocities[atom_idx]
    velocities[atom_idx] = type(vel)(vx, vy, vz)


@wp.kernel
def _initialize_velocities_out_kernel(
    masses: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    random_seed: wp.uint64,
    velocities_out: wp.array(dtype=Any),
):
    """Initialize velocities from Maxwell-Boltzmann distribution (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    mass = masses[atom_idx]
    kT = temperature[0]

    kT_typed = type(mass)(kT)
    sigma = wp.sqrt(kT_typed / mass)

    rng_state = wp.rand_init(int(random_seed), atom_idx)

    # Generate Gaussian-distributed velocities using wp.randn (N(0,1))
    vx = sigma * type(mass)(wp.randn(rng_state))
    vy = sigma * type(mass)(wp.randn(rng_state))
    vz = sigma * type(mass)(wp.randn(rng_state))

    vel_sample = velocities_out[atom_idx]
    velocities_out[atom_idx] = type(vel_sample)(vx, vy, vz)


@wp.kernel
def _batch_initialize_velocities_out_kernel(
    masses: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    random_seed: wp.uint64,
    velocities_out: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
):
    """Initialize velocities from Maxwell-Boltzmann distribution (non-mutating, batched).

    Parameters
    ----------
    masses : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Atomic masses. Shape (num_atoms,).
    temperature : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Temperature. Shape (num_systems,).
    random_seed : wp.uint64
        Random seed.
    velocities_out : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Output velocities. Shape (num_atoms,).
    batch_idx : wp.array(dtype=wp.int32)
        System index for each atom. Shape (num_atoms,).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    mass = masses[atom_idx]
    kT = temperature[system_id]

    kT_typed = type(mass)(kT)
    sigma = wp.sqrt(kT_typed / mass)

    rng_state = wp.rand_init(int(random_seed), atom_idx)

    # Generate Gaussian-distributed velocities using wp.randn (N(0,1))
    vx = sigma * type(mass)(wp.randn(rng_state))
    vy = sigma * type(mass)(wp.randn(rng_state))
    vz = sigma * type(mass)(wp.randn(rng_state))

    vel_sample = velocities_out[atom_idx]
    velocities_out[atom_idx] = type(vel_sample)(vx, vy, vz)


# ==============================================================================
# COM Velocity Division Kernels
# ==============================================================================


@wp.kernel
def _compute_com_from_momentum_kernel(
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
    com_velocity: wp.array(dtype=Any),
):
    """Compute COM velocity from total momentum and mass (single system).

    v_COM = total_momentum / total_mass

    Parameters
    ----------
    total_momentum : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Total momentum. Shape (1,).
    total_mass : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Total mass. Shape (1,).
    com_velocity : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocity. Shape (1,).

    Launch Grid
    -----------
    dim = [1]
    """
    mass = total_mass[0]
    inv_mass = type(mass)(1.0) / mass
    com_velocity[0] = total_momentum[0] * inv_mass


@wp.kernel
def _batch_compute_com_from_momentum_kernel(
    total_momentum: wp.array(dtype=Any),
    total_mass: wp.array(dtype=Any),
    com_velocities: wp.array(dtype=Any),
):
    """Compute COM velocity from total momentum and mass (batched).

    Parameters
    ----------
    total_momentum : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Total momentum. Shape (num_systems,).
    total_mass : wp.array(dtype=Any), e.g., wp.array(dtype=wp.float32)
        Total mass. Shape (num_systems,).
    com_velocities : wp.array(dtype=Any), e.g., wp.array(dtype=wp.vec3f)
        Center of mass velocities. Shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    mass = total_mass[sys_id]
    inv_mass = type(mass)(1.0) / mass
    com_velocities[sys_id] = total_momentum[sys_id] * inv_mass


# ==============================================================================
# Temperature Computation Kernels
# ==============================================================================


@wp.kernel
def _compute_temperature_from_ke_kernel(
    kinetic_energy: wp.array(dtype=Any),
    dof: Any,
    temperature: wp.array(dtype=Any),
):
    """Compute temperature from kinetic energy: T = 2*KE / DOF

    Launch Grid
    -----------
    dim = [1] (or [num_systems] for batched)
    """
    sys_id = wp.tid()
    ke = kinetic_energy[sys_id]
    temperature[sys_id] = type(ke)(2.0) * ke / dof


@wp.kernel
def _batch_compute_temperature_from_ke_kernel(
    kinetic_energies: wp.array(dtype=Any),
    dof: Any,
    temperatures: wp.array(dtype=Any),
):
    """Compute temperature from kinetic energy for batched systems: T = 2*KE / DOF

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    ke = kinetic_energies[sys_id]
    temperatures[sys_id] = type(ke)(2.0) * ke / dof


# ==============================================================================
# Constant Fill Kernels (Pure Warp - no numpy)
# ==============================================================================


@wp.kernel
def _fill_constant_f32_kernel(
    arr: wp.array(dtype=wp.float32),
    value: wp.float32,
):
    """Fill array with constant value (float32).

    Launch Grid
    -----------
    dim = [array_length]
    """
    idx = wp.tid()
    arr[idx] = value


@wp.kernel
def _fill_constant_f64_kernel(
    arr: wp.array(dtype=wp.float64),
    value: wp.float64,
):
    """Fill array with constant value (float64).

    Launch Grid
    -----------
    dim = [array_length]
    """
    idx = wp.tid()
    arr[idx] = value


# ==============================================================================
# Kernel Overloads for Explicit Typing
# ==============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]      # Vector types

# Kinetic energy kernel overloads
_compute_kinetic_energy_kernel_overload = {}
_batch_compute_kinetic_energy_kernel_overload = {}

# Temperature kernel overloads
_compute_temperature_from_ke_kernel_overload = {}
_batch_compute_temperature_from_ke_kernel_overload = {}

# COM velocity kernel overloads
_compute_com_velocity_kernel_overload = {}
_batch_compute_com_velocity_kernel_overload = {}

# Remove COM motion kernel overloads
_remove_com_motion_kernel_overload = {}
_batch_remove_com_motion_kernel_overload = {}
_remove_com_motion_out_kernel_overload = {}
_batch_remove_com_motion_out_kernel_overload = {}

# Initialize velocities kernel overloads
_initialize_velocities_kernel_overload = {}
_batch_initialize_velocities_kernel_overload = {}
_initialize_velocities_out_kernel_overload = {}
_batch_initialize_velocities_out_kernel_overload = {}

for t, v in zip(_T, _V):
    # Kinetic energy kernels (dtype agnostic - output matches input type)
    _compute_kinetic_energy_kernel_overload[v] = wp.overload(
        _compute_kinetic_energy_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t)],
    )
    _batch_compute_kinetic_energy_kernel_overload[v] = wp.overload(
        _batch_compute_kinetic_energy_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )

    # Temperature kernels (keyed by scalar type)
    _compute_temperature_from_ke_kernel_overload[t] = wp.overload(
        _compute_temperature_from_ke_kernel,
        [wp.array(dtype=t), t, wp.array(dtype=t)],
    )
    _batch_compute_temperature_from_ke_kernel_overload[t] = wp.overload(
        _batch_compute_temperature_from_ke_kernel,
        [wp.array(dtype=t), t, wp.array(dtype=t)],
    )

    # COM velocity kernels (now using 1D vector arrays for momentum)
    _compute_com_velocity_kernel_overload[v] = wp.overload(
        _compute_com_velocity_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=v), wp.array(dtype=t)],
    )
    _batch_compute_com_velocity_kernel_overload[v] = wp.overload(
        _batch_compute_com_velocity_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=v), wp.array(dtype=t),
         wp.array(dtype=wp.int32)],
    )

    # Remove COM motion kernels (now using 1D vector arrays for com_velocity)
    _remove_com_motion_kernel_overload[v] = wp.overload(
        _remove_com_motion_kernel,
        [wp.array(dtype=v), wp.array(dtype=v)],
    )
    _batch_remove_com_motion_kernel_overload[v] = wp.overload(
        _batch_remove_com_motion_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=v)],
    )
    _remove_com_motion_out_kernel_overload[v] = wp.overload(
        _remove_com_motion_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=v)],
    )
    _batch_remove_com_motion_out_kernel_overload[v] = wp.overload(
        _batch_remove_com_motion_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=v), wp.array(dtype=v)],
    )

    # Initialize velocities kernels (batch_idx moved to end)
    _initialize_velocities_kernel_overload[v] = wp.overload(
        _initialize_velocities_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t), wp.uint64],
    )
    _batch_initialize_velocities_kernel_overload[v] = wp.overload(
        _batch_initialize_velocities_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t), wp.uint64, wp.array(dtype=wp.int32)],
    )
    _initialize_velocities_out_kernel_overload[v] = wp.overload(
        _initialize_velocities_out_kernel,
        [wp.array(dtype=t), wp.array(dtype=t), wp.uint64, wp.array(dtype=v)],
    )
    _batch_initialize_velocities_out_kernel_overload[v] = wp.overload(
        _batch_initialize_velocities_out_kernel,
        [wp.array(dtype=t), wp.array(dtype=t), wp.uint64, wp.array(dtype=v), wp.array(dtype=wp.int32)],
    )


# ==============================================================================
# Functional Interface
# ==============================================================================


def compute_kinetic_energy(
    velocities: wp.array,
    masses: wp.array,
    kinetic_energy: wp.array = None,
    batch_idx: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> wp.array:
    """
    Compute kinetic energy for single or batched MD systems.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    kinetic_energy : wp.array, optional
        Pre-allocated output array. If None, will be created with same dtype as masses.
        Shape (1,) for single system, (B,) for batched.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    num_systems : int, optional
        Number of systems for batched mode. Default 1.
    device : str, optional
        Warp device. If None, inferred from velocities.

    Returns
    -------
    wp.array
        Kinetic energy (same dtype as masses). Shape (1,) for single, (B,) for batched.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    is_batched = batch_idx is not None

    # Determine output dtype from masses
    scalar_dtype = masses.dtype

    # Allocate output if needed
    if kinetic_energy is None:
        kinetic_energy = wp.zeros(num_systems, dtype=scalar_dtype, device=device)
    else:
        kinetic_energy.zero_()

    vec_dtype = velocities.dtype
    if is_batched:
        wp.launch(
            _batch_compute_kinetic_energy_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, batch_idx, kinetic_energy],
            device=device,
        )
    else:
        wp.launch(
            _compute_kinetic_energy_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, kinetic_energy],
            device=device,
        )

    return kinetic_energy


def compute_temperature(
    velocities: wp.array,
    masses: wp.array,
    num_atoms: int,
    dof: int = None,
    kinetic_energy: wp.array = None,
    batch_idx: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> wp.array:
    """
    Compute instantaneous temperature from kinetic energy.

    Temperature is computed as T = 2*KE / (DOF * k_B), where k_B = 1
    in natural units (so temperature is in energy units).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    num_atoms : int
        Number of atoms (per system for batched).
    dof : int, optional
        Degrees of freedom. If None, uses 3*N - 3.
    kinetic_energy : wp.array, optional
        Pre-computed kinetic energy. If None, will be computed.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    num_systems : int, optional
        Number of systems. Default 1.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Temperature in energy units (k_B*T). Shape (1,) or (B,).
        Dtype matches masses dtype.
    """
    if device is None:
        device = velocities.device

    # Determine scalar dtype from masses
    scalar_dtype = masses.dtype

    # Compute kinetic energy if not provided
    if kinetic_energy is None:
        kinetic_energy = compute_kinetic_energy(
            velocities, masses, batch_idx=batch_idx, num_systems=num_systems, device=device
        )

    # Determine DOF
    if dof is None:
        dof_val = float(3 * num_atoms - 3)
    else:
        dof_val = float(dof)

    # Get typed dof value
    if scalar_dtype == wp.float32:
        dof_typed = wp.float32(dof_val)
    else:
        dof_typed = wp.float64(dof_val)

    # Compute temperature: T = 2*KE / DOF using Warp kernel
    is_batched = batch_idx is not None or num_systems > 1
    n_out = num_systems if is_batched else 1
    temperature = wp.zeros(n_out, dtype=scalar_dtype, device=device)

    if is_batched:
        wp.launch(
            _batch_compute_temperature_from_ke_kernel_overload[scalar_dtype],
            dim=num_systems,
            inputs=[kinetic_energy, dof_typed, temperature],
            device=device,
        )
    else:
        wp.launch(
            _compute_temperature_from_ke_kernel_overload[scalar_dtype],
            dim=1,
            inputs=[kinetic_energy, dof_typed, temperature],
            device=device,
        )

    return temperature


def initialize_velocities(
    velocities: wp.array,
    masses: wp.array,
    temperature: wp.array,
    random_seed: int = 42,
    remove_com: bool = True,
    batch_idx: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> None:
    """
    Initialize velocities from Maxwell-Boltzmann distribution (in-place).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Target temperature (k_B*T in energy units). Shape (1,) or (B,).
    random_seed : int, optional
        Random seed for reproducibility. Default: 42.
    remove_com : bool, optional
        If True, remove center of mass motion after initialization.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    num_systems : int, optional
        Number of systems. Default 1.
    device : str, optional
        Warp device.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    is_batched = batch_idx is not None
    vec_dtype = velocities.dtype

    # Determine scalar dtype from vector dtype
    if vec_dtype == wp.vec3d:
        scalar_dtype = wp.float64
    else:
        scalar_dtype = wp.float32

    if is_batched:
        wp.launch(
            _batch_initialize_velocities_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, temperature, wp.uint64(random_seed), batch_idx],
            device=device,
        )
    else:
        wp.launch(
            _initialize_velocities_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, temperature, wp.uint64(random_seed)],
            device=device,
        )

    if remove_com:
        remove_com_motion(velocities, masses, batch_idx=batch_idx, num_systems=num_systems, device=device)


def initialize_velocities_out(
    masses: wp.array,
    temperature: wp.array,
    velocities_out: wp.array = None,
    random_seed: int = 42,
    remove_com: bool = True,
    batch_idx: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> wp.array:
    """
    Initialize velocities from Maxwell-Boltzmann distribution (non-mutating).

    Parameters
    ----------
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    temperature : wp.array(dtype=wp.float32 or wp.float64)
        Target temperature (k_B*T in energy units). Shape (1,) or (B,).
    velocities_out : wp.array, optional
        Output array. If None, allocated internally (needs dtype hint from masses).
    random_seed : int, optional
        Random seed for reproducibility. Default: 42.
    remove_com : bool, optional
        If True, remove center of mass motion after initialization.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    num_systems : int, optional
        Number of systems. Default 1.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Initialized velocities.
    """
    if device is None:
        device = masses.device

    num_atoms = masses.shape[0]
    is_batched = batch_idx is not None

    # Determine correct dtypes based on masses
    scalar_dtype = masses.dtype
    if scalar_dtype == wp.float64:
        vec_dtype = wp.vec3d
    else:
        vec_dtype = wp.vec3f

    # Allocate output if needed
    if velocities_out is None:
        velocities_out = wp.zeros(num_atoms, dtype=vec_dtype, device=device)

    if is_batched:
        wp.launch(
            _batch_initialize_velocities_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[masses, temperature, wp.uint64(random_seed), velocities_out, batch_idx],
            device=device,
        )
    else:
        wp.launch(
            _initialize_velocities_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[masses, temperature, wp.uint64(random_seed), velocities_out],
            device=device,
        )

    if remove_com:
        velocities_out = remove_com_motion_out(
            velocities_out, masses, batch_idx=batch_idx, num_systems=num_systems, device=device
        )

    return velocities_out


def remove_com_motion(
    velocities: wp.array,
    masses: wp.array,
    batch_idx: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> None:
    """
    Remove center of mass velocity from the system (in-place).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    num_systems : int, optional
        Number of systems. Default 1.
    device : str, optional
        Warp device.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    is_batched = batch_idx is not None

    vec_dtype = velocities.dtype
    # Determine scalar dtype from vector dtype
    scalar_dtype = wp.float32 if vec_dtype == wp.vec3f else wp.float64

    if is_batched:
        # Use 1D vector arrays for momentum and COM velocity
        total_momentum = wp.zeros(num_systems, dtype=vec_dtype, device=device)
        total_mass = wp.zeros(num_systems, dtype=scalar_dtype, device=device)

        wp.launch(
            _batch_compute_com_velocity_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, total_momentum, total_mass, batch_idx],
            device=device,
        )

        # Compute COM velocity using Warp kernel (no numpy)
        com_velocities = wp.zeros(num_systems, dtype=vec_dtype, device=device)
        wp.launch(
            _batch_compute_com_from_momentum_kernel,
            dim=num_systems,
            inputs=[total_momentum, total_mass, com_velocities],
            device=device,
        )

        wp.launch(
            _batch_remove_com_motion_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, batch_idx, com_velocities],
            device=device,
        )
    else:
        # Use 1D vector array for momentum (shape 1)
        total_momentum = wp.zeros(1, dtype=vec_dtype, device=device)
        total_mass = wp.zeros(1, dtype=scalar_dtype, device=device)

        wp.launch(
            _compute_com_velocity_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, total_momentum, total_mass],
            device=device,
        )

        # Compute COM velocity using Warp kernel (no numpy)
        com_velocity = wp.zeros(1, dtype=vec_dtype, device=device)
        wp.launch(
            _compute_com_from_momentum_kernel,
            dim=1,
            inputs=[total_momentum, total_mass, com_velocity],
            device=device,
        )

        wp.launch(
            _remove_com_motion_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, com_velocity],
            device=device,
        )


def remove_com_motion_out(
    velocities: wp.array,
    masses: wp.array,
    velocities_out: wp.array = None,
    batch_idx: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> wp.array:
    """
    Remove center of mass velocity from the system (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    velocities_out : wp.array, optional
        Output array. If None, allocated internally.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    num_systems : int, optional
        Number of systems. Default 1.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Velocities with COM motion removed.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    is_batched = batch_idx is not None

    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    vec_dtype = velocities.dtype
    # Determine scalar dtype from vector dtype
    scalar_dtype = wp.float32 if vec_dtype == wp.vec3f else wp.float64

    if is_batched:
        # Use 1D vector arrays for momentum and COM velocity
        total_momentum = wp.zeros(num_systems, dtype=vec_dtype, device=device)
        total_mass = wp.zeros(num_systems, dtype=scalar_dtype, device=device)

        wp.launch(
            _batch_compute_com_velocity_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, total_momentum, total_mass, batch_idx],
            device=device,
        )

        # Compute COM velocity using Warp kernel (no numpy)
        com_velocities = wp.zeros(num_systems, dtype=vec_dtype, device=device)
        wp.launch(
            _batch_compute_com_from_momentum_kernel,
            dim=num_systems,
            inputs=[total_momentum, total_mass, com_velocities],
            device=device,
        )

        wp.launch(
            _batch_remove_com_motion_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, batch_idx, com_velocities, velocities_out],
            device=device,
        )
    else:
        # Use 1D vector array for momentum (shape 1)
        total_momentum = wp.zeros(1, dtype=vec_dtype, device=device)
        total_mass = wp.zeros(1, dtype=scalar_dtype, device=device)

        wp.launch(
            _compute_com_velocity_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, masses, total_momentum, total_mass],
            device=device,
        )

        # Compute COM velocity using Warp kernel (no numpy)
        com_velocity = wp.zeros(1, dtype=vec_dtype, device=device)
        wp.launch(
            _compute_com_from_momentum_kernel,
            dim=1,
            inputs=[total_momentum, total_mass, com_velocity],
            device=device,
        )

        wp.launch(
            _remove_com_motion_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, com_velocity, velocities_out],
            device=device,
        )

    return velocities_out
