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
Velocity Verlet Integrator Kernels
==================================

GPU-accelerated Warp kernels for the velocity Verlet integrator,
providing time-reversible, symplectic integration for NVE molecular dynamics.

This module provides both mutating (in-place) and non-mutating versions
of each kernel for gradient tracking compatibility.

NAMING CONVENTION
=================

Functions in this module follow a consistent naming scheme:

- **Mutating functions** (e.g., ``velocity_verlet_position_update``):
  Modify arrays in-place for efficiency. Use when gradients are not needed.
  Faster but not compatible with autograd.

- **Non-mutating functions** (e.g., ``velocity_verlet_position_update_out``):
  Append ``_out`` suffix and return new arrays without modifying inputs.
  Use when gradient tracking is required. Compatible with autograd but
  requires additional memory.

This pattern is consistent across all dynamics modules (integrators, optimizers).
See CLAUDE.md architectural patterns section for implementation details.

MATHEMATICAL FORMULATION
========================

The velocity Verlet algorithm is a second-order symplectic integrator:

Position update:

.. math::

    \\mathbf{r}(t + \\Delta t) = \\mathbf{r}(t) + \\mathbf{v}(t) \\Delta t
                                + \\frac{1}{2} \\mathbf{a}(t) \\Delta t^2

Velocity update:

.. math::

    \\mathbf{v}(t + \\Delta t) = \\mathbf{v}(t) + \\frac{1}{2}[\\mathbf{a}(t)
                                + \\mathbf{a}(t + \\Delta t)] \\Delta t

where :math:`\\mathbf{a} = \\mathbf{F}/m` is the acceleration.

TWO-PASS ALGORITHM
==================

**Pass 1 (position_update):**
    - Update positions to r(t+dt)
    - Update velocities to half-step v(t+dt/2) = v(t) + 0.5*a(t)*dt

**[User recalculates forces at new positions]**

**Pass 2 (velocity_finalize):**
    - Complete velocity update: v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt

USAGE EXAMPLE
=============

Mutating (in-place) version::

    for step in range(num_steps):
        # Pass 1: Update positions and half-step velocities
        velocity_verlet_position_update(
            positions, velocities, forces, masses, dt
        )

        # Recalculate forces at new positions
        forces = compute_forces(positions)

        # Pass 2: Complete velocity update
        velocity_verlet_velocity_finalize(
            velocities, forces, masses, dt
        )

Non-mutating version (for gradient tracking)::

    for step in range(num_steps):
        # Pass 1: Compute new positions and velocities
        new_positions, new_velocities = velocity_verlet_position_update_out(
            positions, velocities, forces, masses, dt
        )

        # Recalculate forces at new positions
        new_forces = compute_forces(new_positions)

        # Pass 2: Complete velocity update
        final_velocities = velocity_verlet_velocity_finalize_out(
            new_velocities, new_forces, masses, dt
        )

        positions, velocities, forces = new_positions, final_velocities, new_forces

BATCH MODE
==========

This module supports three execution modes for simulating multiple systems:

**Single System Mode** (default)::

    dt = wp.array([0.001], dtype=wp.float64, device="cuda:0")
    velocity_verlet_position_update(positions, velocities, forces, masses, dt)

**Batch Mode with batch_idx** (atomic operations)::

    # For systems with varying atom counts
    # batch_idx maps each atom to its system
    batch_idx = wp.array([0]*N0 + [1]*N1 + [2]*N2, dtype=wp.int32, device="cuda:0")
    dt = wp.array([dt0, dt1, dt2], dtype=wp.float64, device="cuda:0")  # Per-system timesteps

    velocity_verlet_position_update(
        positions, velocities, forces, masses, dt, batch_idx=batch_idx
    )

    # Launched with dim=num_atoms_total (parallel per-atom operations)

**Batch Mode with atom_ptr** (sequential per-system)::

    # For systems where each thread should process one complete system
    # atom_ptr defines atom ranges: system s owns atoms [atom_ptr[s], atom_ptr[s+1])
    atom_ptr = wp.array([0, N0, N0+N1, N0+N1+N2], dtype=wp.int32, device="cuda:0")
    dt = wp.array([dt0, dt1, dt2], dtype=wp.float64, device="cuda:0")

    velocity_verlet_position_update(
        positions, velocities, forces, masses, dt, atom_ptr=atom_ptr
    )

    # Launched with dim=num_systems (each thread processes one system sequentially)

**Choosing Between batch_idx and atom_ptr:**

- Use **batch_idx** when:
    - Systems have similar sizes
    - You want maximum parallelism (one thread per atom)
    - Memory access patterns are coalesced

- Use **atom_ptr** when:
    - Systems have very different sizes
    - You need per-system operations (reductions, etc.)
    - Each system needs independent sequential processing

REFERENCES
==========

- Swope et al. (1982). J. Chem. Phys. 76, 637 (Velocity Verlet)
- Verlet, L. (1967). Phys. Rev. 159, 98 (Original Verlet method)
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
    "velocity_verlet_position_update",
    "velocity_verlet_velocity_finalize",
    # Non-mutating (output) APIs
    "velocity_verlet_position_update_out",
    "velocity_verlet_velocity_finalize_out",
]


# ==============================================================================
# Mutating Kernels (in-place updates)
# ==============================================================================


@wp.kernel
def _velocity_verlet_position_update_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
):
    """Update positions and half-step velocities (in-place).

    Computes:

    - :math:`\\mathbf{r}(t+\\Delta t) = \\mathbf{r}(t) + \\mathbf{v}(t)\\Delta t + \\frac{1}{2}\\mathbf{a}(t)\\Delta t^2`
    - :math:`\\mathbf{v}_{\\text{half}} = \\mathbf{v}(t) + \\frac{1}{2}\\mathbf{a}(t)\\Delta t`

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

    # Compute acceleration from force
    acc = compute_acceleration_from_force(force, mass)

    # Update position: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    new_pos = velocity_verlet_position_step(pos, vel, acc, dt_val)

    # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
    half_vel = velocity_half_step_from_acceleration(vel, acc, dt_val)

    # Write back
    positions[atom_idx] = new_pos
    velocities[atom_idx] = half_vel


@wp.kernel
def _velocity_verlet_velocity_finalize_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
):
    """Finalize velocity update after force recalculation (in-place).

    Computes:
        v(t+dt) = v_half + 0.5*a(t+dt)*dt

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel_half = velocities[atom_idx]
    force = forces_new[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[0]

    # Compute acceleration at new positions
    acc = compute_acceleration_from_force(force, mass)

    # Complete velocity update: v(t+dt) = v_half + 0.5*a(t+dt)*dt
    new_vel = velocity_half_step_from_acceleration(vel_half, acc, dt_val)

    velocities[atom_idx] = new_vel


# ==============================================================================
# Non-Mutating Kernels (write to output arrays)
# ==============================================================================


@wp.kernel
def _velocity_verlet_position_update_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Update positions and half-step velocities (non-mutating).

    Computes:

    - :math:`\\mathbf{r}(t+\\Delta t) = \\mathbf{r}(t) + \\mathbf{v}(t)\\Delta t + \\frac{1}{2}\\mathbf{a}(t)\\Delta t^2`
    - :math:`\\mathbf{v}_{\\text{half}} = \\mathbf{v}(t) + \\frac{1}{2}\\mathbf{a}(t)\\Delta t`

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

    # Compute acceleration from force
    acc = compute_acceleration_from_force(force, mass)

    # Update position: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    new_pos = velocity_verlet_position_step(pos, vel, acc, dt_val)

    # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
    half_vel = velocity_half_step_from_acceleration(vel, acc, dt_val)

    # Write to output arrays
    positions_out[atom_idx] = new_pos
    velocities_out[atom_idx] = half_vel


@wp.kernel
def _velocity_verlet_velocity_finalize_out_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Finalize velocity update after force recalculation (non-mutating).

    Computes:
        v(t+dt) = v_half + 0.5*a(t+dt)*dt

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    vel_half = velocities[atom_idx]
    force = forces_new[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[0]

    # Compute acceleration at new positions
    acc = compute_acceleration_from_force(force, mass)

    # Complete velocity update: v(t+dt) = v_half + 0.5*a(t+dt)*dt
    new_vel = velocity_half_step_from_acceleration(vel_half, acc, dt_val)

    velocities_out[atom_idx] = new_vel


# ==============================================================================
# Batched Mutating Kernels
# ==============================================================================


@wp.kernel
def _batch_velocity_verlet_position_update_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """Update positions and half-step velocities for batched systems (in-place).

    Each atom uses the timestep for its corresponding system via batch_idx.

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()

    # Get per-system timestep via batch_idx
    dt_val = dt[batch_idx[atom_idx]]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    # Compute acceleration from force
    acc = compute_acceleration_from_force(force, mass)

    # Update position: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    new_pos = velocity_verlet_position_step(pos, vel, acc, dt_val)

    # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
    half_vel = velocity_half_step_from_acceleration(vel, acc, dt_val)

    # Write back
    positions[atom_idx] = new_pos
    velocities[atom_idx] = half_vel


@wp.kernel
def _batch_velocity_verlet_velocity_finalize_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """Finalize velocity update for batched systems (in-place).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()

    dt_val = dt[batch_idx[atom_idx]]

    vel_half = velocities[atom_idx]
    force = forces_new[atom_idx]
    mass = masses[atom_idx]

    # Compute acceleration at new positions
    acc = compute_acceleration_from_force(force, mass)

    # Complete velocity update: v(t+dt) = v_half + 0.5*a(t+dt)*dt
    new_vel = velocity_half_step_from_acceleration(vel_half, acc, dt_val)

    velocities[atom_idx] = new_vel


# ==============================================================================
# Batched Non-Mutating Kernels
# ==============================================================================


@wp.kernel
def _batch_velocity_verlet_position_update_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Update positions and half-step velocities for batched systems (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()

    dt_val = dt[batch_idx[atom_idx]]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    # Compute acceleration from force
    acc = compute_acceleration_from_force(force, mass)

    # Update position: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    new_pos = velocity_verlet_position_step(pos, vel, acc, dt_val)

    # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
    half_vel = velocity_half_step_from_acceleration(vel, acc, dt_val)

    # Write to output arrays
    positions_out[atom_idx] = new_pos
    velocities_out[atom_idx] = half_vel


@wp.kernel
def _batch_velocity_verlet_velocity_finalize_out_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Finalize velocity update for batched systems (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()

    dt_val = dt[batch_idx[atom_idx]]

    vel_half = velocities[atom_idx]
    force = forces_new[atom_idx]
    mass = masses[atom_idx]

    # Compute acceleration at new positions
    acc = compute_acceleration_from_force(force, mass)

    # Complete velocity update: v(t+dt) = v_half + 0.5*a(t+dt)*dt
    new_vel = velocity_half_step_from_acceleration(vel_half, acc, dt_val)

    velocities_out[atom_idx] = new_vel


# ==============================================================================
# Pointer-Based (CSR) Mutating Kernels
# ==============================================================================


@wp.kernel
def _velocity_verlet_position_update_ptr_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """Update positions and half-step velocities using atom_ptr (in-place).

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
        System s owns atoms in range [atom_ptr[s], atom_ptr[s+1]).
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
        pos = positions[i]
        vel = velocities[i]
        force = forces[i]
        mass = masses[i]

        # Compute acceleration from force
        acc = compute_acceleration_from_force(force, mass)

        # Update position: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        new_pos = velocity_verlet_position_step(pos, vel, acc, dt_val)

        # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
        half_vel = velocity_half_step_from_acceleration(vel, acc, dt_val)

        # Write back
        positions[i] = new_pos
        velocities[i] = half_vel


@wp.kernel
def _velocity_verlet_velocity_finalize_ptr_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """Finalize velocity update using atom_ptr (in-place).

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
        vel_half = velocities[i]
        force = forces_new[i]
        mass = masses[i]

        # Compute acceleration at new positions
        acc = compute_acceleration_from_force(force, mass)

        # Complete velocity update: v(t+dt) = v_half + 0.5*a(t+dt)*dt
        new_vel = velocity_half_step_from_acceleration(vel_half, acc, dt_val)

        velocities[i] = new_vel


# ==============================================================================
# Pointer-Based (CSR) Non-Mutating Kernels
# ==============================================================================


@wp.kernel
def _velocity_verlet_position_update_ptr_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Update positions and half-step velocities using atom_ptr (non-mutating).

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

    for i in range(a0, a1):
        pos = positions[i]
        vel = velocities[i]
        force = forces[i]
        mass = masses[i]

        # Compute acceleration from force
        acc = compute_acceleration_from_force(force, mass)

        # Update position: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        new_pos = velocity_verlet_position_step(pos, vel, acc, dt_val)

        # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
        half_vel = velocity_half_step_from_acceleration(vel, acc, dt_val)

        # Write to output arrays
        positions_out[i] = new_pos
        velocities_out[i] = half_vel


@wp.kernel
def _velocity_verlet_velocity_finalize_ptr_out_kernel(
    velocities: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Finalize velocity update using atom_ptr (non-mutating).

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
        vel_half = velocities[i]
        force = forces_new[i]
        mass = masses[i]

        # Compute acceleration at new positions
        acc = compute_acceleration_from_force(force, mass)

        # Complete velocity update: v(t+dt) = v_half + 0.5*a(t+dt)*dt
        new_vel = velocity_half_step_from_acceleration(vel_half, acc, dt_val)

        velocities_out[i] = new_vel


# ==============================================================================
# Kernel Overloads via KernelFamily
# ==============================================================================

# Position update (inplace) -- keyed by vec_dtype
_position_update_families = {
    v: KernelFamily(
        single=register_overloads(
            _velocity_verlet_position_update_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=t)],
            dtype_pairs=((v, t),),
        )[v],
        batch_idx=register_overloads(
            _batch_velocity_verlet_position_update_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
            dtype_pairs=((v, t),),
        )[v],
        atom_ptr=register_overloads(
            _velocity_verlet_position_update_ptr_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
            dtype_pairs=((v, t),),
        )[v],
    )
    for v, t in ((wp.vec3f, wp.float32), (wp.vec3d, wp.float64))
}

# Position update (out) -- keyed by vec_dtype
_position_update_out_families = {
    v: KernelFamily(
        single=register_overloads(
            _velocity_verlet_position_update_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=t),
                           wp.array(dtype=v_), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
        batch_idx=register_overloads(
            _batch_velocity_verlet_position_update_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t),
                           wp.array(dtype=v_), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
        atom_ptr=register_overloads(
            _velocity_verlet_position_update_ptr_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t),
                           wp.array(dtype=v_), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
    )
    for v, t in ((wp.vec3f, wp.float32), (wp.vec3d, wp.float64))
}

# Velocity finalize (inplace) -- keyed by vec_dtype
_velocity_finalize_families = {
    v: KernelFamily(
        single=register_overloads(
            _velocity_verlet_velocity_finalize_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=t)],
            dtype_pairs=((v, t),),
        )[v],
        batch_idx=register_overloads(
            _batch_velocity_verlet_velocity_finalize_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
            dtype_pairs=((v, t),),
        )[v],
        atom_ptr=register_overloads(
            _velocity_verlet_velocity_finalize_ptr_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
            dtype_pairs=((v, t),),
        )[v],
    )
    for v, t in ((wp.vec3f, wp.float32), (wp.vec3d, wp.float64))
}

# Velocity finalize (out) -- keyed by vec_dtype
_velocity_finalize_out_families = {
    v: KernelFamily(
        single=register_overloads(
            _velocity_verlet_velocity_finalize_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
        batch_idx=register_overloads(
            _batch_velocity_verlet_velocity_finalize_out_kernel,
            lambda v_, t: [wp.array(dtype=v_), wp.array(dtype=v_),
                           wp.array(dtype=t), wp.array(dtype=wp.int32),
                           wp.array(dtype=t), wp.array(dtype=v_)],
            dtype_pairs=((v, t),),
        )[v],
        atom_ptr=register_overloads(
            _velocity_verlet_velocity_finalize_ptr_out_kernel,
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


def velocity_verlet_position_update(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> None:
    """
    Perform velocity Verlet position update step (in-place).

    Updates positions to r(t+dt) and velocities to half-step v(t+dt/2).
    After calling this function, recalculate forces at the new positions,
    then call velocity_verlet_velocity_finalize().

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
        Timestep(s). Shape (1,) for single system, (B,) for batched.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from positions.

    See Also
    --------
    velocity_verlet_velocity_finalize : Complete the velocity update
    """
    mode = resolve_execution_mode(batch_idx, atom_ptr)

    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]
    vec_dtype = positions.dtype
    family = _position_update_families[vec_dtype]

    if mode is ExecutionMode.ATOM_PTR:
        dim = atom_ptr.shape[0] - 1  # num_systems
    else:
        dim = num_atoms

    launch_family(
        family,
        mode=mode,
        dim=dim,
        inputs_single=[positions, velocities, forces, masses, dt],
        inputs_batch=[positions, velocities, forces, masses, batch_idx, dt],
        inputs_ptr=[positions, velocities, forces, masses, atom_ptr, dt],
        device=device,
    )


def velocity_verlet_velocity_finalize(
    velocities: wp.array,
    forces_new: wp.array,
    masses: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> None:
    """
    Finalize velocity Verlet velocity update (in-place).

    Completes the velocity update using forces evaluated at the new positions:
    v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Half-step velocities. Shape (N,). MODIFIED in-place to full-step.
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces evaluated at new positions r(t+dt). Shape (N,).
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

    See Also
    --------
    velocity_verlet_position_update : First step of velocity Verlet
    """
    mode = resolve_execution_mode(batch_idx, atom_ptr)

    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    vec_dtype = velocities.dtype
    family = _velocity_finalize_families[vec_dtype]

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


def velocity_verlet_position_update_out(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    positions_out: wp.array,
    velocities_out: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> tuple[wp.array, wp.array]:
    """
    Perform velocity Verlet position update step (non-mutating).

    Writes new positions r(t+dt) and half-step velocities v(t+dt/2) to
    output arrays. Input arrays are NOT modified.

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
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    positions_out : wp.array
        Pre-allocated output array for new positions.  Must match
        ``positions`` in shape, dtype, and device.
    velocities_out : wp.array
        Pre-allocated output array for half-step velocities.  Must match
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
        (positions_out, velocities_out) - New positions and half-step velocities.
    """
    mode = resolve_execution_mode(batch_idx, atom_ptr)

    validate_out_array(positions_out, positions, "positions_out")
    validate_out_array(velocities_out, velocities, "velocities_out")

    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]
    vec_dtype = positions.dtype
    family = _position_update_out_families[vec_dtype]

    if mode is ExecutionMode.ATOM_PTR:
        dim = atom_ptr.shape[0] - 1  # num_systems
    else:
        dim = num_atoms

    launch_family(
        family,
        mode=mode,
        dim=dim,
        inputs_single=[positions, velocities, forces, masses, dt,
                        positions_out, velocities_out],
        inputs_batch=[positions, velocities, forces, masses, batch_idx, dt,
                       positions_out, velocities_out],
        inputs_ptr=[positions, velocities, forces, masses, atom_ptr, dt,
                     positions_out, velocities_out],
        device=device,
    )

    return positions_out, velocities_out


def velocity_verlet_velocity_finalize_out(
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
    Finalize velocity Verlet velocity update (non-mutating).

    Writes full-step velocities v(t+dt) to output array.
    Input arrays are NOT modified.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Half-step velocities v(t+dt/2). Shape (N,).
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions r(t+dt). Shape (N,).
    masses : wp.array(dtype=wp.float32 or wp.float64)
        Atomic masses. Shape (N,).
    dt : wp.array(dtype=wp.float32 or wp.float64)
        Timestep(s). Shape (1,) for single, (B,) for batched.
    velocities_out : wp.array
        Pre-allocated output array for full-step velocities.  Must match
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
        Full-step velocities v(t+dt).
    """
    mode = resolve_execution_mode(batch_idx, atom_ptr)

    validate_out_array(velocities_out, velocities, "velocities_out")

    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    vec_dtype = velocities.dtype
    family = _velocity_finalize_out_families[vec_dtype]

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
