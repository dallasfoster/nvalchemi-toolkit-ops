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

    inv_mass = type(mass)(1.0) / mass

    # Compute acceleration
    acc_x = force[0] * inv_mass
    acc_y = force[1] * inv_mass
    acc_z = force[2] * inv_mass

    # Position update: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

    # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
    half_dt = type(dt_val)(0.5) * dt_val
    half_vel_x = vel[0] + acc_x * half_dt
    half_vel_y = vel[1] + acc_y * half_dt
    half_vel_z = vel[2] + acc_z * half_dt

    # Write back
    positions[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities[atom_idx] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


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

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    new_vel_x = vel_half[0] + force[0] * inv_mass * half_dt
    new_vel_y = vel_half[1] + force[1] * inv_mass * half_dt
    new_vel_z = vel_half[2] + force[2] * inv_mass * half_dt

    velocities[atom_idx] = type(vel_half)(new_vel_x, new_vel_y, new_vel_z)


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

    inv_mass = type(mass)(1.0) / mass

    # Compute acceleration
    acc_x = force[0] * inv_mass
    acc_y = force[1] * inv_mass
    acc_z = force[2] * inv_mass

    # Position update: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

    # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
    half_dt = type(dt_val)(0.5) * dt_val
    half_vel_x = vel[0] + acc_x * half_dt
    half_vel_y = vel[1] + acc_y * half_dt
    half_vel_z = vel[2] + acc_z * half_dt

    # Write to output arrays
    positions_out[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities_out[atom_idx] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


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

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    new_vel_x = vel_half[0] + force[0] * inv_mass * half_dt
    new_vel_y = vel_half[1] + force[1] * inv_mass * half_dt
    new_vel_z = vel_half[2] + force[2] * inv_mass * half_dt

    velocities_out[atom_idx] = type(vel_half)(new_vel_x, new_vel_y, new_vel_z)


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
    system_id = batch_idx[atom_idx]
    dt_val = dt[system_id]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    inv_mass = type(mass)(1.0) / mass

    acc_x = force[0] * inv_mass
    acc_y = force[1] * inv_mass
    acc_z = force[2] * inv_mass

    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

    half_dt = type(dt_val)(0.5) * dt_val
    half_vel_x = vel[0] + acc_x * half_dt
    half_vel_y = vel[1] + acc_y * half_dt
    half_vel_z = vel[2] + acc_z * half_dt

    positions[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities[atom_idx] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


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

    system_id = batch_idx[atom_idx]
    dt_val = dt[system_id]

    vel_half = velocities[atom_idx]
    force = forces_new[atom_idx]
    mass = masses[atom_idx]

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    new_vel_x = vel_half[0] + force[0] * inv_mass * half_dt
    new_vel_y = vel_half[1] + force[1] * inv_mass * half_dt
    new_vel_z = vel_half[2] + force[2] * inv_mass * half_dt

    velocities[atom_idx] = type(vel_half)(new_vel_x, new_vel_y, new_vel_z)


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

    system_id = batch_idx[atom_idx]
    dt_val = dt[system_id]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    inv_mass = type(mass)(1.0) / mass

    acc_x = force[0] * inv_mass
    acc_y = force[1] * inv_mass
    acc_z = force[2] * inv_mass

    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

    half_dt = type(dt_val)(0.5) * dt_val
    half_vel_x = vel[0] + acc_x * half_dt
    half_vel_y = vel[1] + acc_y * half_dt
    half_vel_z = vel[2] + acc_z * half_dt

    positions_out[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities_out[atom_idx] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


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

    system_id = batch_idx[atom_idx]
    dt_val = dt[system_id]

    vel_half = velocities[atom_idx]
    force = forces_new[atom_idx]
    mass = masses[atom_idx]

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    new_vel_x = vel_half[0] + force[0] * inv_mass * half_dt
    new_vel_y = vel_half[1] + force[1] * inv_mass * half_dt
    new_vel_z = vel_half[2] + force[2] * inv_mass * half_dt

    velocities_out[atom_idx] = type(vel_half)(new_vel_x, new_vel_y, new_vel_z)


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
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    dt_val = dt[sys_id]

    half_dt = type(dt_val)(0.5) * dt_val
    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val

    for i in range(a0, a1):
        pos = positions[i]
        vel = velocities[i]
        force = forces[i]
        mass = masses[i]

        inv_mass = type(mass)(1.0) / mass

        # Compute acceleration
        acc_x = force[0] * inv_mass
        acc_y = force[1] * inv_mass
        acc_z = force[2] * inv_mass

        # Position update: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
        new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
        new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

        # Half-step velocity: v_half = v(t) + 0.5*a(t)*dt
        half_vel_x = vel[0] + acc_x * half_dt
        half_vel_y = vel[1] + acc_y * half_dt
        half_vel_z = vel[2] + acc_z * half_dt

        positions[i] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
        velocities[i] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


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
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    dt_val = dt[sys_id]

    half_dt = type(dt_val)(0.5) * dt_val

    for i in range(a0, a1):
        vel_half = velocities[i]
        force = forces_new[i]
        mass = masses[i]

        inv_mass = type(mass)(1.0) / mass

        new_vel_x = vel_half[0] + force[0] * inv_mass * half_dt
        new_vel_y = vel_half[1] + force[1] * inv_mass * half_dt
        new_vel_z = vel_half[2] + force[2] * inv_mass * half_dt

        velocities[i] = type(vel_half)(new_vel_x, new_vel_y, new_vel_z)


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
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    dt_val = dt[sys_id]

    half_dt = type(dt_val)(0.5) * dt_val
    half_dt_sq = type(dt_val)(0.5) * dt_val * dt_val

    for i in range(a0, a1):
        pos = positions[i]
        vel = velocities[i]
        force = forces[i]
        mass = masses[i]

        inv_mass = type(mass)(1.0) / mass

        # Compute acceleration
        acc_x = force[0] * inv_mass
        acc_y = force[1] * inv_mass
        acc_z = force[2] * inv_mass

        # Position update
        new_pos_x = pos[0] + vel[0] * dt_val + acc_x * half_dt_sq
        new_pos_y = pos[1] + vel[1] * dt_val + acc_y * half_dt_sq
        new_pos_z = pos[2] + vel[2] * dt_val + acc_z * half_dt_sq

        # Half-step velocity
        half_vel_x = vel[0] + acc_x * half_dt
        half_vel_y = vel[1] + acc_y * half_dt
        half_vel_z = vel[2] + acc_z * half_dt

        positions_out[i] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
        velocities_out[i] = type(vel)(half_vel_x, half_vel_y, half_vel_z)


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
    a0 = atom_ptr[sys_id]
    a1 = atom_ptr[sys_id + 1]
    dt_val = dt[sys_id]

    half_dt = type(dt_val)(0.5) * dt_val

    for i in range(a0, a1):
        vel_half = velocities[i]
        force = forces_new[i]
        mass = masses[i]

        inv_mass = type(mass)(1.0) / mass

        new_vel_x = vel_half[0] + force[0] * inv_mass * half_dt
        new_vel_y = vel_half[1] + force[1] * inv_mass * half_dt
        new_vel_z = vel_half[2] + force[2] * inv_mass * half_dt

        velocities_out[i] = type(vel_half)(new_vel_x, new_vel_y, new_vel_z)


# ==============================================================================
# Kernel Overloads for Explicit Typing
# ==============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]  # Vector types

# Position update kernel overloads
_velocity_verlet_position_update_kernel_overload = {}
_batch_velocity_verlet_position_update_kernel_overload = {}
_velocity_verlet_position_update_ptr_kernel_overload = {}
_velocity_verlet_position_update_out_kernel_overload = {}
_batch_velocity_verlet_position_update_out_kernel_overload = {}
_velocity_verlet_position_update_ptr_out_kernel_overload = {}

# Velocity finalize kernel overloads
_velocity_verlet_velocity_finalize_kernel_overload = {}
_batch_velocity_verlet_velocity_finalize_kernel_overload = {}
_velocity_verlet_velocity_finalize_ptr_kernel_overload = {}
_velocity_verlet_velocity_finalize_out_kernel_overload = {}
_batch_velocity_verlet_velocity_finalize_out_kernel_overload = {}
_velocity_verlet_velocity_finalize_ptr_out_kernel_overload = {}

for t, v in zip(_T, _V):
    # Position update kernels
    _velocity_verlet_position_update_kernel_overload[v] = wp.overload(
        _velocity_verlet_position_update_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=t),
        ],
    )
    _batch_velocity_verlet_position_update_kernel_overload[v] = wp.overload(
        _batch_velocity_verlet_position_update_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
        ],
    )
    _velocity_verlet_position_update_ptr_kernel_overload[v] = wp.overload(
        _velocity_verlet_position_update_ptr_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
        ],
    )
    _velocity_verlet_position_update_out_kernel_overload[v] = wp.overload(
        _velocity_verlet_position_update_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=v),
            wp.array(dtype=v),
        ],
    )
    _batch_velocity_verlet_position_update_out_kernel_overload[v] = wp.overload(
        _batch_velocity_verlet_position_update_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=v),
            wp.array(dtype=v),
        ],
    )
    _velocity_verlet_position_update_ptr_out_kernel_overload[v] = wp.overload(
        _velocity_verlet_position_update_ptr_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=v),
            wp.array(dtype=v),
        ],
    )

    # Velocity finalize kernels
    _velocity_verlet_velocity_finalize_kernel_overload[v] = wp.overload(
        _velocity_verlet_velocity_finalize_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t)],
    )
    _batch_velocity_verlet_velocity_finalize_kernel_overload[v] = wp.overload(
        _batch_velocity_verlet_velocity_finalize_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
        ],
    )
    _velocity_verlet_velocity_finalize_ptr_kernel_overload[v] = wp.overload(
        _velocity_verlet_velocity_finalize_ptr_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
        ],
    )
    _velocity_verlet_velocity_finalize_out_kernel_overload[v] = wp.overload(
        _velocity_verlet_velocity_finalize_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=v),
        ],
    )
    _batch_velocity_verlet_velocity_finalize_out_kernel_overload[v] = wp.overload(
        _batch_velocity_verlet_velocity_finalize_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=v),
        ],
    )
    _velocity_verlet_velocity_finalize_ptr_out_kernel_overload[v] = wp.overload(
        _velocity_verlet_velocity_finalize_ptr_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=v),
        ],
    )


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

    Example
    -------
    Single system usage::

        import warp as wp
        import numpy as np

        # Setup arrays
        positions = wp.array(np.random.randn(100, 3), dtype=wp.vec3d, device="cuda:0")
        velocities = wp.array(np.random.randn(100, 3), dtype=wp.vec3d, device="cuda:0")
        forces = wp.array(np.random.randn(100, 3), dtype=wp.vec3d, device="cuda:0")
        masses = wp.array(np.ones(100), dtype=wp.float64, device="cuda:0")
        dt = wp.array([0.001], dtype=wp.float64, device="cuda:0")

        # Perform position update
        velocity_verlet_position_update(positions, velocities, forces, masses, dt)

    Batched mode with batch_idx::

        # Setup for 3 systems with different atom counts
        batch_idx = wp.array([0]*30 + [1]*40 + [2]*30, dtype=wp.int32, device="cuda:0")
        dt_batch = wp.array([0.001, 0.002, 0.0015], dtype=wp.float64, device="cuda:0")

        velocity_verlet_position_update(
            positions, velocities, forces, masses, dt_batch, batch_idx=batch_idx
        )

    Batched mode with atom_ptr::

        # Setup for 3 systems: [0:30], [30:70], [70:100]
        atom_ptr = wp.array([0, 30, 70, 100], dtype=wp.int32, device="cuda:0")
        dt_batch = wp.array([0.001, 0.002, 0.0015], dtype=wp.float64, device="cuda:0")

        velocity_verlet_position_update(
            positions, velocities, forces, masses, dt_batch, atom_ptr=atom_ptr
        )

    See Also
    --------
    velocity_verlet_velocity_finalize : Complete the velocity update
    """
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")

    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]
    vec_dtype = positions.dtype

    if atom_ptr is not None:
        # Use atom_ptr mode - launch with dim=num_systems
        num_systems = atom_ptr.shape[0] - 1
        wp.launch(
            _velocity_verlet_position_update_ptr_kernel_overload[vec_dtype],
            dim=num_systems,
            inputs=[positions, velocities, forces, masses, atom_ptr, dt],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode - launch with dim=num_atoms
        wp.launch(
            _batch_velocity_verlet_position_update_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses, batch_idx, dt],
            device=device,
        )
    else:
        # Single system - launch with dim=num_atoms
        wp.launch(
            _velocity_verlet_position_update_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses, dt],
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

    Example
    -------
    Complete MD step with force recalculation::

        import warp as wp

        # After velocity_verlet_position_update(), recalculate forces
        forces_new = compute_forces(positions)  # User-defined force calculation

        # Finalize velocity update
        velocity_verlet_velocity_finalize(velocities, forces_new, masses, dt)

    Full MD loop::

        for step in range(num_steps):
            # Step 1: Position update
            velocity_verlet_position_update(positions, velocities, forces, masses, dt)

            # Step 2: Recalculate forces at new positions
            forces = compute_forces(positions)

            # Step 3: Finalize velocities
            velocity_verlet_velocity_finalize(velocities, forces, masses, dt)

    Batched mode::

        # With batch_idx
        velocity_verlet_velocity_finalize(
            velocities, forces_new, masses, dt_batch, batch_idx=batch_idx
        )

        # With atom_ptr
        velocity_verlet_velocity_finalize(
            velocities, forces_new, masses, dt_batch, atom_ptr=atom_ptr
        )

    See Also
    --------
    velocity_verlet_position_update : First step of velocity Verlet
    """
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")

    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    vec_dtype = velocities.dtype

    if atom_ptr is not None:
        # Use atom_ptr mode - launch with dim=num_systems
        num_systems = atom_ptr.shape[0] - 1
        wp.launch(
            _velocity_verlet_velocity_finalize_ptr_kernel_overload[vec_dtype],
            dim=num_systems,
            inputs=[velocities, forces_new, masses, atom_ptr, dt],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode - launch with dim=num_atoms
        wp.launch(
            _batch_velocity_verlet_velocity_finalize_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, batch_idx, dt],
            device=device,
        )
    else:
        # Single system - launch with dim=num_atoms
        wp.launch(
            _velocity_verlet_velocity_finalize_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, dt],
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
    positions_out: wp.array = None,
    velocities_out: wp.array = None,
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
    positions_out : wp.array, optional
        Output array for new positions. If None, allocated internally.
    velocities_out : wp.array, optional
        Output array for half-step velocities. If None, allocated internally.
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
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")

    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    # Allocate output arrays if needed
    if positions_out is None:
        positions_out = wp.empty_like(positions)
    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    vec_dtype = positions.dtype

    if atom_ptr is not None:
        # Use atom_ptr mode - launch with dim=num_systems
        num_systems = atom_ptr.shape[0] - 1
        wp.launch(
            _velocity_verlet_position_update_ptr_out_kernel_overload[vec_dtype],
            dim=num_systems,
            inputs=[
                positions,
                velocities,
                forces,
                masses,
                atom_ptr,
                dt,
                positions_out,
                velocities_out,
            ],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode - launch with dim=num_atoms
        wp.launch(
            _batch_velocity_verlet_position_update_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                velocities,
                forces,
                masses,
                batch_idx,
                dt,
                positions_out,
                velocities_out,
            ],
            device=device,
        )
    else:
        # Single system - launch with dim=num_atoms
        wp.launch(
            _velocity_verlet_position_update_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                velocities,
                forces,
                masses,
                dt,
                positions_out,
                velocities_out,
            ],
            device=device,
        )

    return positions_out, velocities_out


def velocity_verlet_velocity_finalize_out(
    velocities: wp.array,
    forces_new: wp.array,
    masses: wp.array,
    dt: wp.array,
    velocities_out: wp.array = None,
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
    velocities_out : wp.array, optional
        Output array for full-step velocities. If None, allocated internally.
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
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")

    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]

    # Allocate output array if needed
    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    vec_dtype = velocities.dtype

    if atom_ptr is not None:
        # Use atom_ptr mode - launch with dim=num_systems
        num_systems = atom_ptr.shape[0] - 1
        wp.launch(
            _velocity_verlet_velocity_finalize_ptr_out_kernel_overload[vec_dtype],
            dim=num_systems,
            inputs=[velocities, forces_new, masses, atom_ptr, dt, velocities_out],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode - launch with dim=num_atoms
        wp.launch(
            _batch_velocity_verlet_velocity_finalize_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, batch_idx, dt, velocities_out],
            device=device,
        )
    else:
        # Single system - launch with dim=num_atoms
        wp.launch(
            _velocity_verlet_velocity_finalize_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, dt, velocities_out],
            device=device,
        )

    return velocities_out
