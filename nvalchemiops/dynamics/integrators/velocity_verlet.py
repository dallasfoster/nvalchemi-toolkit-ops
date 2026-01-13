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
    # Fused kernels (single launch for full step)
    "velocity_verlet_step_fused",
    "velocity_verlet_step_fused_out",
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
        r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        v_half = v(t) + 0.5*a(t)*dt

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
        r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        v_half = v(t) + 0.5*a(t)*dt

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
# Kernel Overloads for Explicit Typing
# ==============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]  # Vector types

# Position update kernel overloads
_velocity_verlet_position_update_kernel_overload = {}
_batch_velocity_verlet_position_update_kernel_overload = {}
_velocity_verlet_position_update_out_kernel_overload = {}
_batch_velocity_verlet_position_update_out_kernel_overload = {}

# Velocity finalize kernel overloads
_velocity_verlet_velocity_finalize_kernel_overload = {}
_batch_velocity_verlet_velocity_finalize_kernel_overload = {}
_velocity_verlet_velocity_finalize_out_kernel_overload = {}
_batch_velocity_verlet_velocity_finalize_out_kernel_overload = {}

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
        System index for each atom. Shape (N,). Required for batched mode.
    device : str, optional
        Warp device. If None, inferred from positions.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    vec_dtype = positions.dtype
    if batch_idx is not None:
        wp.launch(
            _batch_velocity_verlet_position_update_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses, batch_idx, dt],
            device=device,
        )
    else:
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
            _batch_velocity_verlet_velocity_finalize_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, batch_idx, dt],
            device=device,
        )
    else:
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
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device. If None, inferred from positions.

    Returns
    -------
    tuple[wp.array, wp.array]
        (positions_out, velocities_out) - New positions and half-step velocities.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    # Allocate output arrays if needed
    if positions_out is None:
        positions_out = wp.empty_like(positions)
    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    vec_dtype = positions.dtype
    if batch_idx is not None:
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
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device. If None, inferred from velocities.

    Returns
    -------
    wp.array
        Full-step velocities v(t+dt).
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]

    # Allocate output array if needed
    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    vec_dtype = velocities.dtype
    if batch_idx is not None:
        wp.launch(
            _batch_velocity_verlet_velocity_finalize_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, batch_idx, dt, velocities_out],
            device=device,
        )
    else:
        wp.launch(
            _velocity_verlet_velocity_finalize_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces_new, masses, dt, velocities_out],
            device=device,
        )

    return velocities_out


# ==============================================================================
# Fused Kernels (Single Launch for Full Step)
# ==============================================================================
# These kernels combine position and velocity updates into a single launch,
# reducing kernel launch overhead. They require both old and new forces.


@wp.kernel
def _velocity_verlet_step_fused_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces_old: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
):
    """Fused velocity Verlet step (in-place).

    Performs complete velocity Verlet update in a single kernel:
        r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt²
        v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt

    Requires forces at both old and new positions.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. MODIFIED in-place.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. MODIFIED in-place.
    forces_old : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at current positions r(t).
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions r(t+dt).
    masses : wp.array
        Atomic masses.
    dt : wp.array
        Timestep.
    """
    atom_idx = wp.tid()

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    f_old = forces_old[atom_idx]
    f_new = forces_new[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[0]

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    # Old acceleration
    acc_old_x = f_old[0] * inv_mass
    acc_old_y = f_old[1] * inv_mass
    acc_old_z = f_old[2] * inv_mass

    # New acceleration
    acc_new_x = f_new[0] * inv_mass
    acc_new_y = f_new[1] * inv_mass
    acc_new_z = f_new[2] * inv_mass

    # Position update: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt²
    half_dt_sq = half_dt * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_old_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_old_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_old_z * half_dt_sq

    # Velocity update: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
    new_vel_x = vel[0] + (acc_old_x + acc_new_x) * half_dt
    new_vel_y = vel[1] + (acc_old_y + acc_new_y) * half_dt
    new_vel_z = vel[2] + (acc_old_z + acc_new_z) * half_dt

    positions[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities[atom_idx] = type(vel)(new_vel_x, new_vel_y, new_vel_z)


@wp.kernel
def _velocity_verlet_step_fused_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces_old: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Fused velocity Verlet step (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    f_old = forces_old[atom_idx]
    f_new = forces_new[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[0]

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    acc_old_x = f_old[0] * inv_mass
    acc_old_y = f_old[1] * inv_mass
    acc_old_z = f_old[2] * inv_mass

    acc_new_x = f_new[0] * inv_mass
    acc_new_y = f_new[1] * inv_mass
    acc_new_z = f_new[2] * inv_mass

    half_dt_sq = half_dt * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_old_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_old_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_old_z * half_dt_sq

    new_vel_x = vel[0] + (acc_old_x + acc_new_x) * half_dt
    new_vel_y = vel[1] + (acc_old_y + acc_new_y) * half_dt
    new_vel_z = vel[2] + (acc_old_z + acc_new_z) * half_dt

    positions_out[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities_out[atom_idx] = type(vel)(new_vel_x, new_vel_y, new_vel_z)


@wp.kernel
def _batch_velocity_verlet_step_fused_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces_old: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """Fused velocity Verlet step for batched systems (in-place).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    f_old = forces_old[atom_idx]
    f_new = forces_new[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[system_id]

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    acc_old_x = f_old[0] * inv_mass
    acc_old_y = f_old[1] * inv_mass
    acc_old_z = f_old[2] * inv_mass

    acc_new_x = f_new[0] * inv_mass
    acc_new_y = f_new[1] * inv_mass
    acc_new_z = f_new[2] * inv_mass

    half_dt_sq = half_dt * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_old_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_old_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_old_z * half_dt_sq

    new_vel_x = vel[0] + (acc_old_x + acc_new_x) * half_dt
    new_vel_y = vel[1] + (acc_old_y + acc_new_y) * half_dt
    new_vel_z = vel[2] + (acc_old_z + acc_new_z) * half_dt

    positions[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities[atom_idx] = type(vel)(new_vel_x, new_vel_y, new_vel_z)


@wp.kernel
def _batch_velocity_verlet_step_fused_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces_old: wp.array(dtype=Any),
    forces_new: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Fused velocity Verlet step for batched systems (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    f_old = forces_old[atom_idx]
    f_new = forces_new[atom_idx]
    mass = masses[atom_idx]
    dt_val = dt[system_id]

    inv_mass = type(mass)(1.0) / mass
    half_dt = type(dt_val)(0.5) * dt_val

    acc_old_x = f_old[0] * inv_mass
    acc_old_y = f_old[1] * inv_mass
    acc_old_z = f_old[2] * inv_mass

    acc_new_x = f_new[0] * inv_mass
    acc_new_y = f_new[1] * inv_mass
    acc_new_z = f_new[2] * inv_mass

    half_dt_sq = half_dt * dt_val
    new_pos_x = pos[0] + vel[0] * dt_val + acc_old_x * half_dt_sq
    new_pos_y = pos[1] + vel[1] * dt_val + acc_old_y * half_dt_sq
    new_pos_z = pos[2] + vel[2] * dt_val + acc_old_z * half_dt_sq

    new_vel_x = vel[0] + (acc_old_x + acc_new_x) * half_dt
    new_vel_y = vel[1] + (acc_old_y + acc_new_y) * half_dt
    new_vel_z = vel[2] + (acc_old_z + acc_new_z) * half_dt

    positions_out[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities_out[atom_idx] = type(vel)(new_vel_x, new_vel_y, new_vel_z)


# ==============================================================================
# Fused Kernel Overloads
# ==============================================================================

_velocity_verlet_step_fused_kernel_overload = {}
_velocity_verlet_step_fused_out_kernel_overload = {}
_batch_velocity_verlet_step_fused_kernel_overload = {}
_batch_velocity_verlet_step_fused_out_kernel_overload = {}

for t, v in zip(_T, _V):
    _velocity_verlet_step_fused_kernel_overload[v] = wp.overload(
        _velocity_verlet_step_fused_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=t),
        ],
    )
    _velocity_verlet_step_fused_out_kernel_overload[v] = wp.overload(
        _velocity_verlet_step_fused_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=t),
            wp.array(dtype=v),
            wp.array(dtype=v),
        ],
    )
    _batch_velocity_verlet_step_fused_kernel_overload[v] = wp.overload(
        _batch_velocity_verlet_step_fused_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=v),
            wp.array(dtype=t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
        ],
    )
    _batch_velocity_verlet_step_fused_out_kernel_overload[v] = wp.overload(
        _batch_velocity_verlet_step_fused_out_kernel,
        [
            wp.array(dtype=v),
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


# ==============================================================================
# Fused Functional Interface
# ==============================================================================


def velocity_verlet_step_fused(
    positions: wp.array,
    velocities: wp.array,
    forces_old: wp.array,
    forces_new: wp.array,
    masses: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Perform complete velocity Verlet step in single kernel (in-place).

    This fused kernel combines position and velocity updates, reducing
    kernel launch overhead. It requires forces at both old and new positions.

    Computes:
        r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt²
        v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,). MODIFIED in-place.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces_old : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at current positions r(t). Shape (N,).
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at new positions r(t+dt). Shape (N,).
    masses : wp.array
        Atomic masses. Shape (N,).
    dt : wp.array
        Timestep(s). Shape (1,) for single, (B,) for batched.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device.

    Notes
    -----
    This kernel assumes forces_new has already been computed at the
    new positions. The typical workflow is:

    1. Store current forces as forces_old
    2. Compute new positions (without this kernel)
    3. Compute forces_new at new positions
    4. Call this fused kernel

    For most use cases, the two-pass approach (position_update + velocity_finalize)
    is more flexible since forces are computed between the passes.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]
    vec_dtype = positions.dtype

    if batch_idx is not None:
        wp.launch(
            _batch_velocity_verlet_step_fused_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                velocities,
                forces_old,
                forces_new,
                masses,
                batch_idx,
                dt,
            ],
            device=device,
        )
    else:
        wp.launch(
            _velocity_verlet_step_fused_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces_old, forces_new, masses, dt],
            device=device,
        )


def velocity_verlet_step_fused_out(
    positions: wp.array,
    velocities: wp.array,
    forces_old: wp.array,
    forces_new: wp.array,
    masses: wp.array,
    dt: wp.array,
    positions_out: wp.array = None,
    velocities_out: wp.array = None,
    batch_idx: wp.array = None,
    device: str = None,
) -> tuple[wp.array, wp.array]:
    """
    Perform complete velocity Verlet step in single kernel (non-mutating).

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions at time t. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities at time t. Shape (N,).
    forces_old : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at positions r(t). Shape (N,).
    forces_new : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces at positions r(t+dt). Shape (N,).
    masses : wp.array
        Atomic masses. Shape (N,).
    dt : wp.array
        Timestep(s). Shape (1,) for single, (B,) for batched.
    positions_out : wp.array, optional
        Output array for positions. If None, allocated internally.
    velocities_out : wp.array, optional
        Output array for velocities. If None, allocated internally.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device.

    Returns
    -------
    tuple[wp.array, wp.array]
        (positions_out, velocities_out) at time t+dt.
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
            _batch_velocity_verlet_step_fused_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                velocities,
                forces_old,
                forces_new,
                masses,
                batch_idx,
                dt,
                positions_out,
                velocities_out,
            ],
            device=device,
        )
    else:
        wp.launch(
            _velocity_verlet_step_fused_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[
                positions,
                velocities,
                forces_old,
                forces_new,
                masses,
                dt,
                positions_out,
                velocities_out,
            ],
            device=device,
        )

    return positions_out, velocities_out
