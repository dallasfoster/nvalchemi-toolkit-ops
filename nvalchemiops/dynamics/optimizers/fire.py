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
FIRE and FIRE2 Optimizer Kernels
================================

GPU-accelerated Warp kernels for FIRE (Fast Inertial Relaxation Engine)
geometry optimization and its improved FIRE2 variant.

This module provides both mutating (in-place) and non-mutating versions
of each kernel for gradient tracking compatibility.

MATHEMATICAL FORMULATION
========================

FIRE uses MD-like dynamics with velocity modification:

Velocity mixing:

.. math::

    \\mathbf{v}(t) \\leftarrow (1-\\alpha) \\mathbf{v}(t)
                              + \\alpha \\hat{\\mathbf{F}}(t) |\\mathbf{v}(t)|

Adaptive parameter update based on power :math:`P = \\mathbf{F} \\cdot \\mathbf{v}`:

If :math:`P > 0` for :math:`N_{\\min}` consecutive steps:
    - :math:`\\Delta t \\leftarrow \\min(\\Delta t \\cdot f_{\\text{inc}}, \\Delta t_{\\max})`
    - :math:`\\alpha \\leftarrow \\alpha \\cdot f_\\alpha`

If :math:`P \\leq 0`:
    - :math:`\\mathbf{v} \\leftarrow 0`
    - :math:`\\Delta t \\leftarrow \\max(\\Delta t \\cdot f_{\\text{dec}}, \\Delta t_{\\min})`
    - :math:`\\alpha \\leftarrow \\alpha_{\\text{start}}`

TYPICAL FIRE PARAMETERS
=======================

- dt_start: 0.1 (initial timestep)
- dt_max: 1.0 (maximum timestep)
- dt_min: 0.01 (minimum timestep)
- n_min: 5 (minimum steps before dt increase)
- f_inc: 1.1 (timestep increase factor)
- f_dec: 0.5 (timestep decrease factor)
- alpha_start: 0.1 (initial mixing parameter)
- f_alpha: 0.99 (alpha decrease factor)

REFERENCES
==========

- Bitzek et al. (2006). Phys. Rev. Lett. 97, 170201 (FIRE)
- Guénolé et al. (2020). Comp. Mat. Sci. 175, 109584 (FIRE2)
"""

from __future__ import annotations

from typing import Any

import warp as wp

__all__ = [
    # FIRE - Mutating APIs
    "fire_compute_diagnostics",
    "fire_velocity_mix",
    "fire_md_step",
    "fire_reset_velocities",
    # FIRE - Non-mutating APIs
    "fire_velocity_mix_out",
    "fire_md_step_out",
    "fire_reset_velocities_out",
    # FIRE2 - Mutating APIs
    "fire2_velocity_update",
    "fire2_md_step",
    # FIRE2 - Non-mutating APIs
    "fire2_velocity_update_out",
    "fire2_md_step_out",
]


# ==============================================================================
# Utility Kernels
# ==============================================================================


# ==============================================================================
# Diagnostic Kernels
# ==============================================================================


@wp.kernel
def _fire_compute_diagnostics_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    power: wp.array(dtype=wp.float64),
    force_norm_sq: wp.array(dtype=wp.float64),
    velocity_norm_sq: wp.array(dtype=wp.float64),
):
    """Compute FIRE power P = F·v and L2 norms.

    Accumulates:
    - power = sum_i(F_i · v_i)
    - force_norm_sq = sum_i(F_i · F_i)
    - velocity_norm_sq = sum_i(v_i · v_i)

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    f = forces[atom_idx]
    v = velocities[atom_idx]

    # Compute in float64 for numerical stability
    f_dot_v = (
        wp.float64(f[0]) * wp.float64(v[0])
        + wp.float64(f[1]) * wp.float64(v[1])
        + wp.float64(f[2]) * wp.float64(v[2])
    )

    f_dot_f = (
        wp.float64(f[0]) * wp.float64(f[0])
        + wp.float64(f[1]) * wp.float64(f[1])
        + wp.float64(f[2]) * wp.float64(f[2])
    )

    v_dot_v = (
        wp.float64(v[0]) * wp.float64(v[0])
        + wp.float64(v[1]) * wp.float64(v[1])
        + wp.float64(v[2]) * wp.float64(v[2])
    )

    wp.atomic_add(power, 0, f_dot_v)
    wp.atomic_add(force_norm_sq, 0, f_dot_f)
    wp.atomic_add(velocity_norm_sq, 0, v_dot_v)


@wp.kernel
def _batch_fire_compute_diagnostics_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    power: wp.array(dtype=wp.float64),
    force_norm_sq: wp.array(dtype=wp.float64),
    velocity_norm_sq: wp.array(dtype=wp.float64),
):
    """Compute per-system FIRE diagnostics.

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    f = forces[atom_idx]
    v = velocities[atom_idx]

    f_dot_v = (
        wp.float64(f[0]) * wp.float64(v[0])
        + wp.float64(f[1]) * wp.float64(v[1])
        + wp.float64(f[2]) * wp.float64(v[2])
    )

    f_dot_f = (
        wp.float64(f[0]) * wp.float64(f[0])
        + wp.float64(f[1]) * wp.float64(f[1])
        + wp.float64(f[2]) * wp.float64(f[2])
    )

    v_dot_v = (
        wp.float64(v[0]) * wp.float64(v[0])
        + wp.float64(v[1]) * wp.float64(v[1])
        + wp.float64(v[2]) * wp.float64(v[2])
    )

    wp.atomic_add(power, system_id, f_dot_v)
    wp.atomic_add(force_norm_sq, system_id, f_dot_f)
    wp.atomic_add(velocity_norm_sq, system_id, v_dot_v)


# ==============================================================================
# Mutating Kernels
# ==============================================================================


@wp.kernel
def _fire_velocity_mix_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    force_norm: wp.array(dtype=Any),
    velocity_norm: wp.array(dtype=Any),
):
    """Apply FIRE velocity mixing (in-place).

    v = (1 - alpha) * v + alpha * F_hat * |v|

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    f = forces[atom_idx]
    v = velocities[atom_idx]
    f_norm = force_norm[0]
    v_norm = velocity_norm[0]
    alpha_val = alpha[0]

    one_minus_alpha = type(alpha_val)(1.0) - alpha_val
    eps = type(f_norm)(1e-10)

    if f_norm > eps:
        velocities[atom_idx] = v * one_minus_alpha + alpha_val * v_norm * f / f_norm
    else:
        velocities[atom_idx] = v * one_minus_alpha


@wp.kernel
def _fire_md_step_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
):
    """FIRE MD-like position/velocity update (in-place).

    v = v + (dt/m) * F
    r = r + dt * v

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

    # Velocity update
    new_vel_x = vel[0] + force[0] * inv_mass * dt_val
    new_vel_y = vel[1] + force[1] * inv_mass * dt_val
    new_vel_z = vel[2] + force[2] * inv_mass * dt_val

    # Position update (using new velocity)
    new_pos_x = pos[0] + new_vel_x * dt_val
    new_pos_y = pos[1] + new_vel_y * dt_val
    new_pos_z = pos[2] + new_vel_z * dt_val

    positions[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities[atom_idx] = type(vel)(new_vel_x, new_vel_y, new_vel_z)


@wp.kernel
def _fire_reset_velocities_kernel(
    velocities: wp.array(dtype=Any),
):
    """Reset all velocities to zero (in-place).

    Called when P <= 0 to restart optimization.

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    vel = velocities[atom_idx]
    velocities[atom_idx] = vel * type(vel[0])(0.0)


# ==============================================================================
# Non-Mutating Kernels
# ==============================================================================


@wp.kernel
def _fire_velocity_mix_out_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    force_norm: wp.array(dtype=Any),
    velocity_norm: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Apply FIRE velocity mixing (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    f = forces[atom_idx]
    v = velocities[atom_idx]
    f_norm = force_norm[0]
    v_norm = velocity_norm[0]
    alpha_val = alpha[0]

    one_minus_alpha = type(alpha_val)(1.0) - alpha_val
    eps = type(f_norm)(1e-10)

    if f_norm > eps:
        velocities_out[atom_idx] = v * one_minus_alpha + alpha_val * v_norm * f / f_norm
    else:
        velocities_out[atom_idx] = v * one_minus_alpha


@wp.kernel
def _fire_md_step_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """FIRE MD-like position/velocity update (non-mutating).

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

    # Velocity update
    new_vel_x = vel[0] + force[0] * inv_mass * dt_val
    new_vel_y = vel[1] + force[1] * inv_mass * dt_val
    new_vel_z = vel[2] + force[2] * inv_mass * dt_val

    # Position update
    new_pos_x = pos[0] + new_vel_x * dt_val
    new_pos_y = pos[1] + new_vel_y * dt_val
    new_pos_z = pos[2] + new_vel_z * dt_val

    positions_out[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities_out[atom_idx] = type(vel)(new_vel_x, new_vel_y, new_vel_z)


@wp.kernel
def _fire_reset_velocities_out_kernel(
    velocities: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Reset all velocities to zero (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    vel = velocities[atom_idx]
    velocities_out[atom_idx] = vel * type(vel[0])(0.0)


# ==============================================================================
# Batched Mutating Kernels
# ==============================================================================


@wp.kernel
def _batch_fire_velocity_mix_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    force_norm: wp.array(dtype=Any),
    velocity_norm: wp.array(dtype=Any),
):
    """Batched FIRE velocity mixing (in-place).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    f = forces[atom_idx]
    v = velocities[atom_idx]
    f_norm = force_norm[system_id]
    v_norm = velocity_norm[system_id]
    alpha_val = alpha[system_id]

    one_minus_alpha = type(alpha_val)(1.0) - alpha_val
    eps = type(f_norm)(1e-10)

    if f_norm > eps:
        velocities[atom_idx] = v * one_minus_alpha + alpha_val * v_norm * f / f_norm
    else:
        velocities[atom_idx] = v * one_minus_alpha


@wp.kernel
def _batch_fire_md_step_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
):
    """Batched FIRE MD step (in-place).

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

    inv_mass = type(mass)(1.0) / mass

    new_vel_x = vel[0] + force[0] * inv_mass * dt_val
    new_vel_y = vel[1] + force[1] * inv_mass * dt_val
    new_vel_z = vel[2] + force[2] * inv_mass * dt_val

    new_pos_x = pos[0] + new_vel_x * dt_val
    new_pos_y = pos[1] + new_vel_y * dt_val
    new_pos_z = pos[2] + new_vel_z * dt_val

    positions[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities[atom_idx] = type(vel)(new_vel_x, new_vel_y, new_vel_z)


@wp.kernel
def _batch_fire_reset_velocities_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.int32),
):
    """Reset velocities for systems where reset_mask == 1 (in-place).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    if reset_mask[system_id] == 1:
        vel = velocities[atom_idx]
        velocities[atom_idx] = vel * type(vel[0])(0.0)


# ==============================================================================
# Batched Non-Mutating Kernels
# ==============================================================================


@wp.kernel
def _batch_fire_velocity_mix_out_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    force_norm: wp.array(dtype=Any),
    velocity_norm: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Batched FIRE velocity mixing (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    f = forces[atom_idx]
    v = velocities[atom_idx]
    f_norm = force_norm[system_id]
    v_norm = velocity_norm[system_id]
    alpha_val = alpha[system_id]

    one_minus_alpha = type(alpha_val)(1.0) - alpha_val
    eps = type(f_norm)(1e-10)

    if f_norm > eps:
        velocities_out[atom_idx] = v * one_minus_alpha + alpha_val * v_norm * f / f_norm
    else:
        velocities_out[atom_idx] = v * one_minus_alpha


@wp.kernel
def _batch_fire_md_step_out_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    positions_out: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Batched FIRE MD step (non-mutating).

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

    inv_mass = type(mass)(1.0) / mass

    new_vel_x = vel[0] + force[0] * inv_mass * dt_val
    new_vel_y = vel[1] + force[1] * inv_mass * dt_val
    new_vel_z = vel[2] + force[2] * inv_mass * dt_val

    new_pos_x = pos[0] + new_vel_x * dt_val
    new_pos_y = pos[1] + new_vel_y * dt_val
    new_pos_z = pos[2] + new_vel_z * dt_val

    positions_out[atom_idx] = type(pos)(new_pos_x, new_pos_y, new_pos_z)
    velocities_out[atom_idx] = type(vel)(new_vel_x, new_vel_y, new_vel_z)


@wp.kernel
def _batch_fire_reset_velocities_out_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.int32),
    velocities_out: wp.array(dtype=Any),
):
    """Reset velocities for systems where reset_mask == 1 (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]
    vel = velocities[atom_idx]

    if reset_mask[system_id] == 1:
        velocities_out[atom_idx] = vel * type(vel[0])(0.0)
    else:
        velocities_out[atom_idx] = vel


# ==============================================================================
# Kernel Overloads for Explicit Typing
# ==============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]      # Vector types

# Diagnostic kernel overloads
_fire_compute_diagnostics_kernel_overload = {}
_batch_fire_compute_diagnostics_kernel_overload = {}

# Velocity mix kernel overloads
_fire_velocity_mix_kernel_overload = {}
_batch_fire_velocity_mix_kernel_overload = {}
_fire_velocity_mix_out_kernel_overload = {}
_batch_fire_velocity_mix_out_kernel_overload = {}

# MD step kernel overloads
_fire_md_step_kernel_overload = {}
_batch_fire_md_step_kernel_overload = {}
_fire_md_step_out_kernel_overload = {}
_batch_fire_md_step_out_kernel_overload = {}

# Reset velocities kernel overloads
_fire_reset_velocities_kernel_overload = {}
_batch_fire_reset_velocities_kernel_overload = {}
_fire_reset_velocities_out_kernel_overload = {}
_batch_fire_reset_velocities_out_kernel_overload = {}

for t, v in zip(_T, _V):
    # Diagnostic kernels (5 args: velocities, forces, power, force_norm_sq, velocity_norm_sq)
    _fire_compute_diagnostics_kernel_overload[v] = wp.overload(
        _fire_compute_diagnostics_kernel,
        [wp.array(dtype=v), wp.array(dtype=v),
         wp.array(dtype=wp.float64), wp.array(dtype=wp.float64), wp.array(dtype=wp.float64)],
    )
    # Batch diagnostic (6 args: velocities, forces, batch_idx, power, force_norm_sq, velocity_norm_sq)
    _batch_fire_compute_diagnostics_kernel_overload[v] = wp.overload(
        _batch_fire_compute_diagnostics_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=wp.int32),
         wp.array(dtype=wp.float64), wp.array(dtype=wp.float64), wp.array(dtype=wp.float64)],
    )

    # Velocity mix kernels (5 args: velocities, forces, alpha, force_norm, velocity_norm)
    _fire_velocity_mix_kernel_overload[v] = wp.overload(
        _fire_velocity_mix_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t)],
    )
    # Batch velocity mix (6 args: velocities, forces, batch_idx, alpha, force_norm, velocity_norm)
    _batch_fire_velocity_mix_kernel_overload[v] = wp.overload(
        _batch_fire_velocity_mix_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=wp.int32),
         wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t)],
    )
    # Velocity mix out (6 args: velocities, forces, alpha, force_norm, velocity_norm, velocities_out)
    _fire_velocity_mix_out_kernel_overload[v] = wp.overload(
        _fire_velocity_mix_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=v)],
    )
    # Batch velocity mix out (7 args)
    _batch_fire_velocity_mix_out_kernel_overload[v] = wp.overload(
        _batch_fire_velocity_mix_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=wp.int32),
         wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=t), wp.array(dtype=v)],
    )

    # MD step kernels
    _fire_md_step_kernel_overload[v] = wp.overload(
        _fire_md_step_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t)],
    )
    _batch_fire_md_step_kernel_overload[v] = wp.overload(
        _batch_fire_md_step_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t),
         wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )
    _fire_md_step_out_kernel_overload[v] = wp.overload(
        _fire_md_step_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=t),
         wp.array(dtype=v), wp.array(dtype=v)],
    )
    _batch_fire_md_step_out_kernel_overload[v] = wp.overload(
        _batch_fire_md_step_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t),
         wp.array(dtype=wp.int32), wp.array(dtype=t), wp.array(dtype=v), wp.array(dtype=v)],
    )

    # Reset velocities kernels
    _fire_reset_velocities_kernel_overload[v] = wp.overload(
        _fire_reset_velocities_kernel,
        [wp.array(dtype=v)],
    )
    _batch_fire_reset_velocities_kernel_overload[v] = wp.overload(
        _batch_fire_reset_velocities_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=wp.int32)],
    )
    _fire_reset_velocities_out_kernel_overload[v] = wp.overload(
        _fire_reset_velocities_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v)],
    )
    _batch_fire_reset_velocities_out_kernel_overload[v] = wp.overload(
        _batch_fire_reset_velocities_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=wp.int32), wp.array(dtype=v)],
    )


# ==============================================================================
# Functional Interface - Diagnostics
# ==============================================================================


def fire_compute_diagnostics(
    velocities: wp.array,
    forces: wp.array,
    power: wp.array = None,
    force_norm_sq: wp.array = None,
    velocity_norm_sq: wp.array = None,
    batch_idx: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> tuple[wp.array, wp.array, wp.array]:
    """
    Compute FIRE diagnostic quantities: power and norms.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    power : wp.array(dtype=wp.float64), optional
        Output for power P = F·v. Shape (1,) or (B,). If None, allocated.
    force_norm_sq : wp.array(dtype=wp.float64), optional
        Output for |F|^2. Shape (1,) or (B,). If None, allocated.
    velocity_norm_sq : wp.array(dtype=wp.float64), optional
        Output for |v|^2. Shape (1,) or (B,). If None, allocated.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    num_systems : int, optional
        Number of systems for batched mode. Default 1.
    device : str, optional
        Warp device. If None, inferred from velocities.

    Returns
    -------
    tuple[wp.array, wp.array, wp.array]
        (power, force_norm_sq, velocity_norm_sq)
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    is_batched = batch_idx is not None

    # Allocate outputs if needed
    if power is None:
        power = wp.zeros(num_systems, dtype=wp.float64, device=device)
    else:
        power.zero_()
    if force_norm_sq is None:
        force_norm_sq = wp.zeros(num_systems, dtype=wp.float64, device=device)
    else:
        force_norm_sq.zero_()
    if velocity_norm_sq is None:
        velocity_norm_sq = wp.zeros(num_systems, dtype=wp.float64, device=device)
    else:
        velocity_norm_sq.zero_()

    vec_dtype = velocities.dtype
    if is_batched:
        wp.launch(
            _batch_fire_compute_diagnostics_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces, batch_idx, power, force_norm_sq, velocity_norm_sq],
            device=device,
        )
    else:
        wp.launch(
            _fire_compute_diagnostics_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces, power, force_norm_sq, velocity_norm_sq],
            device=device,
        )

    return power, force_norm_sq, velocity_norm_sq


# ==============================================================================
# Functional Interface - Mutating
# ==============================================================================


def fire_velocity_mix(
    velocities: wp.array,
    forces: wp.array,
    alpha: wp.array,
    force_norm: wp.array,
    velocity_norm: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Apply FIRE velocity mixing (in-place).

    v = (1 - alpha) * v + alpha * F_hat * |v|

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    alpha : wp.array(dtype=wp.float32 or wp.float64)
        FIRE mixing parameter. Shape (1,) or (B,).
    force_norm : wp.array
        L2 norm of forces. Shape (1,) or (B,).
    velocity_norm : wp.array
        L2 norm of velocities. Shape (1,) or (B,).
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]

    vec_dtype = velocities.dtype
    if batch_idx is not None:
        wp.launch(
            _batch_fire_velocity_mix_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces, batch_idx, alpha, force_norm, velocity_norm],
            device=device,
        )
    else:
        wp.launch(
            _fire_velocity_mix_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces, alpha, force_norm, velocity_norm],
            device=device,
        )


def fire_md_step(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Perform FIRE MD-like position/velocity update (in-place).

    v = v + (dt/m) * F
    r = r + dt * v

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
        Timestep(s). Shape (1,) or (B,).
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device.
    """
    if device is None:
        device = positions.device

    num_atoms = positions.shape[0]

    vec_dtype = positions.dtype
    if batch_idx is not None:
        wp.launch(
            _batch_fire_md_step_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses, batch_idx, dt],
            device=device,
        )
    else:
        wp.launch(
            _fire_md_step_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses, dt],
            device=device,
        )


def fire_reset_velocities(
    velocities: wp.array,
    batch_idx: wp.array = None,
    reset_mask: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> None:
    """
    Reset velocities to zero (in-place).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    reset_mask : wp.array(dtype=wp.int32), optional
        For batched mode, mask indicating which systems to reset.
        If None for batched, all systems are reset (requires num_systems).
    num_systems : int, optional
        Number of systems for batched mode. Required if reset_mask is None
        and batch_idx is provided. Default: 1.
    device : str, optional
        Warp device.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]

    if batch_idx is not None:
        if reset_mask is None:
            reset_mask = wp.ones(num_systems, dtype=wp.int32, device=device)
        vec_dtype = velocities.dtype
        wp.launch(
            _batch_fire_reset_velocities_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, batch_idx, reset_mask],
            device=device,
        )
    else:
        vec_dtype = velocities.dtype
        wp.launch(
            _fire_reset_velocities_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities],
            device=device,
        )


# ==============================================================================
# Functional Interface - Non-Mutating
# ==============================================================================


def fire_velocity_mix_out(
    velocities: wp.array,
    forces: wp.array,
    alpha: wp.array,
    force_norm: wp.array,
    velocity_norm: wp.array,
    velocities_out: wp.array = None,
    batch_idx: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Apply FIRE velocity mixing (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    alpha : wp.array
        FIRE mixing parameter. Shape (1,) or (B,).
    force_norm : wp.array
        L2 norm of forces. Shape (1,) or (B,).
    velocity_norm : wp.array
        L2 norm of velocities. Shape (1,) or (B,).
    velocities_out : wp.array, optional
        Output array. If None, allocated internally.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Mixed velocities.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]

    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    vec_dtype = velocities.dtype
    if batch_idx is not None:
        wp.launch(
            _batch_fire_velocity_mix_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces, batch_idx, alpha, force_norm, velocity_norm,
                    velocities_out],
            device=device,
        )
    else:
        wp.launch(
            _fire_velocity_mix_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces, alpha, force_norm, velocity_norm, velocities_out],
            device=device,
        )

    return velocities_out


def fire_md_step_out(
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
    Perform FIRE MD-like position/velocity update (non-mutating).

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
    dt : wp.array
        Timestep(s). Shape (1,) or (B,).
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
        (positions_out, velocities_out)
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
            _batch_fire_md_step_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses, batch_idx, dt,
                    positions_out, velocities_out],
            device=device,
        )
    else:
        wp.launch(
            _fire_md_step_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[positions, velocities, forces, masses, dt,
                    positions_out, velocities_out],
            device=device,
        )

    return positions_out, velocities_out


def fire_reset_velocities_out(
    velocities: wp.array,
    velocities_out: wp.array = None,
    batch_idx: wp.array = None,
    reset_mask: wp.array = None,
    num_systems: int = 1,
    device: str = None,
) -> wp.array:
    """
    Reset velocities to zero (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    velocities_out : wp.array, optional
        Output array. If None, allocated internally.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    reset_mask : wp.array(dtype=wp.int32), optional
        For batched mode, mask indicating which systems to reset.
        If None for batched, all systems are reset (requires num_systems).
    num_systems : int, optional
        Number of systems for batched mode. Required if reset_mask is None
        and batch_idx is provided. Default: 1.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Velocities (zeroed where applicable).
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]

    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    if batch_idx is not None:
        if reset_mask is None:
            reset_mask = wp.ones(num_systems, dtype=wp.int32, device=device)
        vec_dtype = velocities.dtype
        wp.launch(
            _batch_fire_reset_velocities_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, batch_idx, reset_mask, velocities_out],
            device=device,
        )
    else:
        vec_dtype = velocities.dtype
        wp.launch(
            _fire_reset_velocities_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, velocities_out],
            device=device,
        )

    return velocities_out


# ==============================================================================
# FIRE2 Kernels
# ==============================================================================
# FIRE2 (Guénolé et al. 2020) improves on FIRE by projecting velocities onto
# the force direction before mixing. This provides better convergence for
# stiff systems and avoids oscillatory behavior.
#
# Algorithm:
#   If P = F·v > 0:
#       1. Project velocity onto force: v = (P/|F|²)F
#       2. Mix: v = (1-α)v + α(|v|/|F|)F
#   If P ≤ 0:
#       Reset v = 0


@wp.kernel
def _fire2_velocity_update_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    force_norm_sq: wp.array(dtype=wp.float64),
    power: wp.array(dtype=wp.float64),
):
    """FIRE2 velocity update with projection (in-place).

    If power > 0:
        1. Project velocity onto force direction
        2. Mix with force direction
    If power <= 0:
        Reset velocity to zero

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms.
    alpha : wp.array
        Mixing parameter (scalar array).
    force_norm_sq : wp.array(dtype=wp.float64)
        |F|² (pre-computed sum over all atoms).
    power : wp.array(dtype=wp.float64)
        P = F·v (pre-computed sum over all atoms).
    """
    atom_idx = wp.tid()

    v = velocities[atom_idx]
    f = forces[atom_idx]
    a = alpha[0]
    f_norm_sq = force_norm_sq[0]
    p = power[0]

    if p > wp.float64(0.0) and f_norm_sq > wp.float64(1e-30):
        # Compute velocity norm before projection
        v_norm_sq = (
            wp.float64(v[0]) * wp.float64(v[0])
            + wp.float64(v[1]) * wp.float64(v[1])
            + wp.float64(v[2]) * wp.float64(v[2])
        )
        v_norm = wp.sqrt(v_norm_sq)

        # Force norm
        f_norm = wp.sqrt(f_norm_sq)

        # FIRE2 mixing with projection
        proj_scale = p / f_norm_sq
        mix_scale = wp.float64(a) * v_norm / f_norm
        one_minus_alpha = wp.float64(1.0) - wp.float64(a)

        # v = (1-α) * proj_v + α * (|v|/|F|) * F
        new_vx = type(v[0])(one_minus_alpha * proj_scale * wp.float64(f[0]) + mix_scale * wp.float64(f[0]))
        new_vy = type(v[1])(one_minus_alpha * proj_scale * wp.float64(f[1]) + mix_scale * wp.float64(f[1]))
        new_vz = type(v[2])(one_minus_alpha * proj_scale * wp.float64(f[2]) + mix_scale * wp.float64(f[2]))

        velocities[atom_idx] = type(v)(new_vx, new_vy, new_vz)
    else:
        # Power is negative or zero: reset velocity
        velocities[atom_idx] = v * type(v[0])(0.0)


@wp.kernel
def _fire2_velocity_update_out_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    force_norm_sq: wp.array(dtype=wp.float64),
    power: wp.array(dtype=wp.float64),
    velocities_out: wp.array(dtype=Any),
):
    """FIRE2 velocity update with projection (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()

    v = velocities[atom_idx]
    f = forces[atom_idx]
    a = alpha[0]
    f_norm_sq = force_norm_sq[0]
    p = power[0]

    if p > wp.float64(0.0) and f_norm_sq > wp.float64(1e-30):
        v_norm_sq = (
            wp.float64(v[0]) * wp.float64(v[0])
            + wp.float64(v[1]) * wp.float64(v[1])
            + wp.float64(v[2]) * wp.float64(v[2])
        )
        v_norm = wp.sqrt(v_norm_sq)
        f_norm = wp.sqrt(f_norm_sq)

        proj_scale = p / f_norm_sq
        mix_scale = wp.float64(a) * v_norm / f_norm
        one_minus_alpha = wp.float64(1.0) - wp.float64(a)

        new_vx = type(v[0])(one_minus_alpha * proj_scale * wp.float64(f[0]) + mix_scale * wp.float64(f[0]))
        new_vy = type(v[1])(one_minus_alpha * proj_scale * wp.float64(f[1]) + mix_scale * wp.float64(f[1]))
        new_vz = type(v[2])(one_minus_alpha * proj_scale * wp.float64(f[2]) + mix_scale * wp.float64(f[2]))

        velocities_out[atom_idx] = type(v)(new_vx, new_vy, new_vz)
    else:
        velocities_out[atom_idx] = v * type(v[0])(0.0)


@wp.kernel
def _batch_fire2_velocity_update_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    force_norm_sq: wp.array(dtype=wp.float64),
    power: wp.array(dtype=wp.float64),
):
    """FIRE2 velocity update for batched systems (in-place).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    v = velocities[atom_idx]
    f = forces[atom_idx]
    a = alpha[system_id]
    f_norm_sq = force_norm_sq[system_id]
    p = power[system_id]

    if p > wp.float64(0.0) and f_norm_sq > wp.float64(1e-30):
        v_norm_sq = (
            wp.float64(v[0]) * wp.float64(v[0])
            + wp.float64(v[1]) * wp.float64(v[1])
            + wp.float64(v[2]) * wp.float64(v[2])
        )
        v_norm = wp.sqrt(v_norm_sq)
        f_norm = wp.sqrt(f_norm_sq)

        proj_scale = p / f_norm_sq
        mix_scale = wp.float64(a) * v_norm / f_norm
        one_minus_alpha = wp.float64(1.0) - wp.float64(a)

        new_vx = type(v[0])(one_minus_alpha * proj_scale * wp.float64(f[0]) + mix_scale * wp.float64(f[0]))
        new_vy = type(v[1])(one_minus_alpha * proj_scale * wp.float64(f[1]) + mix_scale * wp.float64(f[1]))
        new_vz = type(v[2])(one_minus_alpha * proj_scale * wp.float64(f[2]) + mix_scale * wp.float64(f[2]))

        velocities[atom_idx] = type(v)(new_vx, new_vy, new_vz)
    else:
        velocities[atom_idx] = v * type(v[0])(0.0)


@wp.kernel
def _batch_fire2_velocity_update_out_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    force_norm_sq: wp.array(dtype=wp.float64),
    power: wp.array(dtype=wp.float64),
    velocities_out: wp.array(dtype=Any),
):
    """FIRE2 velocity update for batched systems (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    v = velocities[atom_idx]
    f = forces[atom_idx]
    a = alpha[system_id]
    f_norm_sq = force_norm_sq[system_id]
    p = power[system_id]

    if p > wp.float64(0.0) and f_norm_sq > wp.float64(1e-30):
        v_norm_sq = (
            wp.float64(v[0]) * wp.float64(v[0])
            + wp.float64(v[1]) * wp.float64(v[1])
            + wp.float64(v[2]) * wp.float64(v[2])
        )
        v_norm = wp.sqrt(v_norm_sq)
        f_norm = wp.sqrt(f_norm_sq)

        proj_scale = p / f_norm_sq
        mix_scale = wp.float64(a) * v_norm / f_norm
        one_minus_alpha = wp.float64(1.0) - wp.float64(a)

        new_vx = type(v[0])(one_minus_alpha * proj_scale * wp.float64(f[0]) + mix_scale * wp.float64(f[0]))
        new_vy = type(v[1])(one_minus_alpha * proj_scale * wp.float64(f[1]) + mix_scale * wp.float64(f[1]))
        new_vz = type(v[2])(one_minus_alpha * proj_scale * wp.float64(f[2]) + mix_scale * wp.float64(f[2]))

        velocities_out[atom_idx] = type(v)(new_vx, new_vy, new_vz)
    else:
        velocities_out[atom_idx] = v * type(v[0])(0.0)


# ==============================================================================
# FIRE2 Kernel Overloads
# ==============================================================================

_fire2_velocity_update_kernel_overload = {}
_fire2_velocity_update_out_kernel_overload = {}
_batch_fire2_velocity_update_kernel_overload = {}
_batch_fire2_velocity_update_out_kernel_overload = {}

for t, v in zip(_T, _V):
    _fire2_velocity_update_kernel_overload[v] = wp.overload(
        _fire2_velocity_update_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t),
         wp.array(dtype=wp.float64), wp.array(dtype=wp.float64)],
    )
    _fire2_velocity_update_out_kernel_overload[v] = wp.overload(
        _fire2_velocity_update_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=t),
         wp.array(dtype=wp.float64), wp.array(dtype=wp.float64), wp.array(dtype=v)],
    )
    _batch_fire2_velocity_update_kernel_overload[v] = wp.overload(
        _batch_fire2_velocity_update_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=wp.int32),
         wp.array(dtype=t), wp.array(dtype=wp.float64), wp.array(dtype=wp.float64)],
    )
    _batch_fire2_velocity_update_out_kernel_overload[v] = wp.overload(
        _batch_fire2_velocity_update_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=v), wp.array(dtype=wp.int32),
         wp.array(dtype=t), wp.array(dtype=wp.float64), wp.array(dtype=wp.float64),
         wp.array(dtype=v)],
    )


# ==============================================================================
# FIRE2 Functional Interface
# ==============================================================================


def fire2_velocity_update(
    velocities: wp.array,
    forces: wp.array,
    alpha: wp.array,
    force_norm_sq: wp.array,
    power: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Apply FIRE2 velocity update with projection (in-place).

    FIRE2 improves on FIRE by projecting velocities onto the force direction
    before mixing, providing better convergence for stiff systems.

    If power > 0:
        1. Project velocity onto force direction: v_proj = (P/|F|²)F
        2. Mix: v = (1-α)v_proj + α(|v|/|F|)F
    If power ≤ 0:
        Reset velocity to zero

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    alpha : wp.array
        FIRE mixing parameter. Shape (1,) or (B,) for batched.
    force_norm_sq : wp.array(dtype=wp.float64)
        Pre-computed |F|² = sum_i(F_i·F_i). Shape (1,) or (B,).
    power : wp.array(dtype=wp.float64)
        Pre-computed P = sum_i(F_i·v_i). Shape (1,) or (B,).
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device.

    Notes
    -----
    Use fire_compute_diagnostics() to compute force_norm_sq and power.

    References
    ----------
    Guénolé et al. (2020). Comp. Mat. Sci. 175, 109584
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]
    vec_dtype = velocities.dtype

    if batch_idx is not None:
        wp.launch(
            _batch_fire2_velocity_update_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces, batch_idx, alpha, force_norm_sq, power],
            device=device,
        )
    else:
        wp.launch(
            _fire2_velocity_update_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces, alpha, force_norm_sq, power],
            device=device,
        )


def fire2_velocity_update_out(
    velocities: wp.array,
    forces: wp.array,
    alpha: wp.array,
    force_norm_sq: wp.array,
    power: wp.array,
    velocities_out: wp.array = None,
    batch_idx: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Apply FIRE2 velocity update with projection (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    alpha : wp.array
        FIRE mixing parameter. Shape (1,) or (B,) for batched.
    force_norm_sq : wp.array(dtype=wp.float64)
        Pre-computed |F|². Shape (1,) or (B,).
    power : wp.array(dtype=wp.float64)
        Pre-computed P = F·v. Shape (1,) or (B,).
    velocities_out : wp.array, optional
        Output array. If None, allocated internally.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Updated velocities.
    """
    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]

    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    vec_dtype = velocities.dtype

    if batch_idx is not None:
        wp.launch(
            _batch_fire2_velocity_update_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces, batch_idx, alpha, force_norm_sq, power,
                    velocities_out],
            device=device,
        )
    else:
        wp.launch(
            _fire2_velocity_update_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, forces, alpha, force_norm_sq, power, velocities_out],
            device=device,
        )

    return velocities_out


def fire2_md_step(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    batch_idx: wp.array = None,
    device: str = None,
) -> None:
    """
    Perform FIRE2 MD step (velocity Verlet position update) in-place.

    This is identical to the FIRE MD step - a standard velocity Verlet
    position update. The FIRE2 difference is in the velocity update,
    not the MD step itself.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,). MODIFIED in-place.
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    masses : wp.array
        Atomic masses. Shape (N,).
    dt : wp.array
        Timestep. Shape (1,) or (B,) for batched.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. Required for batched mode.
    device : str, optional
        Warp device.
    """
    # FIRE2 uses the same MD step as FIRE
    fire_md_step(positions, velocities, forces, masses, dt, batch_idx, device)


def fire2_md_step_out(
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
    Perform FIRE2 MD step (velocity Verlet position update) non-mutating.

    Parameters
    ----------
    positions : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic positions. Shape (N,).
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    forces : wp.array(dtype=wp.vec3f or wp.vec3d)
        Forces on atoms. Shape (N,).
    masses : wp.array
        Atomic masses. Shape (N,).
    dt : wp.array
        Timestep. Shape (1,) or (B,) for batched.
    positions_out : wp.array, optional
        Output array for positions.
    velocities_out : wp.array, optional
        Output array for velocities.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom.
    device : str, optional
        Warp device.

    Returns
    -------
    tuple[wp.array, wp.array]
        (positions_out, velocities_out)
    """
    # FIRE2 uses the same MD step as FIRE
    return fire_md_step_out(
        positions, velocities, forces, masses, dt,
        positions_out, velocities_out, batch_idx, device
    )
