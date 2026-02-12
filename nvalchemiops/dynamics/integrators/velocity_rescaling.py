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
Velocity Rescaling Thermostat
=============================

GPU-accelerated Warp kernels for simple velocity rescaling thermostat.

This module provides both mutating (in-place) and non-mutating versions
of each kernel for gradient tracking compatibility.

MATHEMATICAL FORMULATION
========================

Simple velocity rescaling:

.. math::

    \\mathbf{v}_i \\leftarrow \\mathbf{v}_i \\cdot \\sqrt{\\frac{T_{\\text{target}}}{T_{\\text{current}}}}

where the instantaneous temperature is:

.. math::

    T_{\\text{current}} = \\frac{2 \\cdot KE}{N_{\\text{DOF}} \\cdot k_B}
                        = \\frac{\\sum_i m_i |\\mathbf{v}_i|^2}{N_{\\text{DOF}} \\cdot k_B}

USAGE
=====

Velocity rescaling is useful for:
- Quick equilibration to target temperature
- Simple temperature control (non-canonical sampling)
- Initial velocity scaling before production runs

Note: Velocity rescaling does NOT produce canonical (NVT) sampling.
For proper NVT sampling, use Langevin or Nosé-Hoover thermostats.

BATCH MODE
==========

Supports three execution modes for scaling velocities:

**Single System Mode**::

    from nvalchemiops.dynamics.utils import compute_temperature, compute_kinetic_energy

    # Compute current temperature
    ke = compute_kinetic_energy(velocities, masses)
    T_current = compute_temperature(velocities, masses, num_atoms=100)

    # Compute scaling factor
    factor = compute_rescale_factor(T_current.numpy()[0], T_target=1.0)
    scale_factor = wp.array([factor], dtype=wp.float64, device="cuda:0")

    # Apply rescaling
    velocity_rescale(velocities, scale_factor)

**Batch Mode with batch_idx**::

    # Different scale factors for each system
    batch_idx = wp.array([0]*N0 + [1]*N1 + [2]*N2, dtype=wp.int32, device="cuda:0")
    scale_factors = wp.array([s0, s1, s2], dtype=wp.float64, device="cuda:0")

    velocity_rescale(velocities, scale_factors, batch_idx=batch_idx)

**Batch Mode with atom_ptr**::

    atom_ptr = wp.array([0, N0, N0+N1, N0+N1+N2], dtype=wp.int32, device="cuda:0")
    scale_factors = wp.array([s0, s1, s2], dtype=wp.float64, device="cuda:0")

    velocity_rescale(velocities, scale_factors, atom_ptr=atom_ptr)

REFERENCES
==========

- Berendsen et al. (1984). J. Chem. Phys. 81, 3684 (weak coupling)
- Bussi et al. (2007). J. Chem. Phys. 126, 014101 (stochastic rescaling)
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ..utils.kernel_functions import (
    scale_vector_by_scalar,
)

__all__ = [
    # Mutating APIs
    "velocity_rescale",
    # Non-mutating APIs
    "velocity_rescale_out",
    # Utility
    "compute_rescale_factor",
]


# ==============================================================================
# Rescaling Kernels
# ==============================================================================


@wp.kernel
def _velocity_rescale_kernel(
    velocities: wp.array(dtype=Any),
    scale_factor: wp.array(dtype=Any),
):
    """Rescale all velocities by a single factor (in-place).

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. MODIFIED in-place.
    scale_factor : wp.array
        Scaling factor sqrt(T_target/T_current). Shape (1,).
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    s = scale_factor[0]

    velocities[atom_idx] = scale_vector_by_scalar(v, s)


@wp.kernel
def _velocity_rescale_out_kernel(
    velocities: wp.array(dtype=Any),
    scale_factor: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Rescale all velocities by a single factor (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms]
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    s = scale_factor[0]

    velocities_out[atom_idx] = scale_vector_by_scalar(v, s)


@wp.kernel
def _batch_velocity_rescale_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    scale_factors: wp.array(dtype=Any),
):
    """Rescale velocities with per-system factors (in-place).

    Launch Grid
    -----------
    dim = [num_atoms_total]

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. MODIFIED in-place.
    batch_idx : wp.array(dtype=wp.int32)
        System index for each atom.
    scale_factors : wp.array
        Per-system scaling factors. Shape (B,).
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    s = scale_factors[batch_idx[atom_idx]]

    velocities[atom_idx] = scale_vector_by_scalar(v, s)


@wp.kernel
def _batch_velocity_rescale_out_kernel(
    velocities: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    scale_factors: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Rescale velocities with per-system factors (non-mutating).

    Launch Grid
    -----------
    dim = [num_atoms_total]
    """
    atom_idx = wp.tid()
    v = velocities[atom_idx]
    s = scale_factors[batch_idx[atom_idx]]

    velocities_out[atom_idx] = scale_vector_by_scalar(v, s)


# ==============================================================================
# Pointer-Based (CSR) Rescaling Kernels
# ==============================================================================


@wp.kernel
def _velocity_rescale_ptr_kernel(
    velocities: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    scale_factors: wp.array(dtype=Any),
):
    """Rescale velocities using atom_ptr (in-place).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (num_atoms_total,). MODIFIED in-place.
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
        System s owns atoms in range [atom_ptr[s], atom_ptr[s+1]).
    scale_factors : wp.array(dtype=wp.float32 or wp.float64)
        Per-system scaling factors. Shape (num_systems,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0, a1 = atom_ptr[sys_id], atom_ptr[sys_id + 1]
    s = scale_factors[sys_id]

    for i in range(a0, a1):
        v = velocities[i]
        velocities[i] = scale_vector_by_scalar(v, s)


@wp.kernel
def _velocity_rescale_ptr_out_kernel(
    velocities: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    scale_factors: wp.array(dtype=Any),
    velocities_out: wp.array(dtype=Any),
):
    """Rescale velocities using atom_ptr (non-mutating).

    Each thread processes one system's atoms sequentially.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (num_atoms_total,).
    atom_ptr : wp.array(dtype=wp.int32)
        CSR-style pointers. Shape (num_systems + 1,).
    scale_factors : wp.array(dtype=wp.float32 or wp.float64)
        Per-system scaling factors. Shape (num_systems,).
    velocities_out : wp.array(dtype=wp.vec3f or wp.vec3d)
        Output velocities. Shape (num_atoms_total,).

    Launch Grid
    -----------
    dim = [num_systems]
    """
    sys_id = wp.tid()
    a0, a1 = atom_ptr[sys_id], atom_ptr[sys_id + 1]
    s = scale_factors[sys_id]

    for i in range(a0, a1):
        v = velocities[i]
        velocities_out[i] = scale_vector_by_scalar(v, s)


# ==============================================================================
# Kernel Overloads for Explicit Typing
# ==============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]  # Vector types

_velocity_rescale_kernel_overload = {}
_velocity_rescale_out_kernel_overload = {}
_batch_velocity_rescale_kernel_overload = {}
_batch_velocity_rescale_out_kernel_overload = {}
_velocity_rescale_ptr_kernel_overload = {}
_velocity_rescale_ptr_out_kernel_overload = {}

for t, v in zip(_T, _V):
    _velocity_rescale_kernel_overload[v] = wp.overload(
        _velocity_rescale_kernel,
        [wp.array(dtype=v), wp.array(dtype=t)],
    )
    _velocity_rescale_out_kernel_overload[v] = wp.overload(
        _velocity_rescale_out_kernel,
        [wp.array(dtype=v), wp.array(dtype=t), wp.array(dtype=v)],
    )
    _batch_velocity_rescale_kernel_overload[v] = wp.overload(
        _batch_velocity_rescale_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )
    _batch_velocity_rescale_out_kernel_overload[v] = wp.overload(
        _batch_velocity_rescale_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=v),
        ],
    )
    _velocity_rescale_ptr_kernel_overload[v] = wp.overload(
        _velocity_rescale_ptr_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )
    _velocity_rescale_ptr_out_kernel_overload[v] = wp.overload(
        _velocity_rescale_ptr_out_kernel,
        [
            wp.array(dtype=v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=t),
            wp.array(dtype=v),
        ],
    )


# ==============================================================================
# Functional Interface
# ==============================================================================


def compute_rescale_factor(
    current_temperature: float,
    target_temperature: float,
) -> float:
    """
    Compute velocity rescaling factor.

    Parameters
    ----------
    current_temperature : float
        Current instantaneous temperature.
    target_temperature : float
        Target temperature.

    Returns
    -------
    float
        Scaling factor sqrt(T_target / T_current).

    Notes
    -----
    Returns 1.0 if current_temperature <= 0 to avoid division by zero.
    """
    import math

    if current_temperature <= 0.0:
        return 1.0
    return math.sqrt(target_temperature / current_temperature)


def velocity_rescale(
    velocities: wp.array,
    scale_factor: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> None:
    """
    Rescale velocities to achieve target temperature (in-place).

    Applies v_i *= scale_factor to all velocities.

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,). MODIFIED in-place.
    scale_factor : wp.array
        Scaling factor(s). Shape (1,) for single system, (B,) for batched.
        Typically sqrt(T_target / T_current).
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device. If None, inferred from velocities.

    Example
    -------
    >>> # Compute current temperature
    >>> ke = compute_kinetic_energy(velocities, masses)
    >>> T_current = compute_temperature(ke, ndof)
    >>>
    >>> # Compute rescaling factor
    >>> factor = compute_rescale_factor(T_current, T_target)
    >>> scale = wp.array([factor], dtype=wp.float32, device=device)
    >>>
    >>> # Apply rescaling
    >>> velocity_rescale(velocities, scale)
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
            _velocity_rescale_ptr_kernel_overload[vec_dtype],
            dim=num_systems,
            inputs=[velocities, atom_ptr, scale_factor],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode - launch with dim=num_atoms
        wp.launch(
            _batch_velocity_rescale_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, batch_idx, scale_factor],
            device=device,
        )
    else:
        # Single system - launch with dim=num_atoms
        wp.launch(
            _velocity_rescale_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, scale_factor],
            device=device,
        )


def velocity_rescale_out(
    velocities: wp.array,
    scale_factor: wp.array,
    velocities_out: wp.array = None,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    device: str = None,
) -> wp.array:
    """
    Rescale velocities to achieve target temperature (non-mutating).

    Parameters
    ----------
    velocities : wp.array(dtype=wp.vec3f or wp.vec3d)
        Atomic velocities. Shape (N,).
    scale_factor : wp.array
        Scaling factor(s). Shape (1,) for single system, (B,) for batched.
    velocities_out : wp.array, optional
        Output array. If None, allocated internally.
    batch_idx : wp.array(dtype=wp.int32), optional
        System index for each atom. For batched mode (atomic operations).
    atom_ptr : wp.array(dtype=wp.int32), optional
        CSR-style pointers. Shape (num_systems + 1,). For batched mode (sequential per-system).
    device : str, optional
        Warp device.

    Returns
    -------
    wp.array
        Rescaled velocities.
    """
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")

    if device is None:
        device = velocities.device

    num_atoms = velocities.shape[0]

    if velocities_out is None:
        velocities_out = wp.empty_like(velocities)

    vec_dtype = velocities.dtype

    if atom_ptr is not None:
        # Use atom_ptr mode - launch with dim=num_systems
        num_systems = atom_ptr.shape[0] - 1
        wp.launch(
            _velocity_rescale_ptr_out_kernel_overload[vec_dtype],
            dim=num_systems,
            inputs=[velocities, atom_ptr, scale_factor, velocities_out],
            device=device,
        )
    elif batch_idx is not None:
        # Use batch_idx mode - launch with dim=num_atoms
        wp.launch(
            _batch_velocity_rescale_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, batch_idx, scale_factor, velocities_out],
            device=device,
        )
    else:
        # Single system - launch with dim=num_atoms
        wp.launch(
            _velocity_rescale_out_kernel_overload[vec_dtype],
            dim=num_atoms,
            inputs=[velocities, scale_factor, velocities_out],
            device=device,
        )

    return velocities_out
