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

from ...segment_ops import _compute_ept
from ..utils.kernel_functions import (
    clamp_displacement,
    compute_vf_vv_ff,
    fire_velocity_mixing,
    is_first_atom_of_system,
)


@wp.kernel
def _fire_step_no_downhill_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
):
    """FIRE no-downhill step (single system; launched over atoms).

    This kernel implements a fused FIRE-style update for a *single system* where
    `alpha`, `dt`, and counters are scalar arrays of shape `(1,)`.

    The intended algorithm is the standard "no downhill check" FIRE update:
    - Compute diagnostic scalars:
      - \\(v\\cdot f = \\sum_i \\mathbf{v}_i \\cdot \\mathbf{f}_i\\)
      - \\(v\\cdot v = \\sum_i \\mathbf{v}_i \\cdot \\mathbf{v}_i\\)
      - \\(f\\cdot f = \\sum_i \\mathbf{f}_i \\cdot \\mathbf{f}_i\\)
    - If \\(v\\cdot f > 0\\): mix velocities and (after `n_min` consecutive steps)
      increase `dt` and reduce `alpha`.
    - Else: reset velocities, decrease `dt`, and reset `alpha`.
    - Finally: MD-like update and capped displacement:
      \\(\\mathbf{v} \\leftarrow \\mathbf{v} + \\Delta t\\, \\mathbf{f}/m\\),
      \\(\\Delta \\mathbf{r} = \\Delta t\\, \\mathbf{v}\\),
      \\(\\Delta \\mathbf{r} \\leftarrow \\Delta \\mathbf{r}\\cdot\\min(1, \text{maxstep}/\\|\\Delta \\mathbf{r}\\|)\\),
      \\(\\mathbf{r} \\leftarrow \\mathbf{r} + \\Delta \\mathbf{r}\\).

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3*
        Atomic positions (in-place).
    velocities : wp.array, shape (N,), dtype=wp.vec3*
        Atomic velocities (in-place).
    forces : wp.array, shape (N,), dtype=wp.vec3*
        Forces on atoms.
    masses : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Per-atom masses.
    alpha : wp.array, shape (1,), dtype=wp.float*
        FIRE mixing parameter \\(\alpha\\).
    dt : wp.array, shape (1,), dtype=wp.float*
        FIRE timestep \\(\\Delta t\\).
    alpha_start : wp.array, shape (1,), dtype=wp.float*
        Reset value for \\(\alpha\\) when \\(v\\cdot f \\le 0\\).
    f_alpha : wp.array, shape (1,), dtype=wp.float*
        Multiplicative decay factor for \\(\alpha\\) when progressing.
    dt_min : wp.array, shape (1,), dtype=wp.float*
        Minimum allowed timestep.
    dt_max : wp.array, shape (1,), dtype=wp.float*
        Maximum allowed timestep.
    maxstep : wp.array, shape (1,), dtype=wp.float32
        Maximum displacement magnitude per step (cap applied to `dr`).
        (Note: this is currently float32-typed; callers should pass float32 here.)
    n_steps_positive : wp.array, shape (1,), dtype=wp.int32
        Counter for consecutive steps with \\(v\\cdot f > 0\\).
    n_min : wp.array, shape (1,), dtype=wp.int32
        Threshold for when to start increasing `dt` / decreasing `alpha`.
    f_dec : wp.array, shape (1,), dtype=wp.float*
        Multiplicative decay factor for `dt` when \\(v\\cdot f \\le 0\\).
    f_inc : wp.array, shape (1,), dtype=wp.float*
        Multiplicative growth factor for `dt` after `n_min` positive steps.
    vf, vv, ff : wp.array, shape (1,), dtype=wp.float*
        Accumulators for \\(\\sum v\\cdot f\\), \\(\\sum v\\cdot v\\), \\(\\sum f\\cdot f\\).
        Zeroed internally before each use.

    Launch Grid
    -----------
    dim = [num_atoms]

    Notes
    -----
    - This kernel assumes a *single system*: all scalar control arrays are indexed at `[0]`.
    - `vf/vv/ff` are used as cross-thread accumulators; callers should ensure they are reset to 0
      before each step (and the kernel relies on Warp’s atomic semantics for such updates).
    - `maxstep` is currently typed as `wp.float32` in the signature.
    """
    atom_idx = wp.tid()

    vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[atom_idx], forces[atom_idx])
    vf[0] += vf_val
    vv[0] += vv_val
    ff[0] += ff_val

    vf_mask = vf[0] > type(dt[0])(0.0)
    n_steps_positive[0] = wp.where(vf_mask, n_steps_positive[0] + 1, 0)
    n_steps_positive_mask = n_steps_positive[0] >= n_min[0]

    velocities[atom_idx] = wp.where(
        vf_mask,
        (type(dt[0])(1.0) - alpha[0]) * velocities[atom_idx]
        + (alpha[0] * forces[atom_idx] * wp.sqrt(vv[0] / ff[0])),
        type(dt[0])(0.0) * velocities[atom_idx],
    )
    dt[0] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[0] * f_inc[0], dt_max[0]),
            dt[0],
        ),
        wp.max(dt[0] * f_dec[0], dt_min[0]),
    )
    alpha[0] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[0] * f_alpha[0],
            alpha[0],
        ),
        alpha_start[0],
    )

    # Update velocities with forces
    velocities[atom_idx] += dt[0] * forces[atom_idx] / masses[atom_idx]
    dr = dt[0] * velocities[atom_idx]

    # Scale displacement by maxstep
    dr_clamped = clamp_displacement(dr, maxstep[0])
    positions[atom_idx] += dr_clamped


@wp.kernel(enable_backward=False)
def _fire_uphill_check_kernel(
    energy: wp.array(dtype=Any),
    energy_last: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    uphill_flag: wp.array(dtype=wp.int32),
):
    """Per-system uphill check for FIRE downhill variant.

    Compares current energy to last accepted energy and sets uphill flag.
    Only the first atom per system writes the energy updates.

    Launch Grid
    -----------
    dim = N (total atoms)

    Parameters
    ----------
    energy : wp.array, shape (M,), dtype float*
        Current per-system energies.
    energy_last : wp.array, shape (M,), dtype float*
        Last accepted per-system energies.
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom. **MUST BE SORTED**.
    uphill_flag : wp.array, shape (M,), dtype int32
        OUTPUT: 1 if system is uphill, 0 otherwise.

    Notes
    -----
    - batch_idx MUST be sorted for correct first-atom detection
    - Only first atom per system modifies energy arrays
    - All atoms read uphill_flag for their system
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    # All atoms check uphill condition
    is_uphill = energy[sys] > energy_last[sys]

    # Only first atom per system writes state
    if is_first_atom_of_system(atom_idx, batch_idx):
        if is_uphill:
            uphill_flag[sys] = 1
            energy[sys] = energy_last[sys]  # Revert energy
        else:
            uphill_flag[sys] = 0
            energy_last[sys] = energy[sys]  # Accept energy


@wp.kernel(enable_backward=False)
def _fire_revert_and_reduce_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    positions_last: wp.array(dtype=Any),
    velocities_last: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    uphill_flag: wp.array(dtype=wp.int32),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Revert uphill systems and perform RLE-based reduction.

    For uphill systems, reverts positions/velocities to last accepted state.
    Then performs RLE reduction for vf, vv, ff diagnostics.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype vec3*
        Atomic positions, modified in-place for uphill systems.
    velocities : wp.array, shape (N,), dtype vec3*
        Atomic velocities, modified in-place for uphill systems.
    forces : wp.array, shape (N,), dtype vec3*
        Forces (read-only).
    positions_last : wp.array, shape (N,), dtype vec3*
        Last accepted positions. Modified in-place for downhill systems.
    velocities_last : wp.array, shape (N,), dtype vec3*
        Last accepted velocities. Modified in-place for downhill systems.
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom. **MUST BE SORTED**.
    uphill_flag : wp.array, shape (M,), dtype int32
        Per-system uphill flags from uphill check kernel.
    vf, vv, ff : wp.array, shape (M,), dtype float*
        OUTPUT: Diagnostic accumulators. Zeroed internally before each use.
    N : int32
        Total number of atoms.
    elems_per_thread : int32
        Elements per thread (auto-tuned).

    Notes
    -----
    - Uphill systems: revert from positions_last/velocities_last
    - Downhill systems: update positions_last/velocities_last
    - RLE reduction minimizes atomic operations
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    # First element
    s_cur = batch_idx[start]
    is_uphill = uphill_flag[s_cur] != 0

    if is_uphill:
        positions[start] = positions_last[start]
        velocities[start] = velocities_last[start]
    else:
        positions_last[start] = positions[start]
        velocities_last[start] = velocities[start]

    acc_vf, acc_vv, acc_ff = compute_vf_vv_ff(velocities[start], forces[start])

    # Process remaining elements
    for i in range(start + 1, end):
        s = batch_idx[i]

        # Handle revert/accept on segment boundary
        if s != s_cur:
            # Flush accumulation for previous segment
            wp.atomic_add(vf, s_cur, acc_vf)
            wp.atomic_add(vv, s_cur, acc_vv)
            wp.atomic_add(ff, s_cur, acc_ff)

            # Start new segment
            s_cur = s
            is_uphill = uphill_flag[s] != 0
            acc_vf = type(acc_vf)(0.0)
            acc_vv = type(acc_vv)(0.0)
            acc_ff = type(acc_ff)(0.0)

        # Revert or accept state
        if is_uphill:
            positions[i] = positions_last[i]
            velocities[i] = velocities_last[i]
        else:
            positions_last[i] = positions[i]
            velocities_last[i] = velocities[i]

        # Accumulate diagnostics
        val_vf, val_vv, val_ff = compute_vf_vv_ff(velocities[i], forces[i])
        acc_vf = acc_vf + val_vf
        acc_vv = acc_vv + val_vv
        acc_ff = acc_ff + val_ff

    # Flush final segment
    wp.atomic_add(vf, s_cur, acc_vf)
    wp.atomic_add(vv, s_cur, acc_vv)
    wp.atomic_add(ff, s_cur, acc_ff)


@wp.kernel(enable_backward=False)
def _fire_update_downhill_batch_idx_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    uphill_flag: wp.array(dtype=wp.int32),
):
    """Parameter update for FIRE downhill variant with uphill masking.

    Same as no_downhill update kernel but vf_mask includes uphill check.

    Launch Grid
    -----------
    dim = N (total atoms)

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype vec3*
        Atomic positions, modified in-place.
    velocities : wp.array, shape (N,), dtype vec3*
        Atomic velocities, modified in-place.
    forces : wp.array, shape (N,), dtype vec3*
        Forces (read-only).
    masses : wp.array, shape (N,), dtype float*
        Per-atom masses (read-only).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom. **MUST BE SORTED**.
    alpha, dt, etc. : wp.array, shape (M,), dtype float*
        Per-system FIRE parameters.
    vf, vv, ff : wp.array, shape (M,), dtype float*
        Diagnostic values from reduction kernel (read-only).
    uphill_flag : wp.array, shape (M,), dtype int32
        Per-system uphill flags (read-only).

    Notes
    -----
    - Redundant computation of parameter updates (no synchronization)
    - Only first atom per segment writes dt, alpha, n_steps_positive
    - Uphill systems are masked out from velocity mixing
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    # Snapshot dt before any thread modifies it
    local_dt = dt[sys]
    zero = type(local_dt)(0.0)

    # Redundantly compute parameter updates
    _vf = vf[sys]
    _vv = vv[sys]
    _ff = ff[sys]
    is_uphill = uphill_flag[sys] != 0

    vf_mask = (_vf > zero) and (not is_uphill)
    if vf_mask:
        _nsi = n_steps_positive[sys] + 1
        n_steps_positive_mask = _nsi >= n_min[sys]
        if n_steps_positive_mask:
            new_dt = wp.min(local_dt * f_inc[sys], dt_max[sys])
            new_alpha = alpha[sys] * f_alpha[sys]
        else:
            new_dt = local_dt
            new_alpha = alpha[sys]
    else:
        _nsi = 0
        new_dt = wp.max(local_dt * f_dec[sys], dt_min[sys])
        new_alpha = alpha_start[sys]

    # First atom per segment writes
    if is_first_atom_of_system(atom_idx, batch_idx):
        dt[sys] = new_dt
        alpha[sys] = new_alpha
        n_steps_positive[sys] = _nsi

    # Velocity mixing with uphill masking
    if vf_mask:
        velocities[atom_idx] = fire_velocity_mixing(
            velocities[atom_idx], forces[atom_idx], new_alpha, _vv, _ff
        )
    else:
        velocities[atom_idx] = zero * velocities[atom_idx]

    # Position update
    velocities[atom_idx] = (
        velocities[atom_idx] + local_dt * forces[atom_idx] / masses[atom_idx]
    )
    dr = local_dt * velocities[atom_idx]
    dr_clamped = clamp_displacement(dr, maxstep[sys])
    positions[atom_idx] = positions[atom_idx] + dr_clamped


@wp.kernel(enable_backward=False)
def _fire_reduce_batch_idx_rle_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """RLE-based reduction for FIRE diagnostics (vf, vv, ff).

    Uses run-length encoding to minimize atomic operations: accumulates
    locally while batch_idx stays constant, emits atomic_add only on
    segment boundaries.

    This kernel implements the reduction phase for FIRE optimization,
    computing three per-system inner products:
    - vf[s] = sum(v·f for atoms in system s)
    - vv[s] = sum(v·v for atoms in system s)
    - ff[s] = sum(f·f for atoms in system s)

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    velocities : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic velocities (read-only).
    forces : wp.array, shape (N,), dtype vec3f/vec3d
        Forces on atoms (read-only).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom in [0, M). **MUST BE SORTED**.
    vf : wp.array, shape (M,), dtype float32/float64
        OUTPUT: v·f per system. Zeroed internally before each use.
    vv : wp.array, shape (M,), dtype float32/float64
        OUTPUT: v·v per system. Zeroed internally before each use.
    ff : wp.array, shape (M,), dtype float32/float64
        OUTPUT: f·f per system. Zeroed internally before each use.
    N : int32
        Total number of atoms.
    elems_per_thread : int32
        Elements processed per thread (auto-tuned based on array size).

    Notes
    -----
    - batch_idx MUST be sorted in non-decreasing order for correctness
    - Uses run-length encoding: O(segments) atomic operations instead of O(N)
    - Typically reduces atomics by 100-1000x compared to naive approach
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    # First element
    s_cur = batch_idx[start]
    acc_vf, acc_vv, acc_ff = compute_vf_vv_ff(velocities[start], forces[start])

    # Process remaining elements in chunk
    for i in range(start + 1, end):
        s = batch_idx[i]
        val_vf, val_vv, val_ff = compute_vf_vv_ff(velocities[i], forces[i])
        if s == s_cur:
            # Same segment: accumulate locally
            acc_vf = acc_vf + val_vf
            acc_vv = acc_vv + val_vv
            acc_ff = acc_ff + val_ff
        else:
            # Segment boundary: emit atomic and start new run
            wp.atomic_add(vf, s_cur, acc_vf)
            wp.atomic_add(vv, s_cur, acc_vv)
            wp.atomic_add(ff, s_cur, acc_ff)
            s_cur = s
            acc_vf = val_vf
            acc_vv = val_vv
            acc_ff = val_ff

    # Flush final run
    wp.atomic_add(vf, s_cur, acc_vf)
    wp.atomic_add(vv, s_cur, acc_vv)
    wp.atomic_add(ff, s_cur, acc_ff)


@wp.kernel(enable_backward=False)
def _fire_update_batch_idx_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
):
    """Parameter update, velocity mixing, and position update for FIRE.

    This kernel performs the second phase of FIRE optimization after
    reduction is complete. Each thread redundantly computes per-system
    parameter updates from shared read-only inputs (vf, vv, ff), avoiding
    inter-thread synchronization. Only the first atom per segment writes
    shared state.

    Launch Grid
    -----------
    dim = N (total atoms)

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic positions, modified in-place.
    velocities : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic velocities, modified in-place.
    forces : wp.array, shape (N,), dtype vec3f/vec3d
        Forces on atoms (read-only).
    masses : wp.array, shape (N,), dtype float32/float64
        Per-atom masses (read-only).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom. **MUST BE SORTED**.
    alpha, dt, alpha_start, f_alpha, dt_min, dt_max, maxstep : wp.array, shape (M,), dtype float*
        Per-system FIRE parameters. dt and alpha modified in-place.
    n_steps_positive, n_min : wp.array, shape (M,), dtype int32
        Per-system counters. n_steps_positive modified in-place.
    f_dec, f_inc : wp.array, shape (M,), dtype float*
        Per-system timestep factors (read-only).
    vf, vv, ff : wp.array, shape (M,), dtype float*
        Per-system diagnostic values from reduction kernel (read-only).

    Notes
    -----
    - batch_idx MUST be sorted for correct first-atom-per-segment detection
    - Each thread redundantly computes parameter updates (no synchronization)
    - Only first atom in each segment writes dt, alpha, n_steps_positive
    - Position updates use snapshot of dt before any thread modifies it
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    # Snapshot dt before any thread modifies it (race-condition guard)
    local_dt = dt[sys]
    zero = type(local_dt)(0.0)

    # Redundantly compute per-system parameter updates from read-only inputs
    _vf = vf[sys]
    _vv = vv[sys]
    _ff = ff[sys]

    vf_mask = _vf > zero
    if vf_mask:
        _nsi = n_steps_positive[sys] + 1
        n_steps_positive_mask = _nsi >= n_min[sys]
        if n_steps_positive_mask:
            new_dt = wp.min(local_dt * f_inc[sys], dt_max[sys])
            new_alpha = alpha[sys] * f_alpha[sys]
        else:
            new_dt = local_dt
            new_alpha = alpha[sys]
    else:
        _nsi = 0
        new_dt = wp.max(local_dt * f_dec[sys], dt_min[sys])
        new_alpha = alpha_start[sys]

    # First atom per segment writes updated params
    if is_first_atom_of_system(atom_idx, batch_idx):
        dt[sys] = new_dt
        alpha[sys] = new_alpha
        n_steps_positive[sys] = _nsi

    # Velocity mixing (all atoms)
    if vf_mask:
        velocities[atom_idx] = fire_velocity_mixing(
            velocities[atom_idx], forces[atom_idx], new_alpha, _vv, _ff
        )
    else:
        velocities[atom_idx] = zero * velocities[atom_idx]

    # Update velocities with forces (mass-aware) and positions
    velocities[atom_idx] = (
        velocities[atom_idx] + local_dt * forces[atom_idx] / masses[atom_idx]
    )
    dr = local_dt * velocities[atom_idx]
    dr_clamped = clamp_displacement(dr, maxstep[sys])
    positions[atom_idx] = positions[atom_idx] + dr_clamped


@wp.kernel
def _fire_step_no_downhill_batch_idx_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
):
    r"""FIRE no-downhill step (batched via `batch_idx`; launched over atoms).

    .. deprecated::
        This kernel has race conditions due to non-atomic accumulation.
        It is kept for backward compatibility but should not be used.
        Use the two-kernel approach with _fire_reduce_batch_idx_rle_kernel
        followed by _fire_update_batch_idx_kernel instead, or use the
        atom_ptr kernel which is race-free.

    This kernel applies the same per-system FIRE logic as the single-system kernel,
    but uses `batch_idx[atom] -> sys` to select which system's control parameters
    (`dt`, `alpha`, counters, etc.) apply to each atom.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated positions for all systems (in-place).
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated velocities for all systems (in-place).
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces for all systems.
    masses : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Concatenated masses.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    alpha, dt, alpha_start, f_alpha, dt_min, dt_max, maxstep : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE parameters.
    n_steps_positive, n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system counters/thresholds.
    f_dec, f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system timestep factors.
    vf, vv, ff : wp.array, shape (B,), dtype=wp.float*
        Per-system accumulators for \(v\cdot f\), \(v\cdot v\), \(f\cdot f\).
        Zeroed internally before each use.

    Launch Grid
    -----------
    dim = [num_atoms_total]

    Notes
    -----
    - **DEPRECATED:** This kernel has known race conditions
    - Race conditions occur on lines with += operations on shared arrays
    - Use _fire_reduce_batch_idx_rle_kernel + _fire_update_batch_idx_kernel instead
    - The FIRE logic uses per-system scalar values via `sys = batch_idx[atom_idx]`.
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[atom_idx], forces[atom_idx])
    vf[sys] += vf_val
    vv[sys] += vv_val
    ff[sys] += ff_val

    vf_mask = vf[sys] > type(dt[sys])(0.0)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    velocities[atom_idx] = wp.where(
        vf_mask,
        (type(dt[sys])(1.0) - alpha[sys]) * velocities[atom_idx]
        + (alpha[sys] * forces[atom_idx] * wp.sqrt(vv[sys] / ff[sys])),
        type(dt[sys])(0.0) * velocities[atom_idx],
    )
    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )

    # Update velocities with forces (mass-aware)
    velocities[atom_idx] += dt[sys] * forces[atom_idx] / masses[atom_idx]
    dr = dt[sys] * velocities[atom_idx]
    dr_clamped = clamp_displacement(dr, maxstep[sys])
    positions[atom_idx] += dr_clamped


@wp.kernel
def _fire_step_no_downhill_ptr_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
):
    """FIRE no-downhill step (ptr/CSR batched; launched over systems).

    This is the ptr-based ("CSR") batching formulation analogous to the reference
    implementation you shared:

    - Launch grid is over systems: `dim = [num_systems]`
    - Each thread processes the contiguous atom range:
      `i in [atom_ptr[sys], atom_ptr[sys+1])`
    - All per-system reductions (`vf/vv/ff`) and parameter updates happen within
      the same thread, so no cross-thread synchronization is required.

    Parameters
    ----------
    positions : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated positions for all systems (in-place).
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated velocities for all systems (in-place).
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces for all systems.
    masses : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Concatenated masses.
    alpha, dt, alpha_start, f_alpha, dt_min, dt_max, maxstep : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE parameters.
    n_steps_positive, n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system counters/thresholds.
    f_dec, f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system timestep factors.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32
        CSR pointer giving the start/end atom indices for each system.

    Launch Grid
    -----------
    dim = [num_systems]

    Notes
    -----
    - This formulation is typically the best choice for a fully fused FIRE step
      because the entire system update is carried out within a single thread.
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    # Compute diagnostics within system
    vf = type(dt[sys])(0.0)
    vv = type(dt[sys])(0.0)
    ff = type(dt[sys])(0.0)
    for i in range(a0, a1):
        vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[i], forces[i])
        vf += vf_val
        vv += vv_val
        ff += ff_val

    vf_mask = vf > type(dt[sys])(0.0)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    # Guard against division by zero when forces are zero
    zero = type(dt[sys])(0.0)
    if ff > zero:
        ratio = wp.sqrt(vv / ff)
    else:
        ratio = zero

    for i in range(a0, a1):
        velocities[i] = wp.where(
            vf_mask,
            (type(dt[sys])(1.0) - alpha[sys]) * velocities[i]
            + (alpha[sys] * forces[i] * ratio),
            zero * velocities[i],
        )
    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )
    for i in range(a0, a1):
        velocities[i] += dt[sys] * forces[i] / masses[i]
        dr = dt[sys] * velocities[i]
        dr_clamped = clamp_displacement(dr, maxstep[sys])
        positions[i] += dr_clamped


@wp.kernel
def _fire_step_downhill_kernel(
    energy: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    energy_last: wp.array(dtype=Any),
    positions_last: wp.array(dtype=Any),
    velocities_last: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
):
    """FIRE downhill-check step (single system; launched over atoms).

    This kernel mirrors the reference "downhill check" variant:
    - Tracks previous energy (`energy_last[0]`) and previous positions/velocities
      (`positions_last[:]`, `velocities_last[:]`).
    - If energy increases relative to the last accepted energy, it rolls positions
      back to `positions_last` and marks the step as uphill.
    - Applies the FIRE power criterion (vf = sum(F·v)) and only mixes velocities
      when vf > 0 and the step is not uphill.
    - Updates dt/alpha and performs an MD-like update with mass-aware acceleration
      and a global maxstep cap based on ||dr|| over the whole system.

    Launch Grid
    -----------
    dim = [num_atoms]

    Parameters
    ----------
    energy : wp.array, shape (1,), dtype=wp.float*
        Current system energy (single value). Updated in-place if uphill.
    forces : wp.array, shape (N,), dtype=wp.vec3*
        Forces on atoms.
    positions : wp.array, shape (N,), dtype=wp.vec3*
        Atomic positions (in-place).
    velocities : wp.array, shape (N,), dtype=wp.vec3*
        Atomic velocities (in-place).
    masses : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Per-atom masses.
    alpha : wp.array, shape (1,), dtype=wp.float*
        FIRE mixing parameter \\(\alpha\\).
    dt : wp.array, shape (1,), dtype=wp.float*
        FIRE timestep \\(\\Delta t\\).
    alpha_start : wp.array, shape (1,), dtype=wp.float*
        Reset value for \\(\alpha\\) when uphill or \\(v\\cdot f \\le 0\\).
    f_alpha : wp.array, shape (1,), dtype=wp.float*
        Multiplicative decay factor for \\(\alpha\\) when progressing.
    dt_min : wp.array, shape (1,), dtype=wp.float*
        Minimum allowed timestep.
    dt_max : wp.array, shape (1,), dtype=wp.float*
        Maximum allowed timestep.
    maxstep : wp.array, shape (1,), dtype=wp.float*
        Maximum displacement magnitude per step (cap applied to `dr`).
    n_steps_positive : wp.array, shape (1,), dtype=wp.int32
        Counter for consecutive steps with \\(v\\cdot f > 0\\).
    n_min : wp.array, shape (1,), dtype=wp.int32
        Threshold for when to start increasing `dt` / decreasing `alpha`.
    f_dec : wp.array, shape (1,), dtype=wp.float*
        Multiplicative decay factor for `dt` when uphill or \\(v\\cdot f \\le 0\\).
    f_inc : wp.array, shape (1,), dtype=wp.float*
        Multiplicative growth factor for `dt` after `n_min` positive steps.
    energy_last : wp.array, shape (1,), dtype=wp.float*
        Last accepted energy; used to detect uphill steps.
    positions_last : wp.array, shape (N,), dtype=wp.vec3*
        Last accepted positions; used for rollback.
    velocities_last : wp.array, shape (N,), dtype=wp.vec3*
        Last accepted velocities; used for rollback.
    vf, vv, ff : wp.array, shape (1,), dtype=wp.float*
        Accumulators for \\(\\sum v\\cdot f\\), \\(\\sum v\\cdot v\\), \\(\\sum f\\cdot f\\).
        Zeroed internally before each use.

    Launch Grid
    -----------
    dim = [num_atoms]

    Notes
    -----
    - This kernel matches the *launch style* of `_fire_step_no_downhill_kernel`:
      it is launched with one thread per atom (`atom_idx = wp.tid()`).
    - Scalar control/state arrays (`dt`, `alpha`, `energy_last`, etc.) are shape `(1,)`
      and are intended to be updated coherently each step.
    - `vf/vv/ff` are per-step accumulators (shape `(1,)`) and must be cleared by the caller
      before launching this kernel.
    """
    atom_idx = wp.tid()

    # Uphill check
    is_uphill = False
    if energy[0] > energy_last[0]:
        is_uphill = True
        energy[0] = energy_last[0]
        positions[atom_idx] = positions_last[atom_idx]
        velocities[atom_idx] = velocities_last[atom_idx]
    else:
        # Update saved state (accepted state)
        energy_last[0] = energy[0]
        positions_last[atom_idx] = positions[atom_idx]
        velocities_last[atom_idx] = velocities[atom_idx]

    # (3) Accumulate diagnostics
    vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[atom_idx], forces[atom_idx])
    vf[0] += vf_val
    vv[0] += vv_val
    ff[0] += ff_val

    vf_mask = (vf[0] > type(dt[0])(0.0)) and (not is_uphill)
    n_steps_positive[0] = wp.where(vf_mask, n_steps_positive[0] + 1, 0)
    n_steps_positive_mask = n_steps_positive[0] >= n_min[0]

    # (4) Velocity mixing per-atom
    velocities[atom_idx] = wp.where(
        vf_mask,
        (type(dt[0])(1.0) - alpha[0]) * velocities[atom_idx]
        + (alpha[0] * forces[atom_idx] * wp.sqrt(vv[0] / ff[0])),
        type(dt[0])(0.0) * velocities[atom_idx],
    )

    # (5) Update dt/alpha once
    if atom_idx == 0:
        dt[0] = wp.where(
            vf_mask,
            wp.where(
                n_steps_positive_mask,
                wp.min(dt[0] * f_inc[0], dt_max[0]),
                dt[0],
            ),
            wp.max(dt[0] * f_dec[0], dt_min[0]),
        )
        alpha[0] = wp.where(
            vf_mask,
            wp.where(
                n_steps_positive_mask,
                alpha[0] * f_alpha[0],
                alpha[0],
            ),
            alpha_start[0],
        )

    # (6) MD-like update + per-atom maxstep cap (same style as no-downhill)
    velocities[atom_idx] += dt[0] * forces[atom_idx] / masses[atom_idx]
    dr = dt[0] * velocities[atom_idx]
    dr_clamped = clamp_displacement(dr, maxstep[0])
    positions[atom_idx] += dr_clamped


@wp.kernel
def _fire_step_downhill_batch_idx_kernel(
    energy: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    energy_last: wp.array(dtype=Any),
    positions_last: wp.array(dtype=Any),
    velocities_last: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
):
    r"""FIRE downhill-check step (batched via `batch_idx`; launched over atoms).

    .. deprecated::
        This kernel has race conditions due to non-atomic operations on shared state.
        Use the 3-kernel approach: _fire_uphill_check_kernel, _fire_revert_and_reduce_kernel,
        and _fire_update_downhill_batch_idx_kernel instead, or use the atom_ptr kernel.

    This is the batched analogue of the single-system downhill kernel. Each atom
    reads its system id via `batch_idx` and uses per-system scalars for the FIRE
    controls and downhill bookkeeping.

    Parameters
    ----------
    energy : wp.array, shape (B,), dtype=wp.float*
        Per-system energies. Each system uses `energy[sys]` for uphill checks.
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces.
    positions : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated positions (in-place).
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated velocities (in-place).
    masses : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Concatenated masses.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.
    alpha, dt, alpha_start, f_alpha, dt_min, dt_max, maxstep : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE parameters.
    n_steps_positive, n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system counters/thresholds.
    f_dec, f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system timestep factors.
    energy_last : wp.array, shape (B,), dtype=wp.float*
        Per-system last accepted energies for downhill checks.
    positions_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Per-atom last accepted positions.
    velocities_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Per-atom last accepted velocities.
    vf, vv, ff : wp.array, shape (B,), dtype=wp.float*
        Per-system accumulators for \(v\cdot f\), \(v\cdot v\), \(f\cdot f\).
        Zeroed internally before each use.

    Launch Grid
    -----------
    dim = [num_atoms_total]

    Notes
    -----
    - This formulation is convenient but relies on per-step accumulator arrays.
    - Uphill detection is per system: each atom checks `energy[sys]` vs
      `energy_last[sys]` and uses the per-system flag to gate velocity mixing.
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    # Uphill check
    is_uphill = False
    if energy[sys] > energy_last[sys]:
        is_uphill = True
        energy[sys] = energy_last[sys]
        positions[atom_idx] = positions_last[atom_idx]
        velocities[atom_idx] = velocities_last[atom_idx]
    else:
        energy_last[sys] = energy[sys]
        positions_last[atom_idx] = positions[atom_idx]
        velocities_last[atom_idx] = velocities[atom_idx]

    vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[atom_idx], forces[atom_idx])
    vf[sys] += vf_val
    vv[sys] += vv_val
    ff[sys] += ff_val

    vf_mask = (vf[sys] > type(dt[sys])(0.0)) and (not is_uphill)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    velocities[atom_idx] = wp.where(
        vf_mask,
        (type(dt[sys])(1.0) - alpha[sys]) * velocities[atom_idx]
        + (alpha[sys] * forces[atom_idx] * wp.sqrt(vv[sys] / ff[sys])),
        type(dt[sys])(0.0) * velocities[atom_idx],
    )

    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )

    velocities[atom_idx] += dt[sys] * forces[atom_idx] / masses[atom_idx]
    dr = dt[sys] * velocities[atom_idx]
    dr_clamped = clamp_displacement(dr, maxstep[sys])
    positions[atom_idx] += dr_clamped


@wp.kernel
def _fire_step_downhill_ptr_kernel(
    energy: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    maxstep: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    energy_last: wp.array(dtype=Any),
    positions_last: wp.array(dtype=Any),
    velocities_last: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
):
    """FIRE downhill-check step (ptr/CSR batched; launched over systems).

    This is the ptr-based ("CSR") batched formulation: each thread owns a full
    system range `[atom_ptr[sys], atom_ptr[sys+1])` and performs the downhill
    check, FIRE updates, and MD-like step for that system without cross-thread
    synchronization.

    Parameters
    ----------
    energy : wp.array, shape (B,), dtype=wp.float*
        Per-system energies.
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces.
    positions : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated positions (in-place).
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated velocities (in-place).
    masses : wp.array, shape (N_total,), dtype=wp.float32 or wp.float64
        Concatenated masses.
    alpha, dt, alpha_start, f_alpha, dt_min, dt_max, maxstep : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE parameters.
    n_steps_positive, n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system counters/thresholds.
    f_dec, f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system timestep factors.
    energy_last : wp.array, shape (B,), dtype=wp.float*
        Per-system last accepted energies.
    positions_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Per-atom last accepted positions.
    velocities_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Per-atom last accepted velocities.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32
        CSR pointer giving the start/end atom indices for each system.

    Launch Grid
    -----------
    dim = [num_systems]

    Notes
    -----
    - This formulation is the most natural way to keep the downhill logic fully
      fused because each system is processed by a single thread.
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    # Uphill check
    is_uphill = False
    if energy[sys] > energy_last[sys]:
        is_uphill = True
        energy[sys] = energy_last[sys]
        for i in range(a0, a1):
            positions[i] = positions_last[i]
            velocities[i] = velocities_last[i]
    else:
        energy_last[sys] = energy[sys]
        for i in range(a0, a1):
            positions_last[i] = positions[i]
            velocities_last[i] = velocities[i]

    vf = type(dt[sys])(0.0)
    vv = type(dt[sys])(0.0)
    ff = type(dt[sys])(0.0)
    for i in range(a0, a1):
        vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[i], forces[i])
        vf += vf_val
        vv += vv_val
        ff += ff_val

    vf_mask = (vf > type(dt[sys])(0.0)) and (not is_uphill)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    # Guard against division by zero when forces are zero
    zero = type(dt[sys])(0.0)
    if ff > zero:
        ratio = wp.sqrt(vv / ff)
    else:
        ratio = zero

    for i in range(a0, a1):
        velocities[i] = wp.where(
            vf_mask,
            (type(dt[sys])(1.0) - alpha[sys]) * velocities[i]
            + (alpha[sys] * forces[i] * ratio),
            zero * velocities[i],
        )

    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )

    for i in range(a0, a1):
        velocities[i] += dt[sys] * forces[i] / masses[i]
        dr = dt[sys] * velocities[i]
        dr_clamped = clamp_displacement(dr, maxstep[sys])
        positions[i] += dr_clamped


@wp.kernel
def _fire_update_params_no_downhill_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
):
    r"""FIRE parameter update (no downhill; single system).

    Computes diagnostic scalars (\\(v\\cdot f\\), \\(v\\cdot v\\), \\(f\\cdot f\\)),
    performs velocity mixing, and updates `dt`, `alpha`, and `n_steps_positive`
    **without** performing any MD step (no position update).

    This kernel is intended to be used when the caller wants to decouple the
    FIRE parameter/velocity update from the MD integration step.

    Parameters
    ----------
    velocities : wp.array, shape (N,), dtype=wp.vec3*
        Atomic velocities (in-place; mixed according to FIRE rule).
    forces : wp.array, shape (N,), dtype=wp.vec3*
        Forces on atoms.
    alpha : wp.array, shape (1,), dtype=wp.float*
        FIRE mixing parameter \\(\\alpha\\).
    dt : wp.array, shape (1,), dtype=wp.float*
        FIRE timestep \\(\\Delta t\\).
    alpha_start : wp.array, shape (1,), dtype=wp.float*
        Reset value for \\(\\alpha\\) when \\(v\\cdot f \\le 0\\).
    f_alpha : wp.array, shape (1,), dtype=wp.float*
        Multiplicative decay factor for \\(\\alpha\\) when progressing.
    dt_min : wp.array, shape (1,), dtype=wp.float*
        Minimum allowed timestep.
    dt_max : wp.array, shape (1,), dtype=wp.float*
        Maximum allowed timestep.
    n_steps_positive : wp.array, shape (1,), dtype=wp.int32
        Counter for consecutive steps with \\(v\\cdot f > 0\\).
    n_min : wp.array, shape (1,), dtype=wp.int32
        Threshold for when to start increasing `dt` / decreasing `alpha`.
    f_dec : wp.array, shape (1,), dtype=wp.float*
        Multiplicative decay factor for `dt` when \\(v\\cdot f \\le 0\\).
    f_inc : wp.array, shape (1,), dtype=wp.float*
        Multiplicative growth factor for `dt` after `n_min` positive steps.
    vv : wp.array, shape (1,), dtype=wp.float*
        Accumulator for \\(\\sum v\\cdot v\\). Zeroed internally before each use.
    ff : wp.array, shape (1,), dtype=wp.float*
        Accumulator for \\(\\sum f\\cdot f\\). Zeroed internally before each use.
    vf : wp.array, shape (1,), dtype=wp.float*
        Accumulator for \\(\\sum v\\cdot f\\). Zeroed internally before each use.

    Launch Grid
    -----------
    dim = [num_atoms]

    Notes
    -----
    - This kernel does NOT perform the MD step (velocity integration + position update).
    - The caller is responsible for performing the MD step separately after this kernel.
    - `vf/vv/ff` are cross-thread accumulators and zeroed internally before each use.
    """
    atom_idx = wp.tid()

    # Accumulate diagnostics
    vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[atom_idx], forces[atom_idx])
    vf[0] += vf_val
    vv[0] += vv_val
    ff[0] += ff_val

    vf_mask = vf[0] > type(dt[0])(0.0)
    n_steps_positive[0] = wp.where(vf_mask, n_steps_positive[0] + 1, 0)
    n_steps_positive_mask = n_steps_positive[0] >= n_min[0]

    # Velocity mixing (no MD step)
    velocities[atom_idx] = wp.where(
        vf_mask,
        (type(dt[0])(1.0) - alpha[0]) * velocities[atom_idx]
        + (alpha[0] * forces[atom_idx] * wp.sqrt(vv[0] / ff[0])),
        type(dt[0])(0.0) * velocities[atom_idx],
    )

    if atom_idx == 0:
        dt[0] = wp.where(
            vf_mask,
            wp.where(
                n_steps_positive_mask,
                wp.min(dt[0] * f_inc[0], dt_max[0]),
                dt[0],
            ),
            wp.max(dt[0] * f_dec[0], dt_min[0]),
        )
        alpha[0] = wp.where(
            vf_mask,
            wp.where(
                n_steps_positive_mask,
                alpha[0] * f_alpha[0],
                alpha[0],
            ),
            alpha_start[0],
        )


@wp.kernel
def _fire_update_params_no_downhill_batch_idx_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
):
    r"""FIRE parameter update (no downhill; batch_idx).

    Computes per-system diagnostic scalars (\\(v\\cdot f\\), \\(v\\cdot v\\),
    \\(f\\cdot f\\)), performs velocity mixing, and updates per-system `dt`,
    `alpha`, and `n_steps_positive` **without** performing any MD step.

    Parameters
    ----------
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated atomic velocities (in-place; mixed according to FIRE rule).
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces on atoms.
    alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE mixing parameter \\(\\alpha\\).
    dt : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE timestep \\(\\Delta t\\).
    alpha_start : wp.array, shape (B,), dtype=wp.float*
        Per-system reset value for \\(\\alpha\\).
    f_alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system multiplicative decay factor for \\(\\alpha\\).
    dt_min : wp.array, shape (B,), dtype=wp.float*
        Per-system minimum allowed timestep.
    dt_max : wp.array, shape (B,), dtype=wp.float*
        Per-system maximum allowed timestep.
    n_steps_positive : wp.array, shape (B,), dtype=wp.int32
        Per-system counter for consecutive steps with \\(v\\cdot f > 0\\).
    n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system threshold for when to start increasing `dt`.
    f_dec : wp.array, shape (B,), dtype=wp.float*
        Per-system decay factor for `dt` when \\(v\\cdot f \\le 0\\).
    f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system growth factor for `dt` after `n_min` positive steps.
    vf : wp.array, shape (B,), dtype=wp.float*
        Per-system accumulator for \\(\\sum v\\cdot f\\). Zeroed internally before each use.
    vv : wp.array, shape (B,), dtype=wp.float*
        Per-system accumulator for \\(\\sum v\\cdot v\\). Zeroed internally before each use.
    ff : wp.array, shape (B,), dtype=wp.float*
        Per-system accumulator for \\(\\sum f\\cdot f\\). Zeroed internally before each use.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.

    Launch Grid
    -----------
    dim = [num_atoms_total]

    Notes
    -----
    - This kernel does NOT perform the MD step (velocity integration + position update).
    - The caller is responsible for performing the MD step separately after this kernel.
    - `vf/vv/ff` are per-system accumulators and zeroed internally before each use.
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[atom_idx], forces[atom_idx])
    vf[sys] += vf_val
    vv[sys] += vv_val
    ff[sys] += ff_val

    vf_mask = vf[sys] > type(dt[sys])(0.0)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    velocities[atom_idx] = wp.where(
        vf_mask,
        (type(dt[sys])(1.0) - alpha[sys]) * velocities[atom_idx]
        + (alpha[sys] * forces[atom_idx] * wp.sqrt(vv[sys] / ff[sys])),
        type(dt[sys])(0.0) * velocities[atom_idx],
    )

    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )


@wp.kernel
def _fire_update_params_no_downhill_ptr_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
):
    r"""FIRE parameter update (no downhill; ptr/CSR).

    Each thread owns a full system range `[atom_ptr[sys], atom_ptr[sys+1])` and
    computes the diagnostic scalars (\\(v\\cdot f\\), \\(v\\cdot v\\), \\(f\\cdot f\\)),
    performs velocity mixing, and updates per-system `dt`, `alpha`, and
    `n_steps_positive` **without** performing any MD step.

    Parameters
    ----------
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated atomic velocities (in-place; mixed according to FIRE rule).
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces on atoms.
    alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE mixing parameter \\(\\alpha\\).
    dt : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE timestep \\(\\Delta t\\).
    alpha_start : wp.array, shape (B,), dtype=wp.float*
        Per-system reset value for \\(\\alpha\\).
    f_alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system multiplicative decay factor for \\(\\alpha\\).
    dt_min : wp.array, shape (B,), dtype=wp.float*
        Per-system minimum allowed timestep.
    dt_max : wp.array, shape (B,), dtype=wp.float*
        Per-system maximum allowed timestep.
    n_steps_positive : wp.array, shape (B,), dtype=wp.int32
        Per-system counter for consecutive steps with \\(v\\cdot f > 0\\).
    n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system threshold for when to start increasing `dt`.
    f_dec : wp.array, shape (B,), dtype=wp.float*
        Per-system decay factor for `dt` when \\(v\\cdot f \\le 0\\).
    f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system growth factor for `dt` after `n_min` positive steps.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32
        CSR pointer giving the start/end atom indices for each system.

    Launch Grid
    -----------
    dim = [num_systems]

    Notes
    -----
    - This kernel does NOT perform the MD step (velocity integration + position update).
    - Each system is processed by a single thread, so no cross-thread synchronization
      is required for the diagnostic reductions.
    """
    sys = wp.tid()

    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]
    vv = type(dt[sys])(0.0)
    ff = type(dt[sys])(0.0)
    vf = type(dt[sys])(0.0)
    for i in range(a0, a1):
        vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[i], forces[i])
        vf += vf_val
        vv += vv_val
        ff += ff_val

    vf_mask = vf > type(dt[sys])(0.0)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    # Guard against division by zero when forces are zero
    zero = type(dt[sys])(0.0)
    if ff > zero:
        ratio = wp.sqrt(vv / ff)
    else:
        ratio = zero

    for i in range(a0, a1):
        velocities[i] = wp.where(
            vf_mask,
            (type(dt[sys])(1.0) - alpha[sys]) * velocities[i]
            + (alpha[sys] * forces[i] * ratio),
            zero * velocities[i],
        )
    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )


@wp.kernel
def _fire_update_params_downhill_kernel(
    energy: wp.array(dtype=Any),
    energy_last: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    positions_last: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    velocities_last: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
):
    r"""FIRE parameter update (downhill; single system).

    Performs the downhill check (rolling back positions/velocities if energy increased),
    computes diagnostic scalars (\\(v\\cdot f\\), \\(v\\cdot v\\), \\(f\\cdot f\\)),
    applies velocity mixing, and updates `dt`, `alpha`, and `n_steps_positive`
    **without** performing any MD step.

    Parameters
    ----------
    energy : wp.array, shape (1,), dtype=wp.float*
        Current system energy. Rolled back to `energy_last` if uphill.
    energy_last : wp.array, shape (1,), dtype=wp.float*
        Last accepted energy; used to detect uphill steps.
    positions : wp.array, shape (N,), dtype=wp.vec3*
        Atomic positions (in-place; rolled back if uphill).
    positions_last : wp.array, shape (N,), dtype=wp.vec3*
        Last accepted positions; used for rollback and updated on accept.
    velocities : wp.array, shape (N,), dtype=wp.vec3*
        Atomic velocities (in-place; rolled back if uphill, then mixed).
    velocities_last : wp.array, shape (N,), dtype=wp.vec3*
        Last accepted velocities; used for rollback and updated on accept.
    forces : wp.array, shape (N,), dtype=wp.vec3*
        Forces on atoms.
    alpha : wp.array, shape (1,), dtype=wp.float*
        FIRE mixing parameter \\(\\alpha\\).
    dt : wp.array, shape (1,), dtype=wp.float*
        FIRE timestep \\(\\Delta t\\).
    alpha_start : wp.array, shape (1,), dtype=wp.float*
        Reset value for \\(\\alpha\\) when uphill or \\(v\\cdot f \\le 0\\).
    f_alpha : wp.array, shape (1,), dtype=wp.float*
        Multiplicative decay factor for \\(\\alpha\\) when progressing.
    dt_min : wp.array, shape (1,), dtype=wp.float*
        Minimum allowed timestep.
    dt_max : wp.array, shape (1,), dtype=wp.float*
        Maximum allowed timestep.
    n_steps_positive : wp.array, shape (1,), dtype=wp.int32
        Counter for consecutive steps with \\(v\\cdot f > 0\\).
    n_min : wp.array, shape (1,), dtype=wp.int32
        Threshold for when to start increasing `dt` / decreasing `alpha`.
    f_dec : wp.array, shape (1,), dtype=wp.float*
        Multiplicative decay factor for `dt` when uphill or \\(v\\cdot f \\le 0\\).
    f_inc : wp.array, shape (1,), dtype=wp.float*
        Multiplicative growth factor for `dt` after `n_min` positive steps.
    vv : wp.array, shape (1,), dtype=wp.float*
        Accumulator for \\(\\sum v\\cdot v\\). Zeroed internally before each use.
    ff : wp.array, shape (1,), dtype=wp.float*
        Accumulator for \\(\\sum f\\cdot f\\). Zeroed internally before each use.
    vf : wp.array, shape (1,), dtype=wp.float*
        Accumulator for \\(\\sum v\\cdot f\\). Zeroed internally before each use.

    Launch Grid
    -----------
    dim = [num_atoms]

    Notes
    -----
    - This kernel does NOT perform the MD step (velocity integration + position update).
    - If energy > energy_last, the step is marked as uphill and positions/velocities
      are rolled back to `*_last` arrays. Energy is also rolled back.
    - If energy <= energy_last, the `*_last` arrays are updated with current values.
    - Velocity mixing only occurs if \\(v\\cdot f > 0\\) AND the step is not uphill.
    - `vf/vv/ff` are cross-thread accumulators and zeroed internally before each use.
    """
    atom_idx = wp.tid()

    is_uphill = False
    if energy[0] > energy_last[0]:
        is_uphill = True
        energy[0] = energy_last[0]
        positions[atom_idx] = positions_last[atom_idx]
        velocities[atom_idx] = velocities_last[atom_idx]
    else:
        energy_last[0] = energy[0]
        positions_last[atom_idx] = positions[atom_idx]
        velocities_last[atom_idx] = velocities[atom_idx]

    vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[atom_idx], forces[atom_idx])
    vf[0] += vf_val
    vv[0] += vv_val
    ff[0] += ff_val

    vf_mask = (vf[0] > type(dt[0])(0.0)) and (not is_uphill)
    n_steps_positive[0] = wp.where(vf_mask, n_steps_positive[0] + 1, 0)
    n_steps_positive_mask = n_steps_positive[0] >= n_min[0]

    velocities[atom_idx] = wp.where(
        vf_mask,
        (type(dt[0])(1.0) - alpha[0]) * velocities[atom_idx]
        + (alpha[0] * forces[atom_idx] * wp.sqrt(vv[0] / ff[0])),
        type(dt[0])(0.0) * velocities[atom_idx],
    )

    if atom_idx == 0:
        dt[0] = wp.where(
            vf_mask,
            wp.where(
                n_steps_positive_mask,
                wp.min(dt[0] * f_inc[0], dt_max[0]),
                dt[0],
            ),
            wp.max(dt[0] * f_dec[0], dt_min[0]),
        )
        alpha[0] = wp.where(
            vf_mask,
            wp.where(
                n_steps_positive_mask,
                alpha[0] * f_alpha[0],
                alpha[0],
            ),
            alpha_start[0],
        )


@wp.kernel
def _fire_update_params_downhill_batch_idx_kernel(
    energy: wp.array(dtype=Any),
    energy_last: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    positions_last: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    velocities_last: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    vv: wp.array(dtype=Any),
    ff: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
):
    r"""FIRE parameter update (downhill; batch_idx).

    Performs per-system downhill check (rolling back positions/velocities if energy
    increased), computes per-system diagnostic scalars, applies velocity mixing,
    and updates per-system `dt`, `alpha`, and `n_steps_positive` **without**
    performing any MD step.

    Parameters
    ----------
    energy : wp.array, shape (B,), dtype=wp.float*
        Per-system current energies. Rolled back if uphill.
    energy_last : wp.array, shape (B,), dtype=wp.float*
        Per-system last accepted energies.
    positions : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated atomic positions (in-place; rolled back if uphill).
    positions_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated last accepted positions.
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated atomic velocities (in-place; rolled back if uphill, then mixed).
    velocities_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated last accepted velocities.
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces on atoms.
    alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE mixing parameter \\(\\alpha\\).
    dt : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE timestep \\(\\Delta t\\).
    alpha_start : wp.array, shape (B,), dtype=wp.float*
        Per-system reset value for \\(\\alpha\\).
    f_alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system multiplicative decay factor for \\(\\alpha\\).
    dt_min : wp.array, shape (B,), dtype=wp.float*
        Per-system minimum allowed timestep.
    dt_max : wp.array, shape (B,), dtype=wp.float*
        Per-system maximum allowed timestep.
    n_steps_positive : wp.array, shape (B,), dtype=wp.int32
        Per-system counter for consecutive steps with \\(v\\cdot f > 0\\).
    n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system threshold for when to start increasing `dt`.
    f_dec : wp.array, shape (B,), dtype=wp.float*
        Per-system decay factor for `dt` when uphill or \\(v\\cdot f \\le 0\\).
    f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system growth factor for `dt` after `n_min` positive steps.
    vv : wp.array, shape (B,), dtype=wp.float*
        Per-system accumulator for \\(\\sum v\\cdot v\\). Zeroed internally before each use.
    ff : wp.array, shape (B,), dtype=wp.float*
        Per-system accumulator for \\(\\sum f\\cdot f\\). Zeroed internally before each use.
    vf : wp.array, shape (B,), dtype=wp.float*
        Per-system accumulator for \\(\\sum v\\cdot f\\). Zeroed internally before each use.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32
        System index for each atom.

    Launch Grid
    -----------
    dim = [num_atoms_total]

    Notes
    -----
    - This kernel does NOT perform the MD step (velocity integration + position update).
    - Uphill detection is per system: each atom checks `energy[sys]` vs `energy_last[sys]`.
    - `vf/vv/ff` are per-system accumulators and zeroed internally before each use.
    """
    atom_idx = wp.tid()
    sys = batch_idx[atom_idx]

    is_uphill = False
    if energy[sys] > energy_last[sys]:
        is_uphill = True
        energy[sys] = energy_last[sys]
        positions[atom_idx] = positions_last[atom_idx]
        velocities[atom_idx] = velocities_last[atom_idx]
    else:
        energy_last[sys] = energy[sys]
        positions_last[atom_idx] = positions[atom_idx]
        velocities_last[atom_idx] = velocities[atom_idx]

    vf_val, vv_val, ff_val = compute_vf_vv_ff(velocities[atom_idx], forces[atom_idx])
    vf[sys] += vf_val
    vv[sys] += vv_val
    ff[sys] += ff_val

    vf_mask = (vf[sys] > type(dt[sys])(0.0)) and (not is_uphill)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    velocities[atom_idx] = wp.where(
        vf_mask,
        (type(dt[sys])(1.0) - alpha[sys]) * velocities[atom_idx]
        + (alpha[sys] * forces[atom_idx] * wp.sqrt(vv[sys] / ff[sys])),
        type(dt[sys])(0.0) * velocities[atom_idx],
    )

    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )


@wp.kernel
def _fire_update_params_downhill_ptr_kernel(
    energy: wp.array(dtype=Any),
    energy_last: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    positions_last: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    velocities_last: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    alpha_start: wp.array(dtype=Any),
    f_alpha: wp.array(dtype=Any),
    dt_min: wp.array(dtype=Any),
    dt_max: wp.array(dtype=Any),
    n_steps_positive: wp.array(dtype=wp.int32),
    n_min: wp.array(dtype=wp.int32),
    f_dec: wp.array(dtype=Any),
    f_inc: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
):
    r"""FIRE parameter update (downhill; ptr/CSR).

    Each thread owns a full system range `[atom_ptr[sys], atom_ptr[sys+1])` and
    performs the downhill check, computes diagnostic scalars, applies velocity
    mixing, and updates per-system `dt`, `alpha`, and `n_steps_positive`
    **without** performing any MD step.

    Parameters
    ----------
    energy : wp.array, shape (B,), dtype=wp.float*
        Per-system current energies. Rolled back if uphill.
    energy_last : wp.array, shape (B,), dtype=wp.float*
        Per-system last accepted energies.
    positions : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated atomic positions (in-place; rolled back if uphill).
    positions_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated last accepted positions.
    velocities : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated atomic velocities (in-place; rolled back if uphill, then mixed).
    velocities_last : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated last accepted velocities.
    forces : wp.array, shape (N_total,), dtype=wp.vec3*
        Concatenated forces on atoms.
    alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE mixing parameter \\(\\alpha\\).
    dt : wp.array, shape (B,), dtype=wp.float*
        Per-system FIRE timestep \\(\\Delta t\\).
    alpha_start : wp.array, shape (B,), dtype=wp.float*
        Per-system reset value for \\(\\alpha\\).
    f_alpha : wp.array, shape (B,), dtype=wp.float*
        Per-system multiplicative decay factor for \\(\\alpha\\).
    dt_min : wp.array, shape (B,), dtype=wp.float*
        Per-system minimum allowed timestep.
    dt_max : wp.array, shape (B,), dtype=wp.float*
        Per-system maximum allowed timestep.
    n_steps_positive : wp.array, shape (B,), dtype=wp.int32
        Per-system counter for consecutive steps with \\(v\\cdot f > 0\\).
    n_min : wp.array, shape (B,), dtype=wp.int32
        Per-system threshold for when to start increasing `dt`.
    f_dec : wp.array, shape (B,), dtype=wp.float*
        Per-system decay factor for `dt` when uphill or \\(v\\cdot f \\le 0\\).
    f_inc : wp.array, shape (B,), dtype=wp.float*
        Per-system growth factor for `dt` after `n_min` positive steps.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32
        CSR pointer giving the start/end atom indices for each system.

    Launch Grid
    -----------
    dim = [num_systems]

    Notes
    -----
    - This kernel does NOT perform the MD step (velocity integration + position update).
    - Each system is processed by a single thread, so no cross-thread synchronization
      is required for the diagnostic reductions or rollback.
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    # Downhill check
    is_uphill = False
    if energy[sys] > energy_last[sys]:
        is_uphill = True
        energy[sys] = energy_last[sys]
        for i in range(a0, a1):
            positions[i] = positions_last[i]
            velocities[i] = velocities_last[i]
    else:
        energy_last[sys] = energy[sys]
        for i in range(a0, a1):
            positions_last[i] = positions[i]
            velocities_last[i] = velocities[i]

    # Compute diagnostics
    vf = type(dt[sys])(0.0)
    vv = type(dt[sys])(0.0)
    ff = type(dt[sys])(0.0)
    for i in range(a0, a1):
        vf += wp.dot(velocities[i], forces[i])
        vv += wp.dot(velocities[i], velocities[i])
        ff += wp.dot(forces[i], forces[i])

    vf_mask = (vf > type(dt[sys])(0.0)) and (not is_uphill)
    n_steps_positive[sys] = wp.where(vf_mask, n_steps_positive[sys] + 1, 0)
    n_steps_positive_mask = n_steps_positive[sys] >= n_min[sys]

    # Guard against division by zero when forces are zero
    zero = type(dt[sys])(0.0)
    if ff > zero:
        ratio = wp.sqrt(vv / ff)
    else:
        ratio = zero

    # Velocity mixing
    for i in range(a0, a1):
        velocities[i] = wp.where(
            vf_mask,
            (type(dt[sys])(1.0) - alpha[sys]) * velocities[i]
            + (alpha[sys] * forces[i] * ratio),
            zero * velocities[i],
        )

    dt[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            wp.min(dt[sys] * f_inc[sys], dt_max[sys]),
            dt[sys],
        ),
        wp.max(dt[sys] * f_dec[sys], dt_min[sys]),
    )
    alpha[sys] = wp.where(
        vf_mask,
        wp.where(
            n_steps_positive_mask,
            alpha[sys] * f_alpha[sys],
            alpha[sys],
        ),
        alpha_start[sys],
    )


# =============================================================================
# Kernel Overloads for Explicit Typing
# =============================================================================

_T = [wp.float32, wp.float64]  # Scalar types
_V = [wp.vec3f, wp.vec3d]  # Vector types

# Step kernels (with MD integration)
_fire_step_no_downhill_kernel_overload = {}
_fire_step_no_downhill_batch_idx_kernel_overload = {}
_fire_step_no_downhill_ptr_kernel_overload = {}
_fire_step_downhill_kernel_overload = {}
_fire_step_downhill_batch_idx_kernel_overload = {}
_fire_step_downhill_ptr_kernel_overload = {}

# RLE-based kernels (race-condition free)
_fire_reduce_batch_idx_rle_kernel_overload = {}
_fire_update_batch_idx_kernel_overload = {}
_fire_uphill_check_kernel_overload = {}
_fire_revert_and_reduce_kernel_overload = {}
_fire_update_downhill_batch_idx_kernel_overload = {}

# Update-only kernels (no MD integration)
_fire_update_params_no_downhill_kernel_overload = {}
_fire_update_params_no_downhill_batch_idx_kernel_overload = {}
_fire_update_params_no_downhill_ptr_kernel_overload = {}
_fire_update_params_downhill_kernel_overload = {}
_fire_update_params_downhill_batch_idx_kernel_overload = {}
_fire_update_params_downhill_ptr_kernel_overload = {}

for t, v in zip(_T, _V):
    # Step kernels: no-downhill variants
    _fire_step_no_downhill_kernel_overload[v] = wp.overload(
        _fire_step_no_downhill_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # masses
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep (always f32)
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
        ],
    )

    _fire_step_no_downhill_batch_idx_kernel_overload[v] = wp.overload(
        _fire_step_no_downhill_batch_idx_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # masses
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
        ],
    )

    _fire_step_no_downhill_ptr_kernel_overload[v] = wp.overload(
        _fire_step_no_downhill_ptr_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # masses
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=wp.int32),  # atom_ptr
        ],
    )

    # RLE-based reduction kernel (race-condition free)
    _fire_reduce_batch_idx_rle_kernel_overload[v] = wp.overload(
        _fire_reduce_batch_idx_rle_kernel,
        [
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.int32,  # N
            wp.int32,  # elems_per_thread
        ],
    )

    # RLE-based update kernel (race-condition free)
    _fire_update_batch_idx_kernel_overload[v] = wp.overload(
        _fire_update_batch_idx_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # masses
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
        ],
    )

    # RLE-based downhill kernels (race-condition free)
    _fire_uphill_check_kernel_overload[v] = wp.overload(
        _fire_uphill_check_kernel,
        [
            wp.array(dtype=t),  # energy
            wp.array(dtype=t),  # energy_last
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # uphill_flag
        ],
    )

    _fire_revert_and_reduce_kernel_overload[v] = wp.overload(
        _fire_revert_and_reduce_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=v),  # positions_last
            wp.array(dtype=v),  # velocities_last
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=wp.int32),  # uphill_flag
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.int32,  # N
            wp.int32,  # elems_per_thread
        ],
    )

    _fire_update_downhill_batch_idx_kernel_overload[v] = wp.overload(
        _fire_update_downhill_batch_idx_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # masses
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.array(dtype=wp.int32),  # uphill_flag
        ],
    )

    # Step kernels: downhill variants
    _fire_step_downhill_kernel_overload[v] = wp.overload(
        _fire_step_downhill_kernel,
        [
            wp.array(dtype=t),  # energy
            wp.array(dtype=v),  # forces
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=t),  # masses
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # energy_last
            wp.array(dtype=v),  # positions_last
            wp.array(dtype=v),  # velocities_last
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
        ],
    )

    _fire_step_downhill_batch_idx_kernel_overload[v] = wp.overload(
        _fire_step_downhill_batch_idx_kernel,
        [
            wp.array(dtype=t),  # energy
            wp.array(dtype=v),  # forces
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=t),  # masses
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # energy_last
            wp.array(dtype=v),  # positions_last
            wp.array(dtype=v),  # velocities_last
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
        ],
    )

    _fire_step_downhill_ptr_kernel_overload[v] = wp.overload(
        _fire_step_downhill_ptr_kernel,
        [
            wp.array(dtype=t),  # energy
            wp.array(dtype=v),  # forces
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # velocities
            wp.array(dtype=t),  # masses
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=t),  # maxstep
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # energy_last
            wp.array(dtype=v),  # positions_last
            wp.array(dtype=v),  # velocities_last
            wp.array(dtype=wp.int32),  # atom_ptr
        ],
    )

    # Update-only kernels: no-downhill variants
    _fire_update_params_no_downhill_kernel_overload[v] = wp.overload(
        _fire_update_params_no_downhill_kernel,
        [
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.array(dtype=t),  # vf
        ],
    )

    _fire_update_params_no_downhill_batch_idx_kernel_overload[v] = wp.overload(
        _fire_update_params_no_downhill_batch_idx_kernel,
        [
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vf
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.array(dtype=wp.int32),  # batch_idx
        ],
    )

    _fire_update_params_no_downhill_ptr_kernel_overload[v] = wp.overload(
        _fire_update_params_no_downhill_ptr_kernel,
        [
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=wp.int32),  # atom_ptr
        ],
    )

    # Update-only kernels: downhill variants
    _fire_update_params_downhill_kernel_overload[v] = wp.overload(
        _fire_update_params_downhill_kernel,
        [
            wp.array(dtype=t),  # energy
            wp.array(dtype=t),  # energy_last
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # positions_last
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # velocities_last
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.array(dtype=t),  # vf
        ],
    )

    _fire_update_params_downhill_batch_idx_kernel_overload[v] = wp.overload(
        _fire_update_params_downhill_batch_idx_kernel,
        [
            wp.array(dtype=t),  # energy
            wp.array(dtype=t),  # energy_last
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # positions_last
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # velocities_last
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=t),  # vv
            wp.array(dtype=t),  # ff
            wp.array(dtype=t),  # vf
            wp.array(dtype=wp.int32),  # batch_idx
        ],
    )

    _fire_update_params_downhill_ptr_kernel_overload[v] = wp.overload(
        _fire_update_params_downhill_ptr_kernel,
        [
            wp.array(dtype=t),  # energy
            wp.array(dtype=t),  # energy_last
            wp.array(dtype=v),  # positions
            wp.array(dtype=v),  # positions_last
            wp.array(dtype=v),  # velocities
            wp.array(dtype=v),  # velocities_last
            wp.array(dtype=v),  # forces
            wp.array(dtype=t),  # alpha
            wp.array(dtype=t),  # dt
            wp.array(dtype=t),  # alpha_start
            wp.array(dtype=t),  # f_alpha
            wp.array(dtype=t),  # dt_min
            wp.array(dtype=t),  # dt_max
            wp.array(dtype=wp.int32),  # n_steps_positive
            wp.array(dtype=wp.int32),  # n_min
            wp.array(dtype=t),  # f_dec
            wp.array(dtype=t),  # f_inc
            wp.array(dtype=wp.int32),  # atom_ptr
        ],
    )


# =============================================================================
# Public API: Unified FIRE Step Functions
# =============================================================================


def fire_step(
    # Core DOFs (required)
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    # FIRE control parameters (required)
    alpha: wp.array,
    dt: wp.array,
    alpha_start: wp.array,
    f_alpha: wp.array,
    dt_min: wp.array,
    dt_max: wp.array,
    maxstep: wp.array,
    n_steps_positive: wp.array,
    n_min: wp.array,
    f_dec: wp.array,
    f_inc: wp.array,
    # Scratch arrays
    uphill_flag: wp.array,
    # Accumulators (required for single/batch_idx; ignored for ptr)
    vf: wp.array = None,
    vv: wp.array = None,
    ff: wp.array = None,
    # Batching (mutually exclusive - if neither, assumes single system)
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    # Downhill check (optional - provide ALL or NONE)
    energy: wp.array = None,
    energy_last: wp.array = None,
    positions_last: wp.array = None,
    velocities_last: wp.array = None,
    # Device
    device: str = None,
) -> None:
    """
    Unified FIRE optimization step with MD integration.

    This function dispatches to the appropriate kernel based on:
    - Batching mode: single system, batch_idx, or atom_ptr
    - Downhill check: enabled if all downhill arrays are provided

    Parameters
    ----------
    positions : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*
        Atomic positions (modified in-place).
    velocities : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*
        Atomic velocities (modified in-place).
    forces : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*
        Forces on atoms.
    masses : wp.array, shape (N,) or (N_total,), dtype=wp.float*
        Per-atom masses.
    alpha : wp.array, shape (1,) or (B,), dtype=wp.float*
        FIRE mixing parameter.
    dt : wp.array, shape (1,) or (B,), dtype=wp.float*
        FIRE timestep.
    alpha_start : wp.array, shape (1,) or (B,), dtype=wp.float*
        Reset value for alpha.
    f_alpha : wp.array, shape (1,) or (B,), dtype=wp.float*
        Alpha decay factor.
    dt_min : wp.array, shape (1,) or (B,), dtype=wp.float*
        Minimum timestep.
    dt_max : wp.array, shape (1,) or (B,), dtype=wp.float*
        Maximum timestep.
    maxstep : wp.array, shape (1,) or (B,), dtype=wp.float*
        Maximum displacement per step.
    n_steps_positive : wp.array, shape (1,) or (B,), dtype=wp.int32
        Counter for consecutive positive power steps.
    n_min : wp.array, shape (1,) or (B,), dtype=wp.int32
        Steps before dt increase / alpha decrease.
    f_dec : wp.array, shape (1,) or (B,), dtype=wp.float*
        Timestep decrease factor.
    f_inc : wp.array, shape (1,) or (B,), dtype=wp.float*
        Timestep increase factor.
    vf, vv, ff : wp.array, shape (1,) or (B,), dtype=wp.float*
        Accumulators for diagnostics. Zeroed internally before each use.
        Required for single/batch_idx modes. Ignored for atom_ptr mode.
    uphill_flag : wp.array, shape (B,), dtype=wp.int32, optional
        Scratch array for uphill detection. Shape (B,) where B = num_systems.
        Only used when downhill_enabled=True and batch_idx is provided.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32, optional
        System index per atom. If provided, uses batch_idx kernel.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32, optional
        CSR pointers for atom ranges. If provided, uses ptr kernel.
    energy : wp.array, shape (1,) or (B,), dtype=wp.float*, optional
        Current energies (for downhill check).
    energy_last : wp.array, shape (1,) or (B,), dtype=wp.float*, optional
        Last accepted energies (for downhill check).
    positions_last : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*, optional
        Last accepted positions (for downhill rollback).
    velocities_last : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*, optional
        Last accepted velocities (for downhill rollback).
    device : str, optional
        Warp device.

    Examples
    --------
    Single system (no downhill):

    >>> fire_step(positions, velocities, forces, masses,
    ...           alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
    ...           maxstep, n_steps_positive, n_min, f_dec, f_inc,
    ...           vf, vv, ff)

    Batched with batch_idx:

    >>> fire_step(positions, velocities, forces, masses,
    ...           alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
    ...           maxstep, n_steps_positive, n_min, f_dec, f_inc,
    ...           vf, vv, ff, batch_idx=batch_idx)

    Batched with atom_ptr:

    >>> fire_step(positions, velocities, forces, masses,
    ...           alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
    ...           maxstep, n_steps_positive, n_min, f_dec, f_inc,
    ...           atom_ptr=atom_ptr)

    With downhill check:

    >>> fire_step(positions, velocities, forces, masses,
    ...           alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
    ...           maxstep, n_steps_positive, n_min, f_dec, f_inc,
    ...           vf, vv, ff,
    ...           energy=energy, energy_last=energy_last,
    ...           positions_last=positions_last, velocities_last=velocities_last)
    """
    if device is None:
        device = positions.device

    if vf is not None:
        vf.zero_()
        vv.zero_()
        ff.zero_()

    num_atoms = positions.shape[0]
    vec_dtype = positions.dtype

    # Determine batching mode
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Cannot specify both batch_idx and atom_ptr")

    # Determine if downhill check is enabled
    downhill_arrays = [energy, energy_last, positions_last, velocities_last]
    downhill_enabled = all(arr is not None for arr in downhill_arrays)
    if any(arr is not None for arr in downhill_arrays) and not downhill_enabled:
        raise ValueError(
            "For downhill check, must provide ALL of: "
            "energy, energy_last, positions_last, velocities_last"
        )

    # Dispatch to appropriate kernel
    if atom_ptr is not None:
        # PTR mode - one thread per system
        num_systems = atom_ptr.shape[0] - 1
        if downhill_enabled:
            wp.launch(
                _fire_step_downhill_ptr_kernel_overload[vec_dtype],
                dim=num_systems,
                inputs=[
                    energy,
                    forces,
                    positions,
                    velocities,
                    masses,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    maxstep,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    energy_last,
                    positions_last,
                    velocities_last,
                    atom_ptr,
                ],
                device=device,
            )
        else:
            wp.launch(
                _fire_step_no_downhill_ptr_kernel_overload[vec_dtype],
                dim=num_systems,
                inputs=[
                    positions,
                    velocities,
                    forces,
                    masses,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    maxstep,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    atom_ptr,
                ],
                device=device,
            )

    elif batch_idx is not None:
        # BATCH_IDX mode - one thread per atom
        num_systems = dt.shape[0]
        if vf is None or vv is None or ff is None:
            raise ValueError("vf, vv, ff accumulators required for batch_idx mode")
        if downhill_enabled:
            # Three-kernel RLE approach for race-free downhill batch_idx mode

            # Kernel 1: Uphill check
            wp.launch(
                _fire_uphill_check_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[
                    energy,
                    energy_last,
                    batch_idx,
                    uphill_flag,
                ],
                device=device,
            )

            # Kernel 2: Revert if uphill + RLE reduction
            sm = max(device.sm_count, 1) if hasattr(device, "sm_count") else 1
            ept = _compute_ept(num_atoms, sm, is_vec3=True)
            dim_reduce = (num_atoms + ept - 1) // ept

            wp.launch(
                _fire_revert_and_reduce_kernel_overload[vec_dtype],
                dim=dim_reduce,
                inputs=[
                    positions,
                    velocities,
                    forces,
                    positions_last,
                    velocities_last,
                    batch_idx,
                    uphill_flag,
                    vf,
                    vv,
                    ff,
                    num_atoms,
                    ept,
                ],
                device=device,
            )

            # Kernel 3: Parameter update + velocity mixing
            wp.launch(
                _fire_update_downhill_batch_idx_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[
                    positions,
                    velocities,
                    forces,
                    masses,
                    batch_idx,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    maxstep,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vf,
                    vv,
                    ff,
                    uphill_flag,
                ],
                device=device,
            )
        else:
            # Two-kernel RLE approach for race-free batch_idx mode
            # Kernel 1: RLE-based reduction
            sm = max(device.sm_count, 1) if hasattr(device, "sm_count") else 1
            ept = _compute_ept(num_atoms, sm, is_vec3=True)
            dim_reduce = (num_atoms + ept - 1) // ept

            wp.launch(
                _fire_reduce_batch_idx_rle_kernel_overload[vec_dtype],
                dim=dim_reduce,
                inputs=[
                    velocities,
                    forces,
                    batch_idx,
                    vf,
                    vv,
                    ff,
                    num_atoms,
                    ept,
                ],
                device=device,
            )

            # Kernel 2: Parameter update + velocity mixing + position update
            wp.launch(
                _fire_update_batch_idx_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[
                    positions,
                    velocities,
                    forces,
                    masses,
                    batch_idx,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    maxstep,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vf,
                    vv,
                    ff,
                ],
                device=device,
            )

    else:
        # SINGLE SYSTEM mode
        if vf is None or vv is None or ff is None:
            raise ValueError("vf, vv, ff accumulators required for single system mode")
        if downhill_enabled:
            wp.launch(
                _fire_step_downhill_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[
                    energy,
                    forces,
                    positions,
                    velocities,
                    masses,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    maxstep,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    energy_last,
                    positions_last,
                    velocities_last,
                    vf,
                    vv,
                    ff,
                ],
                device=device,
            )
        else:
            wp.launch(
                _fire_step_no_downhill_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[
                    positions,
                    velocities,
                    forces,
                    masses,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    maxstep,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vf,
                    vv,
                    ff,
                ],
                device=device,
            )


def fire_update(
    # Core arrays (required)
    velocities: wp.array,
    forces: wp.array,
    # FIRE control parameters (required)
    alpha: wp.array,
    dt: wp.array,
    alpha_start: wp.array,
    f_alpha: wp.array,
    dt_min: wp.array,
    dt_max: wp.array,
    n_steps_positive: wp.array,
    n_min: wp.array,
    f_dec: wp.array,
    f_inc: wp.array,
    # Accumulators (required for single/batch_idx; ignored for ptr)
    vf: wp.array = None,
    vv: wp.array = None,
    ff: wp.array = None,
    # Batching (mutually exclusive)
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    # Downhill check (optional - provide ALL or NONE)
    energy: wp.array = None,
    energy_last: wp.array = None,
    positions: wp.array = None,
    positions_last: wp.array = None,
    velocities_last: wp.array = None,
    # Device
    device: str = None,
) -> None:
    """
    FIRE parameter update and velocity mixing WITHOUT MD integration.

    Use this for variable-cell optimization where you want to:
    1. Pack atomic + cell DOFs into extended arrays
    2. Apply FIRE velocity mixing to extended velocities
    3. Perform your own MD step (e.g., with cell-aware position scaling)

    This function dispatches to the appropriate "update params" kernel based on:
    - Batching mode: single system, batch_idx, or atom_ptr
    - Downhill check: enabled if all downhill arrays are provided

    Parameters
    ----------
    velocities : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*
        Velocities (modified in-place with FIRE mixing).
    forces : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*
        Forces.
    alpha : wp.array, shape (1,) or (B,), dtype=wp.float*
        FIRE mixing parameter.
    dt : wp.array, shape (1,) or (B,), dtype=wp.float*
        FIRE timestep.
    alpha_start : wp.array, shape (1,) or (B,), dtype=wp.float*
        Reset value for alpha.
    f_alpha : wp.array, shape (1,) or (B,), dtype=wp.float*
        Alpha decay factor.
    dt_min : wp.array, shape (1,) or (B,), dtype=wp.float*
        Minimum timestep.
    dt_max : wp.array, shape (1,) or (B,), dtype=wp.float*
        Maximum timestep.
    n_steps_positive : wp.array, shape (1,) or (B,), dtype=wp.int32
        Counter for consecutive positive power steps.
    n_min : wp.array, shape (1,) or (B,), dtype=wp.int32
        Steps before dt increase / alpha decrease.
    f_dec : wp.array, shape (1,) or (B,), dtype=wp.float*
        Timestep decrease factor.
    f_inc : wp.array, shape (1,) or (B,), dtype=wp.float*
        Timestep increase factor.
    vf, vv, ff : wp.array, shape (1,) or (B,), dtype=wp.float*
        Accumulators for diagnostics. Zeroed internally before each use.
        Required for single/batch_idx modes. Ignored for atom_ptr mode.
    batch_idx : wp.array, shape (N_total,), dtype=wp.int32, optional
        System index per atom. If provided, uses batch_idx kernel.
    atom_ptr : wp.array, shape (B+1,), dtype=wp.int32, optional
        CSR pointers for atom ranges. If provided, uses ptr kernel.
    energy : wp.array, shape (1,) or (B,), dtype=wp.float*, optional
        Current energies (for downhill check).
    energy_last : wp.array, shape (1,) or (B,), dtype=wp.float*, optional
        Last accepted energies (for downhill check).
    positions : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*, optional
        Positions (for downhill rollback). Required if downhill enabled.
    positions_last : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*, optional
        Last accepted positions (for downhill rollback).
    velocities_last : wp.array, shape (N,) or (N_total,), dtype=wp.vec3*, optional
        Last accepted velocities (for downhill rollback).
    device : str, optional
        Warp device.

    Examples
    --------
    Variable-cell optimization workflow:

    >>> # Pack extended arrays (atomic + cell DOFs)
    >>> ext_pos = pack_positions_with_cell(positions, cell)
    >>> ext_vel = pack_velocities_with_cell(velocities, cell_velocity)
    >>> ext_forces = pack_forces_with_cell(forces, cell_force)
    >>>
    >>> # FIRE velocity mixing only (no position update)
    >>> fire_update(ext_vel, ext_forces,
    ...             alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
    ...             n_steps_positive, n_min, f_dec, f_inc,
    ...             vf, vv, ff)
    >>>
    >>> # Perform your own MD step with cell-aware scaling
    >>> ext_vel += dt * ext_forces / ext_masses
    >>> ext_pos += dt * ext_vel  # (with maxstep capping)
    >>>
    >>> # Unpack results
    >>> positions, cell = unpack_positions_with_cell(ext_pos, num_atoms)
    """
    if device is None:
        device = velocities.device

    if vf is not None:
        vf.zero_()
        vv.zero_()
        ff.zero_()

    num_atoms = velocities.shape[0]

    # Determine batching mode
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Cannot specify both batch_idx and atom_ptr")

    # Determine if downhill check is enabled
    downhill_arrays = [energy, energy_last, positions, positions_last, velocities_last]
    downhill_enabled = all(arr is not None for arr in downhill_arrays)
    if any(arr is not None for arr in downhill_arrays) and not downhill_enabled:
        raise ValueError(
            "For downhill check, must provide ALL of: "
            "energy, energy_last, positions, positions_last, velocities_last"
        )

    vec_dtype = velocities.dtype

    # Dispatch to appropriate kernel
    if atom_ptr is not None:
        # PTR mode
        num_systems = atom_ptr.shape[0] - 1
        if downhill_enabled:
            wp.launch(
                _fire_update_params_downhill_ptr_kernel_overload[vec_dtype],
                dim=num_systems,
                inputs=[
                    energy,
                    energy_last,
                    positions,
                    positions_last,
                    velocities,
                    velocities_last,
                    forces,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    atom_ptr,
                ],
                device=device,
            )
        else:
            wp.launch(
                _fire_update_params_no_downhill_ptr_kernel_overload[vec_dtype],
                dim=num_systems,
                inputs=[
                    velocities,
                    forces,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    atom_ptr,
                ],
                device=device,
            )

    elif batch_idx is not None:
        # BATCH_IDX mode
        if vf is None or vv is None or ff is None:
            raise ValueError("vf, vv, ff accumulators required for batch_idx mode")
        if downhill_enabled:
            wp.launch(
                _fire_update_params_downhill_batch_idx_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[
                    energy,
                    energy_last,
                    positions,
                    positions_last,
                    velocities,
                    velocities_last,
                    forces,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vv,
                    ff,
                    vf,
                    batch_idx,
                ],
                device=device,
            )
        else:
            wp.launch(
                _fire_update_params_no_downhill_batch_idx_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[
                    velocities,
                    forces,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vf,
                    vv,
                    ff,
                    batch_idx,
                ],
                device=device,
            )

    else:
        # SINGLE SYSTEM mode
        if vf is None or vv is None or ff is None:
            raise ValueError("vf, vv, ff accumulators required for single system mode")
        if downhill_enabled:
            wp.launch(
                _fire_update_params_downhill_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[
                    energy,
                    energy_last,
                    positions,
                    positions_last,
                    velocities,
                    velocities_last,
                    forces,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vv,
                    ff,
                    vf,
                ],
                device=device,
            )
        else:
            wp.launch(
                _fire_update_params_no_downhill_kernel_overload[vec_dtype],
                dim=num_atoms,
                inputs=[
                    velocities,
                    forces,
                    alpha,
                    dt,
                    alpha_start,
                    f_alpha,
                    dt_min,
                    dt_max,
                    n_steps_positive,
                    n_min,
                    f_dec,
                    f_inc,
                    vv,
                    ff,
                    vf,
                ],
                device=device,
            )
