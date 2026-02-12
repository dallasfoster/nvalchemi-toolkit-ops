# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FIRE2 Optimizer Kernels
=======================

GPU-accelerated Warp kernels for the FIRE2 (Fast Inertial Relaxation
Engine v2) geometry optimizer.

This module provides three highly-fused kernels that implement a complete
FIRE2 step in only **3 kernel launches**, minimizing Python-side and
launch overhead.

FIRE2 ALGORITHM (Guenole et al., 2020)
=======================================

Given positions *r*, velocities *v*, and forces *f*:

1. Half-step velocity update:  v += f * dt
2. Compute power:  P = sum(v . f)  per system
3. Adaptive parameter update:
   - If P > 0: increment counter, optionally grow dt, shrink alpha
   - If P <= 0: reset counter, shrink dt, reset alpha
4. Velocity mixing:
   v = (1 - alpha) * v + alpha * sqrt(v.v / f.f) * f
5. Compute step:  step = v * dt
6. Uphill correction:
   if P <= 0:  step = -0.5 * dt * v_mixed;  v = 0
7. Step clamping + position update + coupled dt scaling

KERNEL STRUCTURE
================

Kernel 1 (_fire2_reduce_only):
    Runs-based triple inner-product reduction (vf, v.v, f.f) with
    deferred half-step computed in registers only (no velocity write).

Kernel 2 (_fire2_fused_mix_maxnorm):
    Fuses per-system parameter update, deferred half-step, velocity
    mixing, and runs-based max-norm reduction into a single launch.
    Each thread redundantly computes the parameter update for its
    segment from shared read-only inputs, avoiding inter-thread
    synchronization.

Kernel 3 (_fire2_clamp_apply_recompute):
    Recomputes step from mixed velocities, applies step clamping,
    position update, deferred velocity zeroing for uphill systems,
    and coupled dt scaling.

REFERENCES
==========

- Guenole et al. (2020). Comp. Mat. Sci. 175, 109584 (FIRE2)
- Bitzek et al. (2006). Phys. Rev. Lett. 97, 170201 (FIRE)
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ...segment_ops import _compute_ept

# =============================================================================
# Kernel 1: Triple inner-product reduction (deferred half-step)
# =============================================================================


@wp.kernel(enable_backward=False)
def _fire2_reduce_only_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    vf: wp.array(dtype=Any),
    v_sumsq: wp.array(dtype=Any),
    f_sumsq: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Triple inner-product reduction without writing velocities.

    Computes v_upd = v[i] + f[i] * dt[s] in registers only, then
    accumulates dot(v_upd, f), dot(v_upd, v_upd), dot(f, f) per segment.
    Velocities are NOT modified -- the half-step is deferred to the
    fused mixing kernel.

    Parameters
    ----------
    velocities : wp.array, shape (N,), dtype vec3*
        Atomic velocities, read-only.
    forces : wp.array, shape (N,), dtype vec3*
        Forces on atoms.
    dt : wp.array, shape (M,), dtype float*
        Per-system timestep.
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom.
    vf, v_sumsq, f_sumsq : wp.array, shape (M,), dtype float*
        OUTPUT accumulators. Must be zero-initialized by caller.
    N : int
        Total number of atoms.
    elems_per_thread : int
        Elements processed per thread.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    # First element -- compute v_upd in register, do NOT write back
    s_cur = batch_idx[start]
    v_upd = velocities[start] + forces[start] * dt[s_cur]
    acc_vf = wp.dot(v_upd, forces[start])
    acc_vv = wp.dot(v_upd, v_upd)
    acc_ff = wp.dot(forces[start], forces[start])

    for i in range(start + 1, end):
        s = batch_idx[i]
        v_upd = velocities[i] + forces[i] * dt[s]
        val_vf = wp.dot(v_upd, forces[i])
        val_vv = wp.dot(v_upd, v_upd)
        val_ff = wp.dot(forces[i], forces[i])
        if s == s_cur:
            acc_vf = acc_vf + val_vf
            acc_vv = acc_vv + val_vv
            acc_ff = acc_ff + val_ff
        else:
            wp.atomic_add(vf, s_cur, acc_vf)
            wp.atomic_add(v_sumsq, s_cur, acc_vv)
            wp.atomic_add(f_sumsq, s_cur, acc_ff)
            s_cur = s
            acc_vf = val_vf
            acc_vv = val_vv
            acc_ff = val_ff

    wp.atomic_add(vf, s_cur, acc_vf)
    wp.atomic_add(v_sumsq, s_cur, acc_vv)
    wp.atomic_add(f_sumsq, s_cur, acc_ff)


# =============================================================================
# Kernel 2: Fused param update + deferred halfstep + mix + max-norm
# =============================================================================


@wp.kernel(enable_backward=False)
def _fire2_fused_mix_maxnorm_kernel(
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    vf: wp.array(dtype=Any),
    v_sumsq: wp.array(dtype=Any),
    f_sumsq: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    nsteps_inc: wp.array(dtype=wp.int32),
    max_norm: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
    delaystep: wp.int32,
    dtgrow: Any,
    dtshrink: Any,
    alphashrink: Any,
    alpha0: Any,
    tmax: Any,
    tmin: Any,
):
    """Fused parameter update + deferred half-step + velocity mixing + max-norm.

    Each thread redundantly computes the per-system parameter update
    from shared read-only inputs, avoiding inter-thread synchronization.
    The deferred half-step (from kernel 1) is algebraically combined
    with the velocity mix:

      v[i] = mix_a * v[i] + (mix_a * dt_old + mix_b) * f[i]

    Only the first atom per segment writes updated alpha, dt, nsteps_inc.

    Parameters
    ----------
    velocities : wp.array, shape (N,), dtype vec3*
        Atomic velocities, modified in-place (halfstep + mix applied).
        Must still hold pre-halfstep values (kernel 1 did not write them).
    forces : wp.array, shape (N,), dtype vec3*
        Read-only.
    dt : wp.array, shape (M,), dtype float*
        Per-system timestep. Updated in-place (first atom per segment).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom.
    vf, v_sumsq, f_sumsq : wp.array, shape (M,), dtype float*
        Inner-product accumulators from kernel 1 (read-only here).
    alpha : wp.array, shape (M,), dtype float*
        FIRE2 mixing parameter. Updated in-place (first atom per segment).
    nsteps_inc : wp.array, shape (M,), dtype int32
        Consecutive positive-power counter. Updated (first atom per segment).
    max_norm : wp.array, shape (M,), dtype float*
        OUTPUT max step norm per segment. Must be zero-initialized.
    N : int
        Total number of atoms.
    elems_per_thread : int
        Elements processed per thread.
    delaystep : int
        Minimum positive steps before dt growth.
    dtgrow, dtshrink, alphashrink, alpha0, tmax, tmin : float*
        FIRE2 hyperparameters.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = batch_idx[start]

    # --- Redundant param-update computation for the first segment ---
    _vf = vf[s_cur]
    _vv = v_sumsq[s_cur]
    _ff = f_sumsq[s_cur]
    _a = alpha[s_cur]
    _dt = dt[s_cur]
    dt_old = _dt  # pre-update dt for the deferred half-step

    zero = type(_dt)(0.0)
    one = type(_dt)(1.0)
    w_inc = _vf > zero

    if w_inc:
        _nsi = nsteps_inc[s_cur] + 1
        if _nsi > delaystep:
            _dt = wp.min(dtgrow * _dt, tmax)
            _a = alphashrink * _a
    else:
        _nsi = 0
        _a = alpha0
        _dt = wp.max(dtshrink * _dt, tmin)

    # First atom per segment writes updated params
    if start == 0 or batch_idx[start - 1] != s_cur:
        alpha[s_cur] = _a
        dt[s_cur] = _dt
        nsteps_inc[s_cur] = _nsi

    if _ff > zero:
        ratio = wp.sqrt(_vv / _ff)
    else:
        ratio = zero
    mix_a = one - _a
    mix_b = _a * ratio
    w_dec = not w_inc

    # --- Process first element: deferred halfstep + mix (algebraic combo) ---
    f_coeff = mix_a * dt_old + mix_b
    velocities[start] = mix_a * velocities[start] + f_coeff * forces[start]
    if w_dec:
        max_val = wp.length(type(_dt)(-0.5) * _dt * velocities[start])
    else:
        max_val = wp.length(_dt * velocities[start])

    for i in range(start + 1, end):
        s = batch_idx[i]
        if s != s_cur:
            # Flush max_norm for previous segment
            wp.atomic_max(max_norm, s_cur, max_val)
            s_cur = s

            # --- Redundant param-update computation for new segment ---
            _vf = vf[s]
            _vv = v_sumsq[s]
            _ff = f_sumsq[s]
            _a = alpha[s]
            _dt = dt[s]
            dt_old = _dt

            w_inc = _vf > zero
            if w_inc:
                _nsi = nsteps_inc[s] + 1
                if _nsi > delaystep:
                    _dt = wp.min(dtgrow * _dt, tmax)
                    _a = alphashrink * _a
            else:
                _nsi = 0
                _a = alpha0
                _dt = wp.max(dtshrink * _dt, tmin)

            if batch_idx[i - 1] != s:
                alpha[s] = _a
                dt[s] = _dt
                nsteps_inc[s] = _nsi

            if _ff > zero:
                ratio = wp.sqrt(_vv / _ff)
            else:
                ratio = zero
            mix_a = one - _a
            mix_b = _a * ratio
            w_dec = not w_inc
            f_coeff = mix_a * dt_old + mix_b
            max_val = type(_dt)(0.0)

        # Deferred halfstep + mix (algebraic combo)
        velocities[i] = mix_a * velocities[i] + f_coeff * forces[i]
        if w_dec:
            norm = wp.length(type(_dt)(-0.5) * _dt * velocities[i])
        else:
            norm = wp.length(_dt * velocities[i])
        max_val = wp.max(max_val, norm)

    wp.atomic_max(max_norm, s_cur, max_val)


# =============================================================================
# Kernel 3: Step recompute + clamping + position update + velocity zeroing
# =============================================================================


@wp.kernel(enable_backward=False)
def _fire2_clamp_apply_recompute_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    dt: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    max_norm: wp.array(dtype=Any),
    vf: wp.array(dtype=Any),
    maxstep: Any,
):
    """Step recompute, clamping, position update, velocity zeroing, dt scaling.

    Recomputes the step from mixed velocities rather than reading from a
    pre-computed step buffer.  Also performs deferred velocity zeroing for
    uphill systems (which the mixing kernel skipped).

    For each atom:
      local_dt = dt[s]  (snapshot before any write)
      inv = min(1.0, maxstep / max_norm[s])
      if vf[s] <= 0 (uphill):
          step = -0.5 * local_dt * v[i]
          v[i] = 0
      else:
          step = local_dt * v[i]
      pos[i] += step * inv
      dt[s] = local_dt * inv  (first atom per segment only)

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype vec3*
        Atomic positions, modified in-place.
    velocities : wp.array, shape (N,), dtype vec3*
        Atomic velocities, modified in-place (zeroed for uphill systems).
    dt : wp.array, shape (M,), dtype float*
        Per-system timestep, modified in-place (clamped proportionally).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom.
    max_norm : wp.array, shape (M,), dtype float*
        Max step norm per segment from the mixing kernel.
    vf : wp.array, shape (M,), dtype float*
        v.f inner product per segment from kernel 1.  Uphill if vf[s] <= 0.
    maxstep : float*
        Maximum allowed step size (scalar hyperparameter).

    Launch Grid
    -----------
    dim = N (total atoms)
    """
    tid = wp.tid()
    s = batch_idx[tid]
    # Snapshot dt before any thread writes to it (race-condition guard)
    local_dt = dt[s]
    mn = max_norm[s]
    inv = wp.min(type(mn)(1.0), maxstep / mn)

    if vf[s] <= type(mn)(0.0):
        local_step = type(mn)(-0.5) * local_dt * velocities[tid]
        velocities[tid] = type(velocities[tid])()
    else:
        local_step = local_dt * velocities[tid]

    positions[tid] = positions[tid] + local_step * inv
    # Only first atom in segment updates dt (idx is sorted)
    if tid == 0 or batch_idx[tid - 1] != s:
        dt[s] = local_dt * inv


# =============================================================================
# Overloads
# =============================================================================

_T = [wp.float32, wp.float64]
_V = [wp.vec3f, wp.vec3d]

_fire2_reduce_only_overloads = {}
_fire2_fused_mix_maxnorm_overloads = {}
_fire2_clamp_apply_recompute_overloads = {}

for _t, _v in zip(_T, _V):
    _fire2_reduce_only_overloads[_v] = wp.overload(
        _fire2_reduce_only_kernel,
        [
            wp.array(dtype=_v),  # velocities
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_t),  # dt
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=_t),  # vf
            wp.array(dtype=_t),  # v_sumsq
            wp.array(dtype=_t),  # f_sumsq
            wp.int32,  # N
            wp.int32,  # elems_per_thread
        ],
    )

    _fire2_fused_mix_maxnorm_overloads[_v] = wp.overload(
        _fire2_fused_mix_maxnorm_kernel,
        [
            wp.array(dtype=_v),  # velocities
            wp.array(dtype=_v),  # forces
            wp.array(dtype=_t),  # dt
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=_t),  # vf
            wp.array(dtype=_t),  # v_sumsq
            wp.array(dtype=_t),  # f_sumsq
            wp.array(dtype=_t),  # alpha
            wp.array(dtype=wp.int32),  # nsteps_inc
            wp.array(dtype=_t),  # max_norm
            wp.int32,  # N
            wp.int32,  # elems_per_thread
            wp.int32,  # delaystep
            _t,  # dtgrow
            _t,  # dtshrink
            _t,  # alphashrink
            _t,  # alpha0
            _t,  # tmax
            _t,  # tmin
        ],
    )

    _fire2_clamp_apply_recompute_overloads[_v] = wp.overload(
        _fire2_clamp_apply_recompute_kernel,
        [
            wp.array(dtype=_v),  # positions
            wp.array(dtype=_v),  # velocities
            wp.array(dtype=_t),  # dt
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array(dtype=_t),  # max_norm
            wp.array(dtype=_t),  # vf (v.f inner product)
            _t,  # maxstep
        ],
    )


# =============================================================================
# Public API
# =============================================================================


def fire2_step(
    # Per-atom arrays (N,)
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    batch_idx: wp.array,
    # Per-system state (M,)
    alpha: wp.array,
    dt: wp.array,
    nsteps_inc: wp.array,
    # Scratch buffers (M,)
    vf: wp.array,
    v_sumsq: wp.array,
    f_sumsq: wp.array,
    max_norm: wp.array,
    # Hyperparameters (Python scalars)
    delaystep: int = 60,
    dtgrow: float = 1.05,
    dtshrink: float = 0.75,
    alphashrink: float = 0.985,
    alpha0: float = 0.09,
    tmax: float = 0.08,
    tmin: float = 0.005,
    maxstep: float = 0.1,
    # Device
    device: str = None,
) -> None:
    """Complete FIRE2 optimization step.

    Modifies *positions*, *velocities*, *alpha*, *dt*, and *nsteps_inc* in-place.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic positions.
    velocities : wp.array, shape (N,), dtype vec3f/vec3d
        Atomic velocities.
    forces : wp.array, shape (N,), dtype vec3f/vec3d
        Forces on atoms (read-only).
    batch_idx : wp.array, shape (N,), dtype int32
        Sorted system index per atom (required).
    alpha : wp.array, shape (M,), dtype float*
        FIRE2 mixing parameter.
    dt : wp.array, shape (M,), dtype float*
        Per-system timestep.
    nsteps_inc : wp.array, shape (M,), dtype int32
        Consecutive positive-power step counter.
    vf, v_sumsq, f_sumsq, max_norm : wp.array, shape (M,), dtype float*
        Scratch buffers for reductions. Must be zero-initialized.
    delaystep : int
        Minimum positive steps before dt growth.
    dtgrow, dtshrink : float
        Timestep growth/shrink factors.
    alphashrink : float
        Alpha decay factor.
    alpha0 : float
        Alpha reset value.
    tmax, tmin : float
        Timestep bounds.
    maxstep : float
        Maximum step magnitude per system.
    device : str, optional
        Warp device. Inferred from ``positions`` if not provided.

    Notes
    -----
    - ``batch_idx`` must be sorted; segment reductions assume contiguous
      atom ranges per system.
    - Scratch buffers must be zeroed before each call.

    Examples
    --------
    >>> fire2_step(positions, velocities, forces, batch_idx,
    ...            alpha, dt, nsteps_inc,
    ...            vf, v_sumsq, f_sumsq, max_norm)
    """
    # --- Input validation ---
    N = positions.shape[0]

    if batch_idx is None:
        raise ValueError("batch_idx is required for fire2_step")

    if velocities.shape[0] != N:
        raise ValueError(
            f"velocities length {velocities.shape[0]} != positions length {N}"
        )
    if forces.shape[0] != N:
        raise ValueError(f"forces length {forces.shape[0]} != positions length {N}")
    if batch_idx.shape[0] != N:
        raise ValueError(
            f"batch_idx length {batch_idx.shape[0]} != positions length {N}"
        )

    M = alpha.shape[0]
    if dt.shape[0] != M:
        raise ValueError(f"dt length {dt.shape[0]} != alpha length {M}")
    if nsteps_inc.shape[0] != M:
        raise ValueError(f"nsteps_inc length {nsteps_inc.shape[0]} != alpha length {M}")

    vec_dtype = positions.dtype
    if device is None:
        device = positions.device
    elif isinstance(device, str):
        device = wp.get_device(device)
    sm = max(device.sm_count, 1)

    # Kernel 1: reduce only (no velocity write, deferred to fused kernel)
    ept1 = _compute_ept(N, sm, True)
    dim1 = (N + ept1 - 1) // ept1
    wp.launch(
        _fire2_reduce_only_overloads[vec_dtype],
        dim=dim1,
        inputs=[velocities, forces, dt, batch_idx, vf, v_sumsq, f_sumsq, N, ept1],
        device=device,
    )

    # Kernel 2: param update + deferred halfstep + mix + maxnorm
    ept2 = _compute_ept(N, sm, True)
    dim2 = (N + ept2 - 1) // ept2
    wp.launch(
        _fire2_fused_mix_maxnorm_overloads[vec_dtype],
        dim=dim2,
        inputs=[
            velocities,
            forces,
            dt,
            batch_idx,
            vf,
            v_sumsq,
            f_sumsq,
            alpha,
            nsteps_inc,
            max_norm,
            N,
            ept2,
            delaystep,
            dtgrow,
            dtshrink,
            alphashrink,
            alpha0,
            tmax,
            tmin,
        ],
        device=device,
    )

    # Kernel 3: recompute step + clamp + position update + velocity zeroing
    wp.launch(
        _fire2_clamp_apply_recompute_overloads[vec_dtype],
        dim=N,
        inputs=[
            positions,
            velocities,
            dt,
            batch_idx,
            max_norm,
            vf,  # vf holds v.f; uphill if <= 0
            maxstep,
        ],
        device=device,
    )
