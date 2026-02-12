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
PyTorch Adapter for FIRE2 Optimizer
====================================

Thin wrapper that accepts PyTorch tensors, allocates scratch buffers via
PyTorch's CUDA caching allocator, and calls the pure-Warp FIRE2 kernels.

Entry point:

- :func:`fire2_step_coord` -- coordinate-only optimization.

Modifies inputs in-place.  Scratch buffers can be passed in for
reuse across steps, or left as ``None`` to allocate internally each call
(PyTorch's caching allocator handles efficient reuse).

Tensors are detached from the autograd graph before conversion to Warp.
Where possible, ``return_ctype=True`` is used to avoid ``wp.array``
Python-wrapper overhead.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.dynamics.optimizers.fire2 import (
    _fire2_clamp_apply_recompute_overloads,
    _fire2_fused_mix_maxnorm_overloads,
    _fire2_reduce_only_overloads,
)
from nvalchemiops.segment_ops import _compute_ept

# Torch dtype -> Warp dtype mappings
_TORCH_TO_WP_VEC = {torch.float32: wp.vec3f, torch.float64: wp.vec3d}
_TORCH_TO_WP_SCALAR = {torch.float32: wp.float32, torch.float64: wp.float64}


def fire2_step_coord(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    nsteps_inc: torch.Tensor,
    *,
    delaystep: int = 60,
    dtgrow: float = 1.05,
    dtshrink: float = 0.75,
    alphashrink: float = 0.985,
    alpha0: float = 0.09,
    tmax: float = 0.08,
    tmin: float = 0.005,
    maxstep: float = 0.1,
    vf: torch.Tensor | None = None,
    v_sumsq: torch.Tensor | None = None,
    f_sumsq: torch.Tensor | None = None,
    max_norm: torch.Tensor | None = None,
) -> None:
    """FIRE2 coordinate-only optimization step.

    Modifies *positions*, *velocities*, *alpha*, *dt*, *nsteps_inc* in-place.
    Launches 3 Warp kernels directly using ctypes for minimal overhead.

    Parameters
    ----------
    positions : Tensor, shape (N, 3), dtype float32/float64
        Atomic positions, modified in-place.
    velocities : Tensor, shape (N, 3), dtype float32/float64
        Atomic velocities, modified in-place.
    forces : Tensor, shape (N, 3), dtype float32/float64
        Forces on atoms (read-only).
    batch_idx : Tensor, shape (N,), dtype int32
        Sorted system index per atom.
    alpha : Tensor, shape (M,), dtype float32/float64
        FIRE2 mixing parameter, modified in-place.
    dt : Tensor, shape (M,), dtype float32/float64
        Per-system timestep, modified in-place.
    nsteps_inc : Tensor, shape (M,), dtype int32
        Consecutive positive-power counter, modified in-place.
    delaystep, dtgrow, dtshrink, alphashrink, alpha0, tmax, tmin, maxstep
        FIRE2 hyperparameters.
    vf : Tensor, shape (M,), optional
        Scratch for v.f reduction. Allocated and zeroed if not provided;
        zeroed in-place if provided.
    v_sumsq : Tensor, shape (M,), optional
        Scratch for v.v reduction. Same allocation/zeroing semantics.
    f_sumsq : Tensor, shape (M,), optional
        Scratch for f.f reduction. Same allocation/zeroing semantics.
    max_norm : Tensor, shape (M,), optional
        Scratch for max step norm. Same allocation/zeroing semantics.

    Notes
    -----
    This adapter bypasses the public ``fire2_step`` Warp API and launches
    kernel overloads directly with ``return_ctype=True`` to avoid the
    ``wp.array`` Python wrapper overhead on every call. The launch logic
    mirrors ``fire2_step`` exactly; see
    ``nvalchemiops.dynamics.optimizers.fire2.fire2_step`` for the canonical
    implementation.
    """
    dtype = positions.dtype
    device = positions.device
    N = positions.shape[0]
    M = alpha.shape[0]
    vec_type = _TORCH_TO_WP_VEC[dtype]
    wp_device = wp.device_from_torch(device)
    sm = max(wp_device.sm_count, 1)

    # Scratch buffers: allocate if not provided, zero reduction buffers
    if vf is None:
        vf = torch.zeros(M, dtype=dtype, device=device)
    else:
        vf.zero_()
    if v_sumsq is None:
        v_sumsq = torch.zeros(M, dtype=dtype, device=device)
    else:
        v_sumsq.zero_()
    if f_sumsq is None:
        f_sumsq = torch.zeros(M, dtype=dtype, device=device)
    else:
        f_sumsq.zero_()
    if max_norm is None:
        max_norm = torch.zeros(M, dtype=dtype, device=device)
    else:
        max_norm.zero_()

    # Detach from autograd graph + convert to ctypes (no wp.array overhead)
    positions_c = wp.from_torch(
        positions.detach().contiguous(), dtype=vec_type, return_ctype=True
    )
    velocities_c = wp.from_torch(
        velocities.detach().contiguous(), dtype=vec_type, return_ctype=True
    )
    forces_c = wp.from_torch(
        forces.detach().contiguous(), dtype=vec_type, return_ctype=True
    )
    batch_idx_c = wp.from_torch(
        batch_idx.detach().contiguous(), dtype=wp.int32, return_ctype=True
    )
    alpha_c = wp.from_torch(alpha.detach().contiguous(), return_ctype=True)
    dt_c = wp.from_torch(dt.detach().contiguous(), return_ctype=True)
    nsteps_inc_c = wp.from_torch(
        nsteps_inc.detach().contiguous(), dtype=wp.int32, return_ctype=True
    )
    vf_c = wp.from_torch(vf, return_ctype=True)
    v_sumsq_c = wp.from_torch(v_sumsq, return_ctype=True)
    f_sumsq_c = wp.from_torch(f_sumsq, return_ctype=True)
    max_norm_c = wp.from_torch(max_norm, return_ctype=True)

    # Kernel 1: reduce only (no velocity write, deferred to fused kernel)
    ept1 = _compute_ept(N, sm, True)
    dim1 = (N + ept1 - 1) // ept1
    wp.launch(
        _fire2_reduce_only_overloads[vec_type],
        dim=dim1,
        inputs=[
            velocities_c,
            forces_c,
            dt_c,
            batch_idx_c,
            vf_c,
            v_sumsq_c,
            f_sumsq_c,
            N,
            ept1,
        ],
        device=wp_device,
    )

    # Kernel 2: param update + deferred halfstep + mix + maxnorm
    ept2 = _compute_ept(N, sm, True)
    dim2 = (N + ept2 - 1) // ept2
    wp.launch(
        _fire2_fused_mix_maxnorm_overloads[vec_type],
        dim=dim2,
        inputs=[
            velocities_c,
            forces_c,
            dt_c,
            batch_idx_c,
            vf_c,
            v_sumsq_c,
            f_sumsq_c,
            alpha_c,
            nsteps_inc_c,
            max_norm_c,
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
        device=wp_device,
    )

    # Kernel 3: recompute step + clamp + position update + velocity zeroing
    wp.launch(
        _fire2_clamp_apply_recompute_overloads[vec_type],
        dim=N,
        inputs=[
            positions_c,
            velocities_c,
            dt_c,
            batch_idx_c,
            max_norm_c,
            vf_c,  # vf holds v.f; uphill if <= 0
            maxstep,
        ],
        device=wp_device,
    )
