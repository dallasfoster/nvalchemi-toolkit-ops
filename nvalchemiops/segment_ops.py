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

"""Segmented operations for sorted segment indices.

Provides segmented reductions (sum, component_sum, dot, max_norm) and
broadcasts (mul, add, matvec) over arrays grouped by sorted segment
indices *idx*.

Reduction kernels use a run-length approach: each thread processes a
contiguous chunk of elements, accumulates within runs of identical
segment ids, and emits one atomic per segment boundary.  The chunk size
is auto-tuned based on N and the GPU's SM count.
"""

from __future__ import annotations

from typing import Any

import warp as wp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TILE_SIZE = wp.constant(256)
_BLOCK_DIM = 256

_SCALAR_TYPES = [wp.float32, wp.float64]
_VEC_TYPES = [wp.vec3f, wp.vec3d]
_MAT_TYPES = [wp.mat33f, wp.mat33d]
_VEC_TO_SCALAR = {wp.vec3f: wp.float32, wp.vec3d: wp.float64}
_SCALAR_TO_VEC = {wp.float32: wp.vec3f, wp.float64: wp.vec3d}
_SUPPORTED_TYPES = [wp.float32, wp.float64, wp.vec3f, wp.vec3d]


def _compute_ept(N: int, sm_count: int, is_vec3: bool) -> int:
    """Return elements-per-thread for reduction kernels."""
    w_fill = sm_count * 512
    ept_max = 8 if is_vec3 else 16
    ept_min = 2 if is_vec3 else 4
    ept = max(1, N // w_fill)
    p = 1
    while p < ept:
        p <<= 1
    if p > 1 and (p - ept) > (ept - (p >> 1)):
        p >>= 1
    return max(ept_min, min(p, ept_max))


# ---------------------------------------------------------------------------
# Kernels -- segmented_sum
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _total_sum_tile_kernel(
    x: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    """Block-cooperative total sum (M=1 specialization).

    Each block loads _TILE_SIZE elements, reduces via shared memory,
    and emits one atomic add to out[0].
    """
    i = wp.tid()
    t = wp.tile_load(x, shape=_TILE_SIZE, offset=i * _TILE_SIZE)
    s = wp.tile_sum(t)
    wp.tile_atomic_add(out, s, 0)


@wp.kernel(enable_backward=False)
def _segmented_sum_kernel(
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Run-length segmented sum exploiting sorted *idx*.

    Each thread processes *elems_per_thread* contiguous elements,
    accumulating within runs of identical segment ids and flushing
    one atomic add per segment boundary.
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    acc = x[start]
    for i in range(start + 1, end):
        s = idx[i]
        if s == s_cur:
            acc = acc + x[i]
        else:
            wp.atomic_add(out, s_cur, acc)
            s_cur = s
            acc = x[i]
    wp.atomic_add(out, s_cur, acc)


# ---------------------------------------------------------------------------
# Kernels -- segmented_component_sum
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_component_sum_kernel(
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Sum vec3 components to scalar per segment (runs-based)."""
    t = wp.tid()
    start = t * elems_per_thread
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    _x = x[start]
    acc = _x[0] + _x[1] + _x[2]
    for i in range(start + 1, end):
        s = idx[i]
        _x = x[i]
        val = _x[0] + _x[1] + _x[2]
        if s == s_cur:
            acc = acc + val
        else:
            wp.atomic_add(out, s_cur, acc)
            s_cur = s
            acc = val
    wp.atomic_add(out, s_cur, acc)


# ---------------------------------------------------------------------------
# Kernels -- segmented_dot
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_dot_scalar_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Scalar dot-product reduction per segment (runs-based)."""
    t = wp.tid()
    start = t * elems_per_thread
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    acc = x[start] * y[start]
    for i in range(start + 1, end):
        s = idx[i]
        val = x[i] * y[i]
        if s == s_cur:
            acc = acc + val
        else:
            wp.atomic_add(out, s_cur, acc)
            s_cur = s
            acc = val
    wp.atomic_add(out, s_cur, acc)


@wp.kernel(enable_backward=False)
def _segmented_dot_vec_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Vec3 dot-product reduction per segment (runs-based)."""
    t = wp.tid()
    start = t * elems_per_thread
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    acc = wp.dot(x[start], y[start])
    for i in range(start + 1, end):
        s = idx[i]
        val = wp.dot(x[i], y[i])
        if s == s_cur:
            acc = acc + val
        else:
            wp.atomic_add(out, s_cur, acc)
            s_cur = s
            acc = val
    wp.atomic_add(out, s_cur, acc)


# ---------------------------------------------------------------------------
# Kernels -- segmented_max_norm
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_max_norm_kernel(
    x: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Max vector norm per segment (runs-based)."""
    t = wp.tid()
    start = t * elems_per_thread
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    max_val = wp.length(x[start])
    for i in range(start + 1, end):
        s = idx[i]
        val = wp.length(x[i])
        if s == s_cur:
            max_val = wp.max(max_val, val)
        else:
            wp.atomic_max(out, s_cur, max_val)
            s_cur = s
            max_val = val
    wp.atomic_max(out, s_cur, max_val)


@wp.kernel(enable_backward=False)
def _total_max_norm_kernel(
    x: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Total max-norm reduction (M=1 specialization).

    Each thread processes *elems_per_thread* elements and emits one
    atomic_max to out[0].  No segment-boundary logic is needed.
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)
    max_val = wp.length(x[start])
    for i in range(start + 1, end):
        max_val = wp.max(max_val, wp.length(x[i]))
    wp.atomic_max(out, 0, max_val)


# ---------------------------------------------------------------------------
# Kernels -- segmented_axpy (in-place broadcast FMA)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_axpy_kernel(
    y: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    a: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
):
    """y[i] += x[i] * a[idx[i]] (in-place)."""
    tid = wp.tid()
    y[tid] = y[tid] + x[tid] * a[idx[tid]]


# ---------------------------------------------------------------------------
# Kernels -- segmented_inner_products (triple reduction)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_inner_products_scalar_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out_xy: wp.array(dtype=Any),
    out_xx: wp.array(dtype=Any),
    out_yy: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Triple inner-product reduction per segment (scalar, runs-based).

    Computes x*y, x*x, y*y sums per segment in one pass.
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    xi = x[start]
    yi = y[start]
    acc_xy = xi * yi
    acc_xx = xi * xi
    acc_yy = yi * yi
    for i in range(start + 1, end):
        s = idx[i]
        xi = x[i]
        yi = y[i]
        if s == s_cur:
            acc_xy = acc_xy + xi * yi
            acc_xx = acc_xx + xi * xi
            acc_yy = acc_yy + yi * yi
        else:
            wp.atomic_add(out_xy, s_cur, acc_xy)
            wp.atomic_add(out_xx, s_cur, acc_xx)
            wp.atomic_add(out_yy, s_cur, acc_yy)
            s_cur = s
            acc_xy = xi * yi
            acc_xx = xi * xi
            acc_yy = yi * yi
    wp.atomic_add(out_xy, s_cur, acc_xy)
    wp.atomic_add(out_xx, s_cur, acc_xx)
    wp.atomic_add(out_yy, s_cur, acc_yy)


@wp.kernel(enable_backward=False)
def _segmented_inner_products_vec_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out_xy: wp.array(dtype=Any),
    out_xx: wp.array(dtype=Any),
    out_yy: wp.array(dtype=Any),
    N: wp.int32,
    elems_per_thread: wp.int32,
):
    """Triple inner-product reduction per segment (vec3, runs-based).

    Computes dot(x,y), dot(x,x), dot(y,y) sums per segment in one pass.
    """
    t = wp.tid()
    start = t * elems_per_thread
    if start >= N:
        return
    end = wp.min(start + elems_per_thread, N)

    s_cur = idx[start]
    acc_xy = wp.dot(x[start], y[start])
    acc_xx = wp.dot(x[start], x[start])
    acc_yy = wp.dot(y[start], y[start])
    for i in range(start + 1, end):
        s = idx[i]
        if s == s_cur:
            acc_xy = acc_xy + wp.dot(x[i], y[i])
            acc_xx = acc_xx + wp.dot(x[i], x[i])
            acc_yy = acc_yy + wp.dot(y[i], y[i])
        else:
            wp.atomic_add(out_xy, s_cur, acc_xy)
            wp.atomic_add(out_xx, s_cur, acc_xx)
            wp.atomic_add(out_yy, s_cur, acc_yy)
            s_cur = s
            acc_xy = wp.dot(x[i], y[i])
            acc_xx = wp.dot(x[i], x[i])
            acc_yy = wp.dot(y[i], y[i])
    wp.atomic_add(out_xy, s_cur, acc_xy)
    wp.atomic_add(out_xx, s_cur, acc_xx)
    wp.atomic_add(out_yy, s_cur, acc_yy)


# ---------------------------------------------------------------------------
# Kernels -- segmented_axpby (broadcast linear combination)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_axpby_kernel(
    out: wp.array(dtype=Any),
    a: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
):
    """out[i] = a[idx[i]] * x[i] + b[idx[i]] * y[i]."""
    tid = wp.tid()
    s = idx[tid]
    out[tid] = a[s] * x[tid] + b[s] * y[tid]


# ---------------------------------------------------------------------------
# Kernels -- segmented_mul (broadcast)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_mul_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
):
    """out[i] = x[i] * y[idx[i]]."""
    tid = wp.tid()
    out[tid] = x[tid] * y[idx[tid]]


# ---------------------------------------------------------------------------
# Kernels -- segmented_add (broadcast)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_add_kernel(
    x: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=Any),
):
    """out[i] = x[i] + y[idx[i]] (same-type)."""
    tid = wp.tid()
    out[tid] = x[tid] + y[idx[tid]]


@wp.kernel(enable_backward=False)
def _segmented_add_vec_scalar_f32_kernel(
    x: wp.array(dtype=wp.vec3f),
    y: wp.array(dtype=wp.float32),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3f),
):
    """out[i] = vec3f(x[i][k] + y[idx[i]] for k)."""
    tid = wp.tid()
    _x = x[tid]
    _y = y[idx[tid]]
    out[tid] = wp.vec3f(_x[0] + _y, _x[1] + _y, _x[2] + _y)


@wp.kernel(enable_backward=False)
def _segmented_add_vec_scalar_f64_kernel(
    x: wp.array(dtype=wp.vec3d),
    y: wp.array(dtype=wp.float64),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3d),
):
    """out[i] = vec3d(x[i][k] + y[idx[i]] for k)."""
    tid = wp.tid()
    _x = x[tid]
    _y = y[idx[tid]]
    out[tid] = wp.vec3d(_x[0] + _y, _x[1] + _y, _x[2] + _y)


@wp.kernel(enable_backward=False)
def _segmented_add_scalar_vec_f32_kernel(
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.vec3f),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3f),
):
    """out[i] = vec3f(x[i] + y[idx[i]][k] for k)."""
    tid = wp.tid()
    _x = x[tid]
    _y = y[idx[tid]]
    out[tid] = wp.vec3f(_x + _y[0], _x + _y[1], _x + _y[2])


@wp.kernel(enable_backward=False)
def _segmented_add_scalar_vec_f64_kernel(
    x: wp.array(dtype=wp.float64),
    y: wp.array(dtype=wp.vec3d),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3d),
):
    """out[i] = vec3d(x[i] + y[idx[i]][k] for k)."""
    tid = wp.tid()
    _x = x[tid]
    _y = y[idx[tid]]
    out[tid] = wp.vec3d(_x + _y[0], _x + _y[1], _x + _y[2])


# ---------------------------------------------------------------------------
# Kernels -- segmented_matvec (broadcast)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _segmented_matvec_f32_kernel(
    v: wp.array(dtype=wp.vec3f),
    m: wp.array(dtype=wp.mat33f),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3f),
):
    """out[i] = M[idx[i]]^T @ v[i] (float32)."""
    tid = wp.tid()
    _v = v[tid]
    _m = m[idx[tid]]
    c0 = wp.vec3f(_m[0, 0], _m[1, 0], _m[2, 0])
    c1 = wp.vec3f(_m[0, 1], _m[1, 1], _m[2, 1])
    c2 = wp.vec3f(_m[0, 2], _m[1, 2], _m[2, 2])
    out[tid] = wp.vec3f(wp.dot(c0, _v), wp.dot(c1, _v), wp.dot(c2, _v))


@wp.kernel(enable_backward=False)
def _segmented_matvec_f64_kernel(
    v: wp.array(dtype=wp.vec3d),
    m: wp.array(dtype=wp.mat33d),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3d),
):
    """out[i] = M[idx[i]]^T @ v[i] (float64)."""
    tid = wp.tid()
    _v = v[tid]
    _m = m[idx[tid]]
    c0 = wp.vec3d(_m[0, 0], _m[1, 0], _m[2, 0])
    c1 = wp.vec3d(_m[0, 1], _m[1, 1], _m[2, 1])
    c2 = wp.vec3d(_m[0, 2], _m[1, 2], _m[2, 2])
    out[tid] = wp.vec3d(wp.dot(c0, _v), wp.dot(c1, _v), wp.dot(c2, _v))


# ---------------------------------------------------------------------------
# Overloads (module-level, keyed by dtype)
# ---------------------------------------------------------------------------

_total_sum_tile_overloads = {}
for _t in _SUPPORTED_TYPES:
    _total_sum_tile_overloads[_t] = wp.overload(
        _total_sum_tile_kernel,
        [wp.array(dtype=_t), wp.array(dtype=_t)],
    )

_segmented_sum_overloads = {}
for _t in _SUPPORTED_TYPES:
    _segmented_sum_overloads[_t] = wp.overload(
        _segmented_sum_kernel,
        [wp.array(dtype=_t), wp.array(dtype=wp.int32), wp.array(dtype=_t),
         wp.int32, wp.int32],
    )

_segmented_component_sum_overloads = {}
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_component_sum_overloads[_v] = wp.overload(
        _segmented_component_sum_kernel,
        [wp.array(dtype=_v), wp.array(dtype=wp.int32), wp.array(dtype=_s),
         wp.int32, wp.int32],
    )

_segmented_dot_overloads = {}
for _s in _SCALAR_TYPES:
    _segmented_dot_overloads[_s] = wp.overload(
        _segmented_dot_scalar_kernel,
        [wp.array(dtype=_s), wp.array(dtype=_s), wp.array(dtype=wp.int32),
         wp.array(dtype=_s), wp.int32, wp.int32],
    )
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_dot_overloads[_v] = wp.overload(
        _segmented_dot_vec_kernel,
        [wp.array(dtype=_v), wp.array(dtype=_v), wp.array(dtype=wp.int32),
         wp.array(dtype=_s), wp.int32, wp.int32],
    )

_segmented_max_norm_overloads = {}
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_max_norm_overloads[_v] = wp.overload(
        _segmented_max_norm_kernel,
        [wp.array(dtype=_v), wp.array(dtype=wp.int32), wp.array(dtype=_s),
         wp.int32, wp.int32],
    )

_total_max_norm_overloads = {}
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _total_max_norm_overloads[_v] = wp.overload(
        _total_max_norm_kernel,
        [wp.array(dtype=_v), wp.array(dtype=_s), wp.int32, wp.int32],
    )

_segmented_axpy_overloads = {}
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_axpy_overloads[_v] = wp.overload(
        _segmented_axpy_kernel,
        [wp.array(dtype=_v), wp.array(dtype=_v), wp.array(dtype=_s),
         wp.array(dtype=wp.int32)],
    )
for _s in _SCALAR_TYPES:
    _segmented_axpy_overloads[_s] = wp.overload(
        _segmented_axpy_kernel,
        [wp.array(dtype=_s), wp.array(dtype=_s), wp.array(dtype=_s),
         wp.array(dtype=wp.int32)],
    )

_segmented_inner_products_overloads = {}
for _s in _SCALAR_TYPES:
    _segmented_inner_products_overloads[_s] = wp.overload(
        _segmented_inner_products_scalar_kernel,
        [wp.array(dtype=_s), wp.array(dtype=_s), wp.array(dtype=wp.int32),
         wp.array(dtype=_s), wp.array(dtype=_s), wp.array(dtype=_s),
         wp.int32, wp.int32],
    )
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_inner_products_overloads[_v] = wp.overload(
        _segmented_inner_products_vec_kernel,
        [wp.array(dtype=_v), wp.array(dtype=_v), wp.array(dtype=wp.int32),
         wp.array(dtype=_s), wp.array(dtype=_s), wp.array(dtype=_s),
         wp.int32, wp.int32],
    )

_segmented_axpby_overloads = {}
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_axpby_overloads[_v] = wp.overload(
        _segmented_axpby_kernel,
        [wp.array(dtype=_v), wp.array(dtype=_s), wp.array(dtype=_v),
         wp.array(dtype=_s), wp.array(dtype=_v), wp.array(dtype=wp.int32)],
    )
for _s in _SCALAR_TYPES:
    _segmented_axpby_overloads[_s] = wp.overload(
        _segmented_axpby_kernel,
        [wp.array(dtype=_s), wp.array(dtype=_s), wp.array(dtype=_s),
         wp.array(dtype=_s), wp.array(dtype=_s), wp.array(dtype=wp.int32)],
    )

_segmented_mul_overloads = {}
for _s in _SCALAR_TYPES:
    _segmented_mul_overloads[(_s, _s)] = wp.overload(
        _segmented_mul_kernel,
        [wp.array(dtype=_s), wp.array(dtype=_s), wp.array(dtype=wp.int32),
         wp.array(dtype=_s)],
    )
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_mul_overloads[(_v, _s)] = wp.overload(
        _segmented_mul_kernel,
        [wp.array(dtype=_v), wp.array(dtype=_s), wp.array(dtype=wp.int32),
         wp.array(dtype=_v)],
    )

_segmented_add_overloads = {}
for _t in _SUPPORTED_TYPES:
    _segmented_add_overloads[(_t, _t)] = wp.overload(
        _segmented_add_kernel,
        [wp.array(dtype=_t), wp.array(dtype=_t), wp.array(dtype=wp.int32),
         wp.array(dtype=_t)],
    )
_segmented_add_overloads[(wp.vec3f, wp.float32)] = _segmented_add_vec_scalar_f32_kernel
_segmented_add_overloads[(wp.vec3d, wp.float64)] = _segmented_add_vec_scalar_f64_kernel
_segmented_add_overloads[(wp.float32, wp.vec3f)] = _segmented_add_scalar_vec_f32_kernel
_segmented_add_overloads[(wp.float64, wp.vec3d)] = _segmented_add_scalar_vec_f64_kernel

_segmented_matvec_overloads = {
    wp.float32: _segmented_matvec_f32_kernel,
    wp.float64: _segmented_matvec_f64_kernel,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segmented_sum(
    x: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Segmented sum: ``out[s] = sum(x[i] for i where idx[i] == s)``.

    The caller must zero-initialize *out* before calling this function
    (e.g. via ``out.zero_()`` or ``wp.zeros``).  This avoids a redundant
    kernel launch when the caller already provides a fresh array.

    Parameters
    ----------
    x : wp.array, shape (N,)
        Input values. Supported dtypes: float32, float64, vec3f, vec3d.
    idx : wp.array(dtype=int32), shape (N,)
        Sorted segment indices in ``[0, M)``.
    out : wp.array, shape (M,)
        Output array, same dtype as *x*. Must be zero-initialized.
    """
    N = x.shape[0]
    if N == 0:
        return

    device = x.device
    M = out.shape[0]

    # -- M=1 fast path: tile-based block reduction --------------------------
    if M == 1 and N >= 8192:
        full_blocks = N // _BLOCK_DIM
        wp.launch_tiled(
            _total_sum_tile_overloads[x.dtype],
            dim=full_blocks,
            inputs=[x, out],
            block_dim=_BLOCK_DIM,
        )
        remainder = N - full_blocks * _BLOCK_DIM
        if remainder > 0:
            x_tail = x[full_blocks * _BLOCK_DIM:]
            idx_tail = idx[full_blocks * _BLOCK_DIM:]
            wp.launch(
                _segmented_sum_overloads[x.dtype],
                dim=remainder,
                inputs=[x_tail, idx_tail, out, remainder, 1],
                device=device,
            )
        return

    # -- General path: run-length segmented sum -----------------------------
    ept = _compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_sum_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, N, ept],
        device=device,
    )


def segmented_component_sum(
    x: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Sum vec3 components to scalar per segment.

    ``out[s] = sum(x[i][0] + x[i][1] + x[i][2] for i where idx[i] == s)``

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f / vec3d
    idx : wp.array(dtype=int32), shape (N,), sorted
    out : wp.array, shape (M,), dtype float32 / float64. Must be zero-initialized.
    """
    N = x.shape[0]
    if N == 0:
        return
    device = x.device
    ept = _compute_ept(N, max(device.sm_count, 1), True)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_component_sum_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, N, ept],
        device=device,
    )


def segmented_dot(
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Dot-product reduction per segment.

    Scalar: ``out[s] = sum(x[i] * y[i] for i where idx[i] == s)``
    Vec3:   ``out[s] = sum(dot(x[i], y[i]) for i where idx[i] == s)``

    Parameters
    ----------
    x, y : wp.array, shape (N,), same dtype (float32/float64/vec3f/vec3d)
    idx : wp.array(dtype=int32), shape (N,), sorted
    out : wp.array, shape (M,), scalar dtype. Must be zero-initialized.
    """
    N = x.shape[0]
    if N == 0:
        return
    device = x.device
    ept = _compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_dot_overloads[x.dtype],
        dim=dim,
        inputs=[x, y, idx, out, N, ept],
        device=device,
    )


def segmented_max_norm(
    x: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Max vector norm per segment.

    ``out[s] = max(length(x[i]) for i where idx[i] == s)``

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f / vec3d
    idx : wp.array(dtype=int32), shape (N,), sorted
    out : wp.array, shape (M,), scalar dtype. Must be zero-initialized.
    """
    N = x.shape[0]
    if N == 0:
        return
    device = x.device
    M = out.shape[0]

    # -- M=1 fast path: large EPT to minimize atomic_max contention ----------
    if M == 1 and N >= 8192:
        sm = max(device.sm_count, 1)
        ept = max(64, N // (sm * 4))
        dim = (N + ept - 1) // ept
        wp.launch(
            _total_max_norm_overloads[x.dtype],
            dim=dim,
            inputs=[x, out, N, ept],
            device=device,
        )
        return

    # -- General path: run-length segmented max norm -------------------------
    ept = _compute_ept(N, max(device.sm_count, 1), True)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_max_norm_overloads[x.dtype],
        dim=dim,
        inputs=[x, idx, out, N, ept],
        device=device,
    )


def segmented_axpy(
    y: wp.array,
    x: wp.array,
    a: wp.array,
    idx: wp.array,
) -> None:
    """In-place broadcast FMA: ``y[i] += x[i] * a[idx[i]]``.

    Parameters
    ----------
    y : wp.array, shape (N,), dtype vec3f/vec3d/float32/float64
        Accumulator, modified in-place.
    x : wp.array, shape (N,), same vec dtype as *y*
    a : wp.array, shape (M,), scalar dtype matching *y*'s precision
    idx : wp.array(dtype=int32), shape (N,)
    """
    N = y.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_axpy_overloads[y.dtype],
        dim=N,
        inputs=[y, x, a, idx],
        device=y.device,
    )


def segmented_inner_products(
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    out_xy: wp.array,
    out_xx: wp.array,
    out_yy: wp.array,
) -> None:
    """Triple inner-product reduction per segment in one pass.

    Scalar: ``out_xy[s] = sum(x[i]*y[i])``, etc.
    Vec3:   ``out_xy[s] = sum(dot(x[i],y[i]))``, etc.

    Parameters
    ----------
    x, y : wp.array, shape (N,), same dtype
    idx : wp.array(dtype=int32), shape (N,), sorted
    out_xy, out_xx, out_yy : wp.array, shape (M,), scalar dtype.
        Must be zero-initialized.
    """
    N = x.shape[0]
    if N == 0:
        return
    device = x.device
    ept = _compute_ept(N, max(device.sm_count, 1), x.dtype in _VEC_TYPES)
    dim = (N + ept - 1) // ept
    wp.launch(
        _segmented_inner_products_overloads[x.dtype],
        dim=dim,
        inputs=[x, y, idx, out_xy, out_xx, out_yy, N, ept],
        device=device,
    )


def segmented_axpby(
    out: wp.array,
    a: wp.array,
    x: wp.array,
    b: wp.array,
    y: wp.array,
    idx: wp.array,
) -> None:
    """Broadcast linear combination: ``out[i] = a[idx[i]] * x[i] + b[idx[i]] * y[i]``.

    Parameters
    ----------
    out : wp.array, shape (N,)
    a : wp.array, shape (M,), scalar dtype
    x : wp.array, shape (N,), vec3 or scalar
    b : wp.array, shape (M,), scalar dtype
    y : wp.array, shape (N,), same dtype as *x*
    idx : wp.array(dtype=int32), shape (N,)
    """
    N = out.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_axpby_overloads[x.dtype],
        dim=N,
        inputs=[out, a, x, b, y, idx],
        device=out.device,
    )


def segmented_mul(
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Broadcast multiply: ``out[i] = x[i] * y[idx[i]]``.

    Parameters
    ----------
    x : wp.array, shape (N,)
    y : wp.array, shape (M,)
    idx : wp.array(dtype=int32), shape (N,)
    out : wp.array, shape (N,)
    """
    N = x.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_mul_overloads[(x.dtype, y.dtype)],
        dim=N,
        inputs=[x, y, idx, out],
        device=x.device,
    )


def segmented_add(
    x: wp.array,
    y: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Broadcast add: ``out[i] = x[i] + y[idx[i]]``.

    Supports mixed types (vec3 + scalar, scalar + vec3).

    Parameters
    ----------
    x : wp.array, shape (N,)
    y : wp.array, shape (M,)
    idx : wp.array(dtype=int32), shape (N,)
    out : wp.array, shape (N,)
    """
    N = x.shape[0]
    if N == 0:
        return
    wp.launch(
        _segmented_add_overloads[(x.dtype, y.dtype)],
        dim=N,
        inputs=[x, y, idx, out],
        device=x.device,
    )


def segmented_matvec(
    v: wp.array,
    m: wp.array,
    idx: wp.array,
    out: wp.array,
) -> None:
    """Per-segment matrix-vector multiply: ``out[i] = M[idx[i]]^T @ v[i]``.

    Parameters
    ----------
    v : wp.array, shape (N,), dtype vec3f / vec3d
    m : wp.array, shape (M,), dtype mat33f / mat33d
    idx : wp.array(dtype=int32), shape (N,)
    out : wp.array, shape (N,), same dtype as *v*.
    """
    N = v.shape[0]
    if N == 0:
        return
    scalar_type = _VEC_TO_SCALAR[v.dtype]
    wp.launch(
        _segmented_matvec_overloads[scalar_type],
        dim=N,
        inputs=[v, m, idx, out],
        device=v.device,
    )
