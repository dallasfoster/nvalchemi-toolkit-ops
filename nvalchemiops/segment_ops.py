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
    """Block-cooperative total sum for single-segment case (M=1 specialization).

    Each block loads _TILE_SIZE elements, reduces via shared memory using
    tile operations, and emits one atomic add to out[0]. This provides
    optimal performance for the common case of reducing all elements to
    a single output value.

    Launch Grid
    -----------
    dim = [num_blocks], where num_blocks = N // TILE_SIZE
    block_dim = _BLOCK_DIM

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32/float64/vec3f/vec3d
        Input values to sum.
    out : wp.array, shape (1,), dtype matches x
        OUTPUT: Accumulated sum. Must be zero-initialized by caller.

    Notes
    -----
    - Uses warp tile operations for efficient block-level reduction
    - Only processes full blocks; remainder handled separately by caller
    - Requires N >= 8192 for efficiency (checked by caller)
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
    """Run-length segmented sum exploiting sorted segment indices.

    Computes ``out[s] = sum(x[i] for i where idx[i] == s)`` using a
    run-length encoding approach. Each thread processes a contiguous
    chunk of elements, accumulates within runs of identical segment IDs,
    and emits one atomic add per segment boundary.

    This algorithm exploits spatial locality when idx is sorted, minimizing
    atomic contention by accumulating locally before writing to global memory.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32/float64/vec3f/vec3d
        Input values to sum per segment.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M). Must be sorted in non-decreasing order.
    out : wp.array, shape (M,), dtype matches x
        OUTPUT: Per-segment sums. Must be zero-initialized by caller.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread (auto-tuned based on array size).

    Notes
    -----
    - Requires idx to be sorted for correctness
    - Each thread may span multiple segments; atomic adds occur at boundaries
    - Elements-per-thread is auto-tuned based on total count and SM count
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
    """Sum vec3 components to scalar per segment using run-length encoding.

    Computes ``out[s] = sum(x[i][0] + x[i][1] + x[i][2] for i where idx[i] == s)``.
    This is equivalent to summing all vector components within each segment.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        Input vec3 values.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype float32/float64
        OUTPUT: Scalar sums of all components per segment. Must be zero-initialized.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - Output dtype is scalar (float32 for vec3f, float64 for vec3d)
    - Useful for computing total kinetic energy or similar scalar reductions
    """
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
    """Scalar element-wise product reduction per segment.

    Computes ``out[s] = sum(x[i] * y[i] for i where idx[i] == s)``.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32/float64
        First input array.
    y : wp.array, shape (N,), dtype matches x
        Second input array.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype matches x
        OUTPUT: Dot product per segment. Must be zero-initialized.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - For scalar arrays, computes element-wise product sum
    - See _segmented_dot_vec_kernel for vector dot products
    """
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
    """Vec3 dot-product reduction per segment.

    Computes ``out[s] = sum(dot(x[i], y[i]) for i where idx[i] == s)``
    using vector dot products.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        First input vector array.
    y : wp.array, shape (N,), dtype matches x
        Second input vector array.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype float32/float64
        OUTPUT: Scalar dot product sum per segment. Must be zero-initialized.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - Output is scalar (float32 for vec3f input, float64 for vec3d input)
    - Useful for computing v·f power in FIRE optimizer
    """
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
    """Maximum vector norm per segment using run-length encoding.

    Computes ``out[s] = max(length(x[i]) for i where idx[i] == s)``.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        Input vector array.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out : wp.array, shape (M,), dtype float32/float64
        OUTPUT: Maximum vector norm per segment. Must be zero-initialized.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - Useful for convergence checks in geometry optimization (e.g., max force magnitude)
    - Output is scalar (float32 for vec3f input, float64 for vec3d input)
    """
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
    """Total maximum norm reduction for single-segment case (M=1 specialization).

    Computes ``out[0] = max(length(x[i]) for all i)``. Each thread processes
    a chunk of elements and emits one atomic_max to the global output.
    No segment-boundary logic is needed since there is only one segment.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        Input vector array.
    out : wp.array, shape (1,), dtype float32/float64
        OUTPUT: Maximum norm across all elements. Must be zero-initialized.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread (large EPT reduces atomic contention).

    Notes
    -----
    - Uses large elements-per-thread to minimize atomic_max contention
    - Requires N >= 8192 for efficiency (checked by caller)
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
    """In-place broadcast fused multiply-add (AXPY): y[i] += x[i] * a[idx[i]].

    Performs a segment-indexed AXPY operation where each element is scaled
    by a per-segment coefficient before adding to the accumulator.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    y : wp.array, shape (N,), dtype vec3f/vec3d/float32/float64
        Accumulator array, modified in-place.
    x : wp.array, shape (N,), dtype matches y
        Input array to scale and add.
    a : wp.array, shape (M,), dtype float32/float64
        Per-segment scalar coefficients.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M) (need not be sorted for this kernel).

    Notes
    -----
    - Modifies y in-place
    - Supports both scalar and vector types
    - For vectors, scalar coefficient a[s] is broadcast to all components
    """
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
    """Triple inner-product reduction per segment for scalar arrays.

    Computes three dot products in a single pass:
    - ``out_xy[s] = sum(x[i] * y[i] for i where idx[i] == s)``
    - ``out_xx[s] = sum(x[i] * x[i] for i where idx[i] == s)``
    - ``out_yy[s] = sum(y[i] * y[i] for i where idx[i] == s)``

    This fused operation is more efficient than three separate reductions.

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32/float64
        First input array.
    y : wp.array, shape (N,), dtype matches x
        Second input array.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out_xy : wp.array, shape (M,), dtype matches x
        OUTPUT: x·y per segment. Must be zero-initialized.
    out_xx : wp.array, shape (M,), dtype matches x
        OUTPUT: x·x per segment. Must be zero-initialized.
    out_yy : wp.array, shape (M,), dtype matches x
        OUTPUT: y·y per segment. Must be zero-initialized.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - Useful for FIRE2 optimizer which needs v·f, v·v, and f·f simultaneously
    - Reduces memory traffic compared to three separate kernel launches
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
    """Triple inner-product reduction per segment for vector arrays.

    Computes three dot products in a single pass:
    - ``out_xy[s] = sum(dot(x[i], y[i]) for i where idx[i] == s)``
    - ``out_xx[s] = sum(dot(x[i], x[i]) for i where idx[i] == s)``
    - ``out_yy[s] = sum(dot(y[i], y[i]) for i where idx[i] == s)``

    Launch Grid
    -----------
    dim = ceil(N / elems_per_thread)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d
        First input vector array.
    y : wp.array, shape (N,), dtype matches x
        Second input vector array.
    idx : wp.array, shape (N,), dtype int32
        Sorted segment indices in [0, M).
    out_xy : wp.array, shape (M,), dtype float32/float64
        OUTPUT: x·y per segment. Must be zero-initialized.
    out_xx : wp.array, shape (M,), dtype float32/float64
        OUTPUT: x·x per segment. Must be zero-initialized.
    out_yy : wp.array, shape (M,), dtype float32/float64
        OUTPUT: y·y per segment. Must be zero-initialized.
    N : int32
        Total number of elements.
    elems_per_thread : int32
        Number of elements processed per thread.

    Notes
    -----
    - Output is scalar (float32 for vec3f input, float64 for vec3d input)
    - Useful for FIRE2 optimizer which needs velocity·force, velocity·velocity,
      and force·force simultaneously for adaptive parameter updates
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
    """Broadcast linear combination: out[i] = a[idx[i]] * x[i] + b[idx[i]] * y[i].

    Computes a segment-indexed linear combination where each element is
    scaled by per-segment coefficients before summing.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    out : wp.array, shape (N,), dtype vec3f/vec3d/float32/float64
        OUTPUT: Result of linear combination.
    a : wp.array, shape (M,), dtype float32/float64
        Per-segment scalar coefficients for x.
    x : wp.array, shape (N,), dtype matches out
        First input array.
    b : wp.array, shape (M,), dtype float32/float64
        Per-segment scalar coefficients for y.
    y : wp.array, shape (N,), dtype matches out
        Second input array.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M) (need not be sorted for this kernel).

    Notes
    -----
    - Supports both scalar and vector types
    - For vectors, scalar coefficients are broadcast to all components
    - Common in iterative solvers and optimization algorithms
    """
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
    """Broadcast multiply: out[i] = x[i] * y[idx[i]].

    Multiplies each element by a per-segment value.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d/float32/float64
        Per-element input array.
    y : wp.array, shape (M,), dtype float32/float64 or matches x
        Per-segment broadcast values.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M) (need not be sorted for this kernel).
    out : wp.array, shape (N,), dtype matches x
        OUTPUT: Element-wise product.

    Notes
    -----
    - Supports scalar-scalar and vector-scalar multiplication
    - For vector-scalar, scalar is broadcast to all components
    """
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
    """Broadcast add: out[i] = x[i] + y[idx[i]] (same-type variant).

    Adds a per-segment value to each element.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f/vec3d/float32/float64
        Per-element input array.
    y : wp.array, shape (M,), dtype matches x
        Per-segment broadcast values.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M) (need not be sorted for this kernel).
    out : wp.array, shape (N,), dtype matches x
        OUTPUT: Element-wise sum.

    Notes
    -----
    - This variant requires x and y to have the same dtype
    - See _segmented_add_vec_scalar_* for mixed-type variants
    """
    tid = wp.tid()
    out[tid] = x[tid] + y[idx[tid]]


@wp.kernel(enable_backward=False)
def _segmented_add_vec_scalar_f32_kernel(
    x: wp.array(dtype=wp.vec3f),
    y: wp.array(dtype=wp.float32),
    idx: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec3f),
):
    """Broadcast add: out[i] = x[i] + y[idx[i]] (vec3f + float32 variant).

    Adds a per-segment scalar to each component of a vector.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f
        Per-element input vectors.
    y : wp.array, shape (M,), dtype float32
        Per-segment scalar values to broadcast.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M).
    out : wp.array, shape (N,), dtype vec3f
        OUTPUT: out[i] = [x[i][0] + y[s], x[i][1] + y[s], x[i][2] + y[s]].

    Notes
    -----
    - Scalar is broadcast to all three vector components
    - Float32 precision variant
    """
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
    """Broadcast add: out[i] = x[i] + y[idx[i]] (vec3d + float64 variant).

    Adds a per-segment scalar to each component of a vector.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3d
        Per-element input vectors.
    y : wp.array, shape (M,), dtype float64
        Per-segment scalar values to broadcast.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M).
    out : wp.array, shape (N,), dtype vec3d
        OUTPUT: out[i] = [x[i][0] + y[s], x[i][1] + y[s], x[i][2] + y[s]].

    Notes
    -----
    - Scalar is broadcast to all three vector components
    - Float64 precision variant
    """
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
    """Broadcast add: out[i] = x[i] + y[idx[i]] (float32 + vec3f variant).

    Adds per-element scalars to per-segment vectors component-wise.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float32
        Per-element scalar values.
    y : wp.array, shape (M,), dtype vec3f
        Per-segment vectors to broadcast.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M).
    out : wp.array, shape (N,), dtype vec3f
        OUTPUT: out[i] = [x[i] + y[s][0], x[i] + y[s][1], x[i] + y[s][2]].

    Notes
    -----
    - Scalar is added to each vector component
    - Float32 precision variant
    """
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
    """Broadcast add: out[i] = x[i] + y[idx[i]] (float64 + vec3d variant).

    Adds per-element scalars to per-segment vectors component-wise.

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    x : wp.array, shape (N,), dtype float64
        Per-element scalar values.
    y : wp.array, shape (M,), dtype vec3d
        Per-segment vectors to broadcast.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M).
    out : wp.array, shape (N,), dtype vec3d
        OUTPUT: out[i] = [x[i] + y[s][0], x[i] + y[s][1], x[i] + y[s][2]].

    Notes
    -----
    - Scalar is added to each vector component
    - Float64 precision variant
    """
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
    """Per-segment matrix-vector multiply: out[i] = M[idx[i]]^T @ v[i] (float32).

    Applies a per-segment 3x3 matrix transformation to each vector.
    Uses transpose convention (column-major multiply).

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    v : wp.array, shape (N,), dtype vec3f
        Per-element input vectors.
    m : wp.array, shape (M,), dtype mat33f
        Per-segment 3x3 transformation matrices.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M).
    out : wp.array, shape (N,), dtype vec3f
        OUTPUT: Transformed vectors.

    Notes
    -----
    - Uses transpose convention: M^T @ v (column vectors)
    - Useful for coordinate transformations in NPT barostats
    - Float32 precision variant
    """
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
    """Per-segment matrix-vector multiply: out[i] = M[idx[i]]^T @ v[i] (float64).

    Applies a per-segment 3x3 matrix transformation to each vector.
    Uses transpose convention (column-major multiply).

    Launch Grid
    -----------
    dim = N (total elements)

    Parameters
    ----------
    v : wp.array, shape (N,), dtype vec3d
        Per-element input vectors.
    m : wp.array, shape (M,), dtype mat33d
        Per-segment 3x3 transformation matrices.
    idx : wp.array, shape (N,), dtype int32
        Segment indices in [0, M).
    out : wp.array, shape (N,), dtype vec3d
        OUTPUT: Transformed vectors.

    Notes
    -----
    - Uses transpose convention: M^T @ v (column vectors)
    - Useful for coordinate transformations in NPT barostats
    - Float64 precision variant
    """
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
        [
            wp.array(dtype=_t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_t),
            wp.int32,
            wp.int32,
        ],
    )

_segmented_component_sum_overloads = {}
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_component_sum_overloads[_v] = wp.overload(
        _segmented_component_sum_kernel,
        [
            wp.array(dtype=_v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_s),
            wp.int32,
            wp.int32,
        ],
    )

_segmented_dot_overloads = {}
for _s in _SCALAR_TYPES:
    _segmented_dot_overloads[_s] = wp.overload(
        _segmented_dot_scalar_kernel,
        [
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_s),
            wp.int32,
            wp.int32,
        ],
    )
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_dot_overloads[_v] = wp.overload(
        _segmented_dot_vec_kernel,
        [
            wp.array(dtype=_v),
            wp.array(dtype=_v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_s),
            wp.int32,
            wp.int32,
        ],
    )

_segmented_max_norm_overloads = {}
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_max_norm_overloads[_v] = wp.overload(
        _segmented_max_norm_kernel,
        [
            wp.array(dtype=_v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_s),
            wp.int32,
            wp.int32,
        ],
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
        [
            wp.array(dtype=_v),
            wp.array(dtype=_v),
            wp.array(dtype=_s),
            wp.array(dtype=wp.int32),
        ],
    )
for _s in _SCALAR_TYPES:
    _segmented_axpy_overloads[_s] = wp.overload(
        _segmented_axpy_kernel,
        [
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=wp.int32),
        ],
    )

_segmented_inner_products_overloads = {}
for _s in _SCALAR_TYPES:
    _segmented_inner_products_overloads[_s] = wp.overload(
        _segmented_inner_products_scalar_kernel,
        [
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.int32,
            wp.int32,
        ],
    )
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_inner_products_overloads[_v] = wp.overload(
        _segmented_inner_products_vec_kernel,
        [
            wp.array(dtype=_v),
            wp.array(dtype=_v),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.int32,
            wp.int32,
        ],
    )

_segmented_axpby_overloads = {}
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_axpby_overloads[_v] = wp.overload(
        _segmented_axpby_kernel,
        [
            wp.array(dtype=_v),
            wp.array(dtype=_s),
            wp.array(dtype=_v),
            wp.array(dtype=_s),
            wp.array(dtype=_v),
            wp.array(dtype=wp.int32),
        ],
    )
for _s in _SCALAR_TYPES:
    _segmented_axpby_overloads[_s] = wp.overload(
        _segmented_axpby_kernel,
        [
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=wp.int32),
        ],
    )

_segmented_mul_overloads = {}
for _s in _SCALAR_TYPES:
    _segmented_mul_overloads[(_s, _s)] = wp.overload(
        _segmented_mul_kernel,
        [
            wp.array(dtype=_s),
            wp.array(dtype=_s),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_s),
        ],
    )
for _v, _s in zip(_VEC_TYPES, _SCALAR_TYPES):
    _segmented_mul_overloads[(_v, _s)] = wp.overload(
        _segmented_mul_kernel,
        [
            wp.array(dtype=_v),
            wp.array(dtype=_s),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_v),
        ],
    )

_segmented_add_overloads = {}
for _t in _SUPPORTED_TYPES:
    _segmented_add_overloads[(_t, _t)] = wp.overload(
        _segmented_add_kernel,
        [
            wp.array(dtype=_t),
            wp.array(dtype=_t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=_t),
        ],
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
    """Compute per-segment sum using run-length encoded reduction.

    Performs a segmented reduction that sums all elements belonging to each
    segment: ``out[s] = sum(x[i] for i where idx[i] == s)``.

    **IMPORTANT:** The caller must zero-initialize ``out`` before calling
    (e.g., ``out.zero_()`` or ``wp.zeros``). This avoids a redundant kernel
    launch when the caller already provides a fresh array.

    **CRITICAL:** The ``idx`` array **MUST** be sorted in non-decreasing
    order for correctness. Unsorted indices will produce incorrect results.

    Parameters
    ----------
    x : wp.array, shape (N,)
        Input values to sum per segment.
        Supported dtypes: ``float32``, ``float64``, ``vec3f``, ``vec3d``.
    idx : wp.array(dtype=int32), shape (N,)
        **Sorted** segment indices in ``[0, M)``. Each ``idx[i]`` indicates
        which segment element ``x[i]`` belongs to. **MUST BE SORTED**.
    out : wp.array, shape (M,)
        Output array containing per-segment sums, same dtype as ``x``.
        **Must be zero-initialized by caller before calling this function.**

    Examples
    --------
    Sum per-atom forces to per-system totals:

    >>> import warp as wp
    >>> # 8 atoms in 3 systems: [0,0,0,1,1,2,2,2]
    >>> batch_idx = wp.array([0,0,0,1,1,2,2,2], dtype=wp.int32)
    >>> forces = wp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=wp.float32)
    >>> system_forces = wp.zeros(3, dtype=wp.float32)
    >>> segmented_sum(forces, batch_idx, system_forces)
    >>> print(system_forces.numpy())  # [6.0, 9.0, 21.0]

    See Also
    --------
    segmented_dot : Dot product reduction per segment
    segmented_component_sum : Sum vector components to scalar per segment
    segmented_max_norm : Maximum vector norm per segment
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
            x_tail = x[full_blocks * _BLOCK_DIM :]
            idx_tail = idx[full_blocks * _BLOCK_DIM :]
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
    """Sum all vector components to scalar per segment using RLE reduction.

    Computes the total sum of all x, y, z components within each segment:

    ``out[s] = sum(x[i][0] + x[i][1] + x[i][2] for i where idx[i] == s)``

    This is useful for computing total magnitudes across all dimensions,
    such as summing all force components or all velocity components per system.

    Uses run-length encoding to minimize atomic operations (O(M) instead of O(N)).

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f / vec3d
        Input 3D vectors. Each vector's components will be summed.
    idx : wp.array(dtype=int32), shape (N,)
        **Sorted** segment indices in ``[0, M)``. **MUST BE SORTED**.
    out : wp.array, shape (M,), dtype float32 / float64
        Output scalar sums per segment. Precision matches ``x``.
        **Must be zero-initialized by caller.**

    Examples
    --------
    Sum all velocity components per system:

    >>> velocities = wp.array([[1,2,3], [4,5,6]], dtype=wp.vec3f)  # 2 atoms
    >>> batch_idx = wp.array([0, 0], dtype=wp.int32)  # same system
    >>> total = wp.zeros(1, dtype=wp.float32)
    >>> segmented_component_sum(velocities, batch_idx, total)
    >>> print(total.numpy())  # [21.0] = (1+2+3) + (4+5+6)

    Notes
    -----
    - Output dtype is scalar (float32/float64) matching input precision
    - Requires sorted ``idx`` array for correctness
    - Caller must zero-initialize ``out`` before calling

    See Also
    --------
    segmented_sum : Element-wise sum (preserves vector dtype)
    segmented_dot : Dot product per segment
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
    """Compute per-segment dot product reduction using RLE.

    Performs segmented dot product that handles both scalar and vector types:

    - **Scalar types**: ``out[s] = sum(x[i] * y[i] for i where idx[i] == s)``
    - **Vector types**: ``out[s] = sum(dot(x[i], y[i]) for i where idx[i] == s)``

    Common use cases include computing kinetic energy (v·v per system),
    power (F·v per system), or any pairwise inner product reduction.

    Uses run-length encoding to minimize atomic operations.

    Parameters
    ----------
    x, y : wp.array, shape (N,)
        Input arrays for dot product. Must have same dtype.
        Supported: ``float32``, ``float64``, ``vec3f``, ``vec3d``.
    idx : wp.array(dtype=int32), shape (N,)
        **Sorted** segment indices in ``[0, M)``. **MUST BE SORTED**.
    out : wp.array, shape (M,), dtype float32 / float64
        Output scalar dot products per segment. Precision matches ``x``/``y``.
        **Must be zero-initialized by caller.**

    Examples
    --------
    Compute kinetic energy per system (v·v):

    >>> velocities = wp.array([[1,0,0], [0,2,0], [3,0,0]], dtype=wp.vec3f)
    >>> batch_idx = wp.array([0, 0, 1], dtype=wp.int32)  # 2 atoms in sys 0, 1 in sys 1
    >>> v_dot_v = wp.zeros(2, dtype=wp.float32)
    >>> segmented_dot(velocities, velocities, batch_idx, v_dot_v)
    >>> print(v_dot_v.numpy())  # [5.0, 9.0] = (1^2 + 2^2), (3^2)

    Compute power per system (F·v):

    >>> forces = wp.array([[1,1,1], [2,2,2]], dtype=wp.vec3f)
    >>> velocities = wp.array([[1,0,0], [0,1,0]], dtype=wp.vec3f)
    >>> batch_idx = wp.array([0, 0], dtype=wp.int32)
    >>> power = wp.zeros(1, dtype=wp.float32)
    >>> segmented_dot(forces, velocities, batch_idx, power)
    >>> print(power.numpy())  # [3.0] = (1*1 + 0*1 + 0*1) + (0*2 + 1*2 + 0*2)

    Notes
    -----
    - For vectors, computes standard 3D dot product per element
    - Output is always scalar type matching input precision
    - Requires sorted ``idx`` for correctness
    - Caller must zero-initialize ``out``

    See Also
    --------
    segmented_inner_products : Compute x·y, x·x, and y·y in one pass
    segmented_sum : Element-wise sum per segment
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
    """Compute maximum vector norm (length) per segment using RLE reduction.

    Finds the maximum Euclidean norm across all vectors in each segment:

    ``out[s] = max(length(x[i]) for i where idx[i] == s)``

    where ``length(x[i]) = sqrt(x[i][0]^2 + x[i][1]^2 + x[i][2]^2)``.

    This is commonly used for convergence checking in geometry optimization
    (e.g., FIRE optimizer) by finding the maximum force magnitude per system.

    Uses run-length encoding with atomic max operations. For single-segment
    cases (M=1) with large arrays (N≥8192), uses optimized large elements-per-thread
    to minimize atomic contention.

    Parameters
    ----------
    x : wp.array, shape (N,), dtype vec3f / vec3d
        Input 3D vectors. Norm is computed for each vector.
    idx : wp.array(dtype=int32), shape (N,)
        **Sorted** segment indices in ``[0, M)``. **MUST BE SORTED**.
    out : wp.array, shape (M,), dtype float32 / float64
        Output maximum norms per segment. Precision matches ``x``.
        **Must be zero-initialized by caller.**

    Examples
    --------
    Find maximum force magnitude per system (for convergence check):

    >>> forces = wp.array([[3,4,0], [1,0,0], [0,5,12]], dtype=wp.vec3f)
    >>> batch_idx = wp.array([0, 0, 1], dtype=wp.int32)
    >>> max_forces = wp.zeros(2, dtype=wp.float32)
    >>> segmented_max_norm(forces, batch_idx, max_forces)
    >>> print(max_forces.numpy())  # [5.0, 13.0] = max(5.0, 1.0), max(13.0)
    >>> # Check convergence: system 0 converged if max_forces[0] < threshold

    Notes
    -----
    - Computes Euclidean norm (L2 norm) for each vector
    - Uses ``atomic_max`` for segment reduction
    - Special optimization for M=1 (single segment) with N≥8192
    - Requires sorted ``idx`` for correctness
    - Caller must zero-initialize ``out``

    See Also
    --------
    segmented_dot : For computing squared norms (v·v) per segment
    segmented_component_sum : For L1-like component sums
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
    """In-place segmented broadcast FMA (fused multiply-add): ``y[i] += x[i] * a[idx[i]]``.

    Broadcasts per-segment scalar values to per-atom vectors and accumulates:
    for each atom ``i``, multiplies ``x[i]`` by the scalar ``a[idx[i]]``
    and adds the result to ``y[i]``.

    This is the BLAS AXPY operation generalized to segmented/batched contexts,
    useful for applying per-system scalars (like timesteps, learning rates,
    or temperature factors) to per-atom quantities.

    **IMPORTANT**: This operation modifies ``y`` in-place.

    Parameters
    ----------
    y : wp.array, shape (N,)
        Accumulator array, **modified in-place**.
        Supported dtypes: ``vec3f``, ``vec3d``, ``float32``, ``float64``.
    x : wp.array, shape (N,)
        Input array to scale and add. Must have same dtype as ``y``.
    a : wp.array, shape (M,)
        Per-segment scalar multipliers. Dtype is scalar precision matching ``y``
        (``float32`` for ``vec3f``/``float32``, ``float64`` for ``vec3d``/``float64``).
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)`` for broadcast. Need not be sorted.

    Examples
    --------
    Apply per-system timestep scaling to velocities:

    >>> velocities = wp.array([[1,0,0], [0,1,0]], dtype=wp.vec3f)  # 2 atoms
    >>> accelerations = wp.array([[2,0,0], [0,3,0]], dtype=wp.vec3f)
    >>> dt_per_system = wp.array([0.5, 0.1], dtype=wp.float32)  # different dt
    >>> batch_idx = wp.array([0, 1], dtype=wp.int32)
    >>> segmented_axpy(velocities, accelerations, dt_per_system, batch_idx)
    >>> # velocities[0] = [1,0,0] + [2,0,0]*0.5 = [2,0,0]
    >>> # velocities[1] = [0,1,0] + [0,3,0]*0.1 = [0,1.3,0]

    Notes
    -----
    - Operation is in-place on ``y``
    - ``idx`` need not be sorted (no RLE optimization needed for broadcast)
    - Common use: velocity updates, gradient descent steps, scaled accumulation
    - Named "axpy" after BLAS convention: alpha*x + y

    See Also
    --------
    segmented_axpby : Generalized linear combination (a*x + b*y)
    segmented_mul : Broadcast multiply (out = x * y[idx])
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
    """Compute three inner products per segment in one fused pass using RLE.

    Efficiently computes all three combinations of dot products in a single
    kernel launch:

    - **Scalar types**: ``out_xy[s] = sum(x[i]*y[i])``, ``out_xx[s] = sum(x[i]*x[i])``, ``out_yy[s] = sum(y[i]*y[i])``
    - **Vector types**: ``out_xy[s] = sum(dot(x[i],y[i]))``, ``out_xx[s] = sum(dot(x[i],x[i]))``, ``out_yy[s] = sum(dot(y[i],y[i]))``

    This is significantly more efficient than calling ``segmented_dot`` three
    times separately, as it reuses loads and performs all reductions in one pass.

    Common use case: FIRE optimizer diagnostics where you need v·f, v·v, and f·f
    simultaneously for parameter updates.

    Uses run-length encoding to minimize atomic operations.

    Parameters
    ----------
    x, y : wp.array, shape (N,)
        Input arrays for inner products. Must have same dtype.
        Supported: ``float32``, ``float64``, ``vec3f``, ``vec3d``.
    idx : wp.array(dtype=int32), shape (N,)
        **Sorted** segment indices in ``[0, M)``. **MUST BE SORTED**.
    out_xy, out_xx, out_yy : wp.array, shape (M,), dtype float32 / float64
        Output inner products per segment. Precision matches ``x``/``y``.
        **All three arrays must be zero-initialized by caller.**

    Examples
    --------
    Compute FIRE optimizer diagnostics (v·f, v·v, f·f):

    >>> velocities = wp.array([[1,0,0], [0,2,0]], dtype=wp.vec3f)
    >>> forces = wp.array([[1,1,0], [2,0,0]], dtype=wp.vec3f)
    >>> batch_idx = wp.array([0, 0], dtype=wp.int32)
    >>> vf = wp.zeros(1, dtype=wp.float32)
    >>> vv = wp.zeros(1, dtype=wp.float32)
    >>> ff = wp.zeros(1, dtype=wp.float32)
    >>> segmented_inner_products(velocities, forces, batch_idx, vf, vv, ff)
    >>> # vf[0] = 1.0 (1*1 + 0*2)
    >>> # vv[0] = 5.0 (1^2 + 2^2)
    >>> # ff[0] = 6.0 (1^2 + 1^2 + 2^2)

    Notes
    -----
    - **Performance**: ~3x faster than three separate ``segmented_dot`` calls
    - All output arrays must be zero-initialized
    - Requires sorted ``idx`` for RLE optimization
    - Commonly used in optimization algorithms (FIRE, conjugate gradient)

    See Also
    --------
    segmented_dot : Single dot product per segment
    segmented_sum : Element-wise sum per segment
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

    Generalization of BLAS AXPBY to segmented/batched contexts. Broadcasts
    per-segment scalars ``a`` and ``b`` to per-atom arrays, computing a
    two-term linear combination.

    This is useful for operations like velocity updates with multiple force
    components, gradient blending in optimization, or any operation requiring
    weighted combination of per-atom quantities with per-system weights.

    **Memory**: Writes output to separate array (does not modify inputs).

    Parameters
    ----------
    out : wp.array, shape (N,)
        Output array for results. Must have same dtype as ``x``/``y``.
        Supported: ``vec3f``, ``vec3d``, ``float32``, ``float64``.
    a : wp.array, shape (M,), dtype float32 / float64
        Per-segment scalar multipliers for ``x``. Precision matches ``out``.
    x : wp.array, shape (N,)
        First input array. Must have same dtype as ``y`` and ``out``.
    b : wp.array, shape (M,), dtype float32 / float64
        Per-segment scalar multipliers for ``y``. Precision matches ``out``.
    y : wp.array, shape (N,)
        Second input array. Must have same dtype as ``x`` and ``out``.
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)`` for broadcast. Need not be sorted.

    Examples
    --------
    Blend two force contributions with per-system weights:

    >>> forces_lj = wp.array([[1,0,0], [2,0,0]], dtype=wp.vec3f)
    >>> forces_coul = wp.array([[0,1,0], [0,2,0]], dtype=wp.vec3f)
    >>> weight_lj = wp.array([0.8, 1.0], dtype=wp.float32)
    >>> weight_coul = wp.array([0.2, 1.0], dtype=wp.float32)
    >>> batch_idx = wp.array([0, 1], dtype=wp.int32)
    >>> forces_total = wp.zeros(2, dtype=wp.vec3f)
    >>> segmented_axpby(forces_total, weight_lj, forces_lj,
    ...                 weight_coul, forces_coul, batch_idx)
    >>> # forces_total[0] = 0.8*[1,0,0] + 0.2*[0,1,0] = [0.8, 0.2, 0]
    >>> # forces_total[1] = 1.0*[2,0,0] + 1.0*[0,2,0] = [2, 2, 0]

    Notes
    -----
    - ``idx`` need not be sorted (broadcast operation, no RLE optimization)
    - All arrays (``x``, ``y``, ``out``) must have same dtype
    - Scalar precision of ``a``/``b`` must match vector precision of ``x``/``y``/``out``
    - Common special case: ``a=dt``, ``b=1`` for velocity Verlet-style updates

    See Also
    --------
    segmented_axpy : Simpler one-term version (out = y + a*x)
    segmented_mul : Broadcast multiply only
    segmented_add : Broadcast add only
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

    Broadcasts per-segment scalar values to per-atom arrays and multiplies.
    Supports mixed-type operations (vector × scalar, scalar × scalar).

    Common use cases include:
    - Applying per-system scaling factors (timesteps, learning rates, temperature)
    - Mass-weighting force calculations
    - Applying per-system unit conversions

    **Supports mixed types**: ``vec3 × scalar``, ``scalar × scalar``.

    Parameters
    ----------
    x : wp.array, shape (N,)
        Per-atom input values to scale.
        Supported: ``vec3f``, ``vec3d``, ``float32``, ``float64``.
    y : wp.array, shape (M,)
        Per-segment scalar multipliers.
        Supported: ``float32`` (for ``vec3f``/``float32``), ``float64`` (for ``vec3d``/``float64``).
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)`` for broadcast. Need not be sorted.
    out : wp.array, shape (N,)
        Output array for scaled results. Dtype matches ``x`` for vec3 types,
        or result precision for scalar types.

    Examples
    --------
    Apply per-system timestep scaling to forces:

    >>> forces = wp.array([[1,2,3], [4,5,6]], dtype=wp.vec3f)
    >>> dt_per_system = wp.array([0.5, 0.1], dtype=wp.float32)
    >>> batch_idx = wp.array([0, 1], dtype=wp.int32)
    >>> scaled_forces = wp.zeros(2, dtype=wp.vec3f)
    >>> segmented_mul(forces, dt_per_system, batch_idx, scaled_forces)
    >>> # scaled_forces[0] = [1,2,3] * 0.5 = [0.5, 1.0, 1.5]
    >>> # scaled_forces[1] = [4,5,6] * 0.1 = [0.4, 0.5, 0.6]

    Mass-weighting scalar quantities:

    >>> energies = wp.array([100.0, 200.0, 150.0], dtype=wp.float32)
    >>> masses = wp.array([1.0, 2.0], dtype=wp.float32)
    >>> batch_idx = wp.array([0, 1, 1], dtype=wp.int32)
    >>> weighted = wp.zeros(3, dtype=wp.float32)
    >>> segmented_mul(energies, masses, batch_idx, weighted)
    >>> # weighted = [100.0, 400.0, 300.0]

    Notes
    -----
    - ``idx`` need not be sorted (broadcast operation)
    - For vector × scalar: broadcasts scalar to all components
    - For scalar × scalar: standard element-wise multiply with broadcast
    - Does not modify inputs

    See Also
    --------
    segmented_add : Broadcast addition
    segmented_axpy : Fused multiply-add (y += a*x with broadcast)
    segmented_axpby : Linear combination with broadcast
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

    Broadcasts per-segment values to per-atom arrays and adds.
    Supports mixed-type operations (vector + scalar, scalar + scalar).

    Common use cases include:
    - Adding per-system offsets or biases to per-atom quantities
    - Shifting coordinates by per-system origin vectors
    - Adding per-system baseline energies to per-atom energies

    **Supports mixed types**: ``vec3 + scalar``, ``scalar + vec3``, ``scalar + scalar``.

    Parameters
    ----------
    x : wp.array, shape (N,)
        Per-atom input values.
        Supported: ``vec3f``, ``vec3d``, ``float32``, ``float64``.
    y : wp.array, shape (M,)
        Per-segment values to broadcast and add.
        - For ``x`` with vec3 type: ``y`` can be scalar (broadcasts to all components) or vec3
        - For ``x`` with scalar type: ``y`` must be scalar
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)`` for broadcast. Need not be sorted.
    out : wp.array, shape (N,)
        Output array for results. Dtype typically matches ``x``
        (or promoted type for mixed operations).

    Examples
    --------
    Add per-system offset to per-atom positions:

    >>> positions = wp.array([[0,0,0], [1,0,0]], dtype=wp.vec3f)
    >>> offsets = wp.array([[10,20,30], [5,5,5]], dtype=wp.vec3f)
    >>> batch_idx = wp.array([0, 1], dtype=wp.int32)
    >>> shifted_positions = wp.zeros(2, dtype=wp.vec3f)
    >>> segmented_add(positions, offsets, batch_idx, shifted_positions)
    >>> # shifted_positions[0] = [0,0,0] + [10,20,30] = [10,20,30]
    >>> # shifted_positions[1] = [1,0,0] + [5,5,5] = [6,5,5]

    Add per-system baseline energy to per-atom energies:

    >>> atom_energies = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
    >>> baseline_per_system = wp.array([100.0, 200.0], dtype=wp.float32)
    >>> batch_idx = wp.array([0, 0, 1], dtype=wp.int32)
    >>> total_energies = wp.zeros(3, dtype=wp.float32)
    >>> segmented_add(atom_energies, baseline_per_system, batch_idx, total_energies)
    >>> # total_energies = [101.0, 102.0, 203.0]

    Mixed type: add scalar offset to all vector components:

    >>> vectors = wp.array([[1,2,3]], dtype=wp.vec3f)
    >>> scalar_offset = wp.array([10.0], dtype=wp.float32)
    >>> batch_idx = wp.array([0], dtype=wp.int32)
    >>> result = wp.zeros(1, dtype=wp.vec3f)
    >>> segmented_add(vectors, scalar_offset, batch_idx, result)
    >>> # result[0] = [1,2,3] + [10,10,10] = [11,12,13]

    Notes
    -----
    - ``idx`` need not be sorted (broadcast operation)
    - For vec3 + scalar: broadcasts scalar to all three components
    - For scalar operations: standard broadcast addition
    - Does not modify inputs

    See Also
    --------
    segmented_mul : Broadcast multiplication
    segmented_axpy : Fused multiply-add with broadcast
    segmented_axpby : Linear combination with broadcast
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

    Broadcasts per-segment 3×3 matrices to per-atom 3D vectors and performs
    matrix-vector multiplication. Each atom is transformed by its segment's matrix.

    **Note**: Uses **transpose** of the matrix: ``M^T @ v`` (row-major multiplication).

    Common use cases:
    - Coordinate transformations (rotation, reflection, scaling)
    - Applying per-system cell transformations to per-atom positions
    - Basis transformations in reciprocal space calculations
    - Applying per-system strain tensors to per-atom quantities

    Parameters
    ----------
    v : wp.array, shape (N,), dtype vec3f / vec3d
        Per-atom 3D input vectors to transform.
    m : wp.array, shape (M,), dtype mat33f / mat33d
        Per-segment 3×3 transformation matrices. Precision must match ``v``
        (``mat33f`` for ``vec3f``, ``mat33d`` for ``vec3d``).
    idx : wp.array(dtype=int32), shape (N,)
        Segment indices in ``[0, M)`` for broadcast. Need not be sorted.
    out : wp.array, shape (N,), dtype vec3f / vec3d
        Output transformed vectors. Must have same dtype as ``v``.

    Examples
    --------
    Apply per-system rotation to per-atom positions:

    >>> import warp as wp
    >>> import math
    >>> # 90-degree rotation around z-axis for system 0
    >>> theta = math.pi / 2
    >>> rot_z = wp.mat33f([
    ...     [math.cos(theta), -math.sin(theta), 0],
    ...     [math.sin(theta),  math.cos(theta), 0],
    ...     [0, 0, 1]
    ... ])
    >>> # Identity for system 1
    >>> identity = wp.mat33f([[1,0,0], [0,1,0], [0,0,1]])
    >>> matrices = wp.array([rot_z, identity], dtype=wp.mat33f)
    >>> positions = wp.array([[1,0,0], [0,1,0]], dtype=wp.vec3f)
    >>> batch_idx = wp.array([0, 1], dtype=wp.int32)
    >>> rotated = wp.zeros(2, dtype=wp.vec3f)
    >>> segmented_matvec(positions, matrices, batch_idx, rotated)
    >>> # rotated[0] ≈ [0,1,0] (rotated 90° around z)
    >>> # rotated[1] = [0,1,0] (unchanged by identity)

    Apply periodic boundary cell transformation:

    >>> # Fractional coordinates → Cartesian
    >>> cell_matrix = wp.mat33f([[10,0,0], [0,10,0], [0,0,10]])  # cubic cell
    >>> cells = wp.array([cell_matrix], dtype=wp.mat33f)
    >>> fractional = wp.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]], dtype=wp.vec3f)
    >>> batch_idx = wp.array([0, 0], dtype=wp.int32)
    >>> cartesian = wp.zeros(2, dtype=wp.vec3f)
    >>> segmented_matvec(fractional, cells, batch_idx, cartesian)
    >>> # cartesian = [[5,5,5], [2.5,2.5,2.5]]

    Notes
    -----
    - **Transpose convention**: Computes ``M^T @ v``, not ``M @ v``
    - ``idx`` need not be sorted (broadcast operation)
    - Matrix and vector precision must match (float32/float64)
    - Does not modify inputs
    - Typical use in MD: cell transformations, coordinate basis changes

    See Also
    --------
    segmented_mul : Broadcast scalar multiplication
    segmented_add : Broadcast vector addition
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
