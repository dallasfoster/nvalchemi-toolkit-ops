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

"""Tests for nvalchemiops.segment_ops."""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.segment_ops import (
    segmented_add,
    segmented_axpby,
    segmented_axpy,
    segmented_component_sum,
    segmented_dot,
    segmented_inner_products,
    segmented_matvec,
    segmented_max_norm,
    segmented_mul,
    segmented_sum,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numpy_segmented_sum(x_np: np.ndarray, idx_np: np.ndarray, M: int) -> np.ndarray:
    """Reference segmented sum computed on the host."""
    if x_np.ndim == 1:
        out = np.zeros(M, dtype=x_np.dtype)
    else:
        out = np.zeros((M, x_np.shape[1]), dtype=x_np.dtype)
    np.add.at(out, idx_np, x_np)
    return out


def _make_arrays(x_np, idx_np, M, wp_dtype, device):
    """Build warp arrays from numpy data."""
    x = wp.array(x_np, dtype=wp_dtype, device=device)
    idx = wp.array(idx_np.astype(np.int32), device=device)
    out = wp.zeros(M, dtype=wp_dtype, device=device)
    return x, idx, out


def _wp_vec_array(x_np, wp_dtype, device):
    """Convert (N,3) numpy to warp vec3 array."""
    return wp.array([tuple(r) for r in x_np], dtype=wp_dtype, device=device)


def _make_segments(N, M, rng):
    """Return sorted idx_np of length N with values in [0, M)."""
    return np.sort(rng.integers(0, M, size=N).astype(np.int32))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    return wp.get_device()


# ---------------------------------------------------------------------------
# Scalar tests
# ---------------------------------------------------------------------------


class TestScalarSegmentReduce:
    """Segmented sum for float16, float32 and float64."""

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-5),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_uniform_segments(self, device, wp_dtype, np_dtype, rtol):
        N, M = 1200, 12
        seg_len = N // M
        idx_np = np.repeat(np.arange(M, dtype=np.int32), seg_len)
        x_np = np.random.default_rng(42).standard_normal(N).astype(np_dtype)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp_dtype, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-5),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_skewed_segments(self, device, wp_dtype, np_dtype, rtol):
        """Mix of large and tiny segments."""
        rng = np.random.default_rng(7)
        lengths = [500, 1, 3, 200, 2, 1, 50, 1, 1, 100]
        M = len(lengths)
        idx_np = np.concatenate(
            [np.full(length, s, dtype=np.int32) for s, length in enumerate(lengths)]
        )
        N = len(idx_np)
        x_np = rng.standard_normal(N).astype(np_dtype)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp_dtype, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_single_segment(self, device):
        N, M = 1000, 1
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal(N).astype(np.float32)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp.float32, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

    def test_all_singletons(self, device):
        N = 256
        M = N
        x_np = np.arange(N, dtype=np.float32)
        idx_np = np.arange(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp.float32, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-6)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-4),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_single_segment_large(self, device, wp_dtype, np_dtype, rtol):
        """Large M=1 case -- exercises the tile block-reduction path."""
        N, M = 10_000, 1
        rng = np.random.default_rng(55)
        x_np = rng.standard_normal(N).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp_dtype, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_single_segment_not_aligned(self, device):
        """M=1 with N not a multiple of tile size -- exercises tail path."""
        N, M = 1_000, 1  # 1000 % 256 == 232
        rng = np.random.default_rng(77)
        x_np = rng.standard_normal(N).astype(np.float32)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x, idx, out = _make_arrays(x_np, idx_np, M, wp.float32, device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4)

    def test_empty_input(self, device):
        M = 5
        x = wp.zeros(0, dtype=wp.float32, device=device)
        idx = wp.zeros(0, dtype=wp.int32, device=device)
        out = wp.zeros(M, dtype=wp.float32, device=device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_array_equal(out.numpy(), np.zeros(M, dtype=np.float32))


# ---------------------------------------------------------------------------
# Vector tests
# ---------------------------------------------------------------------------


class TestVectorSegmentReduce:
    """Segmented sum for vec3h, vec3f and vec3d."""

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-4),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_uniform_segments(self, device, wp_dtype, np_dtype, rtol):
        N, M = 600, 6
        seg_len = N // M
        idx_np = np.repeat(np.arange(M, dtype=np.int32), seg_len)
        x_np = np.random.default_rng(99).standard_normal((N, 3)).astype(np_dtype)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x_tuples = [tuple(row) for row in x_np]
        x = wp.array(x_tuples, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-4),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_single_segment_vec(self, device, wp_dtype, np_dtype, rtol):
        N, M = 200, 1
        rng = np.random.default_rng(11)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x = wp.array([tuple(r) for r in x_np], dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-3),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_single_segment_large_vec(self, device, wp_dtype, np_dtype, rtol):
        """Large M=1 vec3 case -- exercises the tile block-reduction path."""
        N, M = 10_000, 1
        rng = np.random.default_rng(88)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = _numpy_segmented_sum(x_np, idx_np, M)
        x = wp.array([tuple(r) for r in x_np], dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)


# ---------------------------------------------------------------------------
# segmented_component_sum tests
# ---------------------------------------------------------------------------


class TestSegmentedComponentSum:
    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-4),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_uniform(self, device, wp_vec, np_dtype, rtol):
        N, M = 600, 6
        rng = np.random.default_rng(10)
        idx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)

        ref = np.zeros(M, dtype=np_dtype)
        np.add.at(ref, idx_np, x_np.sum(axis=1))

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = wp.float32 if np_dtype == np.float32 else wp.float64
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_component_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-4),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_skewed(self, device, wp_vec, np_dtype, rtol):
        rng = np.random.default_rng(20)
        lengths = [300, 1, 50, 2, 100]
        M = len(lengths)
        idx_np = np.concatenate(
            [np.full(length, s, dtype=np.int32) for s, length in enumerate(lengths)]
        )
        N = len(idx_np)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)

        ref = np.zeros(M, dtype=np_dtype)
        np.add.at(ref, idx_np, x_np.sum(axis=1))

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = wp.float32 if np_dtype == np.float32 else wp.float64
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_component_sum(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)


# ---------------------------------------------------------------------------
# segmented_dot tests
# ---------------------------------------------------------------------------


class TestSegmentedDot:
    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-4),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar(self, device, wp_dtype, np_dtype, rtol):
        N, M = 1200, 12
        rng = np.random.default_rng(30)
        idx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(N).astype(np_dtype)

        ref = np.zeros(M, dtype=np_dtype)
        np.add.at(ref, idx_np, x_np * y_np)

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp_dtype, device=device)

        segmented_dot(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-3),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_vec3(self, device, wp_vec, np_dtype, rtol):
        N, M = 600, 6
        rng = np.random.default_rng(31)
        idx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal((N, 3)).astype(np_dtype)

        ref = np.zeros(M, dtype=np_dtype)
        np.add.at(ref, idx_np, (x_np * y_np).sum(axis=1))

        x = _wp_vec_array(x_np, wp_vec, device)
        y = _wp_vec_array(y_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = wp.float32 if np_dtype == np.float32 else wp.float64
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_dot(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)


# ---------------------------------------------------------------------------
# segmented_max_norm tests
# ---------------------------------------------------------------------------


class TestSegmentedMaxNorm:
    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-5),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_uniform(self, device, wp_vec, np_dtype, rtol):
        N, M = 600, 6
        rng = np.random.default_rng(40)
        idx_np = np.repeat(np.arange(M, dtype=np.int32), N // M)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)

        norms = np.linalg.norm(x_np, axis=1).astype(np_dtype)
        ref = np.zeros(M, dtype=np_dtype)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                ref[s] = norms[mask].max()

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = wp.float32 if np_dtype == np.float32 else wp.float64
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_max_norm(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    def test_skewed(self, device):
        rng = np.random.default_rng(41)
        lengths = [200, 1, 3, 100, 50]
        M = len(lengths)
        idx_np = np.concatenate(
            [np.full(length, s, dtype=np.int32) for s, length in enumerate(lengths)]
        )
        N = len(idx_np)
        x_np = rng.standard_normal((N, 3)).astype(np.float32)

        norms = np.linalg.norm(x_np, axis=1).astype(np.float32)
        ref = np.zeros(M, dtype=np.float32)
        for s in range(M):
            mask = idx_np == s
            if mask.any():
                ref[s] = norms[mask].max()

        x = _wp_vec_array(x_np, wp.vec3f, device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(M, dtype=wp.float32, device=device)

        segmented_max_norm(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-5),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_single_segment_large(self, device, wp_vec, np_dtype, rtol):
        """Large M=1 case -- exercises the total max-norm fast path."""
        N, M = 10_000, 1
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.zeros(N, dtype=np.int32)

        ref = np.array([np.linalg.norm(x_np, axis=1).max()], dtype=np_dtype)

        x = _wp_vec_array(x_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out_dtype = wp.float32 if np_dtype == np.float32 else wp.float64
        out = wp.zeros(M, dtype=out_dtype, device=device)

        segmented_max_norm(x, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)


# ---------------------------------------------------------------------------
# segmented_mul tests
# ---------------------------------------------------------------------------


class TestSegmentedMul:
    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-5),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar_same_type(self, device, wp_dtype, np_dtype, rtol):
        N, M = 500, 5
        rng = np.random.default_rng(50)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(M).astype(np_dtype)

        ref = x_np * y_np[idx_np]

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_dtype, device=device)

        segmented_mul(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3f, wp.float32, np.float32, 1e-5),
            (wp.vec3d, wp.float64, np.float64, 1e-12),
        ],
    )
    def test_vec_scalar(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        """vec3 * scalar broadcast."""
        N, M = 500, 5
        rng = np.random.default_rng(51)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal(M).astype(np_dtype)

        ref = x_np * y_np[idx_np, None]

        x = _wp_vec_array(x_np, wp_vec, device)
        y = wp.array(y_np, dtype=wp_scalar, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_mul(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)


# ---------------------------------------------------------------------------
# segmented_add tests
# ---------------------------------------------------------------------------


class TestSegmentedAdd:
    @pytest.mark.parametrize(
        "wp_dtype,np_dtype,rtol",
        [
            (wp.float32, np.float32, 1e-5),
            (wp.float64, np.float64, 1e-12),
        ],
    )
    def test_scalar_same_type(self, device, wp_dtype, np_dtype, rtol):
        N, M = 500, 5
        rng = np.random.default_rng(60)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal(M).astype(np_dtype)

        ref = x_np + y_np[idx_np]

        x = wp.array(x_np, dtype=wp_dtype, device=device)
        y = wp.array(y_np, dtype=wp_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_dtype, device=device)

        segmented_add(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-5),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_vec_same_type(self, device, wp_vec, np_dtype, rtol):
        """vec3 + vec3."""
        N, M = 500, 5
        rng = np.random.default_rng(63)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal((M, 3)).astype(np_dtype)

        ref = x_np + y_np[idx_np]

        x = _wp_vec_array(x_np, wp_vec, device)
        y = _wp_vec_array(y_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_add(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-5),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_vec_scalar(self, device, wp_vec, np_dtype, rtol):
        """vec3 + scalar broadcast."""
        N, M = 500, 5
        rng = np.random.default_rng(61)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal(M).astype(np_dtype)

        ref = x_np + y_np[idx_np, None]

        x = _wp_vec_array(x_np, wp_vec, device)
        scalar_dtype = wp.float32 if np_dtype == np.float32 else wp.float64
        y = wp.array(y_np, dtype=scalar_dtype, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_add(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,np_dtype,rtol",
        [
            (wp.vec3f, np.float32, 1e-5),
            (wp.vec3d, np.float64, 1e-12),
        ],
    )
    def test_scalar_vec(self, device, wp_vec, np_dtype, rtol):
        """scalar + vec3 broadcast."""
        N, M = 500, 5
        rng = np.random.default_rng(62)
        idx_np = _make_segments(N, M, rng)
        x_np = rng.standard_normal(N).astype(np_dtype)
        y_np = rng.standard_normal((M, 3)).astype(np_dtype)

        ref = x_np[:, None] + y_np[idx_np]

        scalar_dtype = wp.float32 if np_dtype == np.float32 else wp.float64
        x = wp.array(x_np, dtype=scalar_dtype, device=device)
        y = _wp_vec_array(y_np, wp_vec, device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_add(x, y, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)


# ---------------------------------------------------------------------------
# segmented_matvec tests
# ---------------------------------------------------------------------------


class TestSegmentedMatvec:
    @pytest.mark.parametrize(
        "wp_vec,wp_mat,np_dtype,rtol",
        [
            (wp.vec3f, wp.mat33f, np.float32, 1e-4),
            (wp.vec3d, wp.mat33d, np.float64, 1e-12),
        ],
    )
    def test_basic(self, device, wp_vec, wp_mat, np_dtype, rtol):
        N, M = 500, 5
        rng = np.random.default_rng(70)
        idx_np = _make_segments(N, M, rng)
        v_np = rng.standard_normal((N, 3)).astype(np_dtype)
        m_np = rng.standard_normal((M, 3, 3)).astype(np_dtype)

        # Reference: M^T @ v
        ref = np.zeros((N, 3), dtype=np_dtype)
        for i in range(N):
            ref[i] = m_np[idx_np[i]].T @ v_np[i]

        v = _wp_vec_array(v_np, wp_vec, device)
        m_tuples = [tuple(tuple(row) for row in mat) for mat in m_np]
        m = wp.array(m_tuples, dtype=wp_mat, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_matvec(v, m, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), ref, rtol=rtol)

    @pytest.mark.parametrize(
        "wp_vec,wp_mat,np_dtype,rtol",
        [
            (wp.vec3f, wp.mat33f, np.float32, 1e-4),
            (wp.vec3d, wp.mat33d, np.float64, 1e-12),
        ],
    )
    def test_identity_matrices(self, device, wp_vec, wp_mat, np_dtype, rtol):
        """With identity matrices, output should equal input."""
        N, M = 200, 4
        rng = np.random.default_rng(71)
        idx_np = _make_segments(N, M, rng)
        v_np = rng.standard_normal((N, 3)).astype(np_dtype)
        m_np = np.stack([np.eye(3, dtype=np_dtype)] * M)

        v = _wp_vec_array(v_np, wp_vec, device)
        m_tuples = [tuple(tuple(row) for row in mat) for mat in m_np]
        m = wp.array(m_tuples, dtype=wp_mat, device=device)
        idx = wp.array(idx_np, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_matvec(v, m, idx, out)
        wp.synchronize()
        np.testing.assert_allclose(out.numpy(), v_np, rtol=rtol)


# ---------------------------------------------------------------------------
# segmented_axpy tests
# ---------------------------------------------------------------------------


class TestSegmentedAxpy:
    """Tests for segmented_axpy: y[i] += x[i] * a[idx[i]]."""

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3f, wp.float32, np.float32, 1e-5),
            (wp.vec3d, wp.float64, np.float64, 1e-12),
        ],
    )
    def test_basic(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        rng = np.random.default_rng(0)
        N, M = 100, 4
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal((N, 3)).astype(np_dtype)
        a_np = rng.standard_normal(M).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_vec, device=device)
        y = wp.array(y_np.copy(), dtype=wp_vec, device=device)
        a = wp.array(a_np, dtype=wp_scalar, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)

        segmented_axpy(y, x, a, idx)
        wp.synchronize()

        expected = y_np + x_np * a_np[idx_np, None]
        result = y.numpy()
        np.testing.assert_allclose(result, expected, rtol=rtol)


# ---------------------------------------------------------------------------
# segmented_inner_products tests
# ---------------------------------------------------------------------------


class TestSegmentedInnerProducts:
    """Tests for segmented_inner_products: triple reduction."""

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3f, wp.float32, np.float32, 1e-4),
            (wp.vec3d, wp.float64, np.float64, 1e-10),
        ],
    )
    def test_vec(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        rng = np.random.default_rng(1)
        N, M = 200, 3
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal((N, 3)).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_vec, device=device)
        y = wp.array(y_np, dtype=wp_vec, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)
        out_xy = wp.zeros(M, dtype=wp_scalar, device=device)
        out_xx = wp.zeros(M, dtype=wp_scalar, device=device)
        out_yy = wp.zeros(M, dtype=wp_scalar, device=device)

        segmented_inner_products(x, y, idx, out_xy, out_xx, out_yy)
        wp.synchronize()

        ref_xy = np.zeros(M, dtype=np_dtype)
        ref_xx = np.zeros(M, dtype=np_dtype)
        ref_yy = np.zeros(M, dtype=np_dtype)
        for i in range(N):
            s = idx_np[i]
            ref_xy[s] += np.dot(x_np[i], y_np[i])
            ref_xx[s] += np.dot(x_np[i], x_np[i])
            ref_yy[s] += np.dot(y_np[i], y_np[i])

        np.testing.assert_allclose(out_xy.numpy(), ref_xy, rtol=rtol)
        np.testing.assert_allclose(out_xx.numpy(), ref_xx, rtol=rtol)
        np.testing.assert_allclose(out_yy.numpy(), ref_yy, rtol=rtol)

    def test_scalar(self, device):
        rng = np.random.default_rng(2)
        N, M = 150, 5
        x_np = rng.standard_normal(N).astype(np.float32)
        y_np = rng.standard_normal(N).astype(np.float32)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp.float32, device=device)
        y = wp.array(y_np, dtype=wp.float32, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)
        out_xy = wp.zeros(M, dtype=wp.float32, device=device)
        out_xx = wp.zeros(M, dtype=wp.float32, device=device)
        out_yy = wp.zeros(M, dtype=wp.float32, device=device)

        segmented_inner_products(x, y, idx, out_xy, out_xx, out_yy)
        wp.synchronize()

        ref_xy = np.zeros(M, dtype=np.float32)
        ref_xx = np.zeros(M, dtype=np.float32)
        ref_yy = np.zeros(M, dtype=np.float32)
        for i in range(N):
            s = idx_np[i]
            ref_xy[s] += x_np[i] * y_np[i]
            ref_xx[s] += x_np[i] * x_np[i]
            ref_yy[s] += y_np[i] * y_np[i]

        np.testing.assert_allclose(out_xy.numpy(), ref_xy, rtol=1e-4)
        np.testing.assert_allclose(out_xx.numpy(), ref_xx, rtol=1e-4)
        np.testing.assert_allclose(out_yy.numpy(), ref_yy, rtol=1e-4)


# ---------------------------------------------------------------------------
# segmented_axpby tests
# ---------------------------------------------------------------------------


class TestSegmentedAxpby:
    """Tests for segmented_axpby: out = a[s]*x + b[s]*y."""

    @pytest.mark.parametrize(
        "wp_vec,wp_scalar,np_dtype,rtol",
        [
            (wp.vec3f, wp.float32, np.float32, 1e-5),
            (wp.vec3d, wp.float64, np.float64, 1e-12),
        ],
    )
    def test_basic(self, device, wp_vec, wp_scalar, np_dtype, rtol):
        rng = np.random.default_rng(3)
        N, M = 120, 4
        x_np = rng.standard_normal((N, 3)).astype(np_dtype)
        y_np = rng.standard_normal((N, 3)).astype(np_dtype)
        a_np = rng.standard_normal(M).astype(np_dtype)
        b_np = rng.standard_normal(M).astype(np_dtype)
        idx_np = np.sort(rng.integers(0, M, N)).astype(np.int32)

        x = wp.array(x_np, dtype=wp_vec, device=device)
        y = wp.array(y_np, dtype=wp_vec, device=device)
        a = wp.array(a_np, dtype=wp_scalar, device=device)
        b = wp.array(b_np, dtype=wp_scalar, device=device)
        idx = wp.array(idx_np, dtype=wp.int32, device=device)
        out = wp.zeros(N, dtype=wp_vec, device=device)

        segmented_axpby(out, a, x, b, y, idx)
        wp.synchronize()

        expected = a_np[idx_np, None] * x_np + b_np[idx_np, None] * y_np
        np.testing.assert_allclose(
            out.numpy(),
            expected,
            rtol=rtol,
        )
