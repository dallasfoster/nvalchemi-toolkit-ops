# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Tests for neighbor utility warp launchers."""

import pytest
import torch
import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    _make_tile_exclusive_scan_int32,
    _tile_exclusive_scan_int32_kernel_cache,
    compute_naive_num_shifts,
    exclusive_scan_int32,
    zero_array,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype

from .test_utils import create_simple_cubic_system

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")
dtypes = [torch.float32, torch.float64]


@pytest.mark.parametrize("device", devices)
class TestNeighborUtilsWpLaunchers:
    """Test the public launcher API for neighbor utilities."""

    @pytest.mark.parametrize("dtype", dtypes)
    def test_compute_naive_num_shifts(self, device, dtype):
        """Test compute_naive_num_shifts launcher."""
        _, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 1.5

        # Prepare output arrays
        shift_range_per_dimension = torch.zeros(3, dtype=torch.int32, device=device)
        num_shifts = torch.zeros(1, dtype=torch.int32, device=device)

        # Convert to warp arrays
        wp_dtype = get_wp_dtype(dtype)
        wp_mat_dtype = get_wp_mat_dtype(dtype)

        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool)
        wp_shift_range = wp.from_torch(shift_range_per_dimension, dtype=wp.vec3i)
        wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.int32)

        # Call launcher
        compute_naive_num_shifts(
            wp_cell,
            cutoff,
            wp_pbc,
            wp_num_shifts,
            wp_shift_range,
            wp_dtype,
            device,
        )

        # Verify results
        assert torch.all(shift_range_per_dimension >= 0), (
            "Shift ranges should be non-negative"
        )
        assert num_shifts.item() > 0, "Should have at least one shift"

    def test_zero_array(self, device):
        """Test zero_array launcher."""
        # Test data with non-zero values
        test_array = torch.full((100,), 42, dtype=torch.int32, device=device)

        # Convert to warp array
        wp_array = wp.from_torch(test_array, dtype=wp.int32)

        # Call launcher
        zero_array(wp_array, device)

        # Should be all zeros
        assert torch.all(test_array == 0), "Array should be zeroed"

    def test_zero_array_empty(self, device):
        """Test zero_array with empty array."""
        test_array = torch.empty(0, dtype=torch.int32, device=device)

        # Convert to warp array
        wp_array = wp.from_torch(test_array, dtype=wp.int32)

        # Call launcher - should handle gracefully
        zero_array(wp_array, device)

        assert test_array.shape == (0,)

    @pytest.mark.parametrize("n", [1, 2, 16, 64, 256, 1024, 4096])
    def test_exclusive_scan_int32_correctness(self, device, n):
        """``exclusive_scan_int32`` matches ``torch.cumsum`` (exclusive variant)
        for a range of array sizes spanning < block-dim and > block-dim cases.
        Runs on both CPU and CUDA devices to verify the tile-scan kernel is
        portable across back-ends.
        """
        torch.manual_seed(0)
        a = torch.randint(0, 100, (n,), dtype=torch.int32, device=device)
        out = torch.zeros(n, dtype=torch.int32, device=device)
        wa = wp.from_torch(a, dtype=wp.int32)
        wo = wp.from_torch(out, dtype=wp.int32)

        exclusive_scan_int32(wa, wo, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        expected = torch.cumsum(a, dim=0) - a  # exclusive prefix sum
        assert torch.equal(out, expected), (
            f"exclusive_scan_int32(n={n}, device={device}) disagrees with "
            f"torch.cumsum reference"
        )

    def test_exclusive_scan_int32_all_zeros(self, device):
        """All-zero input must produce an all-zero exclusive scan."""
        n = 128
        a = torch.zeros(n, dtype=torch.int32, device=device)
        out = torch.full((n,), -1, dtype=torch.int32, device=device)  # poison
        exclusive_scan_int32(
            wp.from_torch(a, dtype=wp.int32),
            wp.from_torch(out, dtype=wp.int32),
            device,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        assert torch.all(out == 0)

    def test_exclusive_scan_int32_all_ones(self, device):
        """All-one input produces 0, 1, 2, ..., n-1 — equivalent to arange."""
        n = 256
        a = torch.ones(n, dtype=torch.int32, device=device)
        out = torch.zeros(n, dtype=torch.int32, device=device)
        exclusive_scan_int32(
            wp.from_torch(a, dtype=wp.int32),
            wp.from_torch(out, dtype=wp.int32),
            device,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        expected = torch.arange(n, dtype=torch.int32, device=device)
        assert torch.equal(out, expected)

    def test_exclusive_scan_int32_kernel_cache_reuse(self, device):
        """Two calls at the same length must reuse the cached specialised
        kernel rather than recompile. We snapshot the cached kernel object
        identity across the two calls.
        """
        n = 333  # arbitrary, unlikely to collide with other tests
        # Drop any pre-existing cached kernel for this size (e.g. from a
        # prior test in the same session) so we observe a single insert.
        _tile_exclusive_scan_int32_kernel_cache.pop(n, None)

        a = torch.arange(1, n + 1, dtype=torch.int32, device=device)
        out = torch.zeros(n, dtype=torch.int32, device=device)
        exclusive_scan_int32(
            wp.from_torch(a, dtype=wp.int32),
            wp.from_torch(out, dtype=wp.int32),
            device,
        )
        first = _tile_exclusive_scan_int32_kernel_cache.get(n)
        assert first is not None, "first launch should have populated the cache"

        # Second call at the same length — kernel object must be the same
        # instance (not a fresh compile).
        out.zero_()
        exclusive_scan_int32(
            wp.from_torch(a, dtype=wp.int32),
            wp.from_torch(out, dtype=wp.int32),
            device,
        )
        second = _tile_exclusive_scan_int32_kernel_cache.get(n)
        assert second is first, "cache hit should return the same kernel"

    def test_exclusive_scan_int32_large_values(self, device):
        """Stress the int32 range by scanning values near INT32_MAX / n;
        the scan must not overflow within bounds and the result must still
        match the torch reference."""
        n = 512
        # Pick values whose total stays within int32 (< 2**31 - 1 ≈ 2.1e9).
        a = torch.full((n,), 1_000_000, dtype=torch.int32, device=device)
        out = torch.zeros(n, dtype=torch.int32, device=device)
        exclusive_scan_int32(
            wp.from_torch(a, dtype=wp.int32),
            wp.from_torch(out, dtype=wp.int32),
            device,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        expected = torch.cumsum(a, dim=0) - a
        assert torch.equal(out, expected)

    def test_make_tile_exclusive_scan_int32_factory_caches(self, device):
        """The kernel factory itself must return the same object for repeated
        calls at the same ``tile_dim`` — the public launcher relies on it for
        reuse, but downstream callers may also use the factory directly
        (e.g. when capturing into a CUDA graph and wanting an upfront warmup)."""
        n = 777
        _tile_exclusive_scan_int32_kernel_cache.pop(n, None)
        k1 = _make_tile_exclusive_scan_int32(n)
        k2 = _make_tile_exclusive_scan_int32(n)
        assert k1 is k2
        # Different size yields a different specialised kernel.
        k3 = _make_tile_exclusive_scan_int32(n + 1)
        assert k3 is not k1
