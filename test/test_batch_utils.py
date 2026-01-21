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
Comprehensive test suite for nvalchemiops.batch_utils module.

Tests cover:
- Index/pointer conversion utilities
- Per-system scalar reductions (sum, max, min, mean)
- Per-system vector reductions (sum, max_norm, rms_norm)
- Broadcasting from per-system to per-atom arrays
- Both float32 and float64 precision
- Both batch_idx and atom_ptr modes
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.batch_utils import (
    atom_ptr_to_batch_idx,
    atoms_per_system,
    batch_idx_to_atom_ptr,
    # Broadcasting
    broadcast_to_atoms,
    create_atom_ptr,
    # Index/pointer conversion
    create_batch_idx,
    max_norm_per_system,
    max_per_system,
    mean_per_system,
    min_per_system,
    rms_norm_per_system,
    # Scalar reductions
    sum_per_system,
    # Vector reductions
    sum_vectors_per_system,
)


@pytest.fixture(scope="module")
def device():
    """Get the default Warp device."""
    return wp.get_device()


@pytest.fixture
def simple_batch_data(device):
    """Create simple batch data for testing."""
    # 3 systems with 10, 15, 12 atoms
    atom_counts = [10, 15, 12]
    num_systems = len(atom_counts)
    total_atoms = sum(atom_counts)

    atom_counts_wp = wp.array(np.array(atom_counts, dtype=np.int32), device=device)

    # Expected values
    expected_atom_ptr = np.array([0, 10, 25, 37], dtype=np.int32)
    expected_batch_idx = np.concatenate(
        [
            np.full(10, 0, dtype=np.int32),
            np.full(15, 1, dtype=np.int32),
            np.full(12, 2, dtype=np.int32),
        ]
    )

    return {
        "atom_counts": atom_counts,
        "atom_counts_wp": atom_counts_wp,
        "num_systems": num_systems,
        "total_atoms": total_atoms,
        "expected_atom_ptr": expected_atom_ptr,
        "expected_batch_idx": expected_batch_idx,
    }


# =============================================================================
# Test Index/Pointer Conversion
# =============================================================================


class TestIndexPointerConversion:
    """Tests for batch_idx <-> atom_ptr conversion utilities."""

    def test_create_atom_ptr(self, device, simple_batch_data):
        """Test creating atom_ptr from atom_counts."""
        atom_ptr = create_atom_ptr(simple_batch_data["atom_counts_wp"], device=device)
        wp.synchronize()

        np.testing.assert_array_equal(
            atom_ptr.numpy(),
            simple_batch_data["expected_atom_ptr"],
        )

    def test_create_batch_idx_with_total_atoms(self, device, simple_batch_data):
        """Test creating batch_idx with total_atoms provided (no sync)."""
        batch_idx = create_batch_idx(
            simple_batch_data["atom_counts_wp"],
            total_atoms=simple_batch_data["total_atoms"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_array_equal(
            batch_idx.numpy(),
            simple_batch_data["expected_batch_idx"],
        )

    def test_create_batch_idx_with_preallocated(self, device, simple_batch_data):
        """Test creating batch_idx with pre-allocated array (no sync)."""
        batch_idx = wp.zeros(
            simple_batch_data["total_atoms"], dtype=wp.int32, device=device
        )
        result = create_batch_idx(
            simple_batch_data["atom_counts_wp"],
            batch_idx=batch_idx,
            device=device,
        )
        wp.synchronize()

        assert result is batch_idx  # Should be same array
        np.testing.assert_array_equal(
            batch_idx.numpy(),
            simple_batch_data["expected_batch_idx"],
        )

    def test_create_batch_idx_auto_sync(self, device, simple_batch_data):
        """Test creating batch_idx without total_atoms (triggers sync)."""
        batch_idx = create_batch_idx(
            simple_batch_data["atom_counts_wp"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_array_equal(
            batch_idx.numpy(),
            simple_batch_data["expected_batch_idx"],
        )

    def test_atom_ptr_to_batch_idx(self, device, simple_batch_data):
        """Test converting atom_ptr to batch_idx."""
        atom_ptr = wp.array(
            simple_batch_data["expected_atom_ptr"], dtype=wp.int32, device=device
        )
        batch_idx = atom_ptr_to_batch_idx(
            atom_ptr,
            total_atoms=simple_batch_data["total_atoms"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_array_equal(
            batch_idx.numpy(),
            simple_batch_data["expected_batch_idx"],
        )

    def test_batch_idx_to_atom_ptr(self, device, simple_batch_data):
        """Test converting batch_idx to atom_ptr."""
        batch_idx = wp.array(
            simple_batch_data["expected_batch_idx"], dtype=wp.int32, device=device
        )
        atom_ptr = batch_idx_to_atom_ptr(
            batch_idx,
            num_systems=simple_batch_data["num_systems"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_array_equal(
            atom_ptr.numpy(),
            simple_batch_data["expected_atom_ptr"],
        )

    def test_roundtrip_atom_ptr_batch_idx(self, device, simple_batch_data):
        """Test roundtrip: atom_counts -> atom_ptr -> batch_idx -> atom_ptr."""
        # atom_counts -> atom_ptr
        atom_ptr1 = create_atom_ptr(simple_batch_data["atom_counts_wp"], device=device)

        # atom_ptr -> batch_idx
        batch_idx = atom_ptr_to_batch_idx(
            atom_ptr1,
            total_atoms=simple_batch_data["total_atoms"],
            device=device,
        )

        # batch_idx -> atom_ptr
        atom_ptr2 = batch_idx_to_atom_ptr(
            batch_idx,
            num_systems=simple_batch_data["num_systems"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_array_equal(atom_ptr1.numpy(), atom_ptr2.numpy())

    def test_atoms_per_system_from_atom_ptr(self, device, simple_batch_data):
        """Test getting atom counts from atom_ptr."""
        atom_ptr = wp.array(
            simple_batch_data["expected_atom_ptr"], dtype=wp.int32, device=device
        )
        counts = atoms_per_system(atom_ptr=atom_ptr, device=device)
        wp.synchronize()

        np.testing.assert_array_equal(
            counts.numpy(),
            np.array(simple_batch_data["atom_counts"], dtype=np.int32),
        )

    def test_atoms_per_system_from_batch_idx(self, device, simple_batch_data):
        """Test getting atom counts from batch_idx."""
        batch_idx = wp.array(
            simple_batch_data["expected_batch_idx"], dtype=wp.int32, device=device
        )
        counts = atoms_per_system(
            batch_idx=batch_idx,
            num_systems=simple_batch_data["num_systems"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_array_equal(
            counts.numpy(),
            np.array(simple_batch_data["atom_counts"], dtype=np.int32),
        )


# =============================================================================
# Test Per-System Scalar Reductions
# =============================================================================


class TestScalarReductions:
    """Tests for per-system scalar reduction utilities."""

    @pytest.fixture
    def scalar_data(self, device, simple_batch_data):
        """Create scalar test data."""
        np.random.seed(42)
        total_atoms = simple_batch_data["total_atoms"]
        atom_counts = simple_batch_data["atom_counts"]

        # Generate random per-atom values
        values_np = np.random.randn(total_atoms).astype(np.float64)

        # Compute expected results
        expected_sum = np.zeros(len(atom_counts), dtype=np.float64)
        expected_max = np.full(len(atom_counts), -np.inf, dtype=np.float64)
        expected_min = np.full(len(atom_counts), np.inf, dtype=np.float64)
        expected_mean = np.zeros(len(atom_counts), dtype=np.float64)

        offset = 0
        for s, count in enumerate(atom_counts):
            sys_values = values_np[offset : offset + count]
            expected_sum[s] = sys_values.sum()
            expected_max[s] = sys_values.max()
            expected_min[s] = sys_values.min()
            expected_mean[s] = sys_values.mean()
            offset += count

        values_wp = wp.array(values_np, dtype=wp.float64, device=device)
        atom_ptr = wp.array(
            simple_batch_data["expected_atom_ptr"], dtype=wp.int32, device=device
        )
        batch_idx = wp.array(
            simple_batch_data["expected_batch_idx"], dtype=wp.int32, device=device
        )

        return {
            "values": values_wp,
            "atom_ptr": atom_ptr,
            "batch_idx": batch_idx,
            "num_systems": len(atom_counts),
            "expected_sum": expected_sum,
            "expected_max": expected_max,
            "expected_min": expected_min,
            "expected_mean": expected_mean,
        }

    def test_sum_per_system_atom_ptr(self, device, scalar_data):
        """Test sum reduction using atom_ptr mode."""
        result = sum_per_system(
            scalar_data["values"],
            atom_ptr=scalar_data["atom_ptr"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_allclose(
            result.numpy(),
            scalar_data["expected_sum"],
            rtol=1e-10,
        )

    def test_sum_per_system_batch_idx(self, device, scalar_data):
        """Test sum reduction using batch_idx mode."""
        result = sum_per_system(
            scalar_data["values"],
            batch_idx=scalar_data["batch_idx"],
            num_systems=scalar_data["num_systems"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_allclose(
            result.numpy(),
            scalar_data["expected_sum"],
            rtol=1e-10,
        )

    def test_max_per_system(self, device, scalar_data):
        """Test max reduction using atom_ptr mode."""
        result = max_per_system(
            scalar_data["values"],
            atom_ptr=scalar_data["atom_ptr"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_allclose(
            result.numpy(),
            scalar_data["expected_max"],
            rtol=1e-10,
        )

    def test_min_per_system(self, device, scalar_data):
        """Test min reduction using atom_ptr mode."""
        result = min_per_system(
            scalar_data["values"],
            atom_ptr=scalar_data["atom_ptr"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_allclose(
            result.numpy(),
            scalar_data["expected_min"],
            rtol=1e-10,
        )

    def test_mean_per_system(self, device, scalar_data):
        """Test mean reduction using atom_ptr mode."""
        result = mean_per_system(
            scalar_data["values"],
            atom_ptr=scalar_data["atom_ptr"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_allclose(
            result.numpy(),
            scalar_data["expected_mean"],
            rtol=1e-10,
        )

    def test_sum_per_system_preallocated(self, device, scalar_data):
        """Test sum reduction with pre-allocated result array."""
        result = wp.zeros(scalar_data["num_systems"], dtype=wp.float64, device=device)
        returned = sum_per_system(
            scalar_data["values"],
            atom_ptr=scalar_data["atom_ptr"],
            result=result,
            device=device,
        )
        wp.synchronize()

        assert returned is result
        np.testing.assert_allclose(
            result.numpy(),
            scalar_data["expected_sum"],
            rtol=1e-10,
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_sum_per_system_dtypes(self, device, simple_batch_data, dtype):
        """Test sum reduction with different dtypes."""
        np.random.seed(42)
        total_atoms = simple_batch_data["total_atoms"]
        values_np = np.random.randn(total_atoms).astype(dtype)

        wp_dtype = wp.float32 if dtype == np.float32 else wp.float64
        values_wp = wp.array(values_np, dtype=wp_dtype, device=device)
        atom_ptr = wp.array(
            simple_batch_data["expected_atom_ptr"], dtype=wp.int32, device=device
        )

        result = sum_per_system(values_wp, atom_ptr=atom_ptr, device=device)
        wp.synchronize()

        # Compute expected
        expected = np.zeros(simple_batch_data["num_systems"], dtype=dtype)
        offset = 0
        for s, count in enumerate(simple_batch_data["atom_counts"]):
            expected[s] = values_np[offset : offset + count].sum()
            offset += count

        rtol = 1e-5 if dtype == np.float32 else 1e-10
        np.testing.assert_allclose(result.numpy(), expected, rtol=rtol)


# =============================================================================
# Test Per-System Vector Reductions
# =============================================================================


class TestVectorReductions:
    """Tests for per-system vector reduction utilities."""

    @pytest.fixture
    def vector_data(self, device, simple_batch_data):
        """Create vector test data."""
        np.random.seed(123)
        total_atoms = simple_batch_data["total_atoms"]
        atom_counts = simple_batch_data["atom_counts"]

        # Generate random per-atom vectors
        vectors_np = np.random.randn(total_atoms, 3).astype(np.float64)

        # Compute expected results
        expected_sum = np.zeros((len(atom_counts), 3), dtype=np.float64)
        expected_max_norm = np.zeros(len(atom_counts), dtype=np.float64)
        expected_rms_norm = np.zeros(len(atom_counts), dtype=np.float64)

        offset = 0
        for s, count in enumerate(atom_counts):
            sys_vectors = vectors_np[offset : offset + count]
            expected_sum[s] = sys_vectors.sum(axis=0)
            norms = np.linalg.norm(sys_vectors, axis=1)
            expected_max_norm[s] = norms.max()
            expected_rms_norm[s] = np.sqrt((norms**2).mean())
            offset += count

        # Convert to warp vec3d format
        vectors_wp = wp.array(
            [tuple(v) for v in vectors_np], dtype=wp.vec3d, device=device
        )
        atom_ptr = wp.array(
            simple_batch_data["expected_atom_ptr"], dtype=wp.int32, device=device
        )
        batch_idx = wp.array(
            simple_batch_data["expected_batch_idx"], dtype=wp.int32, device=device
        )

        return {
            "vectors": vectors_wp,
            "atom_ptr": atom_ptr,
            "batch_idx": batch_idx,
            "num_systems": len(atom_counts),
            "expected_sum": expected_sum,
            "expected_max_norm": expected_max_norm,
            "expected_rms_norm": expected_rms_norm,
        }

    def test_sum_vectors_per_system_atom_ptr(self, device, vector_data):
        """Test vector sum reduction using atom_ptr mode."""
        result = sum_vectors_per_system(
            vector_data["vectors"],
            atom_ptr=vector_data["atom_ptr"],
            device=device,
        )
        wp.synchronize()

        result_np = np.array(result.numpy().tolist())
        np.testing.assert_allclose(
            result_np,
            vector_data["expected_sum"],
            rtol=1e-10,
        )

    def test_sum_vectors_per_system_batch_idx(self, device, vector_data):
        """Test vector sum reduction using batch_idx mode."""
        result = sum_vectors_per_system(
            vector_data["vectors"],
            batch_idx=vector_data["batch_idx"],
            num_systems=vector_data["num_systems"],
            device=device,
        )
        wp.synchronize()

        result_np = np.array(result.numpy().tolist())
        np.testing.assert_allclose(
            result_np,
            vector_data["expected_sum"],
            rtol=1e-10,
        )

    def test_max_norm_per_system_atom_ptr(self, device, vector_data):
        """Test max norm reduction using atom_ptr mode."""
        result = max_norm_per_system(
            vector_data["vectors"],
            atom_ptr=vector_data["atom_ptr"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_allclose(
            result.numpy(),
            vector_data["expected_max_norm"],
            rtol=1e-10,
        )

    def test_max_norm_per_system_batch_idx(self, device, vector_data):
        """Test max norm reduction using batch_idx mode."""
        result = max_norm_per_system(
            vector_data["vectors"],
            batch_idx=vector_data["batch_idx"],
            num_systems=vector_data["num_systems"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_allclose(
            result.numpy(),
            vector_data["expected_max_norm"],
            rtol=1e-10,
        )

    def test_rms_norm_per_system(self, device, vector_data):
        """Test RMS norm reduction using atom_ptr mode."""
        result = rms_norm_per_system(
            vector_data["vectors"],
            atom_ptr=vector_data["atom_ptr"],
            device=device,
        )
        wp.synchronize()

        np.testing.assert_allclose(
            result.numpy(),
            vector_data["expected_rms_norm"],
            rtol=1e-10,
        )

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype",
        [
            (wp.vec3f, np.float32),
            (wp.vec3d, np.float64),
        ],
    )
    def test_vector_reductions_dtypes(
        self, device, simple_batch_data, vec_dtype, scalar_dtype
    ):
        """Test vector reductions with different dtypes."""
        np.random.seed(456)
        total_atoms = simple_batch_data["total_atoms"]
        vectors_np = np.random.randn(total_atoms, 3).astype(scalar_dtype)

        vectors_wp = wp.array(
            [tuple(v) for v in vectors_np], dtype=vec_dtype, device=device
        )
        atom_ptr = wp.array(
            simple_batch_data["expected_atom_ptr"], dtype=wp.int32, device=device
        )

        result = max_norm_per_system(vectors_wp, atom_ptr=atom_ptr, device=device)
        wp.synchronize()

        # Compute expected
        expected = np.zeros(simple_batch_data["num_systems"], dtype=scalar_dtype)
        offset = 0
        for s, count in enumerate(simple_batch_data["atom_counts"]):
            norms = np.linalg.norm(vectors_np[offset : offset + count], axis=1)
            expected[s] = norms.max()
            offset += count

        rtol = 1e-5 if scalar_dtype == np.float32 else 1e-10
        np.testing.assert_allclose(result.numpy(), expected, rtol=rtol)


# =============================================================================
# Test Broadcasting
# =============================================================================


class TestBroadcasting:
    """Tests for broadcasting per-system values to per-atom arrays."""

    def test_broadcast_to_atoms_batch_idx(self, device, simple_batch_data):
        """Test broadcasting using batch_idx mode."""
        per_system_values = wp.array(
            [100.0, 200.0, 300.0], dtype=wp.float64, device=device
        )
        batch_idx = wp.array(
            simple_batch_data["expected_batch_idx"], dtype=wp.int32, device=device
        )

        result = broadcast_to_atoms(
            per_system_values,
            batch_idx=batch_idx,
            device=device,
        )
        wp.synchronize()

        # Expected: each atom gets its system's value
        expected = np.concatenate(
            [
                np.full(10, 100.0),
                np.full(15, 200.0),
                np.full(12, 300.0),
            ]
        )
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_broadcast_to_atoms_atom_ptr(self, device, simple_batch_data):
        """Test broadcasting using atom_ptr mode."""
        per_system_values = wp.array(
            [100.0, 200.0, 300.0], dtype=wp.float64, device=device
        )
        atom_ptr = wp.array(
            simple_batch_data["expected_atom_ptr"], dtype=wp.int32, device=device
        )

        result = broadcast_to_atoms(
            per_system_values,
            atom_ptr=atom_ptr,
            total_atoms=simple_batch_data["total_atoms"],
            device=device,
        )
        wp.synchronize()

        expected = np.concatenate(
            [
                np.full(10, 100.0),
                np.full(15, 200.0),
                np.full(12, 300.0),
            ]
        )
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_broadcast_to_atoms_preallocated(self, device, simple_batch_data):
        """Test broadcasting with pre-allocated output array."""
        per_system_values = wp.array(
            [100.0, 200.0, 300.0], dtype=wp.float64, device=device
        )
        batch_idx = wp.array(
            simple_batch_data["expected_batch_idx"], dtype=wp.int32, device=device
        )
        per_atom_values = wp.zeros(
            simple_batch_data["total_atoms"], dtype=wp.float64, device=device
        )

        result = broadcast_to_atoms(
            per_system_values,
            batch_idx=batch_idx,
            per_atom_values=per_atom_values,
            device=device,
        )
        wp.synchronize()

        assert result is per_atom_values
        expected = np.concatenate(
            [
                np.full(10, 100.0),
                np.full(15, 200.0),
                np.full(12, 300.0),
            ]
        )
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_broadcast_vectors(self, device, simple_batch_data):
        """Test broadcasting vector values."""
        per_system_values = wp.array(
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            dtype=wp.vec3d,
            device=device,
        )
        batch_idx = wp.array(
            simple_batch_data["expected_batch_idx"], dtype=wp.int32, device=device
        )

        result = broadcast_to_atoms(
            per_system_values,
            batch_idx=batch_idx,
            device=device,
        )
        wp.synchronize()

        result_np = np.array(result.numpy().tolist())
        assert result_np.shape == (simple_batch_data["total_atoms"], 3)

        # Check first system atoms have (1, 0, 0)
        np.testing.assert_array_equal(result_np[:10], np.array([[1, 0, 0]] * 10))
        # Check second system atoms have (0, 1, 0)
        np.testing.assert_array_equal(result_np[10:25], np.array([[0, 1, 0]] * 15))
        # Check third system atoms have (0, 0, 1)
        np.testing.assert_array_equal(result_np[25:], np.array([[0, 0, 1]] * 12))


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_system(self, device):
        """Test with a single system."""
        atom_counts = wp.array([50], dtype=wp.int32, device=device)

        atom_ptr = create_atom_ptr(atom_counts, device=device)
        batch_idx = create_batch_idx(atom_counts, total_atoms=50, device=device)
        wp.synchronize()

        np.testing.assert_array_equal(atom_ptr.numpy(), [0, 50])
        np.testing.assert_array_equal(batch_idx.numpy(), np.zeros(50, dtype=np.int32))

    def test_single_atom_per_system(self, device):
        """Test with one atom per system."""
        atom_counts = wp.array([1, 1, 1, 1], dtype=wp.int32, device=device)

        atom_ptr = create_atom_ptr(atom_counts, device=device)
        batch_idx = create_batch_idx(atom_counts, total_atoms=4, device=device)
        wp.synchronize()

        np.testing.assert_array_equal(atom_ptr.numpy(), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(batch_idx.numpy(), [0, 1, 2, 3])

    def test_large_batch(self, device):
        """Test with many systems."""
        np.random.seed(789)
        num_systems = 100
        atom_counts_np = np.random.randint(5, 20, size=num_systems).astype(np.int32)
        total_atoms = int(atom_counts_np.sum())

        atom_counts = wp.array(atom_counts_np, device=device)
        atom_ptr = create_atom_ptr(atom_counts, device=device)
        batch_idx = create_batch_idx(
            atom_counts, total_atoms=total_atoms, device=device
        )
        wp.synchronize()

        # Verify consistency
        atom_ptr_np = atom_ptr.numpy()
        batch_idx_np = batch_idx.numpy()

        for s in range(num_systems):
            start, end = atom_ptr_np[s], atom_ptr_np[s + 1]
            assert np.all(batch_idx_np[start:end] == s)

    def test_error_no_batch_info(self, device):
        """Test that error is raised when neither batch_idx nor atom_ptr provided."""
        values = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device=device)

        with pytest.raises(ValueError, match="Either batch_idx or atom_ptr"):
            sum_per_system(values, device=device)

    def test_error_no_num_systems(self, device):
        """Test that error is raised when num_systems not provided with batch_idx."""
        values = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device=device)
        batch_idx = wp.array([0, 0, 1], dtype=wp.int32, device=device)

        with pytest.raises(ValueError, match="num_systems required"):
            sum_per_system(values, batch_idx=batch_idx, device=device)


# =============================================================================
# Test Sync-Free Patterns
# =============================================================================


class TestSyncFreePatterns:
    """Tests demonstrating sync-free usage patterns."""

    def test_sync_free_workflow(self, device):
        """Test a complete workflow without host-device synchronization."""
        # Setup (known at initialization time)
        atom_counts = [10, 15, 12]
        num_systems = len(atom_counts)
        total_atoms = sum(atom_counts)

        # Pre-allocate arrays where we can
        atom_counts_wp = wp.array(np.array(atom_counts, dtype=np.int32), device=device)
        batch_idx = wp.zeros(total_atoms, dtype=wp.int32, device=device)
        result = wp.zeros(num_systems, dtype=wp.float64, device=device)

        # Create indices - atom_ptr is returned (small allocation, num_systems+1)
        atom_ptr = create_atom_ptr(atom_counts_wp, device=device)
        create_batch_idx(atom_counts_wp, batch_idx=batch_idx, device=device)

        # Create random data
        np.random.seed(999)
        values_np = np.random.randn(total_atoms).astype(np.float64)
        values = wp.array(values_np, dtype=wp.float64, device=device)

        # Compute sum (no sync needed - all sizes known)
        sum_per_system(values, atom_ptr=atom_ptr, result=result, device=device)

        # Only sync at the end when we need results
        wp.synchronize()

        # Verify
        expected = np.zeros(num_systems)
        offset = 0
        for s, count in enumerate(atom_counts):
            expected[s] = values_np[offset : offset + count].sum()
            offset += count

        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
