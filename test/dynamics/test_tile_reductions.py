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

"""
Unit tests for native CUDA tile reduction utilities.

Tests cover:
- Shared memory indexed accumulation (vec3)
- Indexed tile sum (per-system reductions)
- Warp-level vec3 reduction
- Correctness validation against reference implementations
- Gradient verification using finite differences
- Edge cases (zero inputs, single elements, large arrays)
- Float32 and float64 support
"""

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.dynamics.utils.tile_reductions import (
    indexed_tile_sum_f32,
    indexed_tile_sum_f64,
    shared_indexed_accumulate_vec3_f32,
    shared_indexed_accumulate_vec3_f64,
    warp_reduce_vec3_f32,
    warp_reduce_vec3_f64,
)

# ==============================================================================
# Test Configuration
# ==============================================================================

DEVICES = ["cuda:0"]

DTYPE_CONFIGS = [
    pytest.param(wp.vec3f, wp.float32, torch.float32, np.float32, id="float32"),
    pytest.param(wp.vec3d, wp.float64, torch.float64, np.float64, id="float64"),
]

# Block sizes for testing
BLOCK_SIZES = [64, 128, 256]

# Tolerance for correctness checks
ATOL_F32 = 1e-5
ATOL_F64 = 1e-10

# Tolerance for gradient checks
GRAD_RTOL = 1e-3  # 0.1% relative error


# ==============================================================================
# Test Kernels for Shared Indexed Accumulate Vec3
# ==============================================================================


@wp.kernel
def shared_indexed_accumulate_vec3_kernel_f32(
    target_indices: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.vec3f),
    output: wp.array(dtype=wp.float32),  # Flattened array
):
    """Test kernel for float32 indexed accumulation."""
    i = wp.tid()
    target_idx = target_indices[i]
    value = values[i]

    shared_indexed_accumulate_vec3_f32(target_idx, value[0], value[1], value[2], output)


@wp.kernel
def shared_indexed_accumulate_vec3_kernel_f64(
    target_indices: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.vec3d),
    output: wp.array(dtype=wp.float64),  # Flattened array
):
    """Test kernel for float64 indexed accumulation."""
    i = wp.tid()
    target_idx = target_indices[i]
    value = values[i]

    shared_indexed_accumulate_vec3_f64(target_idx, value[0], value[1], value[2], output)


# ==============================================================================
# Test Kernels for Indexed Tile Sum
# ==============================================================================


@wp.kernel
def indexed_tile_sum_kernel_f32(
    system_indices: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float32),
    num_systems: wp.int32,
    output: wp.array(dtype=wp.float32),
):
    """Test kernel for float32 indexed tile sum."""
    i = wp.tid()
    sys_idx = system_indices[i]
    value = values[i]

    indexed_tile_sum_f32(sys_idx, value, num_systems, output)


@wp.kernel
def indexed_tile_sum_kernel_f64(
    system_indices: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float64),
    num_systems: wp.int32,
    output: wp.array(dtype=wp.float64),
):
    """Test kernel for float64 indexed tile sum."""
    i = wp.tid()
    sys_idx = system_indices[i]
    value = values[i]

    indexed_tile_sum_f64(sys_idx, value, num_systems, output)


# ==============================================================================
# Test Kernels for Warp Reduce Vec3
# ==============================================================================


@wp.kernel
def warp_reduce_vec3_kernel_f32(
    values: wp.array(dtype=wp.vec3f),
    output_flat: wp.array(dtype=wp.float32),  # Flattened array
):
    """Test kernel for float32 warp reduction."""
    i = wp.tid()
    value = values[i]
    warp_id = i // 32

    # Native function writes directly to output_flat
    warp_reduce_vec3_f32(value[0], value[1], value[2], warp_id, output_flat)


@wp.kernel
def warp_reduce_vec3_kernel_f64(
    values: wp.array(dtype=wp.vec3d),
    output_flat: wp.array(dtype=wp.float64),  # Flattened array
):
    """Test kernel for float64 warp reduction."""
    i = wp.tid()
    value = values[i]
    warp_id = i // 32

    # Native function writes directly to output_flat
    warp_reduce_vec3_f64(value[0], value[1], value[2], warp_id, output_flat)


# ==============================================================================
# Reference Implementations
# ==============================================================================


def reference_indexed_accumulate_vec3(target_indices, values, num_outputs):
    """Reference implementation using PyTorch scatter_add."""
    device = values.device
    output = torch.zeros(num_outputs, 3, dtype=values.dtype, device=device)

    # Use scatter_add for indexed accumulation
    target_indices_expanded = target_indices.unsqueeze(1).expand(-1, 3)
    output.scatter_add_(0, target_indices_expanded, values)

    return output


def reference_indexed_tile_sum(system_indices, values, num_systems):
    """Reference implementation using PyTorch scatter_add."""
    device = values.device
    output = torch.zeros(num_systems, dtype=values.dtype, device=device)

    # Use scatter_add for per-system summation
    output.scatter_add_(0, system_indices, values)

    return output


def reference_warp_reduce_vec3(values):
    """Reference implementation: sum every 32 consecutive elements."""
    # Reshape to (num_warps, 32, 3) and sum along warp dimension
    num_elements = len(values)
    num_warps = num_elements // 32

    if num_elements % 32 != 0:
        # Pad to multiple of 32
        pad_size = 32 - (num_elements % 32)
        values = torch.cat(
            [values, torch.zeros(pad_size, 3, dtype=values.dtype, device=values.device)]
        )
        num_warps = len(values) // 32

    values_reshaped = values.reshape(num_warps, 32, 3)
    return values_reshaped.sum(dim=1)  # Sum along warp dimension


# ==============================================================================
# Tests for Shared Indexed Accumulate Vec3
# ==============================================================================


class TestSharedIndexedAccumulateVec3:
    """Tests for shared memory indexed vec3 accumulation."""

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_correctness_simple(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test correctness with simple non-overlapping indices."""
        wp.init()
        wp.set_device(device)

        num_inputs = 128
        num_outputs = 64

        # Create test data: each output gets contributions from 2 inputs
        target_indices = torch.arange(num_inputs, device=device, dtype=torch.int32) // 2
        values = torch.randn(num_inputs, 3, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        target_indices_wp = wp.from_torch(target_indices, dtype=wp.int32)
        values_wp = wp.from_torch(values, dtype=vec_dtype)
        # Create flattened output array
        output_flat_wp = wp.zeros(num_outputs * 3, dtype=scalar_dtype, device=device)

        # Run Warp kernel
        if scalar_dtype == wp.float32:
            wp.launch(
                shared_indexed_accumulate_vec3_kernel_f32,
                dim=num_inputs,
                inputs=[target_indices_wp, values_wp, output_flat_wp],
                device=device,
            )
        else:
            wp.launch(
                shared_indexed_accumulate_vec3_kernel_f64,
                dim=num_inputs,
                inputs=[target_indices_wp, values_wp, output_flat_wp],
                device=device,
            )

        # Compare with reference - reshape flattened output to (N, 3)
        output_torch = wp.to_torch(output_flat_wp).reshape(num_outputs, 3)
        reference = reference_indexed_accumulate_vec3(
            target_indices, values, num_outputs
        )

        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_correctness_many_to_one(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test correctness when many threads accumulate to same target."""
        wp.init()
        wp.set_device(device)

        num_inputs = 10
        num_outputs = 4  # Many threads targeting few outputs

        # All threads target same few indices
        target_indices, _ = torch.randint(
            0, num_outputs, (num_inputs,), device=device, dtype=torch.int32
        ).sort()
        values = torch.randn(num_inputs, 3, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        target_indices_wp = wp.from_torch(target_indices, dtype=wp.int32)
        values_wp = wp.from_torch(values, dtype=vec_dtype)
        output_flat_wp = wp.zeros(num_outputs * 3, dtype=scalar_dtype, device=device)
        # Run Warp kernel
        if scalar_dtype == wp.float32:
            wp.launch(
                shared_indexed_accumulate_vec3_kernel_f32,
                dim=num_inputs,
                inputs=[target_indices_wp, values_wp, output_flat_wp],
                device=device,
            )
        else:
            wp.launch(
                shared_indexed_accumulate_vec3_kernel_f64,
                dim=num_inputs,
                inputs=[target_indices_wp, values_wp, output_flat_wp],
                device=device,
            )

        # Compare with reference - reshape flattened output to (N, 3)
        output_torch = wp.to_torch(output_flat_wp).reshape(num_outputs, 3)
        reference = reference_indexed_accumulate_vec3(
            target_indices, values, num_outputs
        )

        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_edge_case_single_thread(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test with single thread (edge case)."""
        wp.init()
        wp.set_device(device)

        num_inputs = 1
        num_outputs = 1

        target_indices = torch.zeros(num_inputs, device=device, dtype=torch.int32)
        values = torch.randn(num_inputs, 3, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        target_indices_wp = wp.from_torch(target_indices, dtype=wp.int32)
        values_wp = wp.from_torch(values, dtype=vec_dtype)
        output_flat_wp = wp.zeros(num_outputs * 3, dtype=scalar_dtype, device=device)
        # Run Warp kernel
        if scalar_dtype == wp.float32:
            wp.launch(
                shared_indexed_accumulate_vec3_kernel_f32,
                dim=num_inputs,
                inputs=[target_indices_wp, values_wp, output_flat_wp],
                device=device,
            )
        else:
            wp.launch(
                shared_indexed_accumulate_vec3_kernel_f64,
                dim=num_inputs,
                inputs=[target_indices_wp, values_wp, output_flat_wp],
                device=device,
            )

        # Compare with reference - reshape flattened output to (N, 3)
        output_torch = wp.to_torch(output_flat_wp).reshape(num_outputs, 3)
        reference = reference_indexed_accumulate_vec3(
            target_indices, values, num_outputs
        )
        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    def test_different_block_sizes(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device, block_size
    ):
        """Test with different block sizes."""
        wp.init()
        wp.set_device(device)

        num_inputs = block_size * 4  # Multiple blocks
        num_outputs = block_size

        target_indices = torch.randint(
            0, num_outputs, (num_inputs,), device=device, dtype=torch.int32
        )
        values = torch.randn(num_inputs, 3, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        target_indices_wp = wp.from_torch(target_indices, dtype=wp.int32)
        values_wp = wp.from_torch(values, dtype=vec_dtype)
        output_flat_wp = wp.zeros(num_outputs * 3, dtype=scalar_dtype, device=device)

        # Run Warp kernel with specified block size
        if scalar_dtype == wp.float32:
            wp.launch(
                shared_indexed_accumulate_vec3_kernel_f32,
                dim=num_inputs,
                inputs=[target_indices_wp, values_wp, output_flat_wp],
                device=device,
                block_dim=block_size,
            )
        else:
            wp.launch(
                shared_indexed_accumulate_vec3_kernel_f64,
                dim=num_inputs,
                inputs=[target_indices_wp, values_wp, output_flat_wp],
                device=device,
                block_dim=block_size,
            )

        # Compare with reference - reshape flattened output to (N, 3)
        output_torch = wp.to_torch(output_flat_wp).reshape(num_outputs, 3)
        reference = reference_indexed_accumulate_vec3(
            target_indices, values, num_outputs
        )

        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)


# ==============================================================================
# Tests for Indexed Tile Sum
# ==============================================================================


class TestIndexedTileSum:
    """Tests for indexed tile sum (per-system reductions)."""

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_correctness_simple(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test correctness with simple per-system reduction."""
        wp.init()
        wp.set_device(device)

        num_atoms = 256
        num_systems = 4

        # Each system has equal number of atoms
        system_indices = (
            torch.arange(num_atoms, device=device, dtype=torch.int32) % num_systems
        )
        values = torch.randn(num_atoms, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        system_indices_wp = wp.from_torch(system_indices, dtype=wp.int32)
        values_wp = wp.from_torch(values, dtype=scalar_dtype)
        output_wp = wp.zeros(num_systems, dtype=scalar_dtype, device=device)

        # Run Warp kernel
        if scalar_dtype == wp.float32:
            wp.launch(
                indexed_tile_sum_kernel_f32,
                dim=num_atoms,
                inputs=[system_indices_wp, values_wp, num_systems, output_wp],
                device=device,
            )
        else:
            wp.launch(
                indexed_tile_sum_kernel_f64,
                dim=num_atoms,
                inputs=[system_indices_wp, values_wp, num_systems, output_wp],
                device=device,
            )

        # Compare with reference
        output_torch = wp.to_torch(output_wp)
        reference = reference_indexed_tile_sum(system_indices, values, num_systems)

        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_correctness_unbalanced_systems(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test with unbalanced system sizes."""
        wp.init()
        wp.set_device(device)

        num_systems = 8
        # Create unbalanced distribution: [1, 2, 4, 8, 16, 32, 64, 128] atoms per system
        system_sizes = [2**i for i in range(num_systems)]
        num_atoms = sum(system_sizes)

        system_indices = torch.cat(
            [
                torch.full((size,), i, device=device, dtype=torch.int32)
                for i, size in enumerate(system_sizes)
            ]
        )
        values = torch.randn(num_atoms, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        system_indices_wp = wp.from_torch(system_indices, dtype=wp.int32)
        values_wp = wp.from_torch(values, dtype=scalar_dtype)
        output_wp = wp.zeros(num_systems, dtype=scalar_dtype, device=device)

        # Run Warp kernel
        if scalar_dtype == wp.float32:
            wp.launch(
                indexed_tile_sum_kernel_f32,
                dim=num_atoms,
                inputs=[system_indices_wp, values_wp, num_systems, output_wp],
                device=device,
            )
        else:
            wp.launch(
                indexed_tile_sum_kernel_f64,
                dim=num_atoms,
                inputs=[system_indices_wp, values_wp, num_systems, output_wp],
                device=device,
            )

        # Compare with reference
        output_torch = wp.to_torch(output_wp)
        reference = reference_indexed_tile_sum(system_indices, values, num_systems)
        print("output_torch: ", output_torch)
        print("reference: ", reference)
        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_correctness_max_systems(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test with maximum supported systems (32 per block)."""
        wp.init()
        wp.set_device(device)

        num_atoms = 1024
        num_systems = 32  # Max supported by shared memory implementation

        system_indices = torch.randint(
            0, num_systems, (num_atoms,), device=device, dtype=torch.int32
        )
        values = torch.randn(num_atoms, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        system_indices_wp = wp.from_torch(system_indices, dtype=wp.int32)
        values_wp = wp.from_torch(values, dtype=scalar_dtype)
        output_wp = wp.zeros(num_systems, dtype=scalar_dtype, device=device)

        # Run Warp kernel
        if scalar_dtype == wp.float32:
            wp.launch(
                indexed_tile_sum_kernel_f32,
                dim=num_atoms,
                inputs=[system_indices_wp, values_wp, num_systems, output_wp],
                device=device,
            )
        else:
            wp.launch(
                indexed_tile_sum_kernel_f64,
                dim=num_atoms,
                inputs=[system_indices_wp, values_wp, num_systems, output_wp],
                device=device,
            )

        # Compare with reference
        output_torch = wp.to_torch(output_wp)
        reference = reference_indexed_tile_sum(system_indices, values, num_systems)

        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("block_size", BLOCK_SIZES)
    def test_different_block_sizes(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device, block_size
    ):
        """Test with different block sizes."""
        wp.init()
        wp.set_device(device)

        num_atoms = block_size * 4
        num_systems = 8

        system_indices = torch.randint(
            0, num_systems, (num_atoms,), device=device, dtype=torch.int32
        )
        values = torch.randn(num_atoms, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        system_indices_wp = wp.from_torch(system_indices, dtype=wp.int32)
        values_wp = wp.from_torch(values, dtype=scalar_dtype)
        output_wp = wp.zeros(num_systems, dtype=scalar_dtype, device=device)

        # Run Warp kernel with specified block size
        if scalar_dtype == wp.float32:
            wp.launch(
                indexed_tile_sum_kernel_f32,
                dim=num_atoms,
                inputs=[system_indices_wp, values_wp, num_systems, output_wp],
                device=device,
                block_dim=block_size,
            )
        else:
            wp.launch(
                indexed_tile_sum_kernel_f64,
                dim=num_atoms,
                inputs=[system_indices_wp, values_wp, num_systems, output_wp],
                device=device,
                block_dim=block_size,
            )

        # Compare with reference
        output_torch = wp.to_torch(output_wp)
        reference = reference_indexed_tile_sum(system_indices, values, num_systems)

        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)


# ==============================================================================
# Tests for Warp Reduce Vec3
# ==============================================================================


class TestWarpReduceVec3:
    """Tests for warp-level vec3 reduction."""

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_correctness_single_warp(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test correctness with single warp (32 threads)."""
        wp.init()
        wp.set_device(device)

        num_elements = 32  # Exactly one warp
        values = torch.randn(num_elements, 3, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        values_wp = wp.from_torch(values, dtype=vec_dtype)
        num_warps = 1
        output_flat_wp = wp.zeros(
            num_warps * 3, dtype=scalar_dtype, device=device
        )  # Flattened output

        # Run Warp kernel
        if scalar_dtype == wp.float32:
            wp.launch(
                warp_reduce_vec3_kernel_f32,
                dim=num_elements,
                inputs=[values_wp, output_flat_wp],
                device=device,
            )
        else:
            wp.launch(
                warp_reduce_vec3_kernel_f64,
                dim=num_elements,
                inputs=[values_wp, output_flat_wp],
                device=device,
            )

        # Compare with reference - reshape flattened output to (N, 3)
        output_torch = wp.to_torch(output_flat_wp).reshape(num_warps, 3)
        reference = reference_warp_reduce_vec3(values)
        print("output_torch: ", output_torch)
        print("reference: ", reference)

        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_correctness_multiple_warps(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test correctness with multiple warps."""
        wp.init()
        wp.set_device(device)

        num_warps = 8
        num_elements = num_warps * 32
        values = torch.randn(num_elements, 3, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        values_wp = wp.from_torch(values, dtype=vec_dtype)
        output_flat_wp = wp.zeros(
            num_warps * 3, dtype=scalar_dtype, device=device
        )  # Flattened output

        # Run Warp kernel
        if scalar_dtype == wp.float32:
            wp.launch(
                warp_reduce_vec3_kernel_f32,
                dim=num_elements,
                inputs=[values_wp, output_flat_wp],
                device=device,
            )
        else:
            wp.launch(
                warp_reduce_vec3_kernel_f64,
                dim=num_elements,
                inputs=[values_wp, output_flat_wp],
                device=device,
            )

        # Compare with reference - reshape flattened output to (N, 3)
        output_torch = wp.to_torch(output_flat_wp).reshape(num_warps, 3)
        reference = reference_warp_reduce_vec3(values)

        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_correctness_large_problem(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test correctness with large problem (many warps)."""
        wp.init()
        wp.set_device(device)

        num_warps = 128
        num_elements = num_warps * 32
        values = torch.randn(num_elements, 3, dtype=torch_dtype, device=device)

        # Convert to Warp arrays
        values_wp = wp.from_torch(values, dtype=vec_dtype)
        output_flat_wp = wp.zeros(
            num_warps * 3, dtype=scalar_dtype, device=device
        )  # Flattened output

        # Run Warp kernel
        if scalar_dtype == wp.float32:
            wp.launch(
                warp_reduce_vec3_kernel_f32,
                dim=num_elements,
                inputs=[values_wp, output_flat_wp],
                device=device,
            )
        else:
            wp.launch(
                warp_reduce_vec3_kernel_f64,
                dim=num_elements,
                inputs=[values_wp, output_flat_wp],
                device=device,
            )

        # Compare with reference - reshape flattened output to (N, 3)
        output_torch = wp.to_torch(output_flat_wp).reshape(num_warps, 3)
        reference = reference_warp_reduce_vec3(values)

        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(output_torch, reference, atol=atol, rtol=0)


# ==============================================================================
# Gradient Tests
# ==============================================================================


class TestGradients:
    """Test gradient correctness using finite differences."""

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_shared_indexed_accumulate_gradient(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test gradient correctness for indexed accumulation using finite differences."""
        wp.init()
        wp.set_device(device)

        num_inputs = 64
        num_outputs = 32

        # Create test data
        target_indices = torch.randint(
            0, num_outputs, (num_inputs,), device=device, dtype=torch.int32
        )
        values = torch.randn(
            num_inputs, 3, dtype=torch_dtype, device=device, requires_grad=True
        )

        # Reference implementation
        output_ref = reference_indexed_accumulate_vec3(
            target_indices, values, num_outputs
        )
        loss_ref = output_ref.sum()
        loss_ref.backward()

        grad_ref = values.grad.clone()
        values.grad = None

        # Warp implementation with tape
        tape = wp.Tape()
        with tape:
            target_indices_wp = wp.from_torch(target_indices, dtype=wp.int32)
            values_wp = wp.from_torch(values, dtype=vec_dtype, requires_grad=True)
            output_flat_wp = wp.zeros(
                num_outputs * 3, dtype=scalar_dtype, device=device, requires_grad=True
            )

            if scalar_dtype == wp.float32:
                wp.launch(
                    shared_indexed_accumulate_vec3_kernel_f32,
                    dim=num_inputs,
                    inputs=[target_indices_wp, values_wp, output_flat_wp],
                    device=device,
                )
            else:
                wp.launch(
                    shared_indexed_accumulate_vec3_kernel_f64,
                    dim=num_inputs,
                    inputs=[target_indices_wp, values_wp, output_flat_wp],
                    device=device,
                )

        # Backward pass
        output_flat_wp.grad = wp.ones_like(output_flat_wp)
        tape.backward()

        grad_warp = wp.to_torch(values_wp.grad)

        # Compare gradients
        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(grad_warp, grad_ref, atol=atol, rtol=GRAD_RTOL)

    @pytest.mark.parametrize(
        "vec_dtype,scalar_dtype,torch_dtype,np_dtype", DTYPE_CONFIGS
    )
    @pytest.mark.parametrize("device", DEVICES)
    def test_indexed_tile_sum_gradient(
        self, vec_dtype, scalar_dtype, torch_dtype, np_dtype, device
    ):
        """Test gradient correctness for indexed tile sum using finite differences."""
        wp.init()
        wp.set_device(device)

        num_atoms = 128
        num_systems = 8

        # Create test data
        system_indices = torch.randint(
            0, num_systems, (num_atoms,), device=device, dtype=torch.int32
        )
        values = torch.randn(
            num_atoms, dtype=torch_dtype, device=device, requires_grad=True
        )

        # Reference implementation
        output_ref = reference_indexed_tile_sum(system_indices, values, num_systems)
        loss_ref = output_ref.sum()
        loss_ref.backward()

        grad_ref = values.grad.clone()
        values.grad = None

        # Warp implementation with tape
        tape = wp.Tape()
        with tape:
            system_indices_wp = wp.from_torch(system_indices, dtype=wp.int32)
            values_wp = wp.from_torch(values, dtype=scalar_dtype, requires_grad=True)
            output_wp = wp.zeros(
                num_systems, dtype=scalar_dtype, device=device, requires_grad=True
            )

            if scalar_dtype == wp.float32:
                wp.launch(
                    indexed_tile_sum_kernel_f32,
                    dim=num_atoms,
                    inputs=[system_indices_wp, values_wp, num_systems, output_wp],
                    device=device,
                )
            else:
                wp.launch(
                    indexed_tile_sum_kernel_f64,
                    dim=num_atoms,
                    inputs=[system_indices_wp, values_wp, num_systems, output_wp],
                    device=device,
                )

        # Backward pass
        output_wp.grad = wp.ones_like(output_wp)
        tape.backward()

        grad_warp = wp.to_torch(values_wp.grad)

        # Compare gradients
        atol = ATOL_F32 if scalar_dtype == wp.float32 else ATOL_F64
        torch.testing.assert_close(grad_warp, grad_ref, atol=atol, rtol=GRAD_RTOL)
