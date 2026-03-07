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

"""Native CUDA Shared Memory Utilities for High-Performance Reductions.

This module provides custom CUDA implementations using Warp's native function
feature to leverage shared memory for significant performance improvements over
standard atomic operations.

**Key Features:**
- Shared memory indexed accumulation (reduces global atomic contention)
- Indexed tile sum for batched per-system reductions
- Warp-level shuffle reductions (no shared memory needed)
- Full gradient support via adjoint snippets for automatic differentiation
- Float32 and float64 variants for all utilities

**Expected Performance Improvements:**
- Indexed accumulation: 2-3x speedup (reduces global atomics by block size factor)
- Indexed tile sum: 3-5x speedup (batched reductions)
- Warp reduce: 1.5-2x speedup (shuffle operations faster than atomics)

**Standalone Design:**
These utilities are designed to be benchmarked against pure Warp implementations.
Enable/disable via environment variable: ALCHEMI_USE_NATIVE_CUDA

Examples
--------
Using shared indexed accumulation in a force kernel::

    from nvalchemiops.dynamics.utils.tile_reductions import shared_indexed_accumulate_vec3_f32

    @wp.kernel
    def lj_forces_kernel(
        positions: wp.array(dtype=wp.vec3f),
        neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
        forces: wp.array(dtype=wp.float32),  # Flattened array
    ):
        i = wp.tid()
        force_i_x = 0.0
        force_i_y = 0.0
        force_i_z = 0.0

        for k in range(max_neighbors):
            j = neighbor_matrix[i, k]
            if j < 0:
                break

            # ... compute force_ij_x, force_ij_y, force_ij_z ...

            # Accumulate to atom i (local)
            force_i_x += force_ij_x
            force_i_y += force_ij_y
            force_i_z += force_ij_z

            # Accumulate to neighbor j (using shared memory)
            shared_indexed_accumulate_vec3_f32(j, -force_ij_x, -force_ij_y, -force_ij_z, forces)

        # Write atom i's force
        forces[i * 3 + 0] = force_i_x
        forces[i * 3 + 1] = force_i_y
        forces[i * 3 + 2] = force_i_z


Using indexed tile sum for per-system reductions::

    from nvalchemiops.dynamics.utils.tile_reductions import indexed_tile_sum_f32

    @wp.kernel
    def compute_system_energies(
        atom_energies: wp.array(dtype=wp.float32),
        system_idx: wp.array(dtype=wp.int32),
        num_systems: wp.int32,
        output: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        sys_idx = system_idx[i]
        energy = atom_energies[i]

        # Indexed reduction directly to system output
        indexed_tile_sum_f32(sys_idx, energy, num_systems, output)


Using warp-level reduction before atomic write::

    from nvalchemiops.dynamics.utils.tile_reductions import warp_reduce_vec3_f32

    @wp.kernel
    def force_reduction_kernel(
        local_forces_x: wp.array(dtype=wp.float32),
        local_forces_y: wp.array(dtype=wp.float32),
        local_forces_z: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),  # Flattened array
    ):
        i = wp.tid()
        force_x = local_forces_x[i]
        force_y = local_forces_y[i]
        force_z = local_forces_z[i]

        # Reduce within warp
        result_x = wp.float32(0.0)
        result_y = wp.float32(0.0)
        result_z = wp.float32(0.0)
        warp_reduce_vec3_f32(force_x, force_y, force_z, result_x, result_y, result_z)

        # Only lane 0 writes
        if i % 32 == 0:
            wp.atomic_add(output, 0, result_x)
            wp.atomic_add(output, 1, result_y)
            wp.atomic_add(output, 2, result_z)

References
----------
- Warp Custom Native Functions: https://nvidia.github.io/warp/user_guide/differentiability.html#custom-native-functions
- Example Implementations: https://gitlab-master.nvidia.com/sarichardson/cualchemi/-/tree/sarichardson/native_cell_list/nvalchemiops/neighborlist/natives
"""

import warp as wp

__all__ = [
    "shared_indexed_accumulate_vec3_f32",
    "shared_indexed_accumulate_vec3_f64",
    "indexed_tile_sum_f32",
    "indexed_tile_sum_f64",
    "warp_reduce_vec3_f32",
    "warp_reduce_vec3_f64",
]

# =============================================================================
# Utility 1: Shared Memory Indexed Accumulation (vec3)
# =============================================================================
# Purpose: Accumulate vec3 values to indexed locations using shared memory
# to reduce global atomic contention.
#
# Expected Speedup: 2-3x for force accumulation (reduces global atomics by
# factor of block_size)
# =============================================================================

# -----------------------------------------------------------------------------
# Float32 Variant
# -----------------------------------------------------------------------------


def _shared_indexed_accumulate_vec3_f32_impl(
    target_idx: wp.int32,
    value_x: wp.float32,
    value_y: wp.float32,
    value_z: wp.float32,
    global_output: wp.array(dtype=wp.float32),
):
    """Python stub for native CUDA implementation."""
    ...


shared_indexed_accumulate_vec3_f32 = wp.Function(
    func=_shared_indexed_accumulate_vec3_f32_impl,
    key="shared_indexed_accumulate_vec3_f32",
    namespace="",
    native_snippet="""
    // Thread and block info
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Shared memory for accumulation
    __shared__ int original_indices[1024];  // Original target indices (unchanged)
    __shared__ int status[1024];  // Status: 0=active, -1=merged
    __shared__ float shared_values_x[1024];
    __shared__ float shared_values_y[1024];
    __shared__ float shared_values_z[1024];
    __shared__ int max_active_tid;  // Track maximum active thread

    // Initialize shared max_active_tid
    if (tid == 0) {
        max_active_tid = -1;
    }

    // Initialize ALL shared memory slots to sentinels and zeros (not just tid)
    for (int i = tid; i < 1024; i += block_size) {
        original_indices[i] = -1;
        status[i] = -1;
        shared_values_x[i] = 0.0f;
        shared_values_y[i] = 0.0f;
        shared_values_z[i] = 0.0f;
    }
    __syncthreads();

    // Now write this thread's actual data and track max active tid
    if (target_idx >= 0) {
        original_indices[tid] = target_idx;
        status[tid] = 0;
        shared_values_x[tid] = value_x;
        shared_values_y[tid] = value_y;
        shared_values_z[tid] = value_z;
        atomicMax(&max_active_tid, tid);
    }
    __syncthreads();

    // Check if this thread is the first (lowest tid) with this target_idx
    bool is_first = true;
    for (int i = 0; i < tid; i++) {
        if (original_indices[i] >= 0 && original_indices[i] == target_idx) {
            is_first = false;
            status[tid] = -1;  // Mark this thread as merged
            break;
        }
    }
    __syncthreads();

    // Only the first thread with each target accumulates values from later threads
    // Only loop up to max_active_tid to avoid garbage from inactive threads
    if (is_first && target_idx >= 0) {
        for (int i = tid + 1; i <= max_active_tid; i++) {
            if (original_indices[i] >= 0 && original_indices[i] == target_idx) {
                shared_values_x[tid] += shared_values_x[i];
                shared_values_y[tid] += shared_values_y[i];
                shared_values_z[tid] += shared_values_z[i];
                status[i] = -1;
            }
        }
    }
    __syncthreads();

    // Write merged results to global memory (only first thread per target)
    if (status[tid] >= 0) {
        atomicAdd(&global_output[target_idx * 3 + 0], shared_values_x[tid]);
        atomicAdd(&global_output[target_idx * 3 + 1], shared_values_y[tid]);
        atomicAdd(&global_output[target_idx * 3 + 2], shared_values_z[tid]);
    }
""",
    adj_native_snippet="""
    // Gradient flows backward: adj_value += adj_output[target_idx]
    // This is the transpose of the accumulation operation
    adj_value_x += adj_global_output[target_idx * 3 + 0];
    adj_value_y += adj_global_output[target_idx * 3 + 1];
    adj_value_z += adj_global_output[target_idx * 3 + 2];
""",
)

# -----------------------------------------------------------------------------
# Float64 Variant
# -----------------------------------------------------------------------------


def _shared_indexed_accumulate_vec3_f64_impl(
    target_idx: wp.int32,
    value_x: wp.float64,
    value_y: wp.float64,
    value_z: wp.float64,
    global_output: wp.array(dtype=wp.float64),
):
    """Python stub for native CUDA implementation."""
    ...


shared_indexed_accumulate_vec3_f64 = wp.Function(
    func=_shared_indexed_accumulate_vec3_f64_impl,
    key="shared_indexed_accumulate_vec3_f64",
    namespace="",
    native_snippet="""
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    __shared__ int original_indices[1024];
    __shared__ int status[1024];
    __shared__ double shared_values_x[1024];
    __shared__ double shared_values_y[1024];
    __shared__ double shared_values_z[1024];
    __shared__ int max_active_tid;  // Track maximum active thread

    // Initialize shared max_active_tid
    if (tid == 0) {
        max_active_tid = -1;
    }

    // Initialize ALL shared memory slots to sentinels and zeros
    for (int i = tid; i < 1024; i += block_size) {
        original_indices[i] = -1;
        status[i] = -1;
        shared_values_x[i] = 0.0;
        shared_values_y[i] = 0.0;
        shared_values_z[i] = 0.0;
    }
    __syncthreads();

    // Write this thread's actual data and track max active tid
    if (target_idx >= 0) {
        original_indices[tid] = target_idx;
        status[tid] = 0;
        shared_values_x[tid] = value_x;
        shared_values_y[tid] = value_y;
        shared_values_z[tid] = value_z;
        atomicMax(&max_active_tid, tid);
    }
    __syncthreads();

    bool is_first = true;
    for (int i = 0; i < tid; i++) {
        if (original_indices[i] >= 0 && original_indices[i] == target_idx) {
            is_first = false;
            status[tid] = -1;
            break;
        }
    }
    __syncthreads();

    // Only loop up to max_active_tid to avoid garbage from inactive threads
    if (is_first && target_idx >= 0) {
        for (int i = tid + 1; i <= max_active_tid; i++) {
            if (original_indices[i] >= 0 && original_indices[i] == target_idx) {
                shared_values_x[tid] += shared_values_x[i];
                shared_values_y[tid] += shared_values_y[i];
                shared_values_z[tid] += shared_values_z[i];
                status[i] = -1;
            }
        }
    }
    __syncthreads();

    if (status[tid] >= 0) {
        atomicAdd(&global_output[target_idx * 3 + 0], shared_values_x[tid]);
        atomicAdd(&global_output[target_idx * 3 + 1], shared_values_y[tid]);
        atomicAdd(&global_output[target_idx * 3 + 2], shared_values_z[tid]);
    }
""",
    adj_native_snippet="""
    adj_value_x += adj_global_output[target_idx * 3 + 0];
    adj_value_y += adj_global_output[target_idx * 3 + 1];
    adj_value_z += adj_global_output[target_idx * 3 + 2];
""",
)

# =============================================================================
# Utility 2: Indexed Tile Sum
# =============================================================================
# Purpose: Perform per-system reductions in batched operations using shared
# memory to avoid loop-based approaches.
#
# Expected Speedup: 3-5x for batched reductions
# =============================================================================

# -----------------------------------------------------------------------------
# Float32 Variant
# -----------------------------------------------------------------------------


def _indexed_tile_sum_f32_impl(
    system_idx: wp.int32,
    value: wp.float32,
    num_systems: wp.int32,
    global_output: wp.array(dtype=wp.float32),
):
    """Python stub for native CUDA implementation."""
    ...


indexed_tile_sum_f32 = wp.Function(
    func=_indexed_tile_sum_f32_impl,
    key="indexed_tile_sum_f32",
    namespace="",
    native_snippet="""
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    __shared__ int original_indices[1024];
    __shared__ int status[1024];
    __shared__ float shared_values[1024];
    __shared__ int max_active_tid;  // Track maximum active thread

    // Initialize shared max_active_tid
    if (tid == 0) {
        max_active_tid = -1;
    }

    // Initialize ALL shared memory slots to sentinels and zeros
    for (int i = tid; i < 1024; i += block_size) {
        original_indices[i] = -1;
        status[i] = -1;
        shared_values[i] = 0.0;
    }
    __syncthreads();

    // Write this thread's actual data and track max active tid
    if (system_idx >= 0) {
        original_indices[tid] = system_idx;
        status[tid] = 0;
        shared_values[tid] = value;
        atomicMax(&max_active_tid, tid);
    }
    __syncthreads();

    bool is_first = true;
    for (int i = 0; i < tid; i++) {
        if (original_indices[i] >= 0 && original_indices[i] == system_idx) {
            is_first = false;
            status[tid] = -1;
            break;
        }
    }
    __syncthreads();

    // Only loop up to max_active_tid to avoid garbage from inactive threads
    if (is_first && system_idx >= 0) {
        for (int i = tid + 1; i <= max_active_tid; i++) {
            if (original_indices[i] >= 0 && original_indices[i] == system_idx) {
                shared_values[tid] += shared_values[i];
                status[i] = -1;
            }
        }
    }
    __syncthreads();

    if (status[tid] >= 0) {
        atomicAdd(&global_output[system_idx], shared_values[tid]);
    }
""",
    adj_native_snippet="""
    adj_value += adj_global_output[system_idx];
""",
)

# -----------------------------------------------------------------------------
# Float64 Variant
# -----------------------------------------------------------------------------


def _indexed_tile_sum_f64_impl(
    system_idx: wp.int32,
    value: wp.float64,
    num_systems: wp.int32,
    global_output: wp.array(dtype=wp.float64),
):
    """Python stub for native CUDA implementation."""
    pass


indexed_tile_sum_f64 = wp.Function(
    func=_indexed_tile_sum_f64_impl,
    key="indexed_tile_sum_f64",
    namespace="",
    native_snippet="""
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    __shared__ int original_indices[1024];
    __shared__ int status[1024];
    __shared__ double shared_values[1024];
    __shared__ int max_active_tid;  // Track maximum active thread

    // Initialize shared max_active_tid
    if (tid == 0) {
        max_active_tid = -1;
    }

    // Initialize ALL shared memory slots to sentinels and zeros
    for (int i = tid; i < 1024; i += block_size) {
        original_indices[i] = -1;
        status[i] = -1;
        shared_values[i] = 0.0;
    }
    __syncthreads();

    // Write this thread's actual data and track max active tid
    if (system_idx >= 0) {
        original_indices[tid] = system_idx;
        status[tid] = 0;
        shared_values[tid] = value;
        atomicMax(&max_active_tid, tid);
    }
    __syncthreads();

    bool is_first = true;
    for (int i = 0; i < tid; i++) {
        if (original_indices[i] >= 0 && original_indices[i] == system_idx) {
            is_first = false;
            status[tid] = -1;
            break;
        }
    }
    __syncthreads();

    // Only loop up to max_active_tid to avoid garbage from inactive threads
    if (is_first && system_idx >= 0) {
        for (int i = tid + 1; i <= max_active_tid; i++) {
            if (original_indices[i] >= 0 && original_indices[i] == system_idx) {
                shared_values[tid] += shared_values[i];
                status[i] = -1;
            }
        }
    }
    __syncthreads();

    if (status[tid] >= 0) {
        atomicAdd(&global_output[system_idx], shared_values[tid]);
    }
""",
    adj_native_snippet="""
    adj_value += adj_global_output[system_idx];
""",
)

# =============================================================================
# Utility 3: Warp-Level Force Reduction
# =============================================================================
# Purpose: Fast warp-level reduction using shuffle operations (no shared
# memory needed for 32 threads).
#
# Expected Speedup: 1.5-2x for force accumulation (warp shuffles faster than
# atomics)
# =============================================================================

# -----------------------------------------------------------------------------
# Float32 Variant
# -----------------------------------------------------------------------------


def _warp_reduce_vec3_f32_impl(
    force_x: wp.float32,
    force_y: wp.float32,
    force_z: wp.float32,
    warp_id: wp.int32,
    global_output: wp.array(dtype=wp.float32),
):
    """Python stub for native CUDA implementation."""
    pass


warp_reduce_vec3_f32 = wp.Function(
    func=_warp_reduce_vec3_f32_impl,
    key="warp_reduce_vec3_f32",
    namespace="",
    native_snippet="""
    // Warp shuffle reduction (no shared memory needed)
    float sum_x = force_x;
    float sum_y = force_y;
    float sum_z = force_z;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_x += __shfl_down_sync(0xffffffff, sum_x, offset);
        sum_y += __shfl_down_sync(0xffffffff, sum_y, offset);
        sum_z += __shfl_down_sync(0xffffffff, sum_z, offset);
    }

    // Lane 0 has the result - write directly to global output
    if ((threadIdx.x & 31) == 0) {
        global_output[warp_id * 3 + 0] = sum_x;
        global_output[warp_id * 3 + 1] = sum_y;
        global_output[warp_id * 3 + 2] = sum_z;
    }
""",
    adj_native_snippet="""
    // Read gradient from global output (only lane 0 has written, but all lanes need gradient)
    float grad_x = 0.0f;
    float grad_y = 0.0f;
    float grad_z = 0.0f;

    if ((threadIdx.x & 31) == 0) {
        grad_x = adj_global_output[warp_id * 3 + 0];
        grad_y = adj_global_output[warp_id * 3 + 1];
        grad_z = adj_global_output[warp_id * 3 + 2];
    }

    // Broadcast gradient to all lanes in the warp
    grad_x = __shfl_sync(0xffffffff, grad_x, 0);
    grad_y = __shfl_sync(0xffffffff, grad_y, 0);
    grad_z = __shfl_sync(0xffffffff, grad_z, 0);

    // Each thread gets the full gradient
    adj_force_x += grad_x;
    adj_force_y += grad_y;
    adj_force_z += grad_z;
""",
)

# -----------------------------------------------------------------------------
# Float64 Variant
# -----------------------------------------------------------------------------


def _warp_reduce_vec3_f64_impl(
    force_x: wp.float64,
    force_y: wp.float64,
    force_z: wp.float64,
    warp_id: wp.int32,
    global_output: wp.array(dtype=wp.float64),
):
    """Python stub for native CUDA implementation."""
    pass


warp_reduce_vec3_f64 = wp.Function(
    func=_warp_reduce_vec3_f64_impl,
    key="warp_reduce_vec3_f64",
    namespace="",
    native_snippet="""
    // Warp shuffle reduction (no shared memory needed)
    double sum_x = force_x;
    double sum_y = force_y;
    double sum_z = force_z;

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_x += __shfl_down_sync(0xffffffff, sum_x, offset);
        sum_y += __shfl_down_sync(0xffffffff, sum_y, offset);
        sum_z += __shfl_down_sync(0xffffffff, sum_z, offset);
    }

    // Lane 0 has the result - write directly to global output
    if ((threadIdx.x & 31) == 0) {
        global_output[warp_id * 3 + 0] = sum_x;
        global_output[warp_id * 3 + 1] = sum_y;
        global_output[warp_id * 3 + 2] = sum_z;
    }
""",
    adj_native_snippet="""
    // Read gradient from global output (only lane 0 has written, but all lanes need gradient)
    double grad_x = 0.0;
    double grad_y = 0.0;
    double grad_z = 0.0;

    if ((threadIdx.x & 31) == 0) {
        grad_x = adj_global_output[warp_id * 3 + 0];
        grad_y = adj_global_output[warp_id * 3 + 1];
        grad_z = adj_global_output[warp_id * 3 + 2];
    }

    // Broadcast gradient to all lanes in the warp
    grad_x = __shfl_sync(0xffffffff, grad_x, 0);
    grad_y = __shfl_sync(0xffffffff, grad_y, 0);
    grad_z = __shfl_sync(0xffffffff, grad_z, 0);

    // Each thread gets the full gradient
    adj_force_x += grad_x;
    adj_force_y += grad_y;
    adj_force_z += grad_z;
""",
)
