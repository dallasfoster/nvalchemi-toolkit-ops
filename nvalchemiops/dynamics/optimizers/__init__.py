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
Geometry Optimizers
===================

GPU-accelerated geometry optimization algorithms.

Available Optimizers
--------------------
FIRE (Fast Inertial Relaxation Engine)
    MD-based optimization with adaptive timestep and velocity mixing.

FIRE2 (Fast Inertial Relaxation Engine v2)
    Improved FIRE with adaptive damping and velocity mixing.

Main API Functions
------------------
fire_step
    Full FIRE step with MD integration. Supports single system,
    batch_idx, and atom_ptr batching modes, with optional downhill check.

fire_update
    FIRE velocity mixing and parameter update WITHOUT MD integration.
    Use for variable-cell optimization with packed extended arrays.

fire2_step
    Complete FIRE2 optimization step.
    Uses batch_idx batching only.

Kernel Selection
----------------
- Neither batch_idx nor atom_ptr: single system kernel
- batch_idx provided: batch_idx kernel (one thread per atom)
- atom_ptr provided: ptr/CSR kernel (one thread per system)
- Downhill arrays provided: downhill variant with energy check
"""

from .fire import (
    _fire_step_downhill_batch_idx_kernel,
    _fire_step_downhill_kernel,
    _fire_step_downhill_ptr_kernel,
    _fire_step_no_downhill_batch_idx_kernel,
    # Low-level kernels (for advanced use)
    _fire_step_no_downhill_kernel,
    _fire_step_no_downhill_ptr_kernel,
    _fire_update_params_downhill_batch_idx_kernel,
    _fire_update_params_downhill_kernel,
    _fire_update_params_downhill_ptr_kernel,
    _fire_update_params_no_downhill_batch_idx_kernel,
    _fire_update_params_no_downhill_kernel,
    _fire_update_params_no_downhill_ptr_kernel,
    # Unified API (recommended)
    fire_step,
    fire_update,
)
from .fire2 import fire2_step

__all__ = [
    # Unified API (recommended)
    "fire_step",
    "fire_update",
    "fire2_step",
    # Low-level kernels (for advanced use)
    "_fire_step_no_downhill_kernel",
    "_fire_step_no_downhill_batch_idx_kernel",
    "_fire_step_no_downhill_ptr_kernel",
    "_fire_step_downhill_kernel",
    "_fire_step_downhill_batch_idx_kernel",
    "_fire_step_downhill_ptr_kernel",
    "_fire_update_params_no_downhill_kernel",
    "_fire_update_params_no_downhill_batch_idx_kernel",
    "_fire_update_params_no_downhill_ptr_kernel",
    "_fire_update_params_downhill_kernel",
    "_fire_update_params_downhill_batch_idx_kernel",
    "_fire_update_params_downhill_ptr_kernel",
]
