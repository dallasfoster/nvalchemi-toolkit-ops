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

GPU-accelerated geometry optimization algorithms with both mutating and
non-mutating APIs for gradient tracking compatibility.

Available Optimizers
--------------------
fire
    FIRE (Fast Inertial Relaxation Engine) optimizer.
    MD-based optimization with adaptive timestep and velocity mixing.

API Patterns
------------
- Mutating APIs: Modify arrays in-place (e.g., `fire_md_step`)
- Non-mutating APIs: Return new arrays (e.g., `fire_md_step_out`)
"""

from .fire import (
    # Diagnostics (shared by FIRE and FIRE2)
    fire_compute_diagnostics,
    # FIRE - Mutating
    fire_velocity_mix,
    fire_md_step,
    fire_reset_velocities,
    # FIRE - Non-mutating
    fire_velocity_mix_out,
    fire_md_step_out,
    fire_reset_velocities_out,
    # FIRE2 - Mutating
    fire2_velocity_update,
    fire2_md_step,
    # FIRE2 - Non-mutating
    fire2_velocity_update_out,
    fire2_md_step_out,
)

__all__ = [
    # Diagnostics (shared by FIRE and FIRE2)
    "fire_compute_diagnostics",
    # FIRE - Mutating
    "fire_velocity_mix",
    "fire_md_step",
    "fire_reset_velocities",
    # FIRE - Non-mutating
    "fire_velocity_mix_out",
    "fire_md_step_out",
    "fire_reset_velocities_out",
    # FIRE2 - Mutating
    "fire2_velocity_update",
    "fire2_md_step",
    # FIRE2 - Non-mutating
    "fire2_velocity_update_out",
    "fire2_md_step_out",
]
