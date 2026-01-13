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
Dynamics Module
===============

GPU-accelerated Warp kernels for molecular dynamics integrators and
geometry optimization algorithms.

All kernels use direct array inputs (no structs) and provide both:
- Mutating (in-place) versions for efficiency
- Non-mutating versions for gradient tracking

Submodules
----------
integrators
    MD integrators: velocity Verlet (NVE), Langevin (NVT), Nosé-Hoover (NVT)
optimizers
    Geometry optimizers: FIRE, FIRE2
utils
    Utility functions: temperature computation, velocity initialization

Example
-------
>>> import warp as wp
>>> from nvalchemiops.dynamics.integrators import velocity_verlet_position_update
>>>
>>> # Mutating (in-place) API
>>> velocity_verlet_position_update(
...     positions, velocities, forces, masses, dt
... )
>>>
>>> # Non-mutating API (for gradient tracking)
>>> from nvalchemiops.dynamics.integrators import velocity_verlet_position_update_out
>>> new_pos, new_vel = velocity_verlet_position_update_out(
...     positions, velocities, forces, masses, dt
... )
"""

from . import integrators, optimizers, utils

__all__ = [
    # Submodules
    "integrators",
    "optimizers",
    "utils",
]
