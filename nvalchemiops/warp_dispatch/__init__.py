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
Library-wide Warp kernel dispatch, overload registration, and validation.

This package provides module-agnostic primitives that any kernel family
in the library can use to build dispatch tables, register dtype overloads,
and validate output arrays.  It contains no dynamics-, neighbor-, or
interaction-specific logic.

Public API
----------
- :data:`DEFAULT_DTYPE_PAIRS`
- :func:`register_overloads`
- :func:`build_dispatch_table`
- :func:`dispatch`
- :func:`validate_out_array`
"""

from .core import (
    DEFAULT_DTYPE_PAIRS,
    build_dispatch_table,
    dispatch,
    register_overloads,
    validate_out_array,
)

__all__ = [
    "DEFAULT_DTYPE_PAIRS",
    "build_dispatch_table",
    "dispatch",
    "register_overloads",
    "validate_out_array",
]
