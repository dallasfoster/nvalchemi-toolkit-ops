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
Unit tests for launch helpers (dispatch, validation, overload registration).

Tests cover:
- ExecutionMode and KernelFamily types
- resolve_execution_mode: all 3 modes + mutual exclusivity error
- launch_family: correct kernel selection per mode, unsupported mode error
- validate_out_array: shape/dtype/device match
- register_overloads: dict keys and values
"""

from typing import Any

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.utils.launch_helpers import (
    ExecutionMode,
    KernelFamily,
    launch_family,
    register_overloads,
    resolve_execution_mode,
    validate_out_array,
)

DEVICE = "cuda:0"


# ==============================================================================
# Tests for ExecutionMode
# ==============================================================================


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_values(self):
        assert ExecutionMode.SINGLE.value == "single"
        assert ExecutionMode.BATCH_IDX.value == "batch_idx"
        assert ExecutionMode.ATOM_PTR.value == "atom_ptr"

    def test_members(self):
        assert len(ExecutionMode) == 3


# ==============================================================================
# Tests for KernelFamily
# ==============================================================================


class TestKernelFamily:
    """Tests for KernelFamily dataclass."""

    def test_all_fields(self):
        family = KernelFamily(single="s", batch_idx="b", atom_ptr="p")
        assert family.single == "s"
        assert family.batch_idx == "b"
        assert family.atom_ptr == "p"

    def test_defaults(self):
        family = KernelFamily(single="s")
        assert family.single == "s"
        assert family.batch_idx is None
        assert family.atom_ptr is None

    def test_frozen(self):
        family = KernelFamily(single="s")
        with pytest.raises(AttributeError):
            family.single = "other"


# ==============================================================================
# Tests for resolve_execution_mode
# ==============================================================================


class TestResolveExecutionMode:
    """Tests for resolve_execution_mode."""

    def test_single_mode(self):
        assert resolve_execution_mode(None, None) is ExecutionMode.SINGLE

    def test_batch_idx_mode(self):
        batch_idx = wp.array([0, 0, 1, 1], dtype=wp.int32, device=DEVICE)
        assert resolve_execution_mode(batch_idx, None) is ExecutionMode.BATCH_IDX

    def test_atom_ptr_mode(self):
        atom_ptr = wp.array([0, 2, 4], dtype=wp.int32, device=DEVICE)
        assert resolve_execution_mode(None, atom_ptr) is ExecutionMode.ATOM_PTR

    def test_mutual_exclusivity(self):
        batch_idx = wp.array([0, 0, 1, 1], dtype=wp.int32, device=DEVICE)
        atom_ptr = wp.array([0, 2, 4], dtype=wp.int32, device=DEVICE)
        with pytest.raises(ValueError, match="Provide batch_idx OR atom_ptr, not both"):
            resolve_execution_mode(batch_idx, atom_ptr)


# ==============================================================================
# Tests for launch_family
# ==============================================================================


# Simple kernels for testing dispatch
@wp.kernel
def _test_single_kernel(output: wp.array(dtype=wp.float32)):
    i = wp.tid()
    output[i] = 1.0


@wp.kernel
def _test_batch_kernel(output: wp.array(dtype=wp.float32)):
    i = wp.tid()
    output[i] = 2.0


@wp.kernel
def _test_ptr_kernel(output: wp.array(dtype=wp.float32)):
    i = wp.tid()
    output[i] = 3.0


class TestLaunchFamily:
    """Tests for launch_family."""

    def test_single_mode_launches_single_kernel(self):
        output = wp.zeros(4, dtype=wp.float32, device=DEVICE)
        family = KernelFamily(
            single=_test_single_kernel,
            batch_idx=_test_batch_kernel,
            atom_ptr=_test_ptr_kernel,
        )
        launch_family(
            family,
            mode=ExecutionMode.SINGLE,
            dim=4,
            inputs_single=[output],
            inputs_batch=[output],
            inputs_ptr=[output],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        np.testing.assert_array_equal(output.numpy(), [1.0, 1.0, 1.0, 1.0])

    def test_batch_idx_mode_launches_batch_kernel(self):
        output = wp.zeros(4, dtype=wp.float32, device=DEVICE)
        family = KernelFamily(
            single=_test_single_kernel,
            batch_idx=_test_batch_kernel,
            atom_ptr=_test_ptr_kernel,
        )
        launch_family(
            family,
            mode=ExecutionMode.BATCH_IDX,
            dim=4,
            inputs_single=[output],
            inputs_batch=[output],
            inputs_ptr=[output],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        np.testing.assert_array_equal(output.numpy(), [2.0, 2.0, 2.0, 2.0])

    def test_atom_ptr_mode_launches_ptr_kernel(self):
        output = wp.zeros(4, dtype=wp.float32, device=DEVICE)
        family = KernelFamily(
            single=_test_single_kernel,
            batch_idx=_test_batch_kernel,
            atom_ptr=_test_ptr_kernel,
        )
        launch_family(
            family,
            mode=ExecutionMode.ATOM_PTR,
            dim=4,
            inputs_single=[output],
            inputs_batch=[output],
            inputs_ptr=[output],
            device=DEVICE,
        )
        wp.synchronize_device(DEVICE)
        np.testing.assert_array_equal(output.numpy(), [3.0, 3.0, 3.0, 3.0])

    def test_unsupported_atom_ptr_raises(self):
        family = KernelFamily(single=_test_single_kernel, atom_ptr=None)
        with pytest.raises(ValueError, match="atom_ptr mode not supported"):
            launch_family(
                family,
                mode=ExecutionMode.ATOM_PTR,
                dim=4,
                inputs_single=[],
                inputs_ptr=[],
                device=DEVICE,
            )

    def test_unsupported_batch_idx_raises(self):
        family = KernelFamily(single=_test_single_kernel, batch_idx=None)
        with pytest.raises(ValueError, match="batch_idx mode not supported"):
            launch_family(
                family,
                mode=ExecutionMode.BATCH_IDX,
                dim=4,
                inputs_single=[],
                inputs_batch=[],
                device=DEVICE,
            )


# ==============================================================================
# Tests for validate_out_array
# ==============================================================================


class TestValidateOutArray:
    """Tests for validate_out_array."""

    def test_valid(self):
        ref = wp.zeros(10, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(10, dtype=wp.vec3f, device=DEVICE)
        validate_out_array(out, ref, "test_out")

    def test_shape_mismatch(self):
        ref = wp.zeros(10, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(5, dtype=wp.vec3f, device=DEVICE)
        with pytest.raises(ValueError, match="test_out shape mismatch"):
            validate_out_array(out, ref, "test_out")

    def test_dtype_mismatch(self):
        ref = wp.zeros(10, dtype=wp.vec3f, device=DEVICE)
        out = wp.zeros(10, dtype=wp.vec3d, device=DEVICE)
        with pytest.raises(ValueError, match="test_out dtype mismatch"):
            validate_out_array(out, ref, "test_out")


# ==============================================================================
# Tests for register_overloads
# ==============================================================================


@wp.kernel
def _generic_test_kernel(
    a: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
):
    """Generic kernel for overload registration tests."""
    i = wp.tid()
    pass


class TestRegisterOverloads:
    """Tests for register_overloads."""

    def test_default_dtype_pairs(self):
        overloads = register_overloads(
            _generic_test_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
        )
        assert wp.vec3f in overloads
        assert wp.vec3d in overloads
        assert len(overloads) == 2

    def test_custom_dtype_pairs(self):
        overloads = register_overloads(
            _generic_test_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
            dtype_pairs=((wp.vec3f, wp.float32),),
        )
        assert wp.vec3f in overloads
        assert wp.vec3d not in overloads
        assert len(overloads) == 1

    def test_custom_key_fn(self):
        overloads = register_overloads(
            _generic_test_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
            key_fn=lambda v, t: (v, t),
        )
        assert (wp.vec3f, wp.float32) in overloads
        assert (wp.vec3d, wp.float64) in overloads
        assert len(overloads) == 2

    def test_overload_values_are_not_none(self):
        overloads = register_overloads(
            _generic_test_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
        )
        for key, value in overloads.items():
            assert value is not None, f"Overload for {key} is None"
