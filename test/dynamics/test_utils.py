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
Unit tests for dynamics utility functions.

Tests cover:
- Kinetic energy computation
- Temperature calculation
- Maxwell-Boltzmann velocity initialization
- Center of mass motion removal
- Float32 and float64 support
"""

import pytest
import numpy as np
import warp as wp

from nvalchemiops.dynamics.utils import (
    compute_kinetic_energy,
    compute_temperature,
    initialize_velocities,
    initialize_velocities_out,
    remove_com_motion,
    remove_com_motion_out,
)


# ==============================================================================
# Test Configuration
# ==============================================================================

DEVICES = ["cuda:0"]

DTYPE_CONFIGS = [
    pytest.param(wp.vec3f, wp.float32, np.float32, id="float32"),
    pytest.param(wp.vec3d, wp.float64, np.float64, id="float64"),
]


# ==============================================================================
# Kinetic Energy Tests
# ==============================================================================


class TestKineticEnergy:
    """Test kinetic energy computation functions."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that compute_kinetic_energy executes without error."""
        num_atoms = 100
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        ke = compute_kinetic_energy(velocities, masses, device=device)
        wp.synchronize_device(device)

        assert ke.shape[0] == 1
        ke_val = ke.numpy()[0]
        assert ke_val > 0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_device_inference(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test kinetic energy computation with device inferred from arrays."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        # Call without explicit device
        ke = compute_kinetic_energy(velocities, masses)
        wp.synchronize_device(device)

        assert ke.shape[0] == 1

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_preallocated(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test kinetic energy computation with pre-allocated output array."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        kinetic_energy_out = wp.zeros(1, dtype=dtype_scalar, device=device)

        ke = compute_kinetic_energy(
            velocities, masses, kinetic_energy=kinetic_energy_out, device=device
        )

        wp.synchronize_device(device)
        assert ke is kinetic_energy_out

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_value(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that kinetic energy is computed correctly: KE = 0.5 * sum(m * v^2)."""
        vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np_dtype)
        mass = np.array([1.0, 1.0], dtype=np_dtype)

        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)

        ke = compute_kinetic_energy(velocities, masses, device=device)
        wp.synchronize_device(device)

        expected_ke = 0.5 * (1.0 * 1.0 + 1.0 * 4.0)
        np.testing.assert_allclose(ke.numpy()[0], expected_ke, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_batched(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test batched kinetic energy computation."""
        num_systems = 3
        atoms_per_system = 10
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        ke = compute_kinetic_energy(
            velocities, masses, batch_idx=batch_idx, num_systems=num_systems, device=device
        )
        wp.synchronize_device(device)

        assert ke.shape[0] == num_systems
        ke_vals = ke.numpy()
        for i in range(num_systems):
            assert ke_vals[i] > 0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_kinetic_energy_batched_preallocated(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test batched kinetic energy computation with pre-allocated output."""
        num_systems = 2
        num_atoms = 40

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * 20 + [1] * 20, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        kinetic_energy_out = wp.zeros(num_systems, dtype=dtype_scalar, device=device)

        ke = compute_kinetic_energy(
            velocities, masses, batch_idx=batch_idx, num_systems=num_systems,
            kinetic_energy=kinetic_energy_out, device=device
        )

        wp.synchronize_device(device)
        assert ke is kinetic_energy_out


# ==============================================================================
# Temperature Computation Tests
# ==============================================================================


class TestTemperatureComputation:
    """Test temperature computation functions."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that compute_temperature executes without error."""
        num_atoms = 100
        dof = 3 * num_atoms
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        temp = compute_temperature(velocities, masses, num_atoms, dof=dof, device=device)
        wp.synchronize_device(device)

        assert temp.shape[0] == 1
        temp_val = temp.numpy()[0]
        assert temp_val > 0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_device_inference(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test temperature computation with device inferred from arrays."""
        num_atoms = 50
        dof = 3 * num_atoms

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        # Call without explicit device
        temp = compute_temperature(velocities, masses, num_atoms, dof=dof)
        wp.synchronize_device(device)

        assert temp.shape[0] == 1

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_preallocated_ke(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test temperature computation with pre-computed kinetic energy."""
        num_atoms = 50
        dof = 3 * num_atoms

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        # Pre-compute kinetic energy
        ke = compute_kinetic_energy(velocities, masses, device=device)

        temp = compute_temperature(
            velocities, masses, num_atoms, dof=dof, kinetic_energy=ke, device=device
        )

        wp.synchronize_device(device)
        assert temp.shape[0] == 1

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_value(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that temperature follows kT = 2*KE / dof."""
        num_atoms = 100
        dof = 3 * num_atoms
        np.random.seed(42)

        vel = np.random.randn(num_atoms, 3).astype(np_dtype)
        mass = np.ones(num_atoms, dtype=np_dtype)

        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)

        temp = compute_temperature(velocities, masses, num_atoms, dof=dof, device=device)
        wp.synchronize_device(device)

        ke = 0.5 * np.sum(mass[:, np.newaxis] * vel ** 2)
        expected_temp = 2.0 * ke / dof

        np.testing.assert_allclose(temp.numpy()[0], expected_temp, rtol=1e-4)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_batched(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test batched temperature computation."""
        num_systems = 3
        atoms_per_system = 50
        total_atoms = num_systems * atoms_per_system
        dof_per_system = 3 * atoms_per_system
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        temp = compute_temperature(
            velocities, masses, atoms_per_system, dof=dof_per_system,
            batch_idx=batch_idx, num_systems=num_systems, device=device
        )
        wp.synchronize_device(device)

        assert temp.shape[0] == num_systems

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_temperature_batched_with_ke(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test batched temperature computation with pre-computed kinetic energy."""
        num_systems = 2
        atoms_per_system = 20
        total_atoms = num_systems * atoms_per_system
        dof_per_system = 3 * atoms_per_system

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * atoms_per_system + [1] * atoms_per_system, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        # Pre-compute kinetic energy
        ke = compute_kinetic_energy(
            velocities, masses, batch_idx=batch_idx, num_systems=num_systems, device=device
        )

        temp = compute_temperature(
            velocities, masses, atoms_per_system, dof=dof_per_system,
            kinetic_energy=ke, batch_idx=batch_idx, num_systems=num_systems, device=device
        )

        wp.synchronize_device(device)
        assert temp.shape[0] == num_systems


# ==============================================================================
# Velocity Initialization Tests
# ==============================================================================


class TestVelocityInitialization:
    """Test Maxwell-Boltzmann velocity initialization."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that initialize_velocities executes without error."""
        num_atoms = 100

        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)

        initialize_velocities(velocities, masses, temperature, random_seed=42, device=device)
        wp.synchronize_device(device)

        vel_np = velocities.numpy()
        assert not np.allclose(vel_np, 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_device_inference(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test velocity initialization with device inferred from arrays."""
        num_atoms = 50

        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)

        # Call without explicit device
        initialize_velocities(velocities, masses, temperature, random_seed=42)
        wp.synchronize_device(device)

        vel_np = velocities.numpy()
        assert not np.allclose(vel_np, 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_out_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating velocity initialization."""
        num_atoms = 100

        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)

        velocities = initialize_velocities_out(masses, temperature, random_seed=42, device=device)
        wp.synchronize_device(device)

        assert velocities.shape[0] == num_atoms
        vel_np = velocities.numpy()
        assert not np.allclose(vel_np, 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_out_preallocated(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test velocity initialization with pre-allocated output (COM removal disabled)."""
        num_atoms = 50

        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperature = wp.array([1.0], dtype=dtype_scalar, device=device)
        velocities_out = wp.zeros(num_atoms, dtype=dtype_vec, device=device)

        # Disable COM removal to get the same array back
        vel = initialize_velocities_out(
            masses, temperature, velocities_out=velocities_out, random_seed=42,
            remove_com=False, device=device
        )

        wp.synchronize_device(device)
        assert vel is velocities_out
        assert not np.allclose(vel.numpy(), 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_temperature(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that initialized velocities give correct temperature."""
        num_atoms = 10000
        target_temp = 1.0
        dof = 3 * num_atoms

        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperature = wp.array([target_temp], dtype=dtype_scalar, device=device)

        initialize_velocities(velocities, masses, temperature, random_seed=42, device=device)

        measured_temp = compute_temperature(velocities, masses, num_atoms, dof=dof, device=device)
        wp.synchronize_device(device)

        np.testing.assert_allclose(measured_temp.numpy()[0], target_temp, rtol=0.1)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_batched(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test batched velocity initialization."""
        num_systems = 3
        atoms_per_system = 100
        total_atoms = num_systems * atoms_per_system

        velocities = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperatures = wp.array([0.5, 1.0, 2.0], dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        initialize_velocities(
            velocities, masses, temperatures, random_seed=42, batch_idx=batch_idx, device=device
        )
        wp.synchronize_device(device)

        vel_np = velocities.numpy()
        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            assert not np.allclose(vel_np[start:end], 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_initialize_velocities_out_batched(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating batched velocity initialization."""
        num_atoms = 40
        num_systems = 2

        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * 20 + [1] * 20, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        temperature = wp.array([1.0, 1.0], dtype=dtype_scalar, device=device)

        vel_out = initialize_velocities_out(
            masses, temperature, random_seed=42, batch_idx=batch_idx, device=device
        )

        wp.synchronize_device(device)
        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", [(wp.vec3d, wp.float64, np.float64)])
    def test_initialize_velocities_different_temps(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test batched initialization with different temperatures per system."""
        num_systems = 2
        atoms_per_system = 5000
        total_atoms = num_systems * atoms_per_system
        temps = [0.5, 2.0]
        # DOF is 3*N - 3 because COM motion is removed by default
        dof = 3 * atoms_per_system - 3

        velocities = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        temperatures = wp.array(temps, dtype=dtype_scalar, device=device)
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        initialize_velocities(
            velocities, masses, temperatures, random_seed=42, batch_idx=batch_idx, device=device
        )
        wp.synchronize()

        vel_np = velocities.numpy()
        mass_np = masses.numpy()
        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            print(f"System {sys_id}: start={start}, end={end}")
            sys_vel = vel_np[start:end]
            sys_mass = mass_np[start:end]
            ke = 0.5 * np.sum(sys_mass[:, np.newaxis] * sys_vel ** 2)
            measured_temp = 2.0 * ke / dof
            print(measured_temp, temps[sys_id])
            np.testing.assert_allclose(measured_temp, temps[sys_id], rtol=0.15)


# ==============================================================================
# COM Motion Removal Tests
# ==============================================================================


class TestCOMMotionRemoval:
    """Test center of mass motion removal functions."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that remove_com_motion executes without error."""
        num_atoms = 100
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        remove_com_motion(velocities, masses, device=device)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_device_inference(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test COM removal with device inferred from arrays."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        # Call without explicit device
        remove_com_motion(velocities, masses)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_out_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating COM motion removal."""
        num_atoms = 100
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        vel_out = remove_com_motion_out(velocities, masses, device=device)
        wp.synchronize_device(device)

        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_out_preserves_input(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that non-mutating COM removal preserves input."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )

        vel_orig = velocities.numpy().copy()

        vel_out = remove_com_motion_out(velocities, masses, device=device)
        wp.synchronize_device(device)

        np.testing.assert_array_equal(velocities.numpy(), vel_orig)
        assert not np.allclose(vel_out.numpy(), vel_orig)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_out_preallocated(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test COM removal with pre-allocated output."""
        num_atoms = 50

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        velocities_out = wp.empty_like(velocities)

        vel_out = remove_com_motion_out(
            velocities, masses, velocities_out=velocities_out, device=device
        )

        wp.synchronize_device(device)
        assert vel_out is velocities_out

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_value(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that COM velocity becomes zero after removal."""
        num_atoms = 100
        np.random.seed(42)

        vel = np.random.randn(num_atoms, 3).astype(np_dtype)
        mass = np.ones(num_atoms, dtype=np_dtype)

        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)

        initial_com = np.sum(mass[:, np.newaxis] * vel, axis=0) / np.sum(mass)
        assert np.linalg.norm(initial_com) > 0.01

        remove_com_motion(velocities, masses, device=device)
        wp.synchronize_device(device)

        vel_result = velocities.numpy()
        final_com = np.sum(mass[:, np.newaxis] * vel_result, axis=0) / np.sum(mass)
        np.testing.assert_allclose(final_com, 0.0, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_batched(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test batched COM motion removal."""
        num_systems = 3
        atoms_per_system = 50
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        remove_com_motion(
            velocities, masses, batch_idx=batch_idx, num_systems=num_systems, device=device
        )
        wp.synchronize_device(device)

        vel_result = velocities.numpy()
        mass_result = masses.numpy()
        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            sys_vel = vel_result[start:end]
            sys_mass = mass_result[start:end]
            sys_com = np.sum(sys_mass[:, np.newaxis] * sys_vel, axis=0) / np.sum(sys_mass)
            np.testing.assert_allclose(sys_com, 0.0, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_out_batched(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating batched COM removal."""
        num_atoms = 40
        num_systems = 2

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * 20 + [1] * 20, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

        vel_out = remove_com_motion_out(
            velocities, masses, batch_idx=batch_idx, num_systems=num_systems, device=device
        )

        wp.synchronize_device(device)
        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_remove_com_motion_different_masses(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test COM removal with non-uniform masses."""
        num_atoms = 4
        vel = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np_dtype)
        mass = np.array([1.0, 2.0, 3.0, 4.0], dtype=np_dtype)

        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)

        total_mass = np.sum(mass)
        initial_momentum = np.sum(mass[:, np.newaxis] * vel, axis=0)
        initial_com = initial_momentum / total_mass
        assert np.linalg.norm(initial_com) > 0.01

        remove_com_motion(velocities, masses, device=device)
        wp.synchronize_device(device)

        vel_result = velocities.numpy()
        final_momentum = np.sum(mass[:, np.newaxis] * vel_result, axis=0)
        np.testing.assert_allclose(final_momentum, 0.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
