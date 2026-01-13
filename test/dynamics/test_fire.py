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
Unit tests for FIRE optimizer.

Tests cover:
- Basic API functionality (single and batched)
- Convergence on harmonic potential
- Energy decrease during optimization
- Force convergence
- Morse potential dimer optimization
- Diagnostic computation
- Velocity mixing
- Float32 and float64 support
"""

import numpy as np
import pytest
import warp as wp

from nvalchemiops.dynamics.optimizers import (
    fire_compute_diagnostics,
    fire_md_step,
    fire_md_step_out,
    fire_reset_velocities,
    fire_reset_velocities_out,
    fire_velocity_mix,
    fire_velocity_mix_out,
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
# Helper Functions
# ==============================================================================


@wp.kernel
def _compute_harmonic_forces_kernel_f32(
    positions: wp.array(dtype=wp.vec3f),
    forces: wp.array(dtype=wp.vec3f),
    spring_k: wp.float32,
):
    """Compute harmonic forces F = -k * r (float32)."""
    i = wp.tid()
    pos = positions[i]
    forces[i] = wp.vec3f(-spring_k * pos[0], -spring_k * pos[1], -spring_k * pos[2])


@wp.kernel
def _compute_harmonic_forces_kernel_f64(
    positions: wp.array(dtype=wp.vec3d),
    forces: wp.array(dtype=wp.vec3d),
    spring_k: wp.float64,
):
    """Compute harmonic forces F = -k * r (float64)."""
    i = wp.tid()
    pos = positions[i]
    forces[i] = wp.vec3d(-spring_k * pos[0], -spring_k * pos[1], -spring_k * pos[2])


def compute_harmonic_forces(positions: wp.array, forces: wp.array, k: float):
    """Compute harmonic forces F = -k * r."""
    if positions.dtype == wp.vec3f:
        wp.launch(
            _compute_harmonic_forces_kernel_f32,
            dim=positions.shape[0],
            inputs=[positions, forces, wp.float32(k)],
            device=positions.device,
        )
    else:
        wp.launch(
            _compute_harmonic_forces_kernel_f64,
            dim=positions.shape[0],
            inputs=[positions, forces, wp.float64(k)],
            device=positions.device,
        )


def compute_harmonic_energy(positions: np.ndarray, k: float) -> float:
    """Compute harmonic potential energy E = 0.5 * k * |r|^2."""
    r_sq = np.sum(positions**2)
    return 0.5 * k * r_sq


def compute_max_force(forces: np.ndarray) -> float:
    """Compute maximum force magnitude."""
    force_magnitudes = np.linalg.norm(forces, axis=1)
    return np.max(force_magnitudes)


def compute_morse_forces_and_energy(
    positions: np.ndarray, D_e: float, a: float, r_e: float
) -> tuple[np.ndarray, float]:
    """Compute Morse potential forces and energy for a dimer."""
    forces = np.zeros_like(positions)
    r_vec = positions[1] - positions[0]
    r = np.linalg.norm(r_vec)

    if r > 1e-10:
        r_hat = r_vec / r
        exp_term = np.exp(-a * (r - r_e))
        dVdr = 2 * D_e * a * (1 - exp_term) * exp_term
        forces[1] = -dVdr * r_hat
        forces[0] = dVdr * r_hat
        energy = D_e * (1 - exp_term) ** 2
    else:
        energy = 0.0

    return forces, float(energy)


def run_fire_optimization(
    positions: wp.array,
    velocities: wp.array,
    forces: wp.array,
    masses: wp.array,
    dt: wp.array,
    alpha: wp.array,
    n_positive: wp.array,
    compute_forces_fn,
    max_steps: int = 1000,
    force_tol: float = 1e-5,
    alpha_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    f_alpha: float = 0.99,
    dt_max: float = 0.1,
    device: str = None,
    dtype_scalar=wp.float32,
    np_dtype=np.float32,
    batch_idx: wp.array = None,
    num_systems: int = 1,
):
    """Run FIRE optimization loop."""
    num_atoms = positions.shape[0]

    compute_forces_fn()

    for step in range(max_steps):
        power, force_norm_sq, velocity_norm_sq = fire_compute_diagnostics(
            velocities,
            forces,
            batch_idx=batch_idx,
            num_systems=num_systems,
            device=device,
        )

        wp.synchronize_device(device)
        power_val = power.numpy()[0]
        force_norm_sq_val = force_norm_sq.numpy()[0]
        velocity_norm_sq_val = velocity_norm_sq.numpy()[0]
        max_force = np.sqrt(force_norm_sq_val / num_atoms)

        if max_force < force_tol:
            return step, max_force

        # Compute norms for velocity mixing
        force_norm = wp.array(
            [np.sqrt(force_norm_sq_val)], dtype=dtype_scalar, device=device
        )
        velocity_norm = wp.array(
            [np.sqrt(velocity_norm_sq_val)], dtype=dtype_scalar, device=device
        )

        fire_velocity_mix(
            velocities,
            forces,
            alpha,
            force_norm,
            velocity_norm,
            batch_idx=batch_idx,
            device=device,
        )

        if power_val > 0:
            wp.synchronize_device(device)
            current_n_positive = n_positive.numpy()[0]
            current_dt = dt.numpy()[0]
            current_alpha = alpha.numpy()[0]

            n_positive_new = current_n_positive + 1

            if n_positive_new > n_min:
                new_dt = min(current_dt * f_inc, dt_max)
                new_alpha = current_alpha * f_alpha
                dt_np = np.array([new_dt], dtype=np_dtype)
                alpha_np = np.array([new_alpha], dtype=np_dtype)
                dt = wp.array(dt_np, dtype=dtype_scalar, device=device)
                alpha = wp.array(alpha_np, dtype=dtype_scalar, device=device)

            n_positive = wp.array(
                np.array([n_positive_new], dtype=np.int32),
                dtype=wp.int32,
                device=device,
            )
        else:
            fire_reset_velocities(velocities, batch_idx=batch_idx, device=device)

            wp.synchronize_device(device)
            current_dt = dt.numpy()[0]
            new_dt = current_dt * f_dec

            dt = wp.array(
                np.array([new_dt], dtype=np_dtype), dtype=dtype_scalar, device=device
            )
            alpha = wp.array(
                np.array([alpha_start], dtype=np_dtype),
                dtype=dtype_scalar,
                device=device,
            )
            n_positive = wp.array(
                np.array([0], dtype=np.int32), dtype=wp.int32, device=device
            )

        fire_md_step(
            positions,
            velocities,
            forces,
            masses,
            dt,
            batch_idx=batch_idx,
            device=device,
        )
        compute_forces_fn()

    wp.synchronize_device(device)
    _, force_norm_sq, _ = fire_compute_diagnostics(
        velocities, forces, batch_idx=batch_idx, num_systems=num_systems, device=device
    )
    wp.synchronize_device(device)
    force_norm_sq_val = force_norm_sq.numpy()[0]
    max_force = np.sqrt(force_norm_sq_val / num_atoms)

    return max_steps, max_force


# ==============================================================================
# Single System API Tests
# ==============================================================================


class TestFIREAPI:
    """Test single-system API functionality including non-mutating variants."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_diagnostics_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that fire_compute_diagnostics executes without error."""
        num_atoms = 10
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )

        power, force_norm_sq, velocity_norm_sq = fire_compute_diagnostics(
            velocities, forces, device=device
        )
        wp.synchronize_device(device)

        assert power is not None
        assert force_norm_sq.numpy()[0] > 0.0
        assert velocity_norm_sq.numpy()[0] > 0.0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_compute_diagnostics_with_preallocated_arrays(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test fire_compute_diagnostics with pre-allocated output arrays."""
        num_atoms = 20

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )

        # Pre-allocate diagnostic arrays
        power = wp.zeros(1, dtype=wp.float64, device=device)
        force_norm_sq = wp.zeros(1, dtype=wp.float64, device=device)
        velocity_norm_sq = wp.zeros(1, dtype=wp.float64, device=device)

        p, fn, vn = fire_compute_diagnostics(
            velocities,
            forces,
            num_systems=1,
            power=power,
            force_norm_sq=force_norm_sq,
            velocity_norm_sq=velocity_norm_sq,
            device=device,
        )

        wp.synchronize_device(device)
        assert p is power
        assert fn is force_norm_sq
        assert vn is velocity_norm_sq

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_mix_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that fire_velocity_mix executes without error."""
        num_atoms = 10
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        alpha = wp.array([0.1], dtype=dtype_scalar, device=device)
        force_norm = wp.array([1.0], dtype=dtype_scalar, device=device)
        velocity_norm = wp.array([1.0], dtype=dtype_scalar, device=device)

        fire_velocity_mix(velocities, forces, alpha, force_norm, velocity_norm)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_mix_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating velocity mix preserves input."""
        num_atoms = 20

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        alpha = wp.array([0.1], dtype=dtype_scalar, device=device)
        force_norm = wp.array([1.0], dtype=dtype_scalar, device=device)
        velocity_norm = wp.array([1.0], dtype=dtype_scalar, device=device)

        vel_orig = velocities.numpy().copy()

        vel_out = fire_velocity_mix_out(
            velocities, forces, alpha, force_norm, velocity_norm, device=device
        )

        np.testing.assert_array_equal(velocities.numpy(), vel_orig)
        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_md_step_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that fire_md_step executes without error."""
        num_atoms = 10
        np.random.seed(42)

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.01], dtype=dtype_scalar, device=device)

        fire_md_step(positions, velocities, forces, masses, dt)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_md_step_device_inference(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that device is correctly inferred for fire_md_step."""
        num_atoms = 20

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.001], dtype=dtype_scalar, device=device)

        # Call without explicit device (should infer from positions)
        fire_md_step(positions, velocities, forces, masses, dt)

        wp.synchronize_device(device)
        assert positions.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_md_step_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that non-mutating fire_md_step_out preserves input."""
        num_atoms = 10
        np.random.seed(42)

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.01], dtype=dtype_scalar, device=device)

        pos_orig = positions.numpy().copy()
        vel_orig = velocities.numpy().copy()

        pos_out, vel_out = fire_md_step_out(positions, velocities, forces, masses, dt)
        wp.synchronize_device(device)

        np.testing.assert_array_equal(positions.numpy(), pos_orig)
        np.testing.assert_array_equal(velocities.numpy(), vel_orig)
        assert not np.allclose(pos_out.numpy(), pos_orig)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_reset_velocities_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that fire_reset_velocities executes and zeros velocities."""
        num_atoms = 10
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )

        fire_reset_velocities(velocities)
        wp.synchronize_device(device)

        vel_np = velocities.numpy()
        assert np.allclose(vel_np, 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_reset_velocities_out_preserves_input(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating velocity reset preserves input."""
        num_atoms = 20

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )

        vel_orig = velocities.numpy().copy()

        vel_out = fire_reset_velocities_out(velocities, device=device)

        np.testing.assert_array_equal(velocities.numpy(), vel_orig)
        assert np.allclose(vel_out.numpy(), 0.0)


# ==============================================================================
# Batched API Tests
# ==============================================================================


class TestFIREBatched:
    """Test batched API functionality including non-mutating variants."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_compute_diagnostics_runs(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that batched fire_compute_diagnostics executes correctly."""
        num_systems = 3
        atoms_per_system = 5
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )

        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        power, force_norm_sq, velocity_norm_sq = fire_compute_diagnostics(
            velocities, forces, batch_idx=batch_idx, num_systems=num_systems
        )
        wp.synchronize_device(device)

        force_norm_sq_np = force_norm_sq.numpy()
        velocity_norm_sq_np = velocity_norm_sq.numpy()

        for sys_id in range(num_systems):
            assert force_norm_sq_np[sys_id] > 0.0
            assert velocity_norm_sq_np[sys_id] > 0.0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_velocity_mix_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that batched fire_velocity_mix executes correctly."""
        num_systems = 3
        atoms_per_system = 5
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        alpha = wp.array([0.1, 0.2, 0.3], dtype=dtype_scalar, device=device)
        force_norm = wp.array([1.0, 1.0, 1.0], dtype=dtype_scalar, device=device)
        velocity_norm = wp.array([1.0, 1.0, 1.0], dtype=dtype_scalar, device=device)

        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        fire_velocity_mix(
            velocities, forces, alpha, force_norm, velocity_norm, batch_idx=batch_idx
        )
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_velocity_mix_out(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating velocity mix for batched systems."""
        num_atoms = 40

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * 20 + [1] * 20, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        alpha = wp.array([0.1, 0.1], dtype=dtype_scalar, device=device)
        force_norm = wp.array([1.0, 1.0], dtype=dtype_scalar, device=device)
        velocity_norm = wp.array([1.0, 1.0], dtype=dtype_scalar, device=device)

        vel_out = fire_velocity_mix_out(
            velocities,
            forces,
            alpha,
            force_norm,
            velocity_norm,
            batch_idx=batch_idx,
            device=device,
        )

        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_md_step_runs(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that batched fire_md_step executes correctly."""
        num_systems = 3
        atoms_per_system = 5
        total_atoms = num_systems * atoms_per_system
        np.random.seed(42)

        positions = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
            np.random.randn(total_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype),
            dtype=dtype_scalar,
            device=device,
        )
        dt = wp.array([0.01, 0.02, 0.03], dtype=dtype_scalar, device=device)

        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        fire_md_step(positions, velocities, forces, masses, dt, batch_idx=batch_idx)
        wp.synchronize_device(device)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_md_step_out(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test non-mutating MD step for batched systems."""
        num_atoms = 40

        positions = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype) * 0.1,
            dtype=dtype_vec,
            device=device,
        )
        forces = wp.array(
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
        dt = wp.array([0.001, 0.001], dtype=dtype_scalar, device=device)

        pos_out, vel_out = fire_md_step_out(
            positions,
            velocities,
            forces,
            masses,
            dt,
            batch_idx=batch_idx,
            device=device,
        )

        assert pos_out.shape[0] == num_atoms
        assert vel_out.shape[0] == num_atoms

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_uses_per_system_dt(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that batched version uses per-system timesteps correctly."""
        pos = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        vel = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np_dtype)
        force = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np_dtype)
        mass = np.array([1.0, 1.0], dtype=np_dtype)
        dt_vals = [0.01, 0.02]

        positions = wp.array(pos, dtype=dtype_vec, device=device)
        velocities = wp.array(vel, dtype=dtype_vec, device=device)
        forces = wp.array(force, dtype=dtype_vec, device=device)
        masses = wp.array(mass, dtype=dtype_scalar, device=device)
        dt = wp.array(dt_vals, dtype=dtype_scalar, device=device)
        batch_idx = wp.array([0, 1], dtype=wp.int32, device=device)

        fire_md_step(positions, velocities, forces, masses, dt, batch_idx=batch_idx)
        wp.synchronize_device(device)

        result_pos = positions.numpy()
        result_vel = velocities.numpy()

        assert result_vel[1, 0] > result_vel[0, 0], (
            "System with larger dt should have higher velocity"
        )

        displacement_0 = result_pos[0, 0] - 1.0
        displacement_1 = result_pos[1, 0] - 1.0
        assert displacement_1 > displacement_0

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_reset_velocities(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test velocity reset for batched systems zeros all velocities."""
        num_atoms = 40

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * 20 + [1] * 20, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

        fire_reset_velocities(
            velocities, batch_idx=batch_idx, num_systems=2, device=device
        )

        vel_np = velocities.numpy()
        assert np.allclose(vel_np, 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_reset_velocities_out(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test non-mutating velocity reset for batched systems."""
        num_atoms = 40

        velocities = wp.array(
            np.random.randn(num_atoms, 3).astype(np_dtype),
            dtype=dtype_vec,
            device=device,
        )
        batch_idx = wp.array(
            np.array([0] * 20 + [1] * 20, dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )

        vel_out = fire_reset_velocities_out(
            velocities, batch_idx=batch_idx, num_systems=2, device=device
        )

        assert np.allclose(vel_out.numpy(), 0.0)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_batched_independent_convergence(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that batched systems converge independently."""
        num_systems = 2
        atoms_per_system = 5
        total_atoms = num_systems * atoms_per_system
        spring_k = 1.0

        np.random.seed(42)

        pos_sys0 = np.random.randn(atoms_per_system, 3).astype(np_dtype) * 0.5
        pos_sys1 = np.random.randn(atoms_per_system, 3).astype(np_dtype) * 3.0
        initial_pos = np.vstack([pos_sys0, pos_sys1])

        positions = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        forces = wp.zeros(total_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(total_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        dt = wp.array([0.01, 0.01], dtype=dtype_scalar, device=device)
        alpha = wp.array([0.1, 0.1], dtype=dtype_scalar, device=device)

        batch_idx = wp.array(
            np.repeat(np.arange(num_systems), atoms_per_system).astype(np.int32),
            dtype=wp.int32,
            device=device,
        )

        def compute_forces():
            compute_harmonic_forces(positions, forces, spring_k)

        compute_forces()

        for step in range(500):
            power, force_norm_sq, velocity_norm_sq = fire_compute_diagnostics(
                velocities, forces, batch_idx=batch_idx, num_systems=num_systems
            )
            wp.synchronize_device(device)

            force_norm_np = np.sqrt(force_norm_sq.numpy())
            velocity_norm_np = np.sqrt(velocity_norm_sq.numpy())
            force_norm = wp.array(
                force_norm_np.astype(np_dtype), dtype=dtype_scalar, device=device
            )
            velocity_norm = wp.array(
                velocity_norm_np.astype(np_dtype), dtype=dtype_scalar, device=device
            )

            fire_velocity_mix(
                velocities,
                forces,
                alpha,
                force_norm,
                velocity_norm,
                batch_idx=batch_idx,
            )
            fire_md_step(positions, velocities, forces, masses, dt, batch_idx=batch_idx)
            compute_forces()

        wp.synchronize_device(device)
        final_pos = positions.numpy()

        for sys_id in range(num_systems):
            start = sys_id * atoms_per_system
            end = (sys_id + 1) * atoms_per_system
            max_displacement = np.max(np.abs(final_pos[start:end]))
            assert max_displacement < 0.5, (
                f"System {sys_id} should converge: max displacement = {max_displacement}"
            )


# ==============================================================================
# Physics and Correctness Tests
# ==============================================================================


class TestFIREPhysics:
    """Test physics correctness: convergence, energy decrease, diagnostics, velocity mixing."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_harmonic_converges_to_origin(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that FIRE converges to minimum of harmonic potential (origin)."""
        num_atoms = 10
        spring_k = 1.0
        force_tol = 1e-4

        np.random.seed(42)

        initial_pos = np.random.randn(num_atoms, 3).astype(np_dtype) * 2.0

        initial_max_pos = np.max(np.abs(initial_pos))
        assert initial_max_pos > 0.5, "Initial positions should be far from origin"

        positions = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        dt = wp.array([0.01], dtype=dtype_scalar, device=device)
        alpha = wp.array([0.1], dtype=dtype_scalar, device=device)
        n_positive = wp.array([0], dtype=wp.int32, device=device)

        def compute_forces():
            compute_harmonic_forces(positions, forces, spring_k)

        steps, final_force = run_fire_optimization(
            positions,
            velocities,
            forces,
            masses,
            dt,
            alpha,
            n_positive,
            compute_forces,
            max_steps=2000,
            force_tol=force_tol,
            device=device,
            dtype_scalar=dtype_scalar,
            np_dtype=np_dtype,
        )

        wp.synchronize_device(device)
        final_pos = positions.numpy()

        max_displacement = np.max(np.abs(final_pos))
        assert max_displacement < 0.01, (
            f"Should converge to origin, max displacement: {max_displacement}"
        )
        assert steps < 2000, f"Should converge within max steps, took {steps}"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_energy_decreases_during_optimization(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that potential energy decreases during FIRE optimization."""
        num_atoms = 20
        spring_k = 1.0

        np.random.seed(42)

        initial_pos = np.random.randn(num_atoms, 3).astype(np_dtype) * 2.0

        positions = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        dt = wp.array([0.01], dtype=dtype_scalar, device=device)
        alpha = wp.array([0.1], dtype=dtype_scalar, device=device)
        n_positive = wp.array([0], dtype=wp.int32, device=device)

        initial_energy = compute_harmonic_energy(initial_pos, spring_k)
        assert initial_energy > 1.0, (
            f"Initial energy should be significant: {initial_energy}"
        )

        def compute_forces():
            compute_harmonic_forces(positions, forces, spring_k)

        run_fire_optimization(
            positions,
            velocities,
            forces,
            masses,
            dt,
            alpha,
            n_positive,
            compute_forces,
            max_steps=500,
            force_tol=1e-4,
            device=device,
            dtype_scalar=dtype_scalar,
            np_dtype=np_dtype,
        )

        wp.synchronize_device(device)
        final_pos = positions.numpy()
        final_energy = compute_harmonic_energy(final_pos, spring_k)

        assert final_energy < initial_energy, (
            f"Energy should decrease: {initial_energy} -> {final_energy}"
        )

        energy_reduction = (initial_energy - final_energy) / initial_energy
        assert energy_reduction > 0.99, (
            f"Energy should decrease substantially: {energy_reduction}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_force_convergence(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that maximum force converges to near zero."""
        num_atoms = 20
        spring_k = 1.0
        force_tol = 1e-4

        np.random.seed(42)

        initial_pos = np.random.randn(num_atoms, 3).astype(np_dtype) * 2.0

        positions = wp.array(initial_pos.copy(), dtype=dtype_vec, device=device)
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        forces = wp.zeros(num_atoms, dtype=dtype_vec, device=device)
        masses = wp.array(
            np.ones(num_atoms, dtype=np_dtype), dtype=dtype_scalar, device=device
        )

        dt = wp.array([0.01], dtype=dtype_scalar, device=device)
        alpha = wp.array([0.1], dtype=dtype_scalar, device=device)
        n_positive = wp.array([0], dtype=wp.int32, device=device)

        compute_harmonic_forces(positions, forces, spring_k)
        wp.synchronize_device(device)
        initial_forces = forces.numpy()
        initial_max_force = compute_max_force(initial_forces)

        assert initial_max_force > 0.1, (
            f"Initial forces should be significant: {initial_max_force}"
        )

        def compute_forces():
            compute_harmonic_forces(positions, forces, spring_k)

        steps, final_max_force = run_fire_optimization(
            positions,
            velocities,
            forces,
            masses,
            dt,
            alpha,
            n_positive,
            compute_forces,
            max_steps=2000,
            force_tol=force_tol,
            device=device,
            dtype_scalar=dtype_scalar,
            np_dtype=np_dtype,
        )

        assert final_max_force < force_tol * 10, (
            f"Final max force should be small: {final_max_force}"
        )

        force_reduction = initial_max_force / max(final_max_force, 1e-10)
        assert force_reduction > 1000, (
            f"Forces should decrease by > 1000x: {force_reduction}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_morse_dimer_optimization(self, device):
        """Test FIRE optimization on Morse potential dimer."""
        D_e = 1.0
        a = 2.0
        r_e = 1.5

        num_atoms = 2
        force_tol = 1e-4

        initial_r = 2.5
        initial_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [initial_r, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        masses_np = np.array([1.0, 1.0], dtype=np.float32)

        positions = wp.array(initial_pos.copy(), dtype=wp.vec3f, device=device)
        velocities = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
        forces = wp.zeros(num_atoms, dtype=wp.vec3f, device=device)
        masses = wp.array(masses_np, dtype=wp.float32, device=device)

        dt = wp.array([0.01], dtype=wp.float32, device=device)
        alpha = wp.array([0.1], dtype=wp.float32, device=device)
        n_positive = wp.array([0], dtype=wp.int32, device=device)

        def compute_morse():
            wp.synchronize_device(device)
            pos_np = positions.numpy()
            forces_np, _ = compute_morse_forces_and_energy(pos_np, D_e, a, r_e)
            forces_wp = wp.array(
                forces_np.astype(np.float32), dtype=wp.vec3f, device=device
            )
            wp.copy(forces, forces_wp)

        compute_morse()
        wp.synchronize_device(device)
        initial_forces = forces.numpy()
        initial_max_force = compute_max_force(initial_forces)

        assert initial_max_force > 0.1, (
            f"Initial forces should be significant: {initial_max_force}"
        )

        steps, final_max_force = run_fire_optimization(
            positions,
            velocities,
            forces,
            masses,
            dt,
            alpha,
            n_positive,
            compute_morse,
            max_steps=2000,
            force_tol=force_tol,
            device=device,
        )

        wp.synchronize_device(device)
        final_pos = positions.numpy()
        final_r = np.linalg.norm(final_pos[1] - final_pos[0])

        np.testing.assert_allclose(
            final_r,
            r_e,
            rtol=0.01,
            err_msg=f"Bond length {final_r} should converge to r_e={r_e}",
        )

        assert final_max_force < force_tol * 10, (
            f"Final max force should be small: {final_max_force}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_power_calculation_positive(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test power calculation when v·F > 0 (moving downhill)."""
        velocities = wp.array([[1.0, 0.0, 0.0]], dtype=dtype_vec, device=device)
        forces = wp.array([[2.0, 0.0, 0.0]], dtype=dtype_vec, device=device)

        power, force_norm_sq, velocity_norm_sq = fire_compute_diagnostics(
            velocities, forces, device=device
        )
        wp.synchronize_device(device)

        power_val = power.numpy()[0]
        assert power_val > 0, f"Power should be positive: {power_val}"
        np.testing.assert_allclose(power_val, 2.0, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_power_calculation_negative(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test power calculation when v·F < 0 (moving uphill)."""
        velocities = wp.array([[1.0, 0.0, 0.0]], dtype=dtype_vec, device=device)
        forces = wp.array([[-2.0, 0.0, 0.0]], dtype=dtype_vec, device=device)

        power, force_norm_sq, velocity_norm_sq = fire_compute_diagnostics(
            velocities, forces, device=device
        )
        wp.synchronize_device(device)

        power_val = power.numpy()[0]
        assert power_val < 0, f"Power should be negative: {power_val}"
        np.testing.assert_allclose(power_val, -2.0, rtol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_force_norm_calculation(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test force norm squared calculation."""
        num_atoms = 2

        forces = wp.array(
            [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=dtype_vec, device=device
        )
        velocities = wp.zeros(num_atoms, dtype=dtype_vec, device=device)

        power, force_norm_sq, velocity_norm_sq = fire_compute_diagnostics(
            velocities, forces, device=device
        )
        wp.synchronize_device(device)

        expected_force_norm_sq = 25.0
        np.testing.assert_allclose(
            force_norm_sq.numpy()[0], expected_force_norm_sq, rtol=1e-5
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_velocity_aligns_with_force(
        self, device, dtype_vec, dtype_scalar, np_dtype
    ):
        """Test that velocity_mix aligns velocity toward force direction."""
        velocities = wp.array([[1.0, 0.0, 0.0]], dtype=dtype_vec, device=device)
        forces = wp.array([[0.0, 1.0, 0.0]], dtype=dtype_vec, device=device)
        alpha = wp.array([0.5], dtype=dtype_scalar, device=device)
        force_norm = wp.array([1.0], dtype=dtype_scalar, device=device)
        velocity_norm = wp.array([1.0], dtype=dtype_scalar, device=device)

        fire_velocity_mix(velocities, forces, alpha, force_norm, velocity_norm)
        wp.synchronize_device(device)

        result = velocities.numpy()[0]

        assert result[1] > 0, "Velocity should have component in force direction"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("dtype_vec,dtype_scalar,np_dtype", DTYPE_CONFIGS)
    def test_alpha_zero_no_mixing(self, device, dtype_vec, dtype_scalar, np_dtype):
        """Test that alpha=0 results in no mixing."""
        initial_vel = np.array([[1.0, 0.0, 0.0]], dtype=np_dtype)
        velocities = wp.array(initial_vel.copy(), dtype=dtype_vec, device=device)
        forces = wp.array([[0.0, 1.0, 0.0]], dtype=dtype_vec, device=device)
        alpha = wp.array([0.0], dtype=dtype_scalar, device=device)
        force_norm = wp.array([1.0], dtype=dtype_scalar, device=device)
        velocity_norm = wp.array([1.0], dtype=dtype_scalar, device=device)

        fire_velocity_mix(velocities, forces, alpha, force_norm, velocity_norm)
        wp.synchronize_device(device)

        result = velocities.numpy()
        np.testing.assert_allclose(result, initial_vel, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
