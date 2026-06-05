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

r"""Unit tests for the multipolar Ewald real-space Warp kernels.

Tolerances: float64 kernel-to-kernel parity is bit-for-bit modulo expression
ordering (``atol=1e-15``); closed-form checks against ``math.erfc`` use
``rel=1e-6`` because ``wp_erfc`` is an Abramowitz-Stegun approximation (~1e-7).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.electrostatics import (
    ewald_real_space_energy,
    multipole_real_space_dipole_csr_energy,
    multipole_real_space_monopole_csr_energy,
)

# =============================================================================
# Helpers
# =============================================================================


def _random_system(
    num_atoms: int,
    box_size: float,
    rng: np.random.Generator,
) -> dict:
    """Return a dict of numpy arrays describing a simple charge-neutral system."""
    positions = rng.uniform(0.0, box_size, size=(num_atoms, 3)).astype(np.float64)
    charges = rng.uniform(-1.0, 1.0, size=num_atoms).astype(np.float64)
    charges -= charges.mean()
    # Cell far bigger than the cutoff so no periodic images enter the half list.
    L = max(box_size * 3.0, 100.0)
    cell = np.array([[[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]], dtype=np.float64)
    # Half neighbor list (each pair once): atom i's neighbors are j > i.
    idx_j: list[int] = []
    neighbor_ptr: list[int] = [0]
    unit_shifts: list[list[int]] = []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            idx_j.append(j)
            unit_shifts.append([0, 0, 0])
        neighbor_ptr.append(len(idx_j))
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "idx_j": np.asarray(idx_j, dtype=np.int32),
        "neighbor_ptr": np.asarray(neighbor_ptr, dtype=np.int32),
        "unit_shifts": np.asarray(unit_shifts, dtype=np.int32),
        "num_atoms": num_atoms,
    }


def _triclinic_system(rng: np.random.Generator) -> dict:
    """6-atom system in a non-orthogonal triclinic cell (no periodic shifts used)."""
    positions = rng.uniform(0.0, 5.0, size=(6, 3)).astype(np.float64)
    charges = rng.uniform(-1.0, 1.0, size=6).astype(np.float64)
    charges -= charges.mean()
    cell = np.array(
        [
            [
                [10.0, 0.0, 0.0],
                [2.5, 9.5, 0.0],
                [1.3, 1.1, 11.0],
            ]
        ],
        dtype=np.float64,
    )
    idx_j: list[int] = []
    neighbor_ptr: list[int] = [0]
    unit_shifts: list[list[int]] = []
    for i in range(6):
        for j in range(i + 1, 6):
            idx_j.append(j)
            unit_shifts.append([0, 0, 0])
        neighbor_ptr.append(len(idx_j))
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "idx_j": np.asarray(idx_j, dtype=np.int32),
        "neighbor_ptr": np.asarray(neighbor_ptr, dtype=np.int32),
        "unit_shifts": np.asarray(unit_shifts, dtype=np.int32),
        "num_atoms": 6,
    }


def _to_warp(system: dict, device: str, wp_dtype: type) -> dict:
    """Upload a numpy-backed system dict to Warp arrays on ``device`` at ``wp_dtype``."""
    np_scalar = np.float64 if wp_dtype == wp.float64 else np.float32
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    mat_dtype = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f
    return {
        "positions": wp.from_numpy(
            system["positions"].astype(np_scalar), dtype=vec_dtype, device=device
        ),
        "charges": wp.from_numpy(
            system["charges"].astype(np_scalar), dtype=wp_dtype, device=device
        ),
        "cell": wp.from_numpy(
            system["cell"].astype(np_scalar), dtype=mat_dtype, device=device
        ),
        "idx_j": wp.from_numpy(system["idx_j"], dtype=wp.int32, device=device),
        "neighbor_ptr": wp.from_numpy(
            system["neighbor_ptr"], dtype=wp.int32, device=device
        ),
        "unit_shifts": wp.from_numpy(
            system["unit_shifts"], dtype=wp.vec3i, device=device
        ),
    }


# l=1 dipole T-tensors have a removable 1/σ singularity at σ=0, so collapse
# tests compare the l=1 (zero-dipole) and l=0 kernels at a common σ>0.
_COLLAPSE_SIGMA = 0.5


def _sigma(device: str, wp_dtype: type = wp.float64, value: float = 0.0):
    r"""GTO width ``σ`` array for the real-space launchers.

    At ``σ → 0`` the GTO charge ``T^(0)`` reduces bit-for-bit to the legacy
    monopole Ewald ``erfc(α r)/r`` form, so the default ``σ = 0`` keeps the
    monopole-collapse / closed-form assertions valid.
    """
    np_scalar = np.float64 if wp_dtype == wp.float64 else np.float32
    return wp.from_numpy(
        np.array([value], dtype=np_scalar), dtype=wp_dtype, device=device
    )


def _launch_both(
    system: dict,
    device: str,
    alpha_value: float,
    wp_dtype: type = wp.float64,
    sigma_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run both kernels on the same inputs; return per-atom energies (ref, new).

    ``ref`` is the legacy monopole ``ewald_real_space_energy`` (σ-free); ``new``
    is the GTO ``multipole_real_space_monopole_csr_energy`` at ``sigma_value``
    (default ``σ = 0`` → bit-identical to ``ref``).
    """
    inputs = _to_warp(system, device, wp_dtype)
    alpha = wp.from_numpy(
        np.array(
            [alpha_value],
            dtype=np.float64 if wp_dtype == wp.float64 else np.float32,
        ),
        dtype=wp_dtype,
        device=device,
    )
    num_atoms = system["num_atoms"]
    ref = wp.zeros(num_atoms, dtype=wp.float64, device=device)
    new = wp.zeros(num_atoms, dtype=wp.float64, device=device)

    ewald_real_space_energy(
        inputs["positions"],
        inputs["charges"],
        inputs["cell"],
        inputs["idx_j"],
        inputs["neighbor_ptr"],
        inputs["unit_shifts"],
        alpha,
        ref,
        wp_dtype=wp_dtype,
        device=device,
    )
    multipole_real_space_monopole_csr_energy(
        inputs["positions"],
        inputs["charges"],
        inputs["cell"],
        inputs["idx_j"],
        inputs["neighbor_ptr"],
        inputs["unit_shifts"],
        _sigma(device, wp_dtype, sigma_value),
        alpha,
        new,
        wp_dtype=wp_dtype,
        device=device,
    )
    return ref.numpy(), new.numpy()


# =============================================================================
# Monopole-collapse tests
# =============================================================================


class TestMonopoleCollapse:
    r"""l_max=0 branch must reproduce :func:`ewald_real_space_energy` exactly."""

    def test_two_atoms_closed_form(self, device):
        """Two opposite charges at r=3, alpha=0.3 — closed-form plus parity."""
        system = {
            "positions": np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64),
            "charges": np.array([1.0, -1.0], dtype=np.float64),
            "cell": np.array(
                [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
                dtype=np.float64,
            ),
            "idx_j": np.array([1], dtype=np.int32),
            "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
            "unit_shifts": np.array([[0, 0, 0]], dtype=np.int32),
            "num_atoms": 2,
        }
        alpha = 0.3
        ref, new = _launch_both(system, device, alpha)
        np.testing.assert_allclose(new, ref, rtol=0.0, atol=1e-15)
        expected = 0.5 * 1.0 * -1.0 * math.erfc(alpha * 3.0) / 3.0
        assert float(new.sum()) == pytest.approx(expected, abs=1e-7)

    @pytest.mark.parametrize("seed", [0, 7, 42, 123])
    @pytest.mark.parametrize("num_atoms", [3, 10, 25])
    def test_random_systems_float64(self, device, seed, num_atoms):
        """Random orthogonal systems, float64 — expect kernel-to-kernel parity at 1 ULP."""
        rng = np.random.default_rng(seed)
        system = _random_system(num_atoms, box_size=5.0, rng=rng)
        alpha = 0.4
        ref, new = _launch_both(system, device, alpha, wp_dtype=wp.float64)
        np.testing.assert_allclose(new, ref, rtol=0.0, atol=1e-15)

    def test_triclinic_cell_float64(self, device):
        """Triclinic (non-orthogonal) cell: per-edge periodic shift is zero here,
        but the cell transpose in the kernel still has off-diagonal entries. This
        catches any mismatch in how ``cell[0]`` is applied."""
        rng = np.random.default_rng(19)
        system = _triclinic_system(rng)
        alpha = 0.35
        ref, new = _launch_both(system, device, alpha, wp_dtype=wp.float64)
        np.testing.assert_allclose(new, ref, rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_random_systems_float32(self, device, seed):
        """Float32 inputs (float64 accumulators). Both kernels agree to 1 ULP float64."""
        rng = np.random.default_rng(seed)
        system = _random_system(10, box_size=5.0, rng=rng)
        alpha = 0.5
        ref, new = _launch_both(system, device, alpha, wp_dtype=wp.float32)
        np.testing.assert_allclose(new, ref, rtol=1e-13, atol=1e-15)

    def test_nonzero_unit_shifts(self, device):
        """Pairs with non-zero periodic shifts: verify that cell · shift is
        plumbed identically through both kernels."""
        # Two atoms at (0,0,0) and (1,0,0) in a 5x5x5 cell.
        # Use a periodic shift of (1, 0, 0) so the effective separation is 6.0.
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        charges = np.array([1.0, -1.0], dtype=np.float64)
        cell = np.array(
            [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]], dtype=np.float64
        )
        system = {
            "positions": positions,
            "charges": charges,
            "cell": cell,
            "idx_j": np.array([1], dtype=np.int32),
            "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
            "unit_shifts": np.array([[1, 0, 0]], dtype=np.int32),
            "num_atoms": 2,
        }
        alpha = 0.4
        ref, new = _launch_both(system, device, alpha)
        np.testing.assert_allclose(new, ref, rtol=0.0, atol=1e-15)
        # Effective separation |(1,0,0) + (5,0,0)| = 6.
        expected = 0.5 * 1.0 * -1.0 * math.erfc(alpha * 6.0) / 6.0
        assert float(new.sum()) == pytest.approx(expected, abs=1e-7)


# =============================================================================
# damped_coulomb_T0 @wp.func — sanity check via the public kernel
# =============================================================================


class TestT0InteractionTensor:
    """Indirect tests for ``damped_coulomb_T0`` via the public kernel."""

    def test_matches_analytical_erfc_over_r(self, device):
        """Varying r at fixed alpha traces out erfc(alpha*r)/r exactly."""
        distances = [0.5, 1.0, 2.5, 4.0, 10.0]
        for r in distances:
            system = {
                "positions": np.array(
                    [[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float64
                ),
                "charges": np.array([1.0, 1.0], dtype=np.float64),  # self-repulsion
                "cell": np.array(
                    [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
                    dtype=np.float64,
                ),
                "idx_j": np.array([1], dtype=np.int32),
                "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
                "unit_shifts": np.array([[0, 0, 0]], dtype=np.int32),
                "num_atoms": 2,
            }
            alpha = 0.25
            _, new = _launch_both(system, device, alpha)
            # wp_erfc's ~1e-7 absolute error dominates at large r → use abs tol.
            expected = 0.5 * math.erfc(alpha * r) / r
            assert float(new.sum()) == pytest.approx(expected, abs=1e-7)


# =============================================================================
# l_max = 1 (charges + dipoles)
# =============================================================================


def _launch_dipole(
    *,
    positions: np.ndarray,
    charges: np.ndarray,
    dipoles: np.ndarray,
    cell: np.ndarray,
    idx_j: np.ndarray,
    neighbor_ptr: np.ndarray,
    unit_shifts: np.ndarray,
    alpha_value: float,
    device: str,
    wp_dtype: type = wp.float64,
    sigma_value: float = 0.0,
) -> np.ndarray:
    """Run the l_max=1 kernel on numpy-backed inputs and return per-atom energies.

    ``σ = 0`` (default) makes the charge T^(0) collapse bit-exactly to the legacy
    monopole kernel. Dipole T-tensors have a removable ``1/σ`` singularity at
    ``σ = 0``, so dipole-physics callers pass a small ``σ > 0`` (at ``α → 0`` the
    bare-Coulomb formulas hold for any such ``σ``).
    """
    np_scalar = np.float64 if wp_dtype == wp.float64 else np.float32
    vec_dtype = wp.vec3d if wp_dtype == wp.float64 else wp.vec3f
    mat_dtype = wp.mat33d if wp_dtype == wp.float64 else wp.mat33f
    positions_wp = wp.from_numpy(
        positions.astype(np_scalar), dtype=vec_dtype, device=device
    )
    charges_wp = wp.from_numpy(charges.astype(np_scalar), dtype=wp_dtype, device=device)
    dipoles_wp = wp.from_numpy(
        dipoles.astype(np_scalar), dtype=vec_dtype, device=device
    )
    cell_wp = wp.from_numpy(cell.astype(np_scalar), dtype=mat_dtype, device=device)
    idx_j_wp = wp.from_numpy(idx_j, dtype=wp.int32, device=device)
    neighbor_ptr_wp = wp.from_numpy(neighbor_ptr, dtype=wp.int32, device=device)
    unit_shifts_wp = wp.from_numpy(unit_shifts, dtype=wp.vec3i, device=device)
    alpha_wp = wp.from_numpy(
        np.array([alpha_value], dtype=np_scalar),
        dtype=wp_dtype,
        device=device,
    )
    out = wp.zeros(positions.shape[0], dtype=wp.float64, device=device)
    multipole_real_space_dipole_csr_energy(
        positions_wp,
        charges_wp,
        dipoles_wp,
        cell_wp,
        idx_j_wp,
        neighbor_ptr_wp,
        unit_shifts_wp,
        wp.from_numpy(
            np.array([sigma_value], dtype=np_scalar), dtype=wp_dtype, device=device
        ),
        alpha_wp,
        out,
        wp_dtype=wp_dtype,
        device=device,
    )
    return out.numpy()


def _two_atom_pair_system(
    *,
    distance: float,
    charges: tuple[float, float] = (0.0, 0.0),
    dipole_i: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dipole_j: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    """Two atoms along ``+x`` with the given charges and dipoles."""
    return {
        "positions": np.array(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float64
        ),
        "charges": np.array(charges, dtype=np.float64),
        "dipoles": np.array([dipole_i, dipole_j], dtype=np.float64),
        "cell": np.array(
            [[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]],
            dtype=np.float64,
        ),
        "idx_j": np.array([1], dtype=np.int32),
        "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
        "unit_shifts": np.array([[0, 0, 0]], dtype=np.int32),
    }


class TestDipoleMonopoleCollapse:
    """l_max=1 kernel with ``dipoles = 0`` must reproduce the l_max=0 kernel
    (which in turn matches the legacy monopole Ewald kernel, per
    :class:`TestMonopoleCollapse`)."""

    @pytest.mark.parametrize("seed", [0, 7, 42, 123])
    @pytest.mark.parametrize("num_atoms", [3, 10, 25])
    def test_zero_dipoles_matches_monopole(self, device, seed, num_atoms):
        rng = np.random.default_rng(seed)
        system = _random_system(num_atoms, box_size=5.0, rng=rng)
        alpha = 0.4
        # Run l_max=0.
        inputs = _to_warp(system, device, wp.float64)
        alpha_wp = wp.from_numpy(
            np.array([alpha], dtype=np.float64), dtype=wp.float64, device=device
        )
        monopole_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)
        multipole_real_space_monopole_csr_energy(
            inputs["positions"],
            inputs["charges"],
            inputs["cell"],
            inputs["idx_j"],
            inputs["neighbor_ptr"],
            inputs["unit_shifts"],
            _sigma(device, wp.float64, _COLLAPSE_SIGMA),
            alpha_wp,
            monopole_energies,
            wp_dtype=wp.float64,
            device=device,
        )
        # Run l_max=1 with zero dipoles.
        dipole_energies = _launch_dipole(
            positions=system["positions"],
            charges=system["charges"],
            dipoles=np.zeros((num_atoms, 3), dtype=np.float64),
            cell=system["cell"],
            idx_j=system["idx_j"],
            neighbor_ptr=system["neighbor_ptr"],
            unit_shifts=system["unit_shifts"],
            alpha_value=alpha,
            device=device,
            sigma_value=_COLLAPSE_SIGMA,
        )
        np.testing.assert_allclose(
            dipole_energies, monopole_energies.numpy(), rtol=0.0, atol=1e-15
        )

    def test_triclinic_cell_zero_dipoles(self, device):
        rng = np.random.default_rng(19)
        system = _triclinic_system(rng)
        alpha = 0.35
        num_atoms = system["num_atoms"]
        inputs = _to_warp(system, device, wp.float64)
        alpha_wp = wp.from_numpy(
            np.array([alpha], dtype=np.float64), dtype=wp.float64, device=device
        )
        monopole_energies = wp.zeros(num_atoms, dtype=wp.float64, device=device)
        multipole_real_space_monopole_csr_energy(
            inputs["positions"],
            inputs["charges"],
            inputs["cell"],
            inputs["idx_j"],
            inputs["neighbor_ptr"],
            inputs["unit_shifts"],
            _sigma(device, wp.float64, _COLLAPSE_SIGMA),
            alpha_wp,
            monopole_energies,
            wp_dtype=wp.float64,
            device=device,
        )
        dipole_energies = _launch_dipole(
            positions=system["positions"],
            charges=system["charges"],
            dipoles=np.zeros((num_atoms, 3), dtype=np.float64),
            cell=system["cell"],
            idx_j=system["idx_j"],
            neighbor_ptr=system["neighbor_ptr"],
            unit_shifts=system["unit_shifts"],
            alpha_value=alpha,
            device=device,
            sigma_value=_COLLAPSE_SIGMA,
        )
        np.testing.assert_allclose(
            dipole_energies, monopole_energies.numpy(), rtol=0.0, atol=1e-15
        )


class TestDipoleChargeDipole:
    """Analytical checks for charge-dipole pair interactions in the ``α → 0`` limit."""

    SMALL_ALPHA = 1e-3
    TOL_ABS = 1e-7
    # Small σ>0 avoids the removable 1/σ dipole-tensor singularity at σ=0.
    SMALL_SIGMA = 1e-3

    def test_dipole_at_j_aligned_with_r(self, device):
        """Charge +q at i, dipole (μ, 0, 0) at j, r_ij along +x.

        Physical formula: E = μ · ∇V_q(r_j), V_q(r) = q/|r - r_i|.
        Gradient at r_j = (d, 0, 0) is (-q/d^2, 0, 0). So E = -q·μ/d^2.
        Kernel returns 0.5 * this (half-list prefactor).
        """
        d = 5.0
        q, mu = 1.2, 0.7
        system = _two_atom_pair_system(
            distance=d, charges=(q, 0.0), dipole_j=(mu, 0.0, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        expected = 0.5 * (-q * mu / d**2)
        assert float(energies.sum()) == pytest.approx(expected, abs=self.TOL_ABS)

    def test_charge_at_j_dipole_at_i(self, device):
        """Dipole (μ, 0, 0) at i, charge +q at j, r_ij along +x.

        Standard formula: E = +q·μ/d^2 (positive side of dipole faces charge).
        """
        d = 5.0
        q, mu = 1.2, 0.7
        system = _two_atom_pair_system(
            distance=d, charges=(0.0, q), dipole_i=(mu, 0.0, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        expected = 0.5 * (q * mu / d**2)
        assert float(energies.sum()) == pytest.approx(expected, abs=self.TOL_ABS)

    def test_dipole_perpendicular_to_r(self, device):
        """Charge at i, dipole at j perpendicular to r_ij — energy should be 0."""
        d = 4.0
        q, mu = 0.9, 0.5
        system = _two_atom_pair_system(
            distance=d, charges=(q, 0.0), dipole_j=(0.0, mu, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        assert float(energies.sum()) == pytest.approx(0.0, abs=self.TOL_ABS)


class TestDipoleDipoleDipole:
    """Analytical checks for dipole-dipole pair interactions.

    Standard formula: ``E_dd = (μ_i · μ_j)/r^3 - 3(μ_i · r̂)(μ_j · r̂)/r^3``.
    With the kernel's half-list 0.5 prefactor, expected = 0.5 * E_dd.
    """

    SMALL_ALPHA = 1e-3
    TOL_ABS = 1e-7
    # Small σ>0 avoids the removable 1/σ dipole-tensor singularity at σ=0.
    SMALL_SIGMA = 1e-3

    def test_parallel_along_r(self, device):
        """Both dipoles (μ, 0, 0), r along +x → attractive, -2μ²/d³."""
        d = 5.0
        mu = 1.3
        system = _two_atom_pair_system(
            distance=d, dipole_i=(mu, 0.0, 0.0), dipole_j=(mu, 0.0, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        expected = 0.5 * (-2.0 * mu**2 / d**3)
        assert float(energies.sum()) == pytest.approx(expected, abs=self.TOL_ABS)

    def test_antiparallel_along_r(self, device):
        """Opposing dipoles (μ, 0, 0) and (-μ, 0, 0), r along +x → repulsive, +2μ²/d³."""
        d = 5.0
        mu = 1.3
        system = _two_atom_pair_system(
            distance=d, dipole_i=(mu, 0.0, 0.0), dipole_j=(-mu, 0.0, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        expected = 0.5 * (2.0 * mu**2 / d**3)
        assert float(energies.sum()) == pytest.approx(expected, abs=self.TOL_ABS)

    def test_perpendicular_dipoles(self, device):
        """Dipoles along different transverse axes — zero energy."""
        d = 4.0
        mu = 1.1
        system = _two_atom_pair_system(
            distance=d, dipole_i=(0.0, mu, 0.0), dipole_j=(0.0, 0.0, mu)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        assert float(energies.sum()) == pytest.approx(0.0, abs=self.TOL_ABS)

    def test_parallel_perpendicular_to_r(self, device):
        """Both dipoles (0, μ, 0), r along +x.

        E_dd = μ² /r³ - 0 = +μ²/r³ (repulsive for two side-by-side parallel dipoles).
        """
        d = 5.0
        mu = 1.0
        system = _two_atom_pair_system(
            distance=d, dipole_i=(0.0, mu, 0.0), dipole_j=(0.0, mu, 0.0)
        )
        energies = _launch_dipole(
            **system,
            alpha_value=self.SMALL_ALPHA,
            device=device,
            sigma_value=self.SMALL_SIGMA,
        )
        expected = 0.5 * (mu**2 / d**3)
        assert float(energies.sum()) == pytest.approx(expected, abs=self.TOL_ABS)


class TestDipolePeriodicShifts:
    """Non-zero image shifts with dipoles."""

    def test_shift_preserves_collapse(self, device):
        """With shift (1,0,0) and a 5x5x5 cell → effective separation 6.
        Zero dipoles → matches l_max=0 kernel."""
        system = {
            "positions": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
            "charges": np.array([1.0, -1.0], dtype=np.float64),
            "dipoles": np.zeros((2, 3), dtype=np.float64),
            "cell": np.array(
                [[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]],
                dtype=np.float64,
            ),
            "idx_j": np.array([1], dtype=np.int32),
            "neighbor_ptr": np.array([0, 1, 1], dtype=np.int32),
            "unit_shifts": np.array([[1, 0, 0]], dtype=np.int32),
        }
        alpha = 0.4
        # Zero dipoles → the l=1 kernel reproduces the l=0 kernel at a common σ>0.
        dipole = _launch_dipole(
            **system, alpha_value=alpha, device=device, sigma_value=_COLLAPSE_SIGMA
        )

        ref_system = {k: v for k, v in system.items() if k != "dipoles"}
        ref_system["num_atoms"] = 2
        _, monopole = _launch_both(
            ref_system, device, alpha, sigma_value=_COLLAPSE_SIGMA
        )
        np.testing.assert_allclose(dipole, monopole, rtol=0.0, atol=1e-15)
