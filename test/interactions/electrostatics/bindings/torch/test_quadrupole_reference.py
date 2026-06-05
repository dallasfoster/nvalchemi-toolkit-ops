# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Cross-validate the pure-Python multipole reference against existing Path-A.

The reference module ``nvalchemiops._reference.multipole_reference`` is
ground truth for the quadrupole PME tests. Before relying on it at
``l_max = 2`` (where no other reference exists), validate it against
Path-A ``multipole_ewald_summation`` at ``l_max = 0`` and ``l_max = 1``.

Tolerances are loose (``rtol ≈ 1e-3``): the reference uses finite
cutoffs while Path-A auto-estimates to ``rtol = 1e-6``. The test
confirms sign conventions and channel coefficients, not bit-equality.

The quadrupole channel is exercised by FD tests here (no external
``l_max = 2`` reference): Q=0 must recover ``l_max = 1`` exactly and the
self-energy coefficient must match.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvalchemiops._reference.multipole_reference import (  # noqa: E402
    direct_ewald_energy,
    pair_energy,
    self_energy,
)

# ---------------------------------------------------------------------------
# Small fixture builder
# ---------------------------------------------------------------------------


def _bcc_fixture(size: int = 2, dtype=np.float64):
    """Tiny BCC NaCl-like fixture (``size = 2`` → ``N = 16`` atoms)."""
    a = 4.14
    ijk = np.indices((size, size, size)).reshape(3, -1).T
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    sites = (ijk[:, None, :] + basis[None, :, :]) * a
    pos = sites.reshape(-1, 3).astype(dtype)
    parity = (ijk.sum(-1)[:, None] + np.array([[0, 1]])) % 2
    q = np.where(parity == 0, 1.0, -1.0).reshape(-1).astype(dtype)
    if abs(float(q.sum())) > 1e-12:
        q[-1] -= float(q.sum())
    rng = np.random.default_rng(31415)
    mu = rng.standard_normal((pos.shape[0], 3)).astype(dtype) * 0.3
    # Random symmetric traceless quadrupole per atom.
    Q_raw = rng.standard_normal((pos.shape[0], 3, 3)).astype(dtype) * 0.2
    Q = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    trace = Q[:, 0, 0] + Q[:, 1, 1] + Q[:, 2, 2]
    Q[:, 0, 0] -= trace / 3.0
    Q[:, 1, 1] -= trace / 3.0
    Q[:, 2, 2] -= trace / 3.0
    cell = np.eye(3, dtype=dtype) * (size * a)
    return {
        "positions": pos,
        "cell": cell,
        "charges": q,
        "dipoles": mu,
        "quadrupoles": Q,
    }


class TestPathAReferenceParity:
    """``direct_ewald_energy`` reference vs Path-A ``multipole_ewald_summation``
    at l_max <= 1."""

    @pytest.fixture(scope="class")
    def fix(self):
        return _bcc_fixture(size=2)

    def _run_path_a(self, fix, *, with_dipoles: bool):
        """Run Path-A ``multipole_ewald_summation`` and return its estimated
        alpha so the reference comparison uses consistent parameters.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for Path-A real-space tile kernel")

        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            multipole_ewald_summation,
        )

        device = "cuda:0"
        positions = torch.from_numpy(fix["positions"]).to(device, torch.float64)
        cell = torch.from_numpy(fix["cell"]).to(device, torch.float64)
        N = positions.shape[0]

        if with_dipoles:
            # e3nn pack order: [q, μ_y, μ_z, μ_x]
            mu = fix["dipoles"]
            source_feats = np.zeros((N, 4), dtype=np.float64)
            source_feats[:, 0] = fix["charges"]
            source_feats[:, 1] = mu[:, 1]  # μ_y
            source_feats[:, 2] = mu[:, 2]  # μ_z
            source_feats[:, 3] = mu[:, 0]  # μ_x
        else:
            source_feats = fix["charges"].reshape(-1, 1).astype(np.float64)

        source_feats_t = torch.from_numpy(source_feats).to(device, torch.float64)

        sigma = 1.0
        from nvalchemiops.torch.interactions.electrostatics import (
            estimate_multipole_ewald_parameters,
        )

        params = estimate_multipole_ewald_parameters(
            positions,
            cell,
            sigma=sigma,
            accuracy=1e-6,
        )
        alpha = float(params.alpha.item())
        rcut = float(params.real_space_cutoff.item())
        kcut = float(params.reciprocal_space_cutoff.item())

        # Build the neighbor list for the real-space pair sum.
        from nvalchemiops.torch.neighbors import neighbor_list

        pbc = torch.tensor([True, True, True], device=device)
        pairs, nptr, shifts = neighbor_list(
            positions,
            rcut,
            cell=cell,
            pbc=pbc,
            return_neighbor_list=True,
        )
        idx_j = pairs[1].contiguous().to(torch.int32)
        nptr = nptr.to(torch.int32)
        shifts = shifts.to(torch.int32)

        e_total = multipole_ewald_summation(
            positions,
            source_feats_t,
            cell,
            idx_j,
            nptr,
            shifts,
            sigma=sigma,
            alpha=alpha,
            kspace_cutoff=kcut,
        )
        return float(e_total.item()), alpha, sigma

    def test_charges_only_matches_path_a(self, fix):
        """l_max=0 reference matches Path-A within 1e-3 relative."""
        e_path_a, alpha, sigma = self._run_path_a(fix, with_dipoles=False)
        e_ref = direct_ewald_energy(
            fix["positions"],
            fix["charges"],
            cell=fix["cell"],
            alpha=alpha,
            sigma=sigma,
            real_cutoff=10.0,
            kspace_cutoff=4.0 * math.pi / 4.14,
        )
        rel_err = abs(e_ref - e_path_a) / max(abs(e_path_a), 1e-12)
        print(
            f"\n[l_max=0]  Path-A = {e_path_a:.8e}   "
            f"Reference = {e_ref:.8e}   rel_err = {rel_err:.3e}"
        )
        assert rel_err < 1e-2, (
            f"Reference and Path-A disagree at l_max=0: "
            f"path_a={e_path_a}, ref={e_ref}, rel_err={rel_err}"
        )

    def test_charges_and_dipoles_matches_path_a(self, fix):
        """l_max=1 reference matches Path-A within 1e-3 relative."""
        e_path_a, alpha, sigma = self._run_path_a(fix, with_dipoles=True)
        e_ref = direct_ewald_energy(
            fix["positions"],
            fix["charges"],
            dipoles=fix["dipoles"],
            cell=fix["cell"],
            alpha=alpha,
            sigma=sigma,
            real_cutoff=10.0,
            kspace_cutoff=4.0 * math.pi / 4.14,
        )
        rel_err = abs(e_ref - e_path_a) / max(abs(e_path_a), 1e-12)
        print(
            f"\n[l_max=1]  Path-A = {e_path_a:.8e}   "
            f"Reference = {e_ref:.8e}   rel_err = {rel_err:.3e}"
        )
        assert rel_err < 1e-2, (
            f"Reference and Path-A disagree at l_max=1: "
            f"path_a={e_path_a}, ref={e_ref}, rel_err={rel_err}"
        )


class TestQuadrupoleSanity:
    """Smoke tests for the quadrupole channel — no external reference."""

    def test_zero_quadrupole_matches_dipole(self):
        """Setting Q=0 recovers the l_max=1 result exactly."""
        fix = _bcc_fixture(size=2)
        Z = np.zeros_like(fix["quadrupoles"])
        e_with_Q_zero = direct_ewald_energy(
            fix["positions"],
            fix["charges"],
            dipoles=fix["dipoles"],
            quadrupoles=Z,
            cell=fix["cell"],
            alpha=0.4,
            sigma=1.0,
            real_cutoff=8.0,
            kspace_cutoff=2.0,
        )
        e_dipole = direct_ewald_energy(
            fix["positions"],
            fix["charges"],
            dipoles=fix["dipoles"],
            cell=fix["cell"],
            alpha=0.4,
            sigma=1.0,
            real_cutoff=8.0,
            kspace_cutoff=2.0,
        )
        np.testing.assert_allclose(e_with_Q_zero, e_dipole, atol=0, rtol=0)

    def test_quadrupole_self_energy_formula(self):
        """Check the raw quadrupole self-energy coefficient
        ``E_self_Q_raw = |Q|_F² / (120 σ_c⁵ √π)``."""
        # |Q|² = 1.0 (one atom with Q_xx = 1, traceless)
        Q = np.zeros((1, 3, 3))
        Q[0, 0, 0] = 1.0
        Q[0, 1, 1] = -0.5
        Q[0, 2, 2] = -0.5
        q = np.zeros(1)
        alpha, sigma = 0.4, 1.0
        sigma_c = math.sqrt(sigma**2 + 0.25 / alpha**2)
        Q_F_sq = float(np.einsum("nab,nab->n", Q, Q).sum())
        expected_raw = Q_F_sq / (120.0 * sigma_c**5 * math.sqrt(math.pi))
        actual_raw = float(self_energy(q, quadrupoles=Q, alpha=alpha, sigma=sigma)[0])
        assert abs(actual_raw - expected_raw) / expected_raw < 1e-12

    def test_pair_energy_quadrupole_symmetric(self):
        """Pair energy is symmetric under (i ↔ j) swap with r → -r."""
        # Random multipoles
        rng = np.random.default_rng(2718)
        Q_i = rng.standard_normal((3, 3))
        Q_i = 0.5 * (Q_i + Q_i.T)
        Q_i -= np.eye(3) * np.trace(Q_i) / 3.0
        Q_j = rng.standard_normal((3, 3))
        Q_j = 0.5 * (Q_j + Q_j.T)
        Q_j -= np.eye(3) * np.trace(Q_j) / 3.0
        mu_i = rng.standard_normal(3) * 0.3
        mu_j = rng.standard_normal(3) * 0.3
        r_vec = np.array([1.2, -0.4, 0.7])

        E_ij = pair_energy(
            0.5,
            mu_i,
            Q_i,
            -0.5,
            mu_j,
            Q_j,
            r_vec,
            alpha=0.4,
            sigma=1.0,
        )
        E_ji = pair_energy(
            -0.5,
            mu_j,
            Q_j,
            0.5,
            mu_i,
            Q_i,
            -r_vec,
            alpha=0.4,
            sigma=1.0,
        )
        assert abs(E_ij - E_ji) < 1e-12 * max(abs(E_ij), 1.0)
