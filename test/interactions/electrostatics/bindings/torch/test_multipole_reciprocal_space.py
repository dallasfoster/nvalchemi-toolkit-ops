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

r"""
Integration tests for :func:`multipole_reciprocal_space_energy`.

Covers:

* α→∞ limit convergence to the direct-kspace Path B energy
  (:func:`multipole_electrostatic_energy`). At finite ``α`` the
  Gaussian-damped reciprocal sum includes fewer high-k modes; at large ``α``
  the damping approaches 1 and the two must agree.
* Shape / dtype / device invariants.
* First- and second-order autograd (checks the composition stays on the tape).
* Input validation (negative α, missing k-grid info).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    multipole_reciprocal_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_charges_dipoles,
)


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _system(
    *, n_atoms: int, box_len: float, device: str, seed: int, with_dipoles: bool
):
    rng = np.random.default_rng(seed)
    td = _torch_device(device)
    positions = torch.from_numpy(rng.uniform(0, box_len, size=(n_atoms, 3))).to(
        td, dtype=torch.float64
    )
    charges_np = rng.uniform(-1, 1, size=n_atoms)
    charges_np -= charges_np.mean()
    charges = torch.from_numpy(charges_np).to(td, dtype=torch.float64)
    dipoles = None
    if with_dipoles:
        dipoles = torch.from_numpy(0.3 * rng.standard_normal((n_atoms, 3))).to(
            td, dtype=torch.float64
        )
    cell = torch.eye(3, dtype=torch.float64, device=td) * box_len
    return positions, charges, dipoles, cell


class TestMultipoleReciprocalSpaceLimit:
    """At α→∞ the Gaussian damping factor exp(−k²/(4α²)) → 1 so
    reciprocal-space must converge to the direct-kspace Path B result.
    """

    @pytest.mark.parametrize("with_dipoles", [False, True])
    def test_large_alpha_matches_direct_kspace(self, device, with_dipoles):
        positions, charges, dipoles, cell = _system(
            n_atoms=6, box_len=5.0, device=device, seed=0, with_dipoles=with_dipoles
        )
        sigma, k_cutoff = 1.0, 3.0

        source_feats = pack_charges_dipoles(charges, dipoles)
        e_direct = multipole_electrostatic_energy(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            kspace_cutoff=k_cutoff,
            include_self_interaction=True,
        )
        e_recip = multipole_reciprocal_space_energy(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            alpha=100.0,
            kspace_cutoff=k_cutoff,
        )
        # The residual is O((k_max/α)²) from the exp series; with
        # k_cutoff=3 and α=100 it's ~(3/100)² ≈ 1e-3 magnitude.
        torch.testing.assert_close(e_recip, e_direct, rtol=2e-3, atol=2e-3)

    def test_small_alpha_damps_to_zero(self, device):
        """Heavy Gaussian damping (small α) drives the reciprocal sum toward zero."""
        positions, charges, dipoles, cell = _system(
            n_atoms=6, box_len=5.0, device=device, seed=0, with_dipoles=False
        )
        e = multipole_reciprocal_space_energy(
            positions,
            pack_charges_dipoles(charges, None),
            cell,
            sigma=1.0,
            alpha=0.05,
            kspace_cutoff=3.0,
        )
        assert abs(e.item()) < 1e-10


class TestMultipoleReciprocalSpaceShapes:
    def test_output_scalar_float64(self, device):
        positions, charges, dipoles, cell = _system(
            n_atoms=5, box_len=5.0, device=device, seed=0, with_dipoles=True
        )
        e = multipole_reciprocal_space_energy(
            positions,
            pack_charges_dipoles(charges, dipoles),
            cell,
            sigma=1.0,
            alpha=0.4,
            kspace_cutoff=3.0,
        )
        assert e.shape == ()
        assert e.dtype == torch.float64
        assert e.device.type == _torch_device(device)


class TestMultipoleReciprocalSpaceAutograd:
    def test_first_order_backward(self, device):
        positions_, charges_, dipoles_, cell = _system(
            n_atoms=6, box_len=5.0, device=device, seed=7, with_dipoles=True
        )
        positions = positions_.detach().clone().requires_grad_(True)
        source_feats = pack_charges_dipoles(
            charges_.detach().clone(), dipoles_.detach().clone()
        ).requires_grad_(True)
        e = multipole_reciprocal_space_energy(
            positions,
            source_feats,
            cell,
            sigma=1.0,
            alpha=0.4,
            kspace_cutoff=3.0,
        )
        e.backward()
        for t in (positions, source_feats):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0

    def test_double_backward_force_loss(self, device):
        """create_graph=True force-loss path (reuses MultipoleRhoFunction's 2nd-order)."""
        positions_, charges_, dipoles_, cell = _system(
            n_atoms=5, box_len=5.0, device=device, seed=11, with_dipoles=True
        )
        positions = positions_.detach().clone().requires_grad_(True)
        source_feats = pack_charges_dipoles(
            charges_.detach().clone(), dipoles_.detach().clone()
        ).requires_grad_(True)
        e = multipole_reciprocal_space_energy(
            positions,
            source_feats,
            cell,
            sigma=1.0,
            alpha=0.4,
            kspace_cutoff=3.0,
        )
        (forces_neg,) = torch.autograd.grad(e, positions, create_graph=True)
        loss = (forces_neg**2).sum()
        loss.backward()
        for t in (positions, source_feats):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0


class TestBatchMultipoleReciprocalSpace:
    """Batched reciprocal-space entry — checks the sugar wrapper stays wired."""

    def test_matches_per_system_loop(self, device):
        rng = np.random.default_rng(0)
        td = _torch_device(device)
        n_per, L = 5, 4.0

        def _sys():
            pos = torch.from_numpy(rng.uniform(0, L, (n_per, 3))).to(td, torch.float64)
            chg_np = rng.uniform(-1, 1, n_per)
            chg_np -= chg_np.mean()
            chg = torch.from_numpy(chg_np).to(td, torch.float64)
            mu = torch.from_numpy(0.2 * rng.standard_normal((n_per, 3))).to(
                td, torch.float64
            )
            cell = torch.eye(3, dtype=torch.float64, device=td) * L
            return pos, chg, mu, cell

        systems = [_sys() for _ in range(3)]
        sigma, alpha, kcut = 1.0, 0.4, 3.0

        per_e = torch.stack(
            [
                multipole_reciprocal_space_energy(
                    p,
                    pack_charges_dipoles(c, m),
                    cell,
                    sigma=sigma,
                    alpha=alpha,
                    kspace_cutoff=kcut,
                )
                for (p, c, m, cell) in systems
            ]
        )
        pos_all = torch.cat([s[0] for s in systems])
        chg_all = torch.cat([s[1] for s in systems])
        mu_all = torch.cat([s[2] for s in systems])
        cells = torch.stack([s[3] for s in systems])
        bi = torch.cat(
            [torch.full((n_per,), i, dtype=torch.int32, device=td) for i in range(3)]
        )
        e_batch = multipole_reciprocal_space_energy(
            pos_all,
            pack_charges_dipoles(chg_all, mu_all),
            cells,
            batch_idx=bi,
            sigma=sigma,
            alpha=alpha,
            kspace_cutoff=kcut,
        )
        torch.testing.assert_close(e_batch, per_e, rtol=0, atol=1e-14)

    def test_backward_runs(self, device):
        rng = np.random.default_rng(17)
        td = _torch_device(device)
        n_per = 4
        B = 2
        positions = (
            torch.from_numpy(rng.uniform(0, 5.0, (B * n_per, 3)))
            .to(td, torch.float64)
            .requires_grad_(True)
        )
        charges_np = rng.uniform(-1, 1, B * n_per)
        charges_np -= charges_np.mean()
        charges = torch.from_numpy(charges_np).to(td, torch.float64)
        dipoles = torch.from_numpy(0.2 * rng.standard_normal((B * n_per, 3))).to(
            td, torch.float64
        )
        source_feats = pack_charges_dipoles(charges, dipoles).requires_grad_(True)
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * 5.0 for _ in range(B)]
        )
        bi = torch.cat(
            [torch.full((n_per,), i, dtype=torch.int32, device=td) for i in range(B)]
        )
        e = multipole_reciprocal_space_energy(
            positions,
            source_feats,
            cells,
            batch_idx=bi,
            sigma=1.0,
            alpha=0.4,
            kspace_cutoff=3.0,
        )
        assert e.shape == (B,)
        e.sum().backward()
        for t in (positions, source_feats):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0


class TestMultipoleReciprocalSpaceValidation:
    def test_alpha_must_be_positive(self, device):
        positions, charges, _, cell = _system(
            n_atoms=4, box_len=5.0, device=device, seed=0, with_dipoles=False
        )
        with pytest.raises(ValueError, match="alpha must be positive"):
            multipole_reciprocal_space_energy(
                positions,
                pack_charges_dipoles(charges, None),
                cell,
                sigma=1.0,
                alpha=-0.1,
                kspace_cutoff=3.0,
            )

    def test_requires_k_grid_info(self, device):
        positions, charges, _, cell = _system(
            n_atoms=4, box_len=5.0, device=device, seed=0, with_dipoles=False
        )
        with pytest.raises(ValueError, match="k_vectors"):
            multipole_reciprocal_space_energy(
                positions,
                pack_charges_dipoles(charges, None),
                cell,
                sigma=1.0,
                alpha=0.4,
            )


# ---------------------------------------------------------------------------
# Fused reciprocal-space scalar Function
# ---------------------------------------------------------------------------


class TestMultipoleReciprocalSpaceDipoleFusedScalar:
    r"""Parity tests for ``MultipoleReciprocalSpaceDipoleFusedScalarFunction``.

    The fused Function uses the SAME forward kernels as
    :func:`multipole_scf_step_energy` / :func:`multipole_reciprocal_space_energy`
    and computes ``grad_rho`` from the same scalar-energy formula, so parity is
    bit-exact at fp64.
    """

    @pytest.fixture
    def gpu_system(self):
        if not torch.cuda.is_available():
            pytest.skip("Path A reciprocal kernels are GPU-only")
        positions, charges, dipoles, cell = _system(
            n_atoms=8, box_len=5.0, device="cuda:0", seed=0xCAFE, with_dipoles=True
        )
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        sigma, alpha = 1.0, 0.5
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=[sigma],
            kspace_cutoff=3.0,
            l_max=1,
            alpha=alpha,
            device="cuda:0",
        )
        return {
            "positions": positions,
            "charges": charges,
            "dipoles": dipoles,
            "cell": cell,
            "cache": cache,
        }

    def _run_reference(self, sd, *, with_pos, with_q, with_mu, include_self):
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
            multipole_scf_step_energy,
        )

        p = sd["positions"].clone().detach().requires_grad_(with_pos)
        q = sd["charges"].clone().detach().requires_grad_(with_q)
        mu = sd["dipoles"].clone().detach().requires_grad_(with_mu)
        sf = pack_charges_dipoles(q, mu)
        e = multipole_scf_step_energy(
            sd["cache"], p, sf, include_self_interaction=include_self
        )
        return p, q, mu, e

    def _run_fused(self, sd, *, with_pos, with_q, with_mu, include_self):
        from nvalchemiops.torch.interactions.electrostatics.multipole_autograd import (
            multipole_reciprocal_space_dipole_fused_scalar,
        )

        p = sd["positions"].clone().detach().requires_grad_(with_pos)
        q = sd["charges"].clone().detach().requires_grad_(with_q)
        mu = sd["dipoles"].clone().detach().requires_grad_(with_mu)
        sf = pack_charges_dipoles(q, mu)
        e = multipole_reciprocal_space_dipole_fused_scalar(
            p, sf, sd["cache"], include_self_interaction=include_self
        )
        return p, q, mu, e

    @pytest.mark.parametrize("include_self", [True, False])
    def test_forward_parity(self, gpu_system, include_self):
        """Bit-exact energy parity vs `multipole_scf_step_energy`."""
        _, _, _, e_ref = self._run_reference(
            gpu_system,
            with_pos=False,
            with_q=False,
            with_mu=False,
            include_self=include_self,
        )
        _, _, _, e_f = self._run_fused(
            gpu_system,
            with_pos=False,
            with_q=False,
            with_mu=False,
            include_self=include_self,
        )
        torch.testing.assert_close(e_f, e_ref, rtol=0, atol=0)

    @pytest.mark.parametrize("include_self", [True, False])
    def test_all_grads_parity(self, gpu_system, include_self):
        p_r, q_r, mu_r, e_r = self._run_reference(
            gpu_system,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=include_self,
        )
        e_r.backward()
        p_f, q_f, mu_f, e_f = self._run_fused(
            gpu_system,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=include_self,
        )
        e_f.backward()
        torch.testing.assert_close(p_f.grad, p_r.grad, rtol=0, atol=0)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=0)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=0)

    def test_per_input_gating_pos_only(self, gpu_system):
        p_f, q_f, mu_f, e_f = self._run_fused(
            gpu_system,
            with_pos=True,
            with_q=False,
            with_mu=False,
            include_self=True,
        )
        e_f.backward()
        assert p_f.grad is not None
        assert q_f.grad is None
        assert mu_f.grad is None

    def test_no_grad_path(self, gpu_system):
        _, _, _, e_f = self._run_fused(
            gpu_system,
            with_pos=False,
            with_q=False,
            with_mu=False,
            include_self=True,
        )
        assert not e_f.requires_grad
        assert torch.isfinite(e_f).item()

    def test_weighted_backward(self, gpu_system):
        """Custom upstream scalar grad multiplier; gradients scale linearly.

        Tolerance is 1 ULP: the fused path multiplies by ``upstream`` after the
        backward kernels finish while the reference multiplies ``grad_rho``
        before, so atomic_add ordering gives fp64 ULP drift.
        """
        p_f, q_f, mu_f, e_f = self._run_fused(
            gpu_system,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=True,
        )
        upstream = torch.tensor(2.5, dtype=torch.float64, device=e_f.device)
        e_f.backward(upstream)
        p_r, q_r, mu_r, e_r = self._run_reference(
            gpu_system,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=True,
        )
        e_r.backward(upstream)
        torch.testing.assert_close(p_f.grad, p_r.grad, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=1e-14, atol=1e-15)


class TestBatchMultipoleReciprocalSpaceDipoleFusedScalar:
    r"""Parity tests for the batched Path A reciprocal fused Function."""

    @pytest.fixture
    def gpu_batch(self):
        if not torch.cuda.is_available():
            pytest.skip("Path A reciprocal kernels are GPU-only")
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        sigma, alpha = 1.0, 0.5
        n_systems = 3
        n_per_sys = 6
        sub = []
        for s in range(n_systems):
            p, q, mu, c = _system(
                n_atoms=n_per_sys,
                box_len=5.0,
                device="cuda:0",
                seed=0x300 + s,
                with_dipoles=True,
            )
            sub.append({"positions": p, "charges": q, "dipoles": mu, "cell": c})

        td = "cuda:0"
        positions = torch.cat([s["positions"] for s in sub], dim=0)
        charges = torch.cat([s["charges"] for s in sub], dim=0)
        dipoles = torch.cat([s["dipoles"] for s in sub], dim=0)
        cells = torch.stack([s["cell"] for s in sub], dim=0)
        batch_idx = torch.cat(
            [
                torch.full((n_per_sys,), s_idx, dtype=torch.int32, device=td)
                for s_idx in range(n_systems)
            ]
        )

        cache = prepare_multipole_scf_cache(
            cells,
            sigma=sigma,
            receiver_sigmas=[sigma],
            kspace_cutoff=3.0,
            l_max=1,
            alpha=alpha,
            device=td,
        )
        return {
            "positions": positions,
            "charges": charges,
            "dipoles": dipoles,
            "batch_idx": batch_idx,
            "cache": cache,
            "n_systems": n_systems,
        }

    def _run_reference(self, b, *, with_pos, with_q, with_mu, include_self):
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
            multipole_scf_step_energy,
        )

        p = b["positions"].clone().detach().requires_grad_(with_pos)
        q = b["charges"].clone().detach().requires_grad_(with_q)
        mu = b["dipoles"].clone().detach().requires_grad_(with_mu)
        sf = pack_charges_dipoles(q, mu)
        e = multipole_scf_step_energy(
            b["cache"],
            p,
            sf,
            batch_idx=b["batch_idx"],
            include_self_interaction=include_self,
        )
        return p, q, mu, e

    def _run_fused(self, b, *, with_pos, with_q, with_mu, include_self):
        from nvalchemiops.torch.interactions.electrostatics.multipole_autograd_batch import (
            batch_multipole_reciprocal_space_dipole_fused_scalar,
        )

        p = b["positions"].clone().detach().requires_grad_(with_pos)
        q = b["charges"].clone().detach().requires_grad_(with_q)
        mu = b["dipoles"].clone().detach().requires_grad_(with_mu)
        sf = pack_charges_dipoles(q, mu)
        e = batch_multipole_reciprocal_space_dipole_fused_scalar(
            p,
            sf,
            b["batch_idx"],
            b["cache"],
            include_self_interaction=include_self,
        )
        return p, q, mu, e

    @pytest.mark.parametrize("include_self", [True, False])
    def test_forward_parity(self, gpu_batch, include_self):
        _, _, _, e_ref = self._run_reference(
            gpu_batch,
            with_pos=False,
            with_q=False,
            with_mu=False,
            include_self=include_self,
        )
        _, _, _, e_f = self._run_fused(
            gpu_batch,
            with_pos=False,
            with_q=False,
            with_mu=False,
            include_self=include_self,
        )
        torch.testing.assert_close(e_f, e_ref, rtol=0, atol=0)

    @pytest.mark.parametrize("include_self", [True, False])
    def test_all_grads_parity(self, gpu_batch, include_self):
        p_r, q_r, mu_r, e_r = self._run_reference(
            gpu_batch,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=include_self,
        )
        e_r.sum().backward()
        p_f, q_f, mu_f, e_f = self._run_fused(
            gpu_batch,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=include_self,
        )
        e_f.sum().backward()
        torch.testing.assert_close(p_f.grad, p_r.grad, rtol=0, atol=0)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=0)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=0)

    def test_weighted_backward(self, gpu_batch):
        """Per-system upstream-weight broadcast — non-uniform weights.

        Tolerance is 1 ULP — same atomic_add ordering divergence as the
        single-system weighted_backward test.
        """
        p_f, q_f, mu_f, e_f = self._run_fused(
            gpu_batch,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=False,
        )
        weights = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float64, device=e_f.device)
        (weights * e_f).sum().backward()
        p_r, q_r, mu_r, e_r = self._run_reference(
            gpu_batch,
            with_pos=True,
            with_q=True,
            with_mu=True,
            include_self=False,
        )
        (weights * e_r).sum().backward()
        torch.testing.assert_close(p_f.grad, p_r.grad, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=1e-14, atol=1e-15)
