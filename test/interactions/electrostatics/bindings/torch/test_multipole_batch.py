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
Integration tests for the batched Path B bindings.

Covers the unified batched ``MultipoleSCFCache`` (built from a ``(B, 3, 3)``
cell via ``prepare_multipole_scf_cache``), ``multipole_scf_step_energy`` /
``multipole_scf_step_features`` with ``batch_idx``, and the batched
``multipole_electrostatic_energy`` / ``multipole_electrostatic_features``
(``cell`` ``(B, 3, 3)`` + ``batch_idx``).

Each test exercises a multi-system batch with unequal per-system
``(N_b, K_b)`` and checks: parity against per-system single-system calls;
finite, per-system-equivalent gradients w.r.t. ``(positions, charges,
dipoles)``; and finite double-backward (force-loss) gradients. Pad k-rows in
the batched cache carry zero weights, so the kernels ignore them without
explicit masking; systems with different ``K_b`` verify this implicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    multipole_electrostatic_features,
    multipole_scf_step_energy,
    multipole_scf_step_features,
    pack_multipole_moments,
    prepare_multipole_scf_cache,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_charges_dipoles,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _random_system(seed: int, n_atoms: int, box_len: float, device: str):
    """Build one random system: (positions, charges, dipoles, cell)."""
    rng = np.random.default_rng(seed)
    positions = torch.from_numpy(rng.uniform(0.0, box_len, size=(n_atoms, 3))).to(
        device=device, dtype=torch.float64
    )
    charges_np = rng.uniform(-1.0, 1.0, n_atoms)
    charges_np -= charges_np.mean()
    charges = torch.from_numpy(charges_np).to(device=device, dtype=torch.float64)
    dipoles_np = rng.standard_normal((n_atoms, 3)) * 0.3
    dipoles = torch.from_numpy(dipoles_np).to(device=device, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64, device=device) * box_len
    return positions, charges, dipoles, cell


def _batch_fixture(device: str, *, seed_base: int = 0):
    """Build a 3-system batch with unequal (N_b, box) shapes.

    Returns a dict bundling both the per-system tensors and the flat
    batched tensors, so tests can exercise both paths.
    """
    td = _torch_device(device)
    sizes = [(6, 4.5), (10, 5.2), (4, 3.8)]
    per_system = [
        _random_system(seed_base + b, n, L, td) for b, (n, L) in enumerate(sizes)
    ]

    positions_flat = torch.cat([s[0] for s in per_system], dim=0)
    charges_flat = torch.cat([s[1] for s in per_system], dim=0)
    dipoles_flat = torch.cat([s[2] for s in per_system], dim=0)
    cells = torch.stack([s[3] for s in per_system], dim=0)

    batch_idx = torch.cat(
        [
            torch.full((sizes[b][0],), b, dtype=torch.int32, device=td)
            for b in range(len(sizes))
        ]
    )
    return {
        "sizes": sizes,
        "per_system": per_system,
        "positions": positions_flat,
        "charges": charges_flat,
        "dipoles": dipoles_flat,
        "cells": cells,
        "batch_idx": batch_idx,
        "device": td,
    }


# ---------------------------------------------------------------------------
# Cache shape / pad-row invariants
# ---------------------------------------------------------------------------


class TestBatchCache:
    def test_shapes_and_pad_invariants(self, device):
        td = _torch_device(device)
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * L for L in (4.0, 5.5, 3.5)],
            dim=0,
        )
        cache = prepare_multipole_scf_cache(
            cells, sigma=0.5, receiver_sigmas=[0.7], kspace_cutoff=3.5, l_max=1
        )
        assert cache.is_batched
        assert cache.batch_size == 3
        assert cache.k_vectors.shape == (3, cache.n_k_max, 3)
        assert cache.source_phi_hat.shape == (3, cache.n_k_max, 4, 2)
        assert cache.receiver_phi_hat.shape == (3, cache.n_k_max, 1, 4, 2)

        # Pad rows carry zero weights.
        k_indices = torch.arange(cache.n_k_max, device=td)
        for b in range(3):
            valid = cache.valid_k_counts[b].item()
            pad_mask = k_indices >= valid
            if pad_mask.any():
                assert torch.all(cache.per_k_factor[b][pad_mask] == 0)
                assert torch.all(cache.k_factor_proj[b][pad_mask] == 0)
                assert torch.all(cache.source_phi_hat[b][pad_mask] == 0)
                assert torch.all(cache.receiver_phi_hat[b][pad_mask] == 0)
                assert torch.all((cache.k_vectors[b][pad_mask] == 0).all(dim=-1))


# ---------------------------------------------------------------------------
# SCF step parity vs per-system
# ---------------------------------------------------------------------------


def _per_system_energies_features(
    batch: dict, *, sigma: float, receiver_sigmas: list[float], k_cut: float
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    energies = []
    features = []
    for positions, charges, dipoles, cell in batch["per_system"]:
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=k_cut,
            l_max=1,
        )
        source_feats = pack_charges_dipoles(charges, dipoles)
        e = multipole_scf_step_energy(cache, positions, source_feats)
        f = multipole_scf_step_features(cache, positions, source_feats)
        energies.append(e)
        features.append(f)
    return energies, features


class TestBatchStepParity:
    sigma = 0.5
    receiver_sigmas = [0.7]
    k_cut = 3.5

    def test_energy_bit_parity(self, device):
        batch = _batch_fixture(device)
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            kspace_cutoff=self.k_cut,
            l_max=1,
        )
        e_b = multipole_scf_step_energy(
            cache_b,
            batch["positions"],
            pack_charges_dipoles(batch["charges"], batch["dipoles"]),
            batch_idx=batch["batch_idx"],
        )
        per_e, _ = _per_system_energies_features(
            batch,
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cut=self.k_cut,
        )
        assert e_b.shape == (batch["cells"].shape[0],)
        for b, ref in enumerate(per_e):
            torch.testing.assert_close(e_b[b], ref.reshape(()), rtol=0, atol=1e-12)

    def test_features_parity(self, device):
        batch = _batch_fixture(device)
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            kspace_cutoff=self.k_cut,
            l_max=1,
        )
        f_b = multipole_scf_step_features(
            cache_b,
            batch["positions"],
            pack_charges_dipoles(batch["charges"], batch["dipoles"]),
            batch_idx=batch["batch_idx"],
        )
        _, per_f = _per_system_energies_features(
            batch,
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            k_cut=self.k_cut,
        )
        off = 0
        for b, (n, _) in enumerate(batch["sizes"]):
            torch.testing.assert_close(f_b[off : off + n], per_f[b], rtol=0, atol=5e-13)
            off += n

    def test_charges_only_matches_zero_dipoles(self, device):
        """l_max=0 source_feats must match explicit-zero-dipole l_max=1 source_feats.

        Both paths represent the same physical system (no dipole moment); the
        two code paths go through different caches (l_max=0 vs l_max=1), so
        matching energies verify the l=1 contribution vanishes cleanly when
        the dipole block is zero.
        """
        batch = _batch_fixture(device)
        zeros = torch.zeros_like(batch["dipoles"])

        cache_l0 = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            kspace_cutoff=self.k_cut,
            l_max=0,
        )
        cache_l1 = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            kspace_cutoff=self.k_cut,
            l_max=1,
        )
        e_l0 = multipole_scf_step_energy(
            cache_l0,
            batch["positions"],
            pack_charges_dipoles(batch["charges"], None),
            batch_idx=batch["batch_idx"],
        )
        e_l1_zero = multipole_scf_step_energy(
            cache_l1,
            batch["positions"],
            pack_charges_dipoles(batch["charges"], zeros),
            batch_idx=batch["batch_idx"],
        )
        torch.testing.assert_close(e_l0, e_l1_zero, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Backward gradient parity
# ---------------------------------------------------------------------------


class TestBatchBackwardParity:
    sigma = 0.5
    receiver_sigmas = [0.7]
    k_cut = 3.5

    def test_energy_backward_matches_per_system(self, device):
        """Per-atom gradients of Σ E_b must match per-system ∂E/∂(pos/source_feats)."""
        batch = _batch_fixture(device)

        # Per-system reference grads.
        per_grads = []
        for positions, charges, dipoles, cell in batch["per_system"]:
            p = positions.detach().clone().requires_grad_(True)
            sf = (
                pack_charges_dipoles(charges.detach(), dipoles.detach())
                .clone()
                .requires_grad_(True)
            )
            cache = prepare_multipole_scf_cache(
                cell,
                sigma=self.sigma,
                receiver_sigmas=self.receiver_sigmas,
                kspace_cutoff=self.k_cut,
                l_max=1,
            )
            e = multipole_scf_step_energy(cache, p, sf)
            e.backward()
            per_grads.append((p.grad, sf.grad))

        # Batched grads.
        p_b = batch["positions"].detach().clone().requires_grad_(True)
        sf_b = (
            pack_charges_dipoles(batch["charges"].detach(), batch["dipoles"].detach())
            .clone()
            .requires_grad_(True)
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            kspace_cutoff=self.k_cut,
            l_max=1,
        )
        e_vec = multipole_scf_step_energy(
            cache_b, p_b, sf_b, batch_idx=batch["batch_idx"]
        )
        e_vec.sum().backward()

        off = 0
        for b, (n, _) in enumerate(batch["sizes"]):
            gp_ref, gsf_ref = per_grads[b]
            torch.testing.assert_close(
                p_b.grad[off : off + n], gp_ref, rtol=1e-10, atol=1e-10
            )
            torch.testing.assert_close(
                sf_b.grad[off : off + n], gsf_ref, rtol=1e-10, atol=1e-10
            )
            off += n

    def test_features_backward_finite(self, device):
        """Feature-loss backward produces finite grads for positions and source_feats."""
        batch = _batch_fixture(device)
        p = batch["positions"].detach().clone().requires_grad_(True)
        sf = (
            pack_charges_dipoles(batch["charges"].detach(), batch["dipoles"].detach())
            .clone()
            .requires_grad_(True)
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            kspace_cutoff=self.k_cut,
            l_max=1,
        )
        f = multipole_scf_step_features(cache_b, p, sf, batch_idx=batch["batch_idx"])
        (f**2).sum().backward()
        for t in (p, sf):
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Double-backward (force-loss training case)
# ---------------------------------------------------------------------------


class TestBatchDoubleBackward:
    """Double-backward paths: ensure d(F^2)/dx stays on the autograd tape."""

    sigma = 0.5
    receiver_sigmas = [0.7]
    k_cut = 3.0  # smaller cutoff → cheaper test

    def test_force_loss_backward(self, device):
        batch = _batch_fixture(device)
        p = batch["positions"].detach().clone().requires_grad_(True)
        sf = (
            pack_charges_dipoles(batch["charges"].detach(), batch["dipoles"].detach())
            .clone()
            .requires_grad_(True)
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            kspace_cutoff=self.k_cut,
            l_max=1,
        )
        e_vec = multipole_scf_step_energy(cache_b, p, sf, batch_idx=batch["batch_idx"])
        e_total = e_vec.sum()
        forces = -torch.autograd.grad(e_total, p, create_graph=True)[0]
        # Force-loss surrogate: simulate matching a zero-force target.
        loss = (forces**2).sum()
        loss.backward()
        for t in (p, sf):
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0

    def test_feature_force_loss_backward(self, device):
        """Same but via the feature pipeline."""
        batch = _batch_fixture(device)
        p = batch["positions"].detach().clone().requires_grad_(True)
        sf = (
            pack_charges_dipoles(batch["charges"].detach(), batch["dipoles"].detach())
            .clone()
            .requires_grad_(True)
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            kspace_cutoff=self.k_cut,
            l_max=1,
        )
        f = multipole_scf_step_features(cache_b, p, sf, batch_idx=batch["batch_idx"])
        scalar = f.sum()
        (grad_p,) = torch.autograd.grad(scalar, p, create_graph=True)
        (grad_p**2).sum().backward()
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# One-shot entries: same math, thinner wrapper
# ---------------------------------------------------------------------------


class TestBatchOneShot:
    sigma = 0.5
    receiver_sigmas = [0.7]
    k_cut = 3.5

    def test_energy_matches_scf_step(self, device):
        batch = _batch_fixture(device)
        source_feats = pack_charges_dipoles(batch["charges"], batch["dipoles"])
        e_oneshot = multipole_electrostatic_energy(
            batch["positions"],
            source_feats,
            batch["cells"],
            batch_idx=batch["batch_idx"],
            sigma=self.sigma,
            kspace_cutoff=self.k_cut,
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=[self.sigma],  # one-shot uses sigma as the only receiver σ
            kspace_cutoff=self.k_cut,
            l_max=1,
        )
        e_step = multipole_scf_step_energy(
            cache_b,
            batch["positions"],
            source_feats,
            batch_idx=batch["batch_idx"],
        )
        torch.testing.assert_close(e_oneshot, e_step, rtol=0, atol=1e-12)

    def test_features_matches_scf_step(self, device):
        batch = _batch_fixture(device)
        source_feats = pack_charges_dipoles(batch["charges"], batch["dipoles"])
        f_oneshot = multipole_electrostatic_features(
            batch["positions"],
            source_feats,
            batch["cells"],
            batch_idx=batch["batch_idx"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            kspace_cutoff=self.k_cut,
        )
        cache_b = prepare_multipole_scf_cache(
            batch["cells"],
            sigma=self.sigma,
            receiver_sigmas=self.receiver_sigmas,
            kspace_cutoff=self.k_cut,
            l_max=1,
        )
        f_step = multipole_scf_step_features(
            cache_b,
            batch["positions"],
            source_feats,
            batch_idx=batch["batch_idx"],
        )
        torch.testing.assert_close(f_oneshot, f_step, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestBatchValidation:
    def test_bad_cells_shape(self, device):
        td = _torch_device(device)
        cells = torch.zeros((4,), dtype=torch.float64, device=td)  # not (3,3)/(B,3,3)
        with pytest.raises(ValueError, match="cell must be"):
            prepare_multipole_scf_cache(
                cells, sigma=0.5, receiver_sigmas=[0.7], kspace_cutoff=3.0
            )

    def test_batch_idx_length_mismatch(self, device):
        td = _torch_device(device)
        cells = torch.stack(
            [torch.eye(3, dtype=torch.float64, device=td) * 4.0 for _ in range(2)],
            dim=0,
        )
        cache_b = prepare_multipole_scf_cache(
            cells, sigma=0.5, receiver_sigmas=[0.7], kspace_cutoff=3.0, l_max=1
        )
        pos = torch.zeros((5, 3), dtype=torch.float64, device=td)
        source_feats = torch.zeros((5, 4), dtype=torch.float64, device=td)
        batch_idx_bad = torch.zeros(4, dtype=torch.int32, device=td)  # wrong length
        with pytest.raises(ValueError, match="batch_idx must be"):
            multipole_scf_step_energy(
                cache_b, pos, source_feats, batch_idx=batch_idx_bad
            )


# ---------------------------------------------------------------------------
# l_max=2 batched Path-B energy parity
# ---------------------------------------------------------------------------


def _quadrupole_batch_fixture(device: str, *, seed_base: int = 100):
    """3-system batch with per-atom detraced (traceless) symmetric quadrupoles."""
    td = _torch_device(device)
    sizes = [(6, 4.5), (10, 5.2), (4, 3.8)]
    systems = []
    for b, (n, L) in enumerate(sizes):
        rng = np.random.default_rng(seed_base + b)
        pos = rng.uniform(0.0, L, size=(n, 3))
        q = rng.normal(size=n)
        q -= q.mean()
        mu = rng.normal(size=(n, 3)) * 0.3
        Qr = rng.normal(size=(n, 3, 3)) * 0.1
        Q = 0.5 * (Qr + Qr.transpose(0, 2, 1))
        Q -= (np.trace(Q, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
        systems.append(
            (
                torch.tensor(pos, device=td),
                torch.tensor(q, device=td),
                torch.tensor(mu, device=td),
                torch.tensor(Q, device=td),
                torch.eye(3, dtype=torch.float64, device=td) * L,
            )
        )
    return td, sizes, systems


class TestBatchQuadrupole:
    """Batched Path-B l=2 energy matches per-system (the validated single path)."""

    sigma = 0.5
    k_cut = 12.0

    def test_energy_matches_per_system(self, device):
        td, sizes, systems = _quadrupole_batch_fixture(device)
        pos = torch.cat([s[0] for s in systems], dim=0)
        mm = pack_multipole_moments(
            torch.cat([s[1] for s in systems], dim=0),
            torch.cat([s[2] for s in systems], dim=0),
            torch.cat([s[3] for s in systems], dim=0),
        )
        cells = torch.stack([s[4] for s in systems], dim=0)
        batch_idx = torch.cat(
            [
                torch.full((sizes[b][0],), b, dtype=torch.int32, device=td)
                for b in range(len(sizes))
            ]
        )
        e_b = multipole_electrostatic_energy(
            pos,
            mm,
            cells,
            batch_idx=batch_idx,
            sigma=self.sigma,
            kspace_cutoff=self.k_cut,
        )
        assert e_b.shape == (len(sizes),)
        for b, (p, q, mu, Q, cell) in enumerate(systems):
            e_single = multipole_electrostatic_energy(
                p,
                pack_multipole_moments(q, mu, Q),
                cell,
                sigma=self.sigma,
                kspace_cutoff=self.k_cut,
            )
            torch.testing.assert_close(
                e_b[b], e_single.reshape(()), rtol=1e-9, atol=1e-9
            )

    def test_forces_match_per_system(self, device):
        td, sizes, systems = _quadrupole_batch_fixture(device, seed_base=200)
        pos = torch.cat([s[0] for s in systems], dim=0).requires_grad_(True)
        mm = pack_multipole_moments(
            torch.cat([s[1] for s in systems], dim=0),
            torch.cat([s[2] for s in systems], dim=0),
            torch.cat([s[3] for s in systems], dim=0),
        )
        cells = torch.stack([s[4] for s in systems], dim=0)
        batch_idx = torch.cat(
            [
                torch.full((sizes[b][0],), b, dtype=torch.int32, device=td)
                for b in range(len(sizes))
            ]
        )
        e_b = multipole_electrostatic_energy(
            pos,
            mm,
            cells,
            batch_idx=batch_idx,
            sigma=self.sigma,
            kspace_cutoff=self.k_cut,
        )
        (g_b,) = torch.autograd.grad(e_b.sum(), pos)
        off = 0
        for b, (p, q, mu, Q, cell) in enumerate(systems):
            n = sizes[b][0]
            p_g = p.clone().requires_grad_(True)
            e_single = multipole_electrostatic_energy(
                p_g,
                pack_multipole_moments(q, mu, Q),
                cell,
                sigma=self.sigma,
                kspace_cutoff=self.k_cut,
            )
            (g_s,) = torch.autograd.grad(e_single, p_g)
            torch.testing.assert_close(g_b[off : off + n], g_s, rtol=1e-7, atol=1e-7)
            off += n
