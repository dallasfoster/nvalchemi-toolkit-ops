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

r"""Integration tests for the public ``multipole_electrostatic_energy`` binding.

Parity tests match the k-vectors the binding generates internally so the
reference sees the same grid.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_charges_dipoles,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.math import FIELD_CONSTANT
from nvalchemiops.torch.math.gto import NormMode


def _torch_device(device: str) -> str:
    """Map conftest's ``"cuda:0"`` to PyTorch's ``"cuda"`` form."""
    return "cuda" if "cuda" in device else "cpu"


def _random_system(
    *,
    seed: int = 0,
    n_atoms: int = 8,
    box_len: float = 6.0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    with_dipoles: bool = True,
) -> dict:
    """Build a small neutral random periodic system for parity tests."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, box_len, size=(n_atoms, 3))
    charges = rng.uniform(-1.0, 1.0, n_atoms)
    charges -= charges.mean()
    dipoles = rng.standard_normal((n_atoms, 3)) * 0.3
    cell = np.eye(3) * box_len

    out = {
        "positions": torch.from_numpy(positions).to(device=device, dtype=dtype),
        "charges": torch.from_numpy(charges).to(device=device, dtype=dtype),
        "cell": torch.from_numpy(cell).to(device=device, dtype=dtype),
        "positions_np": positions,
        "charges_np": charges,
        "cell_np": cell,
    }
    if with_dipoles:
        out["dipoles"] = torch.from_numpy(dipoles).to(device=device, dtype=dtype)
        out["dipoles_np"] = dipoles
    else:
        out["dipoles"] = None
        out["dipoles_np"] = np.zeros_like(dipoles)
    return out


class TestBasics:
    """Shape / dtype / device invariants for the public entry point."""

    def test_returns_scalar_float64(self, device):
        td = _torch_device(device)
        sys = _random_system(seed=0, n_atoms=4, device=td, dtype=torch.float64)
        energy = multipole_electrostatic_energy(
            sys["positions"],
            pack_charges_dipoles(sys["charges"], sys["dipoles"]),
            sys["cell"],
            sigma=1.0,
            kspace_cutoff=4.0,
        )
        assert energy.shape == ()
        assert energy.dtype == torch.float64
        assert energy.device.type == td

    def test_float32_positions_give_float64_energy(self, device):
        td = _torch_device(device)
        sys = _random_system(seed=1, n_atoms=4, device=td, dtype=torch.float32)
        energy = multipole_electrostatic_energy(
            sys["positions"],
            pack_charges_dipoles(sys["charges"], sys["dipoles"]),
            sys["cell"],
            sigma=1.0,
            kspace_cutoff=4.0,
        )
        assert energy.dtype == torch.float64

    def test_charges_only_matches_zero_dipole_branch(self, device):
        """Passing ``dipoles=None`` matches passing an explicit zero array."""
        td = _torch_device(device)
        sys = _random_system(seed=2, n_atoms=5, device=td, dtype=torch.float64)
        zero_dipoles = torch.zeros_like(sys["dipoles"])

        e_none = multipole_electrostatic_energy(
            sys["positions"],
            pack_charges_dipoles(sys["charges"], None),
            sys["cell"],
            sigma=1.0,
            kspace_cutoff=4.0,
        )
        e_zero = multipole_electrostatic_energy(
            sys["positions"],
            pack_charges_dipoles(sys["charges"], zero_dipoles),
            sys["cell"],
            sigma=1.0,
            kspace_cutoff=4.0,
        )
        assert torch.allclose(e_none, e_zero, rtol=1e-14, atol=1e-14)

    def test_include_self_interaction_adds_back_self_energy(self, device):
        """include_self_interaction=True equals the other path plus 0.5·E_self."""
        td = _torch_device(device)
        sys = _random_system(seed=3, n_atoms=4, device=td, dtype=torch.float64)
        source_feats = pack_charges_dipoles(sys["charges"], sys["dipoles"])
        e_with = multipole_electrostatic_energy(
            sys["positions"],
            source_feats,
            sys["cell"],
            sigma=1.0,
            kspace_cutoff=4.0,
            include_self_interaction=True,
        )
        e_without = multipole_electrostatic_energy(
            sys["positions"],
            source_feats,
            sys["cell"],
            sigma=1.0,
            kspace_cutoff=4.0,
            include_self_interaction=False,
        )
        # E_self is positive, so include_self=False gives the smaller energy.
        assert float(e_with) != float(e_without)
        assert float(e_with) > float(e_without)


class TestValidation:
    """User-facing input validation errors."""

    def test_bad_positions_shape(self):
        with pytest.raises(ValueError, match="positions"):
            multipole_electrostatic_energy(
                torch.zeros(5),
                torch.zeros((5, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64),
                sigma=1.0,
                kspace_cutoff=4.0,
            )

    def test_source_feats_wrong_length(self):
        with pytest.raises(ValueError, match="multipole_moments"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((3, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                kspace_cutoff=4.0,
            )

    def test_source_feats_wrong_last_dim(self):
        with pytest.raises(ValueError, match="source_feats|last-dim"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 5), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                kspace_cutoff=4.0,
            )

    def test_bad_cell_shape(self):
        with pytest.raises(ValueError, match="cell"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(4, dtype=torch.float64),
                sigma=1.0,
                kspace_cutoff=4.0,
            )

    def test_non_positive_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=0.0,
                kspace_cutoff=4.0,
            )

    def test_non_positive_kspace_cutoff(self):
        with pytest.raises(ValueError, match="kspace_cutoff"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                kspace_cutoff=-1.0,
            )

    def test_missing_both_kspace_cutoff_and_k_vectors(self):
        """At least one of ``kspace_cutoff`` / ``k_vectors`` must be supplied."""
        with pytest.raises(ValueError, match="k_vectors"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
            )

    def test_k_vectors_bad_shape(self):
        with pytest.raises(ValueError, match="k_vectors"):
            multipole_electrostatic_energy(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                k_vectors=torch.zeros(5, dtype=torch.float64),
            )


class TestNormalizationAndKVectors:
    """Normalization handling and precomputed-k-vector equivalence."""

    def test_precomputed_k_vectors_match_internal_generation(self, device):
        """Pre-supplying the same k-grid the binding would build internally gives the same energy."""
        td = _torch_device(device)
        sys = _random_system(seed=13, n_atoms=6, box_len=5.0, device=td)
        kspace_cutoff = 3.5

        source_feats = pack_charges_dipoles(sys["charges"], sys["dipoles"])
        e_internal = multipole_electrostatic_energy(
            sys["positions"],
            source_feats,
            sys["cell"],
            sigma=1.0,
            kspace_cutoff=kspace_cutoff,
        )

        # Same k-grid, fed in explicitly.
        k_half = generate_k_vectors_ewald_summation(sys["cell"], kspace_cutoff)
        k_vecs = torch.cat([k_half.new_zeros((1, 3)), k_half], dim=0).to(
            dtype=torch.float64
        )
        e_external = multipole_electrostatic_energy(
            sys["positions"],
            source_feats,
            sys["cell"],
            sigma=1.0,
            k_vectors=k_vecs,
        )
        assert float(e_internal) == float(e_external)

    def test_accepts_string_and_int_normalize(self, device):
        """``normalize=`` accepts ``NormMode``, int, and lowercase/uppercase strings."""
        td = _torch_device(device)
        sys = _random_system(seed=11, n_atoms=5, device=td, dtype=torch.float64)
        kwargs = dict(
            positions=sys["positions"],
            multipole_moments=pack_charges_dipoles(sys["charges"], sys["dipoles"]),
            cell=sys["cell"],
            sigma=1.0,
            kspace_cutoff=3.5,
        )
        e_enum = multipole_electrostatic_energy(**kwargs, normalize=NormMode.RECEIVER)
        e_int = multipole_electrostatic_energy(
            **kwargs, normalize=int(NormMode.RECEIVER)
        )
        e_str_lower = multipole_electrostatic_energy(**kwargs, normalize="receiver")
        e_str_upper = multipole_electrostatic_energy(**kwargs, normalize="RECEIVER")
        assert float(e_enum) == float(e_int) == float(e_str_lower) == float(e_str_upper)


class TestPhysicalInvariants:
    """Physical sanity checks that don't depend on an external reference."""

    def test_translation_invariance(self, device):
        """Rigidly translating all atoms leaves the energy unchanged (PBC)."""
        td = _torch_device(device)
        sys = _random_system(seed=5, n_atoms=6, box_len=5.0, device=td)
        shift = torch.tensor([1.3, -0.7, 2.1], dtype=torch.float64, device=td)
        source_feats = pack_charges_dipoles(sys["charges"], sys["dipoles"])
        e_before = multipole_electrostatic_energy(
            sys["positions"],
            source_feats,
            sys["cell"],
            sigma=1.0,
            kspace_cutoff=3.5,
        )
        e_after = multipole_electrostatic_energy(
            sys["positions"] + shift,
            source_feats,
            sys["cell"],
            sigma=1.0,
            kspace_cutoff=3.5,
        )
        # |ρ(k)|² is phase-invariant, so the energy agrees to float64 noise.
        np.testing.assert_allclose(
            float(e_before), float(e_after), rtol=1e-12, atol=1e-13
        )

    def test_zero_moments_give_zero_energy(self, device):
        """All-zero charges and dipoles → ``E = 0`` exactly."""
        td = _torch_device(device)
        sys = _random_system(seed=9, n_atoms=4, box_len=5.0, device=td)
        zero_q = torch.zeros_like(sys["charges"])
        zero_d = torch.zeros_like(sys["dipoles"])
        e = multipole_electrostatic_energy(
            sys["positions"],
            pack_charges_dipoles(zero_q, zero_d),
            sys["cell"],
            sigma=1.0,
            kspace_cutoff=3.5,
        )
        assert float(e) == 0.0


# =============================================================================
# l_max=2 (Path-B direct-k Cartesian-quadrupole energy)
# =============================================================================


def _quadrupole_fixture(seed: int, n_atoms: int, L: float, device: str):
    """Neutral random system with detraced (traceless) symmetric quadrupoles."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, L, size=(n_atoms, 3))
    q = rng.normal(size=n_atoms)
    q -= q.mean()
    mu = rng.normal(size=(n_atoms, 3)) * 0.3
    Qr = rng.normal(size=(n_atoms, 3, 3)) * 0.1
    Q = 0.5 * (Qr + Qr.transpose(0, 2, 1))
    Q -= (np.trace(Q, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
    cell = np.eye(3) * L
    return pos, q, mu, Q, cell


def _pathb_reference_quadrupole(pos, q, mu, Q, cell, sigma, kcut):
    """Path-B l=2 oracle: numpy GTO direct-k reciprocal reduction (envelope
    ``exp(-σ²k²)``, no Ewald damping) minus the analytic GTO self overlap.
    """
    F = float(FIELD_CONSTANT)
    pi32 = math.pi**1.5
    V = float(abs(np.linalg.det(cell)))
    G = 2.0 * np.pi * np.linalg.inv(cell).T
    nmax = int(np.ceil(kcut / np.linalg.norm(G, axis=1).min())) + 1
    ks = [
        a * G[0] + b * G[1] + c * G[2]
        for a in range(-nmax, nmax + 1)
        for b in range(-nmax, nmax + 1)
        for c in range(-nmax, nmax + 1)
    ]
    ks = np.array([k for k in ks if 0.0 < np.linalg.norm(k) <= kcut])
    e_recip = 0.0
    for k in ks:
        k2 = float(k @ k)
        e = np.exp(-1j * (pos @ k))
        rho = (q * e).sum() - 1j * ((mu @ k) * e).sum()
        rho -= 0.5 * (np.einsum("a,nab,b->n", k, Q, k) * e).sum()
        e_recip += (
            (4.0 * math.pi / k2)
            * math.exp(-k2 * sigma**2)
            * (rho * rho.conjugate()).real
        )
    e_recip = (F / (4.0 * math.pi)) * 0.5 / V * e_recip
    e_self = (
        (F / (8.0 * pi32 * sigma)) * (q**2).sum()
        + (F / (48.0 * pi32 * sigma**3)) * (mu**2).sum()
        # Cartesian-Frobenius |Q|_F² self uses denom 320 (the angular
        # ⟨(k̂·Q·k̂)²⟩ = (2/15)|Q|_F² factor 3/2 turns 480 into 320).
        + (F / (320.0 * pi32 * sigma**5)) * (Q**2).sum()
    )
    return e_recip - e_self


class TestQuadrupole:
    """Path-B l_max=2 Cartesian-quadrupole energy."""

    def test_matches_reciprocal_reference(self, device):
        """Full Path-B l=2 energy matches recip-reference − analytic self."""
        td = _torch_device(device)
        sigma, kcut = 0.5, 16.0
        pos, q, mu, Q, cell = _quadrupole_fixture(7, 6, 6.0, td)
        mm = pack_multipole_moments(
            torch.tensor(q, device=td),
            torch.tensor(mu, device=td),
            torch.tensor(Q, device=td),
        )
        E = multipole_electrostatic_energy(
            torch.tensor(pos, device=td),
            mm,
            torch.tensor(cell, device=td),
            sigma=sigma,
            kspace_cutoff=kcut,
        )
        E_ref = _pathb_reference_quadrupole(pos, q, mu, Q, cell, sigma, kcut)
        assert abs(float(E) - E_ref) / abs(E_ref) < 1e-8

    def test_zero_quadrupole_matches_dipole(self, device):
        """Zero Q (l=2 packed) is bit-close to the l=1 packed energy."""
        td = _torch_device(device)
        pos, q, mu, _, cell = _quadrupole_fixture(3, 5, 6.0, td)
        pos_t = torch.tensor(pos, device=td)
        cell_t = torch.tensor(cell, device=td)
        Qz = torch.zeros((pos.shape[0], 3, 3), dtype=torch.float64, device=td)
        kw = dict(sigma=0.5, kspace_cutoff=12.0)
        e_l1 = multipole_electrostatic_energy(
            pos_t,
            pack_multipole_moments(
                torch.tensor(q, device=td), torch.tensor(mu, device=td)
            ),
            cell_t,
            **kw,
        )
        e_l2 = multipole_electrostatic_energy(
            pos_t,
            pack_multipole_moments(
                torch.tensor(q, device=td), torch.tensor(mu, device=td), Qz
            ),
            cell_t,
            **kw,
        )
        assert abs(float(e_l2) - float(e_l1)) < 1e-9 * max(abs(float(e_l1)), 1.0)

    def test_grads_match_fd(self, device):
        """∂E/∂{pos,q,μ,Q} via autograd match central finite differences."""
        td = _torch_device(device)
        sigma, kcut = 0.5, 12.0
        pos, q, mu, Q, cell = _quadrupole_fixture(11, 4, 6.0, td)
        cell_t = torch.tensor(cell, device=td)

        def energy(pos_, q_, mu_, Q_):
            mm = pack_multipole_moments(q_, mu_, Q_)
            return multipole_electrostatic_energy(
                pos_, mm, cell_t, sigma=sigma, kspace_cutoff=kcut
            )

        pos_t = torch.tensor(pos, device=td, requires_grad=True)
        q_t = torch.tensor(q, device=td, requires_grad=True)
        mu_t = torch.tensor(mu, device=td, requires_grad=True)
        Q_t = torch.tensor(Q, device=td, requires_grad=True)
        E = energy(pos_t, q_t, mu_t, Q_t)
        gpos, gq, gmu, gQ = torch.autograd.grad(E, [pos_t, q_t, mu_t, Q_t])
        # ∂E/∂Q must be symmetric (Q enters only via k·Q·k).
        assert (gQ - gQ.transpose(-1, -2)).abs().max() < 1e-12

        h = 1e-6
        base = (
            torch.tensor(pos, device=td),
            torch.tensor(q, device=td),
            torch.tensor(mu, device=td),
            torch.tensor(Q, device=td),
        )

        def fd(idx, shape):
            out = torch.zeros(shape, dtype=torch.float64, device=td)
            flat = out.view(-1)
            b0 = base[idx]
            for i in range(flat.numel()):
                args_p, args_m = list(base), list(base)
                bp = b0.clone()
                bp.view(-1)[i] += h
                bm = b0.clone()
                bm.view(-1)[i] -= h
                args_p[idx], args_m[idx] = bp, bm
                flat[i] = (float(energy(*args_p)) - float(energy(*args_m))) / (2 * h)
            return out

        N = pos.shape[0]
        assert (gpos - fd(0, (N, 3))).abs().max() / gpos.abs().max() < 1e-5
        assert (gq - fd(1, (N,))).abs().max() / gq.abs().max() < 1e-5
        assert (gmu - fd(2, (N, 3))).abs().max() / gmu.abs().max() < 1e-5
        # Symmetric-Q FD: symmetrize the per-component FD before comparing.
        fd_Q = fd(3, (N, 3, 3))
        fd_Q = 0.5 * (fd_Q + fd_Q.transpose(-1, -2))
        assert (gQ - fd_Q).abs().max() / fd_Q.abs().max() < 1e-5
