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

r"""Capability + accuracy matrix for the multipole electrostatics entry points.

For every public entry point, covers the training-relevant capability set at
l_max in {0, 1, 2} (packed ``multipole_moments`` of width ``(l_max+1)**2``),
single and batch:

    entry points : direct-k energy | features | Ewald | PME
    capabilities : energy (finite)
                   d/dpositions          (forces / 1st-order)
                   d/dmultipole_moments  (value-loss training)
                   create_graph force-loss (2nd-order)
                   d/dcell               (stress / virial)

Every cell asserts accuracy: analytical autograd vs central finite differences.
Two cross-cutting physics gates supplement the self-consistent FD checks: the
Ewald/PME total must be alpha-independent, and Ewald == PME == direct-k.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    multipole_electrostatic_features,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    multipole_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    multipole_particle_mesh_ewald,
)

# --------------------------------------------------------------------------- #
# Shared parameters / helpers
# --------------------------------------------------------------------------- #

_SIGMA = 0.5
_ALPHA = 0.6
_BOX = 6.0
_RSIG = (1.0, 1.5)
_SIGMA_C = math.sqrt(_SIGMA**2 + 1.0 / (4.0 * _ALPHA**2))
_RCUT = 12.0 * _SIGMA_C  # generous → fixed neighbor list valid under FD
_KCUT = 7.0 / _SIGMA_C
_MESH = (48, 48, 48)
_FD_EPS = 1e-6
_FD_RTOL = 5e-4
_FD_ATOL = 1e-6

ENTRIES = ("directk", "features", "ewald", "pme")
LMAXES = (0, 1, 2)
MODES = ("single", "batch")


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _pack(charges, dipoles, quad, l_max):
    """Packed e3nn ``multipole_moments`` of width ``(l_max+1)**2``."""
    if l_max == 0:
        return charges.unsqueeze(-1).contiguous()
    if l_max == 1:
        return pack_multipole_moments(charges, dipoles)
    return pack_multipole_moments(charges, dipoles, quad)


def _rand_moments(n, l_max, rng, td):
    ch = rng.standard_normal(n)
    ch -= ch.mean()  # neutral (reciprocal q-q needs it)
    charges = torch.tensor(ch, device=td)
    dipoles = torch.tensor(0.4 * rng.standard_normal((n, 3)), device=td)
    A = rng.standard_normal((n, 3, 3))
    quad = torch.tensor(0.5 * (A + A.transpose(0, 2, 1)), device=td)
    return _pack(charges, dipoles, quad, l_max)


def _neigh(pos_np, L, cutoff, atom_offset=0):
    """O(N²·shells) CSR neighbor list for one cubic cell (test-only)."""
    n = pos_np.shape[0]
    shell = int(math.ceil(cutoff / L)) + 1
    idx, counts, shifts = [], [], []
    for i in range(n):
        c = 0
        for a in range(-shell, shell + 1):
            for b in range(-shell, shell + 1):
                for cc in range(-shell, shell + 1):
                    for j in range(n):
                        if i == j and (a, b, cc) == (0, 0, 0):
                            continue
                        d = pos_np[j] - pos_np[i] + np.array([a, b, cc]) * L
                        if np.linalg.norm(d) < cutoff:
                            idx.append(j + atom_offset)
                            shifts.append([a, b, cc])
                            c += 1
        counts.append(c)
    return idx, counts, shifts


def _build_system(mode, l_max, td, seed=0):
    """Return base tensors + (for Ewald/PME) a fixed CSR neighbor list + batch_idx."""
    rng = np.random.default_rng(seed)
    if mode == "single":
        sizes = [3]
    else:
        sizes = [2, 3]
    pos_list, mm_list, cells, bidx = [], [], [], []
    idx, ptr, sh = [], [0], []
    off = 0
    for b, n in enumerate(sizes):
        p = rng.uniform(0.0, _BOX, (n, 3))
        pos_list.append(p)
        cells.append(np.eye(3) * _BOX)
        bidx += [b] * n
        mm_list.append(_rand_moments(n, l_max, rng, td))
        i_b, c_b, s_b = _neigh(p, _BOX, _RCUT, atom_offset=off)
        idx += i_b
        ptr += list(np.cumsum(c_b) + (ptr[-1]))
        sh += s_b
        off += n
    pos = torch.tensor(np.concatenate(pos_list), device=td)
    mm = torch.cat(mm_list)
    out = {
        "pos": pos,
        "mm": mm,
        "idx_j": torch.tensor(idx, dtype=torch.int32, device=td),
        "neighbor_ptr": torch.tensor(ptr, dtype=torch.int32, device=td),
        "unit_shifts": torch.tensor(sh, dtype=torch.int32, device=td).reshape(-1, 3),
    }
    if mode == "single":
        out["cell"] = torch.tensor(cells[0], device=td)
        out["batch_idx"] = None
    else:
        out["cell"] = torch.tensor(np.stack(cells), device=td)
        out["batch_idx"] = torch.tensor(bidx, dtype=torch.int32, device=td)
    return out


def _make_value_fn(entry, mode, l_max, sys):
    """Return ``value(pos, mm, cell) -> scalar`` for one matrix cell.

    Energy entries reduce to ``E.sum()``; the feature tensor is contracted with
    fixed sin-weights so the matrix harness is uniform.
    """
    bidx = sys["batch_idx"]
    idx_j, ptr, sh = sys["idx_j"], sys["neighbor_ptr"], sys["unit_shifts"]
    fml = l_max  # receiver cap matches source for the matrix

    def _feat_weights(f):
        w = torch.sin(
            torch.arange(f.numel(), device=f.device, dtype=torch.float64)
        ).reshape(f.shape)
        return (f * w).sum()

    if entry == "directk":
        if mode == "single":

            def value(pos, mm, cell):
                return multipole_electrostatic_energy(
                    pos, mm, cell, sigma=_SIGMA, kspace_cutoff=_KCUT
                )
        else:

            def value(pos, mm, cell):
                return multipole_electrostatic_energy(
                    pos, mm, cell, batch_idx=bidx, sigma=_SIGMA, kspace_cutoff=_KCUT
                ).sum()
    elif entry == "features":
        if mode == "single":

            def value(pos, mm, cell):
                f = multipole_electrostatic_features(
                    pos,
                    mm,
                    cell,
                    sigma=_SIGMA,
                    receiver_sigmas=list(_RSIG),
                    kspace_cutoff=_KCUT,
                    feature_max_l=fml,
                )
                return _feat_weights(f)
        else:

            def value(pos, mm, cell):
                f = multipole_electrostatic_features(
                    pos,
                    mm,
                    cell,
                    batch_idx=bidx,
                    sigma=_SIGMA,
                    receiver_sigmas=list(_RSIG),
                    kspace_cutoff=_KCUT,
                    feature_max_l=fml,
                )
                return _feat_weights(f)
    elif entry == "ewald":

        def value(pos, mm, cell):
            return multipole_ewald_summation(
                pos,
                mm,
                cell,
                idx_j,
                ptr,
                sh,
                sigma=_SIGMA,
                alpha=_ALPHA,
                kspace_cutoff=_KCUT,
                batch_idx=bidx,
            ).sum()
    elif entry == "pme":

        def value(pos, mm, cell):
            return multipole_particle_mesh_ewald(
                pos,
                mm,
                cell,
                idx_j,
                ptr,
                sh,
                sigma=_SIGMA,
                alpha=_ALPHA,
                mesh_dimensions=_MESH,
                batch_idx=bidx,
            ).sum()
    else:  # pragma: no cover
        raise ValueError(entry)
    return value


def _fd_grad(scalar_fn, x):
    """Central finite-difference gradient of scalar_fn w.r.t. tensor x (float64)."""
    base = x.detach().clone()
    fd = torch.zeros_like(base)
    flat = base.reshape(-1)
    for i in range(flat.numel()):
        xp = base.clone().reshape(-1)
        xp[i] += _FD_EPS
        xm = base.clone().reshape(-1)
        xm[i] -= _FD_EPS
        fp = float(scalar_fn(xp.reshape(base.shape)))
        fm = float(scalar_fn(xm.reshape(base.shape)))
        fd.reshape(-1)[i] = (fp - fm) / (2.0 * _FD_EPS)
    return fd


_FEATURE_STRESS_XFAIL = "features ∂/∂cell create-free 1st-order may be partial; tracked"


# --------------------------------------------------------------------------- #
# The matrix
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("l_max", LMAXES)
class TestCapabilityMatrix:
    """One class instance per (entry, mode, l_max) cell."""

    def test_energy_finite(self, entry, mode, l_max, device):
        td = _torch_device(device)
        sys = _build_system(mode, l_max, td, seed=1)
        value = _make_value_fn(entry, mode, l_max, sys)
        v = value(sys["pos"], sys["mm"], sys["cell"])
        assert torch.isfinite(v).all()
        assert v.ndim == 0  # we reduced everything to a scalar

    def test_grad_positions_fd(self, entry, mode, l_max, device):
        """Forces: ∂value/∂positions vs central FD."""
        td = _torch_device(device)
        sys = _build_system(mode, l_max, td, seed=2)
        value = _make_value_fn(entry, mode, l_max, sys)
        mm, cell = sys["mm"], sys["cell"]
        p = sys["pos"].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(value(p, mm, cell), p)
        fd = _fd_grad(lambda x: value(x, mm, cell), sys["pos"])
        torch.testing.assert_close(g, fd, rtol=_FD_RTOL, atol=_FD_ATOL)

    def test_grad_moments_fd(self, entry, mode, l_max, device):
        """Value-loss training: ∂value/∂multipole_moments vs central FD."""
        td = _torch_device(device)
        sys = _build_system(mode, l_max, td, seed=3)
        value = _make_value_fn(entry, mode, l_max, sys)
        pos, cell = sys["pos"], sys["cell"]
        m = sys["mm"].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(value(pos, m, cell), m)
        fd = _fd_grad(lambda x: value(pos, x, cell), sys["mm"])
        torch.testing.assert_close(g, fd, rtol=_FD_RTOL, atol=_FD_ATOL)

    def test_force_loss_create_graph_fd(self, entry, mode, l_max, device, request):
        """Force-loss training: d(||d value/d pos||^2)/d moments vs FD (2nd-order)."""
        td = _torch_device(device)
        sys = _build_system(mode, l_max, td, seed=4)
        value = _make_value_fn(entry, mode, l_max, sys)
        pos, cell = sys["pos"], sys["cell"]

        def force_loss(m):
            p = pos.clone().requires_grad_(True)
            (forces,) = torch.autograd.grad(value(p, m, cell), p, create_graph=True)
            return (forces**2).sum()

        m = sys["mm"].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(force_loss(m), m)
        fd = _fd_grad(lambda x: float(force_loss(x)), sys["mm"])
        torch.testing.assert_close(g, fd, rtol=_FD_RTOL, atol=1e-5)

    def test_grad_cell_fd(self, entry, mode, l_max, device, request):
        """Stress / virial: d value/d cell vs central FD (fixed topological nlist)."""
        td = _torch_device(device)
        sys = _build_system(mode, l_max, td, seed=5)
        value = _make_value_fn(entry, mode, l_max, sys)
        pos, mm = sys["pos"], sys["mm"]
        c = sys["cell"].clone().requires_grad_(True)
        (g,) = torch.autograd.grad(value(pos, mm, c), c)
        fd = _fd_grad(lambda x: value(pos, mm, x), sys["cell"])
        torch.testing.assert_close(g, fd, rtol=_FD_RTOL, atol=_FD_ATOL)


# --------------------------------------------------------------------------- #
# Cross-cutting physics gates (accuracy beyond self-consistent FD)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("l_max", LMAXES)
class TestCrossMethodPhysics:
    """Ewald/PME/direct-k must agree, and the Ewald total must be α-independent."""

    def _system(self, td, seed=11):
        rng = np.random.default_rng(seed)
        n = 4
        pos_np = rng.uniform(0.0, _BOX, (n, 3))
        return pos_np, rng

    def test_ewald_alpha_independent(self, l_max, device):
        td = _torch_device(device)
        rng = np.random.default_rng(20 + l_max)
        n = 4
        pos_np = rng.uniform(0.0, _BOX, (n, 3))
        pos = torch.tensor(pos_np, device=td)
        cell = torch.tensor(np.eye(3) * _BOX, device=td)
        mm = _rand_moments(n, l_max, rng, td)
        totals = []
        for alpha in (0.4, 0.6, 0.9):
            sc = math.sqrt(_SIGMA**2 + 1.0 / (4.0 * alpha**2))
            idx, cnt, sh = _neigh(pos_np, _BOX, 12.0 * sc)
            ptr = [0] + list(np.cumsum(cnt))
            totals.append(
                float(
                    multipole_ewald_summation(
                        pos,
                        mm,
                        cell,
                        torch.tensor(idx, dtype=torch.int32, device=td),
                        torch.tensor(ptr, dtype=torch.int32, device=td),
                        torch.tensor(sh, dtype=torch.int32, device=td).reshape(-1, 3),
                        sigma=_SIGMA,
                        alpha=alpha,
                        kspace_cutoff=7.0 / sc,
                    )
                )
            )
        assert max(totals) - min(totals) < 1e-4, (
            f"l={l_max} Ewald total α-dependent: {totals}"
        )

    def test_ewald_pme_directk_agree(self, l_max, device):
        td = _torch_device(device)
        rng = np.random.default_rng(30 + l_max)
        n = 4
        pos_np = rng.uniform(0.0, _BOX, (n, 3))
        pos = torch.tensor(pos_np, device=td)
        cell = torch.tensor(np.eye(3) * _BOX, device=td)
        mm = _rand_moments(n, l_max, rng, td)
        idx, cnt, sh = _neigh(pos_np, _BOX, _RCUT)
        ptr = [0] + list(np.cumsum(cnt))
        ij = torch.tensor(idx, dtype=torch.int32, device=td)
        pt = torch.tensor(ptr, dtype=torch.int32, device=td)
        st = torch.tensor(sh, dtype=torch.int32, device=td).reshape(-1, 3)

        e_b = float(
            multipole_electrostatic_energy(
                pos, mm, cell, sigma=_SIGMA, kspace_cutoff=_KCUT
            )
        )
        e_ewald = float(
            multipole_ewald_summation(
                pos,
                mm,
                cell,
                ij,
                pt,
                st,
                sigma=_SIGMA,
                alpha=_ALPHA,
                kspace_cutoff=_KCUT,
            )
        )
        e_pme = float(
            multipole_particle_mesh_ewald(
                pos,
                mm,
                cell,
                ij,
                pt,
                st,
                sigma=_SIGMA,
                alpha=_ALPHA,
                mesh_dimensions=_MESH,
            )
        )

        # Ewald == direct-k to convergence floor; PME == Ewald to MESH accuracy
        # (the l=2 mesh form factor needs a finer grid for tight agreement —
        # 2e-3 is the honest PME-vs-exact-reciprocal tolerance at mesh=48).
        assert abs(e_ewald - e_b) / abs(e_b) < 1e-4, (e_ewald, e_b)
        assert abs(e_pme - e_ewald) / abs(e_ewald) < 2e-3, (e_pme, e_ewald)
