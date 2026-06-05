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

r"""Integration tests for the batched ``multipole_real_space_energy`` path.

Covers the batched real-space multipole Ewald path (reached via
``multipole_real_space_energy(..., batch_idx=)``): forward parity vs a
per-system loop of :func:`multipole_real_space_energy`, first-order backward
parity on ``(positions, charges, dipoles)``, and second-order
(``create_graph=True`` force-loss) parity for l_max=0 and l_max=1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_real_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_charges_dipoles,
)


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _rand_system(n: int, box: float, device: str, seed: int):
    rng = np.random.default_rng(seed)
    td = _torch_device(device)
    pos = torch.from_numpy(rng.uniform(0, box, (n, 3))).to(td, torch.float64)
    chg_np = rng.uniform(-1, 1, n)
    chg_np -= chg_np.mean()
    chg = torch.from_numpy(chg_np).to(td, torch.float64)
    cell = torch.eye(3, dtype=torch.float64, device=td) * box
    idx_j_l, nptr_l, sh_l = [], [0], []
    for i in range(n):
        for j in range(n):
            if j != i:
                idx_j_l.append(j)
                sh_l.append([0, 0, 0])
        nptr_l.append(len(idx_j_l))
    return (
        pos,
        chg,
        cell,
        torch.tensor(idx_j_l, dtype=torch.int32, device=td),
        torch.tensor(nptr_l, dtype=torch.int32, device=td),
        torch.tensor(sh_l, dtype=torch.int32, device=td),
    )


def _flatten_batch(systems):
    """Stitch a list of per-system fixtures into the flat batched form."""
    n_per = [s[0].shape[0] for s in systems]
    pos = torch.cat([s[0] for s in systems])
    chg = torch.cat([s[1] for s in systems])
    cells = torch.stack([s[2] for s in systems])
    idx_j_flat_l, nptr_flat_l, sh_flat_l = [], [0], []
    atom_off = 0
    for s in systems:
        idx_j_flat_l.append(s[3] + atom_off)
        sh_flat_l.append(s[5])
        nptr_np = s[4].cpu().numpy()
        for k in range(1, len(nptr_np)):
            nptr_flat_l.append(nptr_flat_l[-1] + int(nptr_np[k] - nptr_np[k - 1]))
        atom_off += s[0].shape[0]
    idx_j_flat = torch.cat(idx_j_flat_l)
    nptr_flat = torch.tensor(nptr_flat_l, dtype=torch.int32, device=pos.device)
    sh_flat = torch.cat(sh_flat_l)
    bi = torch.cat(
        [
            torch.full((n_per[i],), i, dtype=torch.int32, device=pos.device)
            for i in range(len(systems))
        ]
    )
    return pos, chg, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per


class TestBatchedMonopoleForward:
    def test_forward_bit_parity_vs_per_system(self, device):
        systems = [
            _rand_system(5, 4.0, device, 0),
            _rand_system(4, 5.0, device, 1),
            _rand_system(6, 3.8, device, 2),
        ]
        alphas_np = [0.3, 0.4, 0.5]
        td = _torch_device(device)
        alphas = torch.tensor(alphas_np, dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_e = []
        for i, s in enumerate(systems):
            pos, chg, cell, idx_j, nptr, sh = s
            a = alphas[i : i + 1]
            per_e.append(
                multipole_real_space_energy(
                    pos,
                    pack_charges_dipoles(chg, None),
                    cell,
                    idx_j,
                    nptr,
                    sh,
                    sigmas[i : i + 1],
                    a,
                )
            )

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        e_batch = multipole_real_space_energy(
            pos_all,
            pack_charges_dipoles(chg_all, None),
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )

        assert e_batch.shape == (sum(n_per),)
        assert e_batch.dtype == torch.float64
        off = 0
        for i, n in enumerate(n_per):
            torch.testing.assert_close(e_batch[off : off + n], per_e[i], rtol=0, atol=0)
            off += n


class TestBatchedMonopoleBackward:
    def test_backward_bit_parity_vs_per_system(self, device):
        systems = [_rand_system(5, 4.0, device, 100), _rand_system(4, 5.0, device, 101)]
        td = _torch_device(device)
        alphas = torch.tensor([0.3, 0.4], dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_gp, per_gc = [], []
        for i, s in enumerate(systems):
            pos_, chg_, cell, idx_j, nptr, sh = s
            p = pos_.detach().clone().requires_grad_(True)
            sf_ = pack_charges_dipoles(chg_.detach().clone(), None).requires_grad_(True)
            e = multipole_real_space_energy(
                p, sf_, cell, idx_j, nptr, sh, sigmas[i : i + 1], alphas[i : i + 1]
            )
            e.sum().backward()
            per_gp.append(p.grad)
            per_gc.append(sf_.grad[..., 0])

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        pos_all = pos_all.requires_grad_(True)
        sf_all = pack_charges_dipoles(chg_all, None).requires_grad_(True)
        e_batch = multipole_real_space_energy(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )
        e_batch.sum().backward()

        off = 0
        for i, n in enumerate(n_per):
            torch.testing.assert_close(
                pos_all.grad[off : off + n], per_gp[i], rtol=0, atol=0
            )
            torch.testing.assert_close(
                sf_all.grad[off : off + n, 0], per_gc[i], rtol=0, atol=0
            )
            off += n


def _rand_dipoles(n: int, device: str, seed: int) -> torch.Tensor:
    td = _torch_device(device)
    rng = np.random.default_rng(seed + 1000)
    return torch.from_numpy(0.3 * rng.standard_normal((n, 3))).to(td, torch.float64)


class TestBatchedDipoleForward:
    """Batched l_max=1 (charges + dipoles) forward parity."""

    def test_forward_bit_parity_vs_per_system(self, device):
        systems = [
            _rand_system(5, 4.0, device, 100),
            _rand_system(4, 5.0, device, 101),
            _rand_system(6, 3.8, device, 102),
        ]
        dipoles_per = [
            _rand_dipoles(s[0].shape[0], device, seed=i) for i, s in enumerate(systems)
        ]
        td = _torch_device(device)
        alphas = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_e = []
        for i, s in enumerate(systems):
            pos, chg, cell, idx_j, nptr, sh = s
            dip = dipoles_per[i]
            per_e.append(
                multipole_real_space_energy(
                    pos,
                    pack_charges_dipoles(chg, dip),
                    cell,
                    idx_j,
                    nptr,
                    sh,
                    sigmas[i : i + 1],
                    alphas[i : i + 1],
                )
            )

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        dip_all = torch.cat(dipoles_per)
        e_batch = multipole_real_space_energy(
            pos_all,
            pack_charges_dipoles(chg_all, dip_all),
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )
        assert e_batch.shape == (sum(n_per),)
        off = 0
        for i, n in enumerate(n_per):
            torch.testing.assert_close(e_batch[off : off + n], per_e[i], rtol=0, atol=0)
            off += n


class TestBatchedDipoleBackward:
    """Batched l_max=1 first-order backward parity on positions/charges/dipoles."""

    def test_backward_bit_parity_vs_per_system(self, device):
        systems = [
            _rand_system(5, 4.0, device, 200),
            _rand_system(4, 5.0, device, 201),
        ]
        dipoles_per = [
            _rand_dipoles(s[0].shape[0], device, seed=i) for i, s in enumerate(systems)
        ]
        td = _torch_device(device)
        alphas = torch.tensor([0.3, 0.4], dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_gp, per_gc, per_gd = [], [], []
        for i, s in enumerate(systems):
            pos_, chg_, cell, idx_j, nptr, sh = s
            p = pos_.detach().clone().requires_grad_(True)
            sf_ = pack_charges_dipoles(
                chg_.detach().clone(), dipoles_per[i].detach().clone()
            ).requires_grad_(True)
            e = multipole_real_space_energy(
                p, sf_, cell, idx_j, nptr, sh, sigmas[i : i + 1], alphas[i : i + 1]
            )
            e.sum().backward()
            per_gp.append(p.grad)
            per_gc.append(sf_.grad[..., 0])
            per_gd.append(sf_.grad[..., [3, 1, 2]])

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        dip_all = torch.cat(dipoles_per)
        pos_all = pos_all.requires_grad_(True)
        sf_all = pack_charges_dipoles(chg_all, dip_all).requires_grad_(True)
        e_batch = multipole_real_space_energy(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )
        e_batch.sum().backward()

        chg_all_grad = sf_all.grad[..., 0]
        dip_all_grad = sf_all.grad[..., [3, 1, 2]]

        off = 0
        for i, n in enumerate(n_per):
            # Backward uses atomic_add, so ordering can introduce ~float64 ULP
            # differences. Use a tight but non-zero tolerance.
            torch.testing.assert_close(
                pos_all.grad[off : off + n], per_gp[i], rtol=0, atol=1e-14
            )
            torch.testing.assert_close(
                chg_all_grad[off : off + n], per_gc[i], rtol=0, atol=1e-14
            )
            torch.testing.assert_close(
                dip_all_grad[off : off + n], per_gd[i], rtol=0, atol=1e-14
            )
            off += n

    def test_force_loss_parity_vs_per_system(self, device):
        """Batched l_max=1 double-backward matches per-system force-loss gradients."""
        systems = [
            _rand_system(5, 4.0, device, 300),
            _rand_system(4, 5.0, device, 301),
        ]
        dipoles_per = [
            _rand_dipoles(s[0].shape[0], device, seed=i) for i, s in enumerate(systems)
        ]
        td = _torch_device(device)
        alphas = torch.tensor([0.3, 0.4], dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_gp, per_gc, per_gd = [], [], []
        for i, s in enumerate(systems):
            pos_, chg_, cell, idx_j, nptr, sh = s
            p = pos_.detach().clone().requires_grad_(True)
            sf_ = pack_charges_dipoles(
                chg_.detach().clone(), dipoles_per[i].detach().clone()
            ).requires_grad_(True)
            e = multipole_real_space_energy(
                p, sf_, cell, idx_j, nptr, sh, sigmas[i : i + 1], alphas[i : i + 1]
            )
            (forces_neg,) = torch.autograd.grad(e.sum(), p, create_graph=True)
            (forces_neg**2).sum().backward()
            per_gp.append(p.grad)
            per_gc.append(sf_.grad[..., 0])
            per_gd.append(sf_.grad[..., [3, 1, 2]])

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        dip_all = torch.cat(dipoles_per)
        pos_all = pos_all.detach().clone().requires_grad_(True)
        sf_all = (
            pack_charges_dipoles(chg_all, dip_all).detach().clone().requires_grad_(True)
        )
        e_batch = multipole_real_space_energy(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )
        (forces_neg_batch,) = torch.autograd.grad(
            e_batch.sum(), pos_all, create_graph=True
        )
        (forces_neg_batch**2).sum().backward()

        chg_all_grad = sf_all.grad[..., 0]
        dip_all_grad = sf_all.grad[..., [3, 1, 2]]

        off = 0
        for i, n in enumerate(n_per):
            torch.testing.assert_close(
                pos_all.grad[off : off + n], per_gp[i], rtol=0, atol=1e-12
            )
            torch.testing.assert_close(
                chg_all_grad[off : off + n], per_gc[i], rtol=0, atol=1e-12
            )
            torch.testing.assert_close(
                dip_all_grad[off : off + n], per_gd[i], rtol=0, atol=1e-12
            )
            off += n


class TestBatchedMonopoleValidation:
    def test_bad_source_feats_shape(self, device):
        s = _rand_system(4, 4.0, device, 0)
        pos, chg, cell, idx_j, nptr, sh = s
        td = _torch_device(device)
        cells = cell.unsqueeze(0)
        alphas = torch.tensor([0.3], dtype=torch.float64, device=td)
        sigmas = torch.tensor([1.0], dtype=torch.float64, device=td)
        bi = torch.zeros(pos.shape[0], dtype=torch.int32, device=td)
        # Wrong N (3 vs 4 positions), still a valid (N, 1) trailing dim.
        bad_sf = torch.zeros(3, 1, dtype=torch.float64, device=td)
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_real_space_energy(
                pos, bad_sf, cells, idx_j, nptr, sh, sigmas, alphas, batch_idx=bi
            )

    def test_force_loss_parity_vs_per_system(self, device):
        """Batched l_max=0 double-backward matches per-system force-loss gradients."""
        systems = [
            _rand_system(5, 4.0, device, 400),
            _rand_system(4, 5.0, device, 401),
        ]
        td = _torch_device(device)
        alphas = torch.tensor([0.3, 0.4], dtype=torch.float64, device=td)
        sigmas = torch.full_like(alphas, 1.0)

        per_gp, per_gc = [], []
        for i, s in enumerate(systems):
            pos_, chg_, cell, idx_j, nptr, sh = s
            p = pos_.detach().clone().requires_grad_(True)
            sf_ = pack_charges_dipoles(chg_.detach().clone(), None).requires_grad_(True)
            e = multipole_real_space_energy(
                p, sf_, cell, idx_j, nptr, sh, sigmas[i : i + 1], alphas[i : i + 1]
            )
            (forces_neg,) = torch.autograd.grad(e.sum(), p, create_graph=True)
            (forces_neg**2).sum().backward()
            per_gp.append(p.grad)
            per_gc.append(sf_.grad[..., 0])

        pos_all, chg_all, cells, idx_j_flat, nptr_flat, sh_flat, bi, n_per = (
            _flatten_batch(systems)
        )
        pos_all = pos_all.detach().clone().requires_grad_(True)
        sf_all = (
            pack_charges_dipoles(chg_all, None).detach().clone().requires_grad_(True)
        )
        e_batch = multipole_real_space_energy(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigmas,
            alphas,
            batch_idx=bi,
        )
        (forces_neg_batch,) = torch.autograd.grad(
            e_batch.sum(), pos_all, create_graph=True
        )
        (forces_neg_batch**2).sum().backward()

        chg_all_grad = sf_all.grad[..., 0]
        off = 0
        for i, n in enumerate(n_per):
            torch.testing.assert_close(
                pos_all.grad[off : off + n], per_gp[i], rtol=0, atol=1e-12
            )
            torch.testing.assert_close(
                chg_all_grad[off : off + n], per_gc[i], rtol=0, atol=1e-12
            )
            off += n
