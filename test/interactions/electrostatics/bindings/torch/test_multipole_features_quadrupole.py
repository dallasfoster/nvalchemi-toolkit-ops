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

r"""l=2 receiver feature tests for the ``feature_max_l=2`` feature extractor."""

from __future__ import annotations

import numpy as np
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_features,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    cartesian_quadrupole_to_e3nn,
    pack_multipole_moments,
    split_packed_for_kernels,
)

_SIGMA = 1.0
_RSIG = [1.0, 1.5]
_KCUT = 4.0
_BOX = 5.0


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _rand_moments(n, seed, td, *, l2=True):
    rng = np.random.default_rng(seed)
    ch = rng.standard_normal(n)
    ch -= ch.mean()
    charges = torch.from_numpy(ch).to(td, torch.float64)
    dip = torch.from_numpy(rng.standard_normal((n, 3))).to(td, torch.float64)
    if not l2:
        return pack_multipole_moments(charges, dip)
    A = rng.standard_normal((n, 3, 3))
    Q = 0.5 * (A + A.transpose(0, 2, 1))
    tr = np.einsum("nii->n", Q) / 3.0
    for d in range(3):
        Q[:, d, d] -= tr
    quad = torch.from_numpy(Q).to(td, torch.float64)
    return pack_multipole_moments(charges, dip, quad)


def _system(n, seed, td, *, l2=True):
    rng = np.random.default_rng(seed + 999)
    pos = torch.from_numpy(rng.uniform(0.0, _BOX, (n, 3))).to(td, torch.float64)
    cell = torch.eye(3, dtype=torch.float64, device=td) * _BOX
    mm = _rand_moments(n, seed, td, l2=l2)
    return pos, cell, mm


def _features(pos, mm, cell, *, feature_max_l):
    return multipole_electrostatic_features(
        pos,
        mm,
        cell,
        sigma=_SIGMA,
        receiver_sigmas=_RSIG,
        kspace_cutoff=_KCUT,
        feature_max_l=feature_max_l,
    )


class TestFeaturesQuadrupole:
    def test_shape(self, device):
        td = _torch_device(device)
        pos, cell, mm = _system(5, 0, td)
        f = _features(pos, mm, cell, feature_max_l=2)
        assert f.shape == (5, len(_RSIG) * 9)
        assert f.dtype == torch.float64

    def test_decoupling_l1_block_unchanged(self, device):
        """feature_max_l=2's l≤1 columns equal the feature_max_l=1 output."""
        td = _torch_device(device)
        pos, cell, mm = _system(6, 1, td)
        f1 = _features(pos, mm, cell, feature_max_l=1)
        f2 = _features(pos, mm, cell, feature_max_l=2)
        nsig = len(_RSIG)
        torch.testing.assert_close(f2[:, : nsig * 4], f1, rtol=0, atol=1e-12)

    def test_l2_block_matches_numpy_projection(self, device):
        """The 5 l=2 channels match an independent numpy projection + self-subtract."""
        td = _torch_device(device)
        pos, cell, mm = _system(5, 2, td)
        nsig = len(_RSIG)
        f2 = _features(pos, mm, cell, feature_max_l=2).detach().cpu().numpy()

        # Reconstruct ρ/V, then project the l=2 receiver block in numpy and lay
        # out per the (σ, m) permuted convention.
        from nvalchemiops.torch.interactions.electrostatics.multipole_autograd import (
            MultipoleRhoFunction,
            MultipoleRhoQFunction,
            _compute_structure_factor_table,
            _l2_receiver_block,
        )
        from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
            prepare_multipole_scf_cache,
        )

        cache = prepare_multipole_scf_cache(
            cell,
            sigma=_SIGMA,
            receiver_sigmas=_RSIG,
            kspace_cutoff=_KCUT,
            l_max=2,
            feature_max_l=2,
        )
        src_l1, quad_cart, _ = split_packed_for_kernels(mm)
        ch = src_l1[:, 0].contiguous()
        dip = src_l1[:, [3, 1, 2]].contiguous()
        rho = MultipoleRhoFunction.apply(
            ch, dip, pos, cache.source_phi_hat, cache.k_vectors, cache
        )
        rho = rho + MultipoleRhoQFunction.apply(
            quad_cart, pos, cache.source_coeff2, cache.k_vectors, cache
        )
        V = (cache.per_k_factor.unsqueeze(-1) * rho).detach().cpu().numpy()

        cos, sin = _compute_structure_factor_table(pos, cache)
        cosn, sinn = cos.detach().cpu().numpy(), sin.detach().cpu().numpy()
        recv2 = _l2_receiver_block(cache).detach().cpu().numpy()
        kfp = cache.k_factor_proj.detach().cpu().numpy()
        inv = 1.0 / (2 * np.pi) ** 3
        a = V[:, None, None, 0] * recv2[..., 0] + V[:, None, None, 1] * recv2[..., 1]
        b = V[:, None, None, 0] * recv2[..., 1] - V[:, None, None, 1] * recv2[..., 0]
        raw_l2 = (
            2.0
            * inv
            * (
                np.einsum("k,ksm,ki->ism", kfp, a, cosn)
                + np.einsum("k,ksm,ki->ism", kfp, b, sinn)
            )
        )
        src_e3nn_l2 = (
            cartesian_quadrupole_to_e3nn(quad_cart.to(torch.float64))
            .detach()
            .cpu()
            .numpy()
        )
        foc2 = cache.feature_overlap_l2.detach().cpu().numpy()
        feat_l2 = raw_l2 - src_e3nn_l2[:, None, :] * foc2[None, :, None]

        off = nsig * 4
        ref = np.empty((pos.shape[0], 5 * nsig))
        for s in range(nsig):
            ref[:, s * 5 : (s + 1) * 5] = feat_l2[:, s, :]
        np.testing.assert_allclose(f2[:, off:], ref, rtol=0, atol=1e-9)

    def test_grads_match_fd(self, device):
        td = _torch_device(device)
        pos, cell, mm = _system(4, 3, td)
        w = torch.sin(
            torch.arange(4 * len(_RSIG) * 9, device=td, dtype=torch.float64)
        ).reshape(4, len(_RSIG) * 9)

        def loss(p, m):
            return (_features(p, m, cell, feature_max_l=2) * w).sum()

        pl = pos.clone().requires_grad_(True)
        ml = mm.clone().requires_grad_(True)
        loss(pl, ml).backward()
        gp, gm = pl.grad.clone(), ml.grad.clone()

        eps = 1e-6
        fdp = torch.zeros_like(gp)
        for i in range(pos.shape[0]):
            for d in range(3):
                pp = pos.clone()
                pp[i, d] += eps
                pmn = pos.clone()
                pmn[i, d] -= eps
                fdp[i, d] = (loss(pp, mm) - loss(pmn, mm)) / (2 * eps)
        torch.testing.assert_close(gp, fdp, rtol=2e-5, atol=1e-7)

        fdm = torch.zeros_like(gm)
        for i in range(pos.shape[0]):
            for c in range(9):
                mp = mm.clone()
                mp[i, c] += eps
                mn = mm.clone()
                mn[i, c] -= eps
                fdm[i, c] = (loss(pos, mp) - loss(pos, mn)) / (2 * eps)
        torch.testing.assert_close(gm, fdm, rtol=2e-5, atol=1e-7)

    def test_batched_matches_single(self, device):
        td = _torch_device(device)
        pos0, cell, mm0 = _system(4, 4, td)
        pos1, _, mm1 = _system(6, 5, td)
        posb = torch.cat([pos0, pos1])
        mmb = torch.cat([mm0, mm1])
        bidx = torch.cat(
            [
                torch.zeros(4, dtype=torch.int32, device=td),
                torch.ones(6, dtype=torch.int32, device=td),
            ]
        )
        cellb = torch.stack([cell, cell])
        fb = multipole_electrostatic_features(
            posb,
            mmb,
            cellb,
            batch_idx=bidx,
            sigma=_SIGMA,
            receiver_sigmas=_RSIG,
            kspace_cutoff=_KCUT,
            feature_max_l=2,
        )
        f0 = _features(pos0, mm0, cell, feature_max_l=2)
        f1 = _features(pos1, mm1, cell, feature_max_l=2)
        torch.testing.assert_close(fb[:4], f0, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(fb[4:], f1, rtol=1e-10, atol=1e-12)

    def test_force_loss_create_graph(self, device):
        """create_graph=True force-loss grad w.r.t. moments must match FD."""
        td = _torch_device(device)
        pos, cell, mm = _system(4, 7, td)
        n = pos.shape[0]
        w = torch.sin(
            torch.arange(n * len(_RSIG) * 9, device=td, dtype=torch.float64)
        ).reshape(n, len(_RSIG) * 9)

        def floss(m):
            p = pos.clone().requires_grad_(True)
            f = _features(p, m, cell, feature_max_l=2)
            (forces,) = torch.autograd.grad((f * w).sum(), p, create_graph=True)
            return (forces**2).sum()

        ml = mm.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(floss(ml), ml)
        eps = 1e-6
        fd = torch.zeros_like(g)
        for i in range(n):
            for c in range(9):
                mp = mm.clone()
                mp[i, c] += eps
                mn = mm.clone()
                mn[i, c] -= eps
                fd[i, c] = (floss(mp).item() - floss(mn).item()) / (2 * eps)
        torch.testing.assert_close(g, fd, rtol=3e-5, atol=1e-6)

    def test_force_loss_create_graph_batched(self, device):
        """Batched analog of :meth:`test_force_loss_create_graph`."""
        td = _torch_device(device)
        pos0, cell, mm0 = _system(3, 8, td)
        pos1, _, mm1 = _system(4, 9, td)
        pos = torch.cat([pos0, pos1])
        mm = torch.cat([mm0, mm1])
        bidx = torch.cat(
            [
                torch.zeros(3, dtype=torch.int32, device=td),
                torch.ones(4, dtype=torch.int32, device=td),
            ]
        )
        cellb = torch.stack([cell, cell])
        n = pos.shape[0]
        w = torch.cos(
            torch.arange(n * len(_RSIG) * 9, device=td, dtype=torch.float64)
        ).reshape(n, len(_RSIG) * 9)

        def floss(m):
            p = pos.clone().requires_grad_(True)
            f = multipole_electrostatic_features(
                p,
                m,
                cellb,
                batch_idx=bidx,
                sigma=_SIGMA,
                receiver_sigmas=_RSIG,
                kspace_cutoff=_KCUT,
                feature_max_l=2,
            )
            (forces,) = torch.autograd.grad((f * w).sum(), p, create_graph=True)
            return (forces**2).sum()

        ml = mm.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(floss(ml), ml)
        eps = 1e-6
        fd = torch.zeros_like(g)
        for i in range(n):
            for c in range(9):
                mp = mm.clone()
                mp[i, c] += eps
                mn = mm.clone()
                mn[i, c] -= eps
                fd[i, c] = (floss(mp).item() - floss(mn).item()) / (2 * eps)
        torch.testing.assert_close(g, fd, rtol=3e-5, atol=1e-6)

    def test_source_l1_with_feature_l2_decoupled(self, device):
        """feature_max_l=2 works with an l≤1 source (no quadrupole)."""
        td = _torch_device(device)
        pos, cell, mm_l1 = _system(5, 6, td, l2=False)
        f = _features(pos, mm_l1, cell, feature_max_l=2)
        assert f.shape == (5, len(_RSIG) * 9)
        off = len(_RSIG) * 4
        assert f[:, off:].abs().max() > 1e-6
