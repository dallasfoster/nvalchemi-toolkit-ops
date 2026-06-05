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

r"""Path A ≡ Path B parity for multipole Ewald.

``multipole_ewald_summation`` (GTO-Ewald real-space + damped reciprocal sum +
analytical self-energy) must equal ``multipole_electrostatic_energy`` (direct
k-space, Path B) to within the accumulated ``wp_erfc`` floor (~1.5e-7 per pair).
Tolerances scale with N_pair × 1e-7: ``|Δ| ≤ 5e-4`` for BCC supercells,
``≤ 1e-6`` for 2-atom cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    multipole_ewald_scf_step_energy,
    multipole_ewald_summation,
    prepare_multipole_scf_cache,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_charges_dipoles,
)


def _bcc(n: int, a: float = 4.14):
    """Alternating-charge BCC supercell (neutral, l_max=0 capable)."""
    p = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p.append((i * a, j * a, k * a))
                p.append(((i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a))
    return np.array(p), np.eye(3) * (n * a)


def _neigh(positions: np.ndarray, L: float, cutoff: float):
    """O(N² · shells) neighbor list covering all periodic images within ``cutoff``.

    Test-only; production should use the real neighbor-list builder. Test
    systems are tiny (~16 atoms) so the O(N²) overhead is negligible.
    """
    N = positions.shape[0]
    shell = int(math.ceil(cutoff / L)) + 1
    idx_j, nptr, shifts = [], [0], []
    for i in range(N):
        for sa in range(-shell, shell + 1):
            for sb in range(-shell, shell + 1):
                for sc in range(-shell, shell + 1):
                    for j in range(N):
                        if j == i and (sa, sb, sc) == (0, 0, 0):
                            continue
                        r = positions[j] - positions[i] + np.array([sa, sb, sc]) * L
                        if np.linalg.norm(r) < cutoff:
                            idx_j.append(j)
                            shifts.append([sa, sb, sc])
        nptr.append(len(idx_j))
    return (
        np.array(idx_j, np.int32),
        np.array(nptr, np.int32),
        np.array(shifts, np.int32),
    )


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _build(n: int, device: str, seed: int, l_max: int):
    """BCC system with alternating charges (and optional random dipoles)."""
    rng = np.random.default_rng(seed)
    pos_np, cell_np = _bcc(n)
    N = pos_np.shape[0]
    chg_np = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    if abs(chg_np.sum()) > 1e-12:
        chg_np[-1] -= chg_np.sum()
    dip_np = 0.3 * rng.standard_normal((N, 3)) if l_max >= 1 else None

    td = _torch_device(device)
    pos = torch.from_numpy(pos_np).to(td, torch.float64)
    chg = torch.from_numpy(chg_np).to(td, torch.float64)
    dip = torch.from_numpy(dip_np).to(td, torch.float64) if dip_np is not None else None
    cell = torch.from_numpy(cell_np).to(td, torch.float64)
    source_feats = pack_charges_dipoles(chg, dip)
    return pos, source_feats, cell, cell_np, pos_np


def _path_a_vs_b(
    device: str, n: int, sigma: float, alpha: float, l_max: int, seed: int
) -> float:
    pos, source_feats, cell, cell_np, pos_np = _build(n, device, seed, l_max)
    L = cell_np[0, 0]
    td = _torch_device(device)

    # Real-space cutoff governed by σ_c; 10·σ_c gives ULP-level parity.
    sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
    cutoff = 10.0 * sigma_c
    idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
    idx_j = torch.from_numpy(idx_j_np).to(td)
    nptr = torch.from_numpy(nptr_np).to(td)
    sh = torch.from_numpy(sh_np).to(td)

    # k·σ_c = 6 ⇒ Gaussian damping exp(-36) ~ 1e-16 at cutoff.
    kcut = 6.0 / sigma_c

    E_B = float(
        multipole_electrostatic_energy(
            pos, source_feats, cell, sigma=sigma, kspace_cutoff=kcut
        )
    )
    E_A = float(
        multipole_ewald_summation(
            pos,
            source_feats,
            cell,
            idx_j,
            nptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            kspace_cutoff=kcut,
        )
    )
    return E_A - E_B


class TestPathAEquivPathB:
    """Path A (real + reciprocal − self) must equal Path B (direct k-space)."""

    @pytest.mark.parametrize("alpha", [0.3, 0.4, 0.6, 0.9])
    @pytest.mark.parametrize("sigma", [0.8, 1.0, 1.2])
    def test_monopole_bcc(self, device, sigma: float, alpha: float):
        """l_max=0 BCC supercell: |Δ| bounded by accumulated wp_erfc error."""
        delta = _path_a_vs_b(device, n=2, sigma=sigma, alpha=alpha, l_max=0, seed=41)
        assert abs(delta) < 5e-4, f"σ={sigma}  α={alpha}  Δ={delta:.3e}"

    @pytest.mark.parametrize("alpha", [0.3, 0.4, 0.6, 0.9])
    @pytest.mark.parametrize("sigma", [0.8, 1.0, 1.2])
    def test_dipole_bcc(self, device, sigma: float, alpha: float):
        """l_max=1 BCC supercell: dipole + charge cross terms."""
        delta = _path_a_vs_b(device, n=2, sigma=sigma, alpha=alpha, l_max=1, seed=47)
        assert abs(delta) < 5e-4, f"σ={sigma}  α={alpha}  Δ={delta:.3e}"

    @pytest.mark.parametrize("alpha", [0.5, 1.0])
    def test_two_atom_sigma1(self, device, alpha: float):
        """2-atom (+1, −1) at separation 3 in large box — tighter tolerance."""
        td = _torch_device(device)
        L = 30.0
        pos_np = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        cell_np = np.eye(3) * L
        chg = torch.tensor([1.0, -1.0], dtype=torch.float64, device=td)
        sf = pack_charges_dipoles(chg, None)
        pos = torch.from_numpy(pos_np).to(td, torch.float64)
        cell = torch.from_numpy(cell_np).to(td, torch.float64)
        sigma = 1.0
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
        idx_j = torch.from_numpy(idx_j_np).to(td)
        nptr = torch.from_numpy(nptr_np).to(td)
        sh = torch.from_numpy(sh_np).to(td)
        kcut = 6.0 / sigma_c
        E_B = float(
            multipole_electrostatic_energy(
                pos, sf, cell, sigma=sigma, kspace_cutoff=kcut
            )
        )
        E_A = float(
            multipole_ewald_summation(
                pos,
                sf,
                cell,
                idx_j,
                nptr,
                sh,
                sigma=sigma,
                alpha=alpha,
                kspace_cutoff=kcut,
            )
        )
        assert abs(E_A - E_B) < 1e-4, (
            f"α={alpha}  E_A={E_A:.6f}  E_B={E_B:.6f}  Δ={E_A - E_B:.3e}"
        )

    def test_alpha_limit_recovers_pathb(self, device):
        """Large α: all energy in real-space, should still match Path B."""
        delta = _path_a_vs_b(device, n=2, sigma=1.0, alpha=2.0, l_max=1, seed=53)
        assert abs(delta) < 1e-4, f"α=2.0 Δ={delta:.3e}"


class TestBatchedEwaldSummation:
    """Batched ``multipole_ewald_summation`` (``batch_idx`` ≠ None) must equal
    a per-system loop of the single-system variant to the same precision as
    forward / backward bit-parity tests — i.e. zero drift because each thread
    still owns a unique ``atom_i`` in the batched kernels.
    """

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_batch_matches_per_system_loop(self, device, l_max: int):
        """Stack B identical systems and verify E_A_batch[b] ≈ E_A_single per b."""
        td = _torch_device(device)
        sigma, alpha = 1.0, 0.6
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        # Distinct seeds exercise the batch_idx dispatch, not replicated copies.
        systems = []
        for seed in (41, 47, 53):
            pos, sf, cell, cell_np, pos_np = _build(
                n=2, device=device, seed=seed, l_max=l_max
            )
            L = cell_np[0, 0]
            idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
            idx_j = torch.from_numpy(idx_j_np).to(td)
            nptr = torch.from_numpy(nptr_np).to(td)
            sh = torch.from_numpy(sh_np).to(td)
            systems.append(
                (
                    pos,
                    sf,
                    cell,
                    idx_j,
                    nptr,
                    sh,
                    cell_np,
                    pos_np,
                    idx_j_np,
                    nptr_np,
                    sh_np,
                )
            )

        # Per-system reference.
        per_system_e = []
        for sys_tup in systems:
            pos, sf, cell, idx_j, nptr, sh = sys_tup[:6]
            e = float(
                multipole_ewald_summation(
                    pos,
                    sf,
                    cell,
                    idx_j,
                    nptr,
                    sh,
                    sigma=sigma,
                    alpha=alpha,
                    kspace_cutoff=kcut,
                )
            )
            per_system_e.append(e)

        # Stitch a batched call.
        pos_all = torch.cat([s[0] for s in systems], dim=0)
        sf_all = torch.cat([s[1] for s in systems], dim=0)
        cells = torch.stack(
            [s[2].squeeze(0) if s[2].ndim == 3 else s[2] for s in systems], dim=0
        )
        n_per = [s[0].shape[0] for s in systems]
        batch_idx = torch.cat(
            [
                torch.full((n,), b, dtype=torch.int32, device=td)
                for b, n in enumerate(n_per)
            ]
        )
        # Flat CSR: offset idx_j by cumulative atom count per system; stitch nptr.
        idx_j_flat = []
        nptr_flat = [0]
        sh_flat = []
        atom_off = 0
        for s in systems:
            idx_j_flat.append(s[8] + atom_off)
            sh_flat.append(s[10])
            nptr_np = s[9]
            for k in range(1, len(nptr_np)):
                nptr_flat.append(nptr_flat[-1] + int(nptr_np[k] - nptr_np[k - 1]))
            atom_off += s[0].shape[0]
        idx_j_flat = torch.from_numpy(np.concatenate(idx_j_flat).astype(np.int32)).to(
            td
        )
        nptr_flat = torch.from_numpy(np.asarray(nptr_flat, dtype=np.int32)).to(td)
        sh_flat = torch.from_numpy(np.concatenate(sh_flat).astype(np.int32)).to(td)

        e_batch = multipole_ewald_summation(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigma=sigma,
            alpha=alpha,
            kspace_cutoff=kcut,
            batch_idx=batch_idx,
        )
        assert e_batch.shape == (len(systems),)
        for b, e_ref in enumerate(per_system_e):
            assert abs(float(e_batch[b]) - e_ref) < 1e-10, (
                f"sys {b} l_max={l_max}: batch={float(e_batch[b]):.6e} "
                f"single={e_ref:.6e}"
            )

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_batch_matches_path_b(self, device, l_max: int):
        """Batched Path A ≡ Path B (direct k-space, per-system loop)."""
        td = _torch_device(device)
        sigma, alpha = 1.0, 0.6
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        # Two distinct systems.
        systems = []
        per_b = []
        for seed in (71, 73):
            pos, sf, cell, cell_np, pos_np = _build(
                n=2, device=device, seed=seed, l_max=l_max
            )
            per_b.append(
                float(
                    multipole_electrostatic_energy(
                        pos, sf, cell, sigma=sigma, kspace_cutoff=kcut
                    )
                )
            )
            L = cell_np[0, 0]
            idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
            systems.append((pos, sf, cell, cell_np, pos_np, idx_j_np, nptr_np, sh_np))

        pos_all = torch.cat([s[0] for s in systems], dim=0)
        sf_all = torch.cat([s[1] for s in systems], dim=0)
        cells = torch.stack(
            [s[2].squeeze(0) if s[2].ndim == 3 else s[2] for s in systems], dim=0
        )
        n_per = [s[0].shape[0] for s in systems]
        batch_idx = torch.cat(
            [
                torch.full((n,), b, dtype=torch.int32, device=td)
                for b, n in enumerate(n_per)
            ]
        )
        idx_j_flat = []
        nptr_flat = [0]
        sh_flat = []
        atom_off = 0
        for s in systems:
            idx_j_flat.append(s[5] + atom_off)
            sh_flat.append(s[7])
            nptr_np = s[6]
            for k in range(1, len(nptr_np)):
                nptr_flat.append(nptr_flat[-1] + int(nptr_np[k] - nptr_np[k - 1]))
            atom_off += s[0].shape[0]
        idx_j_flat = torch.from_numpy(np.concatenate(idx_j_flat).astype(np.int32)).to(
            td
        )
        nptr_flat = torch.from_numpy(np.asarray(nptr_flat, dtype=np.int32)).to(td)
        sh_flat = torch.from_numpy(np.concatenate(sh_flat).astype(np.int32)).to(td)

        e_batch = multipole_ewald_summation(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigma=sigma,
            alpha=alpha,
            kspace_cutoff=kcut,
            batch_idx=batch_idx,
        )
        for b, e_b_ref in enumerate(per_b):
            assert abs(float(e_batch[b]) - e_b_ref) < 5e-4, (
                f"sys {b} l_max={l_max}: A_batch={float(e_batch[b]):.6e} "
                f"B={e_b_ref:.6e}"
            )


class TestEwaldSCFStepEnergy:
    """Cache-aware ``multipole_ewald_scf_step_energy`` must match the
    one-shot ``multipole_ewald_summation`` bit-for-bit when the cache is
    built with matching (σ, α, kspace_cutoff)."""

    @pytest.mark.parametrize("alpha", [0.4, 0.9])
    @pytest.mark.parametrize("l_max", [0, 1])
    def test_single_matches_one_shot(self, device, l_max: int, alpha: float):
        td = _torch_device(device)
        sigma = 1.0
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        pos, sf, cell, cell_np, pos_np = _build(
            n=2, device=device, seed=11, l_max=l_max
        )
        L = cell_np[0, 0]
        idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
        idx_j = torch.from_numpy(idx_j_np).to(td)
        nptr = torch.from_numpy(nptr_np).to(td)
        sh = torch.from_numpy(sh_np).to(td)

        E_one_shot = float(
            multipole_ewald_summation(
                pos,
                sf,
                cell,
                idx_j,
                nptr,
                sh,
                sigma=sigma,
                alpha=alpha,
                kspace_cutoff=kcut,
            )
        )

        cache = prepare_multipole_scf_cache(
            cell.squeeze(0) if cell.ndim == 3 else cell,
            sigma=sigma,
            alpha=alpha,
            receiver_sigmas=[sigma],
            kspace_cutoff=kcut,
            l_max=l_max,
            device=pos.device,
        )
        E_cached = float(
            multipole_ewald_scf_step_energy(cache, pos, sf, idx_j, nptr, sh)
        )
        assert abs(E_cached - E_one_shot) < 1e-10, (
            f"l_max={l_max} α={alpha}: cached={E_cached:.6e} "
            f"one-shot={E_one_shot:.6e}  |Δ|={abs(E_cached - E_one_shot):.3e}"
        )

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_batched_matches_one_shot(self, device, l_max: int):
        td = _torch_device(device)
        sigma, alpha = 1.0, 0.6
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        systems = []
        for seed in (31, 37, 41):
            pos, sf, cell, cell_np, pos_np = _build(
                n=2, device=device, seed=seed, l_max=l_max
            )
            L = cell_np[0, 0]
            idx_j_np, nptr_np, sh_np = _neigh(pos_np, L, cutoff)
            systems.append((pos, sf, cell, cell_np, pos_np, idx_j_np, nptr_np, sh_np))

        # Stitch flat tensors.
        pos_all = torch.cat([s[0] for s in systems], dim=0)
        sf_all = torch.cat([s[1] for s in systems], dim=0)
        cells = torch.stack(
            [s[2].squeeze(0) if s[2].ndim == 3 else s[2] for s in systems], dim=0
        )
        n_per = [s[0].shape[0] for s in systems]
        batch_idx = torch.cat(
            [
                torch.full((n,), b, dtype=torch.int32, device=td)
                for b, n in enumerate(n_per)
            ]
        )
        idx_j_flat, nptr_flat, sh_flat = [], [0], []
        atom_off = 0
        for s in systems:
            idx_j_flat.append(s[5] + atom_off)
            sh_flat.append(s[7])
            nptr_np = s[6]
            for k in range(1, len(nptr_np)):
                nptr_flat.append(nptr_flat[-1] + int(nptr_np[k] - nptr_np[k - 1]))
            atom_off += s[0].shape[0]
        idx_j_flat = torch.from_numpy(np.concatenate(idx_j_flat).astype(np.int32)).to(
            td
        )
        nptr_flat = torch.from_numpy(np.asarray(nptr_flat, dtype=np.int32)).to(td)
        sh_flat = torch.from_numpy(np.concatenate(sh_flat).astype(np.int32)).to(td)

        E_one_shot = multipole_ewald_summation(
            pos_all,
            sf_all,
            cells,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            sigma=sigma,
            alpha=alpha,
            kspace_cutoff=kcut,
            batch_idx=batch_idx,
        )

        batch_cache = prepare_multipole_scf_cache(
            cells,
            sigma=sigma,
            alpha=alpha,
            receiver_sigmas=[sigma],
            kspace_cutoff=kcut,
            l_max=l_max,
            device=pos_all.device,
        )
        E_cached = multipole_ewald_scf_step_energy(
            batch_cache,
            pos_all,
            sf_all,
            idx_j_flat,
            nptr_flat,
            sh_flat,
            batch_idx=batch_idx,
        )
        torch.testing.assert_close(E_cached, E_one_shot, rtol=0, atol=1e-10)

    def test_path_b_cache_rejected(self, device):
        """Passing a Path B cache (``alpha=None``) should raise."""
        td = _torch_device(device)
        pos, sf, cell, cell_np, pos_np = _build(n=2, device=device, seed=1, l_max=0)
        idx_j_np, nptr_np, sh_np = _neigh(pos_np, cell_np[0, 0], 5.0)
        idx_j = torch.from_numpy(idx_j_np).to(td)
        nptr = torch.from_numpy(nptr_np).to(td)
        sh = torch.from_numpy(sh_np).to(td)
        # Path B cache: no alpha.
        cache_b = prepare_multipole_scf_cache(
            cell.squeeze(0) if cell.ndim == 3 else cell,
            sigma=1.0,
            receiver_sigmas=[1.0],
            kspace_cutoff=3.0,
            l_max=0,
            device=pos.device,
        )
        with pytest.raises(ValueError, match="requires an Ewald cache"):
            multipole_ewald_scf_step_energy(cache_b, pos, sf, idx_j, nptr, sh)


class TestCudaCpuTileRouting:
    r"""CUDA ``multipole_ewald_summation`` (tile real-space kernel) and CPU
    (CSR fallback) must produce the same energy and gradients on the same
    (σ, α, geometry) input, bit-for-bit at float64 modulo atomic-add ordering.
    """

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_forward_cuda_matches_cpu(self, l_max):
        """Same system, tile kernel on CUDA ≡ CSR kernel on CPU at 1e-10 rel."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA for the tile path")
        sigma, alpha = 1.0, 0.6
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        out: dict[str, float] = {}
        for device in ("cpu", "gpu"):
            pos, sf, cell, cell_np, pos_np = _build(
                n=2, device=device, seed=7, l_max=l_max
            )
            td = _torch_device(device)
            idx_j_np, nptr_np, sh_np = _neigh(pos_np, cell_np[0, 0], cutoff)
            idx_j = torch.from_numpy(idx_j_np).to(td)
            nptr = torch.from_numpy(nptr_np).to(td)
            sh = torch.from_numpy(sh_np).to(td)
            out[device] = float(
                multipole_ewald_summation(
                    pos,
                    sf,
                    cell,
                    idx_j,
                    nptr,
                    sh,
                    sigma=sigma,
                    alpha=alpha,
                    kspace_cutoff=kcut,
                )
            )
        assert abs(out["cpu"] - out["gpu"]) / max(abs(out["cpu"]), 1e-300) < 1e-10, (
            f"CUDA tile path ({out['gpu']:.15e}) disagrees with CPU CSR path "
            f"({out['cpu']:.15e}) — tile routing broke the energy invariant."
        )

    @pytest.mark.parametrize("l_max", [0, 1])
    def test_backward_cuda_matches_cpu(self, l_max):
        """Gradients from tile backward (CUDA) ≡ CSR backward (CPU) at 1e-10 rel."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA for the tile path")
        sigma, alpha = 1.0, 0.6
        sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        cutoff = 10.0 * sigma_c
        kcut = 6.0 / sigma_c

        grads: dict[str, torch.Tensor] = {}
        for device in ("cpu", "gpu"):
            pos, sf, cell, cell_np, pos_np = _build(
                n=2, device=device, seed=7, l_max=l_max
            )
            pos.requires_grad_(True)
            sf.requires_grad_(True)
            td = _torch_device(device)
            idx_j_np, nptr_np, sh_np = _neigh(pos_np, cell_np[0, 0], cutoff)
            idx_j = torch.from_numpy(idx_j_np).to(td)
            nptr = torch.from_numpy(nptr_np).to(td)
            sh = torch.from_numpy(sh_np).to(td)
            energy = multipole_ewald_summation(
                pos,
                sf,
                cell,
                idx_j,
                nptr,
                sh,
                sigma=sigma,
                alpha=alpha,
                kspace_cutoff=kcut,
            )
            energy.backward()
            grads[f"{device}_pos"] = pos.grad.detach().cpu().clone()
            grads[f"{device}_sf"] = sf.grad.detach().cpu().clone()

        torch.testing.assert_close(
            grads["gpu_pos"], grads["cpu_pos"], rtol=1e-10, atol=1e-13
        )
        torch.testing.assert_close(
            grads["gpu_sf"], grads["cpu_sf"], rtol=1e-10, atol=1e-13
        )
