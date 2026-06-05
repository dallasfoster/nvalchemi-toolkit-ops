# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct-k-space l_max=2 reciprocal energy + 1st-order grads.

Validates the Cartesian-quadrupole reciprocal channel and the
``multipole_ewald_summation`` composite at l_max=2.

Oracle strategy
---------------
* Reciprocal parity is checked against the numpy reciprocal reference, whose
  reciprocal term is trustworthy (its real-space QQ term is buggy, so the full
  reference total is not used as the l=2 oracle).
* Composite parity is cross-checked against the PME composite, which shares the
  identical (independently FD-validated) real-space kernel + analytical self
  term and differs only in the reciprocal method.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    cartesian_quadrupole_to_e3nn,
    dipole_cartesian_to_spherical,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
    multipole_reciprocal_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    FIELD_CONSTANT,
    multipole_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    multipole_pme_reciprocal_space,
)

F = float(FIELD_CONSTANT)


def _fixture(seed=11, N=4):
    rng = np.random.default_rng(seed)
    L = 6.0
    cell_np = np.eye(3) * L
    pos = rng.uniform(0, L, size=(N, 3))
    q = rng.normal(size=N)
    q -= q.mean()
    mu = rng.normal(size=(N, 3))
    Qr = rng.normal(size=(N, 3, 3))
    Q = 0.5 * (Qr + Qr.transpose(0, 2, 1))
    return cell_np, pos, q, mu, Q, L


def _sf(q, mu):
    # e3nn packing [q, mu_y, mu_z, mu_x]
    return torch.tensor(
        np.concatenate([q[:, None], mu[:, [1, 2, 0]]], axis=1), dtype=torch.float64
    )


def _mm(q, mu, Q=None):
    """Packed e3nn ``multipole_moments``: l<=1 block (+ traceless l=2 from a
    Cartesian Q via the converter)."""
    from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
        cartesian_quadrupole_to_e3nn,
    )

    sf = _sf(q, mu)
    if Q is None:
        return sf
    Qt = Q if torch.is_tensor(Q) else torch.tensor(Q, dtype=torch.float64)
    return torch.cat([sf, cartesian_quadrupole_to_e3nn(Qt)], dim=-1)


def _half_kgrid(cell_np, kcut):
    """Canonical half k-set (origin + one of each ±k pair) + the full ±set."""
    G = 2.0 * np.pi * np.linalg.inv(cell_np).T
    nmax = int(np.ceil(kcut / np.linalg.norm(G, axis=1).min())) + 1
    half = []
    for a in range(-nmax, nmax + 1):
        for b in range(-nmax, nmax + 1):
            for c in range(-nmax, nmax + 1):
                k = a * G[0] + b * G[1] + c * G[2]
                if not (0 < np.linalg.norm(k) <= kcut):
                    continue
                if (a > 0) or (a == 0 and b > 0) or (a == 0 and b == 0 and c > 0):
                    half.append(k)
    half = np.array(half)
    return half


def _ref_recip(pos, q, mu, Q, ks_full, sigma, alpha, V):
    E = 0.0
    for k in ks_full:
        k2 = float(k @ k)
        e = np.exp(-1j * (pos @ k))
        rho = (q * e).sum() - 1j * ((mu @ k) * e).sum()
        rho -= 0.5 * (np.einsum("a,nab,b->n", k, Q, k) * e).sum()
        E += (
            (4.0 * math.pi / k2)
            * math.exp(-k2 * (0.25 / alpha**2 + sigma**2))
            * (rho * rho.conjugate()).real
        )
    return (F / (4.0 * math.pi)) * 0.5 / V * E


@pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8])
def test_reciprocal_quadrupole_parity_vs_reference(sigma):
    """Direct-k reciprocal energy matches the numpy reciprocal reference."""
    alpha, kcut = 0.45, 9.0
    cell_np, pos, q, mu, Q, L = _fixture()
    V = L**3
    half = _half_kgrid(cell_np, kcut)
    ks_prod = np.vstack([np.zeros((1, 3)), half])
    ks_full = np.vstack([half, -half])

    E = multipole_reciprocal_space_energy(
        torch.tensor(pos),
        _mm(q, mu, Q),
        torch.tensor(cell_np),
        sigma=sigma,
        alpha=alpha,
        k_vectors=torch.tensor(ks_prod, dtype=torch.float64),
    )
    # Reference uses the SAME traceless Q the converter feeds the kernel.
    Q_tl = Q - np.eye(3)[None] * (np.trace(Q, axis1=1, axis2=2)[:, None, None] / 3.0)
    E_ref = _ref_recip(pos, q, mu, Q_tl, ks_full, sigma, alpha, V)
    assert abs(float(E) - E_ref) / abs(E_ref) < 1e-9


def test_reciprocal_quadrupole_grads_match_fd():
    """Autograd ∂E/∂{pos,q,μ,Q} match central finite differences."""
    sigma, alpha, kcut = 0.5, 0.45, 9.0
    cell_np, pos, q, mu, Q, L = _fixture()
    cell = torch.tensor(cell_np)
    kvec = torch.tensor(
        np.vstack([np.zeros((1, 3)), _half_kgrid(cell_np, kcut)]), dtype=torch.float64
    )

    def energy(pos_, q_, mu_, Q_):
        sf_ = torch.cat([q_[:, None], dipole_cartesian_to_spherical(mu_)], dim=1)
        mm_ = torch.cat([sf_, cartesian_quadrupole_to_e3nn(Q_)], dim=1)
        return multipole_reciprocal_space_energy(
            pos_,
            mm_,
            cell,
            sigma=sigma,
            alpha=alpha,
            k_vectors=kvec,
        )

    pos_t = torch.tensor(pos, requires_grad=True)
    q_t = torch.tensor(q, requires_grad=True)
    mu_t = torch.tensor(mu, requires_grad=True)
    Q_t = torch.tensor(Q, requires_grad=True)
    E = energy(pos_t, q_t, mu_t, Q_t)
    gpos, gq, gmu, gQ = torch.autograd.grad(E, [pos_t, q_t, mu_t, Q_t])

    h = 1e-6
    base = (torch.tensor(pos), torch.tensor(q), torch.tensor(mu), torch.tensor(Q))

    def fd(idx, shape):
        out = torch.zeros(shape, dtype=torch.float64)
        flat = out.view(-1)
        b0 = base[idx].clone()
        for i in range(flat.numel()):
            args_p = list(base)
            args_m = list(base)
            bp = b0.clone()
            bp.view(-1)[i] += h
            bm = b0.clone()
            bm.view(-1)[i] -= h
            args_p[idx] = bp
            args_m[idx] = bm
            flat[i] = (float(energy(*args_p)) - float(energy(*args_m))) / (2 * h)
        return out

    N = pos.shape[0]
    assert (gpos - fd(0, (N, 3))).abs().max() / gpos.abs().max() < 1e-5
    assert (gq - fd(1, (N,))).abs().max() / gq.abs().max() < 1e-5
    assert (gmu - fd(2, (N, 3))).abs().max() / gmu.abs().max() < 1e-5
    fd_Q_free = fd(3, (N, 3, 3))
    fd_Q = 0.5 * (fd_Q_free + fd_Q_free.transpose(-1, -2))  # kernel emits symmetric
    assert (gQ - fd_Q).abs().max() / fd_Q.abs().max() < 1e-5
    assert (gQ - gQ.transpose(-1, -2)).abs().max() < 1e-12


def test_reciprocal_quadrupole_none_unchanged():
    """quadrupoles=None reproduces the l<=1 reciprocal exactly (no regression)."""
    sigma, alpha, kcut = 0.5, 0.45, 9.0
    cell_np, pos, q, mu, Q, L = _fixture()
    kvec = torch.tensor(
        np.vstack([np.zeros((1, 3)), _half_kgrid(cell_np, kcut)]), dtype=torch.float64
    )
    args = dict(sigma=sigma, alpha=alpha, k_vectors=kvec)
    E_none = multipole_reciprocal_space_energy(
        torch.tensor(pos), _sf(q, mu), torch.tensor(cell_np), **args
    )
    E_q0 = multipole_reciprocal_space_energy(
        torch.tensor(pos),
        _mm(q, mu, np.zeros((pos.shape[0], 3, 3))),
        torch.tensor(cell_np),
        **args,
    )
    assert abs(float(E_none) - float(E_q0)) < 1e-12


def _build_csr(pos, cell_np, cutoff):
    N = len(pos)
    Lv = np.diag(cell_np)
    ii, jj, sh = [], [], []
    for i in range(N):
        for j in range(N):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        if i == j and sx == sy == sz == 0:
                            continue
                        d = pos[j] - pos[i] + np.array([sx, sy, sz]) * Lv
                        if np.linalg.norm(d) < cutoff:
                            ii.append(i)
                            jj.append(j)
                            sh.append((sx, sy, sz))
    order = sorted(range(len(ii)), key=lambda k: (ii[k], jj[k]))
    ii = [ii[k] for k in order]
    jj = [jj[k] for k in order]
    sh = [sh[k] for k in order]
    ptr = np.zeros(N + 1, dtype=np.int32)
    for i in ii:
        ptr[i + 1] += 1
    ptr = np.cumsum(ptr).astype(np.int32)
    return (
        torch.tensor(jj, dtype=torch.int32),
        torch.tensor(ptr, dtype=torch.int32),
        torch.tensor(sh, dtype=torch.int32).reshape(-1, 3),
    )


def test_composite_quadrupole_matches_pme_composite():
    """Full direct-k Ewald composite (l=2) == PME composite to mesh accuracy.

    Both share the identical (FD-validated) real-space kernel + analytical
    self; they differ only in the reciprocal method.
    """
    sigma, alpha = 0.5, 0.45
    real_cut, kcut = 12.0, 12.0
    cell_np, pos, q, mu, Q, L = _fixture()
    # Detrace Q: the migrated Ewald path is e3nn-traceless; the PME oracle
    # (still Cartesian, trace-ful) must use the same traceless Q to agree.
    Q = Q - np.eye(3)[None] * (np.trace(Q, axis1=1, axis2=2)[:, None, None] / 3.0)
    idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
    cell = torch.tensor(cell_np)
    sf = _sf(q, mu)
    pos_t = torch.tensor(pos)
    Q_t = torch.tensor(Q)

    E_dk = float(
        multipole_ewald_summation(
            pos_t,
            _mm(q, mu, Q),
            cell,
            idx_j,
            ptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            kspace_cutoff=kcut,
        )
    )

    cs = F / (4.0 * math.pi)
    e_real = cs * float(
        multipole_real_space_quadrupole_energy(
            pos_t,
            sf[:, 0],
            sf[:, [3, 1, 2]].contiguous(),
            Q_t,
            cell.unsqueeze(0),
            idx_j,
            ptr,
            sh,
            torch.tensor([sigma], dtype=torch.float64),
            torch.tensor([alpha], dtype=torch.float64),
        ).sum()
    )
    # multipole_pme_reciprocal_space already returns recip - self - bg.
    e_recip_pme = float(
        multipole_pme_reciprocal_space(
            pos_t,
            torch.cat([sf, cartesian_quadrupole_to_e3nn(Q_t)], dim=-1),
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=(48, 48, 48),
        )
    )
    E_pme = e_real + e_recip_pme
    assert abs(E_dk - E_pme) / abs(E_pme) < 2e-4


def test_composite_quadrupole_batched_matches_per_system():
    """Batched l=2 composite (2 systems) == single-system composite per system,
    and is autograd-connected across the flat moment tensors."""
    sigma, alpha, real_cut, kcut = 0.5, 0.45, 12.0, 12.0
    systems = [_fixture(11), _fixture(22)]

    # Per-system single composite.
    E_single = []
    for cell_np, pos, q, mu, Q, L in systems:
        idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
        E = multipole_ewald_summation(
            torch.tensor(pos),
            _mm(q, mu, Q),
            torch.tensor(cell_np),
            idx_j,
            ptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            kspace_cutoff=kcut,
        )
        E_single.append(float(E))

    # Concatenate into a flat batch with offset CSR + batch_idx.
    pos_all, q_all, mu_all, Q_all, cells, bidx = [], [], [], [], [], []
    idxj_all, ptr_all, sh_all = [], [0], []
    off, ptr_off = 0, 0
    for b, (cell_np, pos, q, mu, Q, L) in enumerate(systems):
        idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
        n = pos.shape[0]
        pos_all.append(pos)
        q_all.append(q)
        mu_all.append(mu)
        Q_all.append(Q)
        cells.append(cell_np)
        bidx += [b] * n
        idxj_all.append(idx_j.numpy() + off)
        ptr_all += (ptr.numpy()[1:] + ptr_off).tolist()
        sh_all.append(sh.numpy())
        off += n
        ptr_off += len(idx_j)
    pos_all = np.concatenate(pos_all)
    q_all = np.concatenate(q_all)
    mu_all = np.concatenate(mu_all)
    Q_all = np.concatenate(Q_all)
    cells = np.stack(cells)
    sf = _sf(q_all, mu_all)
    idx_j = torch.tensor(np.concatenate(idxj_all), dtype=torch.int32)
    ptr = torch.tensor(ptr_all, dtype=torch.int32)
    sh = torch.tensor(np.concatenate(sh_all), dtype=torch.int32)
    bidx = torch.tensor(bidx, dtype=torch.int32)
    pos_t = torch.tensor(pos_all, requires_grad=True)
    Q_t = torch.tensor(Q_all, requires_grad=True)

    mm_b = torch.cat([sf, cartesian_quadrupole_to_e3nn(Q_t)], dim=-1)
    Eb = multipole_ewald_summation(
        pos_t,
        mm_b,
        torch.tensor(cells),
        idx_j,
        ptr,
        sh,
        sigma=sigma,
        alpha=alpha,
        kspace_cutoff=kcut,
        batch_idx=bidx,
    )
    assert Eb.shape == (2,)
    for i in range(2):
        assert abs(float(Eb[i]) - E_single[i]) / abs(E_single[i]) < 1e-9
    gpos, gQ = torch.autograd.grad(Eb.sum(), [pos_t, Q_t])
    assert torch.isfinite(gpos).all() and torch.isfinite(gQ).all()


def test_pme_composite_quadrupole_matches_directk_and_batched():
    """multipole_particle_mesh_ewald(..., quadrupoles=Q) matches the direct-k
    Ewald composite (PME mesh accuracy) single-system; the batched l=2 path
    reproduces the single-system result bit-for-bit (B=1 system)."""
    from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
        multipole_particle_mesh_ewald,
    )

    sigma, alpha, real_cut, kcut = 0.5, 0.45, 12.0, 12.0
    cell_np, pos, q, mu, Q, L = _fixture()
    # Detrace Q so the traceless Ewald path matches the trace-ful PME oracle.
    Q = Q - np.eye(3)[None] * (np.trace(Q, axis1=1, axis2=2)[:, None, None] / 3.0)
    idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
    cell = torch.tensor(cell_np)
    sf = _sf(q, mu)
    pos_t = torch.tensor(pos, requires_grad=True)
    Q_t = torch.tensor(Q, requires_grad=True)

    E_dk = float(
        multipole_ewald_summation(
            torch.tensor(pos),
            _mm(q, mu, Q),
            cell,
            idx_j,
            ptr,
            sh,
            sigma=sigma,
            alpha=alpha,
            kspace_cutoff=kcut,
        )
    )
    E_pme = multipole_particle_mesh_ewald(
        pos_t,
        torch.cat([sf, cartesian_quadrupole_to_e3nn(Q_t)], dim=-1),
        cell,
        idx_j,
        ptr,
        sh,
        sigma=sigma,
        alpha=alpha,
        mesh_dimensions=(48, 48, 48),
    )
    assert abs(float(E_pme) - E_dk) / abs(E_dk) < 2e-4
    gpos, gQ = torch.autograd.grad(E_pme, [pos_t, Q_t])
    assert torch.isfinite(gpos).all() and torch.isfinite(gQ).all()
    assert (gQ - gQ.transpose(-1, -2)).abs().max() < 1e-10

    # A single-system batch reproduces the single-system composite bit-for-bit
    # (energy + forces + ∂E/∂Q).
    bidx = torch.zeros(pos.shape[0], dtype=torch.int32)
    pos_b = torch.tensor(pos, requires_grad=True)
    Q_b = torch.tensor(Q, requires_grad=True)
    E_batch = multipole_particle_mesh_ewald(
        pos_b,
        torch.cat([sf, cartesian_quadrupole_to_e3nn(Q_b)], dim=-1),
        cell.reshape(1, 3, 3),
        idx_j,
        ptr,
        sh,
        sigma=sigma,
        alpha=alpha,
        mesh_dimensions=(48, 48, 48),
        batch_idx=bidx,
    )
    assert E_batch.shape == (1,)
    torch.testing.assert_close(E_batch[0], E_pme.detach(), rtol=1e-9, atol=1e-9)
    gpos_b, gQ_b = torch.autograd.grad(E_batch.sum(), [pos_b, Q_b])
    torch.testing.assert_close(gpos_b, gpos, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(gQ_b, gQ, rtol=1e-8, atol=1e-8)


def test_composite_quadrupole_autograd_connected():
    """Composite l=2 is finite + autograd-connected on all moment channels."""
    sigma, alpha = 0.5, 0.45
    cell_np, pos, q, mu, Q, L = _fixture()
    idx_j, ptr, sh = _build_csr(pos, cell_np, 12.0)
    cell = torch.tensor(cell_np)
    pos_t = torch.tensor(pos, requires_grad=True)
    sf = _sf(q, mu).requires_grad_(True)
    Q_t = torch.tensor(Q, requires_grad=True)
    mm = torch.cat([sf, cartesian_quadrupole_to_e3nn(Q_t)], dim=-1)
    E = multipole_ewald_summation(
        pos_t,
        mm,
        cell,
        idx_j,
        ptr,
        sh,
        sigma=sigma,
        alpha=alpha,
        kspace_cutoff=12.0,
    )
    assert torch.isfinite(E)
    gpos, gsf, gQ = torch.autograd.grad(E, [pos_t, sf, Q_t])
    assert torch.isfinite(gpos).all()
    assert torch.isfinite(gsf).all()
    assert torch.isfinite(gQ).all()
    assert (gQ - gQ.transpose(-1, -2)).abs().max() < 1e-12


# =============================================================================
# Reciprocal cell-grad (stress) through the composite Ewald path
# =============================================================================


def _fd_cell_grad(energy_fn, cell_np, h=1e-6):
    """Central-difference ∂E/∂cell (3, 3)."""
    fd = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            cp = cell_np.copy()
            cp[a, b] += h
            cm = cell_np.copy()
            cm[a, b] -= h
            fd[a, b] = (float(energy_fn(cp)) - float(energy_fn(cm))) / (2 * h)
    return fd


@pytest.mark.parametrize("with_quad", [False, True])
def test_composite_stress_matches_fd(with_quad):
    """∂E/∂cell through the full composite Ewald (real + direct-k recip − self)
    matches FD — l≤1 and l=2, single-system."""
    sigma, alpha, real_cut, kcut = 0.5, 0.45, 9.0, 9.0
    cell_np, pos, q, mu, Q, L = _fixture(7)
    idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
    mm = _mm(q, mu, Q if with_quad else None)
    kw = dict(sigma=sigma, alpha=alpha, kspace_cutoff=kcut)

    def energy(cell_t):
        return multipole_ewald_summation(
            torch.tensor(pos), mm, cell_t, idx_j, ptr, sh, **kw
        )

    cell = torch.tensor(cell_np, requires_grad=True)
    (gcell,) = torch.autograd.grad(energy(cell), [cell])
    fd = _fd_cell_grad(lambda c: energy(torch.tensor(c)), cell_np)
    rel = np.abs(gcell.numpy() - fd).max() / (np.abs(fd).max() + 1e-30)
    assert rel < 5e-5, f"stress rel vs FD = {rel:.3e}"


@pytest.mark.parametrize("with_quad", [False, True])
def test_batched_stress_matches_per_system(with_quad):
    """Batched composite stress equals the per-system single-system stress
    (l≤1 and l=2)."""
    sigma, alpha, real_cut, kcut = 0.5, 0.45, 9.0, 9.0
    systems = [_fixture(11), _fixture(22)]

    # per-system single stress
    single_g = []
    for cell_np, pos, q, mu, Q, L in systems:
        idx_j, ptr, sh = _build_csr(pos, cell_np, real_cut)
        kw = dict(sigma=sigma, alpha=alpha, kspace_cutoff=kcut)
        ct = torch.tensor(cell_np, requires_grad=True)
        mm = _mm(q, mu, Q if with_quad else None)
        E = multipole_ewald_summation(torch.tensor(pos), mm, ct, idx_j, ptr, sh, **kw)
        single_g.append(torch.autograd.grad(E, [ct])[0].numpy())

    # batched
    pos_all, q_all, mu_all, Q_all, cells, bidx = [], [], [], [], [], []
    idxj, ptr_all, sh_all, off, po = [], [0], [], 0, 0
    for b, (cell_np, pos, q, mu, Q, L) in enumerate(systems):
        ij, pt, s = _build_csr(pos, cell_np, real_cut)
        n = pos.shape[0]
        pos_all.append(pos)
        q_all.append(q)
        mu_all.append(mu)
        Q_all.append(Q)
        cells.append(cell_np)
        bidx += [b] * n
        idxj.append(ij.numpy() + off)
        ptr_all += (pt.numpy()[1:] + po).tolist()
        sh_all.append(s.numpy())
        off += n
        po += len(ij)
    mm = _mm(
        np.concatenate(q_all),
        np.concatenate(mu_all),
        np.concatenate(Q_all) if with_quad else None,
    )
    cells_t = torch.tensor(np.stack(cells), requires_grad=True)
    kw = dict(
        sigma=sigma,
        alpha=alpha,
        kspace_cutoff=kcut,
        batch_idx=torch.tensor(bidx, dtype=torch.int32),
    )
    Eb = multipole_ewald_summation(
        torch.tensor(np.concatenate(pos_all)),
        mm,
        cells_t,
        torch.tensor(np.concatenate(idxj), dtype=torch.int32),
        torch.tensor(ptr_all, dtype=torch.int32),
        torch.tensor(np.concatenate(sh_all), dtype=torch.int32),
        **kw,
    )
    (gb,) = torch.autograd.grad(Eb.sum(), [cells_t])
    for b in range(2):
        rel = np.abs(gb.numpy()[b] - single_g[b]).max() / (
            np.abs(single_g[b]).max() + 1e-30
        )
        assert rel < 1e-9, f"system {b}: batched stress rel = {rel:.3e}"


# =============================================================================
# Direct-k l=2 Q-channel double-back (create_graph=True force/stress)
# =============================================================================


def test_reciprocal_quadrupole_pos_hvp_matches_fd():
    """create_graph=True pos-HVP through the l=2 reciprocal (force-loss with Q
    present) matches a central-difference HVP."""
    sigma, alpha, kcut = 0.5, 0.45, 8.0
    cell_np, pos, q, mu, Q, L = _fixture(seed=3)
    cell = torch.tensor(cell_np)
    mm = _mm(q, mu, Q)
    rng = np.random.default_rng(1)
    v = torch.tensor(rng.normal(size=pos.shape))

    def grad_pos(p, create):
        pt = torch.tensor(p, requires_grad=True)
        E = multipole_reciprocal_space_energy(
            pt, mm, cell, sigma=sigma, alpha=alpha, kspace_cutoff=kcut
        )
        return torch.autograd.grad(E, [pt], create_graph=create)[0]

    pt = torch.tensor(pos, requires_grad=True)
    E = multipole_reciprocal_space_energy(
        pt, mm, cell, sigma=sigma, alpha=alpha, kspace_cutoff=kcut
    )
    gp = torch.autograd.grad(E, [pt], create_graph=True)[0]
    hvp = torch.autograd.grad((gp * v).sum(), [pt])[0]

    h = 1e-6
    fd = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            pp = pos.copy()
            pp[i, d] += h
            pm = pos.copy()
            pm[i, d] -= h
            fd[i, d] = float(
                ((grad_pos(pp, False) - grad_pos(pm, False)) * v).sum()
            ) / (2 * h)
    rel = np.abs(hvp.detach().numpy() - fd).max() / (np.abs(fd).max() + 1e-30)
    assert rel < 1e-4, f"pos-HVP rel = {rel:.3e}"


def test_reciprocal_quadrupole_q_hvp_matches_fd():
    """create_graph=True Q-HVP (and mixed ∂²E/∂r∂Q) through the l=2 reciprocal
    match finite differences."""
    sigma, alpha, kcut = 0.5, 0.45, 8.0
    cell_np, pos, q, mu, Q, L = _fixture(seed=3)
    cell = torch.tensor(cell_np)
    sf = _sf(q, mu)
    rng = np.random.default_rng(2)
    vQr = rng.normal(size=Q.shape)
    vQ = torch.tensor(0.5 * (vQr + vQr.transpose(0, 2, 1)))

    def grad_Q(Qm):
        Qt = torch.tensor(Qm, requires_grad=True)
        E = multipole_reciprocal_space_energy(
            torch.tensor(pos),
            torch.cat([sf, cartesian_quadrupole_to_e3nn(Qt)], dim=-1),
            cell,
            sigma=sigma,
            alpha=alpha,
            kspace_cutoff=kcut,
        )
        return torch.autograd.grad(E, [Qt], create_graph=True)[0]

    Qt = torch.tensor(Q, requires_grad=True)
    E = multipole_reciprocal_space_energy(
        torch.tensor(pos),
        torch.cat([sf, cartesian_quadrupole_to_e3nn(Qt)], dim=-1),
        cell,
        sigma=sigma,
        alpha=alpha,
        kspace_cutoff=kcut,
    )
    gQ = torch.autograd.grad(E, [Qt], create_graph=True)[0]
    qhvp = torch.autograd.grad((gQ * vQ).sum(), [Qt])[0]

    h = 1e-6
    fd = np.zeros_like(Q)
    for i in range(Q.shape[0]):
        for a in range(3):
            for b in range(3):
                Qp = Q.copy()
                Qp[i, a, b] += h
                Qm = Q.copy()
                Qm[i, a, b] -= h
                fd[i, a, b] = float(((grad_Q(Qp) - grad_Q(Qm)) * vQ).sum()) / (2 * h)
    rel = np.abs(qhvp.detach().numpy() - fd).max() / (np.abs(fd).max() + 1e-30)
    assert rel < 1e-4, f"Q-HVP rel = {rel:.3e}"


def test_reciprocal_quadrupole_batched_hvp_matches_per_system():
    """Batched l=2 reciprocal pos-HVP + Q-HVP equal the per-system single-system
    HVPs (validates the batched K_a/K_b/K_c indexing). Same cell per system so
    the per-system k-grids align with the batched (padded) grid."""
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
        prepare_multipole_scf_cache,
    )
    from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import (
        multipole_scf_step_energy,
    )

    sigma, alpha, kcut, rs = 0.5, 0.45, 8.0, [0.8]
    L = 6.0
    cell_np = np.eye(3) * L
    rng = np.random.default_rng(7)
    Ns = [4, 3]

    def mk(n):
        p = rng.uniform(0, L, (n, 3))
        qq = rng.normal(size=n)
        qq -= qq.mean()
        m = rng.normal(size=(n, 3))
        Qr = rng.normal(size=(n, 3, 3))
        return (
            p,
            np.concatenate([qq[:, None], m[:, [1, 2, 0]]], 1),
            0.5 * (Qr + Qr.transpose(0, 2, 1)),
        )

    sysd = [mk(n) for n in Ns]
    pos_all = np.concatenate([s[0] for s in sysd])
    sf_all = np.concatenate([s[1] for s in sysd])
    Q_all = np.concatenate([s[2] for s in sysd])
    bidx = np.concatenate([[b] * Ns[b] for b in range(len(Ns))]).astype(np.int32)

    cacheb = prepare_multipole_scf_cache(
        torch.tensor(np.stack([cell_np] * len(Ns))),
        sigma=sigma,
        receiver_sigmas=rs,
        kspace_cutoff=kcut,
        l_max=2,
        alpha=alpha,
    )
    ptb = torch.tensor(pos_all, requires_grad=True)
    Qtb = torch.tensor(Q_all, requires_grad=True)
    Eb = multipole_scf_step_energy(
        cacheb,
        ptb,
        torch.tensor(sf_all),
        batch_idx=torch.tensor(bidx),
        include_self_interaction=True,
        quadrupoles=Qtb,
    )
    gpb, gQb = torch.autograd.grad(Eb.sum(), [ptb, Qtb], create_graph=True)
    vpos = torch.tensor(rng.normal(size=pos_all.shape))
    vQr = rng.normal(size=Q_all.shape)
    vQ = torch.tensor(0.5 * (vQr + vQr.transpose(0, 2, 1)))
    poshvp_b = torch.autograd.grad((gpb * vpos).sum(), [ptb], retain_graph=True)[0]
    qhvp_b = torch.autograd.grad((gQb * vQ).sum(), [Qtb], retain_graph=True)[0]

    off = 0
    for b, (p, sf, Q) in enumerate(sysd):
        n = Ns[b]
        sl = slice(off, off + n)
        off += n
        cache = prepare_multipole_scf_cache(
            torch.tensor(cell_np),
            sigma=sigma,
            receiver_sigmas=rs,
            kspace_cutoff=kcut,
            l_max=2,
            alpha=alpha,
        )
        pt = torch.tensor(p, requires_grad=True)
        Qt = torch.tensor(Q, requires_grad=True)
        E = multipole_scf_step_energy(
            cache, pt, torch.tensor(sf), include_self_interaction=True, quadrupoles=Qt
        )
        gp, gQ = torch.autograd.grad(E, [pt, Qt], create_graph=True)
        ph = torch.autograd.grad((gp * vpos[sl]).sum(), [pt], retain_graph=True)[0]
        qh = torch.autograd.grad((gQ * vQ[sl]).sum(), [Qt], retain_graph=True)[0]
        assert (ph - poshvp_b[sl].detach()).abs().max() < 1e-12
        assert (qh - qhvp_b[sl].detach()).abs().max() < 1e-12
