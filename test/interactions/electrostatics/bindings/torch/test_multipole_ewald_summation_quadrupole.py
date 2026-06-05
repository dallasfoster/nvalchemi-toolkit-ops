# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-level integration tests for LMAX=2 (Cartesian-Q convention)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    cartesian_quadrupole_to_e3nn,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    _multipole_ewald_self_energy_per_atom,
    multipole_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)


def _make_4atom_pbc(dtype=torch.float64, device="cpu"):
    rng = np.random.default_rng(20260613)
    n = 4
    L = 6.0
    pos = torch.tensor(
        rng.uniform(0.0, L, size=(n, 3)),
        dtype=dtype,
        device=device,
    )
    q = torch.tensor(rng.normal(size=(n,)), dtype=dtype, device=device)
    mu = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype, device=device)
    Q_raw = rng.normal(size=(n, 3, 3))
    Q_sym = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    Q = torch.tensor(Q_sym, dtype=dtype, device=device)
    cell = torch.tensor(
        [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
        dtype=dtype,
        device=device,
    ).unsqueeze(0)

    # source_feats (N, 4) packed: charges + dipoles in e3nn (y, z, x) order
    # at indices 1, 2, 3.
    sf = torch.empty(n, 4, dtype=dtype, device=device)
    sf[:, 0] = q
    sf[:, 1] = mu[:, 1]  # y (m=-1)
    sf[:, 2] = mu[:, 2]  # z (m=0)
    sf[:, 3] = mu[:, 0]  # x (m=+1)

    # Build a CSR full neighbor list within cutoff=4.0 with PBC.
    cutoff = 4.0
    idx_j_list, shifts_list = [], []
    pos_np = pos.cpu().numpy()
    for i in range(n):
        nb, sh = [], []
        for j in range(n):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        if i == j and (sx, sy, sz) == (0, 0, 0):
                            continue
                        disp = pos_np[j] - pos_np[i] + np.array([sx, sy, sz]) * L
                        if np.linalg.norm(disp) < cutoff:
                            nb.append(j)
                            sh.append((sx, sy, sz))
        idx_j_list.append(nb)
        shifts_list.append(sh)
    counts = [len(ell) for ell in idx_j_list]
    ptr = np.cumsum([0] + counts)
    idx_j = torch.tensor(
        [j for nbrs in idx_j_list for j in nbrs],
        dtype=torch.int32,
        device=device,
    )
    neighbor_ptr = torch.tensor(ptr, dtype=torch.int32, device=device)
    unit_shifts = torch.tensor(
        [s for shifts in shifts_list for s in shifts],
        dtype=torch.int32,
        device=device,
    ).reshape(-1, 3)

    # Packed e3nn moments (N, 9): l<=1 block + traceless l=2 (the converter
    # drops the trace of the fixture's symmetric Q).
    multipole_moments = torch.cat(
        [sf, cartesian_quadrupole_to_e3nn(Q)], dim=-1
    ).contiguous()
    return dict(
        positions=pos,
        charges=q,
        dipoles=mu,
        quadrupoles=Q,
        source_feats=sf,
        multipole_moments=multipole_moments,
        cell=cell,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )


def test_self_energy_quadrupole_term_is_additive():
    """The LMAX=2 self-energy is the LMAX=1 self-energy plus a |Q|_F^2 term."""
    p = _make_4atom_pbc()
    sigma, alpha = 0.5, 0.35
    e1 = _multipole_ewald_self_energy_per_atom(p["source_feats"], sigma, alpha).sum()
    e2 = _multipole_ewald_self_energy_per_atom(
        p["source_feats"],
        sigma,
        alpha,
        quadrupoles=p["quadrupoles"],
    ).sum()
    sigma_c = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
    from nvalchemiops.torch.math import FIELD_CONSTANT

    # l=2 self-energy denominator is 320: the angular (k·Q·k)² ↔
    # Cartesian-Frobenius |Q|_F² factor of 3/2.
    expected_Q_term = (
        FIELD_CONSTANT
        / (320.0 * math.pi**1.5 * sigma_c**5)
        * (p["quadrupoles"].to(torch.float64) ** 2).sum()
    )
    diff = (e2 - e1 - expected_Q_term).abs().item()
    assert diff < 1e-12, f"self-energy decomposition mismatch: diff={diff:.3e}"


def test_quadrupole_total_is_alpha_independent():
    """The composite Ewald l=2 total must be α-independent.

    α only splits the real/reciprocal work; the total cannot depend on it.
    Sweeps α with σ_c-scaled cutoffs (constant truncation error) and asserts
    the q+μ+Q total is flat to the convergence floor.
    """
    rng = np.random.default_rng(7)
    n, L, sigma = 4, 6.0, 0.5
    pos_np = rng.uniform(0.0, L, size=(n, 3))
    q = rng.normal(size=(n,))
    q -= q.mean()
    mu = rng.normal(size=(n, 3))
    Qr = rng.normal(size=(n, 3, 3))
    Q = 0.5 * (Qr + Qr.transpose(0, 2, 1))
    cell = torch.tensor(np.eye(3) * L)
    sf = torch.empty(n, 4, dtype=torch.float64)
    sf[:, 0] = torch.tensor(q)
    sf[:, 1], sf[:, 2], sf[:, 3] = (
        torch.tensor(mu[:, 1]),
        torch.tensor(mu[:, 2]),
        torch.tensor(mu[:, 0]),
    )
    mm = torch.cat(
        [sf, cartesian_quadrupole_to_e3nn(torch.tensor(Q))], dim=-1
    ).contiguous()
    pos = torch.tensor(pos_np)

    def _csr(cutoff):
        shell = int(math.ceil(cutoff / L)) + 1
        idx, ptr, sh = [], [0], []
        for i in range(n):
            for sx in range(-shell, shell + 1):
                for sy in range(-shell, shell + 1):
                    for sz in range(-shell, shell + 1):
                        for j in range(n):
                            if i == j and (sx, sy, sz) == (0, 0, 0):
                                continue
                            d = pos_np[j] - pos_np[i] + np.array([sx, sy, sz]) * L
                            if np.linalg.norm(d) < cutoff:
                                idx.append(j)
                                sh.append((sx, sy, sz))
            ptr.append(len(idx))
        return (
            torch.tensor(idx, dtype=torch.int32),
            torch.tensor(ptr, dtype=torch.int32),
            torch.tensor(sh, dtype=torch.int32).reshape(-1, 3),
        )

    totals = []
    for alpha in (0.4, 0.6, 0.9):
        sc = math.sqrt(sigma**2 + 1.0 / (4.0 * alpha**2))
        idx_j, ptr, sh = _csr(12.0 * sc)
        E = float(
            multipole_ewald_summation(
                pos,
                mm,
                cell,
                idx_j,
                ptr,
                sh,
                sigma=sigma,
                alpha=alpha,
                kspace_cutoff=7.0 / sc,
            )
        )
        totals.append(E)
    spread = max(totals) - min(totals)
    assert spread < 1e-4, (
        f"l=2 Ewald total is α-dependent: {totals} (spread={spread:.3e})"
    )


def test_multipole_ewald_summation_quadrupole_computes():
    """Single-system composite at l_max=2 computes a finite, autograd-connected
    total (direct-k reciprocal + real-space + self)."""
    p = _make_4atom_pbc()
    pos = p["positions"].clone().requires_grad_(True)
    mm = p["multipole_moments"].clone().requires_grad_(True)
    E = multipole_ewald_summation(
        pos,
        mm,
        p["cell"],
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
        sigma=0.5,
        alpha=0.35,
        kspace_cutoff=6.0,
    )
    assert E.ndim == 0 and torch.isfinite(E)
    gpos, gmm = torch.autograd.grad(E, [pos, mm])
    assert torch.isfinite(gpos).all() and torch.isfinite(gmm).all()
    # gradient flows to the l=2 (5-component) e3nn block.
    assert gmm.shape == (p["positions"].shape[0], 9)
    assert gmm[:, 4:9].abs().max() > 0.0


def test_multipole_ewald_summation_quadrupole_batched_matches_single():
    """A B=1 batch equals the single-system composite at l_max=2."""
    p = _make_4atom_pbc()
    n = p["positions"].shape[0]
    batch_idx = torch.zeros(n, dtype=torch.int32)
    cell_b = p["cell"].reshape(1, 3, 3) if p["cell"].ndim == 2 else p["cell"]
    cell_s = p["cell"].reshape(3, 3) if p["cell"].ndim == 3 else p["cell"]

    E_single = multipole_ewald_summation(
        p["positions"],
        p["multipole_moments"],
        cell_s,
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
        sigma=0.5,
        alpha=0.35,
        kspace_cutoff=6.0,
    )
    E_batch = multipole_ewald_summation(
        p["positions"],
        p["multipole_moments"],
        cell_b,
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
        sigma=0.5,
        alpha=0.35,
        kspace_cutoff=6.0,
        batch_idx=batch_idx,
    )
    assert E_batch.shape == (1,)
    assert torch.isfinite(E_batch).all()
    assert abs(float(E_batch[0]) - float(E_single)) / abs(float(E_single)) < 1e-9


def test_multipole_ewald_summation_validates_packed_shape():
    """An unsupported ``multipole_moments`` trailing dim raises ValueError
    (must be 1 / 4 / 9 for l_max 0 / 1 / 2)."""
    p = _make_4atom_pbc()
    bad_mm = torch.zeros(p["positions"].shape[0], 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="multipole_moments last-dim must be"):
        multipole_ewald_summation(
            p["positions"],
            bad_mm,
            p["cell"],
            p["idx_j"],
            p["neighbor_ptr"],
            p["unit_shifts"],
            sigma=0.5,
            alpha=0.35,
            kspace_cutoff=2.0,
        )


def test_multipole_real_space_quadrupole_callable_with_4atom_periodic():
    """Real-space LMAX=2 entry point produces a finite, autograd-connected energy."""
    p = _make_4atom_pbc()
    positions = p["positions"].clone().requires_grad_(True)
    sigma = torch.tensor([0.5], dtype=torch.float64)
    alpha = torch.tensor([0.35], dtype=torch.float64)
    E = multipole_real_space_quadrupole_energy(
        positions,
        p["charges"],
        p["dipoles"],
        p["quadrupoles"],
        p["cell"],
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
        sigma,
        alpha,
    )
    # Real-space LMAX=2 returns per-atom (N,); total is .sum().
    assert E.shape == (p["positions"].shape[0],)
    assert torch.isfinite(E).all()
    (grad_pos,) = torch.autograd.grad(E.sum(), [positions])
    assert torch.isfinite(grad_pos).all()
