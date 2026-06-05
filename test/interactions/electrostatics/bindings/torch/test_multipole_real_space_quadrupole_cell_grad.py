# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cell-gradient tests for LMAX=2 real-space Ewald."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)


def _periodic_pair(dtype=torch.float64, device="cpu"):
    """4-atom periodic system with PBC shifts so cell appears in the math."""
    rng = np.random.default_rng(20260612)
    n = 4
    L = 6.0
    pos = torch.tensor(
        rng.uniform(0.0, L, size=(n, 3)),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    q = torch.tensor(
        rng.normal(size=(n,)), dtype=dtype, device=device, requires_grad=True
    )
    mu = torch.tensor(
        rng.normal(size=(n, 3)), dtype=dtype, device=device, requires_grad=True
    )
    Q_raw = rng.normal(size=(n, 3, 3))
    Q_sym = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    Q = torch.tensor(Q_sym, dtype=dtype, device=device, requires_grad=True)
    cell = (
        torch.tensor(
            [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
            dtype=dtype,
            device=device,
        )
        .unsqueeze(0)
        .clone()
    )
    cell.requires_grad_(True)
    sigma = torch.tensor([0.5], dtype=dtype, device=device)
    alpha = torch.tensor([0.35], dtype=dtype, device=device)

    cutoff = 4.0
    idx_j_list = []
    shifts_list = []
    pos_np = pos.detach().cpu().numpy()
    for i in range(n):
        nbrs_i = []
        shifts_i = []
        for j in range(n):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sz in (-1, 0, 1):
                        if i == j and sx == 0 and sy == 0 and sz == 0:
                            continue
                        disp = pos_np[j] - pos_np[i] + np.array([sx, sy, sz]) * L
                        if np.linalg.norm(disp) < cutoff:
                            nbrs_i.append(j)
                            shifts_i.append((sx, sy, sz))
        idx_j_list.append(nbrs_i)
        shifts_list.append(shifts_i)

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

    return dict(
        positions=pos,
        charges=q,
        dipoles=mu,
        quadrupoles=Q,
        cell=cell,
        sigma=sigma,
        alpha=alpha,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )


def test_quadrupole_cell_grad_returns_finite():
    """Smoke test: cell.requires_grad=True produces a finite ∂E/∂cell."""
    p = _periodic_pair()
    E = multipole_real_space_quadrupole_energy(**p).sum()
    (grad_cell,) = torch.autograd.grad(E, [p["cell"]])
    assert grad_cell.shape == (1, 3, 3)
    assert torch.isfinite(grad_cell).all()
    assert grad_cell.abs().max() > 0


def test_quadrupole_cell_grad_matches_fd():
    """∂E/∂cell matches FD on cell entries."""
    p = _periodic_pair()
    cell0 = p["cell"].detach().clone()

    p_an = {**p, "cell": cell0.clone().requires_grad_(True)}
    E_an = multipole_real_space_quadrupole_energy(**p_an).sum()
    (grad_cell_an,) = torch.autograd.grad(E_an, [p_an["cell"]])

    h = 1e-5
    grad_cell_fd = torch.zeros_like(cell0)
    for a in range(3):
        for b in range(3):
            cell_p = cell0.clone()
            cell_p[0, a, b] += h
            cell_p.requires_grad_(False)
            p_p = {
                **p,
                "cell": cell_p,
                "positions": p["positions"].detach().clone(),
                "charges": p["charges"].detach().clone(),
                "dipoles": p["dipoles"].detach().clone(),
                "quadrupoles": p["quadrupoles"].detach().clone(),
            }
            E_p = multipole_real_space_quadrupole_energy(**p_p).sum().item()

            cell_m = cell0.clone()
            cell_m[0, a, b] -= h
            cell_m.requires_grad_(False)
            p_m = {
                **p,
                "cell": cell_m,
                "positions": p["positions"].detach().clone(),
                "charges": p["charges"].detach().clone(),
                "dipoles": p["dipoles"].detach().clone(),
                "quadrupoles": p["quadrupoles"].detach().clone(),
            }
            E_m = multipole_real_space_quadrupole_energy(**p_m).sum().item()

            grad_cell_fd[0, a, b] = (E_p - E_m) / (2 * h)

    diff = (grad_cell_an - grad_cell_fd).abs().max().item()
    denom = max(grad_cell_an.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"cell grad mismatch: max_abs={diff:.3e}, rel={rel:.3e}\n"
        f"analytical={grad_cell_an}\nFD={grad_cell_fd}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_quadrupole_cell_grad_cuda():
    """Smoke test on CUDA — exercises tile cell-grad path."""
    p = _periodic_pair(device="cuda")
    E = multipole_real_space_quadrupole_energy(**p).sum()
    (grad_cell,) = torch.autograd.grad(E, [p["cell"]])
    assert torch.isfinite(grad_cell).all()
    assert grad_cell.abs().max() > 0
