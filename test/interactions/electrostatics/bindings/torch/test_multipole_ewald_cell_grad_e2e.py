# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end cell-grad tests through the LMAX=0/1
``MultipoleRealSpace*FusedScalarFunction`` wrappers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    BatchMultipoleRealSpaceDipoleFusedScalarFunction,
    BatchMultipoleRealSpaceMonopoleFusedScalarFunction,
    MultipoleRealSpaceDipoleFusedScalarFunction,
    MultipoleRealSpaceMonopoleFusedScalarFunction,
)


def _periodic_4atom(dtype=torch.float64, device="cpu"):
    rng = np.random.default_rng(20260615)
    n = 4
    L = 6.0
    pos = torch.tensor(rng.uniform(0.0, L, size=(n, 3)), dtype=dtype, device=device)
    q = torch.tensor(rng.normal(size=(n,)), dtype=dtype, device=device)
    mu = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype, device=device)
    cell = torch.tensor(
        [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
        dtype=dtype,
        device=device,
    ).unsqueeze(0)
    sigma = torch.tensor([0.5], dtype=dtype, device=device)
    alpha = torch.tensor([0.35], dtype=dtype, device=device)
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
    return dict(
        positions=pos,
        charges=q,
        dipoles=mu,
        cell=cell,
        sigma=sigma,
        alpha=alpha,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
    )


def test_monopole_fused_scalar_cell_grad_via_autograd():
    """MultipoleRealSpaceMonopoleFusedScalarFunction propagates cell-grad."""
    p = _periodic_4atom()
    cell = p["cell"].clone().requires_grad_(True)
    E = MultipoleRealSpaceMonopoleFusedScalarFunction.apply(
        p["positions"],
        p["charges"],
        cell,
        p["sigma"],
        p["alpha"],
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
    )
    (grad_cell,) = torch.autograd.grad(E, [cell])
    assert grad_cell.shape == (1, 3, 3)
    assert torch.isfinite(grad_cell).all()
    assert grad_cell.abs().max() > 0


def test_dipole_fused_scalar_cell_grad_via_autograd():
    """MultipoleRealSpaceDipoleFusedScalarFunction propagates cell-grad."""
    p = _periodic_4atom()
    cell = p["cell"].clone().requires_grad_(True)
    E = MultipoleRealSpaceDipoleFusedScalarFunction.apply(
        p["positions"],
        p["charges"],
        p["dipoles"],
        cell,
        p["sigma"],
        p["alpha"],
        p["idx_j"],
        p["neighbor_ptr"],
        p["unit_shifts"],
    )
    (grad_cell,) = torch.autograd.grad(E, [cell])
    assert grad_cell.shape == (1, 3, 3)
    assert torch.isfinite(grad_cell).all()
    assert grad_cell.abs().max() > 0


@pytest.mark.parametrize("lmax", [0, 1])
def test_fused_scalar_cell_grad_matches_fd(lmax):
    """Cell-grad through the FusedScalar wrapper matches FD on the energy."""
    p = _periodic_4atom()
    if lmax == 0:
        Fn = MultipoleRealSpaceMonopoleFusedScalarFunction

        def forward(c):
            return Fn.apply(
                p["positions"],
                p["charges"],
                c,
                p["sigma"],
                p["alpha"],
                p["idx_j"],
                p["neighbor_ptr"],
                p["unit_shifts"],
            )
    else:
        Fn = MultipoleRealSpaceDipoleFusedScalarFunction

        def forward(c):
            return Fn.apply(
                p["positions"],
                p["charges"],
                p["dipoles"],
                c,
                p["sigma"],
                p["alpha"],
                p["idx_j"],
                p["neighbor_ptr"],
                p["unit_shifts"],
            )

    cell0 = p["cell"].detach().clone()
    cell_grad_in = cell0.clone().requires_grad_(True)
    E = forward(cell_grad_in)
    (grad_an,) = torch.autograd.grad(E, [cell_grad_in])

    h = 1e-5
    grad_fd = torch.zeros_like(cell0)
    for a in range(3):
        for b in range(3):
            c_p = cell0.clone()
            c_p[0, a, b] += h
            E_p = forward(c_p).item()
            c_m = cell0.clone()
            c_m[0, a, b] -= h
            E_m = forward(c_m).item()
            grad_fd[0, a, b] = (E_p - E_m) / (2 * h)
    diff = (grad_an - grad_fd).abs().max().item()
    denom = max(grad_an.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, f"LMAX={lmax}: rel_err={rel:.3e}\nan={grad_an}\nfd={grad_fd}"


def _periodic_4atom_batched_2sys(dtype=torch.float64):
    """Two periodic systems with the same atom count, flat-packed."""
    p1 = _periodic_4atom(dtype=dtype)
    p2 = _periodic_4atom(dtype=dtype)
    n1 = p1["positions"].shape[0]
    n2 = p2["positions"].shape[0]

    positions = torch.cat([p1["positions"], p2["positions"]], dim=0)
    charges = torch.cat([p1["charges"], p2["charges"]], dim=0)
    dipoles = torch.cat([p1["dipoles"], p2["dipoles"]], dim=0)
    cells = torch.cat([p1["cell"], p2["cell"]], dim=0)  # (2, 3, 3)
    sigmas = torch.tensor([0.5, 0.5], dtype=dtype)
    alphas = torch.tensor([0.35, 0.35], dtype=dtype)
    batch_idx = torch.cat(
        [
            torch.zeros(n1, dtype=torch.int32),
            torch.ones(n2, dtype=torch.int32),
        ]
    )
    # Flat CSR — offset system 2's idx_j by n1.
    idx_j = torch.cat(
        [
            p1["idx_j"],
            p2["idx_j"] + n1,
        ]
    )
    nptr1 = p1["neighbor_ptr"]
    nptr2 = p2["neighbor_ptr"]
    neighbor_ptr = torch.cat(
        [
            nptr1,
            nptr2[1:] + nptr1[-1],
        ]
    )
    unit_shifts = torch.cat([p1["unit_shifts"], p2["unit_shifts"]], dim=0)
    return dict(
        positions=positions,
        charges=charges,
        dipoles=dipoles,
        cells=cells,
        sigmas=sigmas,
        alphas=alphas,
        idx_j=idx_j,
        neighbor_ptr=neighbor_ptr,
        unit_shifts=unit_shifts,
        batch_idx=batch_idx,
    )


@pytest.mark.parametrize("lmax", [0, 1])
def test_batch_fused_scalar_cell_grad_matches_fd(lmax):
    """Batched cell-grad matches FD per-system."""
    p = _periodic_4atom_batched_2sys()
    if lmax == 0:
        Fn = BatchMultipoleRealSpaceMonopoleFusedScalarFunction

        def forward(c):
            return Fn.apply(
                p["positions"],
                p["charges"],
                c,
                p["sigmas"],
                p["alphas"],
                p["idx_j"],
                p["neighbor_ptr"],
                p["unit_shifts"],
                p["batch_idx"],
            )
    else:
        Fn = BatchMultipoleRealSpaceDipoleFusedScalarFunction

        def forward(c):
            return Fn.apply(
                p["positions"],
                p["charges"],
                p["dipoles"],
                c,
                p["sigmas"],
                p["alphas"],
                p["idx_j"],
                p["neighbor_ptr"],
                p["unit_shifts"],
                p["batch_idx"],
            )

    cell0 = p["cells"].detach().clone()
    cells_grad_in = cell0.clone().requires_grad_(True)
    E_per_system = forward(cells_grad_in)  # (B,)
    # Sum over batch to get scalar; the per-system grad_cell is unweighted = 1 each
    total = E_per_system.sum()
    (grad_an,) = torch.autograd.grad(total, [cells_grad_in])
    # FD on each (system_b, a, b) entry
    h = 1e-5
    grad_fd = torch.zeros_like(cell0)
    B = cell0.shape[0]
    for b_sys in range(B):
        for a in range(3):
            for b in range(3):
                c_p = cell0.clone()
                c_p[b_sys, a, b] += h
                E_p = forward(c_p).sum().item()
                c_m = cell0.clone()
                c_m[b_sys, a, b] -= h
                E_m = forward(c_m).sum().item()
                grad_fd[b_sys, a, b] = (E_p - E_m) / (2 * h)
    diff = (grad_an - grad_fd).abs().max().item()
    denom = max(grad_an.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"batched LMAX={lmax}: rel_err={rel:.3e}\nan={grad_an}\nfd={grad_fd}"
    )
