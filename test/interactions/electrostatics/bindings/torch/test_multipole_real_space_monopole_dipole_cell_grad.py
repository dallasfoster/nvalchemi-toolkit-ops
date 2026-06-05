# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-level cell-gradient tests for LMAX=0 and LMAX=1.

FD oracle: differentiate the CSR energy launcher w.r.t. cell entries.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

from nvalchemiops.interactions.electrostatics.multipole_ewald_cell_grad import (
    multipole_real_space_dipole_csr_cell_grad,
    multipole_real_space_monopole_csr_cell_grad,
)
from nvalchemiops.interactions.electrostatics.multipole_ewald_kernels import (
    multipole_real_space_dipole_csr_energy_fused,
    multipole_real_space_monopole_csr_energy_fused,
)


def _periodic_4atom(dtype=torch.float64):
    rng = np.random.default_rng(20260614)
    n = 4
    L = 6.0
    pos = torch.tensor(rng.uniform(0.0, L, size=(n, 3)), dtype=dtype)
    q = torch.tensor(rng.normal(size=(n,)), dtype=dtype)
    mu = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype)
    cell = torch.tensor(
        [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
        dtype=dtype,
    ).unsqueeze(0)
    sigma = torch.tensor([0.5], dtype=dtype)
    alpha = torch.tensor([0.35], dtype=dtype)

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
    )
    neighbor_ptr = torch.tensor(ptr, dtype=torch.int32)
    unit_shifts = torch.tensor(
        [s for shifts in shifts_list for s in shifts],
        dtype=torch.int32,
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


def _launch_monopole_grad_cell(p, half_neighbor_list=False):
    """Launch the LMAX=0 cell-grad kernel and return the (1,3,3) grad."""
    grad_cell = torch.zeros_like(p["cell"]).contiguous()
    multipole_real_space_monopole_csr_cell_grad(
        wp.from_torch(p["positions"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["charges"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["cell"].contiguous(), dtype=wp.mat33d),
        wp.from_torch(p["idx_j"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["neighbor_ptr"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["unit_shifts"].contiguous(), dtype=wp.vec3i),
        wp.from_torch(p["sigma"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["alpha"].contiguous(), dtype=wp.float64),
        wp.from_torch(grad_cell, dtype=wp.mat33d),
        device="cpu",
        half_neighbor_list=half_neighbor_list,
    )
    return grad_cell


def _launch_monopole_energy_total(p):
    """Launch LMAX=0 fused energy kernel and return scalar total energy."""
    n = p["positions"].shape[0]
    energies = torch.zeros(n, dtype=torch.float64)
    grad_pos = torch.zeros((n, 3), dtype=torch.float64)
    grad_q = torch.zeros(n, dtype=torch.float64)
    multipole_real_space_monopole_csr_energy_fused(
        wp.from_torch(p["positions"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["charges"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["cell"].contiguous(), dtype=wp.mat33d),
        wp.from_torch(p["idx_j"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["neighbor_ptr"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["unit_shifts"].contiguous(), dtype=wp.vec3i),
        wp.from_torch(p["sigma"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["alpha"].contiguous(), dtype=wp.float64),
        wp.from_torch(energies, dtype=wp.float64),
        wp.from_torch(grad_pos, dtype=wp.vec3d),
        wp.from_torch(grad_q, dtype=wp.float64),
        with_pos_grad=False,
        with_charge_grad=False,
        wp_dtype=wp.float64,
        device="cpu",
    )
    return energies.sum().item()


def test_monopole_cell_grad_matches_fd():
    """LMAX=0 cell-grad kernel matches FD on the energy kernel."""
    p = _periodic_4atom()
    grad_an = _launch_monopole_grad_cell(p)

    h = 1e-5
    grad_fd = torch.zeros_like(p["cell"])
    cell0 = p["cell"].clone()
    for a in range(3):
        for b in range(3):
            cell_p = cell0.clone()
            cell_p[0, a, b] += h
            E_p = _launch_monopole_energy_total({**p, "cell": cell_p})
            cell_m = cell0.clone()
            cell_m[0, a, b] -= h
            E_m = _launch_monopole_energy_total({**p, "cell": cell_m})
            grad_fd[0, a, b] = (E_p - E_m) / (2 * h)

    diff = (grad_an - grad_fd).abs().max().item()
    denom = max(grad_an.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"LMAX=0 cell grad mismatch rel={rel:.3e}\nanalytical={grad_an}\nFD={grad_fd}"
    )


def _launch_dipole_grad_cell(p, half_neighbor_list=False):
    grad_cell = torch.zeros_like(p["cell"]).contiguous()
    multipole_real_space_dipole_csr_cell_grad(
        wp.from_torch(p["positions"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["charges"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["dipoles"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["cell"].contiguous(), dtype=wp.mat33d),
        wp.from_torch(p["idx_j"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["neighbor_ptr"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["unit_shifts"].contiguous(), dtype=wp.vec3i),
        wp.from_torch(p["sigma"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["alpha"].contiguous(), dtype=wp.float64),
        wp.from_torch(grad_cell, dtype=wp.mat33d),
        device="cpu",
        half_neighbor_list=half_neighbor_list,
    )
    return grad_cell


def _launch_dipole_energy_total(p):
    n = p["positions"].shape[0]
    energies = torch.zeros(n, dtype=torch.float64)
    grad_pos = torch.zeros((n, 3), dtype=torch.float64)
    grad_q = torch.zeros(n, dtype=torch.float64)
    grad_mu = torch.zeros((n, 3), dtype=torch.float64)
    multipole_real_space_dipole_csr_energy_fused(
        wp.from_torch(p["positions"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["charges"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["dipoles"].contiguous(), dtype=wp.vec3d),
        wp.from_torch(p["cell"].contiguous(), dtype=wp.mat33d),
        wp.from_torch(p["idx_j"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["neighbor_ptr"].contiguous(), dtype=wp.int32),
        wp.from_torch(p["unit_shifts"].contiguous(), dtype=wp.vec3i),
        wp.from_torch(p["sigma"].contiguous(), dtype=wp.float64),
        wp.from_torch(p["alpha"].contiguous(), dtype=wp.float64),
        wp.from_torch(energies, dtype=wp.float64),
        wp.from_torch(grad_pos, dtype=wp.vec3d),
        wp.from_torch(grad_q, dtype=wp.float64),
        wp.from_torch(grad_mu, dtype=wp.vec3d),
        with_pos_grad=False,
        with_charge_grad=False,
        with_dipole_grad=False,
        wp_dtype=wp.float64,
        device="cpu",
    )
    return energies.sum().item()


def test_dipole_cell_grad_matches_fd():
    """LMAX=1 cell-grad kernel matches FD on the energy kernel."""
    p = _periodic_4atom()
    grad_an = _launch_dipole_grad_cell(p)

    h = 1e-5
    grad_fd = torch.zeros_like(p["cell"])
    cell0 = p["cell"].clone()
    for a in range(3):
        for b in range(3):
            cell_p = cell0.clone()
            cell_p[0, a, b] += h
            E_p = _launch_dipole_energy_total({**p, "cell": cell_p})
            cell_m = cell0.clone()
            cell_m[0, a, b] -= h
            E_m = _launch_dipole_energy_total({**p, "cell": cell_m})
            grad_fd[0, a, b] = (E_p - E_m) / (2 * h)

    diff = (grad_an - grad_fd).abs().max().item()
    denom = max(grad_an.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"LMAX=1 cell grad mismatch rel={rel:.3e}\nanalytical={grad_an}\nFD={grad_fd}"
    )


@pytest.mark.parametrize("lmax", [0, 1])
def test_monopole_dipole_half_vs_full_neighbor_list_match(lmax):
    """Half-list with scale=1.0 produces same output as full-list with scale=0.5."""
    p_full = _periodic_4atom()
    if lmax == 0:
        grad_full = _launch_monopole_grad_cell(p_full, half_neighbor_list=False)
    else:
        grad_full = _launch_dipole_grad_cell(p_full, half_neighbor_list=False)

    # Build a half list: keep only edges (i, j) with j > i (or j == i and shift > 0).
    nbptr_np = p_full["neighbor_ptr"].cpu().numpy()
    idx_j_np = p_full["idx_j"].cpu().numpy()
    sh_np = p_full["unit_shifts"].cpu().numpy()

    half_idx_j: list[int] = []
    half_shifts: list[tuple[int, int, int]] = []
    half_counts: list[int] = [0] * len(p_full["positions"])
    for i in range(len(p_full["positions"])):
        k_start, k_end = int(nbptr_np[i]), int(nbptr_np[i + 1])
        for k in range(k_start, k_end):
            j = int(idx_j_np[k])
            s = tuple(int(x) for x in sh_np[k])
            # canonical: (j > i) OR (j == i AND shift > (0,0,0))
            if j > i or (j == i and s > (0, 0, 0)):
                half_idx_j.append(j)
                half_shifts.append(s)
                half_counts[i] += 1
    half_ptr = np.cumsum([0] + half_counts)
    p_half = dict(p_full)
    p_half["idx_j"] = torch.tensor(half_idx_j, dtype=torch.int32)
    p_half["neighbor_ptr"] = torch.tensor(half_ptr, dtype=torch.int32)
    p_half["unit_shifts"] = torch.tensor(half_shifts, dtype=torch.int32).reshape(-1, 3)

    if lmax == 0:
        grad_half = _launch_monopole_grad_cell(p_half, half_neighbor_list=True)
    else:
        grad_half = _launch_dipole_grad_cell(p_half, half_neighbor_list=True)

    diff = (grad_full - grad_half).abs().max().item()
    denom = max(grad_full.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 1e-12, (
        f"LMAX={lmax}: half vs full disagree, rel={rel:.3e}\n"
        f"full={grad_full}\nhalf={grad_half}"
    )
