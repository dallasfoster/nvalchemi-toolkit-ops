# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Double-backward (create_graph=True) tests for LMAX=2 real-space.

Verifies the create_graph Hessian-vector product against FD of the
1st-order grads.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)


def _two_atom_pair(dtype=torch.float64, device="cpu", half_neighbor_list=False):
    """Build a simple 2-atom non-periodic pair for testing.

    With half_neighbor_list=True, only the (0, 1) edge is listed; with
    False (default), both (0, 1) and (1, 0) are listed.
    """
    rng = np.random.default_rng(20260610)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    q = torch.tensor(
        rng.normal(size=(2,)), dtype=dtype, device=device, requires_grad=True
    )
    mu = torch.tensor(
        rng.normal(size=(2, 3)), dtype=dtype, device=device, requires_grad=True
    )
    Q_raw = rng.normal(size=(2, 3, 3))
    Q_sym = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    Q = torch.tensor(Q_sym, dtype=dtype, device=device, requires_grad=True)
    cell = (
        torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 100.0
    )  # large box → no PBC
    sigma = torch.tensor([0.5], dtype=dtype, device=device)
    alpha = torch.tensor([0.35], dtype=dtype, device=device)
    if half_neighbor_list:
        # Only atom 0 sees atom 1; atom 1 has no neighbors
        idx_j = torch.tensor([1], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
        unit_shifts = torch.zeros((1, 3), dtype=torch.int32, device=device)
    else:
        # Full: atom 0 sees atom 1, atom 1 sees atom 0
        idx_j = torch.tensor([1, 0], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        unit_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
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


def test_quadrupole_2nd_backward_create_graph_true_runs():
    """Smoke test: ``create_graph=True`` no longer raises NotImplementedError."""
    p = _two_atom_pair()
    E = multipole_real_space_quadrupole_energy(**p).sum()
    grads = torch.autograd.grad(
        E,
        [p["positions"], p["charges"], p["dipoles"], p["quadrupoles"]],
        create_graph=True,
    )
    assert all(g.requires_grad for g in grads), (
        "grads from create_graph=True must have requires_grad=True"
    )


def test_quadrupole_2nd_backward_returns_finite():
    """The full ∂²E/∂param² is computable and finite."""
    p = _two_atom_pair()
    E = multipole_real_space_quadrupole_energy(**p).sum()
    (grad_pos,) = torch.autograd.grad(E, [p["positions"]], create_graph=True)
    loss = (grad_pos**2).sum()
    grads_2nd = torch.autograd.grad(
        loss,
        [p["positions"]],
        retain_graph=False,
    )
    g = grads_2nd[0]
    assert torch.isfinite(g).all(), "2nd backward produced non-finite values"
    assert g.abs().max() > 0, "2nd backward should be non-zero"


def test_quadrupole_2nd_backward_fd_match_positions():
    """FD on the 1st-order grad matches the autograd 2nd backward, positions."""
    p = _two_atom_pair()

    def grad_pos_at(positions, create_graph):
        params = {**p, "positions": positions}
        E = multipole_real_space_quadrupole_energy(**params).sum()
        (g,) = torch.autograd.grad(E, [positions], create_graph=create_graph)
        return g

    v = torch.randn_like(p["positions"])
    p_grad = p["positions"]
    p_grad.requires_grad_(True)
    grad_pos = grad_pos_at(p_grad, create_graph=True)
    loss = (grad_pos * v).sum()
    (autograd_hvp,) = torch.autograd.grad(loss, [p_grad])

    h = 1e-5
    fd_hvp = torch.zeros_like(p["positions"])
    pos_base = p["positions"].detach().clone()
    for a in range(p["positions"].shape[0]):
        for k in range(3):
            pos_p_v = pos_base.clone()
            pos_p_v[a, k] = pos_base[a, k] + h
            pos_p_v.requires_grad_(True)
            gp_plus = grad_pos_at(pos_p_v, create_graph=False)
            pos_m_v = pos_base.clone()
            pos_m_v[a, k] = pos_base[a, k] - h
            pos_m_v.requires_grad_(True)
            gp_minus = grad_pos_at(pos_m_v, create_graph=False)
            fd_hvp[a, k] = ((gp_plus - gp_minus) * v).sum() / (2 * h)

    diff = (autograd_hvp - fd_hvp).abs().max().item()
    denom = max(autograd_hvp.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"autograd 2nd backward disagrees with FD: max_abs={diff:.3e}, "
        f"rel={rel:.3e}, autograd={autograd_hvp}, fd={fd_hvp}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_quadrupole_2nd_backward_create_graph_true_runs_cuda():
    """Smoke test on CUDA — exercises tile path."""
    p = _two_atom_pair(device="cuda")
    E = multipole_real_space_quadrupole_energy(**p).sum()
    grads = torch.autograd.grad(
        E,
        [p["positions"], p["charges"], p["dipoles"], p["quadrupoles"]],
        create_graph=True,
    )
    assert all(g.requires_grad for g in grads)


def test_quadrupole_2nd_backward_half_vs_full_neighbor_list_match():
    """Half and full neighbor lists produce the same 2nd-order outputs.

    Invokes the CSR kernel directly to test the half/full plumbing in isolation.
    """
    import warp as wp

    from nvalchemiops.interactions.electrostatics.multipole_ewald_quadrupole_2nd_backward import (
        multipole_real_space_quadrupole_csr_energy_2nd_backward,
    )

    rng = np.random.default_rng(20260611)
    dtype_t = torch.float64

    n = 2
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=dtype_t)
    q = torch.tensor(rng.normal(size=(n,)), dtype=dtype_t)
    mu = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype_t)
    Q_raw = rng.normal(size=(n, 3, 3))
    Q = torch.tensor(0.5 * (Q_raw + Q_raw.transpose(0, 2, 1)), dtype=dtype_t)
    cell = (torch.eye(3, dtype=dtype_t) * 100.0).unsqueeze(0)
    sigma = torch.tensor([0.5], dtype=dtype_t)
    alpha = torch.tensor([0.35], dtype=dtype_t)
    ge = torch.tensor(rng.normal(size=(n,)), dtype=torch.float64)
    gp = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype_t)
    gc = torch.tensor(rng.normal(size=(n,)), dtype=dtype_t)
    gd = torch.tensor(rng.normal(size=(n, 3)), dtype=dtype_t)
    gQ_raw = rng.normal(size=(n, 3, 3))
    gQ = torch.tensor(0.5 * (gQ_raw + gQ_raw.transpose(0, 2, 1)), dtype=dtype_t)

    def run(half_list):
        if half_list:
            idx_j = torch.tensor([1], dtype=torch.int32)
            ptr = torch.tensor([0, 1, 1], dtype=torch.int32)
            sh = torch.zeros((1, 3), dtype=torch.int32)
        else:
            idx_j = torch.tensor([1, 0], dtype=torch.int32)
            ptr = torch.tensor([0, 1, 2], dtype=torch.int32)
            sh = torch.zeros((2, 3), dtype=torch.int32)

        gg_ge_2nd = torch.zeros(n, dtype=torch.float64)
        gg_pos_2nd = torch.zeros((n, 3), dtype=dtype_t)
        gg_q_2nd = torch.zeros(n, dtype=dtype_t)
        gg_mu_2nd = torch.zeros((n, 3), dtype=dtype_t)
        gg_Q_2nd = torch.zeros((n, 3, 3), dtype=dtype_t)

        multipole_real_space_quadrupole_csr_energy_2nd_backward(
            wp.from_torch(pos.contiguous(), dtype=wp.vec3d),
            wp.from_torch(q.contiguous(), dtype=wp.float64),
            wp.from_torch(mu.contiguous(), dtype=wp.vec3d),
            wp.from_torch(Q.contiguous(), dtype=wp.mat33d),
            wp.from_torch(cell.contiguous(), dtype=wp.mat33d),
            wp.from_torch(idx_j.contiguous(), dtype=wp.int32),
            wp.from_torch(ptr.contiguous(), dtype=wp.int32),
            wp.from_torch(sh.contiguous(), dtype=wp.vec3i),
            wp.from_torch(sigma.contiguous(), dtype=wp.float64),
            wp.from_torch(alpha.contiguous(), dtype=wp.float64),
            wp.from_torch(ge.contiguous(), dtype=wp.float64),
            wp.from_torch(gp.contiguous(), dtype=wp.vec3d),
            wp.from_torch(gc.contiguous(), dtype=wp.float64),
            wp.from_torch(gd.contiguous(), dtype=wp.vec3d),
            wp.from_torch(gQ.contiguous(), dtype=wp.mat33d),
            wp.from_torch(gg_ge_2nd, dtype=wp.float64),
            wp.from_torch(gg_pos_2nd, dtype=wp.vec3d),
            wp.from_torch(gg_q_2nd, dtype=wp.float64),
            wp.from_torch(gg_mu_2nd, dtype=wp.vec3d),
            wp.from_torch(gg_Q_2nd, dtype=wp.mat33d),
            device="cpu",
            half_neighbor_list=half_list,
        )
        return dict(
            ge=gg_ge_2nd.clone(),
            pos=gg_pos_2nd.clone(),
            q=gg_q_2nd.clone(),
            mu=gg_mu_2nd.clone(),
            Q=gg_Q_2nd.clone(),
        )

    out_full = run(half_list=False)
    out_half = run(half_list=True)

    for key in ("ge", "pos", "q", "mu", "Q"):
        diff = (out_full[key] - out_half[key]).abs().max().item()
        denom = max(out_full[key].abs().max().item(), 1e-12)
        rel = diff / denom
        assert rel < 1e-12, (
            f"{key}: half vs full disagree (rel={rel:.2e}). "
            f"full={out_full[key]}, half={out_half[key]}"
        )


@pytest.mark.parametrize("channel", ["charges", "dipoles"])
def test_quadrupole_2nd_backward_fd_match_other_channels(channel):
    """FD on the 1st-order grad matches autograd 2nd backward for non-position channels."""
    p = _two_atom_pair()
    x_ref = p[channel]

    def grad_x_at(x, create_graph):
        params = {**p, channel: x}
        E = multipole_real_space_quadrupole_energy(**params).sum()
        (g,) = torch.autograd.grad(E, [x], create_graph=create_graph)
        return g

    v = torch.randn_like(x_ref)
    x = x_ref.detach().clone().requires_grad_(True)
    grad_x = grad_x_at(x, create_graph=True)
    loss = (grad_x * v).sum()
    (autograd_hvp,) = torch.autograd.grad(loss, [x])

    h = 1e-5
    fd_hvp = torch.zeros_like(x_ref)
    flat = fd_hvp.view(-1)
    base = x_ref.detach().clone()
    for k in range(flat.shape[0]):
        x_p = base.clone()
        x_p.view(-1)[k] = base.view(-1)[k] + h
        x_p.requires_grad_(True)
        gp = grad_x_at(x_p, create_graph=False)
        x_m = base.clone()
        x_m.view(-1)[k] = base.view(-1)[k] - h
        x_m.requires_grad_(True)
        gm = grad_x_at(x_m, create_graph=False)
        flat[k] = ((gp - gm) * v).sum() / (2 * h)

    diff = (autograd_hvp - fd_hvp).abs().max().item()
    denom = max(autograd_hvp.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"{channel}: autograd 2nd backward disagrees with FD: "
        f"max_abs={diff:.3e}, rel={rel:.3e}"
    )


def test_quadrupole_2nd_backward_fd_match_quadrupoles_symmetric():
    """FD-match for Q channel — symmetric perturbation matching the kernel's
    symmetric-Q contract."""
    p = _two_atom_pair()
    Q_ref = p["quadrupoles"]

    def grad_Q_at(Q, create_graph):
        params = {**p, "quadrupoles": Q}
        E = multipole_real_space_quadrupole_energy(**params).sum()
        (g,) = torch.autograd.grad(E, [Q], create_graph=create_graph)
        return g

    # v must be symmetric to match the kernel's symmetric-Q output convention.
    v_raw = torch.randn_like(Q_ref)
    v = 0.5 * (v_raw + v_raw.transpose(-1, -2))
    Q = Q_ref.detach().clone().requires_grad_(True)
    grad_Q = grad_Q_at(Q, create_graph=True)
    loss = (grad_Q * v).sum()
    (autograd_hvp,) = torch.autograd.grad(loss, [Q])

    # Symmetric perturbation: perturb [a,b] and [b,a] together by h; off-diagonal
    # entries divide by 2 to match the kernel's symmetric free-index emit.
    h = 1e-5
    fd_hvp = torch.zeros_like(Q_ref)
    base = Q_ref.detach().clone()
    n_atoms = base.shape[0]
    for n in range(n_atoms):
        for a in range(3):
            for b in range(3):
                Q_p = base.clone()
                Q_p[n, a, b] = base[n, a, b] + h
                if a != b:
                    Q_p[n, b, a] = base[n, b, a] + h
                Q_p.requires_grad_(True)
                gp = grad_Q_at(Q_p, create_graph=False)
                Q_m = base.clone()
                Q_m[n, a, b] = base[n, a, b] - h
                if a != b:
                    Q_m[n, b, a] = base[n, b, a] - h
                Q_m.requires_grad_(True)
                gm = grad_Q_at(Q_m, create_graph=False)
                fd_val = ((gp - gm) * v).sum() / (2 * h)
                if a == b:
                    fd_hvp[n, a, b] = fd_val
                else:
                    fd_hvp[n, a, b] = 0.5 * fd_val
                    fd_hvp[n, b, a] = 0.5 * fd_val

    diff = (autograd_hvp - fd_hvp).abs().max().item()
    denom = max(autograd_hvp.abs().max().item(), 1e-10)
    rel = diff / denom
    assert rel < 5e-4, (
        f"quadrupoles: autograd 2nd backward disagrees with symmetric FD: "
        f"max_abs={diff:.3e}, rel={rel:.3e}"
    )


def test_quadrupole_batched_2nd_backward_matches_per_system():
    """Batched LMAX=2 second-order backward (force-loss create_graph) equals the
    per-system single-system 2nd-back, atom-for-atom."""
    from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
        multipole_real_space_quadrupole_energy,
    )

    dev = "cpu"
    s0 = _two_atom_pair(device=dev)
    rng = np.random.default_rng(7)
    s1 = _two_atom_pair(device=dev)
    nps = 2
    pos = torch.cat(
        [s0["positions"].detach(), s1["positions"].detach()], 0
    ).requires_grad_(True)
    q = torch.cat([s0["charges"].detach(), s1["charges"].detach()], 0)
    mu = torch.cat([s0["dipoles"].detach(), s1["dipoles"].detach()], 0)
    Q = torch.cat([s0["quadrupoles"].detach(), s1["quadrupoles"].detach()], 0)
    cells = torch.cat([s0["cell"], s1["cell"]], 0)
    sigmas = torch.tensor([0.5, 0.5], dtype=torch.float64, device=dev)
    alphas = torch.tensor([0.35, 0.35], dtype=torch.float64, device=dev)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=dev)
    idx_j = torch.tensor([1, 0, 3, 2], dtype=torch.int32, device=dev)
    ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=dev)
    sh = torch.zeros((4, 3), dtype=torch.int32, device=dev)
    v = torch.tensor(rng.normal(size=(2 * nps, 3)), dtype=torch.float64, device=dev)

    E = multipole_real_space_quadrupole_energy(
        pos,
        q,
        mu,
        Q,
        cells,
        idx_j,
        ptr,
        sh,
        sigmas,
        alphas,
        batch_idx=batch_idx,
    ).sum()
    forces = -torch.autograd.grad(E, pos, create_graph=True)[0]
    (gb,) = torch.autograd.grad((forces * v).sum(), [pos])

    # Per-system single-system 2nd-back.
    off = 0
    for b, s in enumerate([s0, s1]):
        sl = slice(off, off + nps)
        off += nps
        ps = s["positions"].detach().clone().requires_grad_(True)
        Es = multipole_real_space_quadrupole_energy(
            ps,
            s["charges"].detach(),
            s["dipoles"].detach(),
            s["quadrupoles"].detach(),
            s["cell"],
            s["idx_j"],
            s["neighbor_ptr"],
            s["unit_shifts"],
            s["sigma"],
            s["alpha"],
        ).sum()
        Fs = -torch.autograd.grad(Es, ps, create_graph=True)[0]
        (gs,) = torch.autograd.grad((Fs * v[sl]).sum(), [ps])
        rel = (gb[sl] - gs).abs().max() / (gs.abs().max() + 1e-30)
        assert rel < 1e-10, f"system {b}: batched 2nd-back rel = {rel:.3e}"
