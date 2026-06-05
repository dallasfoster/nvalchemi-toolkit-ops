"""Smoke test for the LMAX=2 torch wrapper.

Forward returns a per-atom ``(N,)`` vector (``.sum()`` is the total energy);
backward returns finite gradients matching FD on the wrapper itself; the
batched variant returns ``(B,)`` per-system energies. Full FD correctness is
covered at the kernel level elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)


def _csr_neighbors_for_n_atoms(positions: torch.Tensor, cutoff: float):
    """O(N²) toy CSR neighbor list for the smoke test."""
    n = positions.shape[0]
    device = positions.device
    pairs_i = []
    pairs_j = []
    shifts = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = torch.norm(positions[i] - positions[j]).item()
            if d < cutoff:
                pairs_i.append(i)
                pairs_j.append(j)
                shifts.append([0, 0, 0])
    # Sort by (i, j) to build CSR.
    order = sorted(range(len(pairs_i)), key=lambda k: (pairs_i[k], pairs_j[k]))
    pairs_i = [pairs_i[k] for k in order]
    pairs_j = [pairs_j[k] for k in order]
    shifts = [shifts[k] for k in order]

    idx_j = torch.tensor(pairs_j, dtype=torch.int32, device=device)
    neighbor_ptr = torch.zeros(n + 1, dtype=torch.int32, device=device)
    for ii in pairs_i:
        neighbor_ptr[ii + 1] += 1
    neighbor_ptr = torch.cumsum(neighbor_ptr, dim=0).to(torch.int32)
    unit_shifts = torch.tensor(shifts, dtype=torch.int32, device=device)
    return idx_j, neighbor_ptr, unit_shifts


def _symmetric_random_Q(n, rng):
    Q = np.zeros((n, 3, 3))
    for k in range(n):
        raw = rng.normal(size=(3, 3))
        Q[k] = 0.5 * (raw + raw.T)
    return Q


def _build_inputs(
    device,
    n_atoms=4,
    sigma=0.5,
    alpha=0.35,
    positions_grad=True,
    charges_grad=True,
    dipoles_grad=True,
    quadrupoles_grad=True,
):
    rng = np.random.default_rng(20260606)
    pos = rng.uniform(0, 5, size=(n_atoms, 3))
    q = rng.normal(size=(n_atoms,))
    mu = rng.normal(size=(n_atoms, 3))
    Q = _symmetric_random_Q(n_atoms, rng)
    cell = np.eye(3) * 20.0

    pos_t = torch.tensor(
        pos, dtype=torch.float64, device=device, requires_grad=positions_grad
    )
    q_t = torch.tensor(
        q, dtype=torch.float64, device=device, requires_grad=charges_grad
    )
    mu_t = torch.tensor(
        mu, dtype=torch.float64, device=device, requires_grad=dipoles_grad
    )
    Q_t = torch.tensor(
        Q, dtype=torch.float64, device=device, requires_grad=quadrupoles_grad
    )
    cell_t = torch.tensor(cell, dtype=torch.float64, device=device).unsqueeze(0)
    sigma_t = torch.tensor([sigma], dtype=torch.float64, device=device)
    alpha_t = torch.tensor([alpha], dtype=torch.float64, device=device)

    cutoff = 10.0
    idx_j, neighbor_ptr, unit_shifts = _csr_neighbors_for_n_atoms(pos_t, cutoff)
    return (
        pos_t,
        q_t,
        mu_t,
        Q_t,
        cell_t,
        sigma_t,
        alpha_t,
        idx_j,
        neighbor_ptr,
        unit_shifts,
    )


def test_quadrupole_forward_returns_per_atom():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    pos, q, mu, Q, cell, sigma, alpha, idx_j, np_ptr, sh = _build_inputs(device)
    E = multipole_real_space_quadrupole_energy(
        pos, q, mu, Q, cell, idx_j, np_ptr, sh, sigma, alpha
    )
    assert E.shape == (pos.shape[0],), (
        f"expected per-atom (N,); got shape {tuple(E.shape)}"
    )
    assert torch.isfinite(E).all(), f"non-finite energy: {E}"


def test_quadrupole_backward_returns_gradients_for_all_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    pos, q, mu, Q, cell, sigma, alpha, idx_j, np_ptr, sh = _build_inputs(device)
    E = multipole_real_space_quadrupole_energy(
        pos, q, mu, Q, cell, idx_j, np_ptr, sh, sigma, alpha
    )
    E.sum().backward()
    for name, t in [("pos", pos), ("q", q), ("mu", mu), ("Q", Q)]:
        assert t.grad is not None, f"no gradient on {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite grad on {name}"


def test_quadrupole_backward_matches_fd_at_loose_tol():
    """FD check on positions to confirm autograd direction is correct.

    Loose tol (1e-4): fp64 with FD step h=1e-3 (truncation ~1e-6), just an
    order-of-magnitude check — full correctness validated at the kernel level.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    pos, q, mu, Q, cell, sigma, alpha, idx_j, np_ptr, sh = _build_inputs(device)
    E = multipole_real_space_quadrupole_energy(
        pos, q, mu, Q, cell, idx_j, np_ptr, sh, sigma, alpha
    )
    E.sum().backward()

    h = 1e-3
    grad_pos_an = pos.grad.detach().clone()
    pos_d = pos.detach().clone()
    for atom_k in range(min(2, pos.shape[0])):
        for d in range(3):
            pos_plus = pos_d.clone()
            pos_plus[atom_k, d] += h
            pos_minus = pos_d.clone()
            pos_minus[atom_k, d] -= h
            E_plus = multipole_real_space_quadrupole_energy(
                pos_plus,
                q.detach(),
                mu.detach(),
                Q.detach(),
                cell,
                idx_j,
                np_ptr,
                sh,
                sigma,
                alpha,
            ).sum()
            E_minus = multipole_real_space_quadrupole_energy(
                pos_minus,
                q.detach(),
                mu.detach(),
                Q.detach(),
                cell,
                idx_j,
                np_ptr,
                sh,
                sigma,
                alpha,
            ).sum()
            fd = ((E_plus - E_minus) / (2 * h)).item()
            assert abs(grad_pos_an[atom_k, d].item() - fd) < 1e-4, (
                f"atom {atom_k} d{d}: an={grad_pos_an[atom_k, d].item():+.4e} "
                f"fd={fd:+.4e}"
            )


def test_batch_quadrupole_forward_returns_per_system_energies():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    pos0, q0, mu0, Q0, cell0, sigma, alpha, idx_j0, ptr0, sh0 = _build_inputs(device)
    pos1, q1, mu1, Q1, cell1, _, _, idx_j1, ptr1, sh1 = _build_inputs(device)

    n0 = pos0.shape[0]
    n1 = pos1.shape[0]
    pos_b = torch.cat([pos0.detach(), pos1.detach()], dim=0).requires_grad_(True)
    q_b = torch.cat([q0.detach(), q1.detach()], dim=0).requires_grad_(True)
    mu_b = torch.cat([mu0.detach(), mu1.detach()], dim=0).requires_grad_(True)
    Q_b = torch.cat([Q0.detach(), Q1.detach()], dim=0).requires_grad_(True)
    cells_b = torch.cat([cell0, cell1], dim=0)
    sigmas_b = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    alphas_b = torch.tensor([0.35, 0.35], dtype=torch.float64, device=device)

    batch_idx = torch.cat(
        [
            torch.zeros(n0, dtype=torch.int32, device=device),
            torch.ones(n1, dtype=torch.int32, device=device),
        ]
    )

    idx_j_offset = idx_j1 + n0
    idx_j_b = torch.cat([idx_j0, idx_j_offset])
    ptr_b = torch.cat(
        [
            ptr0[:-1],
            ptr1 + ptr0[-1],
        ]
    )
    unit_shifts_b = torch.cat([sh0, sh1], dim=0)

    E_b = multipole_real_space_quadrupole_energy(
        pos_b,
        q_b,
        mu_b,
        Q_b,
        cells_b,
        idx_j_b,
        ptr_b,
        unit_shifts_b,
        sigmas_b,
        alphas_b,
        batch_idx=batch_idx,
    )
    assert E_b.shape == (2,), f"expected (2,); got {tuple(E_b.shape)}"
    assert torch.isfinite(E_b).all(), f"non-finite energy: {E_b}"

    E_b.sum().backward()
    for name, t in [("pos_b", pos_b), ("q_b", q_b), ("mu_b", mu_b), ("Q_b", Q_b)]:
        assert t.grad is not None, f"no gradient on {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite grad on {name}"
