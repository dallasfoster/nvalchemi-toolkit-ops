# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Forward-energy tests for the l_max = 2 (quadrupole) PME extension.

Three categories:

1. ``quadrupoles = 0`` parity — ``Q = zeros((N, 3, 3))`` must bit-match
   the l_max=1 path.
2. Non-zero ``Q`` smoke test — produces a finite energy delta from the
   l_max=1 baseline.
3. Quadrupole self-energy formula ``F · |Q|² / (320 π^{3/2} σ_c^5)``
   matches the value in ``multipole_pme_energy_corrections``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvalchemiops.torch.interactions.electrostatics import (  # noqa: E402
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (  # noqa: E402
    multipole_pme_energy_corrections,
    multipole_pme_reciprocal_space,
)


def _bcc_quadrupole_fixture(size: int = 2, device: str = "cuda:0"):
    """BCC NaCl-like fixture with random symmetric traceless quadrupoles."""
    a = 4.14
    ijk = np.indices((size, size, size)).reshape(3, -1).T
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    sites = (ijk[:, None, :] + basis[None, :, :]) * a
    pos = sites.reshape(-1, 3).astype(np.float64)
    parity = (ijk.sum(-1)[:, None] + np.array([[0, 1]])) % 2
    q = np.where(parity == 0, 1.0, -1.0).reshape(-1).astype(np.float64)
    if abs(float(q.sum())) > 1e-12:
        q[-1] -= float(q.sum())
    rng = np.random.default_rng(31415)
    mu = rng.standard_normal((pos.shape[0], 3)).astype(np.float64) * 0.3
    Q_raw = rng.standard_normal((pos.shape[0], 3, 3)).astype(np.float64) * 0.2
    Q = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    trace = Q[:, 0, 0] + Q[:, 1, 1] + Q[:, 2, 2]
    Q[:, 0, 0] -= trace / 3.0
    Q[:, 1, 1] -= trace / 3.0
    Q[:, 2, 2] -= trace / 3.0
    cell = np.eye(3, dtype=np.float64) * (size * a)
    return {
        "positions": torch.from_numpy(pos).to(device, torch.float64),
        "cell": torch.from_numpy(cell).to(device, torch.float64),
        "charges": torch.from_numpy(q).to(device, torch.float64),
        "dipoles": torch.from_numpy(mu).to(device, torch.float64),
        "quadrupoles": torch.from_numpy(Q).to(device, torch.float64),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuadrupolePMEForwardEnergy:
    """l_max = 2 forward energy tests."""

    @pytest.fixture(scope="class")
    def fix(self):
        return _bcc_quadrupole_fixture(size=2)

    def _e_recip(self, fix, *, quadrupoles, dipoles=None):
        # An l_max=2 packed tensor must carry the (zero) dipole block.
        if quadrupoles is not None and dipoles is None:
            dipoles = torch.zeros_like(fix["dipoles"])
        mm = pack_multipole_moments(fix["charges"], dipoles, quadrupoles)
        return float(
            multipole_pme_reciprocal_space(
                fix["positions"],
                mm,
                fix["cell"],
                sigma=1.0,
                alpha=0.4632,
                mesh_dimensions=(32, 32, 32),
                spline_order=4,
            ).item()
        )

    def test_zero_quadrupole_matches_dipole_no_dipoles(self, fix):
        """quadrupoles=zeros gives bit-identical result to quadrupoles=None,
        with no dipoles."""
        N = fix["positions"].shape[0]
        Q_zero = torch.zeros((N, 3, 3), dtype=torch.float64, device="cuda:0")
        e_with_Q_zero = self._e_recip(fix, quadrupoles=Q_zero, dipoles=None)
        e_no_Q = self._e_recip(fix, quadrupoles=None, dipoles=None)
        # ``Q=0`` adds nothing (each per-atom Q_eff is zero).
        assert abs(e_with_Q_zero - e_no_Q) < 1e-9 * max(abs(e_no_Q), 1.0), (
            f"Q=0 should match Q=None: e_Q={e_with_Q_zero}, e_noQ={e_no_Q}, "
            f"diff={e_with_Q_zero - e_no_Q}"
        )

    def test_zero_quadrupole_matches_dipole_with_dipoles(self, fix):
        """quadrupoles=zeros gives bit-identical result with dipoles present."""
        N = fix["positions"].shape[0]
        Q_zero = torch.zeros((N, 3, 3), dtype=torch.float64, device="cuda:0")
        e_with_Q_zero = self._e_recip(fix, quadrupoles=Q_zero, dipoles=fix["dipoles"])
        e_no_Q = self._e_recip(fix, quadrupoles=None, dipoles=fix["dipoles"])
        assert abs(e_with_Q_zero - e_no_Q) < 1e-9 * max(abs(e_no_Q), 1.0), (
            f"Q=0 should match Q=None with dipoles: "
            f"e_Q={e_with_Q_zero}, e_noQ={e_no_Q}, "
            f"diff={e_with_Q_zero - e_no_Q}"
        )

    def test_nonzero_quadrupole_changes_energy(self, fix):
        """Non-zero Q must produce a measurable energy delta from Q=0.

        Multipole interactions partially cancel, so the delta is
        conservatively bounded from below at ``10·ULP`` of the baseline.
        """
        e_baseline = self._e_recip(fix, quadrupoles=None, dipoles=fix["dipoles"])
        e_with_Q = self._e_recip(
            fix,
            quadrupoles=fix["quadrupoles"],
            dipoles=fix["dipoles"],
        )
        delta = e_with_Q - e_baseline
        print(
            f"\n  e_baseline (lmax=1)  = {e_baseline:.4f}"
            f"\n  e_with_Q   (lmax=2)  = {e_with_Q:.4f}"
            f"\n  Q-channel delta      = {delta:.4f}"
        )
        threshold = max(10.0 * 2.22e-16 * abs(e_baseline), 1e-4)
        assert abs(delta) > threshold, (
            f"Expected non-trivial Q contribution; got delta={delta} "
            f"(threshold={threshold})"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuadrupolePMEvsReference:
    """PME at l_max = 2 vs the independent Python direct-Ewald reference."""

    def test_pme_reciprocal_quadrupole_matches_reference(self):
        """PME reciprocal half (E_recip − E_self) at l_max = 2 matches the
        reference's ``direct_ewald_reciprocal_minus_self`` (no real-space sum)."""
        from nvalchemiops._reference.multipole_reference import (
            direct_ewald_reciprocal_minus_self,
        )

        fix = _bcc_quadrupole_fixture(size=2)
        sigma, alpha = 1.0, 0.4632
        mesh = (32, 32, 32)

        e_pme = float(
            multipole_pme_reciprocal_space(
                fix["positions"],
                pack_multipole_moments(
                    fix["charges"], fix["dipoles"], fix["quadrupoles"]
                ),
                fix["cell"],
                sigma=sigma,
                alpha=alpha,
                mesh_dimensions=mesh,
                spline_order=4,
            ).item()
        )

        pos_np = fix["positions"].cpu().numpy()
        q_np = fix["charges"].cpu().numpy()
        mu_np = fix["dipoles"].cpu().numpy()
        Q_np = fix["quadrupoles"].cpu().numpy()
        cell_np = fix["cell"].cpu().numpy()
        e_ref = direct_ewald_reciprocal_minus_self(
            pos_np,
            q_np,
            dipoles=mu_np,
            quadrupoles=Q_np,
            cell=cell_np,
            alpha=alpha,
            sigma=sigma,
            kspace_cutoff=5.0,
        )
        rel_err = abs(e_pme - e_ref) / max(abs(e_ref), 1.0)
        print(
            f"\n  PME recip-only (l_max=2)        = {e_pme:.6f}"
            f"\n  Reference recip-minus-self      = {e_ref:.6f}"
            f"\n  rel_err                         = {rel_err:.3e}"
        )
        # PME spline-truncation at order=4, mesh=32³ gives ~1e-3 relative.
        assert rel_err < 5e-3, (
            f"PME l_max=2 recip vs reference rel_err = {rel_err:.3e} > 5e-3 "
            f"(PME = {e_pme}, ref = {e_ref})"
        )


def _small_quadrupole_fixture(device: str = "cuda:0"):
    """4-atom diagonal fixture with symmetric traceless Q.

    The small box + tighter mesh keep all atoms clear of B-spline cell
    breakpoints, so FD perturbations sit in the smooth spline interior
    (needed for sub-1e-4 FD agreement on the Q channel's 3rd derivative).
    """
    dtype = torch.float64
    L = 3.0
    cell = torch.eye(3, dtype=dtype, device=device) * L
    positions = torch.tensor(
        [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
        dtype=dtype,
        device=device,
    )
    charges = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=dtype, device=device)
    dipoles = torch.tensor(
        [[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1], [0.05, 0.05, 0.05], [0.3, -0.2, 0.1]],
        dtype=dtype,
        device=device,
    )
    rng = np.random.default_rng(42)
    Q_raw = rng.standard_normal((4, 3, 3)).astype(np.float64) * 0.2
    Q_sym = 0.5 * (Q_raw + Q_raw.transpose(0, 2, 1))
    trace = Q_sym[:, 0, 0] + Q_sym[:, 1, 1] + Q_sym[:, 2, 2]
    Q_sym[:, 0, 0] -= trace / 3.0
    Q_sym[:, 1, 1] -= trace / 3.0
    Q_sym[:, 2, 2] -= trace / 3.0
    Q = torch.from_numpy(Q_sym).to(device, dtype)
    return {
        "positions": positions,
        "cell": cell,
        "charges": charges,
        "dipoles": dipoles,
        "quadrupoles": Q,
        "sigma": 1.0,
        "alpha": 0.4632,
        "mesh": (16, 16, 16),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuadrupoleAutogradBackward:
    """FD-vs-autograd tests for the l_max=2 PME backward.

    Validates all four diff-input gradients (positions, charges, dipoles,
    quadrupoles) against central-difference FD. The position gradient
    exercises the full ``∂L/∂r = q·∂B + μ·∂²B + (1/2)Q:∂³B`` chain rule.
    Uses the small diagonal fixture so all spline weights sit in the
    smooth interior of their pieces.
    """

    @staticmethod
    def _energy(
        positions,
        charges,
        dipoles,
        quadrupoles,
        cell,
        *,
        sigma,
        alpha,
        mesh,
        spline_order=4,
    ):
        mm = pack_multipole_moments(charges, dipoles, quadrupoles)
        return multipole_pme_reciprocal_space(
            positions,
            mm,
            cell,
            sigma=sigma,
            alpha=alpha,
            mesh_dimensions=mesh,
            spline_order=spline_order,
        )

    def test_position_gradient_fd(self):
        """``∂E/∂r`` via autograd matches central-difference FD."""
        fix = _small_quadrupole_fixture()
        pos_leaf = fix["positions"].detach().clone().requires_grad_(True)
        e = self._energy(
            pos_leaf,
            fix["charges"],
            fix["dipoles"],
            fix["quadrupoles"],
            fix["cell"],
            sigma=fix["sigma"],
            alpha=fix["alpha"],
            mesh=fix["mesh"],
        )
        e.backward()
        grad_analytical = pos_leaf.grad.detach().clone()

        atom_idx, axis_idx = 3, 1
        eps = 1e-4
        with torch.no_grad():
            pos_plus = fix["positions"].clone()
            pos_plus[atom_idx, axis_idx] += eps
            e_plus = self._energy(
                pos_plus,
                fix["charges"],
                fix["dipoles"],
                fix["quadrupoles"],
                fix["cell"],
                sigma=fix["sigma"],
                alpha=fix["alpha"],
                mesh=fix["mesh"],
            ).item()
            pos_minus = fix["positions"].clone()
            pos_minus[atom_idx, axis_idx] -= eps
            e_minus = self._energy(
                pos_minus,
                fix["charges"],
                fix["dipoles"],
                fix["quadrupoles"],
                fix["cell"],
                sigma=fix["sigma"],
                alpha=fix["alpha"],
                mesh=fix["mesh"],
            ).item()
        grad_fd = (e_plus - e_minus) / (2 * eps)
        grad_an = float(grad_analytical[atom_idx, axis_idx].item())
        rel_err = abs(grad_an - grad_fd) / max(abs(grad_fd), 1e-12)
        print(
            f"\n  ∂E/∂r[{atom_idx},{axis_idx}] (analytical) = {grad_an:.6e}"
            f"\n  ∂E/∂r[{atom_idx},{axis_idx}] (FD)         = {grad_fd:.6e}"
            f"\n  rel_err                                  = {rel_err:.3e}"
        )
        assert rel_err < 1e-4, (
            f"FD vs analytical position gradient: "
            f"analytical={grad_an}, FD={grad_fd}, rel_err={rel_err}"
        )

    def test_cell_gradient_spread_fd(self):
        """``∂(K·ρ)/∂cell_inv_t`` via autograd matches central-difference FD.

        Isolates the unified-spread backward cell-gradient path. Loss is
        ``Σ_g K(g) · ρ(g)``; gradient flows to ``cell_inv_t`` through all
        three M-paths (theta = Mr, μ_frac = Mμ, Qe = MQM^T). Positions are
        shifted off integer mesh cells so FD doesn't straddle a breakpoint.
        """
        fix = _small_quadrupole_fixture()
        nx, ny, nz = fix["mesh"]
        spline_order = 4
        L = 3.0
        M_base = (
            torch.diag(torch.full((3,), 1.0 / L, dtype=torch.float64, device="cuda:0"))
            .unsqueeze(0)
            .contiguous()
        )
        positions = fix["positions"] + torch.tensor(
            [0.13, 0.07, 0.19], dtype=torch.float64, device="cuda:0"
        )
        torch.manual_seed(0)
        K = torch.randn(*fix["mesh"], dtype=torch.float64, device="cuda:0") * 0.3

        def call(M):
            rho = torch.ops.nvalchemiops.multipole_pme_spread_unified(
                positions,
                fix["charges"],
                fix["dipoles"],
                fix["quadrupoles"],
                M,
                nx,
                ny,
                nz,
                spline_order,
                2,
            )
            return (K * rho).sum()

        M_leaf = M_base.detach().clone().requires_grad_(True)
        loss = call(M_leaf)
        loss.backward()
        grad_M_an = M_leaf.grad.detach().clone()

        eps = 1e-5
        grad_M_fd = torch.zeros_like(M_base)
        for c in range(3):
            for d in range(3):
                with torch.no_grad():
                    Mp = M_base.clone()
                    Mp[0, c, d] += eps
                    lp = call(Mp).item()
                    Mm = M_base.clone()
                    Mm[0, c, d] -= eps
                    lm = call(Mm).item()
                grad_M_fd[0, c, d] = (lp - lm) / (2 * eps)
        max_abs_err = (grad_M_an - grad_M_fd).abs().max().item()
        rel_err = max_abs_err / max(grad_M_fd.abs().max().item(), 1e-12)
        print(f"\n  max abs error = {max_abs_err:.3e}")
        print(f"  rel_err       = {rel_err:.3e}")
        assert rel_err < 1e-4, (
            f"FD vs analytical cell gradient: max_abs_err={max_abs_err}, "
            f"rel_err={rel_err}"
        )

    def test_cell_gradient_pme_chain_fd(self):
        """End-to-end ``∂E_recip/∂cell`` via autograd matches FD.

        Goes through the full PME chain: ``cell -> cell_inv_t -> spread +
        k_squared -> FFT -> convolve -> IFFT -> energy``. Cell gradient
        flows through ``cell_inv_t`` (spread), ``volume = det(cell)`` and
        ``k_squared`` (convolve backward).
        """
        fix = _small_quadrupole_fixture()
        # Shift positions off mesh-cell boundaries.
        positions = fix["positions"] + torch.tensor(
            [0.13, 0.07, 0.19], dtype=torch.float64, device="cuda:0"
        )
        L = 3.0
        cell_diag = torch.tensor([L, L, L], dtype=torch.float64, device="cuda:0")

        mm = pack_multipole_moments(fix["charges"], fix["dipoles"], fix["quadrupoles"])

        def energy(cell_d):
            cell = torch.diag(cell_d)
            return multipole_pme_reciprocal_space(
                positions,
                mm,
                cell,
                sigma=fix["sigma"],
                alpha=fix["alpha"],
                mesh_dimensions=fix["mesh"],
                spline_order=4,
            )

        cell_leaf = cell_diag.detach().clone().requires_grad_(True)
        e = energy(cell_leaf)
        e.backward()
        grad_an = cell_leaf.grad.detach().clone()

        # FD on each diagonal component of the cell.
        eps = 1e-5
        grad_fd = torch.zeros_like(cell_diag)
        for i in range(3):
            with torch.no_grad():
                cp = cell_diag.clone()
                cp[i] += eps
                ep = energy(cp).item()
                cm = cell_diag.clone()
                cm[i] -= eps
                em = energy(cm).item()
            grad_fd[i] = (ep - em) / (2 * eps)
        max_abs_err = (grad_an - grad_fd).abs().max().item()
        rel_err = max_abs_err / max(grad_fd.abs().max().item(), 1e-12)
        print(
            f"\n  analytical ∂E/∂cell_diag = {grad_an.cpu().tolist()}"
            f"\n  FD         ∂E/∂cell_diag = {grad_fd.cpu().tolist()}"
            f"\n  rel_err                  = {rel_err:.3e}"
        )
        assert rel_err < 1e-4, (
            f"FD vs analytical cell gradient (PME chain): "
            f"analytical={grad_an}, FD={grad_fd}, rel_err={rel_err}"
        )

    def test_quadrupole_gradient_fd(self):
        """``∂E/∂Q`` via autograd matches central-difference FD.

        Perturbs Q SYMMETRICALLY (``Q[i, α, β]`` and ``Q[i, β, α]``
        together), matching the kernel's symmetric ``(1/2) Q : H``
        contraction convention.
        """
        fix = _small_quadrupole_fixture()
        atom_idx = 2
        ai, bi = 0, 1  # off-diagonal pair

        Q_leaf = fix["quadrupoles"].detach().clone().requires_grad_(True)
        e = self._energy(
            fix["positions"],
            fix["charges"],
            fix["dipoles"],
            Q_leaf,
            fix["cell"],
            sigma=fix["sigma"],
            alpha=fix["alpha"],
            mesh=fix["mesh"],
        )
        e.backward()
        # Symmetric DOF gradient = grad[ai,bi] + grad[bi,ai].
        grad_an = float(
            Q_leaf.grad[atom_idx, ai, bi].item() + Q_leaf.grad[atom_idx, bi, ai].item()
        )

        eps = 1e-4
        with torch.no_grad():
            Q_plus = fix["quadrupoles"].clone()
            Q_plus[atom_idx, ai, bi] += eps
            Q_plus[atom_idx, bi, ai] += eps  # symmetric
            e_plus = self._energy(
                fix["positions"],
                fix["charges"],
                fix["dipoles"],
                Q_plus,
                fix["cell"],
                sigma=fix["sigma"],
                alpha=fix["alpha"],
                mesh=fix["mesh"],
            ).item()
            Q_minus = fix["quadrupoles"].clone()
            Q_minus[atom_idx, ai, bi] -= eps
            Q_minus[atom_idx, bi, ai] -= eps  # symmetric
            e_minus = self._energy(
                fix["positions"],
                fix["charges"],
                fix["dipoles"],
                Q_minus,
                fix["cell"],
                sigma=fix["sigma"],
                alpha=fix["alpha"],
                mesh=fix["mesh"],
            ).item()
        grad_fd = (e_plus - e_minus) / (2 * eps)
        rel_err = abs(grad_an - grad_fd) / max(abs(grad_fd), 1e-12)
        print(
            f"\n  ∂E/∂Q_sym[{atom_idx},{ai},{bi}] (analytical) = {grad_an:.6e}"
            f"\n  ∂E/∂Q_sym[{atom_idx},{ai},{bi}] (FD)         = {grad_fd:.6e}"
            f"\n  rel_err                                    = {rel_err:.3e}"
        )
        assert rel_err < 1e-4, (
            f"FD vs analytical Q gradient (symmetric DOF): "
            f"analytical={grad_an}, FD={grad_fd}, rel_err={rel_err}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQuadrupoleSelfEnergyFormula:
    """Validate the sympy-derived quadrupole self-energy formula."""

    def test_quadrupole_self_energy_coefficient(self):
        """Q self-energy = F |Q|²_F / (320 π^{3/2} σ_c^5)."""
        from nvalchemiops.torch.math import FIELD_CONSTANT

        device = "cuda:0"
        N = 4
        rng = np.random.default_rng(7)
        Q_np = rng.standard_normal((N, 3, 3)).astype(np.float64) * 0.5
        Q_np = 0.5 * (Q_np + Q_np.transpose(0, 2, 1))
        trace = Q_np[:, 0, 0] + Q_np[:, 1, 1] + Q_np[:, 2, 2]
        for k in range(3):
            Q_np[:, k, k] -= trace / 3
        Q = torch.from_numpy(Q_np).to(device, torch.float64)
        charges = torch.zeros(N, dtype=torch.float64, device=device)
        volume = torch.tensor(1000.0, dtype=torch.float64, device=device)

        alpha = 0.5
        for sigma in [0.5, 1.0, 1.5]:
            sigma_c = math.sqrt(sigma**2 + 0.25 / alpha**2)
            corr = float(
                multipole_pme_energy_corrections(
                    charges,
                    dipoles=None,
                    quadrupoles=Q,
                    sigma=sigma,
                    alpha=alpha,
                    volume=volume,
                ).item()
            )
            # l=2 self denom is 320 (angular ⟨(k̂·Q·k̂)²⟩ = (2/15)|Q|_F²).
            expected = float(
                FIELD_CONSTANT
                / (320.0 * math.pi**1.5 * sigma_c**5)
                * float((Q_np * Q_np).sum())
            )
            assert abs(corr - expected) / max(abs(expected), 1e-12) < 1e-10, (
                f"σ={sigma}: corr={corr}, expected={expected}, "
                f"rel_err={(corr - expected) / expected}"
            )
