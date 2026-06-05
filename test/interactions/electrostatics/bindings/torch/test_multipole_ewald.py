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

r"""Integration tests for :func:`multipole_real_space_energy`.

Covers FD correctness of the analytical backward, double-backward
(force-loss) FD correctness for l_max=0/1, shape/dtype/device invariants,
and input validation. Composite physical correctness vs Path B lives in
``test_multipole_ewald_summation.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_real_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_charges_dipoles,
    pack_multipole_moments,
)


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _build_system(
    *,
    n_atoms: int,
    box_len: float,
    device: str,
    seed: int,
    full_list: bool = True,
):
    """Charge-neutral random system with explicit (full or half) neighbor list."""
    rng = np.random.default_rng(seed)
    positions_np = rng.uniform(0.0, box_len, size=(n_atoms, 3)).astype(np.float64)
    charges_np = rng.uniform(-1.0, 1.0, size=n_atoms).astype(np.float64)
    charges_np -= charges_np.mean()
    L = max(box_len * 3.0, 100.0)
    cell_np = np.array([[[L, 0, 0], [0, L, 0], [0, 0, L]]], dtype=np.float64)

    idx_j_list: list[int] = []
    nptr = [0]
    shifts_list: list[list[int]] = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if j == i:
                continue
            if not full_list and j < i:
                continue
            idx_j_list.append(j)
            shifts_list.append([0, 0, 0])
        nptr.append(len(idx_j_list))

    td = _torch_device(device)
    positions = torch.from_numpy(positions_np).to(td)
    charges = torch.from_numpy(charges_np).to(td)
    cell = torch.from_numpy(cell_np).to(td)
    idx_j = torch.tensor(idx_j_list, dtype=torch.int32, device=td)
    neighbor_ptr = torch.tensor(nptr, dtype=torch.int32, device=td)
    unit_shifts = torch.tensor(shifts_list, dtype=torch.int32, device=td)
    alpha = torch.tensor([0.4], dtype=torch.float64, device=td)
    sigma = torch.tensor([1.0], dtype=torch.float64, device=td)
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "idx_j": idx_j,
        "neighbor_ptr": neighbor_ptr,
        "unit_shifts": unit_shifts,
        "alpha": alpha,
        "sigma": sigma,
        "n_atoms": n_atoms,
    }


class TestMultipoleRealSpaceEnergyForward:
    def test_output_shape_dtype_device(self, device):
        """Per-atom energy vector shape / dtype / device."""
        sys_dict = _build_system(n_atoms=6, box_len=5.0, device=device, seed=0)
        e = multipole_real_space_energy(
            sys_dict["positions"],
            pack_charges_dipoles(sys_dict["charges"], None),
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        assert e.shape == (sys_dict["n_atoms"],)
        assert e.dtype == torch.float64
        assert e.device.type == _torch_device(device)

    def test_accepts_unbatched_cell(self, device):
        """``(3, 3)`` cell tensor is unsqueezed to ``(1, 3, 3)`` internally."""
        sys_dict = _build_system(n_atoms=4, box_len=5.0, device=device, seed=1)
        cell_2d = sys_dict["cell"].squeeze(0)
        e = multipole_real_space_energy(
            sys_dict["positions"],
            pack_charges_dipoles(sys_dict["charges"], None),
            cell_2d,
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        assert e.shape == (sys_dict["n_atoms"],)


class TestMultipoleRealSpaceEnergyBackward:
    def test_finite_difference(self, device):
        """Analytical backward matches central FD within the wp_erfc approximation floor."""
        sys_dict = _build_system(n_atoms=5, box_len=4.0, device=device, seed=3)
        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), None
        ).requires_grad_(True)

        def _loss(positions, source_feats):
            e = multipole_real_space_energy(
                positions,
                source_feats,
                sys_dict["cell"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )
            # Deterministic per-atom weighting so FD exercises both direct
            # (a=i) and cross (a!=i) contributions.
            w = torch.linspace(0.3, 1.7, e.shape[0], dtype=e.dtype, device=e.device)
            return (w * e).sum()

        L = _loss(pos, sf)
        L.backward()

        h = 1e-5
        for i in range(sys_dict["n_atoms"]):
            for a in range(3):
                pos_plus = pos.detach().clone()
                pos_minus = pos.detach().clone()
                pos_plus[i, a] += h
                pos_minus[i, a] -= h
                fd = (
                    _loss(pos_plus, sf.detach()).item()
                    - _loss(pos_minus, sf.detach()).item()
                ) / (2 * h)
                assert abs(pos.grad[i, a].item() - fd) < 1e-6, (
                    f"pos grad FD mismatch at ({i},{a}): analytical={pos.grad[i, a].item()}, fd={fd}"
                )

        chg_grad = sf.grad[..., 0]
        for i in range(sys_dict["n_atoms"]):
            sf_plus = sf.detach().clone()
            sf_minus = sf.detach().clone()
            sf_plus[i, 0] += h
            sf_minus[i, 0] -= h
            fd = (
                _loss(pos.detach(), sf_plus).item()
                - _loss(pos.detach(), sf_minus).item()
            ) / (2 * h)
            assert abs(chg_grad[i].item() - fd) < 1e-8, (
                f"charge grad FD mismatch at {i}: analytical={chg_grad[i].item()}, fd={fd}"
            )


class TestMultipoleRealSpaceEnergyMonopoleDoubleBackward:
    """``create_graph=True`` force-loss training path (l_max=0)."""

    def test_force_loss_backward_runs_and_is_finite(self, device):
        sys_dict = _build_system(n_atoms=5, box_len=5.0, device=device, seed=11)

        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), None
        ).requires_grad_(True)

        e = multipole_real_space_energy(
            pos,
            sf,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        total_e = e.sum()
        (forces_neg,) = torch.autograd.grad(total_e, pos, create_graph=True)
        loss = (forces_neg**2).sum()
        loss.backward()

        assert torch.isfinite(pos.grad).all()
        assert pos.grad.abs().sum() > 0
        assert sf.grad is not None
        assert torch.isfinite(sf.grad).all()
        assert sf.grad.abs().sum() > 0

    def test_double_backward_matches_finite_difference(self, device):
        """Second-order grad vs central FD of the analytical first-order backward."""
        sys_dict = _build_system(n_atoms=4, box_len=4.0, device=device, seed=13)

        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), None
        ).requires_grad_(True)

        # Fixed upstream "force weights" to define L'.
        td = _torch_device(device)
        rng = np.random.default_rng(500)
        gg_pos_np = rng.standard_normal((sys_dict["n_atoms"], 3))
        gg_chg_np = rng.standard_normal(sys_dict["n_atoms"])
        gg_pos = torch.from_numpy(gg_pos_np).to(td)
        gg_chg = torch.from_numpy(gg_chg_np).to(td)

        # Initial grad_energies = ∂L/∂E for the first backward.
        ge_np = rng.standard_normal(sys_dict["n_atoms"])
        ge = torch.from_numpy(ge_np).to(td).requires_grad_(True)

        e = multipole_real_space_energy(
            pos,
            sf,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        (grad_pos, grad_sf) = torch.autograd.grad(
            outputs=(e,),
            inputs=(pos, sf),
            grad_outputs=(ge,),
            create_graph=True,
        )
        grad_chg = grad_sf[..., 0]

        # Scalar L' = gg_pos · grad_pos + gg_chg · grad_chg
        lprime = (gg_pos * grad_pos).sum() + (gg_chg * grad_chg).sum()
        lprime.backward()

        h = 1e-5
        pos_np = pos.detach().cpu().numpy().copy()
        sf_np = sf.detach().cpu().numpy().copy()
        for i in range(sys_dict["n_atoms"]):
            for a in range(3):
                pos_p = pos_np.copy()
                pos_m = pos_np.copy()
                pos_p[i, a] += h
                pos_m[i, a] -= h

                def _make_and_compute_lprime(p_np):
                    p_local = torch.from_numpy(p_np).to(td).requires_grad_(True)
                    sf_local = torch.from_numpy(sf_np).to(td).requires_grad_(True)
                    e_local = multipole_real_space_energy(
                        p_local,
                        sf_local,
                        sys_dict["cell"],
                        sys_dict["idx_j"],
                        sys_dict["neighbor_ptr"],
                        sys_dict["unit_shifts"],
                        sys_dict["sigma"],
                        sys_dict["alpha"],
                    )
                    (gp_l, gsf_l) = torch.autograd.grad(
                        outputs=(e_local,),
                        inputs=(p_local, sf_local),
                        grad_outputs=(ge.detach(),),
                    )
                    gc_l = gsf_l[..., 0]
                    return float(((gg_pos * gp_l).sum() + (gg_chg * gc_l).sum()).item())

                fd_val = (
                    _make_and_compute_lprime(pos_p) - _make_and_compute_lprime(pos_m)
                ) / (2 * h)
                analytical = pos.grad[i, a].item()
                assert abs(analytical - fd_val) < 5e-6, (
                    f"2nd-order grad_pos FD mismatch at ({i},{a}): "
                    f"analytical={analytical}, fd={fd_val}"
                )


def _random_dipoles(n_atoms: int, device: str, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed + 100)
    return torch.from_numpy(
        rng.standard_normal((n_atoms, 3)).astype(np.float64) * 0.3
    ).to(_torch_device(device))


class TestMultipoleRealSpaceEnergyDipole:
    @pytest.mark.parametrize("seed", [0, 42])
    def test_collapse_to_monopole_when_dipoles_zero(self, device, seed):
        """Zero dipoles ⇒ l_max=1 energy equals l_max=0 energy (monopole collapse)."""
        sys_dict = _build_system(n_atoms=6, box_len=5.0, device=device, seed=seed)
        td = _torch_device(device)
        zero_dipoles = torch.zeros(
            sys_dict["n_atoms"], 3, dtype=torch.float64, device=td
        )

        e_dipole = multipole_real_space_energy(
            sys_dict["positions"],
            pack_charges_dipoles(sys_dict["charges"], zero_dipoles),
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        e_monopole = multipole_real_space_energy(
            sys_dict["positions"],
            pack_charges_dipoles(sys_dict["charges"], None),
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        torch.testing.assert_close(e_dipole, e_monopole, rtol=0, atol=1e-14)

    def test_backward_finite_difference(self, device):
        """Analytical backward matches central FD for positions / charges / dipoles."""
        sys_dict = _build_system(n_atoms=5, box_len=4.0, device=device, seed=3)
        dip = _random_dipoles(sys_dict["n_atoms"], device, seed=3)
        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), dip
        ).requires_grad_(True)

        def _loss(positions, source_feats):
            e = multipole_real_space_energy(
                positions,
                source_feats,
                sys_dict["cell"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )
            w = torch.linspace(0.3, 1.7, e.shape[0], dtype=e.dtype, device=e.device)
            return (w * e).sum()

        L = _loss(pos, sf)
        L.backward()

        chg_grad = sf.grad[..., 0]
        # Dipole grad: e3nn (y, z, x) spherical cols [1:4] → Cartesian (x, y, z).
        dip_grad_cart = sf.grad[..., [3, 1, 2]]

        h = 1e-5
        for i in range(sys_dict["n_atoms"]):
            for a in range(3):
                pos_plus = pos.detach().clone()
                pos_minus = pos.detach().clone()
                pos_plus[i, a] += h
                pos_minus[i, a] -= h
                fd = (
                    _loss(pos_plus, sf.detach()).item()
                    - _loss(pos_minus, sf.detach()).item()
                ) / (2 * h)
                assert abs(pos.grad[i, a].item() - fd) < 1e-5, (
                    f"pos grad FD mismatch at ({i},{a}): "
                    f"analytical={pos.grad[i, a].item()}, fd={fd}"
                )

        for i in range(sys_dict["n_atoms"]):
            sf_plus = sf.detach().clone()
            sf_minus = sf.detach().clone()
            sf_plus[i, 0] += h
            sf_minus[i, 0] -= h
            fd = (
                _loss(pos.detach(), sf_plus).item()
                - _loss(pos.detach(), sf_minus).item()
            ) / (2 * h)
            assert abs(chg_grad[i].item() - fd) < 1e-8, (
                f"charge grad FD mismatch at {i}: "
                f"analytical={chg_grad[i].item()}, fd={fd}"
            )

        # Cartesian axis a=0,1,2 (x,y,z) maps to source_feats col (3, 1, 2).
        cart_to_sph_col = (3, 1, 2)
        for i in range(sys_dict["n_atoms"]):
            for a in range(3):
                sf_plus = sf.detach().clone()
                sf_minus = sf.detach().clone()
                col = cart_to_sph_col[a]
                sf_plus[i, col] += h
                sf_minus[i, col] -= h
                fd = (
                    _loss(pos.detach(), sf_plus).item()
                    - _loss(pos.detach(), sf_minus).item()
                ) / (2 * h)
                assert abs(dip_grad_cart[i, a].item() - fd) < 1e-8, (
                    f"dipole grad FD mismatch at ({i},{a}): "
                    f"analytical={dip_grad_cart[i, a].item()}, fd={fd}"
                )


class TestMultipoleRealSpaceEnergyDipoleDoubleBackward:
    """Analytical l_max=1 second-order kernel (charges + dipoles).

    Exercises force-loss training and verifies the 2nd-order ∂²/∂pos
    gradient against central FD of the first-order backward.
    """

    def test_force_loss_backward_runs_and_is_finite(self, device):
        sys_dict = _build_system(n_atoms=5, box_len=5.0, device=device, seed=42)
        dipoles = _random_dipoles(sys_dict["n_atoms"], device, seed=42)
        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), dipoles
        ).requires_grad_(True)

        e = multipole_real_space_energy(
            pos,
            sf,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        total_e = e.sum()
        (forces_neg,) = torch.autograd.grad(total_e, pos, create_graph=True)
        loss = (forces_neg**2).sum()
        loss.backward()
        for t in (pos, sf):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all()
            assert t.grad.abs().sum() > 0

    def test_double_backward_matches_finite_difference(self, device):
        """Second-order ∂²/∂pos of L' vs central FD of the first-order backward."""
        sys_dict = _build_system(n_atoms=4, box_len=4.0, device=device, seed=13)
        td = _torch_device(device)
        rng = np.random.default_rng(500)

        dipoles = _random_dipoles(sys_dict["n_atoms"], device, seed=13)
        pos = sys_dict["positions"].detach().clone().requires_grad_(True)
        sf = pack_charges_dipoles(
            sys_dict["charges"].detach().clone(), dipoles
        ).requires_grad_(True)

        gg_pos = torch.from_numpy(rng.standard_normal((sys_dict["n_atoms"], 3))).to(td)
        gg_chg = torch.from_numpy(rng.standard_normal(sys_dict["n_atoms"])).to(td)
        gg_dip = torch.from_numpy(rng.standard_normal((sys_dict["n_atoms"], 3))).to(td)
        ge_np = rng.standard_normal(sys_dict["n_atoms"])
        ge = torch.from_numpy(ge_np).to(td).requires_grad_(True)

        e = multipole_real_space_energy(
            pos,
            sf,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )
        (grad_pos, grad_sf) = torch.autograd.grad(
            outputs=(e,),
            inputs=(pos, sf),
            grad_outputs=(ge,),
            create_graph=True,
        )
        grad_chg = grad_sf[..., 0]
        # Cartesian (x, y, z) dipole grad from spherical (y, z, x) columns.
        grad_dip = grad_sf[..., [3, 1, 2]]
        lprime = (
            (gg_pos * grad_pos).sum()
            + (gg_chg * grad_chg).sum()
            + (gg_dip * grad_dip).sum()
        )
        lprime.backward()

        h = 1e-5
        pos_np = pos.detach().cpu().numpy().copy()
        sf_np = sf.detach().cpu().numpy().copy()

        def _lprime_at(p_np):
            p_local = torch.from_numpy(p_np).to(td).requires_grad_(True)
            sf_local = torch.from_numpy(sf_np).to(td).requires_grad_(True)
            e_local = multipole_real_space_energy(
                p_local,
                sf_local,
                sys_dict["cell"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )
            (gp_l, gsf_l) = torch.autograd.grad(
                outputs=(e_local,),
                inputs=(p_local, sf_local),
                grad_outputs=(ge.detach(),),
            )
            gc_l = gsf_l[..., 0]
            gd_l = gsf_l[..., [3, 1, 2]]
            return float(
                (
                    (gg_pos * gp_l).sum()
                    + (gg_chg * gc_l).sum()
                    + (gg_dip * gd_l).sum()
                ).item()
            )

        for i in range(sys_dict["n_atoms"]):
            for a in range(3):
                pp = pos_np.copy()
                pm = pos_np.copy()
                pp[i, a] += h
                pm[i, a] -= h
                fd_val = (_lprime_at(pp) - _lprime_at(pm)) / (2 * h)
                analytical = pos.grad[i, a].item()
                # Looser than lmax=0: l_max=1 position grad involves the A'''
                # term, amplifying the wp_erfc floor via higher radial derivatives.
                assert abs(analytical - fd_val) < 1e-4, (
                    f"2nd-order grad_pos FD mismatch at ({i},{a}): "
                    f"analytical={analytical}, fd={fd_val}"
                )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
class TestMultipoleRealSpaceDipoleCompile:
    r"""The :math:`l_{max}=1` single-system path is a ``torch.library.custom_op``
    chain (not a ``torch.autograd.Function``), so ``torch.compile`` must trace it
    as one opaque graph node — no graph break on the struct-dtype
    ``wp.from_torch`` calls — and produce eager-identical results.
    """

    def _inputs(self):
        sys_dict = _build_system(n_atoms=12, box_len=6.0, device="cuda:0", seed=7)
        rng = np.random.default_rng(11)
        dipoles = torch.from_numpy(
            (0.3 * rng.standard_normal((sys_dict["n_atoms"], 3))).astype(np.float64)
        ).to("cuda")
        source_feats = pack_charges_dipoles(sys_dict["charges"], dipoles)
        return sys_dict, source_feats

    def _energy(self, sys_dict, source_feats, positions):
        return multipole_real_space_energy(
            positions,
            source_feats,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )

    def test_no_graph_breaks(self):
        """``torch.compile`` captures the path in a single graph (0 breaks)."""
        torch._dynamo.reset()
        sys_dict, source_feats = self._inputs()
        explanation = torch._dynamo.explain(
            lambda: self._energy(sys_dict, source_feats, sys_dict["positions"])
        )()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self):
        """Compiled forward + first-order grad match eager bit-for-bit."""
        torch._dynamo.reset()
        sys_dict, source_feats = self._inputs()
        pos_eager = sys_dict["positions"].detach().clone().requires_grad_(True)
        pos_comp = sys_dict["positions"].detach().clone().requires_grad_(True)

        e_eager = self._energy(sys_dict, source_feats, pos_eager)
        compiled = torch.compile(lambda p: self._energy(sys_dict, source_feats, p))
        e_comp = compiled(pos_comp)
        torch.testing.assert_close(e_comp, e_eager)

        (g_eager,) = torch.autograd.grad(e_eager.sum(), pos_eager)
        (g_comp,) = torch.autograd.grad(e_comp.sum(), pos_comp)
        torch.testing.assert_close(g_comp, g_eager)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
class TestMultipoleRealSpaceMonopoleCompile:
    r"""``torch.compile`` regression for the :math:`l_{max}=0` single-system path
    (``nvalchemiops::multipole_real_space_monopole`` custom_op chain)."""

    def _inputs(self):
        sys_dict = _build_system(n_atoms=12, box_len=6.0, device="cuda:0", seed=5)
        return sys_dict, pack_charges_dipoles(sys_dict["charges"], None)

    def _energy(self, sys_dict, source_feats, positions):
        return multipole_real_space_energy(
            positions,
            source_feats,
            sys_dict["cell"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )

    def test_no_graph_breaks(self):
        """``torch.compile`` captures the path in a single graph (0 breaks)."""
        torch._dynamo.reset()
        sys_dict, source_feats = self._inputs()
        explanation = torch._dynamo.explain(
            lambda: self._energy(sys_dict, source_feats, sys_dict["positions"])
        )()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self):
        """Compiled forward + first-order grad match eager bit-for-bit."""
        torch._dynamo.reset()
        sys_dict, source_feats = self._inputs()
        pos_eager = sys_dict["positions"].detach().clone().requires_grad_(True)
        pos_comp = sys_dict["positions"].detach().clone().requires_grad_(True)

        e_eager = self._energy(sys_dict, source_feats, pos_eager)
        compiled = torch.compile(lambda p: self._energy(sys_dict, source_feats, p))
        e_comp = compiled(pos_comp)
        torch.testing.assert_close(e_comp, e_eager)

        (g_eager,) = torch.autograd.grad(e_eager.sum(), pos_eager)
        (g_comp,) = torch.autograd.grad(e_comp.sum(), pos_comp)
        torch.testing.assert_close(g_comp, g_eager)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
class TestMultipoleRealSpaceQuadrupoleCompile:
    r"""``torch.compile`` regression for the :math:`l_{max}=2` single-system path
    (``nvalchemiops::multipole_real_space_quadrupole`` custom_op chain, whose
    backward conditionally fires the cell-grad kernel)."""

    def _inputs(self):
        sys_dict = _build_system(n_atoms=10, box_len=6.0, device="cuda:0", seed=9)
        rng = np.random.default_rng(13)
        n = sys_dict["n_atoms"]
        dipoles = torch.from_numpy(
            (0.3 * rng.standard_normal((n, 3))).astype(np.float64)
        ).to("cuda")
        q = 0.2 * rng.standard_normal((n, 3, 3))
        q = 0.5 * (q + q.transpose(0, 2, 1))
        q -= (q.trace(axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
        quads = torch.from_numpy(q.astype(np.float64)).to("cuda")
        moments = pack_multipole_moments(sys_dict["charges"], dipoles, quads)
        return sys_dict, moments

    def _energy(self, sys_dict, moments, positions, cell=None):
        return multipole_real_space_energy(
            positions,
            moments,
            sys_dict["cell"] if cell is None else cell,
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
            sys_dict["sigma"],
            sys_dict["alpha"],
        )

    def test_no_graph_breaks(self):
        """``torch.compile`` captures the l=2 path in a single graph (0 breaks)."""
        torch._dynamo.reset()
        sys_dict, moments = self._inputs()
        explanation = torch._dynamo.explain(
            lambda: self._energy(sys_dict, moments, sys_dict["positions"])
        )()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self):
        """Compiled forward + first-order grad match eager bit-for-bit."""
        torch._dynamo.reset()
        sys_dict, moments = self._inputs()
        pos_eager = sys_dict["positions"].detach().clone().requires_grad_(True)
        pos_comp = sys_dict["positions"].detach().clone().requires_grad_(True)

        e_eager = self._energy(sys_dict, moments, pos_eager)
        compiled = torch.compile(lambda p: self._energy(sys_dict, moments, p))
        e_comp = compiled(pos_comp)
        torch.testing.assert_close(e_comp, e_eager)

        (g_eager,) = torch.autograd.grad(e_eager.sum(), pos_eager)
        (g_comp,) = torch.autograd.grad(e_comp.sum(), pos_comp)
        torch.testing.assert_close(g_comp, g_eager)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
@pytest.mark.parametrize("l_max", [0, 1, 2])
class TestBatchMultipoleRealSpaceCompile:
    r"""``torch.compile`` regression for the batched real-space path
    (``batch_multipole_real_space_*`` custom_op chains), all ``l_max``."""

    def _batch(self, l_max):
        a = _build_system(n_atoms=8, box_len=6.0, device="cuda:0", seed=1)
        b = _build_system(n_atoms=11, box_len=6.0, device="cuda:0", seed=2)
        n0 = a["n_atoms"]
        rng = np.random.default_rng(3)

        def _moments(sysd):
            n = sysd["n_atoms"]
            if l_max == 0:
                return pack_charges_dipoles(sysd["charges"], None)
            dip = torch.from_numpy(
                (0.3 * rng.standard_normal((n, 3))).astype(np.float64)
            ).to("cuda")
            if l_max == 1:
                return pack_charges_dipoles(sysd["charges"], dip)
            q = 0.2 * rng.standard_normal((n, 3, 3))
            q = 0.5 * (q + q.transpose(0, 2, 1))
            q -= (q.trace(axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
            quads = torch.from_numpy(q.astype(np.float64)).to("cuda")
            return pack_multipole_moments(sysd["charges"], dip, quads)

        positions = torch.cat([a["positions"], b["positions"]])
        moments = torch.cat([_moments(a), _moments(b)])
        idx = torch.cat([a["idx_j"], b["idx_j"] + n0])
        m0 = a["idx_j"].numel()
        nptr = torch.cat([a["neighbor_ptr"], b["neighbor_ptr"][1:] + m0]).to(
            torch.int32
        )
        shifts = torch.cat([a["unit_shifts"], b["unit_shifts"]])
        batch_idx = torch.cat(
            [
                torch.zeros(n0, dtype=torch.int32, device="cuda"),
                torch.ones(b["n_atoms"], dtype=torch.int32, device="cuda"),
            ]
        )
        cells = torch.cat([a["cell"], b["cell"]])  # each (1,3,3) -> (2,3,3)
        sig = torch.tensor([1.0, 1.0], dtype=torch.float64, device="cuda")
        alp = torch.tensor([0.4, 0.4], dtype=torch.float64, device="cuda")
        return positions, moments, cells, idx, nptr, shifts, sig, alp, batch_idx

    def _energy(self, packed, positions):
        _, moments, cells, idx, nptr, shifts, sig, alp, batch_idx = packed
        return multipole_real_space_energy(
            positions, moments, cells, idx, nptr, shifts, sig, alp, batch_idx=batch_idx
        )

    def test_no_graph_breaks(self, l_max):
        """``torch.compile`` captures the batched path in a single graph."""
        torch._dynamo.reset()
        packed = self._batch(l_max)
        explanation = torch._dynamo.explain(lambda: self._energy(packed, packed[0]))()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self, l_max):
        """Compiled forward + first-order grad match eager bit-for-bit."""
        torch._dynamo.reset()
        packed = self._batch(l_max)
        pos_eager = packed[0].detach().clone().requires_grad_(True)
        pos_comp = packed[0].detach().clone().requires_grad_(True)
        e_eager = self._energy(packed, pos_eager)
        compiled = torch.compile(lambda p: self._energy(packed, p))
        e_comp = compiled(pos_comp)
        torch.testing.assert_close(e_comp, e_eager)
        (g_eager,) = torch.autograd.grad(e_eager.sum(), pos_eager)
        (g_comp,) = torch.autograd.grad(e_comp.sum(), pos_comp)
        torch.testing.assert_close(g_comp, g_eager)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
@pytest.mark.parametrize("l_max", [0, 1])
class TestMultipoleRealSpaceFusedScalarCompile:
    r"""``torch.compile`` regression for the fused-scalar real-space ops
    (``multipole_real_space_{monopole,dipole}_fused``) used by the Ewald
    composite: one fused launch returns the scalar energy + per-atom grads."""

    def _inputs(self, l_max):
        s = _build_system(n_atoms=10, box_len=6.0, device="cuda:0", seed=4)
        dip = None
        if l_max == 1:
            rng = np.random.default_rng(8)
            dip = torch.from_numpy(
                (0.3 * rng.standard_normal((s["n_atoms"], 3))).astype(np.float64)
            ).to("cuda")
        return s, s["cell"], dip

    def _energy(self, s, cell, dip, positions):
        if dip is None:
            return torch.ops.nvalchemiops.multipole_real_space_monopole_fused(
                positions,
                s["charges"],
                cell,
                s["sigma"],
                s["alpha"],
                s["idx_j"],
                s["neighbor_ptr"],
                s["unit_shifts"],
                False,
            )[0]
        return torch.ops.nvalchemiops.multipole_real_space_dipole_fused(
            positions,
            s["charges"],
            dip,
            cell,
            s["sigma"],
            s["alpha"],
            s["idx_j"],
            s["neighbor_ptr"],
            s["unit_shifts"],
            False,
        )[0]

    def test_no_graph_breaks(self, l_max):
        torch._dynamo.reset()
        s, cell, dip = self._inputs(l_max)
        explanation = torch._dynamo.explain(
            lambda: self._energy(s, cell, dip, s["positions"])
        )()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self, l_max):
        torch._dynamo.reset()
        s, cell, dip = self._inputs(l_max)
        pos_e = s["positions"].detach().clone().requires_grad_(True)
        pos_c = s["positions"].detach().clone().requires_grad_(True)
        e_eager = self._energy(s, cell, dip, pos_e)
        compiled = torch.compile(lambda p: self._energy(s, cell, dip, p))
        e_comp = compiled(pos_c)
        torch.testing.assert_close(e_comp, e_eager)
        (g_eager,) = torch.autograd.grad(e_eager, pos_e)
        (g_comp,) = torch.autograd.grad(e_comp, pos_c)
        torch.testing.assert_close(g_comp, g_eager)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile / Warp path is GPU-only here"
)
@pytest.mark.parametrize("l_max", [0, 1])
class TestBatchMultipoleRealSpaceFusedScalarCompile:
    r"""``torch.compile`` regression for the batched fused-scalar ops
    (``batch_multipole_real_space_{monopole,dipole}_fused``) used by the batched
    Ewald composite."""

    def _batch(self, l_max):
        a = _build_system(n_atoms=8, box_len=6.0, device="cuda:0", seed=1)
        b = _build_system(n_atoms=11, box_len=6.0, device="cuda:0", seed=2)
        n0 = a["n_atoms"]
        positions = torch.cat([a["positions"], b["positions"]])
        idx = torch.cat([a["idx_j"], b["idx_j"] + n0])
        m0 = a["idx_j"].numel()
        nptr = torch.cat([a["neighbor_ptr"], b["neighbor_ptr"][1:] + m0]).to(
            torch.int32
        )
        shifts = torch.cat([a["unit_shifts"], b["unit_shifts"]])
        batch_idx = torch.cat(
            [
                torch.zeros(n0, dtype=torch.int32, device="cuda"),
                torch.ones(b["n_atoms"], dtype=torch.int32, device="cuda"),
            ]
        )
        cells = torch.cat([a["cell"], b["cell"]])  # each (1,3,3) -> (2,3,3)
        sig = torch.tensor([1.0, 1.0], dtype=torch.float64, device="cuda")
        alp = torch.tensor([0.4, 0.4], dtype=torch.float64, device="cuda")
        charges = torch.cat([a["charges"], b["charges"]])
        dip = None
        if l_max == 1:
            rng = np.random.default_rng(6)
            dip = torch.from_numpy(
                (0.3 * rng.standard_normal((n0 + b["n_atoms"], 3))).astype(np.float64)
            ).to("cuda")
        return positions, charges, dip, cells, idx, nptr, shifts, sig, alp, batch_idx

    def _energy(self, packed, positions):
        _, charges, dip, cells, idx, nptr, shifts, sig, alp, batch_idx = packed
        if dip is None:
            return torch.ops.nvalchemiops.batch_multipole_real_space_monopole_fused(
                positions, charges, cells, sig, alp, idx, nptr, shifts, batch_idx, False
            )[0]
        return torch.ops.nvalchemiops.batch_multipole_real_space_dipole_fused(
            positions,
            charges,
            dip,
            cells,
            sig,
            alp,
            idx,
            nptr,
            shifts,
            batch_idx,
            False,
        )[0]

    def test_no_graph_breaks(self, l_max):
        torch._dynamo.reset()
        packed = self._batch(l_max)
        explanation = torch._dynamo.explain(lambda: self._energy(packed, packed[0]))()
        assert explanation.graph_break_count == 0, explanation.break_reasons

    def test_compiled_matches_eager(self, l_max):
        torch._dynamo.reset()
        packed = self._batch(l_max)
        pos_e = packed[0].detach().clone().requires_grad_(True)
        pos_c = packed[0].detach().clone().requires_grad_(True)
        e_eager = self._energy(packed, pos_e)
        compiled = torch.compile(lambda p: self._energy(packed, p))
        e_comp = compiled(pos_c)
        torch.testing.assert_close(e_comp, e_eager)
        (g_eager,) = torch.autograd.grad(e_eager.sum(), pos_e)
        (g_comp,) = torch.autograd.grad(e_comp.sum(), pos_c)
        torch.testing.assert_close(g_comp, g_eager)


class TestMultipoleRealSpaceValidation:
    def test_bad_source_feats_shape(self, device):
        """source_feats with a mismatched leading dim raises ValueError."""
        sys_dict = _build_system(n_atoms=4, box_len=5.0, device=device, seed=0)
        # Wrong N (3 vs 4 positions), still a valid (N, 1) trailing dim.
        bad_sf = torch.zeros(3, 1, dtype=torch.float64, device=_torch_device(device))
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_real_space_energy(
                sys_dict["positions"],
                bad_sf,
                sys_dict["cell"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )

    def test_bad_positions_shape(self, device):
        sys_dict = _build_system(n_atoms=4, box_len=5.0, device=device, seed=0)
        bad_pos = sys_dict["positions"][:, :2]  # (N, 2)
        with pytest.raises(ValueError, match="positions must be"):
            multipole_real_space_energy(
                bad_pos,
                pack_charges_dipoles(sys_dict["charges"], None),
                sys_dict["cell"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )

    def test_batched_cell_rejected(self, device):
        """A ``(B, 3, 3)`` cell with B>1 raises on the single-system entry."""
        sys_dict = _build_system(n_atoms=4, box_len=5.0, device=device, seed=0)
        batched_cell = sys_dict["cell"].repeat(2, 1, 1)  # (2, 3, 3)
        with pytest.raises(ValueError, match="batch size"):
            multipole_real_space_energy(
                sys_dict["positions"],
                pack_charges_dipoles(sys_dict["charges"], None),
                batched_cell,
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
                sys_dict["sigma"],
                sys_dict["alpha"],
            )


class TestMultipoleRealSpaceMonopoleFusedScalar:
    r"""Parity tests for ``MultipoleRealSpaceMonopoleFusedScalarFunction``.

    The fused Function returns scalar energy, computes analytical gradients
    in forward when inputs require grad, and backward is a pure scalar
    broadcast. Parity vs the layered ``MultipoleRealSpaceMonopoleFunction`` +
    backward pair should hold to ULP at fp64.
    """

    @pytest.fixture
    def gpu_system(self):
        if not torch.cuda.is_available():
            pytest.skip("Fused path is GPU-only")
        return _build_system(n_atoms=8, box_len=5.0, device="cuda:0", seed=0xCAFE)

    def _run_fused(self, sys_dict, *, with_pos: bool, with_q: bool):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceMonopoleFusedScalarFunction,
        )

        positions = sys_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = sys_dict["charges"].clone().detach().requires_grad_(with_q)
        return (
            positions,
            charges,
            MultipoleRealSpaceMonopoleFusedScalarFunction.apply(
                positions,
                charges,
                sys_dict["cell"],
                sys_dict["sigma"],
                sys_dict["alpha"],
                sys_dict["idx_j"],
                sys_dict["neighbor_ptr"],
                sys_dict["unit_shifts"],
            ),
        )

    def _run_reference(self, sys_dict, *, with_pos: bool, with_q: bool):
        """Existing layered path — per-atom energies, summed externally."""
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceMonopoleFunction,
        )

        positions = sys_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = sys_dict["charges"].clone().detach().requires_grad_(with_q)
        per_atom = MultipoleRealSpaceMonopoleFunction.apply(
            positions,
            charges,
            sys_dict["cell"],
            sys_dict["sigma"],
            sys_dict["alpha"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
        )
        return positions, charges, per_atom.sum()

    def test_forward_parity_vs_reference(self, gpu_system):
        """Energy sum matches reference path to ULP at fp64."""
        _, _, e_fused = self._run_fused(gpu_system, with_pos=False, with_q=False)
        _, _, e_ref = self._run_reference(gpu_system, with_pos=False, with_q=False)
        torch.testing.assert_close(e_fused, e_ref, rtol=0, atol=1e-15)

    def test_grad_pos_parity_vs_reference(self, gpu_system):
        """Position gradient matches reference path to ULP at fp64."""
        pos_f, _, e_f = self._run_fused(gpu_system, with_pos=True, with_q=False)
        e_f.backward()
        pos_r, _, e_r = self._run_reference(gpu_system, with_pos=True, with_q=False)
        e_r.backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)

    def test_grad_q_parity_vs_reference(self, gpu_system):
        """Charge gradient matches reference path to ULP at fp64."""
        _, q_f, e_f = self._run_fused(gpu_system, with_pos=False, with_q=True)
        e_f.backward()
        _, q_r, e_r = self._run_reference(gpu_system, with_pos=False, with_q=True)
        e_r.backward()
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)

    def test_both_grads_parity_vs_reference(self, gpu_system):
        """Joint pos+charge gradient matches reference path to ULP."""
        pos_f, q_f, e_f = self._run_fused(gpu_system, with_pos=True, with_q=True)
        e_f.backward()
        pos_r, q_r, e_r = self._run_reference(gpu_system, with_pos=True, with_q=True)
        e_r.backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)

    def test_per_input_grad_gating_pos_only(self, gpu_system):
        """When only positions.requires_grad, charges.grad stays None."""
        pos_f, q_f, e_f = self._run_fused(gpu_system, with_pos=True, with_q=False)
        e_f.backward()
        assert pos_f.grad is not None
        assert q_f.grad is None  # never required grad

    def test_per_input_grad_gating_charge_only(self, gpu_system):
        """When only charges.requires_grad, positions.grad stays None."""
        pos_f, q_f, e_f = self._run_fused(gpu_system, with_pos=False, with_q=True)
        e_f.backward()
        assert pos_f.grad is None
        assert q_f.grad is not None

    def test_no_grad_path_runs(self, gpu_system):
        """No-grad inputs run the energy-only kernel and produce no graph."""
        _, _, e = self._run_fused(gpu_system, with_pos=False, with_q=False)
        assert not e.requires_grad
        assert torch.isfinite(e).item()

    def test_backward_does_no_kernel_work(self, gpu_system):
        """Backward is a scalar broadcast over pre-computed gradient tensors."""
        pos_f, q_f, e_f = self._run_fused(gpu_system, with_pos=True, with_q=True)
        upstream = torch.tensor(2.5, dtype=torch.float64, device=e_f.device)
        e_f.backward(upstream)
        pos_r, q_r, e_r = self._run_reference(gpu_system, with_pos=True, with_q=True)
        e_r.backward(upstream)
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)

    def test_double_backward_raises(self, gpu_system):
        """Fused path doesn't support 2nd-order grad; first backward is finite."""
        pos_f, q_f, e_f = self._run_fused(gpu_system, with_pos=True, with_q=False)
        (grad_pos,) = torch.autograd.grad(e_f, pos_f, create_graph=True)
        assert torch.isfinite(grad_pos).all()

    @pytest.fixture
    def cpu_system(self):
        return _build_system(n_atoms=8, box_len=5.0, device="cpu", seed=0xC0FFEE)

    def test_cpu_path_runs(self, cpu_system):
        """CSR fused kernel runs on CPU."""
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceMonopoleFusedScalarFunction,
        )

        positions = cpu_system["positions"].clone().detach().requires_grad_(True)
        charges = cpu_system["charges"].clone().detach().requires_grad_(True)
        e = MultipoleRealSpaceMonopoleFusedScalarFunction.apply(
            positions,
            charges,
            cpu_system["cell"],
            cpu_system["sigma"],
            cpu_system["alpha"],
            cpu_system["idx_j"],
            cpu_system["neighbor_ptr"],
            cpu_system["unit_shifts"],
        )
        e.backward()
        assert torch.isfinite(positions.grad).all()
        assert torch.isfinite(charges.grad).all()
        assert torch.isfinite(e).item()

    def test_cpu_parity_vs_layered_csr(self, cpu_system):
        """Bit-for-bit parity vs the layered CSR path on CPU."""
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceMonopoleFunction,
            MultipoleRealSpaceMonopoleFusedScalarFunction,
        )

        pos_f = cpu_system["positions"].clone().detach().requires_grad_(True)
        q_f = cpu_system["charges"].clone().detach().requires_grad_(True)
        e_f = MultipoleRealSpaceMonopoleFusedScalarFunction.apply(
            pos_f,
            q_f,
            cpu_system["cell"],
            cpu_system["sigma"],
            cpu_system["alpha"],
            cpu_system["idx_j"],
            cpu_system["neighbor_ptr"],
            cpu_system["unit_shifts"],
        )
        e_f.backward()

        pos_r = cpu_system["positions"].clone().detach().requires_grad_(True)
        q_r = cpu_system["charges"].clone().detach().requires_grad_(True)
        e_r = MultipoleRealSpaceMonopoleFunction.apply(
            pos_r,
            q_r,
            cpu_system["cell"],
            cpu_system["sigma"],
            cpu_system["alpha"],
            cpu_system["idx_j"],
            cpu_system["neighbor_ptr"],
            cpu_system["unit_shifts"],
        ).sum()
        e_r.backward()

        torch.testing.assert_close(e_f, e_r, rtol=0, atol=1e-15)
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)


class TestBatchMultipoleRealSpaceMonopoleFusedScalar:
    r"""Parity tests for ``BatchMultipoleRealSpaceMonopoleFusedScalarFunction``.

    Returns per-system scalar tensor ``(B,)``; backward is per-atom
    weight broadcast (``grad_E[batch_idx[i]]`` per atom).
    """

    def _build_batch_system(
        self, *, device: str, n_per_sys: int = 8, n_systems: int = 3
    ):
        """Stack n_systems independent _build_system fixtures with shared sigma/alpha."""
        sub = []
        for s in range(n_systems):
            sd = _build_system(
                n_atoms=n_per_sys, box_len=5.0, device=device, seed=0x100 + s
            )
            sub.append(sd)

        # Flat CSR: offset each system's idx_j by the cumulative atom count.
        td = sub[0]["positions"].device
        positions = torch.cat([s["positions"] for s in sub], dim=0)
        charges = torch.cat([s["charges"] for s in sub], dim=0)
        cells = torch.cat([s["cell"] for s in sub], dim=0)  # (B, 3, 3)
        sigmas = torch.cat([s["sigma"] for s in sub], dim=0)  # (B,)
        alphas = torch.cat([s["alpha"] for s in sub], dim=0)  # (B,)

        offset_atoms = 0
        offset_edges = 0
        idx_j_chunks = []
        nptr_chunks = []
        unit_shifts_chunks = []
        batch_idx_chunks = []

        nptr_chunks.append(torch.tensor([0], dtype=torch.int32, device=td))
        for s_idx, s in enumerate(sub):
            idx_j_chunks.append(s["idx_j"] + offset_atoms)
            unit_shifts_chunks.append(s["unit_shifts"])
            # Shift by cumulative edge count and drop the leading zero.
            sub_nptr = s["neighbor_ptr"][1:] + offset_edges
            nptr_chunks.append(sub_nptr)
            batch_idx_chunks.append(
                torch.full((s["n_atoms"],), s_idx, dtype=torch.int32, device=td)
            )
            offset_atoms += s["n_atoms"]
            offset_edges += s["idx_j"].shape[0]

        return {
            "positions": positions,
            "charges": charges,
            "cells": cells,
            "sigmas": sigmas,
            "alphas": alphas,
            "idx_j": torch.cat(idx_j_chunks, dim=0),
            "neighbor_ptr": torch.cat(nptr_chunks, dim=0),
            "unit_shifts": torch.cat(unit_shifts_chunks, dim=0),
            "batch_idx": torch.cat(batch_idx_chunks, dim=0),
            "n_systems": n_systems,
            "n_per_sys": n_per_sys,
        }

    @pytest.fixture
    def gpu_batch(self):
        if not torch.cuda.is_available():
            pytest.skip("Fused batched tile path is GPU-only")
        return self._build_batch_system(device="cuda:0")

    @pytest.fixture
    def cpu_batch(self):
        return self._build_batch_system(device="cpu")

    def _run_fused(self, batch_dict, *, with_pos: bool, with_q: bool):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            BatchMultipoleRealSpaceMonopoleFusedScalarFunction,
        )

        positions = batch_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = batch_dict["charges"].clone().detach().requires_grad_(with_q)
        per_sys = BatchMultipoleRealSpaceMonopoleFusedScalarFunction.apply(
            positions,
            charges,
            batch_dict["cells"],
            batch_dict["sigmas"],
            batch_dict["alphas"],
            batch_dict["idx_j"],
            batch_dict["neighbor_ptr"],
            batch_dict["unit_shifts"],
            batch_dict["batch_idx"],
        )
        return positions, charges, per_sys

    def _run_reference(self, batch_dict, *, with_pos: bool, with_q: bool):
        """Existing layered batched path → per-atom energies → scatter to per-system."""
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            BatchMultipoleRealSpaceMonopoleFunction,
        )

        positions = batch_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = batch_dict["charges"].clone().detach().requires_grad_(with_q)
        per_atom = BatchMultipoleRealSpaceMonopoleFunction.apply(
            positions,
            charges,
            batch_dict["cells"],
            batch_dict["sigmas"],
            batch_dict["alphas"],
            batch_dict["idx_j"],
            batch_dict["neighbor_ptr"],
            batch_dict["unit_shifts"],
            batch_dict["batch_idx"],
        )
        per_sys = torch.zeros(
            batch_dict["n_systems"], dtype=torch.float64, device=per_atom.device
        )
        per_sys.scatter_add_(0, batch_dict["batch_idx"].to(torch.int64), per_atom)
        return positions, charges, per_sys

    def test_gpu_forward_parity(self, gpu_batch):
        """Per-system energies match reference path to ULP at fp64 (GPU tile)."""
        _, _, e_f = self._run_fused(gpu_batch, with_pos=False, with_q=False)
        _, _, e_r = self._run_reference(gpu_batch, with_pos=False, with_q=False)
        torch.testing.assert_close(e_f, e_r, rtol=0, atol=1e-15)

    def test_gpu_grad_pos_parity(self, gpu_batch):
        """Position gradient matches reference path on GPU tile."""
        pos_f, _, e_f = self._run_fused(gpu_batch, with_pos=True, with_q=False)
        e_f.sum().backward()
        pos_r, _, e_r = self._run_reference(gpu_batch, with_pos=True, with_q=False)
        e_r.sum().backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)

    def test_gpu_both_grads_parity(self, gpu_batch):
        """Joint pos + charge gradient parity on GPU tile."""
        pos_f, q_f, e_f = self._run_fused(gpu_batch, with_pos=True, with_q=True)
        e_f.sum().backward()
        pos_r, q_r, e_r = self._run_reference(gpu_batch, with_pos=True, with_q=True)
        e_r.sum().backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)

    def test_cpu_forward_parity(self, cpu_batch):
        """Forward parity on CPU CSR path."""
        _, _, e_f = self._run_fused(cpu_batch, with_pos=False, with_q=False)
        _, _, e_r = self._run_reference(cpu_batch, with_pos=False, with_q=False)
        torch.testing.assert_close(e_f, e_r, rtol=0, atol=1e-15)

    def test_cpu_both_grads_parity(self, cpu_batch):
        """Joint pos + charge gradient parity on CPU CSR path."""
        pos_f, q_f, e_f = self._run_fused(cpu_batch, with_pos=True, with_q=True)
        e_f.sum().backward()
        pos_r, q_r, e_r = self._run_reference(cpu_batch, with_pos=True, with_q=True)
        e_r.sum().backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)

    def test_per_system_weighted_backward(self, gpu_batch):
        """Non-uniform per-system upstream weights (B,) propagate correctly."""
        pos_f, q_f, e_f = self._run_fused(gpu_batch, with_pos=True, with_q=True)
        weights = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float64, device=e_f.device)
        (weights * e_f).sum().backward()

        pos_r, q_r, e_r = self._run_reference(gpu_batch, with_pos=True, with_q=True)
        (weights * e_r).sum().backward()

        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)


def _random_dipoles_dipole(n: int, device: str, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((n, 3)) * 0.4).to(
        _torch_device(device), dtype=torch.float64
    )


class TestMultipoleRealSpaceDipoleFusedScalar:
    r"""Parity tests for ``MultipoleRealSpaceDipoleFusedScalarFunction`` (lmax=1).

    The fused Function returns scalar total energy and stashes analytical
    (`grad_pos`, `grad_q`, `grad_mu`) in ctx for backward. Parity is anchored
    against the layered ``MultipoleRealSpaceFunction`` summed externally.
    """

    @pytest.fixture
    def gpu_system(self):
        if not torch.cuda.is_available():
            pytest.skip("Fused tile path is GPU-only")
        sys_dict = _build_system(n_atoms=8, box_len=5.0, device="cuda:0", seed=0xCAFE)
        sys_dict["dipoles"] = _random_dipoles_dipole(8, "cuda:0", 0xC0FE)
        return sys_dict

    @pytest.fixture
    def cpu_system(self):
        sys_dict = _build_system(n_atoms=8, box_len=5.0, device="cpu", seed=0xC0FFEE)
        sys_dict["dipoles"] = _random_dipoles_dipole(8, "cpu", 0xBEEF)
        return sys_dict

    def _run_fused(self, sys_dict, *, with_pos: bool, with_q: bool, with_mu: bool):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceDipoleFusedScalarFunction,
        )

        positions = sys_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = sys_dict["charges"].clone().detach().requires_grad_(with_q)
        dipoles = sys_dict["dipoles"].clone().detach().requires_grad_(with_mu)
        e = MultipoleRealSpaceDipoleFusedScalarFunction.apply(
            positions,
            charges,
            dipoles,
            sys_dict["cell"],
            sys_dict["sigma"],
            sys_dict["alpha"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
        )
        return positions, charges, dipoles, e

    def _run_reference(self, sys_dict, *, with_pos: bool, with_q: bool, with_mu: bool):
        """Existing layered lmax=1 path."""
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            MultipoleRealSpaceFunction,
        )

        positions = sys_dict["positions"].clone().detach().requires_grad_(with_pos)
        charges = sys_dict["charges"].clone().detach().requires_grad_(with_q)
        dipoles = sys_dict["dipoles"].clone().detach().requires_grad_(with_mu)
        per_atom = MultipoleRealSpaceFunction.apply(
            positions,
            charges,
            dipoles,
            sys_dict["cell"],
            sys_dict["sigma"],
            sys_dict["alpha"],
            sys_dict["idx_j"],
            sys_dict["neighbor_ptr"],
            sys_dict["unit_shifts"],
        )
        return positions, charges, dipoles, per_atom.sum()

    def test_gpu_forward_parity(self, gpu_system):
        """Energy parity at ULP fp64 (no grads)."""
        _, _, _, e_f = self._run_fused(
            gpu_system, with_pos=False, with_q=False, with_mu=False
        )
        _, _, _, e_r = self._run_reference(
            gpu_system, with_pos=False, with_q=False, with_mu=False
        )
        torch.testing.assert_close(e_f, e_r, rtol=0, atol=1e-15)

    def test_gpu_all_grads_parity(self, gpu_system):
        """Joint pos + charge + dipole gradient parity on GPU tile."""
        pos_f, q_f, mu_f, e_f = self._run_fused(
            gpu_system, with_pos=True, with_q=True, with_mu=True
        )
        e_f.backward()
        pos_r, q_r, mu_r, e_r = self._run_reference(
            gpu_system, with_pos=True, with_q=True, with_mu=True
        )
        e_r.backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)

    def test_gpu_dipole_grad_only(self, gpu_system):
        """Dipole gradient alone parity (charges + positions detached)."""
        _, _, mu_f, e_f = self._run_fused(
            gpu_system, with_pos=False, with_q=False, with_mu=True
        )
        e_f.backward()
        _, _, mu_r, e_r = self._run_reference(
            gpu_system, with_pos=False, with_q=False, with_mu=True
        )
        e_r.backward()
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)

    def test_gpu_per_input_gating(self, gpu_system):
        """Only requested gradient slots get populated."""
        pos_f, q_f, mu_f, e_f = self._run_fused(
            gpu_system, with_pos=True, with_q=False, with_mu=True
        )
        e_f.backward()
        assert pos_f.grad is not None
        assert q_f.grad is None
        assert mu_f.grad is not None

    def test_cpu_all_grads_parity(self, cpu_system):
        """Same parity test on CPU CSR fused path."""
        pos_f, q_f, mu_f, e_f = self._run_fused(
            cpu_system, with_pos=True, with_q=True, with_mu=True
        )
        e_f.backward()
        pos_r, q_r, mu_r, e_r = self._run_reference(
            cpu_system, with_pos=True, with_q=True, with_mu=True
        )
        e_r.backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)


class TestBatchMultipoleRealSpaceDipoleFusedScalar:
    r"""Parity tests for ``BatchMultipoleRealSpaceDipoleFusedScalarFunction`` (batched lmax=1)."""

    def _build_batch_system(
        self, *, device: str, n_per_sys: int = 8, n_systems: int = 3
    ):
        sub = []
        for s in range(n_systems):
            sd = _build_system(
                n_atoms=n_per_sys, box_len=5.0, device=device, seed=0x200 + s
            )
            sd["dipoles"] = _random_dipoles_dipole(n_per_sys, device, 0xD00 + s)
            sub.append(sd)

        td = sub[0]["positions"].device
        positions = torch.cat([s["positions"] for s in sub], dim=0)
        charges = torch.cat([s["charges"] for s in sub], dim=0)
        dipoles = torch.cat([s["dipoles"] for s in sub], dim=0)
        cells = torch.cat([s["cell"] for s in sub], dim=0)
        sigmas = torch.cat([s["sigma"] for s in sub], dim=0)
        alphas = torch.cat([s["alpha"] for s in sub], dim=0)

        offset_atoms = 0
        offset_edges = 0
        idx_j_chunks = []
        nptr_chunks = [torch.tensor([0], dtype=torch.int32, device=td)]
        unit_shifts_chunks = []
        batch_idx_chunks = []

        for s_idx, s in enumerate(sub):
            idx_j_chunks.append(s["idx_j"] + offset_atoms)
            unit_shifts_chunks.append(s["unit_shifts"])
            sub_nptr = s["neighbor_ptr"][1:] + offset_edges
            nptr_chunks.append(sub_nptr)
            batch_idx_chunks.append(
                torch.full((s["n_atoms"],), s_idx, dtype=torch.int32, device=td)
            )
            offset_atoms += s["n_atoms"]
            offset_edges += s["idx_j"].shape[0]

        return {
            "positions": positions,
            "charges": charges,
            "dipoles": dipoles,
            "cells": cells,
            "sigmas": sigmas,
            "alphas": alphas,
            "idx_j": torch.cat(idx_j_chunks, dim=0),
            "neighbor_ptr": torch.cat(nptr_chunks, dim=0),
            "unit_shifts": torch.cat(unit_shifts_chunks, dim=0),
            "batch_idx": torch.cat(batch_idx_chunks, dim=0),
            "n_systems": n_systems,
        }

    @pytest.fixture
    def gpu_batch(self):
        if not torch.cuda.is_available():
            pytest.skip("Fused tile path is GPU-only")
        return self._build_batch_system(device="cuda:0")

    @pytest.fixture
    def cpu_batch(self):
        return self._build_batch_system(device="cpu")

    def _run_fused(self, b, *, with_pos, with_q, with_mu):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            BatchMultipoleRealSpaceDipoleFusedScalarFunction,
        )

        positions = b["positions"].clone().detach().requires_grad_(with_pos)
        charges = b["charges"].clone().detach().requires_grad_(with_q)
        dipoles = b["dipoles"].clone().detach().requires_grad_(with_mu)
        per_sys = BatchMultipoleRealSpaceDipoleFusedScalarFunction.apply(
            positions,
            charges,
            dipoles,
            b["cells"],
            b["sigmas"],
            b["alphas"],
            b["idx_j"],
            b["neighbor_ptr"],
            b["unit_shifts"],
            b["batch_idx"],
        )
        return positions, charges, dipoles, per_sys

    def _run_reference(self, b, *, with_pos, with_q, with_mu):
        from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
            BatchMultipoleRealSpaceFunction,
        )

        positions = b["positions"].clone().detach().requires_grad_(with_pos)
        charges = b["charges"].clone().detach().requires_grad_(with_q)
        dipoles = b["dipoles"].clone().detach().requires_grad_(with_mu)
        per_atom = BatchMultipoleRealSpaceFunction.apply(
            positions,
            charges,
            dipoles,
            b["cells"],
            b["sigmas"],
            b["alphas"],
            b["idx_j"],
            b["neighbor_ptr"],
            b["unit_shifts"],
            b["batch_idx"],
        )
        per_sys = torch.zeros(
            b["n_systems"], dtype=torch.float64, device=per_atom.device
        )
        per_sys.scatter_add_(0, b["batch_idx"].to(torch.int64), per_atom)
        return positions, charges, dipoles, per_sys

    def test_gpu_forward_parity(self, gpu_batch):
        _, _, _, e_f = self._run_fused(
            gpu_batch, with_pos=False, with_q=False, with_mu=False
        )
        _, _, _, e_r = self._run_reference(
            gpu_batch, with_pos=False, with_q=False, with_mu=False
        )
        torch.testing.assert_close(e_f, e_r, rtol=0, atol=1e-15)

    def test_gpu_all_grads_parity(self, gpu_batch):
        """Joint pos + charge + dipole gradient parity, batched."""
        pos_f, q_f, mu_f, e_f = self._run_fused(
            gpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        e_f.sum().backward()
        pos_r, q_r, mu_r, e_r = self._run_reference(
            gpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        e_r.sum().backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)

    def test_cpu_all_grads_parity(self, cpu_batch):
        pos_f, q_f, mu_f, e_f = self._run_fused(
            cpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        e_f.sum().backward()
        pos_r, q_r, mu_r, e_r = self._run_reference(
            cpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        e_r.sum().backward()
        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)

    def test_per_system_weighted_backward(self, gpu_batch):
        pos_f, q_f, mu_f, e_f = self._run_fused(
            gpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        weights = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float64, device=e_f.device)
        (weights * e_f).sum().backward()

        pos_r, q_r, mu_r, e_r = self._run_reference(
            gpu_batch, with_pos=True, with_q=True, with_mu=True
        )
        (weights * e_r).sum().backward()

        torch.testing.assert_close(pos_f.grad, pos_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(q_f.grad, q_r.grad, rtol=0, atol=1e-15)
        torch.testing.assert_close(mu_f.grad, mu_r.grad, rtol=0, atol=1e-15)
