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

r"""
Cross-cutting tests for the multipole autograd + torch.compile surface.

``multipole_scf_step_energy`` wraps the rho(k) assembly Warp kernel in a
``torch.autograd.Function`` with analytical backward for positions and
source_feats; ``multipole_scf_step_features`` adds the feature-projection
Function on top. The ``torch.compile`` tests exercise the autograd.Function /
Dynamo boundary: the compiled graph must register each Function as an opaque
differentiable primitive and replay its backward on ``.backward()``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    multipole_electrostatic_features,
    multipole_scf_step_energy,
    multipole_scf_step_features,
    prepare_multipole_scf_cache,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_charges_dipoles,
)


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _build_test_system(*, seed: int, n_atoms: int, box_len: float, device: str):
    rng = np.random.default_rng(seed)
    positions = torch.from_numpy(rng.uniform(0.0, box_len, size=(n_atoms, 3))).to(
        device=device, dtype=torch.float64
    )
    charges_np = rng.uniform(-1.0, 1.0, n_atoms)
    charges_np -= charges_np.mean()
    charges = torch.from_numpy(charges_np).to(device=device, dtype=torch.float64)
    dipoles = torch.from_numpy(rng.standard_normal((n_atoms, 3)) * 0.3).to(
        device=device, dtype=torch.float64
    )
    cell = torch.eye(3, dtype=torch.float64, device=device) * box_len
    source_feats = pack_charges_dipoles(charges, dipoles)
    return positions, charges, dipoles, cell, source_feats


# =============================================================================
# torch.compile
# =============================================================================


class TestTorchCompile:
    r"""``torch.compile`` smoke tests — the custom op should appear as a single opaque node and numerics must match eager."""

    def test_compile_scf_step_energy(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=0, n_atoms=5, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )

        def step(sf):
            return multipole_scf_step_energy(cache, positions, sf)

        e_eager = step(source_feats)
        compiled = torch.compile(step, fullgraph=False)
        e_compiled = compiled(source_feats)
        # Tolerate <=1-ULP float64 summation-order drift from graph breaks at
        # the Warp-op boundary.
        np.testing.assert_allclose(
            float(e_eager), float(e_compiled), rtol=1e-14, atol=1e-14
        )

    def test_compile_scf_step_features(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=1, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.7, 1.3],
            kspace_cutoff=3.5,
        )

        def step(sf):
            return multipole_scf_step_features(cache, positions, sf)

        f_eager = step(source_feats)
        compiled = torch.compile(step, fullgraph=False)
        f_compiled = compiled(source_feats)
        # Tolerate <=1-ULP float64 summation-order drift.
        np.testing.assert_allclose(
            f_eager.detach().cpu().numpy(),
            f_compiled.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_compile_one_shot_energy(self, device):
        """One-shot binding survives ``torch.compile`` with <=1-ULP drift.

        The one-shot path rebuilds the SCF cache on every call (including a
        scipy ``compute_overlap_constants`` call that causes dynamo graph
        breaks), so a few trailing ULPs of summation-order noise slip in. The
        step-level tests above reuse a pre-built cache and stay bit-exact.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=2, n_atoms=4, box_len=5.0, device=td
        )

        def fn(sf):
            return multipole_electrostatic_energy(
                positions,
                sf,
                cell,
                sigma=1.0,
                kspace_cutoff=3.5,
            )

        e_eager = fn(source_feats)
        compiled = torch.compile(fn, fullgraph=False)
        e_compiled = compiled(source_feats)
        np.testing.assert_allclose(
            float(e_eager), float(e_compiled), rtol=1e-12, atol=1e-14
        )

    def test_compile_one_shot_features(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=3, n_atoms=4, box_len=5.0, device=td
        )

        def fn(sf):
            return multipole_electrostatic_features(
                positions,
                sf,
                cell,
                sigma=1.0,
                receiver_sigmas=[0.8, 1.2],
                kspace_cutoff=3.5,
            )

        f_eager = fn(source_feats)
        compiled = torch.compile(fn, fullgraph=False)
        f_compiled = compiled(source_feats)
        np.testing.assert_allclose(
            f_eager.detach().cpu().numpy(),
            f_compiled.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
        )


# =============================================================================
# Autograd (forward-only current landing)
# =============================================================================


class TestAutogradEnergy:
    """``multipole_scf_step_energy`` autograd tests.

    Analytical moment gradients flow through ``MultipoleRhoFunction``: the
    rho-pipeline contribution reaches ``source_feats`` alongside the
    self-interaction term; the position gradient is wired separately.
    """

    def test_forward_with_requires_grad_does_not_raise(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=0, n_atoms=5, box_len=5.0, device=td
        )
        sf = source_feats.clone().requires_grad_(True)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )
        e = multipole_scf_step_energy(cache, positions, sf)
        # requires_grad: the self-interaction torch subtract combines the
        # detached raw energy with grad-tracking charge/dipole terms.
        assert e.requires_grad

    @pytest.mark.parametrize("seed", [11, 23, 31])
    def test_gradcheck_source_feats(self, device, seed):
        r"""``gradcheck`` on ``source_feats``.

        The full gradient is the sum of the self-interaction torch term and
        the rho-pipeline term; ``gradcheck`` verifies both together vs FD.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=seed, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )

        def fn(sf):
            return multipole_scf_step_energy(cache, positions, sf)

        sf = source_feats.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(fn, (sf,), eps=1e-6, atol=1e-4)

    @pytest.mark.parametrize("seed", [11, 23, 31])
    def test_gradcheck_positions(self, device, seed):
        r"""``gradcheck`` on positions.

        The position gradient flows via
        ``_position_gradient_from_rhok_kernel`` (analytical backward, one Warp
        launch). source_feats is held fixed; the next test does the joint check.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=seed, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )

        def fn(p):
            return multipole_scf_step_energy(cache, p, source_feats)

        p = positions.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(fn, (p,), eps=1e-6, atol=1e-4)

    @pytest.mark.parametrize("seed", [11, 23, 31])
    def test_gradcheck_joint(self, device, seed):
        r"""``gradcheck`` passes on all inputs simultaneously.

        Confirms that ``MultipoleRhoFunction``'s backward produces
        consistent gradients for both slots (positions, source_feats)
        under the same cotangent path, not just any one at a time.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=seed, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )

        def fn(p, sf):
            return multipole_scf_step_energy(cache, p, sf)

        p = positions.clone().requires_grad_(True)
        sf = source_feats.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(fn, (p, sf), eps=1e-6, atol=1e-4)

    def test_positions_grad_flows_through_one_shot_binding(self, device):
        """The ``multipole_electrostatic_energy`` one-shot binding also reports a position gradient."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=17, n_atoms=4, box_len=5.0, device=td
        )
        p = positions.clone().requires_grad_(True)
        e = multipole_electrostatic_energy(
            p, source_feats, cell, sigma=1.0, kspace_cutoff=3.5
        )
        e.backward()
        assert p.grad is not None
        assert p.grad.shape == positions.shape
        assert float(p.grad.abs().max()) > 0.0

    def test_warp_pipeline_contributes_to_gradients(self, device):
        r"""The full gradient differs from the self-interaction term alone.

        Regression anchor: fails if a future refactor drops the analytical
        backward and leaves the Warp-pipeline branch detached.
        """
        td = _torch_device(device)
        positions, charges, dipoles, cell, source_feats = _build_test_system(
            seed=11, n_atoms=6, box_len=5.0, device=td
        )
        sf = source_feats.clone().requires_grad_(True)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )
        e = multipole_scf_step_energy(cache, positions, sf)
        e.backward()
        # Extract charge / dipole(Cartesian) gradients from source_feats.grad.
        # sph layout: [:, 0] = charge, [:, 1:4] = (mu_y, mu_z, mu_x).
        # Cartesian (x, y, z) dipole grads live at columns [3, 1, 2].
        chg_grad = sf.grad.detach()[..., 0]
        dip_grad_cart = sf.grad.detach()[..., [3, 1, 2]]
        # Self-interaction contribution alone.
        self_int_c = -cache.source_overlap_constants[0].detach() * charges.detach()
        self_int_d = -cache.source_overlap_constants[1].detach() * dipoles.detach()
        # The full gradient must include a non-zero Warp-pipeline term
        # on top of the self-interaction.
        warp_contrib_c = float((chg_grad - self_int_c).abs().max())
        warp_contrib_d = float((dip_grad_cart - self_int_d).abs().max())
        assert warp_contrib_c > 1e-6, (
            f"Warp contribution to charges.grad unexpectedly zero: "
            f"max |Δ| = {warp_contrib_c:.2e}"
        )
        assert warp_contrib_d > 1e-6, (
            f"Warp contribution to dipoles.grad unexpectedly zero: "
            f"max |Δ| = {warp_contrib_d:.2e}"
        )

    def test_include_self_interaction_true_output_still_requires_grad(self, device):
        r"""With ``include_self_interaction=True``, the output still requires grad.

        The rho-pipeline flows through ``MultipoleRhoFunction``, so the output
        is autograd-connected regardless of the self-interaction subtract.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=23, n_atoms=4, box_len=5.0, device=td
        )
        sf = source_feats.clone().requires_grad_(True)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )
        e = multipole_scf_step_energy(
            cache, positions, sf, include_self_interaction=True
        )
        assert e.requires_grad
        e.backward()
        assert sf.grad is not None
        assert float(sf.grad.abs().max()) > 0.0


class TestAutogradFeatures:
    """``multipole_scf_step_features`` autograd tests.

    Analytical gradients flow through the feature projection:
    ``MultipoleProjectRawFeaturesFunction`` handles d/dV and d/dr;
    ``MultipoleRhoFunction`` handles d/d source_feats through the rho->V
    chain; the self-interaction subtract and output permutation are
    autograd-native torch ops.
    """

    def test_output_requires_grad(self, device):
        """The feature output is autograd-connected to its inputs."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=0, n_atoms=4, box_len=5.0, device=td
        )
        sf = source_feats.clone().requires_grad_(True)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.8, 1.2],
            kspace_cutoff=3.5,
        )
        f = multipole_scf_step_features(cache, positions, sf)
        assert f.requires_grad

    def test_one_shot_features_autograd_connected(self, device):
        """The one-shot binding also produces a grad-tracking feature tensor."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=4, n_atoms=4, box_len=5.0, device=td
        )
        sf = source_feats.clone().requires_grad_(True)
        f = multipole_electrostatic_features(
            positions,
            sf,
            cell,
            sigma=1.0,
            receiver_sigmas=[0.8, 1.2],
            kspace_cutoff=3.5,
        )
        assert f.requires_grad

    @pytest.mark.parametrize("seed", [11, 23, 31])
    def test_gradcheck_features(self, device, seed):
        r"""``gradcheck`` on (positions, source_feats) for features.

        Exercises the full autograd path: MultipoleRhoFunction -> torch
        per_k_factor multiply -> MultipoleProjectRawFeaturesFunction -> torch
        self-int subtract -> torch index_select permutation.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=seed, n_atoms=3, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[0.8, 1.2], kspace_cutoff=3.5
        )

        def fn(p, sf):
            return multipole_scf_step_features(cache, p, sf)

        p = positions.clone().requires_grad_(True)
        sf = source_feats.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(fn, (p, sf), eps=1e-6, atol=1e-4)

    def test_feature_backward_matches_one_shot(self, device):
        """scf_step_features and multipole_electrostatic_features give the same gradients."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=7, n_atoms=4, box_len=5.0, device=td
        )
        sigma = 1.0
        receiver_sigmas = [0.8, 1.2]
        kspace_cutoff = 3.5
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=kspace_cutoff,
        )

        p_step = positions.clone().requires_grad_(True)
        sf_step = source_feats.clone().requires_grad_(True)
        f_step = multipole_scf_step_features(cache, p_step, sf_step)
        f_step.sum().backward()

        p_one = positions.clone().requires_grad_(True)
        sf_one = source_feats.clone().requires_grad_(True)
        f_one = multipole_electrostatic_features(
            p_one,
            sf_one,
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=kspace_cutoff,
        )
        f_one.sum().backward()

        np.testing.assert_allclose(
            p_step.grad.detach().cpu().numpy(),
            p_one.grad.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            sf_step.grad.detach().cpu().numpy(),
            sf_one.grad.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-13,
        )


class TestAutogradOneShotEnergy:
    """One-shot energy binding inherits the step's autograd behavior."""

    def test_one_shot_energy_backward_matches_step(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=5, n_atoms=5, box_len=5.0, device=td
        )
        sf_one = source_feats.clone().requires_grad_(True)
        sf_step = source_feats.clone().requires_grad_(True)

        e_one = multipole_electrostatic_energy(
            positions,
            sf_one,
            cell,
            sigma=1.0,
            kspace_cutoff=3.5,
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )
        e_step = multipole_scf_step_energy(cache, positions, sf_step)
        e_one.backward()
        e_step.backward()
        np.testing.assert_allclose(
            sf_one.grad.detach().cpu().numpy(),
            sf_step.grad.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )


# =============================================================================
# Combined compile + autograd
# =============================================================================


class TestCompileAutograd:
    """``torch.compile``-wrapped autograd still produces the same gradients."""

    def test_compiled_scf_step_energy_backward(self, device):
        """``torch.compile(step)`` + ``.backward()`` reproduces eager gradients on all inputs.

        Exercises the ``torch.autograd.Function`` boundary under
        ``torch.compile`` specifically: the compiled graph must
        register the Function as an opaque differentiable primitive
        and replay its backward when ``.backward()`` is called on the
        compiled output. Regression anchor: if Dynamo changes how
        autograd.Functions are handled, this test catches it.
        """
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _build_test_system(
            seed=7, n_atoms=5, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )

        def step(p, sf):
            return multipole_scf_step_energy(cache, p, sf)

        p_eager = positions.clone().requires_grad_(True)
        sf_eager = source_feats.clone().requires_grad_(True)
        e_eager = step(p_eager, sf_eager)
        e_eager.backward()

        compiled = torch.compile(step, fullgraph=False)
        p_compiled = positions.clone().requires_grad_(True)
        sf_compiled = source_feats.clone().requires_grad_(True)
        e_compiled = compiled(p_compiled, sf_compiled)
        e_compiled.backward()

        # rtol=1e-12 tolerates <=1-ULP graph-break reordering in the cos/sin
        # fresh-compute path.
        np.testing.assert_allclose(
            p_eager.grad.detach().cpu().numpy(),
            p_compiled.grad.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            sf_eager.grad.detach().cpu().numpy(),
            sf_compiled.grad.detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-14,
        )


# =============================================================================
# Double-backward (force / stress loss support)
# =============================================================================


class TestDoubleBackward:
    r"""Second-order autograd: ``create_graph=True`` on d E/d r so MLIP losses
    of the form ``l = w(E) + w(F) + w(S)`` flow gradients back to source_feats
    (and cell via cell autograd) through the full position gradient.

    The second-order derivative kernels are verified vs FD at the kernel level
    elsewhere; here we check that the autograd glue composes correctly.
    """

    @staticmethod
    def _system(device: str, n_atoms: int = 4, box_len: float = 6.0, seed: int = 13):
        return _build_test_system(
            seed=seed, n_atoms=n_atoms, box_len=box_len, device=device
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_gradgradcheck_energy_moments(self, device):
        """``gradgradcheck`` on d E/d source_feats — the moment-grad double-backward path."""
        positions, _, _, cell, source_feats = self._system(device)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            kspace_cutoff=1.5,
            l_max=1,
        )
        pos = positions.clone()  # detached — not the variable we differentiate here
        sf = source_feats.clone().requires_grad_(True)

        def f(sf_):
            return multipole_scf_step_energy(cache, pos, sf_)

        assert torch.autograd.gradgradcheck(f, (sf,), eps=1e-6, atol=1e-4)

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_gradgradcheck_energy_positions(self, device):
        """``gradgradcheck`` on d E/d r — the position-Hessian path."""
        positions, _, _, cell, source_feats = self._system(device)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            kspace_cutoff=1.5,
            l_max=1,
        )
        sf = source_feats.clone()
        p = positions.clone().requires_grad_(True)

        def f(p_):
            return multipole_scf_step_energy(cache, p_, sf)

        assert torch.autograd.gradgradcheck(f, (p,), eps=1e-6, atol=1e-4)

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_force_loss_backprop_to_moments(self, device):
        """End-to-end MLIP-style: ``forces = -d E/d r`` with
        ``create_graph=True``; then ``force_loss.backward()`` must flow
        gradients back to source_feats (the d F/d theta = -d^2 E/(d r d theta) path).
        """
        positions, _, _, cell, source_feats = self._system(device)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            kspace_cutoff=1.5,
            l_max=1,
        )
        sf = source_feats.clone().requires_grad_(True)
        p = positions.clone().requires_grad_(True)

        energy = multipole_scf_step_energy(cache, p, sf)
        (forces_neg,) = torch.autograd.grad(energy, p, create_graph=True)
        forces = -forces_neg
        # Any scalar loss on forces — check non-zero gradient to source_feats.
        loss = (forces * forces).sum()
        (grad_sf,) = torch.autograd.grad(loss, (sf,), retain_graph=False)
        assert grad_sf.abs().max().item() > 0.0, (
            "force loss produced no source_feats gradient"
        )

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_force_loss_backprop_to_positions(self, device):
        """Position Hessian diagonal: d F/d r. Cheap smoke check — require a
        non-zero gradient and that the pipeline doesn't raise."""
        positions, _, _, cell, source_feats = self._system(device)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            kspace_cutoff=1.5,
            l_max=1,
        )
        sf = source_feats.clone()
        p = positions.clone().requires_grad_(True)

        energy = multipole_scf_step_energy(cache, p, sf)
        (forces_neg,) = torch.autograd.grad(energy, p, create_graph=True)
        loss = forces_neg.pow(2).sum()
        (gp,) = torch.autograd.grad(loss, (p,))
        assert gp.abs().max().item() > 0.0

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_stress_via_cell_autograd(self, device):
        """First-order stress: ``-d E/d cell`` via cell autograd. The cache
        carries autograd through ``(source_phi_hat, receiver_phi_hat,
        per_k_factor, volume)``, so a gradient w.r.t. ``cell`` materializes
        without any manual virial computation."""
        positions, _, _, _cell, source_feats = self._system(device)
        box_len = 6.0
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .mul_(box_len)
            .requires_grad_(True)
        )
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            kspace_cutoff=1.5,
            l_max=1,
        )
        energy = multipole_scf_step_energy(cache, positions, source_feats)
        (gc,) = torch.autograd.grad(energy, cell)
        assert gc.abs().max().item() > 0.0, "expected nonzero ∂E/∂cell"

    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
    )
    def test_features_force_like_double_backward(self, device):
        """Feature-step variant of the force-loss path."""
        positions, _, _, cell, source_feats = self._system(device)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.5,
            receiver_sigmas=[1.5],
            kspace_cutoff=1.5,
            l_max=1,
        )
        sf = source_feats.clone().requires_grad_(True)
        p = positions.clone().requires_grad_(True)

        feats = multipole_scf_step_features(cache, p, sf)
        # A positions-depending scalar: Σ_{i,col} feats[i, col].
        s = feats.sum()
        (gp,) = torch.autograd.grad(s, p, create_graph=True)
        # Scalar loss that's a function of the position gradient of the
        # feature sum; backprop to source_feats.
        loss = gp.pow(2).sum()
        (grad_sf,) = torch.autograd.grad(loss, (sf,))
        assert grad_sf.abs().max().item() > 0.0
