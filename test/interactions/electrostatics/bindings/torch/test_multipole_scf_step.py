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

r"""Integration tests for :func:`multipole_scf_step_energy` / :func:`multipole_scf_step_features`.

Prove the two-phase ``prepare_cache + scf_step`` pattern is numerically
equivalent to the one-shot bindings.
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
from nvalchemiops.torch.math.gto import NormMode


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _random_system(
    *, seed: int, n_atoms: int, box_len: float, device: str, with_dipoles: bool = True
):
    rng = np.random.default_rng(seed)
    positions = torch.from_numpy(rng.uniform(0.0, box_len, size=(n_atoms, 3))).to(
        device=device, dtype=torch.float64
    )
    charges_np = rng.uniform(-1.0, 1.0, n_atoms)
    charges_np -= charges_np.mean()
    charges = torch.from_numpy(charges_np).to(device=device, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64, device=device) * box_len
    dipoles = None
    if with_dipoles:
        dipoles_np = rng.standard_normal((n_atoms, 3)) * 0.3
        dipoles = torch.from_numpy(dipoles_np).to(device=device, dtype=torch.float64)
    source_feats = pack_charges_dipoles(charges, dipoles)
    return positions, charges, dipoles, cell, source_feats


class TestEnergyStep:
    def test_shape_and_dtype(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _random_system(
            seed=0, n_atoms=5, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )
        e = multipole_scf_step_energy(cache, positions, source_feats)
        assert e.shape == ()
        assert e.dtype == torch.float64
        assert e.device.type == td

    @pytest.mark.parametrize("seed", [0, 17, 42])
    @pytest.mark.parametrize("with_dipoles", [True, False])
    def test_parity_with_one_shot_energy(self, device, seed, with_dipoles):
        """scf_step_energy matches multipole_electrostatic_energy to float64 ULP."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _random_system(
            seed=seed, n_atoms=6, box_len=5.0, device=td, with_dipoles=with_dipoles
        )
        sigma = 1.0
        kspace_cutoff = 3.5
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=[1.0],
            kspace_cutoff=kspace_cutoff,
            l_max=1 if with_dipoles else 0,
        )
        e_scf = multipole_scf_step_energy(cache, positions, source_feats)
        e_one_shot = multipole_electrostatic_energy(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            kspace_cutoff=kspace_cutoff,
        )
        np.testing.assert_allclose(
            float(e_scf), float(e_one_shot), rtol=1e-14, atol=1e-14
        )

    def test_charges_only_matches_zero_dipole_branch(self, device):
        td = _torch_device(device)
        positions, charges, _, cell, source_feats_none = _random_system(
            seed=11, n_atoms=5, box_len=5.0, device=td, with_dipoles=False
        )
        zero_d = torch.zeros((5, 3), dtype=torch.float64, device=td)
        source_feats_zero = pack_charges_dipoles(charges, zero_d)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )
        e_none = multipole_scf_step_energy(cache, positions, source_feats_none)
        e_zero = multipole_scf_step_energy(cache, positions, source_feats_zero)
        assert torch.allclose(e_none, e_zero, rtol=1e-14, atol=1e-14)

    def test_multi_step_same_cache(self, device):
        """Different moments through one cache give different energies, each matching the one-shot call."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats_a = _random_system(
            seed=23, n_atoms=5, box_len=5.0, device=td
        )
        _, _, _, _, source_feats_b = _random_system(
            seed=31, n_atoms=5, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )
        e_a_scf = multipole_scf_step_energy(cache, positions, source_feats_a)
        e_b_scf = multipole_scf_step_energy(cache, positions, source_feats_b)
        assert float(e_a_scf) != float(e_b_scf)

        e_a_one = multipole_electrostatic_energy(
            positions,
            source_feats_a,
            cell,
            sigma=1.0,
            kspace_cutoff=3.5,
        )
        e_b_one = multipole_electrostatic_energy(
            positions,
            source_feats_b,
            cell,
            sigma=1.0,
            kspace_cutoff=3.5,
        )
        np.testing.assert_allclose(
            float(e_a_scf), float(e_a_one), rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(
            float(e_b_scf), float(e_b_one), rtol=1e-14, atol=1e-14
        )

    def test_rejects_wrong_n_atoms(self, device):
        td = _torch_device(device)
        positions, _, _, cell, _ = _random_system(
            seed=0, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.5
        )
        wrong_source = torch.zeros((3, 1), dtype=torch.float64, device=td)
        with pytest.raises(ValueError):
            multipole_scf_step_energy(cache, positions, wrong_source)


class TestFeatureStep:
    def test_shape_permuted(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _random_system(
            seed=0, n_atoms=5, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.7, 1.3],
            kspace_cutoff=3.5,
        )
        feats = multipole_scf_step_features(cache, positions, source_feats)
        assert feats.shape == (5, 2 * 4)
        assert feats.dtype == torch.float64

    def test_shape_permuted_multi_sigma(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _random_system(
            seed=1, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.8, 1.2],
            kspace_cutoff=3.5,
        )
        feats = multipole_scf_step_features(cache, positions, source_feats)
        assert feats.shape == (4, 2 * 4)

    @pytest.mark.parametrize("seed", [0, 17, 42])
    @pytest.mark.parametrize("with_dipoles", [True, False])
    def test_parity_with_one_shot_features(self, device, seed, with_dipoles):
        """scf_step_features matches multipole_electrostatic_features to float64 ULP."""
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _random_system(
            seed=seed, n_atoms=6, box_len=5.0, device=td, with_dipoles=with_dipoles
        )
        sigma = 1.0
        receiver_sigmas = [0.6, 1.0, 1.4]
        kspace_cutoff = 3.5
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=kspace_cutoff,
            l_max=1 if with_dipoles else 0,
        )
        f_scf = multipole_scf_step_features(cache, positions, source_feats)
        f_one_shot = multipole_electrostatic_features(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=kspace_cutoff,
        )
        np.testing.assert_allclose(
            f_scf.detach().cpu().numpy(),
            f_one_shot.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )

    @pytest.mark.parametrize("mode_name", ["multipoles", "receiver", "none"])
    def test_parity_across_feature_norm_modes(self, device, mode_name):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _random_system(
            seed=7, n_atoms=5, box_len=5.0, device=td
        )
        sigma = 1.0
        receiver_sigmas = [0.8, 1.2]
        kspace_cutoff = 3.5
        mode = NormMode[mode_name.upper()]
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=kspace_cutoff,
            feature_normalize=mode,
        )
        f_scf = multipole_scf_step_features(cache, positions, source_feats)
        f_one_shot = multipole_electrostatic_features(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=kspace_cutoff,
            feature_normalize=mode,
        )
        np.testing.assert_allclose(
            f_scf.detach().cpu().numpy(),
            f_one_shot.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_include_self_interaction(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats = _random_system(
            seed=19, n_atoms=4, box_len=5.0, device=td
        )
        sigma = 1.0
        receiver_sigmas = [1.0]
        kspace_cutoff = 3.5
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=kspace_cutoff,
        )
        f_scf = multipole_scf_step_features(
            cache, positions, source_feats, include_self_interaction=True
        )
        f_one_shot = multipole_electrostatic_features(
            positions,
            source_feats,
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=True,
        )
        np.testing.assert_allclose(
            f_scf.detach().cpu().numpy(),
            f_one_shot.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_multi_step_same_cache(self, device):
        td = _torch_device(device)
        positions, _, _, cell, source_feats_a = _random_system(
            seed=101, n_atoms=4, box_len=5.0, device=td
        )
        _, _, _, _, source_feats_b = _random_system(
            seed=103, n_atoms=4, box_len=5.0, device=td
        )
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.9, 1.1],
            kspace_cutoff=3.5,
        )
        f_a = multipole_scf_step_features(cache, positions, source_feats_a)
        f_b = multipole_scf_step_features(cache, positions, source_feats_b)
        assert float((f_a - f_b).abs().max()) > 1e-10
        f_a_one = multipole_electrostatic_features(
            positions,
            source_feats_a,
            cell,
            sigma=1.0,
            receiver_sigmas=[0.9, 1.1],
            kspace_cutoff=3.5,
        )
        np.testing.assert_allclose(
            f_a.detach().cpu().numpy(),
            f_a_one.detach().cpu().numpy(),
            rtol=1e-14,
            atol=1e-14,
        )
