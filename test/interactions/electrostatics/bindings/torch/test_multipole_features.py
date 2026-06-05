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

r"""Integration tests for ``multipole_electrostatic_features``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_features,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    pack_charges_dipoles,
)


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _random_system(
    *,
    seed: int = 0,
    n_atoms: int = 8,
    box_len: float = 6.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    with_dipoles: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, box_len, size=(n_atoms, 3))
    charges = rng.uniform(-1.0, 1.0, n_atoms)
    charges -= charges.mean()
    dipoles = rng.standard_normal((n_atoms, 3)) * 0.3
    cell = np.eye(3) * box_len
    charges_t = torch.from_numpy(charges).to(device=device, dtype=dtype)
    out = {
        "positions": torch.from_numpy(positions).to(device=device, dtype=dtype),
        "charges": charges_t,
        "cell": torch.from_numpy(cell).to(device=device, dtype=dtype),
        "positions_np": positions,
        "charges_np": charges,
        "cell_np": cell,
    }
    if with_dipoles:
        dipoles_t = torch.from_numpy(dipoles).to(device=device, dtype=dtype)
        out["dipoles"] = dipoles_t
        out["dipoles_np"] = dipoles
        out["source_feats"] = pack_charges_dipoles(charges_t, dipoles_t)
    else:
        out["dipoles"] = None
        out["dipoles_np"] = np.zeros_like(dipoles)
        out["source_feats"] = pack_charges_dipoles(charges_t, None)
    return out


class TestBasics:
    def test_permuted_shape(self, device):
        td = _torch_device(device)
        sys = _random_system(seed=0, n_atoms=4, device=td)
        feats = multipole_electrostatic_features(
            sys["positions"],
            sys["source_feats"],
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=[0.5, 1.0, 1.5],
            kspace_cutoff=4.0,
        )
        assert feats.shape == (4, 3 * 4)
        assert feats.dtype == torch.float64
        assert feats.device.type == td

    def test_permuted_shape_multi_sigma(self, device):
        td = _torch_device(device)
        sys = _random_system(seed=1, n_atoms=4, device=td)
        feats = multipole_electrostatic_features(
            sys["positions"],
            sys["source_feats"],
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=[0.5, 1.0],
            kspace_cutoff=4.0,
        )
        assert feats.shape == (4, 2 * 4)

    def test_charges_only_branch(self, device):
        td = _torch_device(device)
        sys = _random_system(seed=2, n_atoms=5, device=td, with_dipoles=False)
        feats = multipole_electrostatic_features(
            sys["positions"],
            sys["source_feats"],
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=[0.8, 1.2],
            kspace_cutoff=4.0,
        )
        assert feats.shape == (5, 2 * 4)


class TestValidation:
    def test_rejects_empty_receiver_sigmas(self):
        with pytest.raises(ValueError, match="receiver_sigmas"):
            multipole_electrostatic_features(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[],
                kspace_cutoff=4.0,
            )

    def test_rejects_non_positive_receiver_sigma(self):
        with pytest.raises(ValueError, match="receiver_sigmas"):
            multipole_electrostatic_features(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[0.5, -0.1],
                kspace_cutoff=4.0,
            )

    def test_missing_both_kspace_cutoff_and_k_vectors(self):
        with pytest.raises(ValueError, match="k_vectors"):
            multipole_electrostatic_features(
                torch.zeros((4, 3), dtype=torch.float64),
                torch.zeros((4, 1), dtype=torch.float64),
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[1.0],
            )


class TestPhysicalInvariants:
    def test_translation_invariance_of_l0_channel(self, device):
        """l=0 per-atom features are translation-invariant (only cos²+sin² appears)."""
        td = _torch_device(device)
        sys = _random_system(seed=5, n_atoms=6, box_len=5.0, device=td)
        shift = torch.tensor([1.3, -0.7, 2.1], dtype=torch.float64, device=td)
        receiver_sigmas = [0.7, 1.3]
        n_sigma = len(receiver_sigmas)

        f_before = multipole_electrostatic_features(
            sys["positions"],
            sys["source_feats"],
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=3.5,
        )
        f_after = multipole_electrostatic_features(
            sys["positions"] + shift,
            sys["source_feats"],
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=3.5,
        )
        # Columns [0, n_sigma) are the translation-invariant l=0 channels.
        np.testing.assert_allclose(
            f_before[:, :n_sigma].detach().cpu().numpy(),
            f_after[:, :n_sigma].detach().cpu().numpy(),
            rtol=1e-12,
            atol=1e-13,
        )

    def test_zero_source_moments_give_zero_features(self, device):
        """All-zero charges and dipoles → features identically zero."""
        td = _torch_device(device)
        sys = _random_system(seed=9, n_atoms=4, device=td)
        zero_source = torch.zeros_like(sys["source_feats"])
        feats = multipole_electrostatic_features(
            sys["positions"],
            zero_source,
            sys["cell"],
            sigma=1.0,
            receiver_sigmas=[0.7, 1.3],
            kspace_cutoff=3.5,
        )
        np.testing.assert_array_equal(
            feats.detach().cpu().numpy(), np.zeros((4, 2 * 4))
        )
