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

r"""Unit tests for :class:`MultipoleSCFCache` and :func:`prepare_multipole_scf_cache`.

The cache holds only position-independent state (k-vectors, φ̂, per-k factors,
overlap constants, LUTs); the cos/sin structure-factor table lives inside
:class:`MultipoleRhoFunction` instead.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    MultipoleSCFCache,
    prepare_multipole_scf_cache,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.math import FIELD_CONSTANT, compute_overlap_constants
from nvalchemiops.torch.math.gto import NormMode, inv_cl


def _torch_device(device: str) -> str:
    return "cuda" if "cuda" in device else "cpu"


def _cell(*, box_len: float, device: str) -> torch.Tensor:
    return torch.eye(3, dtype=torch.float64, device=device) * box_len


class TestCacheShapes:
    def test_shapes_and_metadata(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.6, 1.0, 1.4],
            kspace_cutoff=3.5,
        )
        assert isinstance(cache, MultipoleSCFCache)
        assert cache.n_sigma == 3
        assert cache.l_max == 1
        assert cache.sigma == 1.0
        assert cache.receiver_sigmas == (0.6, 1.0, 1.4)
        assert cache.density_normalize == NormMode.MULTIPOLES
        assert cache.feature_normalize == NormMode.RECEIVER

        n_k = cache.n_k
        assert cache.k_vectors.shape == (n_k, 3)
        assert cache.k_norm2.shape == (n_k,)
        assert cache.source_phi_hat.shape == (n_k, 4, 2)
        assert cache.receiver_phi_hat.shape == (n_k, 3, 4, 2)
        assert cache.per_k_factor.shape == (n_k,)
        assert cache.k_factor_proj.shape == (n_k,)
        # [l=0, l=1, l=2] source self-overlap constants.
        assert cache.source_overlap_constants.shape == (3,)
        assert cache.feature_overlap_constants.shape == (3, 2)
        # Default feature_max_l=1 → no l=2 receiver block / self constant.
        assert cache.feature_max_l == 1
        assert cache.feature_overlap_l2 is None
        assert cache.out_col_lut_natural.shape == (3, 4)
        assert cache.out_col_lut_permuted.shape == (3, 4)
        assert cache.volume.shape == ()
        assert cache.device.type == td
        # The cache must not hold position-dependent cos/sin tables.
        assert not hasattr(cache, "cosines")
        assert not hasattr(cache, "sines")

    def test_lmax_zero_feature_oc_has_zero_l1_column(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.8],
            kspace_cutoff=3.0,
            l_max=0,
        )
        np.testing.assert_array_equal(
            cache.feature_overlap_constants[:, 1].detach().cpu().numpy(),
            np.zeros(1),
        )
        assert float(cache.source_overlap_constants[1]) == 0.0

    def test_origin_is_row_zero_of_k_vectors(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[1.0],
            kspace_cutoff=3.0,
        )
        np.testing.assert_array_equal(
            cache.k_vectors[0].detach().cpu().numpy(), np.zeros(3)
        )


class TestCacheContents:
    """The cache must be content-equivalent to what the one-shot binding would compute."""

    def test_per_k_factor_matches_closed_form(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.0
        )
        k_norm2 = cache.k_norm2.detach().cpu().numpy()
        expected = np.zeros_like(k_norm2)
        mask = k_norm2 > 0.0
        expected[mask] = FIELD_CONSTANT / k_norm2[mask]
        np.testing.assert_allclose(
            cache.per_k_factor.detach().cpu().numpy(),
            expected,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_k_factor_proj_convention(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell, sigma=1.0, receiver_sigmas=[1.0], kspace_cutoff=3.0
        )
        kfp = cache.k_factor_proj.detach().cpu().numpy()
        assert kfp[0] == 0.5
        assert np.all(kfp[1:] == 1.0)

    def test_feature_overlap_constants_match_host_computation(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        sigma = 1.0
        receiver_sigmas = [0.7, 1.3]
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=sigma,
            receiver_sigmas=receiver_sigmas,
            kspace_cutoff=3.0,
        )
        oc_np = compute_overlap_constants(
            max_L=1,
            sigma_source=sigma,
            sigmas_receive=receiver_sigmas,
            normalize_source=NormMode.MULTIPOLES,
            normalize_receive=NormMode.RECEIVER,
        )
        np.testing.assert_allclose(
            cache.feature_overlap_constants.detach().cpu().numpy(),
            oc_np,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_receiver_phi_at_origin_zero_for_l1(self, device):
        """Receiver φ̂ at k = 0 has zero l=1 block (Y_1^m ∝ k̂ · k vanishes)."""
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[0.8, 1.2],
            kspace_cutoff=3.0,
        )
        phi_origin_l1 = cache.receiver_phi_hat[0, :, 1:4, :].detach().cpu().numpy()
        np.testing.assert_array_equal(phi_origin_l1, np.zeros_like(phi_origin_l1))

    def test_source_phi_hat_l0_matches_closed_form_at_origin(self, device):
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        sigma = 1.0
        cache = prepare_multipole_scf_cache(
            cell, sigma=sigma, receiver_sigmas=[1.0], kspace_cutoff=3.0
        )
        expected = (
            inv_cl(sigma, 0, NormMode.MULTIPOLES)
            * 4.0
            * math.pi
            * math.sqrt(math.pi / 2.0)
            * sigma**3
            / math.sqrt(4.0 * math.pi)
        )
        assert cache.source_phi_hat[0, 0, 0].item() == pytest.approx(
            expected, rel=1e-14
        )
        assert cache.source_phi_hat[0, 0, 1].item() == 0.0


class TestCacheReuse:
    def test_k_vectors_passed_through(self, device):
        """Supplying pre-built k_vectors produces a cache using exactly those."""
        td = _torch_device(device)
        cell = _cell(box_len=5.0, device=td)
        kspace_cutoff = 3.5
        k_half = generate_k_vectors_ewald_summation(cell, kspace_cutoff)
        k = torch.cat([k_half.new_zeros((1, 3)), k_half], dim=0).to(torch.float64)

        cache = prepare_multipole_scf_cache(
            cell,
            sigma=1.0,
            receiver_sigmas=[1.0],
            k_vectors=k,
        )
        np.testing.assert_array_equal(
            cache.k_vectors.detach().cpu().numpy(), k.detach().cpu().numpy()
        )


class TestValidation:
    def test_rejects_empty_receiver_sigmas(self):
        with pytest.raises(ValueError, match="receiver_sigmas"):
            prepare_multipole_scf_cache(
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[],
                kspace_cutoff=3.0,
            )

    def test_rejects_missing_kspace_cutoff_and_k_vectors(self):
        with pytest.raises(ValueError, match="k_vectors"):
            prepare_multipole_scf_cache(
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[1.0],
            )

    def test_rejects_bad_l_max(self):
        # l_max=2 is valid; l_max=3 remains unsupported.
        with pytest.raises(ValueError, match="l_max"):
            prepare_multipole_scf_cache(
                torch.eye(3, dtype=torch.float64) * 5.0,
                sigma=1.0,
                receiver_sigmas=[1.0],
                kspace_cutoff=3.0,
                l_max=3,
            )

    def test_rejects_bad_cell_shape(self):
        with pytest.raises(ValueError, match="cell"):
            prepare_multipole_scf_cache(
                torch.eye(4, dtype=torch.float64),
                sigma=1.0,
                receiver_sigmas=[1.0],
                kspace_cutoff=3.0,
            )
