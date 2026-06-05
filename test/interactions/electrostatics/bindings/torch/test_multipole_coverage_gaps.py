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

r"""Input-validation / error-contract coverage for the multipole electrostatics
public entry points.

The capability matrix and the per-feature suites exercise the *happy paths*
(energy / forces / force-loss / stress, FD-validated). This module covers the
**validation branches** — the ``raise ValueError`` guards on shapes, positive
``sigma``/``alpha``, ``feature_max_l`` range, ``receiver_sigmas`` (list *and*
tensor forms), and ``kspace_cutoff`` — which are real contracts that nothing
else hits. Validation runs before any Warp launch, so these are CPU-only and
fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    multipole_electrostatic_features,
)
from nvalchemiops.torch.interactions.electrostatics._multipole_moments import (
    infer_l_max,
    pack_multipole_moments,
    split_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import (
    multipole_reciprocal_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import (
    multipole_ewald_summation,
    multipole_real_space_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import (
    multipole_real_space_quadrupole_energy,
)
from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import (
    prepare_multipole_scf_cache,
)

_TD = "cpu"
_BOX = 6.0
_KCUT = 7.0


def _pos(n=3):
    return torch.tensor(np.random.default_rng(0).uniform(0.0, _BOX, (n, 3)), device=_TD)


def _charges(n=3):
    q = np.random.default_rng(1).standard_normal(n)
    q -= q.mean()
    return torch.tensor(q, device=_TD).unsqueeze(-1).contiguous()


def _cell():
    return torch.tensor(np.eye(3) * _BOX, device=_TD)


# --------------------------------------------------------------------------- #
# multipole_reciprocal_space_energy (single)
# --------------------------------------------------------------------------- #


class TestReciprocalValidation:
    def test_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            multipole_reciprocal_space_energy(
                torch.zeros(3, 2, device=_TD),
                _charges(),
                _cell(),
                sigma=0.5,
                alpha=0.6,
                kspace_cutoff=_KCUT,
            )

    def test_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_reciprocal_space_energy(
                _pos(),
                torch.zeros(5, 1, device=_TD),
                _cell(),
                sigma=0.5,
                alpha=0.6,
                kspace_cutoff=_KCUT,
            )

    def test_bad_cell(self):
        with pytest.raises(ValueError, match="cell must be"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                torch.zeros(2, 2, device=_TD),
                sigma=0.5,
                alpha=0.6,
                kspace_cutoff=_KCUT,
            )

    def test_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell(),
                sigma=0.0,
                alpha=0.6,
                kspace_cutoff=_KCUT,
            )

    def test_nonpositive_alpha(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell(),
                sigma=0.5,
                alpha=0.0,
                kspace_cutoff=_KCUT,
            )

    def test_missing_kspace_cutoff_and_kvectors(self):
        with pytest.raises(ValueError, match="k_vectors|kspace_cutoff"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell(),
                sigma=0.5,
                alpha=0.6,
            )


# --------------------------------------------------------------------------- #
# batched energy + reciprocal validation
# --------------------------------------------------------------------------- #


class TestBatchedValidation:
    def _bidx(self, n=3):
        return torch.zeros(n, dtype=torch.int32, device=_TD)

    def test_energy_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            multipole_electrostatic_energy(
                torch.zeros(3, 2, device=_TD),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(),
                sigma=0.5,
                kspace_cutoff=_KCUT,
            )

    def test_energy_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_electrostatic_energy(
                _pos(),
                torch.zeros(5, 1, device=_TD),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(),
                sigma=0.5,
                kspace_cutoff=_KCUT,
            )

    def test_energy_bad_cells(self):
        with pytest.raises(ValueError, match="batched cell must be"):
            multipole_electrostatic_energy(
                _pos(),
                _charges(),
                _cell(),
                batch_idx=self._bidx(),
                sigma=0.5,
                kspace_cutoff=_KCUT,
            )

    def test_energy_bad_batch_idx(self):
        with pytest.raises(ValueError, match="batch_idx must match"):
            multipole_electrostatic_energy(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(5),
                sigma=0.5,
                kspace_cutoff=_KCUT,
            )

    def test_reciprocal_bad_cells(self):
        with pytest.raises(ValueError, match="batched cell must be"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell(),
                batch_idx=self._bidx(),
                sigma=0.5,
                alpha=0.6,
                kspace_cutoff=_KCUT,
            )

    def test_reciprocal_bad_batch_idx(self):
        with pytest.raises(ValueError, match="batch_idx must match"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(5),
                sigma=0.5,
                alpha=0.6,
                kspace_cutoff=_KCUT,
            )

    def test_reciprocal_nonpositive_alpha(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            multipole_reciprocal_space_energy(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(),
                sigma=0.5,
                alpha=0.0,
                kspace_cutoff=_KCUT,
            )

    def test_reciprocal_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            multipole_reciprocal_space_energy(
                torch.zeros(3, 2, device=_TD),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(),
                sigma=0.5,
                alpha=0.6,
                kspace_cutoff=_KCUT,
            )

    def test_reciprocal_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_reciprocal_space_energy(
                _pos(),
                torch.zeros(5, 1, device=_TD),
                _cell().unsqueeze(0),
                batch_idx=self._bidx(),
                sigma=0.5,
                alpha=0.6,
                kspace_cutoff=_KCUT,
            )


# --------------------------------------------------------------------------- #
# features validation + receiver_sigmas tensor form
# --------------------------------------------------------------------------- #


class TestFeaturesValidation:
    def test_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            multipole_electrostatic_features(
                torch.zeros(3, 2, device=_TD),
                _charges(),
                _cell(),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_bad_cell(self):
        with pytest.raises(ValueError, match="cell must be"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                torch.zeros(2, 2, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell(),
                sigma=-1.0,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_electrostatic_features(
                _pos(),
                torch.zeros(5, 1, device=_TD),
                _cell(),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_bad_feature_max_l(self):
        with pytest.raises(ValueError, match="feature_max_l must be"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell(),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
                feature_max_l=3,
            )

    def test_receiver_sigmas_tensor_empty(self):
        # Exercises the ``isinstance(receiver_sigmas, torch.Tensor)`` branch.
        with pytest.raises(ValueError, match="receiver_sigmas must be non-empty"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell(),
                sigma=0.5,
                receiver_sigmas=torch.tensor([], dtype=torch.float64),
                kspace_cutoff=_KCUT,
            )

    def test_receiver_sigmas_tensor_valid(self):
        # Tensor receiver_sigmas on the happy path (covers the tolist branch end
        # to end). Tiny system → fast on CPU.
        f = multipole_electrostatic_features(
            _pos(),
            _charges(),
            _cell(),
            sigma=0.5,
            receiver_sigmas=torch.tensor([1.0, 1.5], dtype=torch.float64),
            kspace_cutoff=_KCUT,
        )
        assert torch.isfinite(f).all()

    def test_batch_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            multipole_electrostatic_features(
                torch.zeros(3, 2, device=_TD),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=torch.zeros(3, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_batch_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            multipole_electrostatic_features(
                _pos(),
                torch.zeros(5, 1, device=_TD),
                _cell().unsqueeze(0),
                batch_idx=torch.zeros(3, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_batch_bad_cells(self):
        with pytest.raises(ValueError, match="batched cell must be"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell(),
                batch_idx=torch.zeros(3, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_batch_bad_batch_idx(self):
        with pytest.raises(ValueError, match="batch_idx must match"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=torch.zeros(5, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_batch_receiver_sigmas_tensor_empty(self):
        with pytest.raises(ValueError, match="receiver_sigmas must be non-empty"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=torch.zeros(3, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=torch.tensor([], dtype=torch.float64),
                kspace_cutoff=_KCUT,
            )

    def test_batch_bad_feature_max_l(self):
        with pytest.raises(ValueError, match="feature_max_l must be"):
            multipole_electrostatic_features(
                _pos(),
                _charges(),
                _cell().unsqueeze(0),
                batch_idx=torch.zeros(3, dtype=torch.int32, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
                feature_max_l=5,
            )


# --------------------------------------------------------------------------- #
# SCF cache builder validation (single + batched)
# --------------------------------------------------------------------------- #


class TestScfCacheValidation:
    def test_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            prepare_multipole_scf_cache(
                _cell(),
                sigma=0.0,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_bad_feature_max_l(self):
        with pytest.raises(ValueError, match="feature_max_l must be"):
            prepare_multipole_scf_cache(
                _cell(),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
                feature_max_l=7,
            )

    def test_nonpositive_alpha(self):
        with pytest.raises(ValueError, match="alpha, when given"):
            prepare_multipole_scf_cache(
                _cell(),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
                alpha=-1.0,
            )

    def test_receiver_sigmas_tensor_negative(self):
        # Tensor branch + "all positive" guard.
        with pytest.raises(ValueError, match="receiver_sigmas must all be positive"):
            prepare_multipole_scf_cache(
                _cell(),
                sigma=0.5,
                receiver_sigmas=torch.tensor([-1.0], dtype=torch.float64),
                kspace_cutoff=_KCUT,
            )

    def test_batch_empty_cells(self):
        with pytest.raises(ValueError, match="at least one system"):
            prepare_multipole_scf_cache(
                torch.zeros(0, 3, 3, device=_TD),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_batch_receiver_sigmas_empty(self):
        with pytest.raises(ValueError, match="receiver_sigmas must be non-empty"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=[],
                kspace_cutoff=_KCUT,
            )

    def test_batch_nonpositive_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.0,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
            )

    def test_batch_bad_l_max(self):
        with pytest.raises(ValueError, match="l_max must be"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
                l_max=4,
            )

    def test_batch_bad_feature_max_l(self):
        with pytest.raises(ValueError, match="feature_max_l must be"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
                feature_max_l=9,
            )

    def test_batch_bad_kspace_cutoff(self):
        with pytest.raises(ValueError, match="kspace_cutoff must be"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=-1.0,
            )

    def test_batch_receiver_sigmas_negative(self):
        with pytest.raises(ValueError, match="receiver_sigmas must all be positive"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=torch.tensor([-2.0], dtype=torch.float64),
                kspace_cutoff=_KCUT,
            )

    def test_batch_alpha_nonpositive(self):
        with pytest.raises(ValueError, match="alpha, when given"):
            prepare_multipole_scf_cache(
                _cell().unsqueeze(0),
                sigma=0.5,
                receiver_sigmas=[1.0],
                kspace_cutoff=_KCUT,
                alpha=-0.5,
            )


# --------------------------------------------------------------------------- #
# Auto-parameter-estimation path (alpha / kspace_cutoff inferred)
# --------------------------------------------------------------------------- #


def _halflist(n):
    """Tiny half neighbor list (i<j, zero shift) for an n-atom system."""
    idx, ptr, sh = [], [0], []
    for i in range(n):
        for j in range(i + 1, n):
            idx.append(j)
            sh.append([0, 0, 0])
        ptr.append(len(idx))
    return (
        torch.tensor(idx, dtype=torch.int32, device=_TD),
        torch.tensor(ptr, dtype=torch.int32, device=_TD),
        torch.tensor(sh, dtype=torch.int32, device=_TD).reshape(-1, 3),
    )


class TestEwaldAutoParameters:
    """``multipole_ewald_summation`` with ``alpha``/``kspace_cutoff`` left None
    triggers ``estimate_multipole_ewald_parameters`` (the auto-param path)."""

    def test_single_system_auto_alpha_and_cutoff(self):
        n = 3
        idx, ptr, sh = _halflist(n)
        e = multipole_ewald_summation(
            _pos(n),
            _charges(n),
            _cell(),
            idx,
            ptr,
            sh,
            sigma=0.5,
        )  # alpha=None, kspace_cutoff=None → auto-estimate
        assert torch.isfinite(e).all()

    def test_batched_auto_alpha_identical_cells(self):
        # Two identical-cell systems → auto-estimated alpha agrees across the
        # batch, exercising the (B,)-collapse branch.
        n = 2
        pos = torch.cat([_pos(n), _pos(n)])
        mm = torch.cat([_charges(n), _charges(n)])
        cells = torch.stack([_cell(), _cell()])
        bidx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=_TD)
        # neighbor list: within each 2-atom system, one pair.
        idx = torch.tensor([1, 3], dtype=torch.int32, device=_TD)
        ptr = torch.tensor([0, 1, 1, 2, 2], dtype=torch.int32, device=_TD)
        sh = torch.zeros(2, 3, dtype=torch.int32, device=_TD)
        e = multipole_ewald_summation(
            pos,
            mm,
            cells,
            idx,
            ptr,
            sh,
            sigma=0.5,
            batch_idx=bidx,
            kspace_cutoff=8.0,  # provide kcut; let alpha auto-estimate
        )
        assert torch.isfinite(e).all() and e.shape == (2,)


# --------------------------------------------------------------------------- #
# multipole_real_space_energy(..., batch_idx=) validation
# --------------------------------------------------------------------------- #


class TestBatchRealSpaceValidation:
    def _args(self, **over):
        n = 3
        idx, ptr, sh = _halflist(n)
        d = dict(
            positions=_pos(n),
            multipole_moments=_charges(n),
            cells=_cell().unsqueeze(0),
            idx_j=idx,
            neighbor_ptr=ptr,
            unit_shifts=sh,
            sigmas=torch.tensor([0.5], device=_TD),
            alphas=torch.tensor([0.6], device=_TD),
            batch_idx=torch.zeros(n, dtype=torch.int32, device=_TD),
        )
        d.update(over)
        return d

    def _call(self, **over):
        a = self._args(**over)
        return multipole_real_space_energy(
            a["positions"],
            a["multipole_moments"],
            a["cells"],
            a["idx_j"],
            a["neighbor_ptr"],
            a["unit_shifts"],
            a["sigmas"],
            a["alphas"],
            batch_idx=a["batch_idx"],
        )

    def test_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            self._call(positions=torch.zeros(3, 2, device=_TD))

    def test_bad_moments(self):
        with pytest.raises(ValueError, match="multipole_moments must be"):
            self._call(multipole_moments=torch.zeros(5, 1, device=_TD))

    def test_bad_cells(self):
        with pytest.raises(ValueError, match="cells must be"):
            self._call(cells=_cell())

    def test_bad_alphas(self):
        with pytest.raises(ValueError, match="alphas must be"):
            self._call(alphas=torch.tensor([0.6, 0.7], device=_TD))

    def test_bad_sigmas(self):
        with pytest.raises(ValueError, match="sigmas must be"):
            self._call(sigmas=torch.tensor([0.5, 0.5], device=_TD))

    def test_bad_batch_idx(self):
        with pytest.raises(ValueError, match="batch_idx must match"):
            self._call(batch_idx=torch.zeros(5, dtype=torch.int32, device=_TD))


# --------------------------------------------------------------------------- #
# multipole_real_space_quadrupole_energy validation
# --------------------------------------------------------------------------- #


class TestQuadrupoleRealSpaceValidation:
    def _args(self, **over):
        n = 3
        idx, ptr, sh = _halflist(n)
        d = dict(
            positions=_pos(n),
            charges=_charges(n).squeeze(-1),
            dipoles=torch.zeros(n, 3, device=_TD),
            quadrupoles=torch.zeros(n, 3, 3, device=_TD),
            cell=_cell(),
            idx_j=idx,
            neighbor_ptr=ptr,
            unit_shifts=sh,
            sigma=torch.tensor([0.5], device=_TD),
            alpha=torch.tensor([0.6], device=_TD),
        )
        d.update(over)
        return d

    def _call(self, **over):
        a = self._args(**over)
        return multipole_real_space_quadrupole_energy(
            a["positions"],
            a["charges"],
            a["dipoles"],
            a["quadrupoles"],
            a["cell"],
            a["idx_j"],
            a["neighbor_ptr"],
            a["unit_shifts"],
            a["sigma"],
            a["alpha"],
        )

    def test_bad_positions(self):
        with pytest.raises(ValueError, match="positions must be"):
            self._call(positions=torch.zeros(3, 2, device=_TD))

    def test_bad_cell(self):
        with pytest.raises(ValueError, match="cell must be"):
            self._call(cell=torch.zeros(2, 2, device=_TD))

    def test_bad_dipoles(self):
        with pytest.raises(ValueError, match="dipoles must be"):
            self._call(dipoles=torch.zeros(3, 2, device=_TD))

    def test_bad_quadrupoles(self):
        with pytest.raises(ValueError, match="quadrupoles must be"):
            self._call(quadrupoles=torch.zeros(3, 2, 2, device=_TD))


# --------------------------------------------------------------------------- #
# packed-moment helpers validation
# --------------------------------------------------------------------------- #


class TestMomentHelpersValidation:
    def test_infer_l_max_bad_rank(self):
        with pytest.raises(ValueError, match="rank-2"):
            infer_l_max(torch.zeros(4, device=_TD))

    def test_infer_l_max_bad_last_dim(self):
        with pytest.raises(ValueError, match="last-dim must be"):
            infer_l_max(torch.zeros(3, 7, device=_TD))

    def test_split_bad_last_dim(self):
        with pytest.raises(ValueError, match="last-dim must be|rank-2"):
            split_multipole_moments(torch.zeros(3, 5, device=_TD))

    def test_pack_quadrupoles_without_dipoles(self):
        with pytest.raises(ValueError, match="quadrupoles given without dipoles"):
            pack_multipole_moments(
                torch.zeros(3, device=_TD),
                None,
                torch.zeros(3, 3, 3, device=_TD),
            )
