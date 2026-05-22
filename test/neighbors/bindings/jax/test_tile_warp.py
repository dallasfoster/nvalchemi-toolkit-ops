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

"""Tests for JAX bindings of the single-system cluster-pair tile neighbor list."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors.tile_warp import (
    TILE_GROUP_SIZE,
    estimate_tile_neighbor_list_sizes,
    tile_neighbor_list,
)

from .conftest import requires_gpu

pytestmark = requires_gpu


def _orthorhombic_cell(cell_size: float, dtype=jnp.float32) -> jax.Array:
    return jnp.eye(3, dtype=dtype) * cell_size


def _row_neighbor_set(row: jax.Array, n: int) -> set[int]:
    return {int(x) for x in row[:n]}


class TestTileNeighborListCorrectness:
    """Smoke + small-system correctness tests."""

    def test_single_atom_no_neighbors(self):
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        cell = _orthorhombic_cell(2.0)
        nm, nn, _ = tile_neighbor_list(positions, 3.0, cell, max_neighbors=8)
        assert int(nn.sum()) == 0
        assert nm.shape == (1, 8)

    def test_two_atom_pair(self):
        positions = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=jnp.float32)
        cell = _orthorhombic_cell(2.0)
        nm, nn, _ = tile_neighbor_list(positions, 1.0, cell, max_neighbors=8)
        # Half-fill: exactly one of the two rows owns the pair.
        assert int(nn.sum()) == 1
        owner = 0 if int(nn[0]) == 1 else 1
        other = 1 - owner
        assert int(nm[owner, 0]) == other

    def test_cubic_lattice(self):
        # 4x4x4 = 64 atoms on a simple cubic lattice with spacing 1.0.
        n_side = 4
        coords = [
            [float(ix), float(iy), float(iz)]
            for ix in range(n_side)
            for iy in range(n_side)
            for iz in range(n_side)
        ]
        positions = jnp.array(coords, dtype=jnp.float32)
        cell = _orthorhombic_cell(float(n_side))
        cutoff = 1.1  # picks up just the 6 nearest neighbors
        nm, nn, _ = tile_neighbor_list(positions, cutoff, cell, max_neighbors=64)
        # Sum of half-fill emits should equal 3 * N (each atom has 6
        # nearest neighbors, half-fill stores 3 per atom on average).
        assert int(nn.sum()) == 3 * positions.shape[0]

    def test_non_aligned_N_accepts_padding(self):
        # N not divisible by TILE_GROUP_SIZE — the JAX wrapper pads
        # internally; the kernel uses sentinel Morton codes for padding.
        positions = jnp.array(
            np.random.RandomState(0).uniform(0, 10, size=(33, 3)).astype(np.float32),
        )
        cell = _orthorhombic_cell(10.0)
        nm, nn, _ = tile_neighbor_list(positions, 2.5, cell, max_neighbors=32)
        # Shape sanity; no negative counts.
        assert nm.shape == (33, 32)
        assert bool(jnp.all(nn >= 0))


class TestTileNeighborListFormats:
    """Tests for the three output formats."""

    def test_matrix_vs_coo_pair_count_match(self):
        rng = np.random.RandomState(1)
        positions = jnp.array(rng.uniform(0, 10, size=(64, 3)).astype(np.float32))
        cell = _orthorhombic_cell(10.0)
        cutoff = 2.5

        nm, nn, _ = tile_neighbor_list(positions, cutoff, cell, max_neighbors=64)
        nl, ptr, _ = tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=64,
            format="coo",
        )
        assert int(nn.sum()) == int(nl.shape[1])
        assert int(ptr[-1]) == int(nl.shape[1])

    def test_tile_format_returns_state(self):
        positions = jnp.array(
            np.random.RandomState(2).uniform(0, 10, size=(32, 3)).astype(np.float32),
        )
        cell = _orthorhombic_cell(10.0)
        out = tile_neighbor_list(positions, 2.5, cell, format="tile")
        assert len(out) == 7
        num_tiles, tile_row_group, tile_col_group, *_ = out
        n_tiles = int(num_tiles[0])
        assert n_tiles > 0
        # Tile indices fall within [0, ngroup).
        ngroup = positions.shape[0] // TILE_GROUP_SIZE
        assert int(tile_row_group[:n_tiles].max()) < ngroup
        assert int(tile_col_group[:n_tiles].max()) < ngroup


class TestTileNeighborListErrors:
    """Error path tests."""

    def test_wrong_dtype_raises(self):
        positions = jnp.zeros((32, 3), dtype=jnp.float64)
        cell = _orthorhombic_cell(4.0, dtype=jnp.float64)
        with pytest.raises(TypeError):
            tile_neighbor_list(positions, 1.0, cell, max_neighbors=8)

    def test_invalid_format_raises(self):
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell = _orthorhombic_cell(4.0)
        with pytest.raises(ValueError, match="format"):
            tile_neighbor_list(positions, 1.0, cell, format="bogus")


class TestEstimateSizes:
    """Pure-Python sizing helper tests."""

    def test_aligned_sizes(self):
        n_padded, ngroup, ngroup_padded, max_tiles = estimate_tile_neighbor_list_sizes(
            512,
        )
        assert n_padded == 512
        assert ngroup == 512 // TILE_GROUP_SIZE
        assert ngroup_padded % TILE_GROUP_SIZE == 0 and ngroup_padded > ngroup
        assert max_tiles >= ngroup

    def test_non_aligned_sizes(self):
        n_padded, ngroup, _, _ = estimate_tile_neighbor_list_sizes(33)
        assert n_padded == 64  # ceil(33 / 32) * 32
        assert ngroup == 2

    def test_zero_atoms(self):
        n_padded, ngroup, _, _ = estimate_tile_neighbor_list_sizes(0)
        # Must always reserve at least one tile group.
        assert n_padded == TILE_GROUP_SIZE
        assert ngroup == 1
