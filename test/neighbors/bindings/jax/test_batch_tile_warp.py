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

"""Tests for JAX bindings of the batched cluster-pair tile neighbor list."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.neighbors.batch_tile_warp import (
    TILE_GROUP_SIZE,
    batch_tile_neighbor_list,
    estimate_batch_tile_neighbor_list_sizes,
)

from .conftest import requires_gpu

pytestmark = requires_gpu


def _make_batch(
    sys_sizes: list[int],
    cell_sizes: list[float],
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rng = np.random.RandomState(seed)
    pos_chunks, cells = [], []
    for sz, L in zip(sys_sizes, cell_sizes):
        pos_chunks.append(rng.uniform(0, L, size=(sz, 3)).astype(np.float32))
        cells.append(np.eye(3, dtype=np.float32) * L)
    positions = jnp.array(np.concatenate(pos_chunks, axis=0))
    cell_batch = jnp.array(np.stack(cells, axis=0))
    bp = [0]
    for sz in sys_sizes:
        bp.append(bp[-1] + sz)
    batch_ptr = jnp.array(bp, dtype=jnp.int32)
    return positions, cell_batch, batch_ptr


class TestBatchTileNeighborListCorrectness:
    """Smoke + multi-system tests."""

    def test_single_system_batch(self):
        positions, cell_batch, batch_ptr = _make_batch([64], [10.0], seed=5)
        cutoff = 2.5
        nm, nn, _ = batch_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
        )
        assert nm.shape == (64, 64)
        assert bool(jnp.all(nn >= 0))

    def test_two_systems_different_sizes(self):
        positions, cell_batch, batch_ptr = _make_batch([64, 96], [10.0, 8.0], seed=6)
        cutoff = 2.5
        nm, nn, _ = batch_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=64,
        )
        assert nm.shape == (160, 64)
        # Per-system slices have independent neighbor counts.
        nn_np = np.asarray(nn)
        assert nn_np[:64].sum() > 0
        assert nn_np[64:].sum() > 0

    def test_neighbors_stay_within_system(self):
        """Every emitted neighbor must share a system with its source atom."""
        positions, cell_batch, batch_ptr = _make_batch([48, 80], [9.0, 7.0], seed=7)
        cutoff = 2.5
        nm, nn, _ = batch_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=32,
        )
        N = int(batch_ptr[-1])
        # Per-atom system index.
        sys_sizes = [int(batch_ptr[i + 1] - batch_ptr[i]) for i in range(2)]
        batch_idx = np.concatenate(
            [
                np.zeros(sys_sizes[0], dtype=np.int32),
                np.ones(sys_sizes[1], dtype=np.int32),
            ]
        )
        nm_np = np.asarray(nm)
        nn_np = np.asarray(nn)
        for i in range(N):
            for k in range(int(nn_np[i])):
                j = int(nm_np[i, k])
                if j == N:  # sentinel padding
                    continue
                assert batch_idx[i] == batch_idx[j], (
                    f"Cross-system pair ({i}, {j}) emitted"
                )


class TestBatchTileNeighborListFormats:
    """Tests for the three output formats."""

    def test_matrix_vs_coo_pair_count_match(self):
        positions, cell_batch, batch_ptr = _make_batch([48, 80], [9.0, 7.0], seed=8)
        cutoff = 2.5
        nm, nn, _ = batch_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=32,
        )
        nl, ptr, _ = batch_tile_neighbor_list(
            positions,
            cutoff,
            cell_batch,
            batch_ptr,
            max_neighbors=32,
            format="coo",
        )
        assert int(nn.sum()) == int(nl.shape[1])
        assert int(ptr[-1]) == int(nl.shape[1])

    def test_tile_format_returns_state(self):
        positions, cell_batch, batch_ptr = _make_batch([32, 64], [8.0, 8.0], seed=9)
        out = batch_tile_neighbor_list(
            positions, 2.5, cell_batch, batch_ptr, format="tile"
        )
        assert len(out) == 8
        num_tiles = out[0]
        assert int(num_tiles[0]) > 0


class TestBatchTileNeighborListErrors:
    """Error path tests."""

    def test_wrong_dtype_raises(self):
        positions = jnp.zeros((32, 3), dtype=jnp.float64)
        cell_batch = jnp.eye(3, dtype=jnp.float64)[jnp.newaxis, :, :]
        batch_ptr = jnp.array([0, 32], dtype=jnp.int32)
        with pytest.raises(TypeError):
            batch_tile_neighbor_list(positions, 1.0, cell_batch, batch_ptr)

    def test_bad_cell_shape_raises(self):
        positions = jnp.zeros((32, 3), dtype=jnp.float32)
        cell_batch = jnp.eye(3, dtype=jnp.float32)  # missing system axis
        batch_ptr = jnp.array([0, 32], dtype=jnp.int32)
        with pytest.raises(ValueError, match="cell_batch"):
            batch_tile_neighbor_list(positions, 1.0, cell_batch, batch_ptr)


class TestEstimateBatchSizes:
    """Pure-Python sizing helper tests."""

    def test_aligned_two_systems(self):
        batch_ptr = jnp.array([0, 64, 192], dtype=jnp.int32)
        n_padded, ngroup, _, _, num_systems = estimate_batch_tile_neighbor_list_sizes(
            batch_ptr,
        )
        # Both systems already 32-aligned; n_padded sums to 64 + 128 = 192.
        assert n_padded == 192
        assert ngroup == 192 // TILE_GROUP_SIZE
        assert num_systems == 2

    def test_non_aligned_padding(self):
        # 33 atoms pad to 64; 80 atoms pad to 96. Total padded = 160.
        batch_ptr = jnp.array([0, 33, 113], dtype=jnp.int32)
        n_padded, _, _, _, num_systems = estimate_batch_tile_neighbor_list_sizes(
            batch_ptr,
        )
        assert n_padded == 64 + 96
        assert num_systems == 2
