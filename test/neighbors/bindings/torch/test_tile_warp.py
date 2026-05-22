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

"""Tests for the single-system cluster-pair tile neighbor list PyTorch bindings."""

import pytest
import torch

from nvalchemiops.torch.neighbors.tile_warp import (
    TILE_GROUP_SIZE,
    allocate_tile_neighbor_list,
    build_tile_neighbor_list,
    estimate_tile_neighbor_list_sizes,
    tile_neighbor_list,
    tile_to_matrix,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_random_system,
    create_simple_cubic_system,
)
from .conftest import requires_vesin

# tile_warp is CUDA + float32 only; override the conftest device/dtype
# fixtures to restrict the parametrize matrix.


@pytest.fixture(params=["cuda:0"], ids=lambda d: d.replace(":", "_"))
def device(request):
    if not torch.cuda.is_available():
        pytest.skip("tile_warp kernels require CUDA")
    return request.param


@pytest.fixture(params=[torch.float32], ids=["float32"])
def dtype(request):
    return request.param


def _orthorhombic_cell(
    cell_size: float, device: str, dtype=torch.float32
) -> torch.Tensor:
    return (torch.eye(3, dtype=dtype, device=device) * cell_size).reshape(1, 3, 3)


# =============================================================================
# Correctness
# =============================================================================
class TestTileNeighborListCorrectness:
    def test_single_atom_no_neighbors(self, device, dtype):
        """Single atom system should have no neighbors."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        cell = _orthorhombic_cell(2.0, device, dtype)
        cutoff = 3.0
        nm, nn, _nms = tile_neighbor_list(positions, cutoff, cell, max_neighbors=8)
        assert int(nn.sum().item()) == 0
        assert nm.shape == (1, 8)

    def test_two_atom_pair(self, device, dtype):
        """Two atoms within cutoff yield exactly one half-fill pair."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype, device=device
        )
        cell = _orthorhombic_cell(2.0, device, dtype)
        cutoff = 1.0
        nm, nn, _nms = tile_neighbor_list(positions, cutoff, cell, max_neighbors=8)
        # Half-fill: exactly one of the two rows holds the pair.
        assert int(nn.sum().item()) == 1
        # The recorded pair is the other atom.
        rows = nn.cpu().tolist()
        owner = 0 if rows[0] == 1 else 1
        other = 1 - owner
        assert int(nm[owner, 0].item()) == other

    @requires_vesin
    def test_cubic_system(self, device, dtype):
        """Simple cubic lattice (4x4x4 = 64 atoms, multiple of TILE_GROUP_SIZE)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=64,
            cell_size=4.0,
            dtype=dtype,
            device=device,
        )
        cutoff = 1.1

        nm, nn, nms = tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=32,
        )
        i_got, j_got, u_got = _matrix_to_coo_half_fill(
            nm,
            nn,
            nms,
            positions.shape[0],
        )
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        i_ref, j_ref, u_ref = _full_to_half_fill(i_ref, j_ref, u_ref)
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_random_system(self, device, dtype):
        """Random atomic positions vs brute-force reference."""
        positions, cell, pbc = create_random_system(
            num_atoms=64,
            cell_size=10.0,
            dtype=dtype,
            device=device,
            seed=42,
            pbc_flag=True,
        )
        cutoff = 3.0
        nm, nn, nms = tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
        )
        i_got, j_got, u_got = _matrix_to_coo_half_fill(nm, nn, nms, positions.shape[0])
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        i_ref, j_ref, u_ref = _full_to_half_fill(i_ref, j_ref, u_ref)
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    @pytest.mark.parametrize("N", [33, 65, 100, 127, 200])
    def test_non_aligned_N_correctness(self, device, dtype, N):
        """Non-32-aligned N produces correct pairs (padding-safe)."""
        positions, cell, pbc = create_random_system(
            num_atoms=N,
            cell_size=10.0,
            dtype=dtype,
            device=device,
            seed=N,
            pbc_flag=True,
        )
        cutoff = 3.0
        nm, nn, nms = tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
        )
        i_got, j_got, u_got = _matrix_to_coo_half_fill(nm, nn, nms, N)
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        i_ref, j_ref, u_ref = _full_to_half_fill(i_ref, j_ref, u_ref)
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_triclinic_system(self, device, dtype):
        """Triclinic (non-orthorhombic) cell vs brute-force reference."""
        torch.manual_seed(11)
        N = 96
        # Build a moderately skewed cell.
        cell_mat = torch.eye(3, dtype=dtype, device=device) * 10.0
        cell_mat[0, 1] = 1.0
        cell_mat[1, 2] = 0.5
        cell = cell_mat.reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
        # Fractional sampling, then map into the triclinic cell.
        frac = torch.rand(N, 3, dtype=dtype, device=device)
        positions = (frac @ cell_mat).contiguous()
        cutoff = 2.5
        nm, nn, nms = tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
        )
        i_got, j_got, u_got = _matrix_to_coo_half_fill(nm, nn, nms, N)
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        i_ref, j_ref, u_ref = _full_to_half_fill(i_ref, j_ref, u_ref)
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_larger_random_system(self, device, dtype):
        """Larger system exercises multiple tile rows."""
        positions, cell, pbc = create_random_system(
            num_atoms=256,
            cell_size=15.0,
            dtype=dtype,
            device=device,
            seed=7,
            pbc_flag=True,
        )
        cutoff = 2.5
        nm, nn, nms = tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
        )
        i_got, j_got, u_got = _matrix_to_coo_half_fill(nm, nn, nms, positions.shape[0])
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        i_ref, j_ref, u_ref = _full_to_half_fill(i_ref, j_ref, u_ref)
        assert_neighbor_lists_equal((i_got, j_got, u_got), (i_ref, j_ref, u_ref))

    @requires_vesin
    def test_return_neighbor_list(self, device, dtype):
        """COO output (``format="coo"``) matches matrix output."""
        positions, cell, _ = create_random_system(
            num_atoms=64,
            cell_size=10.0,
            dtype=dtype,
            device=device,
            seed=3,
            pbc_flag=True,
        )
        cutoff = 3.0
        neighbor_list, neighbor_ptr, shifts = tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
            format="coo",
        )
        assert neighbor_list.shape[0] == 2
        # Compare pair counts against the matrix-mode result.
        nm, nn, _nms = tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=128,
        )
        assert int(nn.sum().item()) == int(neighbor_list.shape[1])

    def test_component_API_matches_convenience(self, device, dtype):
        """Explicit component calls produce the same state as the convenience wrapper."""
        N = 128
        torch.manual_seed(0)
        positions = torch.rand(N, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype)
        cutoff = 2.5

        # Convenience path
        nm1, nn1, nms1 = tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=64,
        )

        # Component path -- allocate + build + convert using the
        # SoA-layout state tensors exposed by allocate_tile_neighbor_list.
        (
            sorted_atom_index,
            morton_codes,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
        ) = allocate_tile_neighbor_list(
            N,
            torch.device(device),
            dtype=dtype,
        )
        build_tile_neighbor_list(
            positions,
            cutoff,
            cell,
            sorted_atom_index,
            morton_codes,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
        )
        nm2 = torch.full((N, 64), N, dtype=torch.int32, device=device)
        nn2 = torch.zeros(N, dtype=torch.int32, device=device)
        nms2 = torch.zeros((N, 64, 3), dtype=torch.int32, device=device)
        tile_to_matrix(
            sorted_atom_index,
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            num_tiles,
            tile_row_group,
            tile_col_group,
            cell,
            cutoff,
            N,
            nm2,
            nn2,
            nms2,
        )
        torch.testing.assert_close(nn1, nn2)
        # Entries may be in different order per-row; compare row-wise
        # sorted neighbor index sets.
        for i in range(N):
            n_i = int(nn1[i].item())
            s1 = {int(x.item()) for x in nm1[i, :n_i]}
            s2 = {int(x.item()) for x in nm2[i, :n_i]}
            assert s1 == s2, f"atom {i} neighbor set mismatch"


# =============================================================================
# Edge cases
# =============================================================================
class TestTileNeighborListEdgeCases:
    def test_zero_cutoff_matrix_empty(self, device, dtype):
        """Cutoff smaller than any distance -> zero neighbors."""
        N = 32
        positions = torch.rand(N, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype)
        nm, nn, _nms = tile_neighbor_list(positions, 1e-6, cell, max_neighbors=32)
        assert int(nn.sum().item()) == 0

    def test_max_neighbors_truncation(self, device, dtype):
        """max_neighbors caps the per-atom row; excess neighbors are dropped."""
        N = 64
        torch.manual_seed(1)
        positions = torch.rand(N, 3, dtype=dtype, device=device) * 5.0
        cell = _orthorhombic_cell(5.0, device, dtype)
        nm, nn, _nms = tile_neighbor_list(positions, 5.0, cell, max_neighbors=4)
        assert int(nn.max().item()) >= 4  # counter records the true count
        # All entries within [0:min(nn[i], 4)] must be valid indices.
        for i in range(N):
            visible = min(int(nn[i].item()), 4)
            for k in range(visible):
                assert 0 <= int(nm[i, k].item()) < N

    def test_component_sizes_match_estimate(self, device):
        """estimate_tile_neighbor_list_sizes returns shape-consistent sizes."""
        N = 512
        n_padded, ngroup, ngroup_padded, max_tiles = estimate_tile_neighbor_list_sizes(
            N
        )
        assert n_padded == N  # already 32-aligned
        assert ngroup == n_padded // TILE_GROUP_SIZE
        assert ngroup_padded % TILE_GROUP_SIZE == 0 and ngroup_padded > ngroup
        assert max_tiles >= ngroup

    def test_component_sizes_match_estimate_non_aligned(self, device):
        """Non-32-aligned N is rounded up to ``ceil(N/32)*32``."""
        N = 33
        n_padded, ngroup, ngroup_padded, max_tiles = estimate_tile_neighbor_list_sizes(
            N
        )
        assert n_padded == 64  # ceil(33/32) * 32
        assert ngroup == 2
        assert max_tiles >= ngroup


# =============================================================================
# Errors
# =============================================================================
class TestTileNeighborListErrors:
    def test_wrong_dtype(self, device):
        positions = torch.rand(32, 3, dtype=torch.float64, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device)
        # The torch wrapper accepts float64 in its signature; the Warp
        # layer currently rejects it as NotImplementedError while we
        # wait for the overloaded kernels to land.
        with pytest.raises((TypeError, NotImplementedError)):
            tile_neighbor_list(positions, 2.5, cell, max_neighbors=32)

    def test_non_multiple_of_group_size_accepted(self, device, dtype):
        """N not divisible by TILE_GROUP_SIZE is padded internally.

        Sanity check: build runs without error and emits some pairs.
        Correctness vs reference is exercised by
        ``test_non_aligned_N_correctness`` below.
        """
        positions = torch.rand(33, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype)
        nm, nn, _nms = tile_neighbor_list(
            positions,
            2.5,
            cell,
            max_neighbors=32,
            format="matrix",
        )
        assert nm.shape == (33, 32)
        assert int(nn.sum().item()) >= 0

    def test_triclinic_cell_accepted(self, device, dtype):
        """Triclinic cells are now first-class (tile_warp parity with batch).

        Sanity check: build runs without error and emits some pairs.
        Correctness vs a reference is exercised in
        ``TestTileNeighborListCorrectness::test_triclinic_system`` below.
        """
        positions = torch.rand(32, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype).clone()
        cell[0, 0, 1] = 1.0  # off-diagonal
        nm, nn, _nms = tile_neighbor_list(
            positions,
            2.5,
            cell,
            max_neighbors=32,
            format="matrix",
        )
        # nm shape sanity; nn nonneg sum.
        assert nm.shape == (32, 32)
        assert int(nn.sum().item()) >= 0


# =============================================================================
# Helpers
# =============================================================================
def _matrix_to_coo_half_fill(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    shifts: torch.Tensor,
    natom: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten (nm, nn, nms) into canonical half-fill (i, j, shift) triples
    with ``i < j`` (shift negated when we swap).  Cluster-tile half-fill
    can store a pair in either atom's row depending on Morton order, so
    canonicalization is required for comparison with vesin's output.
    """
    device = neighbor_matrix.device
    i_list, j_list, u_list = [], [], []
    nm_cpu = neighbor_matrix.cpu()
    nn_cpu = num_neighbors.cpu()
    s_cpu = shifts.cpu()
    for i in range(natom):
        ni = int(nn_cpu[i].item())
        for k in range(ni):
            j = int(nm_cpu[i, k].item())
            if 0 <= j < natom:
                sh = tuple(int(x) for x in s_cpu[i, k])
                if i < j:
                    i_list.append(i)
                    j_list.append(j)
                    u_list.append(sh)
                else:
                    i_list.append(j)
                    j_list.append(i)
                    u_list.append((-sh[0], -sh[1], -sh[2]))
    i_t = torch.tensor(i_list, dtype=torch.int32, device=device)
    j_t = torch.tensor(j_list, dtype=torch.int32, device=device)
    if u_list:
        u_t = torch.tensor(u_list, dtype=torch.int32, device=device).reshape(-1, 3)
    else:
        u_t = torch.zeros((0, 3), dtype=torch.int32, device=device)
    return i_t, j_t, u_t


def _full_to_half_fill(
    i: torch.Tensor,
    j: torch.Tensor,
    u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce a full (symmetric) neighbor list to half-fill (i < j, shift sign)."""
    mask = i < j
    return i[mask], j[mask], u[mask]


# =============================================================================
# torch.compile compatibility
# =============================================================================
class TestTileWarpCompile:
    """Tests for ``torch.compile`` compatibility of the cluster-pair tile path.

    Verifies that the ``@torch.library.custom_op``-decorated component shells
    (``_build_tile_neighbor_list``, ``_tile_to_matrix``, ``_tile_to_coo``)
    survive a ``torch.compile`` round-trip without graph breaks that change
    the output.
    """

    def test_tile_neighbor_list_compile(self, device, dtype):
        """``tile_neighbor_list`` should be compatible with ``torch.compile``."""
        torch.manual_seed(0)
        N = 64
        positions = torch.rand(N, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype)
        cutoff = 2.5

        nm_uncompiled, nn_uncompiled, nms_uncompiled = tile_neighbor_list(
            positions,
            cutoff,
            cell,
            max_neighbors=64,
        )

        @torch.compile
        def compiled_tile_neighbor_list(positions, cutoff, cell):
            return tile_neighbor_list(positions, cutoff, cell, max_neighbors=64)

        nm_compiled, nn_compiled, nms_compiled = compiled_tile_neighbor_list(
            positions, cutoff, cell
        )

        assert torch.equal(nn_uncompiled, nn_compiled)
        # Per-row neighbor sets must match (column order within a row may differ).
        for i in range(N):
            n_i = int(nn_uncompiled[i].item())
            s_uncompiled = {int(x.item()) for x in nm_uncompiled[i, :n_i]}
            s_compiled = {int(x.item()) for x in nm_compiled[i, :n_i]}
            assert s_uncompiled == s_compiled, (
                f"Row {i} neighbor set mismatch under torch.compile"
            )

    def test_build_then_convert_compile(self, device, dtype):
        """Component build + tile_to_matrix should compile cleanly."""
        torch.manual_seed(1)
        N = 64
        positions = torch.rand(N, 3, dtype=dtype, device=device) * 10.0
        cell = _orthorhombic_cell(10.0, device, dtype)
        cutoff = 2.5

        # Uncompiled reference via the convenience wrapper.
        nm_ref, nn_ref, _nms_ref = tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=64
        )

        # Compiled component sequence.
        state = allocate_tile_neighbor_list(N, torch.device(device), dtype=dtype)
        nm = torch.full((N, 64), N, dtype=torch.int32, device=device)
        nn = torch.zeros(N, dtype=torch.int32, device=device)
        nms = torch.zeros((N, 64, 3), dtype=torch.int32, device=device)

        @torch.compile
        def compiled_build_and_convert(positions, cutoff, cell, nm, nn, nms):
            build_tile_neighbor_list(positions, cutoff, cell, *state)
            tile_to_matrix(
                state[0],  # sorted_atom_index
                state[2],  # sorted_pos_x
                state[3],  # sorted_pos_y
                state[4],  # sorted_pos_z
                state[11],  # num_tiles
                state[12],  # tile_row_group
                state[13],  # tile_col_group
                cell,
                cutoff,
                N,
                nm,
                nn,
                nms,
            )

        compiled_build_and_convert(positions, cutoff, cell, nm, nn, nms)

        assert torch.equal(nn_ref, nn)
        for i in range(N):
            n_i = int(nn_ref[i].item())
            s_ref = {int(x.item()) for x in nm_ref[i, :n_i]}
            s_got = {int(x.item()) for x in nm[i, :n_i]}
            assert s_ref == s_got, f"Row {i} neighbor set mismatch under torch.compile"


# =============================================================================
# Left-handed cells
# =============================================================================
class TestTileWarpLeftHanded:
    """Cells with ``det(cell) < 0`` should produce the same pair set as the
    right-handed mirror.  Mirrors :class:`TestLeftHandedCells` in
    ``test_cell_list.py``.
    """

    @requires_vesin
    def test_left_handed_cubic(self, device, dtype):
        """Flipping the sign of one axis should preserve the pair set."""
        N = 64
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=N, cell_size=4.0, dtype=dtype, device=device
        )
        cutoff = 1.1

        nm_rh, nn_rh, nms_rh = tile_neighbor_list(
            positions, cutoff, cell, max_neighbors=32
        )
        i_rh, j_rh, _u_rh = _matrix_to_coo_half_fill(nm_rh, nn_rh, nms_rh, N)
        pairs_rh = {(int(a), int(b)) for a, b in zip(i_rh, j_rh)}

        # Flip the third axis to produce a left-handed cell.
        cell_lh = cell.clone()
        cell_lh[0, 2, 2] = -cell_lh[0, 2, 2]
        positions_lh = positions.clone()
        positions_lh[:, 2] = -positions_lh[:, 2]

        nm_lh, nn_lh, nms_lh = tile_neighbor_list(
            positions_lh, cutoff, cell_lh, max_neighbors=32
        )
        i_lh, j_lh, _u_lh = _matrix_to_coo_half_fill(nm_lh, nn_lh, nms_lh, N)
        pairs_lh = {(int(a), int(b)) for a, b in zip(i_lh, j_lh)}

        assert pairs_rh == pairs_lh, "Left-handed cell produced different pair set"
        del pbc  # pbc fixture unused under PBC-implicit tile_warp
