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

"""Batched cluster-pair tile neighbor list — Warp kernels and launchers.

Multi-system variant of ``tile_warp``.  Pure-Warp layer — ``wp.array``
only.  Torch bindings: ``nvalchemiops.torch.neighbors.batch_tile_warp``.

Padding slots within each system carry ``sorted_atom_index == natom``
so the convert kernels' ``i_orig < natom`` filter drops them.

Scope: float32, triclinic-safe, one cell per system, arbitrary
per-system N.
"""

import warp as wp

__all__ = [
    "TILE_GROUP_SIZE",
    "build_batch_tile_neighbor_list",
    "batch_tile_to_matrix",
    "batch_tile_to_coo",
]

TILE_GROUP_SIZE = 32
TILE = wp.constant(TILE_GROUP_SIZE)


# ===========================================================================
# rank2group -- per-group bbox (SoA)
# ===========================================================================
@wp.kernel(enable_backward=False)
def _rank2group_batch_warp_kernel(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
):
    """Compute per-group axis-aligned bounding box over 32-atom clusters.

    Same per-group reduction as :func:`tile_warp._rank2group_warp_kernel`,
    operating on the concatenated multi-system layout.  Padding slots
    inside each system duplicate the system's first real atom, so the
    bounding box stays well-formed without needing a per-system filter.

    Parameters
    ----------
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (ngroup*32,), dtype=wp.float32
        Morton-sorted SoA positions (concatenated across systems).
    group_ctr_x, group_ctr_y, group_ctr_z : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: per-group bounding-box centers.
    group_ext_x, group_ext_y, group_ext_z : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: per-group bounding-box half-extents.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - group_ctr_{x,y,z}, group_ext_{x,y,z} : Filled per group

    Notes
    -----
    - Thread launch: tiled, ``block_dim = TILE_GROUP_SIZE = 32``;
      ``dim = ngroup``.

    See Also
    --------
    _group2tile_batch_warp_kernel : Consumer that uses the bboxes for
        per-system tile-pair filtering.
    """
    g = wp.tid()
    x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=g * TILE)
    y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=g * TILE)
    z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=g * TILE)

    x_min = wp.tile_extract(wp.tile_min(x_tile), 0)
    x_max = wp.tile_extract(wp.tile_max(x_tile), 0)
    y_min = wp.tile_extract(wp.tile_min(y_tile), 0)
    y_max = wp.tile_extract(wp.tile_max(y_tile), 0)
    z_min = wp.tile_extract(wp.tile_min(z_tile), 0)
    z_max = wp.tile_extract(wp.tile_max(z_tile), 0)

    lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
    lane = wp.untile(lane_tile)
    if lane == 0:
        group_ctr_x[g] = wp.float32(0.5) * (x_max + x_min)
        group_ctr_y[g] = wp.float32(0.5) * (y_max + y_min)
        group_ctr_z[g] = wp.float32(0.5) * (z_max + z_min)
        group_ext_x[g] = wp.float32(0.5) * (x_max - x_min)
        group_ext_y[g] = wp.float32(0.5) * (y_max - y_min)
        group_ext_z[g] = wp.float32(0.5) * (z_max - z_min)


@wp.func
def _wrap_triclinic(
    d: wp.vec3f,
    cell: wp.mat33f,
    inv_cell: wp.mat33f,
):
    """Wrap a displacement vector under triclinic minimum-image convention.

    Per-system variant of :func:`tile_warp._wrap_triclinic` — same
    semantics, redeclared here so the kernel module stays self-contained.

    Parameters
    ----------
    d : wp.vec3f
        Cartesian displacement to wrap.
    cell : wp.mat33f
        Cell matrix; rows are lattice vectors a, b, c.
    inv_cell : wp.mat33f
        Inverse cell matrix.

    Returns
    -------
    wp.vec3f, wp.vec3i
        Wrapped Cartesian displacement and integer shift ``(s_a, s_b, s_c)``.
    """
    inv_cell_T = wp.transpose(inv_cell)
    f = inv_cell_T * d
    s_a = -wp.int32(wp.floor(f[0] + wp.float32(0.5)))
    s_b = -wp.int32(wp.floor(f[1] + wp.float32(0.5)))
    s_c = -wp.int32(wp.floor(f[2] + wp.float32(0.5)))
    f = f + wp.vec3f(wp.float32(s_a), wp.float32(s_b), wp.float32(s_c))
    cell_T = wp.transpose(cell)
    d_new = cell_T * f
    return d_new, wp.vec3i(s_a, s_b, s_c)


@wp.func
def _bbox_valid_batch(
    col_idx: wp.int32,
    row_group: wp.int32,
    cg_system_end: wp.int32,
    rg_ctr: wp.vec3f,
    rg_ext: wp.vec3f,
    cg_ctr: wp.vec3f,
    cg_ext: wp.vec3f,
    cell: wp.mat33f,
    inv_cell: wp.mat33f,
    cutoff_sq: wp.float32,
) -> wp.int32:
    """Filter a (row, col) group pair for the batched tile builder.

    Returns 1 only if (a) ``col_idx >= row_group`` (upper-triangular
    filter), (b) ``col_idx < cg_system_end`` (column-side group lies in
    the same system as the row-side group), and (c) the minimum-image
    bounding-box distance is below ``cutoff_sq`` under the row-system's
    cell.

    Parameters
    ----------
    col_idx : wp.int32
        Column-side group index.
    row_group : wp.int32
        Row-side group index.
    cg_system_end : wp.int32
        First group index of the next system (``group_ptr[s + 1]``).
    rg_ctr, rg_ext : wp.vec3f
        Row-side bounding box (center, half-extents).
    cg_ctr, cg_ext : wp.vec3f
        Column-side bounding box.
    cell, inv_cell : wp.mat33f
        Per-system cell + inverse.
    cutoff_sq : wp.float32
        Squared cutoff distance.

    Returns
    -------
    wp.int32
        1 if the pair survives all three filters; 0 otherwise.
    """
    if col_idx < row_group:
        return 0
    if col_idx >= cg_system_end:
        return 0
    d_ctr = wp.vec3f(
        rg_ctr[0] - cg_ctr[0],
        rg_ctr[1] - cg_ctr[1],
        rg_ctr[2] - cg_ctr[2],
    )
    d_wrapped, _s = _wrap_triclinic(d_ctr, cell, inv_cell)
    dx = wp.max(wp.abs(d_wrapped[0]) - rg_ext[0] - cg_ext[0], wp.float32(0.0))
    dy = wp.max(wp.abs(d_wrapped[1]) - rg_ext[1] - cg_ext[1], wp.float32(0.0))
    dz = wp.max(wp.abs(d_wrapped[2]) - rg_ext[2] - cg_ext[2], wp.float32(0.0))
    bbox_dist_sq = dx * dx + dy * dy + dz * dz
    if bbox_dist_sq < cutoff_sq:
        return 1
    return 0


@wp.kernel(enable_backward=False)
def _group2tile_batch_warp_kernel(
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
    group_system: wp.array(dtype=wp.int32),
    group_ptr: wp.array(dtype=wp.int32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    cutoff_sq: wp.float32,
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    max_tiles: wp.int32,
):
    """Enumerate per-system (row, col) group pairs within cutoff.

    Per-system equivalent of
    :func:`tile_warp._group2tile_warp_kernel`.  For each row group, sweeps
    column groups within the SAME system (``[row_group, group_ptr[s+1])``)
    in chunks of TILE_GROUP_SIZE and applies :func:`_bbox_valid_batch`
    under that system's cell.  Surviving pairs are appended to
    ``tile_row_group`` / ``tile_col_group`` / ``tile_system`` via a
    tile-wide inclusive scan + single atomic increment on
    ``num_tiles[0]``.

    Parameters
    ----------
    group_ctr_x, group_ctr_y, group_ctr_z : wp.array, shape (ngroup,), dtype=wp.float32
        Per-group bbox centers.
    group_ext_x, group_ext_y, group_ext_z : wp.array, shape (ngroup,), dtype=wp.float32
        Per-group bbox half-extents.
    group_system : wp.array, shape (ngroup,), dtype=wp.int32
        ``group_system[g]`` is the system index of group ``g``.
    group_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        CSR-style pointer: groups of system ``s`` live in
        ``[group_ptr[s], group_ptr[s+1])``.
    cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
        Per-system cell matrices.
    inv_cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
        Per-system inverse cell matrices.
    cutoff_sq : wp.float32
        Squared cutoff distance.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        OUTPUT (atomic): incremented by the count of emitted pairs.
        Caller must zero before launch.
    tile_row_group : wp.array, shape (max_tiles,), dtype=wp.int32
        OUTPUT: row-side group index of each emitted pair.
    tile_col_group : wp.array, shape (max_tiles,), dtype=wp.int32
        OUTPUT: column-side group index of each emitted pair.
    tile_system : wp.array, shape (max_tiles,), dtype=wp.int32
        OUTPUT: system index of each emitted pair.
    max_tiles : wp.int32
        Capacity of the three tile arrays; over-emission is silently
        dropped.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - num_tiles[0] : Incremented by the number of valid pairs
        - tile_row_group, tile_col_group, tile_system : Filled per emit

    Notes
    -----
    - Thread launch: tiled, ``block_dim = TILE_GROUP_SIZE = 32``;
      ``dim = ngroup``.
    - No cross-system pairs are emitted (the column sweep is bounded by
      ``group_ptr[s+1]``).

    See Also
    --------
    _bbox_valid_batch : Per-pair filter.
    _tile_to_matrix_batch_kernel : Consumer producing per-atom matrix output.
    _tile_to_coo_batch_kernel : Consumer producing flat COO output.
    """
    row_group = wp.tid()
    s = group_system[row_group]
    cg_end = group_ptr[s + 1]
    cell = cell_batch[s]
    inv_cell = inv_cell_batch[s]

    lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
    lane = wp.untile(lane_tile)

    rg_ctr = wp.vec3f(
        group_ctr_x[row_group],
        group_ctr_y[row_group],
        group_ctr_z[row_group],
    )
    rg_ext = wp.vec3f(
        group_ext_x[row_group],
        group_ext_y[row_group],
        group_ext_z[row_group],
    )

    col_start_aligned = (row_group // TILE) * TILE
    for col_start in range(col_start_aligned, cg_end, TILE):
        cg_ctr_x_tile = wp.tile_load(group_ctr_x, shape=TILE, offset=col_start)
        cg_ctr_y_tile = wp.tile_load(group_ctr_y, shape=TILE, offset=col_start)
        cg_ctr_z_tile = wp.tile_load(group_ctr_z, shape=TILE, offset=col_start)
        cg_ext_x_tile = wp.tile_load(group_ext_x, shape=TILE, offset=col_start)
        cg_ext_y_tile = wp.tile_load(group_ext_y, shape=TILE, offset=col_start)
        cg_ext_z_tile = wp.tile_load(group_ext_z, shape=TILE, offset=col_start)

        cg = col_start + lane
        cg_ctr = wp.vec3f(
            wp.untile(cg_ctr_x_tile),
            wp.untile(cg_ctr_y_tile),
            wp.untile(cg_ctr_z_tile),
        )
        cg_ext = wp.vec3f(
            wp.untile(cg_ext_x_tile),
            wp.untile(cg_ext_y_tile),
            wp.untile(cg_ext_z_tile),
        )

        valid = _bbox_valid_batch(
            cg,
            row_group,
            cg_end,
            rg_ctr,
            rg_ext,
            cg_ctr,
            cg_ext,
            cell,
            inv_cell,
            cutoff_sq,
        )
        # Per-emit ``wp.atomic_add`` on the global counter.  Measured
        # equivalent on GB10 to the scan + single-atomic-per-block
        # pattern used by the single-system equivalent — the kernel
        # is bbox-compute-bound, not atomic-bound — so the simpler
        # form wins on readability.
        if valid == 1:
            slot = wp.atomic_add(num_tiles, 0, 1)
            if slot < max_tiles:
                tile_row_group[slot] = row_group
                tile_col_group[slot] = cg
                tile_system[slot] = s


# ===========================================================================
# Format conversion: tile pair list → neighbor_matrix (triclinic + padding-aware).
#
# Same i-broadcast + always-write-shifts optimizations as the single-system
# tile_to_matrix.  Padding slots (``sorted_atom_index[k] == natom``) are filtered by
# the ``i_orig < natom`` / ``j_orig < natom`` guard.
# ===========================================================================
@wp.kernel(enable_backward=False)
def _tile_to_matrix_batch_kernel(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    sorted_atom_index: wp.array(dtype=wp.int32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    cutoff_sq: wp.float32,
    natom: wp.int32,
    max_neighbors: wp.int32,
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array3d(dtype=wp.int32),
):
    """Convert the batched tile pair list into per-atom neighbor_matrix rows.

    Per-system variant of :func:`tile_warp._tile_to_matrix_kernel`.  Each
    block handles one emitted tile pair and uses
    ``tile_system[tile]`` to pick the per-system cell.  Atom indices in
    the neighbor matrix are global (concatenated across systems); no
    cross-system pairs are produced because the upstream
    :func:`_group2tile_batch_warp_kernel` already restricts emission to
    same-system tiles.

    Shifts at active slots are written unconditionally; tail slots are
    left untouched.  Pair with
    :func:`tile_warp.fill_neighbor_matrix_tail` to write the sentinel
    into unused columns.

    Parameters
    ----------
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Morton-sorted SoA positions (concatenated across systems).
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Sort permutation mapping sorted slot to global original atom id.
        Padding slots carry ``sorted_atom_index == natom``.
    cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
        Per-system cell matrices.
    inv_cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
        Per-system inverse cell matrices.
    cutoff_sq : wp.float32
        Squared cutoff distance.
    natom : wp.int32
        Total real atom count across all systems; pairs involving
        ``sorted_atom_index == natom`` are dropped.
    max_neighbors : wp.int32
        Column capacity of ``neighbor_matrix``.
    tile_row_group : wp.array, shape (num_tiles,), dtype=wp.int32
    tile_col_group : wp.array, shape (num_tiles,), dtype=wp.int32
    tile_system : wp.array, shape (num_tiles,), dtype=wp.int32
        System index of each tile pair (drives per-system cell pick).
    neighbor_matrix : wp.array2d, shape (natom, max_neighbors), dtype=wp.int32
        OUTPUT.
    num_neighbors : wp.array, shape (natom,), dtype=wp.int32
        OUTPUT (atomic).  Caller must zero before launch.
    neighbor_matrix_shifts : wp.array3d, shape (natom, max_neighbors, 3), dtype=wp.int32
        OUTPUT.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - neighbor_matrix : Filled at active slots
        - neighbor_matrix_shifts : Filled at active slots
        - num_neighbors : Updated with per-atom counts

    Notes
    -----
    - Thread launch: tiled, ``block_dim = TILE_GROUP_SIZE = 32``;
      ``dim = num_tiles``.

    See Also
    --------
    batch_tile_to_matrix : Python launcher for this kernel.
    tile_warp.fill_neighbor_matrix_tail : Post-conv tail fill.
    _tile_to_coo_batch_kernel : Alternate consumer producing flat COO.
    """
    tile = wp.tid()
    lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
    lane = wp.untile(lane_tile)

    row_group = tile_row_group[tile]
    col_group = tile_col_group[tile]
    s = tile_system[tile]

    j_sorted = col_group * TILE + lane
    j_orig_t = sorted_atom_index[j_sorted]
    pj_x = sorted_pos_x[j_sorted]
    pj_y = sorted_pos_y[j_sorted]
    pj_z = sorted_pos_z[j_sorted]

    # i-broadcast: preload the 32 i-side reads once per tile.
    pi_x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=row_group * TILE)
    pi_y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=row_group * TILE)
    pi_z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=row_group * TILE)
    i_orig_tile = wp.tile_load(sorted_atom_index, shape=TILE, offset=row_group * TILE)

    cell = cell_batch[s]
    inv_cell = inv_cell_batch[s]

    for i_local in range(TILE_GROUP_SIZE):
        i_sorted = row_group * TILE + i_local
        i_orig = wp.tile_extract(i_orig_tile, i_local)
        pi_x = wp.tile_extract(pi_x_tile, i_local)
        pi_y = wp.tile_extract(pi_y_tile, i_local)
        pi_z = wp.tile_extract(pi_z_tile, i_local)

        d = wp.vec3f(pj_x - pi_x, pj_y - pi_y, pj_z - pi_z)
        d_wrapped, s_shift = _wrap_triclinic(d, cell, inv_cell)
        dist_sq = (
            d_wrapped[0] * d_wrapped[0]
            + d_wrapped[1] * d_wrapped[1]
            + d_wrapped[2] * d_wrapped[2]
        )

        valid = wp.int32(0)
        if (
            i_sorted < j_sorted
            and i_orig < natom
            and j_orig_t < natom
            and dist_sq < cutoff_sq
        ):
            valid = wp.int32(1)

        valid_tile = wp.tile(valid)
        scan_tile = wp.tile_scan_inclusive(valid_tile)
        row_count = wp.tile_extract(scan_tile, TILE - 1)

        if row_count > 0:
            base = wp.int32(0)
            if lane == 0:
                base = wp.atomic_add(num_neighbors, i_orig, row_count)
            base_tile = wp.tile_from_thread(
                shape=TILE,
                value=base,
                thread_idx=0,
                storage="shared",
            )
            base_b = wp.untile(base_tile)
            scan_v = wp.untile(scan_tile)
            write_col = base_b + scan_v - 1

            if valid == 1 and write_col < max_neighbors:
                neighbor_matrix[i_orig, write_col] = j_orig_t
                # Always-write shifts (caller can skip nms.zero_()).
                neighbor_matrix_shifts[i_orig, write_col, 0] = s_shift[0]
                neighbor_matrix_shifts[i_orig, write_col, 1] = s_shift[1]
                neighbor_matrix_shifts[i_orig, write_col, 2] = s_shift[2]


@wp.kernel(enable_backward=False)
def _tile_to_coo_batch_kernel(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    sorted_atom_index: wp.array(dtype=wp.int32),
    cell_batch: wp.array(dtype=wp.mat33f),
    inv_cell_batch: wp.array(dtype=wp.mat33f),
    cutoff_sq: wp.float32,
    natom: wp.int32,
    max_pairs: wp.int32,
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    tile_system: wp.array(dtype=wp.int32),
    pair_counter: wp.array(dtype=wp.int32),
    coo_list: wp.array2d(dtype=wp.int32),
    coo_shifts: wp.array2d(dtype=wp.int32),
):
    """Convert the batched tile pair list into a flat COO pair list.

    Per-system equivalent of :func:`tile_warp._tile_to_coo_kernel`.  Each
    block handles one emitted tile and uses ``tile_system[tile]`` to pick
    the per-system cell.  Emitted ``(i, j, shift)`` rows use global atom
    IDs; no cross-system pairs are produced.

    Parameters
    ----------
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Morton-sorted SoA positions.
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Sort permutation; padding slots carry ``sorted_atom_index == natom``.
    cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
    inv_cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
    cutoff_sq : wp.float32
        Squared cutoff distance.
    natom : wp.int32
        Total real atom count across all systems.
    max_pairs : wp.int32
        Row capacity of ``coo_list`` / ``coo_shifts``.
    tile_row_group, tile_col_group, tile_system : wp.array, shape (num_tiles,), dtype=wp.int32
        Tile pair list from :func:`_group2tile_batch_warp_kernel`.
    pair_counter : wp.array, shape (1,), dtype=wp.int32
        OUTPUT (atomic).  Caller must zero before launch.
    coo_list : wp.array2d, shape (max_pairs, 2), dtype=wp.int32
        OUTPUT.
    coo_shifts : wp.array2d, shape (max_pairs, 3), dtype=wp.int32
        OUTPUT.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - pair_counter[0] : Incremented by the count of emitted pairs
        - coo_list : Filled with global ``(i, j)`` indices
        - coo_shifts : Filled with integer shift vectors

    Notes
    -----
    - Thread launch: tiled, ``block_dim = TILE_GROUP_SIZE = 32``;
      ``dim = num_tiles``.

    See Also
    --------
    batch_tile_to_coo : Python launcher for this kernel.
    _tile_to_matrix_batch_kernel : Alternate consumer producing matrix output.
    """
    tile = wp.tid()
    lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
    lane = wp.untile(lane_tile)

    row_group = tile_row_group[tile]
    col_group = tile_col_group[tile]
    s = tile_system[tile]

    j_sorted = col_group * TILE + lane
    j_orig_t = sorted_atom_index[j_sorted]
    pj_x = sorted_pos_x[j_sorted]
    pj_y = sorted_pos_y[j_sorted]
    pj_z = sorted_pos_z[j_sorted]

    pi_x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=row_group * TILE)
    pi_y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=row_group * TILE)
    pi_z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=row_group * TILE)
    i_orig_tile = wp.tile_load(sorted_atom_index, shape=TILE, offset=row_group * TILE)

    cell = cell_batch[s]
    inv_cell = inv_cell_batch[s]

    for i_local in range(TILE_GROUP_SIZE):
        i_sorted = row_group * TILE + i_local
        i_orig = wp.tile_extract(i_orig_tile, i_local)
        pi_x = wp.tile_extract(pi_x_tile, i_local)
        pi_y = wp.tile_extract(pi_y_tile, i_local)
        pi_z = wp.tile_extract(pi_z_tile, i_local)

        d = wp.vec3f(pj_x - pi_x, pj_y - pi_y, pj_z - pi_z)
        d_wrapped, s_shift = _wrap_triclinic(d, cell, inv_cell)
        dist_sq = (
            d_wrapped[0] * d_wrapped[0]
            + d_wrapped[1] * d_wrapped[1]
            + d_wrapped[2] * d_wrapped[2]
        )

        valid = wp.int32(0)
        if (
            i_sorted < j_sorted
            and i_orig < natom
            and j_orig_t < natom
            and dist_sq < cutoff_sq
        ):
            valid = wp.int32(1)

        valid_tile = wp.tile(valid)
        scan_tile = wp.tile_scan_inclusive(valid_tile)
        i_count = wp.tile_extract(scan_tile, TILE - 1)

        if i_count > 0:
            base = wp.int32(0)
            if lane == 0:
                base = wp.atomic_add(pair_counter, 0, i_count)
            base_tile = wp.tile_from_thread(
                shape=TILE,
                value=base,
                thread_idx=0,
                storage="shared",
            )
            base_b = wp.untile(base_tile)
            scan_v = wp.untile(scan_tile)
            write_pos = base_b + scan_v - 1

            if valid == 1 and write_pos < max_pairs:
                coo_list[write_pos, 0] = i_orig
                coo_list[write_pos, 1] = j_orig_t
                coo_shifts[write_pos, 0] = s_shift[0]
                coo_shifts[write_pos, 1] = s_shift[1]
                coo_shifts[write_pos, 2] = s_shift[2]


# ===========================================================================
# Public warp launchers
# ===========================================================================
def build_batch_tile_neighbor_list(
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    group_system: wp.array,
    group_ptr: wp.array,
    cell_batch: wp.array,
    inv_cell_batch: wp.array,
    cutoff: float,
    group_ctr_x: wp.array,
    group_ctr_y: wp.array,
    group_ctr_z: wp.array,
    group_ext_x: wp.array,
    group_ext_y: wp.array,
    group_ext_z: wp.array,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    tile_system: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for batched per-system tile-pair enumeration.

    Runs :func:`_rank2group_batch_warp_kernel` to reduce each 32-atom
    cluster to a bounding box, then :func:`_group2tile_batch_warp_kernel`
    to enumerate per-system (row, col) group pairs surviving a
    bounding-box-vs-bounding-box cutoff filter under each system's cell.

    Callers handle the per-system Morton sort + padding + SoA gather
    upstream (see :mod:`nvalchemiops.torch.neighbors.batch_tile_warp` /
    :mod:`nvalchemiops.jax.neighbors.batch_tile_warp`), plus the
    ``inv_cell_batch`` inverse.

    Parameters
    ----------
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Per-system Morton-sorted SoA positions (concatenated across systems).
    group_system : wp.array, shape (ngroup,), dtype=wp.int32
        System index for each 32-atom group.
    group_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
        CSR-style range of groups per system.
    cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
        Per-system cell matrices.
    inv_cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
        Per-system inverse cell matrices.
    cutoff : float
        Cutoff distance for the bounding-box filter.
    group_ctr_x, group_ctr_y, group_ctr_z : wp.array, shape (ngroup_padded,), dtype=wp.float32
        OUTPUT: per-group bbox centers (overwritten).
    group_ext_x, group_ext_y, group_ext_z : wp.array, shape (ngroup_padded,), dtype=wp.float32
        OUTPUT: per-group bbox half-extents.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        OUTPUT (atomic).  Caller must zero before launch.
    tile_row_group, tile_col_group, tile_system : wp.array, shape (max_tiles,), dtype=wp.int32
        OUTPUT: emitted pair list.
    wp_dtype : type
        Warp dtype.  Only ``wp.float32`` is supported.
    device : str
        Warp device string (e.g. ``"cuda:0"``).

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - group_ctr_{x,y,z}, group_ext_{x,y,z} : Per-group bboxes
        - num_tiles[0] : Count of emitted tile pairs
        - tile_row_group, tile_col_group, tile_system : Emitted pair list

    Raises
    ------
    NotImplementedError
        If ``wp_dtype`` is not ``wp.float32``.

    Notes
    -----
    - This is a low-level warp interface.  Framework bindings should call
      it through :mod:`nvalchemiops.torch.neighbors.batch_tile_warp` /
      :mod:`nvalchemiops.jax.neighbors.batch_tile_warp`.
    - No cross-system pairs are emitted.

    See Also
    --------
    _rank2group_batch_warp_kernel : First kernel launched.
    _group2tile_batch_warp_kernel : Second kernel; performs enumeration.
    batch_tile_to_matrix, batch_tile_to_coo : Downstream consumers.
    nvalchemiops.neighbors.tile_warp.build_tile_neighbor_list : Single-system variant.
    """
    if wp_dtype is not wp.float32:
        raise NotImplementedError(
            "tile_batch_warp kernels currently support float32 only",
        )
    ngroup = sorted_pos_x.shape[0] // TILE_GROUP_SIZE
    max_tiles = tile_row_group.shape[0]

    wp.launch_tiled(
        kernel=_rank2group_batch_warp_kernel,
        dim=[ngroup],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )

    wp.launch_tiled(
        kernel=_group2tile_batch_warp_kernel,
        dim=[ngroup],
        inputs=[
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            group_system,
            group_ptr,
            cell_batch,
            inv_cell_batch,
            wp.float32(cutoff * cutoff),
            num_tiles,
            tile_row_group,
            tile_col_group,
            tile_system,
            int(max_tiles),
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )


def batch_tile_to_matrix(
    sorted_atom_index: wp.array,
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    cell_batch: wp.array,
    inv_cell_batch: wp.array,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    tile_system: wp.array,
    cutoff: float,
    natom: int,
    n_tiles: int,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    neighbor_matrix_shifts: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for batched tile-pair → neighbor_matrix conversion.

    Launches :func:`_tile_to_matrix_batch_kernel` over ``n_tiles`` blocks
    to convert the per-system tile pair list into a global per-atom
    neighbor_matrix.  Atom indices in the matrix are global
    (concatenated across systems); no cross-system pairs are produced
    (the upstream tile builder restricts emission to same-system tiles).

    Parameters
    ----------
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Sort permutation; padding slots carry ``sorted_atom_index == natom``.
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Per-system Morton-sorted positions.
    cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
    inv_cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        Atomic counter populated by :func:`build_batch_tile_neighbor_list`.
    tile_row_group, tile_col_group, tile_system : wp.array, shape (max_tiles,), dtype=wp.int32
        Tile pair list from :func:`build_batch_tile_neighbor_list`.
    cutoff : float
        Cutoff distance.
    natom : int
        Total real atom count across all systems.
    n_tiles : int
        Number of emitted tile pairs.  The launch is skipped when
        ``n_tiles <= 0``.
    neighbor_matrix : wp.array, shape (natom, max_neighbors), dtype=wp.int32
        OUTPUT.
    num_neighbors : wp.array, shape (natom,), dtype=wp.int32
        OUTPUT (atomic).  Caller must zero before launch.
    neighbor_matrix_shifts : wp.array, shape (natom, max_neighbors, 3), dtype=wp.int32
        OUTPUT.
    wp_dtype : type
        Warp dtype.  Only ``wp.float32`` is supported.
    device : str
        Warp device string (e.g. ``"cuda:0"``).

    Returns
    -------
    None
        Modifies outputs in-place; see :func:`_tile_to_matrix_batch_kernel`.

    Raises
    ------
    NotImplementedError
        If ``wp_dtype`` is not ``wp.float32``.

    Notes
    -----
    - This is a low-level warp interface.  Framework bindings should call
      it through :mod:`nvalchemiops.torch.neighbors.batch_tile_warp` /
      :mod:`nvalchemiops.jax.neighbors.batch_tile_warp`.

    See Also
    --------
    _tile_to_matrix_batch_kernel : Kernel that performs the computation.
    build_batch_tile_neighbor_list : Upstream tile-pair builder.
    batch_tile_to_coo : Alternate launcher producing flat COO output.
    """
    if wp_dtype is not wp.float32:
        raise NotImplementedError(
            "tile_batch_warp kernels currently support float32 only",
        )
    if n_tiles <= 0:
        return
    max_neighbors = neighbor_matrix.shape[1]
    wp.launch_tiled(
        kernel=_tile_to_matrix_batch_kernel,
        dim=[n_tiles],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            sorted_atom_index,
            cell_batch,
            inv_cell_batch,
            wp.float32(cutoff * cutoff),
            int(natom),
            int(max_neighbors),
            tile_row_group,
            tile_col_group,
            tile_system,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )


def batch_tile_to_coo(
    sorted_atom_index: wp.array,
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    cell_batch: wp.array,
    inv_cell_batch: wp.array,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    tile_system: wp.array,
    cutoff: float,
    natom: int,
    n_tiles: int,
    max_pairs: int,
    pair_counter: wp.array,
    coo_list: wp.array,
    coo_shifts: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for batched tile-pair → flat COO conversion.

    Launches :func:`_tile_to_coo_batch_kernel` over ``n_tiles`` blocks to
    convert the per-system tile pair list into a flat ``(i, j, shift)``
    COO list using global atom IDs (concatenated across systems).  Both
    halves of each cross-tile pair are emitted (full-fill output).  No
    cross-system pairs are produced.

    Parameters
    ----------
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Sort permutation; padding slots carry ``sorted_atom_index == natom``.
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Per-system Morton-sorted positions.
    cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
    inv_cell_batch : wp.array, shape (num_systems,), dtype=wp.mat33f
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        Atomic counter populated by :func:`build_batch_tile_neighbor_list`.
    tile_row_group, tile_col_group, tile_system : wp.array, shape (max_tiles,), dtype=wp.int32
    cutoff : float
        Cutoff distance.
    natom : int
        Total real atom count across all systems.
    n_tiles : int
        Number of emitted tile pairs.  The launch is skipped when
        ``n_tiles <= 0``.
    max_pairs : int
        Capacity of ``coo_list`` / ``coo_shifts``.
    pair_counter : wp.array, shape (1,), dtype=wp.int32
        OUTPUT (atomic).  Caller must zero before launch.
    coo_list : wp.array, shape (max_pairs, 2), dtype=wp.int32
        OUTPUT.
    coo_shifts : wp.array, shape (max_pairs, 3), dtype=wp.int32
        OUTPUT.
    wp_dtype : type
        Warp dtype.  Only ``wp.float32`` is supported.
    device : str
        Warp device string (e.g. ``"cuda:0"``).

    Returns
    -------
    None
        Modifies outputs in-place; see :func:`_tile_to_coo_batch_kernel`.

    Raises
    ------
    NotImplementedError
        If ``wp_dtype`` is not ``wp.float32``.

    Notes
    -----
    - This is a low-level warp interface.  Framework bindings should call
      it through :mod:`nvalchemiops.torch.neighbors.batch_tile_warp` /
      :mod:`nvalchemiops.jax.neighbors.batch_tile_warp`.

    See Also
    --------
    _tile_to_coo_batch_kernel : Kernel that performs the computation.
    build_batch_tile_neighbor_list : Upstream tile-pair builder.
    batch_tile_to_matrix : Alternate launcher producing matrix output.
    """
    if wp_dtype is not wp.float32:
        raise NotImplementedError(
            "tile_batch_warp kernels currently support float32 only",
        )
    if n_tiles <= 0:
        return
    wp.launch_tiled(
        kernel=_tile_to_coo_batch_kernel,
        dim=[n_tiles],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            sorted_atom_index,
            cell_batch,
            inv_cell_batch,
            wp.float32(cutoff * cutoff),
            int(natom),
            int(max_pairs),
            tile_row_group,
            tile_col_group,
            tile_system,
            pair_counter,
            coo_list,
            coo_shifts,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )
