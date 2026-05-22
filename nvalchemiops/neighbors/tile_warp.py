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

"""Single-system cluster-pair tile neighbor list — Warp kernels and launchers.

GROMACS-style NxM cluster pair list: Morton-sorted atoms grouped into
32-atom clusters (one warp wide).  Pure-Warp layer — ``wp.array`` only,
no torch.  Torch bindings: ``nvalchemiops.torch.neighbors.tile_warp``.

Scope: single system, triclinic-safe, float32, ``N % 32 == 0``.
"""

import warp as wp

__all__ = [
    "TILE_GROUP_SIZE",
    "build_tile_neighbor_list",
    "compute_morton",
    "permute_gather_soa",
    "tile_sort_pairs_warp",
    "tile_to_coo",
    "tile_to_matrix",
]

TILE_GROUP_SIZE = 32
TILE = wp.constant(TILE_GROUP_SIZE)


@wp.func
def _wrap_triclinic(
    d: wp.vec3f,
    cell: wp.mat33f,
    inv_cell: wp.mat33f,
):
    """Wrap a displacement vector under triclinic minimum-image convention.

    Converts ``d`` to fractional coordinates via ``inv_cell``, rounds to
    the nearest integer in each axis to get the shift, then converts the
    wrapped fractional position back to Cartesian via ``cell``.  Returns
    both the wrapped Cartesian displacement and the integer shift vector
    (so callers can record which periodic image was selected).

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


@wp.kernel(enable_backward=False, module="morton_build")
def _compute_morton_kernel(
    positions: wp.array(dtype=wp.vec3f),
    inv_cell: wp.array(dtype=wp.mat33f),
    natom: wp.int32,
    morton_codes: wp.array(dtype=wp.int32),
    sorted_atom_index: wp.array(dtype=wp.int32),
    num_neighbors: wp.array(dtype=wp.int32),
    num_tiles: wp.array(dtype=wp.int32),
):
    """Compute per-atom 30-bit Morton code and initialize tile scratch.

    Maps each atom's fractional coordinates to a 30-bit Morton interleave
    (10 bits per axis) and writes it to ``morton_codes``.  Pads beyond
    ``natom`` with a sentinel code so padding slots sort to the end and
    are dropped by the downstream convert kernels' ``i < natom`` filter.
    Also initializes ``sorted_atom_index`` to the identity permutation
    consumed by the subsequent radix sort, zeros ``num_neighbors``, and
    zeros ``num_tiles[0]``.

    Parameters
    ----------
    positions : wp.array, shape (n_padded,), dtype=wp.vec3f
        Atomic coordinates in the padded layout.
    inv_cell : wp.mat33f
        Inverse cell matrix.  ``transpose(inv_cell) @ pos`` yields the
        fractional coordinates used for bucketing.
    natom : wp.int32
        Real atom count (pre-padding).
    morton_codes : wp.array, shape (n_padded,), dtype=wp.int32
        OUTPUT: 30-bit Morton code per atom; padding slots get
        ``0x40000000`` (one bit above the 30-bit max) so they sort to
        the end.
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        OUTPUT: identity permutation ``sorted_atom_index[i] = i``,
        consumed by the radix sort that follows this kernel.
    num_neighbors : wp.array, shape (natom,), dtype=wp.int32
        OUTPUT: zeroed for downstream atomic accumulation.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        OUTPUT: zeroed by thread 0 (atomic counter for the tile build).

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - morton_codes : Filled with per-atom Morton codes
        - sorted_atom_index : Initialized to identity permutation
        - num_neighbors : Zeroed
        - num_tiles : Element 0 zeroed

    Notes
    -----
    - Thread launch: 1D (n_padded).
    - Bucket resolution is 1024 per axis (clamped); positions wrap to
      ``[0, 1)`` in fractional coords before bucketing.

    See Also
    --------
    compute_morton : Python launcher for this kernel.
    _permute_gather_soa_kernel : Permutes positions into SoA after sort.
    """
    i = wp.tid()
    if i == 0:
        num_tiles[0] = 0
    if i >= natom:
        morton_codes[i] = wp.int32(0x40000000)
        sorted_atom_index[i] = i
        return
    pos = positions[i]
    inv_cell_mat = inv_cell[0]
    inv_cell_T = wp.transpose(inv_cell_mat)
    f = inv_cell_T * pos
    fx = f[0] - wp.floor(f[0])
    fy = f[1] - wp.floor(f[1])
    fz = f[2] - wp.floor(f[2])
    ix = wp.int32(fx * 1024.0)
    iy = wp.int32(fy * 1024.0)
    iz = wp.int32(fz * 1024.0)
    if ix < 0:
        ix = 0
    if ix > 1023:
        ix = 1023
    if iy < 0:
        iy = 0
    if iy > 1023:
        iy = 1023
    if iz < 0:
        iz = 0
    if iz > 1023:
        iz = 1023
    # 10-bit → 30-bit Morton spread (Morton interleave bitmasks).
    ix = ix & 0x000003FF
    ix = (ix | (ix << 16)) & 0x030000FF
    ix = (ix | (ix << 8)) & 0x0300F00F
    ix = (ix | (ix << 4)) & 0x030C30C3
    ix = (ix | (ix << 2)) & 0x09249249
    iy = iy & 0x000003FF
    iy = (iy | (iy << 16)) & 0x030000FF
    iy = (iy | (iy << 8)) & 0x0300F00F
    iy = (iy | (iy << 4)) & 0x030C30C3
    iy = (iy | (iy << 2)) & 0x09249249
    iz = iz & 0x000003FF
    iz = (iz | (iz << 16)) & 0x030000FF
    iz = (iz | (iz << 8)) & 0x0300F00F
    iz = (iz | (iz << 4)) & 0x030C30C3
    iz = (iz | (iz << 2)) & 0x09249249
    morton_codes[i] = (iz << 2) | (iy << 1) | ix
    sorted_atom_index[i] = i
    num_neighbors[i] = 0


def compute_morton(
    positions: wp.array,
    inv_cell: wp.array,
    natom: int,
    morton_codes: wp.array,
    sorted_atom_index: wp.array,
    num_neighbors: wp.array,
    num_tiles: wp.array,
    device: str,
) -> None:
    """Core warp launcher for the fused Morton + identity-init kernel.

    Launches :func:`_compute_morton_kernel` over the padded layout
    ``n_padded = morton_codes.shape[0]`` so padding slots receive a
    sentinel max Morton code that sorts to the end.

    Parameters
    ----------
    positions : wp.array, shape (n_padded,), dtype=wp.vec3f
        Padded position array.
    inv_cell : wp.mat33f
        Inverse cell matrix used to compute fractional coordinates.
    natom : int
        Real atom count (pre-padding).
    morton_codes : wp.array, shape (n_padded,), dtype=wp.int32
        OUTPUT: per-atom Morton codes.
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        OUTPUT: identity permutation consumed by the radix sort.
    num_neighbors : wp.array, shape (natom,), dtype=wp.int32
        OUTPUT: zeroed.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        OUTPUT: element 0 zeroed.
    device : str
        Warp device string (e.g. ``"cuda:0"``).

    Returns
    -------
    None
        Modifies outputs in-place; see :func:`_compute_morton_kernel`.

    Notes
    -----
    - This is a low-level warp interface.  Framework bindings should call
      it through :mod:`nvalchemiops.torch.neighbors.tile_warp` /
      :mod:`nvalchemiops.jax.neighbors.tile_warp`.
    - Output arrays must be pre-allocated by the caller at padded size.

    See Also
    --------
    _compute_morton_kernel : Kernel that performs the computation.
    permute_gather_soa : Followed by SoA gather after the sort.
    """
    n_padded = int(morton_codes.shape[0])
    wp.launch(
        kernel=_compute_morton_kernel,
        dim=n_padded,
        inputs=[
            positions,
            inv_cell,
            int(natom),
            morton_codes,
            sorted_atom_index,
            num_neighbors,
            num_tiles,
        ],
        device=device,
    )


@wp.kernel(enable_backward=False, module="morton_build")
def _permute_gather_soa_kernel(
    positions: wp.array(dtype=wp.vec3f),
    sorted_atom_index: wp.array(dtype=wp.int32),
    natom: wp.int32,
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
):
    """Permute AoS positions into SoA arrays in Morton-sorted order.

    Reads each AoS ``vec3f`` from ``positions[sorted_atom_index[i]]`` and
    scatters its components into the three SoA arrays.  The SoA layout
    lets downstream kernels (``_rank2group_warp_kernel``,
    ``_tile_to_matrix_kernel``) coalesce loads of 32 contiguous atoms via
    a single ``wp.tile_load`` per axis.

    Parameters
    ----------
    positions : wp.array, shape (>=natom,), dtype=wp.vec3f
        AoS positions in original (pre-sort) order.
    sorted_atom_index : wp.array, shape (>=natom,), dtype=wp.int32
        Permutation produced by the Morton sort:
        ``sorted_atom_index[i]`` is the original index at sorted slot ``i``.
    natom : wp.int32
        Real atom count; threads with ``i >= natom`` (padding slots)
        leave their outputs untouched.
    sorted_pos_x : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT: Morton-sorted x positions.
    sorted_pos_y : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT: Morton-sorted y positions.
    sorted_pos_z : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT: Morton-sorted z positions.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - sorted_pos_x, sorted_pos_y, sorted_pos_z : Filled in Morton order

    See Also
    --------
    _compute_morton_kernel : Produces the Morton codes that drive the sort.
    permute_gather_soa : Python launcher for this kernel.
    _rank2group_warp_kernel : Consumes the SoA-sorted positions.
    """
    i = wp.tid()
    if i >= natom:
        return
    src = sorted_atom_index[i]
    pos = positions[src]
    sorted_pos_x[i] = pos[0]
    sorted_pos_y[i] = pos[1]
    sorted_pos_z[i] = pos[2]


def permute_gather_soa(
    positions: wp.array,
    sorted_atom_index: wp.array,
    natom: int,
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    device: str,
) -> None:
    """Core warp launcher for AoS-to-SoA permute-and-gather.

    Launches :func:`_permute_gather_soa_kernel` over ``natom`` threads to
    write Morton-sorted SoA position arrays from a pre-sorted permutation.

    Parameters
    ----------
    positions : wp.array, shape (>=natom,), dtype=wp.vec3f
        AoS positions in original order.
    sorted_atom_index : wp.array, shape (>=natom,), dtype=wp.int32
        Permutation from the Morton radix sort.
    natom : int
        Real atom count.
    sorted_pos_x : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT.
    sorted_pos_y : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT.
    sorted_pos_z : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT.
    device : str
        Warp device string (e.g. ``"cuda:0"``).

    Returns
    -------
    None
        Modifies outputs in-place; see :func:`_permute_gather_soa_kernel`.

    Notes
    -----
    - This is a low-level warp interface.  Framework bindings should call
      it through :mod:`nvalchemiops.torch.neighbors.tile_warp` /
      :mod:`nvalchemiops.jax.neighbors.tile_warp`.
    - Output arrays must be pre-allocated by the caller.

    See Also
    --------
    _permute_gather_soa_kernel : Kernel that performs the computation.
    compute_morton : Initializes the permutation array.
    """
    wp.launch(
        kernel=_permute_gather_soa_kernel,
        dim=int(natom),
        inputs=[
            positions,
            sorted_atom_index,
            int(natom),
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
        ],
        device=device,
    )


# Single-block bitonic sort via ``wp.tile_sort`` (graph-safe).
# Specializations at N=1024 / 2048; above that, callers use CUB radix sort.
TILE_SORT_N_1024 = wp.constant(1024)
TILE_SORT_N_2048 = wp.constant(2048)


@wp.kernel(enable_backward=False, module="warp_tile_sort_1024")
def _tile_sort_kernel_1024(
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
):
    """Single-block in-place bitonic sort of 1024 (key, value) int32 pairs.

    Loads ``keys[:1024]`` and ``values[:1024]`` into shared memory,
    runs ``wp.tile_sort`` to sort the keys (carrying values along), then
    stores back.  Launch with ``dim=[1]``, ``block_dim=512``.

    Parameters
    ----------
    keys : wp.array, shape (>=1024,), dtype=wp.int32
        OUTPUT (in-place): sorted ascending in the first 1024 slots.
    values : wp.array, shape (>=1024,), dtype=wp.int32
        OUTPUT (in-place): permuted to match ``keys``.

    Returns
    -------
    None
        ``keys`` and ``values`` are modified in-place.

    See Also
    --------
    tile_sort_pairs_warp : Dispatch wrapper covering both specializations.
    """
    keys_tile = wp.tile_load(keys, shape=TILE_SORT_N_1024, storage="shared")
    values_tile = wp.tile_load(values, shape=TILE_SORT_N_1024, storage="shared")
    wp.tile_sort(keys_tile, values_tile)
    wp.tile_store(keys, keys_tile)
    wp.tile_store(values, values_tile)


@wp.kernel(enable_backward=False, module="warp_tile_sort_2048")
def _tile_sort_kernel_2048(
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
):
    """Single-block in-place bitonic sort of 2048 (key, value) int32 pairs.

    Same contract as :func:`_tile_sort_kernel_1024` but for N=2048.

    Parameters
    ----------
    keys : wp.array, shape (>=2048,), dtype=wp.int32
        OUTPUT (in-place): sorted ascending in the first 2048 slots.
    values : wp.array, shape (>=2048,), dtype=wp.int32
        OUTPUT (in-place): permuted to match ``keys``.

    Returns
    -------
    None
        ``keys`` and ``values`` are modified in-place.

    See Also
    --------
    _tile_sort_kernel_1024 : Smaller specialization.
    tile_sort_pairs_warp : Dispatch wrapper.
    """
    keys_tile = wp.tile_load(keys, shape=TILE_SORT_N_2048, storage="shared")
    values_tile = wp.tile_load(values, shape=TILE_SORT_N_2048, storage="shared")
    wp.tile_sort(keys_tile, values_tile)
    wp.tile_store(keys, keys_tile)
    wp.tile_store(values, values_tile)


# (kernel, block_dim) — block_dim minimizes per-sort wall time.
_TILE_SORT_SPECIALIZATIONS = {
    1024: (_tile_sort_kernel_1024, 512),
    2048: (_tile_sort_kernel_2048, 512),
}


def tile_sort_pairs_warp(
    keys: wp.array,
    values: wp.array,
    natom: int,
    device: str,
) -> bool:
    """Core warp launcher for in-place key-value sort via ``wp.tile_sort``.

    Dispatches to the size-specialized single-block bitonic-sort kernel
    when ``natom`` matches one of the supported specializations (1024 or
    2048).  Graph-capture safe.  Above ``natom = 2048`` the bitonic-sort
    runtime explodes — callers should detect the ``False`` return and
    fall back to ``wp.utils.radix_sort_pairs`` (CUB; also graph-safe).

    Parameters
    ----------
    keys : wp.array, shape (>=natom,), dtype=wp.int32
        Sort keys; sorted ascending in-place.
    values : wp.array, shape (>=natom,), dtype=wp.int32
        Values carried along with the keys; permuted in-place.
    natom : int
        Number of leading elements to sort.  Must equal a supported
        specialization for the launch to occur.
    device : str
        Warp device string (e.g. ``"cuda:0"``).

    Returns
    -------
    bool
        ``True`` if a specialization kernel was launched; ``False`` if
        ``natom`` was unsupported (no work done, caller must fall back).

    Notes
    -----
    - This is a low-level warp interface.  Framework bindings should call
      it through :mod:`nvalchemiops.torch.neighbors.tile_warp` /
      :mod:`nvalchemiops.jax.neighbors.tile_warp`.

    See Also
    --------
    _tile_sort_kernel_1024 : N=1024 specialization.
    _tile_sort_kernel_2048 : N=2048 specialization.
    """
    spec = _TILE_SORT_SPECIALIZATIONS.get(int(natom))
    if spec is None:
        return False
    kernel, block_dim = spec
    wp.launch_tiled(
        kernel=kernel,
        dim=[1],
        inputs=[keys, values],
        block_dim=block_dim,
        device=device,
    )
    return True


@wp.kernel(enable_backward=False)
def _rank2group_warp_kernel(
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

    Each block reduces 32 consecutive Morton-sorted atoms into a single
    Cartesian axis-aligned bounding box, stored as (center, half-extent)
    in SoA.  Used by :func:`_group2tile_warp_kernel` as the candidate
    filter for tile-pair enumeration.

    Parameters
    ----------
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (ngroup*32,), dtype=wp.float32
        Morton-sorted SoA positions.
    group_ctr_x, group_ctr_y, group_ctr_z : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: per-group bounding-box centers (axis-wise mean of min/max).
    group_ext_x, group_ext_y, group_ext_z : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: per-group bounding-box half-extents.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - group_ctr_{x,y,z} : Filled with per-group bbox centers
        - group_ext_{x,y,z} : Filled with per-group bbox half-extents

    Notes
    -----
    - Thread launch: tiled, ``block_dim = TILE_GROUP_SIZE = 32``.
    - One block per 32-atom group.

    See Also
    --------
    _group2tile_warp_kernel : Consumer that uses the bboxes for pair filtering.
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
def _bbox_valid(
    col_idx: wp.int32,
    row_group: wp.int32,
    rg_ctr: wp.vec3f,
    rg_ext: wp.vec3f,
    cg_ctr: wp.vec3f,
    cg_ext: wp.vec3f,
    cell: wp.mat33f,
    inv_cell: wp.mat33f,
    cutoff_sq: wp.float32,
    ngroup: wp.int32,
) -> wp.int32:
    """Test whether a (row, col) group pair can contain an in-cutoff atom pair.

    Returns 1 only if (a) ``col_idx >= row_group`` (upper-triangular
    filter for half-fill), (b) ``col_idx < ngroup`` (skip padding groups),
    and (c) the minimum-image distance between the two bounding boxes
    is below ``cutoff_sq``.

    Parameters
    ----------
    col_idx : wp.int32
        Column-side group index under consideration.
    row_group : wp.int32
        Row-side group index of the current block.
    rg_ctr, rg_ext : wp.vec3f
        Row-side bounding box (center, half-extents).
    cg_ctr, cg_ext : wp.vec3f
        Column-side bounding box (center, half-extents).
    cell : wp.mat33f
    inv_cell : wp.mat33f
    cutoff_sq : wp.float32
        Squared cutoff distance.
    ngroup : wp.int32
        Number of real (non-padding) groups.

    Returns
    -------
    wp.int32
        1 if the tile pair survives all three filters; 0 otherwise.
    """
    if col_idx < row_group:
        return 0
    if col_idx >= ngroup:
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
def _group2tile_warp_kernel(
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    cutoff_sq: wp.float32,
    ngroup: wp.int32,
    num_tiles: wp.array(dtype=wp.int32),
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    max_tiles: wp.int32,
):
    """Enumerate (row, col) group pairs within cutoff and append to the tile list.

    For each row group, sweeps the column groups ``[row_group, ngroup)``
    in chunks of TILE_GROUP_SIZE and invokes :func:`_bbox_valid` to filter
    by bounding-box distance under PBC.  Surviving (row, col) pairs are
    appended atomically to ``tile_row_group`` / ``tile_col_group``; the
    write position is computed from a tile-wide inclusive scan to amortize
    atomic-add traffic on ``num_tiles[0]``.

    Parameters
    ----------
    group_ctr_x, group_ctr_y, group_ctr_z : wp.array, shape (ngroup,), dtype=wp.float32
        Per-group bbox centers (from :func:`_rank2group_warp_kernel`).
    group_ext_x, group_ext_y, group_ext_z : wp.array, shape (ngroup,), dtype=wp.float32
        Per-group bbox half-extents.
    cell : wp.mat33f
    inv_cell : wp.mat33f
    cutoff_sq : wp.float32
        Squared cutoff distance.
    ngroup : wp.int32
        Number of real groups (padding groups are filtered by :func:`_bbox_valid`).
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        OUTPUT (atomic): incremented by the count of valid tile pairs
        emitted by this kernel.
    tile_row_group : wp.array, shape (max_tiles,), dtype=wp.int32
        OUTPUT: row-side group index of each emitted tile pair.
    tile_col_group : wp.array, shape (max_tiles,), dtype=wp.int32
        OUTPUT: column-side group index of each emitted tile pair.
    max_tiles : wp.int32
        Capacity of ``tile_row_group`` / ``tile_col_group``; writes
        beyond this are silently dropped.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - num_tiles[0] : Incremented by the number of valid pairs
        - tile_row_group : Filled with row-side group indices
        - tile_col_group : Filled with column-side group indices

    Notes
    -----
    - Thread launch: tiled, ``block_dim = TILE_GROUP_SIZE = 32``.
    - Only upper-triangular pairs (``col_group >= row_group``) are
      emitted; consumers either iterate the full 32x32 candidate set per
      tile (full-fill) or apply an atom-level filter (half-fill).

    See Also
    --------
    _bbox_valid : Per-pair filter used by this kernel.
    _tile_to_matrix_kernel : Consumes the emitted tile pair list.
    _tile_to_coo_kernel : Alternate consumer producing flat COO.
    """
    row_group = wp.tid()
    lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
    lane = wp.untile(lane_tile)

    cell_mat = cell[0]
    inv_cell_mat = inv_cell[0]

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
    for col_start in range(col_start_aligned, ngroup, TILE):
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

        valid = _bbox_valid(
            cg,
            row_group,
            rg_ctr,
            rg_ext,
            cg_ctr,
            cg_ext,
            cell_mat,
            inv_cell_mat,
            cutoff_sq,
            ngroup,
        )

        valid_tile = wp.tile(valid)
        scan_tile = wp.tile_scan_inclusive(valid_tile)
        tile_total = wp.tile_extract(scan_tile, TILE - 1)

        if tile_total > 0:
            base = wp.int32(0)
            if lane == 0:
                base = wp.atomic_add(num_tiles, 0, tile_total)
            base_tile = wp.tile_from_thread(
                shape=TILE,
                value=base,
                thread_idx=0,
                storage="shared",
            )
            base_b = wp.untile(base_tile)
            scan_v = wp.untile(scan_tile)
            slot = base_b + scan_v - 1

            if valid == 1 and slot < max_tiles:
                tile_row_group[slot] = row_group
                tile_col_group[slot] = cg


@wp.kernel(enable_backward=False)
def _tile_to_matrix_kernel(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    sorted_atom_index: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    cutoff_sq: wp.float32,
    natom: wp.int32,
    max_neighbors: wp.int32,
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    neighbor_matrix: wp.array2d(dtype=wp.int32),
    num_neighbors: wp.array(dtype=wp.int32),
    neighbor_matrix_shifts: wp.array3d(dtype=wp.int32),
):
    """Convert the tile pair list into per-atom neighbor_matrix rows.

    Each block handles one emitted (row_group, col_group) tile.  The row
    side is preloaded into shared memory (atom IDs + 3 position axes)
    once per tile and broadcast across the inner loop over the 32 column
    atoms.  Per-pair distance is computed under triclinic minimum-image
    PBC; pairs within ``cutoff_sq`` are appended to ``neighbor_matrix``
    rows of both atoms via atomic increments on ``num_neighbors``.

    Shifts at active slots are written unconditionally; tail slots
    (``column >= num_neighbors[atom]``) are left untouched.  Pair with
    :func:`_fill_neighbor_matrix_tail_kernel` to write the sentinel into
    the unused columns of ``neighbor_matrix`` and skip the per-step
    ``neighbor_matrix.fill_`` / ``neighbor_matrix_shifts.zero_`` prefill.

    Parameters
    ----------
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Morton-sorted SoA positions.
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Permutation: ``sorted_atom_index[i]`` is the original atom id at
        sorted slot ``i``.
    cell : wp.mat33f
    inv_cell : wp.mat33f
    cutoff_sq : wp.float32
        Squared cutoff distance.
    natom : wp.int32
        Real atom count; pairs involving sorted slots ``>= natom`` (padding)
        are dropped.
    max_neighbors : wp.int32
        Column capacity of ``neighbor_matrix`` / ``neighbor_matrix_shifts``.
        Writes past this position are silently dropped.
    tile_row_group : wp.array, shape (num_tiles,), dtype=wp.int32
        Row-side group index of each tile pair.
    tile_col_group : wp.array, shape (num_tiles,), dtype=wp.int32
        Column-side group index of each tile pair.
    neighbor_matrix : wp.array2d, shape (natom, max_neighbors), dtype=wp.int32
        OUTPUT: per-atom neighbor indices at columns
        ``[0, num_neighbors[i])``.
    num_neighbors : wp.array, shape (natom,), dtype=wp.int32
        OUTPUT (atomic): per-atom neighbor counts.  Caller must zero
        before launch.
    neighbor_matrix_shifts : wp.array3d, shape (natom, max_neighbors, 3), dtype=wp.int32
        OUTPUT: integer shift vector for each emitted pair.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - neighbor_matrix : Filled with neighbor atom indices at active slots
        - neighbor_matrix_shifts : Filled with shift vectors at active slots
        - num_neighbors : Updated with per-atom counts

    Notes
    -----
    - Thread launch: tiled, ``block_dim = TILE_GROUP_SIZE = 32``;
      ``dim = num_tiles``.
    - Both halves of each cross-tile pair are emitted (full-fill atom
      output) even though the tile pair list itself is half-fill at the
      group level.
    - Tail slots in ``neighbor_matrix`` are not zeroed; use
      :func:`_fill_neighbor_matrix_tail_kernel` afterwards if a sentinel
      is required.

    See Also
    --------
    tile_to_matrix : Python launcher for this kernel.
    fill_neighbor_matrix_tail : Post-conv tail fill that pairs with this kernel.
    _tile_to_coo_kernel : Alternate consumer producing flat COO output.
    """
    tile = wp.tid()
    lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
    lane = wp.untile(lane_tile)

    cell_mat = cell[0]
    inv_cell_mat = inv_cell[0]

    row_group = tile_row_group[tile]
    col_group = tile_col_group[tile]

    j_sorted = col_group * TILE + lane
    j_orig_t = sorted_atom_index[j_sorted]
    pj_x = sorted_pos_x[j_sorted]
    pj_y = sorted_pos_y[j_sorted]
    pj_z = sorted_pos_z[j_sorted]

    # Preload i-side data once per tile; broadcast in the inner loop.
    pi_x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=row_group * TILE)
    pi_y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=row_group * TILE)
    pi_z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=row_group * TILE)
    i_orig_tile = wp.tile_load(sorted_atom_index, shape=TILE, offset=row_group * TILE)

    for i_local in range(TILE_GROUP_SIZE):
        i_sorted = row_group * TILE + i_local
        i_orig = wp.tile_extract(i_orig_tile, i_local)
        pi_x = wp.tile_extract(pi_x_tile, i_local)
        pi_y = wp.tile_extract(pi_y_tile, i_local)
        pi_z = wp.tile_extract(pi_z_tile, i_local)

        d = wp.vec3f(pj_x - pi_x, pj_y - pi_y, pj_z - pi_z)
        d_wrapped, s_shift = _wrap_triclinic(d, cell_mat, inv_cell_mat)
        dist_sq = (
            d_wrapped[0] * d_wrapped[0]
            + d_wrapped[1] * d_wrapped[1]
            + d_wrapped[2] * d_wrapped[2]
        )

        valid = wp.int32(0)
        if (
            i_sorted < j_sorted
            and i_sorted < natom
            and j_sorted < natom
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
                neighbor_matrix_shifts[i_orig, write_col, 0] = s_shift[0]
                neighbor_matrix_shifts[i_orig, write_col, 1] = s_shift[1]
                neighbor_matrix_shifts[i_orig, write_col, 2] = s_shift[2]


@wp.kernel(enable_backward=False)
def _tile_to_coo_kernel(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    sorted_atom_index: wp.array(dtype=wp.int32),
    cell: wp.array(dtype=wp.mat33f),
    inv_cell: wp.array(dtype=wp.mat33f),
    cutoff_sq: wp.float32,
    natom: wp.int32,
    max_pairs: wp.int32,
    tile_row_group: wp.array(dtype=wp.int32),
    tile_col_group: wp.array(dtype=wp.int32),
    pair_counter: wp.array(dtype=wp.int32),
    coo_list: wp.array2d(dtype=wp.int32),
    coo_shifts: wp.array2d(dtype=wp.int32),
):
    """Convert the tile pair list into a flat COO pair list.

    Same launch shape and i-broadcast pattern as
    :func:`_tile_to_matrix_kernel`, but emits each in-cutoff pair as a
    single ``(i, j, shift)`` row in ``coo_list`` / ``coo_shifts`` rather
    than into a per-atom matrix.  Each block coalesces emissions from its
    tile via an intra-tile scan + a single atomic increment on
    ``pair_counter[0]``.

    Parameters
    ----------
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Morton-sorted SoA positions.
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Sort permutation mapping sorted slot to original atom id.
    cell : wp.mat33f
    inv_cell : wp.mat33f
    cutoff_sq : wp.float32
        Squared cutoff distance.
    natom : wp.int32
        Real atom count; pairs involving sorted slots ``>= natom`` (padding)
        are dropped.
    max_pairs : wp.int32
        Row capacity of ``coo_list`` / ``coo_shifts``; writes past this
        position are silently dropped.
    tile_row_group : wp.array, shape (num_tiles,), dtype=wp.int32
        Row-side group index of each tile pair.
    tile_col_group : wp.array, shape (num_tiles,), dtype=wp.int32
        Column-side group index of each tile pair.
    pair_counter : wp.array, shape (1,), dtype=wp.int32
        OUTPUT (atomic): incremented by the number of emitted pairs.
        Caller must zero before launch.
    coo_list : wp.array2d, shape (max_pairs, 2), dtype=wp.int32
        OUTPUT: ``(i_atom, j_atom)`` pairs at rows ``[0, pair_counter[0])``.
    coo_shifts : wp.array2d, shape (max_pairs, 3), dtype=wp.int32
        OUTPUT: integer shift vector per pair.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - pair_counter[0] : Incremented to the total emitted-pair count
        - coo_list : Filled with ``(i, j)`` pairs in emission order
        - coo_shifts : Filled with shift vectors aligned to ``coo_list``

    Notes
    -----
    - Thread launch: tiled, ``block_dim = TILE_GROUP_SIZE = 32``;
      ``dim = num_tiles``.
    - Both halves of each cross-tile pair are emitted (full-fill output).
    - ``coo_list`` rows past ``pair_counter[0]`` are not zeroed; consumers
      gate on the counter.

    See Also
    --------
    tile_to_coo : Python launcher for this kernel.
    _tile_to_matrix_kernel : Alternate consumer producing per-atom matrix rows.
    """
    tile = wp.tid()
    lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
    lane = wp.untile(lane_tile)

    cell_mat = cell[0]
    inv_cell_mat = inv_cell[0]

    row_group = tile_row_group[tile]
    col_group = tile_col_group[tile]

    j_sorted = col_group * TILE + lane
    j_orig_t = sorted_atom_index[j_sorted]
    pj_x = sorted_pos_x[j_sorted]
    pj_y = sorted_pos_y[j_sorted]
    pj_z = sorted_pos_z[j_sorted]

    pi_x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=row_group * TILE)
    pi_y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=row_group * TILE)
    pi_z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=row_group * TILE)
    i_orig_tile = wp.tile_load(sorted_atom_index, shape=TILE, offset=row_group * TILE)

    for i_local in range(TILE_GROUP_SIZE):
        i_sorted = row_group * TILE + i_local
        i_orig = wp.tile_extract(i_orig_tile, i_local)
        pi_x = wp.tile_extract(pi_x_tile, i_local)
        pi_y = wp.tile_extract(pi_y_tile, i_local)
        pi_z = wp.tile_extract(pi_z_tile, i_local)

        d = wp.vec3f(pj_x - pi_x, pj_y - pi_y, pj_z - pi_z)
        d_wrapped, s_shift = _wrap_triclinic(d, cell_mat, inv_cell_mat)
        dist_sq = (
            d_wrapped[0] * d_wrapped[0]
            + d_wrapped[1] * d_wrapped[1]
            + d_wrapped[2] * d_wrapped[2]
        )

        valid = wp.int32(0)
        if (
            i_sorted < j_sorted
            and i_sorted < natom
            and j_sorted < natom
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
# Public launchers
# ===========================================================================
def build_tile_neighbor_list(
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
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
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for tile-pair enumeration on pre-sorted positions.

    Runs two kernels back-to-back: :func:`_rank2group_warp_kernel` reduces
    each 32-atom cluster to a bounding box, then
    :func:`_group2tile_warp_kernel` enumerates (row, col) group pairs that
    survive a bounding-box-vs-bounding-box cutoff filter under minimum-image
    PBC and appends them to the tile pair list.

    Callers must run the Morton sort upstream
    (:func:`compute_morton` + a sort primitive + :func:`permute_gather_soa`)
    so that ``sorted_pos_{x,y,z}`` are in Morton-sorted order before this
    call.

    Parameters
    ----------
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Morton-sorted SoA positions; ``n_padded`` must be a multiple of
        ``TILE_GROUP_SIZE``.
    cell : wp.mat33f
        Cell matrix; rows are lattice vectors a, b, c.
    inv_cell : wp.mat33f
        Inverse cell matrix.
    cutoff : float
        Cutoff distance for the bounding-box filter.
    group_ctr_x, group_ctr_y, group_ctr_z : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: per-group bbox centers (overwritten).
    group_ext_x, group_ext_y, group_ext_z : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: per-group bbox half-extents (overwritten).
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        OUTPUT (atomic): incremented to the total number of emitted pairs.
        Caller must zero this before the call.
    tile_row_group : wp.array, shape (max_tiles,), dtype=wp.int32
        OUTPUT: row-side group indices of emitted pairs.
    tile_col_group : wp.array, shape (max_tiles,), dtype=wp.int32
        OUTPUT: column-side group indices of emitted pairs.
    wp_dtype : type
        Warp dtype.  Only ``wp.float32`` is supported.
    device : str
        Warp device string (e.g. ``"cuda:0"``).

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - group_ctr_{x,y,z}, group_ext_{x,y,z} : Per-group bboxes
        - num_tiles[0] : Set to the count of emitted tile pairs
        - tile_row_group, tile_col_group : Filled with emitted pair indices

    Raises
    ------
    NotImplementedError
        If ``wp_dtype`` is not ``wp.float32``.

    Notes
    -----
    - This is a low-level warp interface.  Framework bindings should call
      it through :mod:`nvalchemiops.torch.neighbors.tile_warp` /
      :mod:`nvalchemiops.jax.neighbors.tile_warp`.
    - Output arrays must be pre-allocated by the caller; sizing helpers
      live in the torch wrapper module (``estimate_tile_neighbor_list_sizes``).
    - Only upper-triangular pairs are emitted (``col_group >= row_group``).

    See Also
    --------
    _rank2group_warp_kernel : First of the two kernels launched.
    _group2tile_warp_kernel : Second kernel; performs the enumeration.
    tile_to_matrix : Consumes the emitted tile pair list (matrix output).
    tile_to_coo : Consumes the emitted tile pair list (COO output).
    """
    if wp_dtype is not wp.float32:
        raise NotImplementedError("tile_warp kernels currently support float32 only")
    ngroup = sorted_pos_x.shape[0] // TILE_GROUP_SIZE
    max_tiles = tile_row_group.shape[0]

    wp.launch_tiled(
        kernel=_rank2group_warp_kernel,
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
        kernel=_group2tile_warp_kernel,
        dim=[ngroup],
        inputs=[
            group_ctr_x,
            group_ctr_y,
            group_ctr_z,
            group_ext_x,
            group_ext_y,
            group_ext_z,
            cell,
            inv_cell,
            wp.float32(cutoff * cutoff),
            int(ngroup),
            num_tiles,
            tile_row_group,
            tile_col_group,
            int(max_tiles),
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )


def tile_to_matrix(
    sorted_atom_index: wp.array,
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
    cutoff: float,
    natom: int,
    n_tiles: int,
    neighbor_matrix: wp.array,
    num_neighbors: wp.array,
    neighbor_matrix_shifts: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for tile-pair → neighbor_matrix conversion.

    Launches :func:`_tile_to_matrix_kernel` over ``n_tiles`` blocks to
    convert the (row_group, col_group) tile pair list into a per-atom
    neighbor_matrix.  Both halves of each cross-tile pair are emitted at
    the atom level so each unordered atom pair appears in both atoms'
    rows.  Triclinic PBC is handled via ``cell`` + ``inv_cell``.

    Active slots are written with neighbor atom IDs and unconditional
    shift vectors; tail slots (``column >= num_neighbors[i]``) are left
    untouched and must be sentinel-filled via
    :func:`fill_neighbor_matrix_tail` if a sentinel is required.

    Parameters
    ----------
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Sort permutation from the Morton sort.
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Morton-sorted SoA positions.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        Atomic counter populated by :func:`build_tile_neighbor_list`.
        Unused by this launcher except for caller bookkeeping.
    tile_row_group : wp.array, shape (max_tiles,), dtype=wp.int32
        Row-side group index of each tile pair.
    tile_col_group : wp.array, shape (max_tiles,), dtype=wp.int32
        Column-side group index of each tile pair.
    cell : wp.mat33f
    inv_cell : wp.mat33f
    cutoff : float
        Cutoff distance.
    natom : int
        Real atom count.
    n_tiles : int
        Number of emitted tile pairs (``num_tiles[0]`` read at launch time).
        The launch is skipped when ``n_tiles <= 0``.
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
        Modifies outputs in-place; see :func:`_tile_to_matrix_kernel`.

    Raises
    ------
    NotImplementedError
        If ``wp_dtype`` is not ``wp.float32``.

    Notes
    -----
    - This is a low-level warp interface.  Framework bindings should call
      it through :mod:`nvalchemiops.torch.neighbors.tile_warp` /
      :mod:`nvalchemiops.jax.neighbors.tile_warp`.
    - Output arrays must be pre-allocated by the caller.

    See Also
    --------
    _tile_to_matrix_kernel : Kernel that performs the computation.
    build_tile_neighbor_list : Upstream tile-pair builder.
    fill_neighbor_matrix_tail : Post-conv tail fill.
    tile_to_coo : Alternate launcher producing flat COO output.
    """
    if wp_dtype is not wp.float32:
        raise NotImplementedError("tile_warp kernels currently support float32 only")
    if n_tiles <= 0:
        return
    max_neighbors = neighbor_matrix.shape[1]
    wp.launch_tiled(
        kernel=_tile_to_matrix_kernel,
        dim=[n_tiles],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            sorted_atom_index,
            cell,
            inv_cell,
            wp.float32(cutoff * cutoff),
            int(natom),
            int(max_neighbors),
            tile_row_group,
            tile_col_group,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )


def tile_to_coo(
    sorted_atom_index: wp.array,
    sorted_pos_x: wp.array,
    sorted_pos_y: wp.array,
    sorted_pos_z: wp.array,
    num_tiles: wp.array,
    tile_row_group: wp.array,
    tile_col_group: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
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
    """Core warp launcher for tile-pair → flat COO conversion.

    Launches :func:`_tile_to_coo_kernel` over ``n_tiles`` blocks to convert
    the (row_group, col_group) tile pair list into a flat
    ``(i_atom, j_atom, shift)`` COO list.  Both halves of each cross-tile
    pair are emitted (full-fill output).  Triclinic PBC is handled via
    ``cell`` + ``inv_cell``.  The active pair count after the call is
    ``pair_counter[0]``.

    Parameters
    ----------
    sorted_atom_index : wp.array, shape (n_padded,), dtype=wp.int32
        Sort permutation from the Morton sort.
    sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (n_padded,), dtype=wp.float32
        Morton-sorted SoA positions.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        Atomic counter populated by :func:`build_tile_neighbor_list`.
        Unused by this launcher except for caller bookkeeping.
    tile_row_group : wp.array, shape (max_tiles,), dtype=wp.int32
        Row-side group index of each tile pair.
    tile_col_group : wp.array, shape (max_tiles,), dtype=wp.int32
        Column-side group index of each tile pair.
    cell : wp.mat33f
    inv_cell : wp.mat33f
    cutoff : float
        Cutoff distance.
    natom : int
        Real atom count.
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
        Modifies outputs in-place; see :func:`_tile_to_coo_kernel`.

    Raises
    ------
    NotImplementedError
        If ``wp_dtype`` is not ``wp.float32``.

    Notes
    -----
    - This is a low-level warp interface.  Framework bindings should call
      it through :mod:`nvalchemiops.torch.neighbors.tile_warp` /
      :mod:`nvalchemiops.jax.neighbors.tile_warp`.
    - Output arrays must be pre-allocated by the caller.

    See Also
    --------
    _tile_to_coo_kernel : Kernel that performs the computation.
    build_tile_neighbor_list : Upstream tile-pair builder.
    tile_to_matrix : Alternate launcher producing per-atom matrix output.
    """
    if wp_dtype is not wp.float32:
        raise NotImplementedError("tile_warp kernels currently support float32 only")
    if n_tiles <= 0:
        return
    wp.launch_tiled(
        kernel=_tile_to_coo_kernel,
        dim=[n_tiles],
        inputs=[
            sorted_pos_x,
            sorted_pos_y,
            sorted_pos_z,
            sorted_atom_index,
            cell,
            inv_cell,
            wp.float32(cutoff * cutoff),
            int(natom),
            int(max_pairs),
            tile_row_group,
            tile_col_group,
            pair_counter,
            coo_list,
            coo_shifts,
        ],
        block_dim=TILE_GROUP_SIZE,
        device=device,
    )
