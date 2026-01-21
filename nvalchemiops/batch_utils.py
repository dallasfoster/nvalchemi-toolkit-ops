# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Batch Utilities for Multi-System Operations
============================================

This module provides GPU-accelerated utilities for working with batched systems,
supporting both indexing modes:

1. **batch_idx mode**: Each atom is tagged with a system index
   - Shape: (N_total,), dtype=int32
   - Value at index i indicates which system atom i belongs to

2. **atom_ptr mode (CSR)**: Atom ranges defined by cumulative pointers
   - Shape: (num_systems + 1,), dtype=int32
   - System s owns atoms in range [atom_ptr[s], atom_ptr[s+1])

The utilities include:
- Index/pointer conversion between representations
- Per-system reductions (sum, max, min, mean)
- Per-system vector operations (max norm, sum vectors)
- Broadcasting from per-system to per-atom arrays

Example
-------
>>> import warp as wp
>>> from nvalchemiops.batch_utils import (
...     create_batch_idx, create_atom_ptr, sum_per_system, max_norm_per_system
... )
>>>
>>> # Create batch representations from atom counts
>>> atom_counts = wp.array([10, 15, 12], dtype=wp.int32, device="cuda:0")
>>> batch_idx = create_batch_idx(atom_counts, device="cuda:0")
>>> atom_ptr = create_atom_ptr(atom_counts, device="cuda:0")
>>>
>>> # Reduce per-atom energies to per-system totals
>>> energies = wp.array([...], dtype=wp.float64, device="cuda:0")  # shape (37,)
>>> system_energies = sum_per_system(energies, atom_ptr=atom_ptr, device="cuda:0")
>>>
>>> # Compute max force magnitude per system
>>> forces = wp.array([...], dtype=wp.vec3d, device="cuda:0")  # shape (37,)
>>> max_forces = max_norm_per_system(forces, atom_ptr=atom_ptr, device="cuda:0")
"""

from __future__ import annotations

from typing import Any

import warp as wp

# =============================================================================
# Index/Pointer Conversion Kernels
# =============================================================================


@wp.kernel
def _create_batch_idx_kernel(
    atom_ptr: wp.array(dtype=wp.int32),
    batch_idx: wp.array(dtype=wp.int32),
):
    """Create batch_idx from atom_ptr by assigning system indices to atoms.

    Each thread handles one system, iterating through its atoms and assigning
    the system index to each atom's batch_idx entry.

    Parameters
    ----------
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer where system s owns atoms [atom_ptr[s], atom_ptr[s+1]).
    batch_idx : wp.array, shape (total_atoms,), dtype=int32
        OUTPUT: System index for each atom. Modified in-place.

    Launch Grid
    -----------
    dim = num_systems
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]
    for i in range(a0, a1):
        batch_idx[i] = wp.int32(sys)


@wp.kernel
def _create_atom_ptr_kernel(
    atom_counts: wp.array(dtype=wp.int32),
    atom_ptr: wp.array(dtype=wp.int32),
):
    """Create atom_ptr from atom_counts using sequential prefix sum.

    This is a single-threaded kernel that computes the cumulative sum of
    atom counts. For large num_systems, a parallel prefix sum would be
    more efficient.

    Parameters
    ----------
    atom_counts : wp.array, shape (num_systems,), dtype=int32
        Number of atoms in each system.
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        OUTPUT: CSR-style pointer array. Modified in-place.
        atom_ptr[0] = 0, atom_ptr[i+1] = atom_ptr[i] + atom_counts[i].

    Launch Grid
    -----------
    dim = 1 (single thread)
    """
    tid = wp.tid()
    if tid == 0:
        atom_ptr[0] = wp.int32(0)
        for i in range(atom_counts.shape[0]):
            atom_ptr[i + 1] = atom_ptr[i] + atom_counts[i]


@wp.kernel
def _atoms_per_system_from_batch_idx_kernel(
    batch_idx: wp.array(dtype=wp.int32),
    atom_counts: wp.array(dtype=wp.int32),
):
    """Count atoms per system from batch_idx using atomic increments.

    Each thread handles one atom, atomically incrementing the count for
    its system. The atom_counts array must be zero-initialized before launch.

    Parameters
    ----------
    batch_idx : wp.array, shape (total_atoms,), dtype=int32
        System index for each atom.
    atom_counts : wp.array, shape (num_systems,), dtype=int32
        OUTPUT: Number of atoms per system. Must be zero-initialized.
        Modified in-place using atomic operations.

    Launch Grid
    -----------
    dim = total_atoms
    """
    i = wp.tid()
    sys = batch_idx[i]
    wp.atomic_add(atom_counts, sys, wp.int32(1))


@wp.kernel
def _atoms_per_system_from_ptr_kernel(
    atom_ptr: wp.array(dtype=wp.int32),
    atom_counts: wp.array(dtype=wp.int32),
):
    """Compute atoms per system from atom_ptr by differencing adjacent pointers.

    Each thread handles one system, computing the count as the difference
    between consecutive pointer values.

    Parameters
    ----------
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer array.
    atom_counts : wp.array, shape (num_systems,), dtype=int32
        OUTPUT: Number of atoms per system. Modified in-place.

    Launch Grid
    -----------
    dim = num_systems
    """
    sys = wp.tid()
    atom_counts[sys] = atom_ptr[sys + 1] - atom_ptr[sys]


# =============================================================================
# Per-System Scalar Reduction Kernels
# =============================================================================


@wp.kernel
def _sum_per_system_ptr_kernel(
    values: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Sum scalar values per system using atom_ptr (CSR mode).

    Each thread handles one system, iterating through its atoms and
    accumulating the sum. This avoids atomic operations and is efficient
    for systems with many atoms.

    Parameters
    ----------
    values : wp.array, shape (total_atoms,), dtype=float32 or float64
        Per-atom scalar values to sum.
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer where system s owns atoms [atom_ptr[s], atom_ptr[s+1]).
    result : wp.array, shape (num_systems,), dtype=same as values
        OUTPUT: Per-system sums. Modified in-place.

    Launch Grid
    -----------
    dim = num_systems
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    total = type(values[0])(0.0)
    for i in range(a0, a1):
        total = total + values[i]
    result[sys] = total


@wp.kernel
def _sum_per_system_batch_idx_kernel(
    values: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Sum scalar values per system using batch_idx with atomic additions.

    Each thread handles one atom, atomically adding its value to the
    corresponding system's sum. The result array must be zero-initialized.

    Parameters
    ----------
    values : wp.array, shape (total_atoms,), dtype=float32 or float64
        Per-atom scalar values to sum.
    batch_idx : wp.array, shape (total_atoms,), dtype=int32
        System index for each atom.
    result : wp.array, shape (num_systems,), dtype=same as values
        OUTPUT: Per-system sums. Must be zero-initialized.
        Modified in-place using atomic operations.

    Launch Grid
    -----------
    dim = total_atoms
    """
    i = wp.tid()
    sys = batch_idx[i]
    wp.atomic_add(result, sys, values[i])


@wp.kernel
def _max_per_system_ptr_kernel(
    values: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Maximum scalar value per system using atom_ptr (CSR mode).

    Each thread handles one system, iterating through its atoms to find
    the maximum value. Systems with no atoms leave result unchanged.

    Parameters
    ----------
    values : wp.array, shape (total_atoms,), dtype=float32 or float64
        Per-atom scalar values.
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer where system s owns atoms [atom_ptr[s], atom_ptr[s+1]).
    result : wp.array, shape (num_systems,), dtype=same as values
        OUTPUT: Per-system maximum values. Should be initialized to -inf
        for systems that may have no atoms. Modified in-place.

    Launch Grid
    -----------
    dim = num_systems
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    if a0 < a1:
        max_val = values[a0]
        for i in range(a0 + 1, a1):
            if values[i] > max_val:
                max_val = values[i]
        result[sys] = max_val


@wp.kernel
def _min_per_system_ptr_kernel(
    values: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Minimum scalar value per system using atom_ptr (CSR mode).

    Each thread handles one system, iterating through its atoms to find
    the minimum value. Systems with no atoms leave result unchanged.

    Parameters
    ----------
    values : wp.array, shape (total_atoms,), dtype=float32 or float64
        Per-atom scalar values.
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer where system s owns atoms [atom_ptr[s], atom_ptr[s+1]).
    result : wp.array, shape (num_systems,), dtype=same as values
        OUTPUT: Per-system minimum values. Should be initialized to +inf
        for systems that may have no atoms. Modified in-place.

    Launch Grid
    -----------
    dim = num_systems
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    if a0 < a1:
        min_val = values[a0]
        for i in range(a0 + 1, a1):
            if values[i] < min_val:
                min_val = values[i]
        result[sys] = min_val


# =============================================================================
# Per-System Vector Reduction Kernels
# =============================================================================


@wp.kernel
def _sum_vectors_per_system_ptr_kernel(
    vectors: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Sum vectors per system using atom_ptr (CSR mode).

    Each thread handles one system, iterating through its atoms and
    accumulating the vector sum. This avoids atomic operations.

    Parameters
    ----------
    vectors : wp.array, shape (total_atoms,), dtype=vec3f or vec3d
        Per-atom vectors to sum.
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer where system s owns atoms [atom_ptr[s], atom_ptr[s+1]).
    result : wp.array, shape (num_systems,), dtype=same as vectors
        OUTPUT: Per-system vector sums. Modified in-place.

    Launch Grid
    -----------
    dim = num_systems
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    total = type(vectors[0])(type(vectors[0][0])(0.0))
    for i in range(a0, a1):
        total = total + vectors[i]
    result[sys] = total


@wp.kernel
def _sum_vectors_per_system_batch_idx_kernel(
    vectors: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Sum vectors per system using batch_idx with atomic additions.

    Each thread handles one atom, atomically adding its vector to the
    corresponding system's sum. The result array must be zero-initialized.

    Note: Warp supports atomic_add on vec3 types, accumulating each component.

    Parameters
    ----------
    vectors : wp.array, shape (total_atoms,), dtype=vec3f or vec3d
        Per-atom vectors to sum.
    batch_idx : wp.array, shape (total_atoms,), dtype=int32
        System index for each atom.
    result : wp.array, shape (num_systems,), dtype=same as vectors
        OUTPUT: Per-system vector sums. Must be zero-initialized.
        Modified in-place using atomic operations.

    Launch Grid
    -----------
    dim = total_atoms
    """
    i = wp.tid()
    sys = batch_idx[i]
    v = vectors[i]
    wp.atomic_add(result, sys, v)


@wp.kernel
def _max_norm_per_system_ptr_kernel(
    vectors: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Maximum vector norm per system using atom_ptr (CSR mode).

    Each thread handles one system, iterating through its atoms to find
    the maximum Euclidean norm. Uses squared norms internally to avoid
    redundant sqrt calls, taking the final sqrt only once.

    Parameters
    ----------
    vectors : wp.array, shape (total_atoms,), dtype=vec3f or vec3d
        Per-atom vectors.
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer where system s owns atoms [atom_ptr[s], atom_ptr[s+1]).
    result : wp.array, shape (num_systems,), dtype=float32 or float64
        OUTPUT: Per-system maximum vector norms. Modified in-place.
        Returns 0 for systems with no atoms.

    Launch Grid
    -----------
    dim = num_systems
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    max_norm_sq = type(vectors[0][0])(0.0)

    for i in range(a0, a1):
        v = vectors[i]
        norm_sq = wp.dot(v, v)
        if norm_sq > max_norm_sq:
            max_norm_sq = norm_sq

    result[sys] = wp.sqrt(max_norm_sq)


@wp.kernel
def _max_norm_per_system_batch_idx_kernel(
    vectors: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Maximum vector norm per system using batch_idx with atomic max.

    Each thread handles one atom, computing its norm and atomically
    updating the maximum for its system. The result array should be
    zero-initialized before launch.

    Parameters
    ----------
    vectors : wp.array, shape (total_atoms,), dtype=vec3f or vec3d
        Per-atom vectors.
    batch_idx : wp.array, shape (total_atoms,), dtype=int32
        System index for each atom.
    result : wp.array, shape (num_systems,), dtype=float32 or float64
        OUTPUT: Per-system maximum vector norms. Should be zero-initialized.
        Modified in-place using atomic max operations.

    Launch Grid
    -----------
    dim = total_atoms
    """
    i = wp.tid()
    sys = batch_idx[i]
    v = vectors[i]
    norm = wp.sqrt(wp.dot(v, v))
    wp.atomic_max(result, sys, norm)


@wp.kernel
def _rms_norm_per_system_ptr_kernel(
    vectors: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Root-mean-square (RMS) vector norm per system using atom_ptr (CSR mode).

    Computes sqrt(mean(|v|²)) for each system, where |v|² is the squared
    Euclidean norm of each vector. This is useful for convergence criteria
    where a single outlier shouldn't dominate (compared to max norm).

    Parameters
    ----------
    vectors : wp.array, shape (total_atoms,), dtype=vec3f or vec3d
        Per-atom vectors.
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer where system s owns atoms [atom_ptr[s], atom_ptr[s+1]).
    result : wp.array, shape (num_systems,), dtype=float32 or float64
        OUTPUT: Per-system RMS vector norms. Modified in-place.
        Returns 0 for systems with no atoms.

    Launch Grid
    -----------
    dim = num_systems
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]

    sum_norm_sq = type(vectors[0][0])(0.0)
    count = a1 - a0

    for i in range(a0, a1):
        v = vectors[i]
        sum_norm_sq += wp.dot(v, v)

    if count > 0:
        result[sys] = wp.sqrt(sum_norm_sq / type(vectors[0][0])(count))
    else:
        result[sys] = type(vectors[0][0])(0.0)


# =============================================================================
# Broadcasting Kernels
# =============================================================================


@wp.kernel
def _broadcast_to_atoms_ptr_kernel(
    per_system_values: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=wp.int32),
    per_atom_values: wp.array(dtype=Any),
):
    """Broadcast per-system values to per-atom array using atom_ptr (CSR mode).

    Each thread handles one system, copying its value to all atoms belonging
    to that system. This is useful for expanding per-system quantities
    (e.g., temperature, box size) to per-atom arrays.

    Parameters
    ----------
    per_system_values : wp.array, shape (num_systems,), dtype=any
        Values for each system (scalars or vectors).
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer where system s owns atoms [atom_ptr[s], atom_ptr[s+1]).
    per_atom_values : wp.array, shape (total_atoms,), dtype=same as per_system_values
        OUTPUT: Values broadcast to each atom. Modified in-place.

    Launch Grid
    -----------
    dim = num_systems
    """
    sys = wp.tid()
    a0 = atom_ptr[sys]
    a1 = atom_ptr[sys + 1]
    val = per_system_values[sys]
    for i in range(a0, a1):
        per_atom_values[i] = val


@wp.kernel
def _broadcast_to_atoms_batch_idx_kernel(
    per_system_values: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    per_atom_values: wp.array(dtype=Any),
):
    """Broadcast per-system values to per-atom array using batch_idx.

    Each thread handles one atom, looking up the value for its system
    and storing it. This is more parallel than the atom_ptr version
    but requires the batch_idx array.

    Parameters
    ----------
    per_system_values : wp.array, shape (num_systems,), dtype=any
        Values for each system (scalars or vectors).
    batch_idx : wp.array, shape (total_atoms,), dtype=int32
        System index for each atom.
    per_atom_values : wp.array, shape (total_atoms,), dtype=same as per_system_values
        OUTPUT: Values broadcast to each atom. Modified in-place.

    Launch Grid
    -----------
    dim = total_atoms
    """
    i = wp.tid()
    sys = batch_idx[i]
    per_atom_values[i] = per_system_values[sys]


# =============================================================================
# Fill Kernels (for min/max initialization without numpy)
# =============================================================================


@wp.kernel
def _fill_scalar_kernel(
    arr: wp.array(dtype=Any),
    value: Any,
):
    """Fill array with a scalar value.

    Parameters
    ----------
    arr : wp.array, shape (N,), dtype=float32 or float64
        OUTPUT: Array to fill. Modified in-place.
    value : scalar
        Value to fill with.

    Launch Grid
    -----------
    dim = N
    """
    i = wp.tid()
    arr[i] = value


@wp.kernel
def _divide_arrays_kernel(
    numerator: wp.array(dtype=Any),
    denominator: wp.array(dtype=wp.int32),
    result: wp.array(dtype=Any),
):
    """Element-wise division of arrays, handling zero denominators.

    Parameters
    ----------
    numerator : wp.array, shape (N,), dtype=float32 or float64
        Numerator values.
    denominator : wp.array, shape (N,), dtype=int32
        Denominator values (atom counts).
    result : wp.array, shape (N,), dtype=same as numerator
        OUTPUT: Result of division. Modified in-place.
        Zero denominators produce zero results.

    Launch Grid
    -----------
    dim = N
    """
    i = wp.tid()
    if denominator[i] > 0:
        result[i] = numerator[i] / type(numerator[0])(denominator[i])
    else:
        result[i] = type(numerator[0])(0.0)


# =============================================================================
# Generate Overloads
# =============================================================================

_SCALAR_TYPES = [wp.float32, wp.float64]
_VEC_TYPES = [wp.vec3f, wp.vec3d]

# Scalar reduction overloads
_sum_per_system_ptr_overloads = {}
_sum_per_system_batch_idx_overloads = {}
_max_per_system_ptr_overloads = {}
_min_per_system_ptr_overloads = {}

for t in _SCALAR_TYPES:
    _sum_per_system_ptr_overloads[t] = wp.overload(
        _sum_per_system_ptr_kernel,
        [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )
    _sum_per_system_batch_idx_overloads[t] = wp.overload(
        _sum_per_system_batch_idx_kernel,
        [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )
    _max_per_system_ptr_overloads[t] = wp.overload(
        _max_per_system_ptr_kernel,
        [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )
    _min_per_system_ptr_overloads[t] = wp.overload(
        _min_per_system_ptr_kernel,
        [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )

# Vector reduction overloads
_sum_vectors_per_system_ptr_overloads = {}
_sum_vectors_per_system_batch_idx_overloads = {}
_max_norm_per_system_ptr_overloads = {}
_max_norm_per_system_batch_idx_overloads = {}
_rms_norm_per_system_ptr_overloads = {}

for v, t in zip(_VEC_TYPES, _SCALAR_TYPES):
    _sum_vectors_per_system_ptr_overloads[v] = wp.overload(
        _sum_vectors_per_system_ptr_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=v)],
    )
    _sum_vectors_per_system_batch_idx_overloads[v] = wp.overload(
        _sum_vectors_per_system_batch_idx_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=v)],
    )
    _max_norm_per_system_ptr_overloads[v] = wp.overload(
        _max_norm_per_system_ptr_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )
    _max_norm_per_system_batch_idx_overloads[v] = wp.overload(
        _max_norm_per_system_batch_idx_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )
    _rms_norm_per_system_ptr_overloads[v] = wp.overload(
        _rms_norm_per_system_ptr_kernel,
        [wp.array(dtype=v), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )

# Broadcast overloads
_broadcast_to_atoms_ptr_overloads = {}
_broadcast_to_atoms_batch_idx_overloads = {}

for t in _SCALAR_TYPES + _VEC_TYPES:
    _broadcast_to_atoms_ptr_overloads[t] = wp.overload(
        _broadcast_to_atoms_ptr_kernel,
        [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )
    _broadcast_to_atoms_batch_idx_overloads[t] = wp.overload(
        _broadcast_to_atoms_batch_idx_kernel,
        [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )

# Fill and divide kernel overloads
_fill_scalar_kernel_overloads = {}
_divide_arrays_kernel_overloads = {}

for t in _SCALAR_TYPES:
    _fill_scalar_kernel_overloads[t] = wp.overload(
        _fill_scalar_kernel,
        [wp.array(dtype=t), t],
    )
    _divide_arrays_kernel_overloads[t] = wp.overload(
        _divide_arrays_kernel,
        [wp.array(dtype=t), wp.array(dtype=wp.int32), wp.array(dtype=t)],
    )


# =============================================================================
# Public API: Index/Pointer Conversion
# =============================================================================


def create_batch_idx(
    atom_counts: wp.array,
    total_atoms: int = None,
    batch_idx: wp.array = None,
    device: str = None,
) -> wp.array:
    """Create batch_idx array from per-system atom counts.

    Parameters
    ----------
    atom_counts : wp.array, shape (num_systems,), dtype=int32
        Number of atoms in each system.
    total_atoms : int, optional
        Total number of atoms (sum of atom_counts). If not provided,
        will sync to compute from atom_counts (avoid if possible).
    batch_idx : wp.array, optional
        Pre-allocated output array, shape (total_atoms,), dtype=int32.
        If provided, total_atoms is inferred from its shape.
    device : str, optional
        Warp device.

    Returns
    -------
    batch_idx : wp.array, shape (total_atoms,), dtype=int32
        System index for each atom.

    Example
    -------
    >>> atom_counts = wp.array([10, 15, 12], dtype=wp.int32, device="cuda:0")
    >>> batch_idx = create_batch_idx(atom_counts, total_atoms=37, device="cuda:0")
    >>> # batch_idx = [0,0,0,...(10 times), 1,1,1,...(15 times), 2,2,2,...(12 times)]
    """
    if device is None:
        device = atom_counts.device

    num_systems = atom_counts.shape[0]

    # First create atom_ptr
    atom_ptr = wp.zeros(num_systems + 1, dtype=wp.int32, device=device)
    wp.launch(
        _create_atom_ptr_kernel,
        dim=1,
        inputs=[atom_counts, atom_ptr],
        device=device,
    )

    # Determine total_atoms
    if batch_idx is not None:
        total_atoms = batch_idx.shape[0]
    elif total_atoms is None:
        # Fallback: sync to get total_atoms (avoid if possible)
        total_atoms = int(atom_counts.numpy().sum())

    # Allocate batch_idx if not provided
    if batch_idx is None:
        batch_idx = wp.zeros(total_atoms, dtype=wp.int32, device=device)

    wp.launch(
        _create_batch_idx_kernel,
        dim=num_systems,
        inputs=[atom_ptr, batch_idx],
        device=device,
    )

    return batch_idx


def create_atom_ptr(
    atom_counts: wp.array,
    atom_ptr: wp.array = None,
    device: str = None,
) -> wp.array:
    """Create atom_ptr (CSR pointer) array from per-system atom counts.

    Parameters
    ----------
    atom_counts : wp.array, shape (num_systems,), dtype=int32
        Number of atoms in each system.
    atom_ptr : wp.array, optional
        Pre-allocated output array, shape (num_systems + 1,), dtype=int32.
    device : str, optional
        Warp device.

    Returns
    -------
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer where system s owns atoms [atom_ptr[s], atom_ptr[s+1]).

    Example
    -------
    >>> atom_counts = wp.array([10, 15, 12], dtype=wp.int32, device="cuda:0")
    >>> atom_ptr = create_atom_ptr(atom_counts, device="cuda:0")
    >>> # atom_ptr = [0, 10, 25, 37]
    """
    if device is None:
        device = atom_counts.device

    num_systems = atom_counts.shape[0]

    if atom_ptr is None:
        atom_ptr = wp.zeros(num_systems + 1, dtype=wp.int32, device=device)

    wp.launch(
        _create_atom_ptr_kernel,
        dim=1,
        inputs=[atom_counts, atom_ptr],
        device=device,
    )

    return atom_ptr


def batch_idx_to_atom_ptr(
    batch_idx: wp.array,
    num_systems: int,
    device: str = None,
) -> wp.array:
    """Convert batch_idx to atom_ptr representation.

    Parameters
    ----------
    batch_idx : wp.array, shape (total_atoms,), dtype=int32
        System index for each atom.
    num_systems : int
        Number of systems.
    device : str, optional
        Warp device.

    Returns
    -------
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer array.
    """
    if device is None:
        device = batch_idx.device

    # Count atoms per system
    atom_counts = wp.zeros(num_systems, dtype=wp.int32, device=device)
    wp.launch(
        _atoms_per_system_from_batch_idx_kernel,
        dim=batch_idx.shape[0],
        inputs=[batch_idx, atom_counts],
        device=device,
    )

    # Convert to atom_ptr
    return create_atom_ptr(atom_counts, device=device)


def atom_ptr_to_batch_idx(
    atom_ptr: wp.array,
    total_atoms: int = None,
    batch_idx: wp.array = None,
    device: str = None,
) -> wp.array:
    """Convert atom_ptr to batch_idx representation.

    Parameters
    ----------
    atom_ptr : wp.array, shape (num_systems + 1,), dtype=int32
        CSR-style pointer array.
    total_atoms : int, optional
        Total number of atoms (atom_ptr[-1]). If not provided,
        will sync to read from atom_ptr (avoid if possible).
    batch_idx : wp.array, optional
        Pre-allocated output array, shape (total_atoms,), dtype=int32.
        If provided, total_atoms is inferred from its shape.
    device : str, optional
        Warp device.

    Returns
    -------
    batch_idx : wp.array, shape (total_atoms,), dtype=int32
        System index for each atom.
    """
    if device is None:
        device = atom_ptr.device

    num_systems = atom_ptr.shape[0] - 1

    # Determine total_atoms
    if batch_idx is not None:
        total_atoms = batch_idx.shape[0]
    elif total_atoms is None:
        # Fallback: sync to get total_atoms (avoid if possible)
        total_atoms = int(atom_ptr.numpy()[-1])

    # Allocate batch_idx if not provided
    if batch_idx is None:
        batch_idx = wp.zeros(total_atoms, dtype=wp.int32, device=device)

    wp.launch(
        _create_batch_idx_kernel,
        dim=num_systems,
        inputs=[atom_ptr, batch_idx],
        device=device,
    )

    return batch_idx


def atoms_per_system(
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    num_systems: int = None,
    device: str = None,
) -> wp.array:
    """Get atom counts per system from either representation.

    Parameters
    ----------
    batch_idx : wp.array, optional
        System index for each atom. Shape (total_atoms,), dtype=int32.
    atom_ptr : wp.array, optional
        CSR-style pointer. Shape (num_systems + 1,), dtype=int32.
    num_systems : int, optional
        Number of systems (required if using batch_idx).
    device : str, optional
        Warp device.

    Returns
    -------
    atom_counts : wp.array, shape (num_systems,), dtype=int32
        Number of atoms in each system.
    """
    if batch_idx is None and atom_ptr is None:
        raise ValueError("Either batch_idx or atom_ptr must be provided")

    if atom_ptr is not None:
        if device is None:
            device = atom_ptr.device
        n_sys = atom_ptr.shape[0] - 1
        atom_counts = wp.zeros(n_sys, dtype=wp.int32, device=device)
        wp.launch(
            _atoms_per_system_from_ptr_kernel,
            dim=n_sys,
            inputs=[atom_ptr, atom_counts],
            device=device,
        )
        return atom_counts
    else:
        if num_systems is None:
            raise ValueError("num_systems required when using batch_idx")
        if device is None:
            device = batch_idx.device
        atom_counts = wp.zeros(num_systems, dtype=wp.int32, device=device)
        wp.launch(
            _atoms_per_system_from_batch_idx_kernel,
            dim=batch_idx.shape[0],
            inputs=[batch_idx, atom_counts],
            device=device,
        )
        return atom_counts


# =============================================================================
# Public API: Per-System Scalar Reductions
# =============================================================================


def sum_per_system(
    values: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    num_systems: int = None,
    result: wp.array = None,
    device: str = None,
) -> wp.array:
    """Sum per-atom scalar values to per-system totals.

    Parameters
    ----------
    values : wp.array, shape (total_atoms,), dtype=float32 or float64
        Per-atom scalar values.
    batch_idx : wp.array, optional
        System index for each atom.
    atom_ptr : wp.array, optional
        CSR-style pointer array. Preferred for efficiency.
    num_systems : int, optional
        Number of systems (required if using batch_idx without atom_ptr).
    result : wp.array, optional
        Pre-allocated output array, shape (num_systems,).
        Must be zero-initialized for batch_idx mode.
    device : str, optional
        Warp device.

    Returns
    -------
    result : wp.array, shape (num_systems,), dtype same as values
        Per-system sum of values.

    Notes
    -----
    When both batch_idx and atom_ptr are provided, atom_ptr is used for efficiency.
    """
    if batch_idx is None and atom_ptr is None:
        raise ValueError("Either batch_idx or atom_ptr must be provided")

    if device is None:
        device = values.device

    dtype = values.dtype

    if atom_ptr is not None:
        n_sys = atom_ptr.shape[0] - 1
        if result is None:
            result = wp.zeros(n_sys, dtype=dtype, device=device)
        wp.launch(
            _sum_per_system_ptr_overloads[dtype],
            dim=n_sys,
            inputs=[values, atom_ptr],
            outputs=[result],
            device=device,
        )
    else:
        if num_systems is None:
            raise ValueError(
                "num_systems required when using batch_idx without atom_ptr"
            )
        if result is None:
            result = wp.zeros(num_systems, dtype=dtype, device=device)
        wp.launch(
            _sum_per_system_batch_idx_overloads[dtype],
            dim=values.shape[0],
            inputs=[values, batch_idx],
            outputs=[result],
            device=device,
        )

    return result


def max_per_system(
    values: wp.array,
    atom_ptr: wp.array,
    result: wp.array = None,
    device: str = None,
) -> wp.array:
    """Maximum per-atom scalar value per system.

    Parameters
    ----------
    values : wp.array, shape (total_atoms,), dtype=float32 or float64
        Per-atom scalar values.
    atom_ptr : wp.array
        CSR-style pointer array.
    result : wp.array, optional
        Pre-allocated output array, shape (num_systems,). If provided,
        avoids allocation (but not initialization to -inf).
    device : str, optional
        Warp device.

    Returns
    -------
    result : wp.array, shape (num_systems,), dtype same as values
        Per-system maximum values.

    Notes
    -----
    Only atom_ptr mode is supported for max reduction (atomic max on floats
    is not reliable across all platforms).
    """
    if device is None:
        device = values.device

    dtype = values.dtype
    n_sys = atom_ptr.shape[0] - 1

    # Allocate result if not provided
    if result is None:
        result = wp.zeros(n_sys, dtype=dtype, device=device)

    # Initialize with -inf using GPU kernel
    if dtype == wp.float32:
        init_val = wp.float32(-3.4028235e38)  # -FLT_MAX (approx -inf)
    else:
        init_val = wp.float64(-1.7976931348623157e308)  # -DBL_MAX (approx -inf)

    wp.launch(
        _fill_scalar_kernel_overloads[dtype],
        dim=n_sys,
        inputs=[result, init_val],
        device=device,
    )

    wp.launch(
        _max_per_system_ptr_overloads[dtype],
        dim=n_sys,
        inputs=[values, atom_ptr],
        outputs=[result],
        device=device,
    )

    return result


def min_per_system(
    values: wp.array,
    atom_ptr: wp.array,
    result: wp.array = None,
    device: str = None,
) -> wp.array:
    """Minimum per-atom scalar value per system.

    Parameters
    ----------
    values : wp.array, shape (total_atoms,), dtype=float32 or float64
        Per-atom scalar values.
    atom_ptr : wp.array
        CSR-style pointer array.
    result : wp.array, optional
        Pre-allocated output array, shape (num_systems,). If provided,
        avoids allocation (but not initialization to +inf).
    device : str, optional
        Warp device.

    Returns
    -------
    result : wp.array, shape (num_systems,), dtype same as values
        Per-system minimum values.
    """
    if device is None:
        device = values.device

    dtype = values.dtype
    n_sys = atom_ptr.shape[0] - 1

    # Allocate result if not provided
    if result is None:
        result = wp.zeros(n_sys, dtype=dtype, device=device)

    # Initialize with +inf using GPU kernel
    if dtype == wp.float32:
        init_val = wp.float32(3.4028235e38)  # FLT_MAX (approx +inf)
    else:
        init_val = wp.float64(1.7976931348623157e308)  # DBL_MAX (approx +inf)

    wp.launch(
        _fill_scalar_kernel_overloads[dtype],
        dim=n_sys,
        inputs=[result, init_val],
        device=device,
    )

    wp.launch(
        _min_per_system_ptr_overloads[dtype],
        dim=n_sys,
        inputs=[values, atom_ptr],
        outputs=[result],
        device=device,
    )

    return result


def mean_per_system(
    values: wp.array,
    atom_ptr: wp.array,
    result: wp.array = None,
    device: str = None,
) -> wp.array:
    """Mean per-atom scalar value per system.

    Parameters
    ----------
    values : wp.array, shape (total_atoms,), dtype=float32 or float64
        Per-atom scalar values.
    atom_ptr : wp.array
        CSR-style pointer array.
    result : wp.array, optional
        Pre-allocated output array, shape (num_systems,).
    device : str, optional
        Warp device.

    Returns
    -------
    result : wp.array, shape (num_systems,), dtype same as values
        Per-system mean values.
    """
    if device is None:
        device = values.device

    dtype = values.dtype
    n_sys = atom_ptr.shape[0] - 1

    # Get sum and count
    sums = sum_per_system(values, atom_ptr=atom_ptr, device=device)
    counts = atoms_per_system(atom_ptr=atom_ptr, device=device)

    # Allocate result if not provided
    if result is None:
        result = wp.zeros(n_sys, dtype=dtype, device=device)

    # Divide on GPU
    wp.launch(
        _divide_arrays_kernel_overloads[dtype],
        dim=n_sys,
        inputs=[sums, counts],
        outputs=[result],
        device=device,
    )

    return result


# =============================================================================
# Public API: Per-System Vector Reductions
# =============================================================================


def sum_vectors_per_system(
    vectors: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    num_systems: int = None,
    result: wp.array = None,
    device: str = None,
) -> wp.array:
    """Sum per-atom vectors to per-system totals.

    Parameters
    ----------
    vectors : wp.array, shape (total_atoms,), dtype=vec3f or vec3d
        Per-atom vectors.
    batch_idx : wp.array, optional
        System index for each atom.
    atom_ptr : wp.array, optional
        CSR-style pointer array. Preferred for efficiency.
    num_systems : int, optional
        Number of systems (required if using batch_idx without atom_ptr).
    result : wp.array, optional
        Pre-allocated output array, shape (num_systems,).
        Must be zero-initialized for batch_idx mode.
    device : str, optional
        Warp device.

    Returns
    -------
    result : wp.array, shape (num_systems,), dtype same as vectors
        Per-system sum of vectors.
    """
    if batch_idx is None and atom_ptr is None:
        raise ValueError("Either batch_idx or atom_ptr must be provided")

    if device is None:
        device = vectors.device

    vec_dtype = vectors.dtype

    if atom_ptr is not None:
        n_sys = atom_ptr.shape[0] - 1
        if result is None:
            result = wp.zeros(n_sys, dtype=vec_dtype, device=device)
        wp.launch(
            _sum_vectors_per_system_ptr_overloads[vec_dtype],
            dim=n_sys,
            inputs=[vectors, atom_ptr],
            outputs=[result],
            device=device,
        )
    else:
        if num_systems is None:
            raise ValueError(
                "num_systems required when using batch_idx without atom_ptr"
            )

        if result is None:
            result = wp.zeros(num_systems, dtype=vec_dtype, device=device)

        wp.launch(
            _sum_vectors_per_system_batch_idx_overloads[vec_dtype],
            dim=vectors.shape[0],
            inputs=[vectors, batch_idx],
            outputs=[result],
            device=device,
        )

    return result


def max_norm_per_system(
    vectors: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    num_systems: int = None,
    result: wp.array = None,
    device: str = None,
) -> wp.array:
    """Maximum vector norm per system.

    Parameters
    ----------
    vectors : wp.array, shape (total_atoms,), dtype=vec3f or vec3d
        Per-atom vectors.
    batch_idx : wp.array, optional
        System index for each atom.
    atom_ptr : wp.array, optional
        CSR-style pointer array. Preferred for efficiency.
    num_systems : int, optional
        Number of systems (required if using batch_idx without atom_ptr).
    result : wp.array, optional
        Pre-allocated output array, shape (num_systems,).
        Must be zero-initialized for batch_idx mode.
    device : str, optional
        Warp device.

    Returns
    -------
    result : wp.array, shape (num_systems,), dtype=float32 or float64
        Per-system maximum vector norm.

    Example
    -------
    >>> forces = wp.array([...], dtype=wp.vec3d, device="cuda:0")
    >>> max_forces = max_norm_per_system(forces, atom_ptr=atom_ptr, device="cuda:0")
    """
    if batch_idx is None and atom_ptr is None:
        raise ValueError("Either batch_idx or atom_ptr must be provided")

    if device is None:
        device = vectors.device

    vec_dtype = vectors.dtype
    scalar_dtype = wp.float32 if vec_dtype == wp.vec3f else wp.float64

    if atom_ptr is not None:
        n_sys = atom_ptr.shape[0] - 1
        if result is None:
            result = wp.zeros(n_sys, dtype=scalar_dtype, device=device)
        wp.launch(
            _max_norm_per_system_ptr_overloads[vec_dtype],
            dim=n_sys,
            inputs=[vectors, atom_ptr],
            outputs=[result],
            device=device,
        )
    else:
        if num_systems is None:
            raise ValueError(
                "num_systems required when using batch_idx without atom_ptr"
            )
        if result is None:
            result = wp.zeros(num_systems, dtype=scalar_dtype, device=device)
        wp.launch(
            _max_norm_per_system_batch_idx_overloads[vec_dtype],
            dim=vectors.shape[0],
            inputs=[vectors, batch_idx],
            outputs=[result],
            device=device,
        )

    return result


def rms_norm_per_system(
    vectors: wp.array,
    atom_ptr: wp.array,
    result: wp.array = None,
    device: str = None,
) -> wp.array:
    """RMS (root mean square) vector norm per system.

    Computes sqrt(mean(|v|²)) for each system.

    Parameters
    ----------
    vectors : wp.array, shape (total_atoms,), dtype=vec3f or vec3d
        Per-atom vectors.
    atom_ptr : wp.array
        CSR-style pointer array.
    result : wp.array, optional
        Pre-allocated output array, shape (num_systems,).
    device : str, optional
        Warp device.

    Returns
    -------
    result : wp.array, shape (num_systems,), dtype=float32 or float64
        Per-system RMS vector norm.

    Notes
    -----
    Only atom_ptr mode is supported for true RMS computation because it
    requires knowing the atom count per system for the mean calculation.
    """
    if device is None:
        device = vectors.device

    vec_dtype = vectors.dtype
    scalar_dtype = wp.float32 if vec_dtype == wp.vec3f else wp.float64
    n_sys = atom_ptr.shape[0] - 1

    if result is None:
        result = wp.zeros(n_sys, dtype=scalar_dtype, device=device)

    wp.launch(
        _rms_norm_per_system_ptr_overloads[vec_dtype],
        dim=n_sys,
        inputs=[vectors, atom_ptr],
        outputs=[result],
        device=device,
    )

    return result


# =============================================================================
# Public API: Broadcasting
# =============================================================================


def broadcast_to_atoms(
    per_system_values: wp.array,
    batch_idx: wp.array = None,
    atom_ptr: wp.array = None,
    total_atoms: int = None,
    per_atom_values: wp.array = None,
    device: str = None,
) -> wp.array:
    """Broadcast per-system values to per-atom array.

    Parameters
    ----------
    per_system_values : wp.array, shape (num_systems,)
        Values for each system.
    batch_idx : wp.array, optional
        System index for each atom.
    atom_ptr : wp.array, optional
        CSR-style pointer array.
    total_atoms : int, optional
        Total number of atoms. Required if using atom_ptr without
        per_atom_values (to avoid sync).
    per_atom_values : wp.array, optional
        Pre-allocated output array, shape (total_atoms,).
        If provided, total_atoms is inferred from its shape.
    device : str, optional
        Warp device.

    Returns
    -------
    per_atom_values : wp.array, shape (total_atoms,)
        Values broadcast to each atom.

    Example
    -------
    >>> temperatures = wp.array([300.0, 350.0, 400.0], dtype=wp.float64, device="cuda:0")
    >>> per_atom_temps = broadcast_to_atoms(temperatures, batch_idx=batch_idx, device="cuda:0")
    """
    if batch_idx is None and atom_ptr is None:
        raise ValueError("Either batch_idx or atom_ptr must be provided")

    if device is None:
        device = per_system_values.device

    dtype = per_system_values.dtype

    if batch_idx is not None:
        n_atoms = batch_idx.shape[0]
        if per_atom_values is None:
            per_atom_values = wp.zeros(n_atoms, dtype=dtype, device=device)
        wp.launch(
            _broadcast_to_atoms_batch_idx_overloads[dtype],
            dim=n_atoms,
            inputs=[per_system_values, batch_idx],
            outputs=[per_atom_values],
            device=device,
        )
    else:
        n_sys = atom_ptr.shape[0] - 1

        # Determine total_atoms
        if per_atom_values is not None:
            total_atoms = per_atom_values.shape[0]
        elif total_atoms is None:
            # Fallback: sync to get total_atoms (avoid if possible)
            total_atoms = int(atom_ptr.numpy()[-1])

        if per_atom_values is None:
            per_atom_values = wp.zeros(total_atoms, dtype=dtype, device=device)

        wp.launch(
            _broadcast_to_atoms_ptr_overloads[dtype],
            dim=n_sys,
            inputs=[per_system_values, atom_ptr],
            outputs=[per_atom_values],
            device=device,
        )

    return per_atom_values
