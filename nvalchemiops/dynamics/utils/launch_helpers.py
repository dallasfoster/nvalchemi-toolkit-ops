# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Launch Helpers for Dynamics Kernels
====================================

Host-side helpers for kernel dispatch, overload registration, and
output validation shared by integrators and (in future) optimizers.

These helpers do **not** generate or modify Warp kernels at runtime.
All overloads are registered eagerly at module import time.

Design Rule
-----------
Warp interface functions must receive ALL arrays pre-allocated.
Output arrays are required, not optional.  Zero-init is the caller's
responsibility.  Only Torch interface functions may accept optional
arguments and allocate/zero as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import warp as wp


# =============================================================================
# Types
# =============================================================================


class ExecutionMode(Enum):
    """Execution mode for batched vs single-system kernel dispatch.

    Three modes are supported:

    - ``SINGLE``: One system, no batch metadata. Scalar arrays have shape ``(1,)``.
    - ``BATCH_IDX``: Multiple systems, each atom tagged with a system index.
      Kernel launches with ``dim=num_atoms``.
    - ``ATOM_PTR``: Multiple systems, CSR-style pointer array defines atom ranges.
      Kernel launches with ``dim=num_systems``.
    """

    SINGLE = "single"
    BATCH_IDX = "batch_idx"
    ATOM_PTR = "atom_ptr"


@dataclass(frozen=True)
class KernelFamily:
    """Container for per-mode kernel overloads.

    Groups the three execution-mode variants of a single logical operation
    so that dispatch helpers can select the correct kernel by mode.

    Parameters
    ----------
    single : object
        Kernel (or overloaded kernel) for single-system mode.
    batch_idx : object or None
        Kernel for ``batch_idx`` mode. None if not supported.
    atom_ptr : object or None
        Kernel for ``atom_ptr`` (CSR pointer) mode. None if not supported.
    """

    single: object
    batch_idx: object | None = None
    atom_ptr: object | None = None


# =============================================================================
# Dispatch
# =============================================================================


def resolve_execution_mode(
    batch_idx: wp.array | None,
    atom_ptr: wp.array | None,
) -> ExecutionMode:
    """Determine execution mode from optional batch metadata arrays.

    Parameters
    ----------
    batch_idx : wp.array or None
        Per-atom system index array (``batch_idx`` mode).
    atom_ptr : wp.array or None
        CSR-style atom pointer array (``atom_ptr`` mode).

    Returns
    -------
    ExecutionMode
        The resolved execution mode.

    Raises
    ------
    ValueError
        If both ``batch_idx`` and ``atom_ptr`` are provided.
    """
    if batch_idx is not None and atom_ptr is not None:
        raise ValueError("Provide batch_idx OR atom_ptr, not both")
    if atom_ptr is not None:
        return ExecutionMode.ATOM_PTR
    if batch_idx is not None:
        return ExecutionMode.BATCH_IDX
    return ExecutionMode.SINGLE


def launch_family(
    family: KernelFamily,
    *,
    mode: ExecutionMode,
    dim: int,
    inputs_single: list,
    inputs_batch: list | None = None,
    inputs_ptr: list | None = None,
    device: str,
) -> None:
    """Launch the correct kernel variant from a :class:`KernelFamily`.

    Selects the kernel matching *mode* and launches it with the
    appropriate inputs and grid dimension.

    Parameters
    ----------
    family : KernelFamily
        Container holding ``single``, ``batch_idx``, and ``atom_ptr``
        kernel variants.
    mode : ExecutionMode
        Resolved execution mode (from :func:`resolve_execution_mode`).
    dim : int
        Launch grid dimension.  For ``SINGLE`` and ``BATCH_IDX`` this is
        typically ``num_atoms``; for ``ATOM_PTR`` it is ``num_systems``.
    inputs_single : list
        Kernel input list for single-system mode.
    inputs_batch : list or None
        Kernel input list for ``batch_idx`` mode.
    inputs_ptr : list or None
        Kernel input list for ``atom_ptr`` mode.
    device : str
        Warp device string.

    Raises
    ------
    ValueError
        If the requested mode's kernel is ``None`` in the family.
    """
    if mode is ExecutionMode.ATOM_PTR:
        kernel = family.atom_ptr
        if kernel is None:
            raise ValueError("atom_ptr mode not supported for this kernel family")
        wp.launch(kernel, dim=dim, inputs=inputs_ptr, device=device)
    elif mode is ExecutionMode.BATCH_IDX:
        kernel = family.batch_idx
        if kernel is None:
            raise ValueError("batch_idx mode not supported for this kernel family")
        wp.launch(kernel, dim=dim, inputs=inputs_batch, device=device)
    else:
        wp.launch(family.single, dim=dim, inputs=inputs_single, device=device)


# =============================================================================
# Overload Registration
# =============================================================================

# Default dtype pairs: (vec3_type, scalar_type)
DEFAULT_DTYPE_PAIRS = (
    (wp.vec3f, wp.float32),
    (wp.vec3d, wp.float64),
)


def register_overloads(
    kernel,
    signature_builder: Callable,
    dtype_pairs: tuple[tuple, ...] = DEFAULT_DTYPE_PAIRS,
    *,
    key_fn: Callable | None = None,
) -> dict:
    """Register ``wp.overload`` variants for each dtype pair.

    Parameters
    ----------
    kernel : warp kernel
        The generic ``@wp.kernel`` function to overload.
    signature_builder : callable
        A function ``(vec_dtype, scalar_dtype) -> list`` that returns
        the overload argument type list for a given dtype pair.
    dtype_pairs : tuple of (vec_dtype, scalar_dtype) tuples
        Dtype pairs to register.  Defaults to ``(vec3f, float32)``
        and ``(vec3d, float64)``.
    key_fn : callable or None
        Optional function ``(vec_dtype, scalar_dtype) -> key`` that
        produces the dictionary key for each overload.  Defaults to
        using ``vec_dtype`` as the key.

    Returns
    -------
    dict
        Mapping from key (default: ``vec_dtype``) to the overloaded
        kernel handle.

    Examples
    --------
    Register single-system rescale kernel overloads::

        overloads = register_overloads(
            _velocity_rescale_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
        )
        # overloads[wp.vec3f] -> overloaded kernel for float32
        # overloads[wp.vec3d] -> overloaded kernel for float64

    Register with a composite key::

        overloads = register_overloads(
            _my_kernel,
            lambda v, t: [wp.array(dtype=v), wp.array(dtype=t)],
            key_fn=lambda v, t: (v, t),
        )
    """
    if key_fn is None:
        key_fn = lambda v, t: v  # noqa: E731

    out = {}
    for vec_dtype, scalar_dtype in dtype_pairs:
        sig = signature_builder(vec_dtype, scalar_dtype)
        key = key_fn(vec_dtype, scalar_dtype)
        out[key] = wp.overload(kernel, sig)
    return out


# =============================================================================
# Validation
# =============================================================================


def validate_out_array(
    out: wp.array,
    reference: wp.array,
    name: str,
) -> None:
    """Validate that an output array matches a reference in shape, dtype, device.

    Parameters
    ----------
    out : wp.array
        Output array to validate.
    reference : wp.array
        Reference array whose shape, dtype, and device are expected.
    name : str
        Human-readable name for error messages.

    Raises
    ------
    ValueError
        If shape, dtype, or device do not match.
    """
    if out.shape != reference.shape:
        raise ValueError(
            f"{name} shape mismatch: expected {reference.shape}, got {out.shape}"
        )
    if out.dtype != reference.dtype:
        raise ValueError(
            f"{name} dtype mismatch: expected {reference.dtype}, got {out.dtype}"
        )
    if str(out.device) != str(reference.device):
        raise ValueError(
            f"{name} device mismatch: expected {reference.device}, got {out.device}"
        )
