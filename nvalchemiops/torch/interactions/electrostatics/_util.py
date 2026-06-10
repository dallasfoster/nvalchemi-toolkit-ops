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

"""Shared utilities for electrostatics PyTorch bindings."""

from __future__ import annotations

import torch

__all__ = [
    "_InjectChargeGrad",
    "_build_electrostatic_result",
    "_combine_electrostatic_outputs",
    "_energy_cotangents",
    "_is_per_system_uniform_cotangent",
    "_is_sync_free_uniform_cotangent",
    "_sum_charge_gradients",
    "_unpack_electrostatic_outputs",
]


def _is_sync_free_uniform_cotangent(grad_energy: torch.Tensor) -> bool:
    """Return whether ``grad_energy`` is known uniform without reading values.

    A scalar, or a tensor broadcast from one element (every non-trivial axis
    has stride 0), is uniform by construction — detectable without a device
    sync. Used to fast-path the common ``energy.sum()`` cotangent.
    """
    if grad_energy.numel() <= 1:
        return True
    return all(
        size <= 1 or stride == 0
        for size, stride in zip(grad_energy.shape, grad_energy.stride(), strict=True)
    )


def _energy_cotangents(
    grad_energy: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_atoms: int,
    num_systems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a per-atom energy cotangent into per-system + per-atom parts.

    Ported from the PR #96 energy-derivative contract. The returned
    ``grad_system`` ``(num_systems,)`` weights collective per-system quantities
    (cell-grad / reciprocal), and ``atom_grad`` ``(num_atoms,)`` weights
    per-atom quantities. Accepts ``grad_energy`` shaped per-system
    ``(num_systems,)``, per-atom ``(num_atoms,)``, or scalar; under the
    sum-reduction contract (``energy.sum()``) the cotangent is uniform so both
    reduce to the (broadcast) loss scale.
    """
    grad = grad_energy.reshape(-1)
    if batch_idx is None:
        if num_systems > 1 and grad.numel() == num_systems:
            grad_system = grad
        elif num_systems > 1 and grad.numel() == 1:
            grad_system = grad.expand(num_systems)
        elif grad.numel() == num_atoms and num_atoms > 0:
            grad_system = grad.mean().reshape(1)
        elif grad.numel() == 0:
            grad_system = grad.new_zeros((1,))
        else:
            grad_system = grad.reshape(-1)[:1]
        atom_grad = grad_system[0].expand(num_atoms)
        return grad_system, atom_grad

    if grad.numel() == num_systems:
        grad_system = grad
    elif grad.numel() == num_atoms:
        sums = grad.new_zeros((num_systems,)).index_add(0, batch_idx, grad)
        counts = grad.new_zeros((num_systems,)).index_add(
            0, batch_idx, torch.ones_like(grad)
        )
        grad_system = sums / counts.clamp_min(1)
    elif grad.numel() == 1:
        grad_system = grad.expand(num_systems)
    else:
        raise RuntimeError(
            "Energy cotangent must be per-system, per-atom, or scalar; got "
            f"{tuple(grad_energy.shape)} for {num_atoms} atoms / {num_systems} systems"
        )
    atom_grad = grad_system.index_select(0, batch_idx)
    return grad_system, atom_grad


def _is_per_system_uniform_cotangent(
    grad_energy: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_systems: int,
) -> bool:
    """Return whether an atom-major cotangent is uniform within each system.

    The PR #96 energy-derivative contract supports per-system-uniform energy
    cotangents (e.g. ``energy.sum()``). Returns ``True`` cheaply when uniformity
    is provable without reading values; otherwise reads (a device sync on CUDA).
    """
    if _is_sync_free_uniform_cotangent(grad_energy):
        return True
    grad = grad_energy.reshape(-1)
    if grad.numel() == 0:
        return True
    if grad.numel() == 1 or grad.numel() == num_systems:
        return True
    if grad.is_cuda:
        return False
    if batch_idx is None:
        return bool(torch.all(grad == grad[0]).item())
    if grad.numel() != batch_idx.numel():
        return False
    idx = batch_idx.to(device=grad.device, dtype=torch.long)
    grad64 = grad.to(torch.float64)
    sys_min = torch.full(
        (num_systems,), float("inf"), dtype=torch.float64, device=grad.device
    ).scatter_reduce(0, idx, grad64, reduce="amin", include_self=False)
    sys_max = torch.full(
        (num_systems,), float("-inf"), dtype=torch.float64, device=grad.device
    ).scatter_reduce(0, idx, grad64, reduce="amax", include_self=False)
    return bool(
        torch.all(sys_min.index_select(0, idx) == sys_max.index_select(0, idx)).item()
    )


@torch.compiler.disable
def _sum_charge_gradients(
    real_space_charge_grads: torch.Tensor,
    reciprocal_charge_grads: torch.Tensor,
) -> torch.Tensor:
    """Sum electrostatic charge gradients eagerly on compiled paths."""
    return real_space_charge_grads + reciprocal_charge_grads


def _build_electrostatic_result(
    energies: torch.Tensor,
    forces: torch.Tensor | None,
    charge_grads: torch.Tensor | None,
    virial: torch.Tensor | None,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Build an output tuple in electrostatics API order."""
    result = [energies]
    if compute_forces and forces is not None:
        result.append(forces)
    if compute_charge_gradients and charge_grads is not None:
        result.append(charge_grads)
    if compute_virial and virial is not None:
        result.append(virial)
    return tuple(result) if len(result) > 1 else result[0]


def _unpack_electrostatic_outputs(
    outputs: torch.Tensor | tuple[torch.Tensor, ...],
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Unpack electrostatics outputs by flag combination without cursor logic."""
    output_tuple = outputs if isinstance(outputs, tuple) else (outputs,)

    if compute_forces and compute_charge_gradients and compute_virial:
        energies, forces, charge_grads, virial = output_tuple
    elif compute_forces and compute_charge_gradients:
        energies, forces, charge_grads = output_tuple
        virial = None
    elif compute_forces and compute_virial:
        energies, forces, virial = output_tuple
        charge_grads = None
    elif compute_charge_gradients and compute_virial:
        energies, charge_grads, virial = output_tuple
        forces = None
    elif compute_forces:
        energies, forces = output_tuple
        charge_grads = None
        virial = None
    elif compute_charge_gradients:
        energies, charge_grads = output_tuple
        forces = None
        virial = None
    elif compute_virial:
        energies, virial = output_tuple
        forces = None
        charge_grads = None
    else:
        (energies,) = output_tuple
        forces = None
        charge_grads = None
        virial = None

    return energies, forces, charge_grads, virial


def _combine_electrostatic_outputs(
    real_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    reciprocal_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    slab_outputs: torch.Tensor | tuple[torch.Tensor, ...] | None,
    compute_forces: bool,
    compute_charge_gradients: bool,
    compute_virial: bool,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Combine real, reciprocal, and optional slab outputs by named fields."""
    real_energies, real_forces, real_charge_grads, real_virial = (
        _unpack_electrostatic_outputs(
            real_outputs,
            compute_forces,
            compute_charge_gradients,
            compute_virial,
        )
    )
    (
        reciprocal_energies,
        reciprocal_forces,
        reciprocal_charge_grads,
        reciprocal_virial,
    ) = _unpack_electrostatic_outputs(
        reciprocal_outputs,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )

    energies = real_energies + reciprocal_energies
    forces = (
        real_forces + reciprocal_forces
        if compute_forces and real_forces is not None and reciprocal_forces is not None
        else None
    )

    if (
        compute_charge_gradients
        and real_charge_grads is not None
        and reciprocal_charge_grads is not None
    ):
        if torch.compiler.is_compiling():
            charge_grads = _sum_charge_gradients(
                real_charge_grads, reciprocal_charge_grads
            )
        else:
            charge_grads = real_charge_grads + reciprocal_charge_grads
    else:
        charge_grads = None

    virial = (
        real_virial + reciprocal_virial
        if compute_virial and real_virial is not None and reciprocal_virial is not None
        else None
    )

    if slab_outputs is not None:
        slab_energies, slab_forces, slab_charge_grads, slab_virial = (
            _unpack_electrostatic_outputs(
                slab_outputs,
                compute_forces,
                compute_charge_gradients,
                compute_virial,
            )
        )
        energies = energies + slab_energies
        if compute_forces and forces is not None and slab_forces is not None:
            forces = forces + slab_forces
        if (
            compute_charge_gradients
            and charge_grads is not None
            and slab_charge_grads is not None
        ):
            charge_grads = charge_grads + slab_charge_grads
        if compute_virial and virial is not None and slab_virial is not None:
            virial = virial + slab_virial

    return _build_electrostatic_result(
        energies,
        forces,
        charge_grads,
        virial,
        compute_forces,
        compute_charge_gradients,
        compute_virial,
    )


class _InjectChargeGrad(torch.autograd.Function):
    """Inject analytical charge gradients into the autograd graph.

    A no-op in the forward pass (returns ``energy`` unchanged).  On backward,
    maps the per-system ``grad_energy`` to per-atom contributions using
    ``batch_idx`` and multiplies by the kernel-computed ``charge_grad``
    (dE/dq), so that ``energy.backward()`` propagates correct gradients
    through the charge pathway without a Warp backward tape.

    Parameters
    ----------
    energy : torch.Tensor
        Per-system energies, shape ``(S,)``.
    charges : torch.Tensor
        Charges with ``requires_grad=True``, shape ``(N,)``.
    charge_grad : torch.Tensor
        Analytical per-atom dE/dq from the forward kernel, shape ``(N,)``.
    batch_idx : torch.Tensor or None
        Per-atom system index, shape ``(N,)``.  ``None`` for single-system.
    """

    @staticmethod
    def forward(energy, charges, charge_grad, batch_idx):
        """Return energy unchanged."""
        return energy

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save charge_grad and batch_idx for backward."""
        _, _, charge_grad, batch_idx = inputs
        ctx.save_for_backward(charge_grad)
        ctx.batch_idx = batch_idx

    @staticmethod
    def backward(ctx, grad_energy):
        """Compute gradients for energy and charges."""
        (charge_grad,) = ctx.saved_tensors
        if ctx.batch_idx is not None:
            atom_grad = grad_energy.index_select(0, ctx.batch_idx)
        else:
            atom_grad = grad_energy.squeeze(0)
        return grad_energy, charge_grad * atom_grad, None, None
