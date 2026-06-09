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

r"""
Electrostatics interactions.

GPU-accelerated, framework-agnostic Warp kernel launchers for computing
long-range electrostatic interactions in molecular simulations. For PyTorch
bindings, see ``nvalchemiops.torch.interactions.electrostatics``.

Available methods:

- **Coulomb** (`coulomb`): direct and damped (erfc) Coulomb energy/forces.
- **Ewald summation** (`ewald`): real-space + reciprocal-space split,
  :math:`O(N^2)` for explicit k-vectors, full autograd support.
- **Particle Mesh Ewald** (`pme`): FFT-based :math:`O(N \log N)` method using
  B-spline charge assignment, full autograd support.
- **Damped Shifted Force** (`dsf`): pairwise :math:`O(N)` summation where both
  potential and forces vanish smoothly at the cutoff; supports
  geometry-dependent charges.
- **Slab correction** (`slab_kernels`): Yeh-Berkowitz / Ballenegger correction
  for 2D-periodic slabs; orthogonal and triclinic cells via projected slab
  normals. PyTorch ``compute_slab_correction()`` and
  ``ewald_summation(..., slab_correction=True)``.
- **Multipole electrostatics** (`multipole_*`): charge/dipole/quadrupole
  (l = 0, 1, 2) Ewald, PME, and electrostatic feature extraction.
"""

# Coulomb
from nvalchemiops.interactions.electrostatics.coulomb import (
    coulomb_energy,
    coulomb_energy_forces,
)

# Ewald
# Multipole direct k-space
from nvalchemiops.interactions.electrostatics.multipole_direct_kspace_kernels import (
    batch_apply_per_k_factor,
    batch_assemble_rho_k_dipole,
    batch_build_structure_factor_table,
    batch_compute_energy_product_per_k,
    batch_eval_gto_fourier_dipole,
    batch_eval_receiver_gto_fourier_dipole,
    batch_position_gradient_from_feature_grad,
    batch_position_gradient_from_rhok,
    batch_project_features_dipole,
    batch_v_gradient_from_feature_grad,
)

# Multipole Ewald
# PME
from nvalchemiops.interactions.electrostatics.pme_kernels import (
    pme_energy_corrections,
    pme_energy_corrections_with_charge_grad,
    pme_green_structure_factor,
)

# Slab correction - Warp launchers (framework-agnostic)
from nvalchemiops.interactions.electrostatics.slab_kernels import (
    slab_correction,
)

# DSF - Warp launchers (framework-agnostic)
from .dsf import (
    dsf_csr,
    dsf_matrix,
)

__all__ = [
    "dsf_csr",
    "dsf_matrix",
    "coulomb_energy",
    "coulomb_energy_forces",
    "slab_correction",
    "pme_green_structure_factor",
    "pme_energy_corrections",
    "pme_energy_corrections_with_charge_grad",
    "ewald_real_space",
    "ewald_reciprocal_space",
    "ewald_summation",
    "compute_slab_correction",
    "particle_mesh_ewald",
    "pme_reciprocal_space",
]

# Deprecated PyTorch functions - lazy import with warning
# These functions have moved to nvalchemiops.torch.interactions.electrostatics
_DEPRECATED_TORCH_EXPORTS = {
    # Ewald
    "ewald_real_space": "nvalchemiops.torch.interactions.electrostatics.ewald",
    "ewald_reciprocal_space": "nvalchemiops.torch.interactions.electrostatics.ewald",
    "ewald_summation": "nvalchemiops.torch.interactions.electrostatics.ewald",
    # PME
    "particle_mesh_ewald": "nvalchemiops.torch.interactions.electrostatics.pme",
    "pme_reciprocal_space": "nvalchemiops.torch.interactions.electrostatics.pme",
    "pme_green_structure_factor": "nvalchemiops.torch.interactions.electrostatics.pme",
    "pme_energy_corrections": "nvalchemiops.torch.interactions.electrostatics.pme",
    "pme_energy_corrections_with_charge_grad": "nvalchemiops.torch.interactions.electrostatics.pme",
    # Slab correction
    "compute_slab_correction": "nvalchemiops.torch.interactions.electrostatics.slab",
    # K-vectors
    "generate_k_vectors_ewald_summation": "nvalchemiops.torch.interactions.electrostatics.k_vectors",
    "generate_k_vectors_pme": "nvalchemiops.torch.interactions.electrostatics.k_vectors",
    # Parameters
    "estimate_ewald_parameters": "nvalchemiops.torch.interactions.electrostatics.parameters",
    "estimate_pme_parameters": "nvalchemiops.torch.interactions.electrostatics.parameters",
    "estimate_pme_mesh_dimensions": "nvalchemiops.torch.interactions.electrostatics.parameters",
    "mesh_spacing_to_dimensions": "nvalchemiops.torch.interactions.electrostatics.parameters",
    "EwaldParameters": "nvalchemiops.torch.interactions.electrostatics.parameters",
    "PMEParameters": "nvalchemiops.torch.interactions.electrostatics.parameters",
}


def __getattr__(name: str):
    """Lazy import with deprecation warning for PyTorch functions."""
    if name in _DEPRECATED_TORCH_EXPORTS:
        import importlib
        import warnings

        module_path = _DEPRECATED_TORCH_EXPORTS[name]
        warnings.warn(
            f"Importing '{name}' from 'nvalchemiops.interactions.electrostatics' is deprecated. "
            f"Please import from '{module_path}' instead. "
            "This will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
