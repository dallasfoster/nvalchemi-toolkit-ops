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
"""

# Coulomb
from nvalchemiops.interactions.electrostatics.coulomb import (
    batch_coulomb_energy,
    batch_coulomb_energy_forces,
    batch_coulomb_energy_forces_matrix,
    batch_coulomb_energy_matrix,
    coulomb_energy,
    coulomb_energy_forces,
    coulomb_energy_forces_matrix,
    coulomb_energy_matrix,
)

# Ewald
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    batch_ewald_real_space_energy,
    batch_ewald_real_space_energy_forces,
    batch_ewald_real_space_energy_forces_charge_grad,
    batch_ewald_real_space_energy_forces_charge_grad_matrix,
    batch_ewald_real_space_energy_forces_matrix,
    batch_ewald_real_space_energy_matrix,
    batch_ewald_reciprocal_space_compute_energy,
    batch_ewald_reciprocal_space_energy_forces,
    batch_ewald_reciprocal_space_energy_forces_charge_grad,
    batch_ewald_reciprocal_space_fill_structure_factors,
    batch_ewald_subtract_self_energy,
    ewald_real_space_energy,
    ewald_real_space_energy_forces,
    ewald_real_space_energy_forces_charge_grad,
    ewald_real_space_energy_forces_charge_grad_matrix,
    ewald_real_space_energy_forces_matrix,
    ewald_real_space_energy_matrix,
    ewald_reciprocal_space_compute_energy,
    ewald_reciprocal_space_energy_forces,
    ewald_reciprocal_space_energy_forces_charge_grad,
    ewald_reciprocal_space_fill_structure_factors,
    ewald_subtract_self_energy,
)

# Multipole direct k-space
from nvalchemiops.interactions.electrostatics.multipole_direct_kspace_kernels import (
    apply_per_k_factor,
    assemble_rho_k_dipole,
    batch_apply_per_k_factor,
    batch_assemble_rho_k_dipole,
    batch_build_structure_factor_table,
    batch_compute_energy_product_per_k,
    batch_eval_gto_fourier_dipole,
    batch_eval_receiver_gto_fourier_dipole,
    batch_eval_receiver_gto_fourier_quadrupole,
    batch_feat_position_grad_backward_grad_raw,
    batch_feat_position_grad_backward_grad_raw_quadrupole,
    batch_feat_position_grad_backward_positions,
    batch_feat_position_grad_backward_positions_quadrupole,
    batch_feat_position_grad_backward_v,
    batch_feat_position_grad_backward_v_quadrupole,
    batch_position_gradient_from_feature_grad,
    batch_position_gradient_from_feature_grad_quadrupole,
    batch_position_gradient_from_rhok,
    batch_project_features_dipole,
    batch_project_features_quadrupole,
    batch_project_kphase_grad,
    batch_project_phihat_grad,
    batch_receiver_phi_hat_backward_dipole,
    batch_receiver_phi_hat_backward_quadrupole,
    batch_rhok_position_grad_backward_grad_rho,
    batch_rhok_position_grad_backward_moments,
    batch_rhok_position_grad_backward_positions,
    batch_source_phi_hat_backward_dipole,
    batch_v_grad_from_feat_grad_backward_positions,
    batch_v_grad_from_feat_grad_backward_positions_quadrupole,
    batch_v_gradient_from_feature_grad,
    batch_v_gradient_from_feature_grad_quadrupole,
    build_structure_factor_table,
    compute_energy_product_per_k,
    eval_gto_fourier_dipole,
    eval_receiver_gto_fourier_dipole,
    eval_receiver_gto_fourier_quadrupole,
    feat_position_grad_backward_grad_raw,
    feat_position_grad_backward_grad_raw_quadrupole,
    feat_position_grad_backward_positions,
    feat_position_grad_backward_positions_quadrupole,
    feat_position_grad_backward_v,
    feat_position_grad_backward_v_quadrupole,
    position_gradient_from_feature_grad,
    position_gradient_from_feature_grad_quadrupole,
    position_gradient_from_rhok,
    project_features_dipole,
    project_features_quadrupole,
    project_kphase_grad_dipole,
    project_phihat_grad_dipole,
    receiver_phi_hat_backward_dipole,
    receiver_phi_hat_backward_quadrupole,
    rhok_position_grad_backward_grad_rho,
    rhok_position_grad_backward_moments,
    rhok_position_grad_backward_positions,
    source_phi_hat_backward_dipole,
    v_grad_from_feat_grad_backward_positions,
    v_grad_from_feat_grad_backward_positions_quadrupole,
    v_gradient_from_feature_grad,
    v_gradient_from_feature_grad_quadrupole,
)

# Multipole Ewald
from nvalchemiops.interactions.electrostatics.multipole_ewald_kernels import (
    batch_multipole_real_space_dipole_csr_energy,
    batch_multipole_real_space_dipole_csr_energy_2nd_backward,
    batch_multipole_real_space_dipole_csr_energy_backward,
    batch_multipole_real_space_dipole_csr_energy_fused,
    batch_multipole_real_space_monopole_csr_energy,
    batch_multipole_real_space_monopole_csr_energy_2nd_backward,
    batch_multipole_real_space_monopole_csr_energy_backward,
    batch_multipole_real_space_monopole_csr_energy_fused,
    multipole_real_space_dipole_csr_energy,
    multipole_real_space_dipole_csr_energy_2nd_backward,
    multipole_real_space_dipole_csr_energy_backward,
    multipole_real_space_dipole_csr_energy_fused,
    multipole_real_space_monopole_csr_energy,
    multipole_real_space_monopole_csr_energy_2nd_backward,
    multipole_real_space_monopole_csr_energy_backward,
    multipole_real_space_monopole_csr_energy_fused,
)

# PME
from nvalchemiops.interactions.electrostatics.pme_kernels import (
    batch_pme_energy_corrections,
    batch_pme_energy_corrections_with_charge_grad,
    batch_pme_green_structure_factor,
    pme_energy_corrections,
    pme_energy_corrections_with_charge_grad,
    pme_green_structure_factor,
)

# DSF
from .dsf import (
    dsf_csr,
    dsf_matrix,
)

__all__ = [
    # DSF
    "dsf_csr",
    "dsf_matrix",
    # Coulomb
    "coulomb_energy",
    "coulomb_energy_forces",
    "coulomb_energy_matrix",
    "coulomb_energy_forces_matrix",
    "batch_coulomb_energy",
    "batch_coulomb_energy_forces",
    "batch_coulomb_energy_matrix",
    "batch_coulomb_energy_forces_matrix",
    # Ewald (real-space)
    "ewald_real_space_energy",
    "ewald_real_space_energy_forces",
    "ewald_real_space_energy_matrix",
    "ewald_real_space_energy_forces_matrix",
    "ewald_real_space_energy_forces_charge_grad",
    "ewald_real_space_energy_forces_charge_grad_matrix",
    "batch_ewald_real_space_energy",
    "batch_ewald_real_space_energy_forces",
    "batch_ewald_real_space_energy_matrix",
    "batch_ewald_real_space_energy_forces_matrix",
    "batch_ewald_real_space_energy_forces_charge_grad",
    "batch_ewald_real_space_energy_forces_charge_grad_matrix",
    # Ewald (reciprocal-space)
    "ewald_reciprocal_space_fill_structure_factors",
    "ewald_reciprocal_space_compute_energy",
    "ewald_subtract_self_energy",
    "ewald_reciprocal_space_energy_forces",
    "ewald_reciprocal_space_energy_forces_charge_grad",
    "batch_ewald_reciprocal_space_fill_structure_factors",
    "batch_ewald_reciprocal_space_compute_energy",
    "batch_ewald_subtract_self_energy",
    "batch_ewald_reciprocal_space_energy_forces",
    "batch_ewald_reciprocal_space_energy_forces_charge_grad",
    # Multipole Ewald
    "multipole_real_space_monopole_csr_energy",
    "multipole_real_space_monopole_csr_energy_backward",
    "multipole_real_space_monopole_csr_energy_2nd_backward",
    "multipole_real_space_monopole_csr_energy_fused",
    "multipole_real_space_dipole_csr_energy",
    "multipole_real_space_dipole_csr_energy_backward",
    "multipole_real_space_dipole_csr_energy_2nd_backward",
    "multipole_real_space_dipole_csr_energy_fused",
    "batch_multipole_real_space_monopole_csr_energy",
    "batch_multipole_real_space_monopole_csr_energy_backward",
    "batch_multipole_real_space_monopole_csr_energy_2nd_backward",
    "batch_multipole_real_space_monopole_csr_energy_fused",
    "batch_multipole_real_space_dipole_csr_energy",
    "batch_multipole_real_space_dipole_csr_energy_backward",
    "batch_multipole_real_space_dipole_csr_energy_2nd_backward",
    "batch_multipole_real_space_dipole_csr_energy_fused",
    # Multipole direct k-space
    "build_structure_factor_table",
    "eval_gto_fourier_dipole",
    "assemble_rho_k_dipole",
    "apply_per_k_factor",
    "compute_energy_product_per_k",
    "eval_receiver_gto_fourier_dipole",
    "eval_receiver_gto_fourier_quadrupole",
    "position_gradient_from_rhok",
    "project_features_dipole",
    "v_gradient_from_feature_grad",
    "position_gradient_from_feature_grad",
    "project_phihat_grad_dipole",
    "project_kphase_grad_dipole",
    "batch_project_phihat_grad",
    "batch_project_kphase_grad",
    # l=2 feature projection
    "project_features_quadrupole",
    "batch_project_features_quadrupole",
    "v_gradient_from_feature_grad_quadrupole",
    "batch_v_gradient_from_feature_grad_quadrupole",
    "position_gradient_from_feature_grad_quadrupole",
    "batch_position_gradient_from_feature_grad_quadrupole",
    # l=2 feature second-order (force-loss / create_graph)
    "feat_position_grad_backward_grad_raw_quadrupole",
    "feat_position_grad_backward_v_quadrupole",
    "feat_position_grad_backward_positions_quadrupole",
    "v_grad_from_feat_grad_backward_positions_quadrupole",
    "batch_feat_position_grad_backward_grad_raw_quadrupole",
    "batch_feat_position_grad_backward_v_quadrupole",
    "batch_feat_position_grad_backward_positions_quadrupole",
    "batch_v_grad_from_feat_grad_backward_positions_quadrupole",
    # Multipole direct k-space - second-order backward kernels
    "source_phi_hat_backward_dipole",
    "receiver_phi_hat_backward_dipole",
    "receiver_phi_hat_backward_quadrupole",
    "batch_eval_receiver_gto_fourier_quadrupole",
    "batch_receiver_phi_hat_backward_quadrupole",
    "rhok_position_grad_backward_grad_rho",
    "rhok_position_grad_backward_moments",
    "rhok_position_grad_backward_positions",
    "feat_position_grad_backward_grad_raw",
    "feat_position_grad_backward_v",
    "feat_position_grad_backward_positions",
    "v_grad_from_feat_grad_backward_positions",
    # Multipole direct k-space - batched K-family
    "batch_source_phi_hat_backward_dipole",
    "batch_receiver_phi_hat_backward_dipole",
    "batch_rhok_position_grad_backward_grad_rho",
    "batch_rhok_position_grad_backward_moments",
    "batch_rhok_position_grad_backward_positions",
    "batch_feat_position_grad_backward_grad_raw",
    "batch_feat_position_grad_backward_v",
    "batch_feat_position_grad_backward_positions",
    "batch_v_grad_from_feat_grad_backward_positions",
    # PME
    "pme_green_structure_factor",
    "batch_pme_green_structure_factor",
    "pme_energy_corrections",
    "batch_pme_energy_corrections",
    "pme_energy_corrections_with_charge_grad",
    "batch_pme_energy_corrections_with_charge_grad",
    # Ewald - PyTorch bindings (deprecated, use nvalchemiops.torch.interactions.electrostatics)
    "ewald_real_space",
    "ewald_reciprocal_space",
    "ewald_summation",
    # PME - PyTorch bindings (deprecated, use nvalchemiops.torch.interactions.electrostatics)
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
