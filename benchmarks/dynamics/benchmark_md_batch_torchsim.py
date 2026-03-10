#!/usr/bin/env python3
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

"""
Torch-sim Comparison Benchmarks (Internal)
===========================================

Compare nvalchemiops batched MD kernels against torch-sim (community PyTorch MD).

This script tests three model configurations:
1. Native LJ: Each backend uses its own Lennard-Jones implementation
2. Common nvalchemiops LJ: Both use nvalchemiops LJ (isolates integrator performance)
3. MACE: Both use torch-sim MACE (real-world MLIP comparison)

NOT FOR PUBLIC RELEASE - Internal testing only

Usage
-----
    python benchmark_md_batch_torchsim.py --config benchmark_config_torchsim.yaml

Output
------
CSV file with extended schema (16 columns including model_type):
- dynamics_md_batch_comparison_<gpu_sku>.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import warp as wp
from shared_utils import (
    BenchmarkResult,
    NvalchemiOpsBenchmark,
    NvalchemiopsLJModel,
    NvalchemiopsModelInterface,
    get_gpu_sku,
    load_config,
    print_batch_benchmark_footer,
    print_batch_benchmark_header,
    print_batch_benchmark_result,
    write_results_csv,
)

# Import torch-sim with graceful fallback
try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.state import SimState
    from torch_sim.typing import StateDict

    TORCH_SIM_AVAILABLE = True
except ImportError:
    TORCH_SIM_AVAILABLE = False
    print("\nWARNING: torch-sim not installed. Only nvalchemiops benchmarks will run.")
    print("Install with: pip install torch-sim-atomistic")
    print(
        "Or from source: git clone https://github.com/TorchSim/torch-sim && cd torch-sim && pip install .\n"
    )

wp.init()
# ==============================================================================
# Model Wrappers and Adapters
# ==============================================================================


class ExternalModelWrapper(NvalchemiopsModelInterface):
    """Wrapper for external models (e.g., torch-sim MACE) to work with nvalchemiops benchmarks.

    This adapter converts between warp arrays and torch tensors, calls an external
    model that conforms to torch-sim's interface, and converts results back.

    Parameters
    ----------
    model : object
        External model with __call__(state) interface (e.g., torch-sim model)
    cell : torch.Tensor
        Unit cell matrix
    masses : torch.Tensor
        Atomic masses
    atomic_numbers : torch.Tensor
        Atomic numbers
    pbc : torch.Tensor
        Periodic boundary conditions
    batch_idx : torch.Tensor or None
        Batch index for each atom
    dtype : torch.dtype
        Data type
    """

    def __init__(
        self,
        model: Any,
        cell: torch.Tensor,
        masses: torch.Tensor,
        atomic_numbers: torch.Tensor,
        pbc: torch.Tensor,
        batch_idx: torch.Tensor | None,
        dtype: torch.dtype,
    ):
        self.model = model
        self.cell = cell
        self.masses = masses
        self.atomic_numbers = atomic_numbers
        self.pbc = pbc
        self.batch_idx = batch_idx
        self.dtype = dtype

        # Check if torch-sim is available and model is torch-sim model
        try:
            import torch_sim as ts

            self.ts_available = True
            self.SimState = ts.SimState
        except ImportError:
            self.ts_available = False
            raise RuntimeError("torch-sim is required for ExternalModelWrapper")

    def compute_forces(
        self,
        wp_positions: wp.array,
        neighbor_matrix: wp.array,
        num_neighbors: wp.array,
        neighbor_shifts: wp.array,
    ) -> tuple[wp.array, wp.array]:
        """Compute energies and forces using external model."""
        # Convert warp positions to torch
        torch_positions = wp.to_torch(wp_positions)

        # Create SimState for torch-sim model
        state = self.SimState(
            positions=torch_positions,
            masses=self.masses,
            cell=self.cell,
            atomic_numbers=self.atomic_numbers,
            pbc=self.pbc.tolist() if isinstance(self.pbc, torch.Tensor) else self.pbc,
            system_idx=self.batch_idx,
        )

        # Call model
        result = self.model(state)

        # Extract energies and forces
        torch_energies = result["energy"]  # Shape: (n_systems,) or scalar
        torch_forces = result["forces"]  # Shape: (n_atoms, 3)

        # Convert back to warp
        from nvalchemiops.types import get_wp_dtype, get_wp_vec_dtype

        wp_dtype = get_wp_dtype(self.dtype)
        wp_vec_dtype = get_wp_vec_dtype(self.dtype)

        # Energy needs to be per-atom for compatibility
        # torch-sim returns per-system energy, we need to replicate per atom
        if torch_energies.dim() == 0:  # Scalar
            torch_energies = torch_energies.unsqueeze(0)

        # Expand energies to per-atom (set all atoms in a system to same energy)
        if self.batch_idx is not None:
            n_atoms = torch_positions.shape[0]
            per_atom_energies = torch.zeros(
                n_atoms, dtype=self.dtype, device=torch_energies.device
            )
            per_atom_energies = torch_energies[self.batch_idx]
        else:
            per_atom_energies = torch_energies.expand(torch_positions.shape[0])

        wp_energies = wp.from_torch(per_atom_energies.contiguous(), dtype=wp_dtype)
        wp_forces = wp.from_torch(torch_forces.contiguous(), dtype=wp_vec_dtype)

        return wp_energies, wp_forces

    def compute_virial(
        self,
        wp_positions: wp.array,
        neighbor_matrix: wp.array,
        num_neighbors: wp.array,
        neighbor_shifts: wp.array,
    ) -> tuple[wp.array, wp.array, wp.array]:
        """Compute energies, forces, and virial using external model."""
        # For now, most external models don't compute virial
        # We could compute it numerically if needed, but for benchmarking
        # we typically don't need NPT with external models
        raise NotImplementedError(
            "Virial computation not supported for external models. "
            "Use native LJ model for NPT/NPH benchmarks."
        )


class NvalchemiopsLJForTorchSim(ModelInterface if TORCH_SIM_AVAILABLE else object):
    """Adapter to use nvalchemiops LJ implementation with torch-sim.

    Wraps nvalchemiops LJ kernels to conform to torch-sim's ModelInterface,
    allowing direct performance comparison of integrators with identical
    force calculations.

    Parameters
    ----------
    epsilon : float
        LJ epsilon parameter (eV)
    sigma : float
        LJ sigma parameter (Å)
    cutoff : float
        Cutoff distance (Å)
    skin : float
        Neighbor list skin distance (Å)
    device : torch.device
        Device for computation
    dtype : torch.dtype
        Data type for computation
    compute_forces : bool
        Whether to compute forces
    compute_stress : bool
        Whether to compute stress
    """

    def __init__(
        self,
        epsilon: float,
        sigma: float,
        cutoff: float,
        skin: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ):
        """Initialize nvalchemiops LJ model for torch-sim."""
        # Store parameters BEFORE calling super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.skin = skin
        self._device = device or torch.device("cuda")
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress

        # Now call parent init
        # Note: torch-sim's ModelInterface may not accept these parameters in __init__
        # so we just call super().__init__() without arguments and rely on our properties
        if TORCH_SIM_AVAILABLE:
            super().__init__()

        # Import nvalchemiops components
        from nvalchemiops.interactions.lj import lj_energy_forces
        from nvalchemiops.neighborlist import neighbor_list

        self.lj_energy_forces = lj_energy_forces
        self.neighbor_list_fn = neighbor_list

        # Neighbor list state (will be initialized on first call)
        self.nl_positions = None
        self.neighbor_matrix = None
        self.num_neighbors = None
        self.neighbor_shifts = None
        self.rebuild_counter = 0
        self.rebuild_interval = 10

    @property
    def device(self) -> torch.device:
        """Return device."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype."""
        return self._dtype

    @property
    def compute_forces(self) -> bool:
        """Return whether to compute forces."""
        return self._compute_forces

    @property
    def compute_stress(self) -> bool:
        """Return whether to compute stress."""
        return self._compute_stress

    def forward(self, state: SimState) -> StateDict:
        """Forward method required by torch-sim ModelInterface (alias for __call__)."""
        return self(state)

    def __call__(self, state: SimState) -> StateDict:
        """Compute energy and forces for a SimState.

        Parameters
        ----------
        state : SimState
            Input state with positions, cell, masses, etc.

        Returns
        -------
        StateDict
            Dictionary with 'energy' and 'forces' keys
        """
        # Check if neighbor list needs rebuild
        need_rebuild = (
            self.nl_positions is None
            or self.rebuild_counter >= self.rebuild_interval
            or self._max_displacement(state.positions) > self.skin / 2
        )

        if need_rebuild:
            self._rebuild_neighbor_list(state)
            self.rebuild_counter = 0
        else:
            self.rebuild_counter += 1

        # Convert torch tensors to warp arrays
        import warp as wp

        from nvalchemiops.types import get_wp_mat_dtype, get_wp_vec_dtype

        wp_vec_dtype = get_wp_vec_dtype(self.dtype)
        wp_mat_dtype = get_wp_mat_dtype(self.dtype)

        wp_positions = wp.from_torch(state.positions.contiguous(), dtype=wp_vec_dtype)

        # Get cell (handle both shapes)
        cell = state.cell[0] if state.cell.dim() == 3 else state.cell
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)

        # Get batch_idx if batched
        wp_batch_idx = None
        if state.system_idx is not None:
            wp_batch_idx = wp.from_torch(
                state.system_idx.to(torch.int32), dtype=wp.int32
            )

        # Compute LJ forces and energy using nvalchemiops kernels
        wp_energies, wp_forces = self.lj_energy_forces(
            positions=wp_positions,
            cell=wp_cell,
            epsilon=self.epsilon,
            sigma=self.sigma,
            cutoff=self.cutoff,
            neighbor_matrix=self.neighbor_matrix,
            neighbor_matrix_shifts=self.neighbor_shifts,
            num_neighbors=self.num_neighbors,
            fill_value=state.positions.shape[0],
            batch_idx=wp_batch_idx,
            device=str(self.device),
        )

        # Convert back to torch
        torch_energies = wp.to_torch(wp_energies)
        torch_forces = wp.to_torch(wp_forces)

        # Aggregate energies per system for torch-sim
        if state.system_idx is not None:
            # Sum energies per system
            n_systems = state.system_idx.max().item() + 1
            system_energies = torch.zeros(
                n_systems, dtype=self.dtype, device=self.device
            )
            system_energies.scatter_add_(0, state.system_idx, torch_energies)
        else:
            system_energies = torch_energies.sum().unsqueeze(0)

        result = {
            "energy": system_energies,
            "forces": torch_forces,
        }

        if self.compute_stress:
            # nvalchemiops stress computation if needed
            # For now, skip or implement if required
            pass

        return result

    def _rebuild_neighbor_list(self, state: SimState):
        """Rebuild neighbor list using nvalchemiops neighbor_list."""
        # Store current positions for displacement check
        self.nl_positions = state.positions.clone()

        # Build neighbor list
        # Get cell (handle both shapes)

        cell = state.cell

        # Determine batch parameters
        batch_idx = state.system_idx

        # Convert pbc to tensor if needed
        pbc_tensor = (
            state.pbc
            if isinstance(state.pbc, torch.Tensor)
            else torch.tensor(state.pbc, device=state.positions.device)
        )

        neighbor_matrix, num_neighbors, neighbor_shifts = self.neighbor_list_fn(
            positions=state.positions,
            cutoff=self.cutoff + self.skin,
            cell=cell,
            pbc=pbc_tensor,
            method="cell_list" if batch_idx is None else "batch_cell_list",
            batch_idx=batch_idx,
        )

        # Convert to warp arrays
        import warp as wp

        self.neighbor_matrix = wp.from_torch(
            neighbor_matrix.to(torch.int32), dtype=wp.int32
        )
        self.num_neighbors = wp.from_torch(
            num_neighbors.to(torch.int32), dtype=wp.int32
        )
        self.neighbor_shifts = wp.from_torch(
            neighbor_shifts.contiguous(), dtype=wp.vec3i
        )

    def _max_displacement(self, current_positions: torch.Tensor) -> float:
        """Calculate maximum atomic displacement since last neighbor list build."""
        if self.nl_positions is None:
            return float("inf")

        displacements = torch.norm(current_positions - self.nl_positions, dim=-1)
        return displacements.max().item()


def create_native_lj_models(
    epsilon: float,
    sigma: float,
    cutoff: float,
    skin: float,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[NvalchemiopsLJModel, Any]:
    """Create native LJ models for both backends.

    Parameters
    ----------
    epsilon : float
        LJ epsilon parameter
    sigma : float
        LJ sigma parameter
    cutoff : float
        Cutoff distance
    skin : float
        Neighbor list skin
    cell : torch.Tensor
        Unit cell matrix
    batch_idx : torch.Tensor
        Batch index for atoms
    device : torch.device
        Device for computation
    dtype : torch.dtype
        Data type

    Returns
    -------
    nvalchemiops_model : NvalchemiopsLJModel
        Native nvalchemiops LJ model wrapper
    torchsim_model : LennardJonesModel or None
        Native torch-sim LJ model (None if torch-sim not available)
    """
    # nvalchemiops native model wrapper
    nvalchemiops_model = NvalchemiopsLJModel(
        epsilon=epsilon,
        sigma=sigma,
        cutoff=cutoff,
        cell=cell,
        batch_idx=batch_idx,
        device=str(device),
        dtype=dtype,
    )

    # torch-sim native model
    torchsim_model = None
    if TORCH_SIM_AVAILABLE:
        from torch_sim.models.lennard_jones import LennardJonesModel

        torchsim_model = LennardJonesModel(
            sigma=sigma,
            epsilon=epsilon,
            cutoff=cutoff,
            device=device,
            dtype=dtype,
            compute_forces=True,
            compute_stress=False,
            use_neighbor_list=True,
        )

    return nvalchemiops_model, torchsim_model


def create_nvalchemiops_lj_models(
    epsilon: float,
    sigma: float,
    cutoff: float,
    skin: float,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[NvalchemiopsLJModel, Any]:
    """Create nvalchemiops LJ models wrapped for both backends.

    Parameters
    ----------
    epsilon : float
        LJ epsilon parameter
    sigma : float
        LJ sigma parameter
    cutoff : float
        Cutoff distance
    skin : float
        Neighbor list skin
    cell : torch.Tensor
        Unit cell matrix
    batch_idx : torch.Tensor
        Batch index for atoms
    device : torch.device
        Device for computation
    dtype : torch.dtype
        Data type

    Returns
    -------
    nvalchemiops_model : NvalchemiopsLJModel
        nvalchemiops LJ model wrapper (same as native)
    torchsim_model : NvalchemiopsLJForTorchSim or None
        Wrapped nvalchemiops LJ for torch-sim (None if torch-sim not available)
    """
    # nvalchemiops uses same native implementation
    nvalchemiops_model = NvalchemiopsLJModel(
        epsilon=epsilon,
        sigma=sigma,
        cutoff=cutoff,
        cell=cell,
        batch_idx=batch_idx,
        device=str(device),
        dtype=dtype,
    )

    # torch-sim uses wrapped nvalchemiops
    torchsim_model = None
    if TORCH_SIM_AVAILABLE:
        torchsim_model = NvalchemiopsLJForTorchSim(
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            skin=skin,
            device=device,
            dtype=dtype,
            compute_forces=True,
            compute_stress=False,
        )

    return nvalchemiops_model, torchsim_model


def create_mace_models(
    model_name: str,
    cell: torch.Tensor,
    masses: torch.Tensor,
    atomic_numbers: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[ExternalModelWrapper, Any]:
    """Create MACE models for both backends.

    Parameters
    ----------
    model_name : str
        MACE model variant ("small", "medium", "large")
    cell : torch.Tensor
        Unit cell matrix
    masses : torch.Tensor
        Atomic masses
    atomic_numbers : torch.Tensor
        Atomic numbers
    pbc : torch.Tensor
        Periodic boundary conditions
    batch_idx : torch.Tensor
        Batch index for atoms
    device : torch.device
        Device for computation
    dtype : torch.dtype
        Data type

    Returns
    -------
    nvalchemiops_model : ExternalModelWrapper
        Wrapper for torch-sim MACE to work with nvalchemiops
    torchsim_model : MaceModel or None
        torch-sim MACE model (None if torch-sim not available)
    """
    if not TORCH_SIM_AVAILABLE:
        raise RuntimeError("torch-sim is required for MACE benchmarks")

    from mace.calculators.foundations_models import mace_mp
    from torch_sim.models.mace import MaceModel

    # Load MACE model
    mace = mace_mp(
        model=model_name,
        return_raw_model=True,
        default_dtype=str(dtype).removeprefix("torch."),
        device=str(device),
    )

    # torch-sim MACE model
    torchsim_model = MaceModel(
        model=mace,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=False,
    )

    # For nvalchemiops, wrap the torch-sim model with ExternalModelWrapper
    nvalchemiops_model = ExternalModelWrapper(
        model=torchsim_model,
        cell=cell,
        masses=masses,
        atomic_numbers=atomic_numbers,
        pbc=pbc,
        batch_idx=batch_idx,
        dtype=dtype,
    )

    return nvalchemiops_model, torchsim_model


# ==============================================================================
# Torch-sim Benchmark Wrapper
# ==============================================================================


class TorchSimBenchmark:
    """Wrapper for torch-sim batched MD simulations.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions (total_atoms, 3)
    cell : torch.Tensor
        Cell matrix (batch_size, 3, 3) or (3, 3)
    masses : torch.Tensor
        Atomic masses (total_atoms,)
    velocities : torch.Tensor
        Initial velocities (total_atoms, 3)
    batch_idx : torch.Tensor
        Batch index for each atom (total_atoms,)
    atom_ptr : torch.Tensor
        Pointer array (batch_size + 1,)
    pbc : torch.Tensor
        Periodic boundary conditions (3,)
    model : ModelInterface
        torch-sim model for force/energy computation
    """

    def __init__(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        masses: torch.Tensor,
        velocities: torch.Tensor,
        batch_idx: torch.Tensor,
        atom_ptr: torch.Tensor,
        pbc: torch.Tensor,
        model: Any,
    ):
        """Initialize torch-sim benchmark with provided model."""
        if not TORCH_SIM_AVAILABLE:
            raise RuntimeError("torch-sim is not available")

        # torch-sim uses system_idx (equivalent to batch_idx)
        self.system_idx = batch_idx
        self.batch_size = atom_ptr.shape[0] - 1
        self.num_atoms_per_system = (atom_ptr[1] - atom_ptr[0]).item()

        # Create SimState
        # torch-sim expects cell shape (n_systems, 3, 3)
        if cell.dim() == 2:
            cell = cell.unsqueeze(0).expand(self.batch_size, -1, -1).contiguous()
        elif cell.shape[0] != self.batch_size:
            cell = cell.reshape(self.batch_size, 3, 3)

        # Atomic numbers (18 = Argon for LJ, will be updated for MACE)
        atomic_numbers = torch.full(
            (positions.shape[0],), 18, device=positions.device, dtype=torch.int
        )

        self.state = ts.SimState(
            positions=positions,
            masses=masses,
            cell=cell,
            atomic_numbers=atomic_numbers,
            pbc=pbc.tolist(),  # torch-sim expects list[bool]
            system_idx=self.system_idx,
        )

        # Set velocities (will be used for initialization)
        self.initial_velocities = velocities

        # Store model
        self.model = model

        self.device = positions.device
        self.dtype = positions.dtype

    def run_velocity_verlet(
        self, dt: float, num_steps: int, warmup_steps: int
    ) -> BenchmarkResult:
        """Run NVE velocity verlet MD.

        Parameters
        ----------
        dt : float
            Timestep in fs
        num_steps : int
            Number of MD steps
        warmup_steps : int
            Number of warmup steps

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and throughput metrics
        """
        # Initialize NVE integrator (kT only used for initial velocities)
        # We'll use existing velocities instead
        kT = torch.tensor(
            300.0 * 8.617333e-5, device=self.device, dtype=self.dtype
        )  # 300K in eV

        # Initialize state with velocities
        state = ts.nve_init(self.state, self.model, kT=kT, seed=42)

        # Override with our initial velocities (converted to momenta)
        state.momenta = self.initial_velocities * state.masses.unsqueeze(-1)

        dt_tensor = torch.tensor(dt, device=self.device, dtype=self.dtype)

        # Warmup
        for _ in range(warmup_steps):
            state = ts.nve_step(state, self.model, dt=dt_tensor)

        # Timed run
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_steps):
            state = ts.nve_step(state, self.model, dt=dt_tensor)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_time = end_time - start_time

        return BenchmarkResult(
            backend="torch-sim",
            name="velocity_verlet",
            num_atoms=self.num_atoms_per_system,
            ensemble="NVE",
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            batch_size=self.batch_size,
        )

    def run_langevin(
        self,
        dt: float,
        num_steps: int,
        temperature: float,
        friction: float,
        warmup_steps: int,
    ) -> BenchmarkResult:
        """Run NVT Langevin MD.

        Parameters
        ----------
        dt : float
            Timestep in fs
        num_steps : int
            Number of MD steps
        temperature : float
            Target temperature in K
        friction : float
            Friction coefficient in 1/fs
        warmup_steps : int
            Number of warmup steps

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and throughput metrics
        """
        # Convert temperature to eV
        kT = torch.tensor(
            temperature * 8.617333e-5, device=self.device, dtype=self.dtype
        )
        gamma = torch.tensor(friction, device=self.device, dtype=self.dtype)

        # Initialize Langevin integrator
        state = ts.nvt_langevin_init(self.state, self.model, kT=kT, seed=42)

        # Override with our initial velocities (converted to momenta)
        state.momenta = self.initial_velocities * state.masses.unsqueeze(-1)

        dt_tensor = torch.tensor(dt, device=self.device, dtype=self.dtype)

        # Warmup
        for _ in range(warmup_steps):
            state = ts.nvt_langevin_step(
                state, self.model, dt=dt_tensor, kT=kT, gamma=gamma
            )

        # Timed run
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_steps):
            state = ts.nvt_langevin_step(
                state, self.model, dt=dt_tensor, kT=kT, gamma=gamma
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        total_time = end_time - start_time

        return BenchmarkResult(
            backend="torch-sim",
            name="langevin",
            num_atoms=self.num_atoms_per_system,
            ensemble="NVT",
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            batch_size=self.batch_size,
        )


# ==============================================================================
# Main Benchmark Loop
# ==============================================================================


def run_benchmarks(config: dict, output_dir: Path) -> None:
    """Run comparison benchmarks across multiple model types.

    Parameters
    ----------
    config : dict
        Benchmark configuration from YAML
    output_dir : Path
        Output directory for CSV files
    """
    comp_config = config.get("torch_sim_comparison", {})
    if not comp_config.get("enabled", False):
        print("Torch-sim comparison benchmarks disabled")
        return

    system_sizes = comp_config.get("system_sizes", [256, 512, 1024])
    batch_sizes = comp_config.get("batch_sizes", [1, 2, 4, 8, 16, 32])
    integrators = comp_config.get("integrators", {})
    model_types_config = comp_config.get("model_types", {})

    potential_config = config.get("potential", {})

    gpu_sku = get_gpu_sku()
    results = []

    print("\nRunning nvalchemiops vs torch-sim Comparison")
    print(f"GPU: {gpu_sku}")

    # Import system creation function
    from benchmark_md_batch import create_batched_system

    # Iterate over enabled model types
    for model_type_name, model_config in model_types_config.items():
        if not model_config.get("enabled", False):
            continue

        print(f"\n{'=' * 70}")
        print(f"Model Type: {model_type_name}")
        print(f"Description: {model_config.get('description', '')}")
        print(f"{'=' * 70}")
        print_batch_benchmark_header()

        # We'll create models inside the loop since they need system-specific info
        # Just validate the model type here
        if model_type_name not in ["native_lj", "nvalchemiops_lj", "mace"]:
            print(f"Unknown model type: {model_type_name}")
            continue

        # Run benchmarks for each system size and batch size
        for num_atoms in system_sizes:
            for batch_size in batch_sizes:
                # Create batched system
                (
                    batch_positions,
                    batch_velocities,
                    batch_cells,
                    batch_masses,
                    batch_idx,
                    atom_ptr,
                ) = create_batched_system(
                    num_atoms_per_system=num_atoms,
                    batch_size=batch_size,
                    lattice_constant=5.26,
                    temperature=300.0,
                    device="cuda",
                    dtype=torch.float64,
                )

                pbc = torch.tensor([True, True, True], device=batch_positions.device)

                # Create atomic numbers (18 = Argon for LJ, will be overridden for MACE if needed)
                atomic_numbers = torch.full(
                    (batch_positions.shape[0],),
                    18,
                    device=batch_positions.device,
                    dtype=torch.int,
                )

                # Create models for this system
                if model_type_name == "native_lj":
                    nv_model, ts_model = create_native_lj_models(
                        epsilon=potential_config.get("epsilon", 0.0104),
                        sigma=potential_config.get("sigma", 3.40),
                        cutoff=potential_config.get("cutoff", 8.5),
                        skin=potential_config.get("skin", 1.0),
                        cell=batch_cells,
                        batch_idx=batch_idx,
                        device=torch.device("cuda"),
                        dtype=torch.float64,
                    )
                elif model_type_name == "nvalchemiops_lj":
                    nv_model, ts_model = create_nvalchemiops_lj_models(
                        epsilon=potential_config.get("epsilon", 0.0104),
                        sigma=potential_config.get("sigma", 3.40),
                        cutoff=potential_config.get("cutoff", 8.5),
                        skin=potential_config.get("skin", 1.0),
                        cell=batch_cells,
                        batch_idx=batch_idx,
                        device=torch.device("cuda"),
                        dtype=torch.float64,
                    )
                elif model_type_name == "mace":
                    nv_model, ts_model = create_mace_models(
                        model_name=model_config.get("model", "small"),
                        cell=batch_cells,
                        masses=batch_masses,
                        atomic_numbers=atomic_numbers,
                        pbc=pbc,
                        batch_idx=batch_idx,
                        device=torch.device("cuda"),
                        dtype=torch.float64,
                    )
                else:
                    continue

                # Velocity Verlet comparison
                if integrators.get("velocity_verlet", {}).get("enabled", False):
                    vv_config = integrators["velocity_verlet"]

                    # nvalchemiops benchmark
                    nv_bench = NvalchemiOpsBenchmark(
                        positions=batch_positions.clone(),
                        cell=batch_cells.clone(),
                        masses=batch_masses.clone(),
                        pbc=pbc,
                        model=nv_model,
                        cutoff=5.0 if model_type_name == "mace" else None,
                        skin=potential_config.get("skin", 1.0),
                        neighbor_rebuild_interval=potential_config.get(
                            "neighbor_rebuild_interval", 10
                        ),
                        velocities=batch_velocities.clone(),
                        batch_idx=batch_idx,
                        atom_ptr=atom_ptr,
                    )

                    nv_result = nv_bench.run_velocity_verlet(
                        dt=vv_config.get("dt", 0.001),
                        num_steps=vv_config.get("steps", 10000),
                        warmup_steps=vv_config.get("warmup_steps", 100),
                    )
                    nv_result.model_type = model_type_name
                    results.append(nv_result)
                    print_batch_benchmark_result(nv_result, is_md=True)

                    # torch-sim benchmark
                    if TORCH_SIM_AVAILABLE and ts_model is not None:
                        ts_bench = TorchSimBenchmark(
                            positions=batch_positions.clone(),
                            cell=batch_cells.clone(),
                            masses=batch_masses.clone(),
                            velocities=batch_velocities.clone(),
                            batch_idx=batch_idx,
                            atom_ptr=atom_ptr,
                            pbc=pbc,
                            model=ts_model,
                        )

                        ts_result = ts_bench.run_velocity_verlet(
                            dt=vv_config.get("dt", 0.001),
                            num_steps=vv_config.get("steps", 10000),
                            warmup_steps=vv_config.get("warmup_steps", 100),
                        )
                        ts_result.model_type = model_type_name
                        results.append(ts_result)
                        print_batch_benchmark_result(ts_result, is_md=True)

                # Langevin comparison
                if integrators.get("langevin", {}).get("enabled", False):
                    lang_config = integrators["langevin"]

                    # nvalchemiops benchmark
                    nv_bench = NvalchemiOpsBenchmark(
                        positions=batch_positions.clone(),
                        cell=batch_cells.clone(),
                        masses=batch_masses.clone(),
                        pbc=pbc,
                        model=nv_model,
                        skin=potential_config.get("skin", 1.0),
                        neighbor_rebuild_interval=potential_config.get(
                            "neighbor_rebuild_interval", 10
                        ),
                        velocities=batch_velocities.clone(),
                        batch_idx=batch_idx,
                        atom_ptr=atom_ptr,
                    )

                    nv_result = nv_bench.run_langevin(
                        dt=lang_config.get("dt", 0.001),
                        num_steps=lang_config.get("steps", 10000),
                        temperature=lang_config.get("temperature", 300.0),
                        friction=lang_config.get("friction", 0.01),
                        warmup_steps=lang_config.get("warmup_steps", 100),
                    )
                    nv_result.model_type = model_type_name
                    results.append(nv_result)
                    print_batch_benchmark_result(nv_result, is_md=True)

                    # torch-sim benchmark
                    if TORCH_SIM_AVAILABLE and ts_model is not None:
                        ts_bench = TorchSimBenchmark(
                            positions=batch_positions.clone(),
                            cell=batch_cells.clone(),
                            masses=batch_masses.clone(),
                            velocities=batch_velocities.clone(),
                            batch_idx=batch_idx,
                            atom_ptr=atom_ptr,
                            pbc=pbc,
                            model=ts_model,
                        )

                        ts_result = ts_bench.run_langevin(
                            dt=lang_config.get("dt", 0.001),
                            num_steps=lang_config.get("steps", 10000),
                            temperature=lang_config.get("temperature", 300.0),
                            friction=lang_config.get("friction", 0.01),
                            warmup_steps=lang_config.get("warmup_steps", 100),
                        )
                        ts_result.model_type = model_type_name
                        results.append(ts_result)
                        print_batch_benchmark_result(ts_result, is_md=True)

        print_batch_benchmark_footer()

    # Write CSV results
    if results:
        output_path = output_dir / f"dynamics_md_batch_comparison_{gpu_sku}.csv"
        write_results_csv(results, output_path)
        print(f"\nWrote results to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare nvalchemiops and torch-sim batched MD"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config_torchsim.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for CSV files",
    )

    args = parser.parse_args()

    if not TORCH_SIM_AVAILABLE:
        print("\nERROR: torch-sim is not installed.")
        print("Install with: pip install torch-sim-atomistic")
        print(
            "Or from source: git clone https://github.com/TorchSim/torch-sim && cd torch-sim && pip install ."
        )
        return

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_benchmarks(config, output_dir)


if __name__ == "__main__":
    main()
