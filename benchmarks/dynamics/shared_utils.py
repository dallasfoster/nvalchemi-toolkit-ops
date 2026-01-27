#!/usr/bin/env python3
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
Shared utilities for dynamics benchmarks.

This module provides common functionality used across all dynamics benchmark scripts:
- Unit conversion constants
- System creation utilities
- Result data structures with CSV export
- Configuration loading
- GPU detection
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml

from nvalchemiops.interactions.lj import lj_energy_forces, lj_energy_forces_virial

wp.init()

# ==============================================================================
# Unit Conversion Constants
# ==============================================================================
# The nvalchemiops dynamics kernels are unitless: they assume a self-consistent
# unit system for (x, v, t, m, E, F).
#
# We use:
# - length: Angstrom (Å)
# - time: femtosecond (fs)
# - energy: electron-volt (eV)
# - mass (internal): eV * fs^2 / Å^2 (so KE = 0.5 * m * v^2 is in eV)
#
# To convert from amu to internal mass units:
#   1 amu = amu_kg / (eV*fs^2/Å^2)_kg
EV_TO_J = 1.602176634e-19
AMU_TO_KG = 1.66053906660e-27
FS_TO_S = 1.0e-15
ANGSTROM_TO_M = 1.0e-10
_MASS_UNIT_KG = EV_TO_J * (FS_TO_S**2) / (ANGSTROM_TO_M**2)
AMU_TO_INTERNAL = AMU_TO_KG / _MASS_UNIT_KG  # ~103.6427

# Boltzmann constant in eV/K
KB_EV = 8.617333262e-5


@dataclass
class BenchmarkResult:
    """Container for benchmark results with CSV export support.

    Parameters
    ----------
    name : str
        Benchmark name (e.g., 'velocity_verlet', 'fire').
    backend : str
        Backend used ('nvalchemiops').
    model_type : str
        Model type used ('native_lj', 'nvalchemiops_lj', 'mace', or None for default).
    ensemble : str
        Ensemble or method type (e.g., 'NVE', 'NVT', 'optimization').
    num_atoms : int
        Number of atoms per system.
    num_steps : int
        Number of simulation/optimization steps.
    dt : float
        Timestep in fs (for MD) or max_step (for optimization).
    warmup_steps : int
        Number of warmup steps excluded from timing.
    total_time : float
        Total time in seconds (excluding warmup).
    step_times : list[float]
        List of individual step times in seconds.
    batch_size : Optional[int]
        Number of systems in batch (None for single-system).
    energies : list[float]
        Energy trajectory (optional).
    temperatures : list[float]
        Temperature trajectory (optional).
    final_ke : Optional[float]
        Final kinetic energy in eV (optional).
    final_pe : Optional[float]
        Final potential energy in eV (optional).
    final_temp : Optional[float]
        Final temperature in K (optional).
    """

    name: str
    backend: str
    ensemble: str
    num_atoms: int
    num_steps: int
    dt: float
    warmup_steps: int
    total_time: float
    step_times: list[float] = field(default_factory=list)
    batch_size: int | None = None
    model_type: str | None = None
    energies: list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    final_ke: float | None = None
    final_pe: float | None = None
    final_temp: float | None = None

    @property
    def avg_step_time_ms(self) -> float:
        """Average time per step in milliseconds."""
        return np.mean(self.step_times) * 1000 if self.step_times else 0.0

    @property
    def throughput_steps_per_s(self) -> float:
        """Calculate steps per second."""
        return self.num_steps / self.total_time if self.total_time > 0 else 0.0

    @property
    def throughput_atom_steps_per_s(self) -> float:
        """Calculate atom-steps per second (throughput metric).

        For batched benchmarks, this uses total_atoms (num_atoms * batch_size)
        to give the total atom-steps across all systems in the batch.
        """
        return (
            self.total_atoms * self.num_steps / self.total_time
            if self.total_time > 0
            else 0.0
        )

    @property
    def total_atoms(self) -> int:
        """Total number of atoms (num_atoms * batch_size for batched)."""
        if self.batch_size is not None:
            return self.num_atoms * self.batch_size
        return self.num_atoms

    @property
    def batch_throughput_system_steps_per_s(self) -> float | None:
        """Complete system steps per second for batched benchmarks."""
        if self.batch_size is not None and self.total_time > 0:
            return self.batch_size * self.num_steps / self.total_time
        return None

    @property
    def is_batched(self) -> bool:
        """Check if this is a batched benchmark."""
        return self.batch_size is not None

    @property
    def ns_per_day(self) -> float:
        """Calculate nanoseconds of simulation time per day of wall-clock time.

        This metric is only meaningful for MD simulations (not optimization).
        For optimization benchmarks, returns NaN.

        Formula: ns/day = (timestep_fs * steps_per_second * 86400 * num_systems) / 1e6
        where timestep_fs is the MD timestep in femtoseconds.
        """
        # Check if this is an optimization benchmark
        if self.ensemble.lower() == "optimization" or self.name.lower() == "fire":
            return float("nan")

        # For MD: calculate ns/day
        num_systems = self.batch_size if self.is_batched else 1
        if self.total_time > 0:
            steps_per_second = self.throughput_steps_per_s
            # timestep (fs) * steps/s * seconds/day / 1e6 (fs to ns)
            return (self.dt * steps_per_second * 86400 * num_systems) / 1e6
        return 0.0

    def to_csv_row(self) -> dict:
        """Convert to dictionary for CSV row.

        Returns appropriate columns based on whether this is a single-system
        or batched benchmark.
        """
        # Common columns for both single and batched
        row = {
            "backend": self.backend,
            "model_type": self.model_type or "",
            "method": self.name,
            "num_atoms": self.num_atoms,
            "ensemble": self.ensemble,
            "steps": self.num_steps,
            "dt": self.dt,
            "warmup_steps": self.warmup_steps,
            "avg_step_time_ms": self.avg_step_time_ms,
            "total_time_s": self.total_time,
            "throughput_steps_per_s": self.throughput_steps_per_s,
            "throughput_atom_steps_per_s": self.throughput_atom_steps_per_s,
            "ns_per_day": self.ns_per_day,
        }

        # Add batch-specific columns if this is a batched benchmark
        if self.is_batched:
            row["batch_size"] = self.batch_size
            row["total_atoms"] = self.total_atoms
            row["batch_throughput_system_steps_per_s"] = (
                self.batch_throughput_system_steps_per_s
            )

        return row

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (backward compatibility)."""
        d = {
            "name": self.name,
            "backend": self.backend,
            "ensemble": self.ensemble,
            "num_atoms": self.num_atoms,
            "batch_size": self.batch_size,
            "num_steps": self.num_steps,
            "total_time_s": self.total_time,
            "steps_per_second": self.throughput_steps_per_s,
            "atom_steps_per_second": self.throughput_atom_steps_per_s,
            "avg_step_time_ms": self.avg_step_time_ms,
        }
        if self.is_batched:
            d["batch_size"] = self.batch_size
            d["total_atoms"] = self.total_atoms
            d["batch_throughput_system_steps_per_s"] = (
                self.batch_throughput_system_steps_per_s
            )
        return d


def write_results_csv(
    results: list[BenchmarkResult],
    output_path: str | Path,
) -> None:
    """Write benchmark results to CSV file.

    Automatically detects whether results are single-system or batched and uses
    the appropriate schema.

    Parameters
    ----------
    results : list[BenchmarkResult]
        List of benchmark results to write.
    output_path : str or Path
        Output CSV file path.

    Raises
    ------
    ValueError
        If results list is empty or contains mixed single/batch results.
    """
    if not results:
        raise ValueError("Results list is empty")

    # Check if all results are consistently batched or single-system
    is_batched = results[0].is_batched
    if not all(r.is_batched == is_batched for r in results):
        raise ValueError("Cannot mix single-system and batched results in same CSV")

    # Define column order based on benchmark type
    if is_batched:
        # Batched benchmark schema (16 columns)
        fieldnames = [
            "backend",
            "model_type",
            "method",
            "num_atoms",
            "ensemble",
            "batch_size",
            "total_atoms",
            "steps",
            "dt",
            "warmup_steps",
            "avg_step_time_ms",
            "total_time_s",
            "throughput_steps_per_s",
            "throughput_atom_steps_per_s",
            "ns_per_day",
            "batch_throughput_system_steps_per_s",
        ]
    else:
        # Single-system benchmark schema (13 columns)
        fieldnames = [
            "backend",
            "model_type",
            "method",
            "num_atoms",
            "ensemble",
            "steps",
            "dt",
            "warmup_steps",
            "avg_step_time_ms",
            "total_time_s",
            "throughput_steps_per_s",
            "throughput_atom_steps_per_s",
            "ns_per_day",
        ]

    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())


def load_config(config_path: str | Path) -> dict:
    """Load benchmark configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_benchmark_header(benchmark_type: str = "MD") -> None:
    """Print a formatted header for benchmark results table.

    Parameters
    ----------
    benchmark_type : str
        Type of benchmark ('MD' or 'Optimization').
    """
    if benchmark_type.upper() == "MD":
        print("\n" + "=" * 110)
        print(
            f"{'Method':<20} {'Atoms':<10} {'Steps':<10} {'Time (s)':<12} "
            f"{'Atom-steps/s':<15} {'ns/day':<12}"
        )
        print("=" * 110)
    else:
        print("\n" + "=" * 100)
        print(
            f"{'Method':<20} {'Atoms':<10} {'Steps':<10} {'Converged':<12} "
            f"{'Time (s)':<12} {'Atom-steps/s':<15}"
        )
        print("=" * 100)


def print_benchmark_result(result: BenchmarkResult, is_md: bool = True) -> None:
    """Print a formatted row for benchmark result.

    Parameters
    ----------
    result : BenchmarkResult
        Benchmark result to print.
    is_md : bool
        Whether this is an MD benchmark (True) or optimization (False).
    """
    method_str = f"{result.name} ({result.ensemble})"

    if is_md:
        # For MD: show ns/day
        ns_day = result.ns_per_day
        ns_day_str = f"{ns_day:.2f}" if not np.isnan(ns_day) else "N/A"
        print(
            f"{method_str:<20} {result.num_atoms:<10} {result.num_steps:<10} "
            f"{result.total_time:<12.3f} {result.throughput_atom_steps_per_s:<15.2e} "
            f"{ns_day_str:<12}"
        )
    else:
        # For optimization: show convergence instead of ns/day
        converged = "Yes" if result.num_steps < 1000 else "No"
        print(
            f"{method_str:<20} {result.num_atoms:<10} {result.num_steps:<10} "
            f"{converged:<12} {result.total_time:<12.3f} "
            f"{result.throughput_atom_steps_per_s:<15.2e}"
        )


def print_benchmark_footer() -> None:
    """Print a formatted footer for benchmark results table."""
    print("=" * 110)


def print_batch_benchmark_header() -> None:
    """Print a formatted header for batched benchmark results table."""
    print("\n" + "=" * 120)
    print(
        f"{'Backend':<18} {'Method':<18} {'Atoms/sys':<12} {'Batch':<8} {'Total':<10} "
        f"{'Steps':<10} {'Time (s)':<12} {'Atom-steps/s':<15} {'ns/day':<12}"
    )
    print("=" * 120)


def print_batch_benchmark_result(result: BenchmarkResult, is_md: bool = True) -> None:
    """Print a formatted row for batched benchmark result.

    Parameters
    ----------
    result : BenchmarkResult
        Batched benchmark result to print.
    is_md : bool
        Whether this is an MD benchmark (True) or optimization (False).
    """
    method_str = f"{result.name}"
    batch_size = result.batch_size if result.batch_size else 1
    total_atoms = result.total_atoms

    if is_md:
        ns_day = result.ns_per_day
        ns_day_str = f"{ns_day:.2f}" if not np.isnan(ns_day) else "N/A"
        print(
            f"{result.backend:<18} {method_str:<18} {result.num_atoms:<12} {batch_size:<8} {total_atoms:<10} "
            f"{result.num_steps:<10} {result.total_time:<12.3f} "
            f"{result.throughput_atom_steps_per_s:<15.2e} {ns_day_str:<12}"
        )
    else:
        # For optimization
        print(
            f"{result.backend:<18} {method_str:<18} {result.num_atoms:<12} {batch_size:<8} {total_atoms:<10} "
            f"{result.num_steps:<10} {result.total_time:<12.3f} "
            f"{result.throughput_atom_steps_per_s:<15.2e}"
        )


def print_batch_benchmark_footer() -> None:
    """Print a formatted footer for batched benchmark results table."""
    print("=" * 120)


def get_gpu_sku() -> str:
    """Get GPU model name for benchmark identification.

    Returns
    -------
    str
        GPU model string (e.g., 'rtx4090', 'a100').
        Returns 'cpu' if no CUDA device available.
        Returns 'unknown_gpu' if detection fails.

    Examples
    --------
    >>> get_gpu_sku()
    'rtx4090'
    """
    try:
        if not torch.cuda.is_available():
            return "cpu"

        # Get device name from PyTorch
        device_name = torch.cuda.get_device_name(0)

        # Clean up device name to create SKU string
        # e.g., "NVIDIA GeForce RTX 4090" -> "rtx4090"
        sku = device_name.lower()
        sku = sku.replace("nvidia", "").replace("geforce", "").replace("tesla", "")
        sku = sku.replace("quadro", "").replace("rtx", "rtx")
        sku = "".join(c for c in sku if c.isalnum())

        # Handle common patterns
        if "rtx" in sku:
            # Extract RTX model number
            idx = sku.find("rtx")
            sku = "rtx" + sku[idx + 3 : idx + 7].strip()
        elif "a100" in sku:
            sku = "a100"
        elif "v100" in sku:
            sku = "v100"
        elif "h100" in sku:
            sku = "h100"
        elif sku:
            # Fallback: take first alphanumeric token
            sku = sku.split()[0] if sku.split() else "unknown_gpu"
        else:
            sku = "unknown_gpu"

        return sku

    except Exception:
        return "unknown_gpu"


def create_lj_system(
    num_atoms: int,
    lattice_constant: float = 5.26,
    temperature: float = 300.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create an FCC argon system for LJ simulations.

    Parameters
    ----------
    num_atoms : int
        Target number of atoms (actual may differ slightly due to FCC structure).
    lattice_constant : float, optional
        FCC lattice constant in Angstroms (default: 5.26 for argon).
    temperature : float, optional
        Initial temperature in Kelvin (default: 300.0).
    device : str, optional
        Torch device (default: 'cuda').
    dtype : torch.dtype, optional
        Data type for tensors (default: torch.float64).

    Returns
    -------
    positions : torch.Tensor
        Atomic positions, shape (N, 3).
    cell : torch.Tensor
        Unit cell matrix, shape (1, 3, 3).
    masses : torch.Tensor
        Atomic masses in internal units, shape (N,).
    velocities : torch.Tensor
        Initial velocities from Maxwell-Boltzmann distribution, shape (N, 3).

    Notes
    -----
    - Creates FCC (face-centered cubic) structure with 4 atoms per unit cell
    - Initializes velocities from Maxwell-Boltzmann distribution
    - Removes center-of-mass momentum
    - Actual number of atoms may differ slightly from target due to FCC structure
    - Uses argon mass: 39.948 amu
    """
    # Calculate supercell size needed
    atoms_per_cell = 4  # FCC has 4 atoms per unit cell
    n_cells = int(np.ceil((num_atoms / atoms_per_cell) ** (1 / 3)))

    # Create FCC lattice basis (4 atoms per unit cell)
    basis = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
        dtype=dtype,
    )

    # Generate supercell positions
    positions_list = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                offset = torch.tensor([i, j, k], dtype=dtype)
                for atom_basis in basis:
                    pos = (offset + atom_basis) * lattice_constant
                    positions_list.append(pos)

    positions = torch.stack(positions_list).to(device)
    actual_num_atoms = positions.shape[0]

    # Create cell matrix (cubic cell)
    cell_length = n_cells * lattice_constant
    cell = torch.eye(3, dtype=dtype, device=device) * cell_length
    cell = cell.unsqueeze(0)  # Shape (1, 3, 3)

    # Create masses (argon: 39.948 amu)
    argon_mass_amu = 39.948
    masses = torch.full(
        (actual_num_atoms,),
        argon_mass_amu * AMU_TO_INTERNAL,
        dtype=dtype,
        device=device,
    )

    # Initialize velocities from Maxwell-Boltzmann distribution
    # For 3D: v ~ N(0, sqrt(kB*T/m)) for each component
    # kB*T in eV, mass in internal units (eV*fs^2/Å^2)
    kB_T = KB_EV * temperature  # eV
    mass_internal = argon_mass_amu * AMU_TO_INTERNAL
    sigma_v = np.sqrt(kB_T / mass_internal)  # Å/fs

    velocities = torch.randn(actual_num_atoms, 3, dtype=dtype, device=device) * sigma_v

    # Remove center-of-mass momentum
    momenta = velocities * masses.unsqueeze(1)
    total_momentum = momenta.sum(dim=0)
    velocities -= total_momentum / masses.sum() / masses.unsqueeze(1)

    return positions, cell, masses, velocities


# ==============================================================================
# Warp Kernels for Virial Conversion
# ==============================================================================


@wp.kernel
def _convert_flat_virial_to_vec9_kernel(
    virial_flat: wp.array(dtype=wp.float64),
    virial_vec9: wp.array(dtype=wp.types.vector(length=9, dtype=wp.float64)),
    negate: wp.bool,
):
    """Convert flat virial array to vec9 format with optional negation.

    The LJ kernels use virial sign convention: W = -Σ r ⊗ F
    The NPT/NPH integrators expect: W = +Σ r ⊗ F
    Therefore we negate when passing to NPT/NPH.

    For batched mode, virial_flat has shape (num_systems*9,) and we process
    each system's 9 elements separately.

    Launched with dim=num_systems.
    """
    sys_id = wp.tid()

    sign = -1.0 if negate else 1.0

    # Offset into flat array for this system
    offset = sys_id * 9

    virial_vec9[sys_id] = wp.types.vector(length=9, dtype=wp.float64)(
        sign * virial_flat[offset + 0],
        sign * virial_flat[offset + 1],
        sign * virial_flat[offset + 2],
        sign * virial_flat[offset + 3],
        sign * virial_flat[offset + 4],
        sign * virial_flat[offset + 5],
        sign * virial_flat[offset + 6],
        sign * virial_flat[offset + 7],
        sign * virial_flat[offset + 8],
    )


@wp.kernel
def _convert_flat_virial_to_vec9_kernel_f32(
    virial_flat: wp.array(dtype=wp.float32),
    virial_vec9: wp.array(dtype=wp.types.vector(length=9, dtype=wp.float32)),
    negate: wp.bool,
):
    """Convert flat virial array to vec9 format with optional negation (float32 version)."""
    sys_id = wp.tid()

    sign = -1.0 if negate else 1.0

    # Offset into flat array for this system
    offset = sys_id * 9

    virial_vec9[sys_id] = wp.types.vector(length=9, dtype=wp.float32)(
        sign * virial_flat[offset + 0],
        sign * virial_flat[offset + 1],
        sign * virial_flat[offset + 2],
        sign * virial_flat[offset + 3],
        sign * virial_flat[offset + 4],
        sign * virial_flat[offset + 5],
        sign * virial_flat[offset + 6],
        sign * virial_flat[offset + 7],
        sign * virial_flat[offset + 8],
    )


def convert_flat_virial_to_vec9(
    virial_flat: wp.array,
    virial_vec9: wp.array,
    negate: bool,
    device: str,
):
    """Convert flat virial array to vec9 format without synchronization.

    Parameters
    ----------
    virial_flat : wp.array
        Flat 9-element array (or multiple of 9 for batched).
    virial_vec9 : wp.array
        Output vec9 array (1 element for single system, B elements for batched).
    negate : bool
        Whether to negate virial (True for NPT/NPH, False otherwise).
    device : str
        Warp device string.
    """
    # Determine dtype and launch appropriate kernel
    if virial_flat.dtype == wp.float32:
        wp.launch(
            kernel=_convert_flat_virial_to_vec9_kernel_f32,
            dim=virial_vec9.shape[0],
            inputs=[virial_flat, virial_vec9, negate],
            device=device,
        )
    else:
        wp.launch(
            kernel=_convert_flat_virial_to_vec9_kernel,
            dim=virial_vec9.shape[0],
            inputs=[virial_flat, virial_vec9, negate],
            device=device,
        )


# ==============================================================================
# Model Interface for Benchmarking
# ==============================================================================


class NvalchemiopsModelInterface:
    """Abstract interface for models used in nvalchemiops benchmarks.

    This interface allows different force calculation methods (LJ, MACE, etc.)
    to be used interchangeably in the benchmark infrastructure.
    """

    def compute_forces(
        self,
        wp_positions: wp.array,
        neighbor_matrix: wp.array,
        num_neighbors: wp.array,
        neighbor_shifts: wp.array,
    ) -> tuple[wp.array, wp.array]:
        """Compute energies and forces.

        Parameters
        ----------
        wp_positions : wp.array
            Atomic positions (warp array)
        neighbor_matrix : wp.array
            Neighbor list matrix
        num_neighbors : wp.array
            Number of neighbors per atom
        neighbor_shifts : wp.array
            PBC shift vectors for neighbors

        Returns
        -------
        wp_energies : wp.array
            Atomic energies
        wp_forces : wp.array
            Atomic forces
        """
        raise NotImplementedError

    def compute_virial(
        self,
        wp_positions: wp.array,
        neighbor_matrix: wp.array,
        num_neighbors: wp.array,
        neighbor_shifts: wp.array,
    ) -> tuple[wp.array, wp.array, wp.array]:
        """Compute energies, forces, and virial.

        Parameters
        ----------
        wp_positions : wp.array
            Atomic positions (warp array)
        neighbor_matrix : wp.array
            Neighbor list matrix
        num_neighbors : wp.array
            Number of neighbors per atom
        neighbor_shifts : wp.array
            PBC shift vectors for neighbors

        Returns
        -------
        wp_energies : wp.array
            Atomic energies
        wp_forces : wp.array
            Atomic forces
        wp_virial : wp.array
            Virial tensor (9 components per system)
        """
        raise NotImplementedError


class NvalchemiopsLJModel(NvalchemiopsModelInterface):
    """Lennard-Jones model using nvalchemiops kernels.

    Parameters
    ----------
    epsilon : float
        LJ epsilon parameter (eV)
    sigma : float
        LJ sigma parameter (Å)
    cutoff : float
        LJ cutoff distance (Å)
    cell : torch.Tensor
        Unit cell matrix
    batch_idx : torch.Tensor or None
        Batch index for each atom (for batched mode)
    device : str
        Warp device string
    dtype : torch.dtype
        Data type
    """

    def __init__(
        self,
        epsilon: float,
        sigma: float,
        cutoff: float,
        cell: torch.Tensor,
        batch_idx: torch.Tensor | None,
        device: str,
        dtype: torch.dtype,
    ):
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.device = device
        self.dtype = dtype

        # Import type helpers
        from nvalchemiops.types import get_wp_mat_dtype

        wp_mat_dtype = get_wp_mat_dtype(dtype)
        self.wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)

        self.batch_idx = batch_idx
        self.wp_batch_idx = (
            None
            if batch_idx is None
            else wp.from_torch(batch_idx.to(torch.int32), dtype=wp.int32)
        )
        self.is_batched = batch_idx is not None

    def compute_forces(
        self,
        wp_positions: wp.array,
        neighbor_matrix: wp.array,
        num_neighbors: wp.array,
        neighbor_shifts: wp.array,
    ) -> tuple[wp.array, wp.array]:
        """Compute LJ energies and forces."""
        # Determine fill_value (num_atoms)
        if hasattr(neighbor_matrix, "shape"):
            fill_value = neighbor_matrix.shape[0]
        else:
            # Fallback
            fill_value = num_neighbors.shape[0]

        wp_energies, wp_forces = lj_energy_forces(
            positions=wp_positions,
            cell=self.wp_cell,
            epsilon=self.epsilon,
            sigma=self.sigma,
            cutoff=self.cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            num_neighbors=num_neighbors,
            fill_value=fill_value,
            batch_idx=self.wp_batch_idx,
            device=self.device,
        )

        return wp_energies, wp_forces

    def compute_virial(
        self,
        wp_positions: wp.array,
        neighbor_matrix: wp.array,
        num_neighbors: wp.array,
        neighbor_shifts: wp.array,
    ) -> tuple[wp.array, wp.array, wp.array]:
        """Compute LJ energies, forces, and virial."""
        # Determine fill_value (num_atoms)
        if hasattr(neighbor_matrix, "shape"):
            fill_value = neighbor_matrix.shape[0]
        else:
            fill_value = num_neighbors.shape[0]

        wp_energies, wp_forces, wp_virial = lj_energy_forces_virial(
            positions=wp_positions,
            cell=self.wp_cell,
            epsilon=self.epsilon,
            sigma=self.sigma,
            cutoff=self.cutoff,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            num_neighbors=num_neighbors,
            fill_value=fill_value,
            batch_idx=self.wp_batch_idx,
            device=self.device,
        )

        return wp_energies, wp_forces, wp_virial


# ==============================================================================
# Unified Benchmark Class
# ==============================================================================


class NvalchemiOpsBenchmark:
    """Unified benchmark class for both single-system and batched simulations.

    This class consolidates MD and optimization benchmarks, automatically detecting
    whether to use single-system or batched mode based on the presence of batch_idx
    and atom_ptr parameters.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions. Shape (N, 3) for single-system or (total_atoms, 3) for batched.
    cell : torch.Tensor
        Unit cell matrix. Shape (1, 3, 3) for single-system or (num_systems, 3, 3) for batched.
    masses : torch.Tensor
        Atomic masses. Shape (N,) for single-system or (total_atoms,) for batched.
    pbc : torch.Tensor
        Periodic boundary conditions, shape (3,).
    model : NvalchemiopsModelInterface, optional
        Model for force/energy computation. If None, uses LJ with epsilon/sigma/cutoff.
    epsilon : float, optional
        LJ epsilon parameter (eV). Used only if model is None.
    sigma : float, optional
        LJ sigma parameter (Å). Used only if model is None.
    cutoff : float, optional
        LJ cutoff distance (Å). Used only if model is None.
    skin : float, optional
        Neighbor list skin distance (Å). Default 1.0.
    neighbor_rebuild_interval : int, optional
        Interval for rebuilding neighbor lists (0 = displacement-based). Default 10.
    velocities : torch.Tensor, optional
        Initial velocities. Required for MD, optional for optimization.
    batch_idx : torch.Tensor, optional
        Batch index for each atom (batched mode only). Shape (total_atoms,).
    atom_ptr : torch.Tensor, optional
        Pointer to start of each batch (batched mode only). Shape (num_systems+1,).

    Notes
    -----
    - Batching is auto-detected: if batch_idx and atom_ptr are provided, batched mode is used.
    - For batched mode, positions/velocities/masses should be concatenated across all systems.
    - Supports integrators: VelocityVerlet, Langevin, NoseHoover, NPT, NPH.
    - Supports optimizer: FIRE.
    - If model is None, automatically creates NvalchemiopsLJModel with epsilon/sigma/cutoff.
    """

    def __init__(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        masses: torch.Tensor,
        pbc: torch.Tensor,
        model: NvalchemiopsModelInterface | None = None,
        epsilon: float | None = None,
        sigma: float | None = None,
        cutoff: float | None = None,
        skin: float = 1.0,
        neighbor_rebuild_interval: int = 10,
        velocities: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
        atom_ptr: torch.Tensor | None = None,
    ):
        # Store torch tensors
        self.torch_positions = positions
        self.torch_cell = cell
        self.torch_masses = masses
        self.torch_velocities = velocities
        self.pbc = pbc

        # Batching parameters
        self.batch_idx = batch_idx
        self.wp_batch_idx = (
            None
            if batch_idx is None
            else wp.from_torch(batch_idx.to(torch.int32), dtype=wp.int32)
        )
        self.atom_ptr = atom_ptr
        self.wp_atom_ptr = (
            None
            if atom_ptr is None
            else wp.from_torch(atom_ptr.to(torch.int32), dtype=wp.int32)
        )
        self.is_batched = batch_idx is not None and atom_ptr is not None

        # Device and dtype setup (needed for model creation)
        self.device = positions.device
        self.dtype = positions.dtype
        self.wp_device = str(self.device)

        # Model setup: use provided model or create LJ model from parameters
        if model is not None:
            self.model = model
            # For neighbor list, we need cutoff
            # Try to get it from model if available
            if hasattr(model, "cutoff"):
                cutoff = model.cutoff
            elif cutoff is None:
                raise ValueError(
                    "cutoff must be provided when using external model without cutoff attribute"
                )
        else:
            # Backward compatibility: create LJ model from parameters
            if epsilon is None or sigma is None or cutoff is None:
                raise ValueError(
                    "Must provide either model or (epsilon, sigma, cutoff)"
                )
            self.model = NvalchemiopsLJModel(
                epsilon=epsilon,
                sigma=sigma,
                cutoff=cutoff,
                cell=cell,
                batch_idx=batch_idx,
                device=self.wp_device,
                dtype=self.dtype,
            )

        # Neighbor list parameters
        self.cutoff = cutoff
        self.skin = skin
        self.neighbor_rebuild_interval = neighbor_rebuild_interval

        # System size
        if self.is_batched:
            self.total_atoms = positions.shape[0]
            self.num_systems = len(atom_ptr) - 1
            self.num_atoms = self.total_atoms // self.num_systems
        else:
            self.num_atoms = positions.shape[0]
            self.total_atoms = self.num_atoms
            self.num_systems = 1

        # Import type helpers
        from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

        self.wp_dtype = get_wp_dtype(self.dtype)
        self.wp_vec_dtype = get_wp_vec_dtype(self.dtype)
        self.wp_mat_dtype = get_wp_mat_dtype(self.dtype)

        # Convert to warp arrays
        self.wp_cell = wp.from_torch(cell, dtype=self.wp_mat_dtype)
        self.wp_masses = wp.from_torch(masses.contiguous(), dtype=self.wp_dtype)

        # Neighbor list state (initialized on first rebuild)
        self._torch_neighbor_matrix = None
        self._torch_num_neighbors = None
        self._torch_neighbor_shifts = None
        self._wp_neighbor_matrix = None
        self._wp_num_neighbors = None
        self._wp_neighbor_shifts = None
        self._ref_positions = positions.clone()  # For displacement-based rebuilding
        self._steps_since_rebuild = 0

        # Initial neighbor list build
        self._rebuild_neighbors()

    def _rebuild_neighbors(self) -> None:
        """Rebuild neighbor lists."""
        from nvalchemiops.neighborlist import neighbor_list

        # Compute neighbor list
        neighbor_matrix, num_neighbors, neighbor_shifts = neighbor_list(
            positions=self.torch_positions,
            cutoff=self.cutoff + self.skin,
            cell=self.torch_cell,
            pbc=self.pbc,
            method="cell_list" if not self.is_batched else "batch_cell_list",
            batch_idx=self.batch_idx,
            batch_ptr=self.atom_ptr,
        )

        # Store torch versions
        self._torch_neighbor_matrix = neighbor_matrix
        self._torch_num_neighbors = num_neighbors
        self._torch_neighbor_shifts = neighbor_shifts

        # Convert to warp
        self._wp_neighbor_matrix = wp.from_torch(neighbor_matrix, dtype=wp.int32)
        self._wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32)
        self._wp_neighbor_shifts = wp.from_torch(neighbor_shifts, dtype=wp.vec3i)
        self._steps_since_rebuild = 0

    def _check_rebuild(self, positions: torch.Tensor | None = None) -> bool:
        """Check if neighbor list needs rebuilding.

        Parameters
        ----------
        positions : torch.Tensor, optional
            Current positions. If None, uses interval-based check.

        Returns
        -------
        bool
            True if rebuild is needed.
        """
        if self.neighbor_rebuild_interval > 0:
            # Interval-based rebuilding
            self._steps_since_rebuild += 1
            return self._steps_since_rebuild >= self.neighbor_rebuild_interval
        else:
            # Displacement-based rebuilding
            if positions is None:
                return False
            max_displacement = (
                torch.norm(positions - self._ref_positions, dim=1).max().item()
            )
            return max_displacement > self.skin / 2.0

    def _compute_forces(
        self, wp_positions: wp.array, compute_virial: bool = False
    ) -> tuple[wp.array, wp.array, wp.array | None]:
        """Compute energies, forces, and optionally virial using the model.

        Parameters
        ----------
        wp_positions : wp.array
            Atomic positions.
        compute_virial : bool, optional
            Whether to compute virial tensor (for NPT/NPH).

        Returns
        -------
        wp_energies : wp.array
            Atomic energies.
        wp_forces : wp.array
            Atomic forces.
        wp_virial : wp.array or None
            Virial tensor (9 components) if compute_virial=True, else None.
        """

        # Check if rebuild needed (for displacement-based)
        if self.neighbor_rebuild_interval == 0:
            positions_torch = wp.to_torch(wp_positions)
            if self._check_rebuild(positions_torch):
                self.torch_positions = positions_torch
                self._rebuild_neighbors()
                self._ref_positions = positions_torch.clone()

        # Call model to compute forces/energies
        if compute_virial:
            wp_energies, wp_forces, wp_virial = self.model.compute_virial(
                wp_positions=wp_positions,
                neighbor_matrix=self._wp_neighbor_matrix,
                num_neighbors=self._wp_num_neighbors,
                neighbor_shifts=self._wp_neighbor_shifts,
            )
        else:
            wp_energies, wp_forces = self.model.compute_forces(
                wp_positions=wp_positions,
                neighbor_matrix=self._wp_neighbor_matrix,
                num_neighbors=self._wp_num_neighbors,
                neighbor_shifts=self._wp_neighbor_shifts,
            )
            wp_virial = None

        return wp_energies, wp_forces, wp_virial

    def _run_warmup(
        self, run_step_fn, warmup_steps: int, log_message: str = "Warmup"
    ) -> None:
        """Run warmup steps.

        Parameters
        ----------
        run_step_fn : callable
            Function to call for each step (takes no arguments).
        warmup_steps : int
            Number of warmup steps.
        log_message : str, optional
            Message to print during warmup.
        """
        if warmup_steps > 0:
            for _ in range(warmup_steps):
                run_step_fn()
            wp.synchronize()

    def _run_timed_loop(
        self,
        run_step_fn,
        num_steps: int,
        log_interval: int = 100,
        log_fn: callable | None = None,
    ) -> tuple[float, list[float]]:
        """Run timed loop for benchmarking.

        Parameters
        ----------
        run_step_fn : callable
            Function to call for each step (takes step index as argument).
        num_steps : int
            Number of steps to run.
        log_interval : int, optional
            Logging interval.
        log_fn : callable, optional
            Logging function called at intervals. Takes step index as argument.

        Returns
        -------
        total_time : float
            Total time in seconds.
        step_times : list[float]
            Individual step times in seconds.
        """
        import time

        wp.synchronize()
        start_time = time.perf_counter()

        for step in range(num_steps):
            run_step_fn(step)

        wp.synchronize()
        total_time = time.perf_counter() - start_time
        step_times = [total_time / num_steps] * num_steps
        return total_time, step_times

    # ========================================================================
    # MD Integrators
    # ========================================================================

    def run_velocity_verlet(
        self,
        dt: float,
        num_steps: int,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Run NVE simulation using velocity Verlet integrator.

        Parameters
        ----------
        dt : float
            Timestep in fs.
        num_steps : int
            Number of simulation steps.
        warmup_steps : int, optional
            Number of warmup steps (excluded from timing).
        log_interval : int, optional
            Logging interval.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.
        """
        from nvalchemiops.dynamics.integrators import (
            velocity_verlet_position_update,
            velocity_verlet_velocity_finalize,
        )
        from nvalchemiops.dynamics.utils import (
            compute_cell_inverse,
            wrap_positions_to_cell,
        )

        # Convert to warp
        wp_positions = wp.from_torch(
            self.torch_positions.clone(), dtype=self.wp_vec_dtype
        )
        wp_velocities = wp.from_torch(
            self.torch_velocities.clone(), dtype=self.wp_vec_dtype
        )
        wp_dt = wp.array([dt], dtype=self.wp_dtype, device=self.wp_device)

        # Pre-compute cell inverse
        wp_cell_inv = compute_cell_inverse(self.wp_cell, device=self.wp_device)

        # Initial forces
        _, wp_forces, _ = self._compute_forces(wp_positions)

        # Warmup
        def warmup_step():
            velocity_verlet_position_update(
                wp_positions,
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            wrap_positions_to_cell(
                wp_positions,
                cells=self.wp_cell,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )
            _, wp_forces_new, _ = self._compute_forces(wp_positions)
            wp.copy(wp_forces, wp_forces_new)
            velocity_verlet_velocity_finalize(
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            if self.neighbor_rebuild_interval > 0:
                self._steps_since_rebuild += 1
                if self._steps_since_rebuild >= self.neighbor_rebuild_interval:
                    self.torch_positions = wp.to_torch(wp_positions)
                    self._rebuild_neighbors()

        self._run_warmup(warmup_step, warmup_steps, "Velocity Verlet warmup")

        # Timed loop
        def run_step(step):
            velocity_verlet_position_update(
                wp_positions,
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            wrap_positions_to_cell(
                wp_positions,
                cells=self.wp_cell,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )
            _, wp_forces_new, _ = self._compute_forces(wp_positions)
            wp.copy(wp_forces, wp_forces_new)
            velocity_verlet_velocity_finalize(
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            if self.neighbor_rebuild_interval > 0:
                self._steps_since_rebuild += 1
                if self._steps_since_rebuild >= self.neighbor_rebuild_interval:
                    self.torch_positions = wp.to_torch(wp_positions)
                    self._rebuild_neighbors()

        total_time, step_times = self._run_timed_loop(run_step, num_steps, log_interval)

        return BenchmarkResult(
            name="velocity_verlet",
            backend="nvalchemiops",
            ensemble="NVE",
            num_atoms=self.num_atoms,
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.num_systems if self.is_batched else None,
        )

    def run_langevin(
        self,
        dt: float,
        num_steps: int,
        temperature: float,
        friction: float,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Run NVT simulation using Langevin dynamics (BAOAB integrator).

        Parameters
        ----------
        dt : float
            Timestep in fs.
        num_steps : int
            Number of simulation steps.
        temperature : float
            Target temperature in K.
        friction : float
            Friction coefficient in 1/fs.
        warmup_steps : int, optional
            Number of warmup steps (excluded from timing).
        log_interval : int, optional
            Logging interval.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.
        """
        from nvalchemiops.dynamics.integrators import (
            langevin_baoab_finalize,
            langevin_baoab_half_step,
        )
        from nvalchemiops.dynamics.utils import (
            compute_cell_inverse,
            wrap_positions_to_cell,
        )

        # Convert temperature to kT (eV)
        kT = temperature * KB_EV

        # Convert to warp
        wp_positions = wp.from_torch(
            self.torch_positions.clone(), dtype=self.wp_vec_dtype
        )
        wp_velocities = wp.from_torch(
            self.torch_velocities.clone(), dtype=self.wp_vec_dtype
        )
        wp_dt = wp.array([dt], dtype=self.wp_dtype, device=self.wp_device)
        wp_temp = wp.array([kT], dtype=self.wp_dtype, device=self.wp_device)
        wp_friction = wp.array([friction], dtype=self.wp_dtype, device=self.wp_device)
        # Pre-compute cell inverse
        wp_cell_inv = compute_cell_inverse(self.wp_cell, device=self.wp_device)

        # Initial forces
        _, wp_forces, _ = self._compute_forces(wp_positions)

        # Warmup
        def warmup_step():
            langevin_baoab_half_step(
                wp_positions,
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                wp_temp,
                wp_friction,
                42,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            wrap_positions_to_cell(
                wp_positions,
                cells=self.wp_cell,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )
            _, wp_forces_new, _ = self._compute_forces(wp_positions)
            wp.copy(wp_forces, wp_forces_new)
            langevin_baoab_finalize(
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            if self.neighbor_rebuild_interval > 0:
                self._steps_since_rebuild += 1
                if self._steps_since_rebuild >= self.neighbor_rebuild_interval:
                    self.torch_positions = wp.to_torch(wp_positions)
                    self._rebuild_neighbors()

        self._run_warmup(warmup_step, warmup_steps, "Langevin warmup")

        # Timed loop
        def run_step(step):
            langevin_baoab_half_step(
                wp_positions,
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                wp_temp,
                wp_friction,
                42,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            wrap_positions_to_cell(
                wp_positions,
                cells=self.wp_cell,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )
            _, wp_forces_new, _ = self._compute_forces(wp_positions)
            wp.copy(wp_forces, wp_forces_new)
            langevin_baoab_finalize(
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            if self.neighbor_rebuild_interval > 0:
                self._steps_since_rebuild += 1
                if self._steps_since_rebuild >= self.neighbor_rebuild_interval:
                    self.torch_positions = wp.to_torch(wp_positions)
                    self._rebuild_neighbors()

        total_time, step_times = self._run_timed_loop(run_step, num_steps, log_interval)

        return BenchmarkResult(
            name="langevin",
            backend="nvalchemiops",
            ensemble="NVT",
            num_atoms=self.num_atoms,
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.num_systems if self.is_batched else None,
        )

    def run_nose_hoover(
        self,
        dt: float,
        num_steps: int,
        temperature: float,
        tau: float,
        chain_length: int = 3,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Run NVT simulation with Nosé-Hoover chains.

        Parameters
        ----------
        dt : float
            Timestep in fs.
        num_steps : int
            Number of simulation steps.
        temperature : float
            Target temperature in K.
        tau : float
            Thermostat time constant in fs.
        chain_length : int, optional
            Number of thermostats in chain.
        warmup_steps : int, optional
            Number of warmup steps (excluded from timing).
        log_interval : int, optional
            Logging interval.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.
        """
        from nvalchemiops.dynamics.integrators import (
            nhc_compute_masses,
            nhc_position_update,
            nhc_thermostat_chain_update,
            nhc_velocity_half_step,
        )
        from nvalchemiops.dynamics.utils import (
            compute_cell_inverse,
            wrap_positions_to_cell,
        )

        # Convert temperature to kT (eV)
        kT = temperature * KB_EV

        # Degrees of freedom (per system)
        ndof = 3 * self.num_atoms - 3

        # Initialize thermostat chain variables
        if self.is_batched:
            eta = torch.zeros(
                self.num_systems, chain_length, device=self.device, dtype=self.dtype
            )
            eta_dot = torch.zeros(
                self.num_systems, chain_length, device=self.device, dtype=self.dtype
            )
        else:
            eta = torch.zeros(chain_length, device=self.device, dtype=self.dtype)
            eta_dot = torch.zeros(chain_length, device=self.device, dtype=self.dtype)

        # Compute thermostat masses
        eta_mass = nhc_compute_masses(
            ndof,
            kT,
            tau,
            chain_length,
            num_systems=self.num_systems,
            device=self.wp_device,
            dtype=self.wp_dtype,
        )

        # Convert to warp
        wp_positions = wp.from_torch(
            self.torch_positions.clone(), dtype=self.wp_vec_dtype
        )
        wp_velocities = wp.from_torch(
            self.torch_velocities.clone(), dtype=self.wp_vec_dtype
        )
        wp_dt = wp.array([dt], dtype=self.wp_dtype, device=self.wp_device)
        wp_temp = wp.array([kT], dtype=self.wp_dtype, device=self.wp_device)
        wp_eta = wp.from_torch(eta, dtype=self.wp_dtype)
        wp_eta_dot = wp.from_torch(eta_dot, dtype=self.wp_dtype)
        wp_ndof = wp.array([ndof], dtype=self.wp_dtype, device=self.wp_device)
        # Pre-compute cell inverse
        wp_cell_inv = compute_cell_inverse(self.wp_cell, device=self.wp_device)

        # Initial forces
        _, wp_forces, _ = self._compute_forces(wp_positions)

        # Warmup
        def warmup_step():
            # NHC integration: thermostat -> velocity half -> position -> forces -> velocity half -> thermostat
            nhc_thermostat_chain_update(
                wp_velocities,
                self.wp_masses,
                wp_eta,
                wp_eta_dot,
                eta_mass,
                wp_temp,
                wp_dt,
                wp_ndof,
                nloops=1,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            nhc_velocity_half_step(
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            nhc_position_update(
                wp_positions,
                wp_velocities,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            wrap_positions_to_cell(
                wp_positions,
                cells=self.wp_cell,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )
            _, wp_forces_new, _ = self._compute_forces(wp_positions)
            wp.copy(wp_forces, wp_forces_new)
            nhc_velocity_half_step(
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            nhc_thermostat_chain_update(
                wp_velocities,
                self.wp_masses,
                wp_eta,
                wp_eta_dot,
                eta_mass,
                wp_temp,
                wp_dt,
                wp_ndof,
                nloops=1,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            if self.neighbor_rebuild_interval > 0:
                self._steps_since_rebuild += 1
                if self._steps_since_rebuild >= self.neighbor_rebuild_interval:
                    self.torch_positions = wp.to_torch(wp_positions)
                    self._rebuild_neighbors()

        self._run_warmup(warmup_step, warmup_steps, "Nosé-Hoover warmup")

        # Timed loop
        def run_step(step):
            nhc_thermostat_chain_update(
                wp_velocities,
                self.wp_masses,
                wp_eta,
                wp_eta_dot,
                eta_mass,
                wp_temp,
                wp_dt,
                wp_ndof,
                nloops=1,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            nhc_velocity_half_step(
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            nhc_position_update(
                wp_positions,
                wp_velocities,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            wrap_positions_to_cell(
                wp_positions,
                cells=self.wp_cell,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )
            _, wp_forces_new, _ = self._compute_forces(wp_positions)
            wp.copy(wp_forces, wp_forces_new)
            nhc_velocity_half_step(
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_dt,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            nhc_thermostat_chain_update(
                wp_velocities,
                self.wp_masses,
                wp_eta,
                wp_eta_dot,
                eta_mass,
                wp_temp,
                wp_dt,
                wp_ndof,
                nloops=1,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )
            if self.neighbor_rebuild_interval > 0:
                self._steps_since_rebuild += 1
                if self._steps_since_rebuild >= self.neighbor_rebuild_interval:
                    self.torch_positions = wp.to_torch(wp_positions)
                    self._rebuild_neighbors()

        total_time, step_times = self._run_timed_loop(run_step, num_steps, log_interval)

        return BenchmarkResult(
            name="nose_hoover",
            backend="nvalchemiops",
            ensemble="NVT",
            num_atoms=self.num_atoms,
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.num_systems if self.is_batched else None,
        )

    def run_npt(
        self,
        dt: float,
        num_steps: int,
        temperature: float,
        pressure: float = 0.0,
        tau_t: float = 100.0,
        tau_p: float = 1000.0,
        chain_length: int = 3,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Run NPT simulation with Nosé-Hoover chains + MTK barostat.

        Parameters
        ----------
        dt : float
            Timestep in fs.
        num_steps : int
            Number of simulation steps.
        temperature : float
            Target temperature in K.
        pressure : float, optional
            Target pressure in bar.
        tau_t : float, optional
            Thermostat time constant in fs.
        tau_p : float, optional
            Barostat time constant in fs.
        chain_length : int, optional
            Number of thermostats in chain.
        warmup_steps : int, optional
            Number of warmup steps (excluded from timing).
        log_interval : int, optional
            Logging interval.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.

        Notes
        -----
        - Supports both single-system and batched modes.
        - Virial computation is required for pressure calculation.
        - Neighbor lists are rebuilt every step due to changing cell.
        """
        from nvalchemiops.dynamics.integrators import (
            compute_barostat_mass,
            nhc_compute_masses,
            run_npt_step,
            vec9d,
            vec9f,
        )
        from nvalchemiops.dynamics.utils import (
            compute_cell_inverse,
            wrap_positions_to_cell,
        )

        # Convert units
        kT = temperature * KB_EV
        target_pressure_evA3 = pressure * 6.2415e-7  # bar to eV/Å³

        # Degrees of freedom per system
        ndof = 3 * self.num_atoms - 3

        # Initialize thermostat chain variables
        eta = torch.zeros(
            self.num_systems, chain_length, device=self.device, dtype=self.dtype
        )
        eta_dot = torch.zeros(
            self.num_systems, chain_length, device=self.device, dtype=self.dtype
        )

        # Compute thermostat masses
        thermostat_masses_1d = nhc_compute_masses(
            ndof,
            kT,
            tau_t,
            chain_length,
            num_systems=self.num_systems,
            device=self.wp_device,
            dtype=self.wp_dtype,
        )
        thermostat_masses = thermostat_masses_1d.reshape(
            (self.num_systems, chain_length)
        )

        # Cell velocity (initially zero)
        cell_velocity = torch.zeros(
            self.num_systems, 3, 3, device=self.device, dtype=self.dtype
        )

        # Compute barostat mass
        cell_mass_single = compute_barostat_mass(
            target_temperature=float(kT),
            tau_p=float(tau_p),
            num_atoms=self.num_atoms,
            dtype=self.wp_dtype,
            device=self.wp_device,
        )
        if self.is_batched:
            # For batched, replicate the single mass to all systems
            cell_mass_torch = (
                wp.to_torch(cell_mass_single).expand(self.num_systems).clone()
            )
            cell_mass = wp.from_torch(cell_mass_torch, dtype=self.wp_dtype)
        else:
            cell_mass = cell_mass_single

        # vec9 type
        vec_type = vec9f if self.dtype == torch.float32 else vec9d

        # Convert to warp arrays
        wp_positions = wp.from_torch(
            self.torch_positions.clone(), dtype=self.wp_vec_dtype
        )
        wp_velocities = wp.from_torch(
            self.torch_velocities.clone(), dtype=self.wp_vec_dtype
        )
        wp_cells = wp.from_torch(self.torch_cell.clone(), dtype=self.wp_mat_dtype)
        wp_cell_velocities = wp.from_torch(cell_velocity, dtype=self.wp_mat_dtype)
        wp_eta = wp.from_torch(eta, dtype=self.wp_dtype)
        wp_eta_dot = wp.from_torch(eta_dot, dtype=self.wp_dtype)
        wp_target_temp = wp.array(
            [kT] * self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )
        wp_target_pressure = wp.array(
            [target_pressure_evA3] * self.num_systems,
            dtype=self.wp_dtype,
            device=self.wp_device,
        )
        wp_virial_vec9 = wp.zeros(
            self.num_systems, dtype=vec_type, device=self.wp_device
        )

        # Initial forces with virial
        _, wp_forces, wp_virial_flat = self._compute_forces(
            wp_positions, compute_virial=True
        )

        # Convert flat virial to vec9 without synchronization (negate for sign convention)
        convert_flat_virial_to_vec9(
            wp_virial_flat, wp_virial_vec9, negate=True, device=self.wp_device
        )

        # For batched mode, need num_atoms_per_system array
        if self.is_batched:
            num_atoms_per_system = wp.array(
                [self.num_atoms] * self.num_systems,
                dtype=wp.int32,
                device=self.wp_device,
            )
        else:
            num_atoms_per_system = None

        # Force computation callback for NPT
        def compute_forces_callback(pos, cells, forces_out, virial_out):
            # Update cell and rebuild neighbors
            self.torch_positions = wp.to_torch(pos)
            self.torch_cell = wp.to_torch(cells)
            self.wp_cell = cells
            self._rebuild_neighbors()

            # Compute forces and virial
            _, new_forces, new_virial = self._compute_forces(pos, compute_virial=True)
            wp.copy(forces_out, new_forces)

            # Convert flat virial to vec9 without synchronization (negate for sign convention)
            convert_flat_virial_to_vec9(
                new_virial, virial_out, negate=True, device=self.wp_device
            )

        # Warmup
        def warmup_step():
            run_npt_step(
                wp_positions,
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_cells,
                wp_cell_velocities,
                wp_virial_vec9,
                wp_eta,
                wp_eta_dot,
                thermostat_masses,
                cell_mass,
                wp_target_temp,
                wp_target_pressure,
                self.num_atoms,
                chain_length,
                dt,
                compute_forces_fn=compute_forces_callback,
                batch_idx=self.wp_batch_idx,
                num_atoms_per_system=num_atoms_per_system,
                device=self.wp_device,
            )
            # Wrap positions
            wp_cell_inv = compute_cell_inverse(wp_cells, device=self.wp_device)
            wrap_positions_to_cell(
                wp_positions,
                cells=wp_cells,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )

        self._run_warmup(warmup_step, warmup_steps, "NPT warmup")

        # Timed loop
        def run_step(step):
            run_npt_step(
                wp_positions,
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_cells,
                wp_cell_velocities,
                wp_virial_vec9,
                wp_eta,
                wp_eta_dot,
                thermostat_masses,
                cell_mass,
                wp_target_temp,
                wp_target_pressure,
                self.num_atoms,
                chain_length,
                dt,
                compute_forces_fn=compute_forces_callback,
                batch_idx=self.wp_batch_idx,
                num_atoms_per_system=num_atoms_per_system,
                device=self.wp_device,
            )
            # Wrap positions
            wp_cell_inv = compute_cell_inverse(wp_cells, device=self.wp_device)
            wrap_positions_to_cell(
                wp_positions,
                cells=wp_cells,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )

        total_time, step_times = self._run_timed_loop(run_step, num_steps, log_interval)

        return BenchmarkResult(
            name="npt",
            backend="nvalchemiops",
            ensemble="NPT",
            num_atoms=self.num_atoms,
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.num_systems if self.is_batched else None,
        )

    def run_nph(
        self,
        dt: float,
        num_steps: int,
        pressure: float = 0.0,
        tau_p: float = 1000.0,
        warmup_steps: int = 100,
        log_interval: int = 100,
    ) -> BenchmarkResult:
        """Run NPH simulation with MTK barostat (no thermostat).

        Parameters
        ----------
        dt : float
            Timestep in fs.
        num_steps : int
            Number of simulation steps.
        pressure : float, optional
            Target pressure in bar.
        tau_p : float, optional
            Barostat time constant in fs.
        warmup_steps : int, optional
            Number of warmup steps (excluded from timing).
        log_interval : int, optional
            Logging interval.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.

        Notes
        -----
        - Supports both single-system and batched modes.
        - Virial computation is required for pressure calculation.
        - Neighbor lists are rebuilt every step due to changing cell.
        """
        from nvalchemiops.dynamics.integrators import (
            compute_barostat_mass,
            run_nph_step,
            vec9d,
            vec9f,
        )
        from nvalchemiops.dynamics.utils import (
            compute_cell_inverse,
            wrap_positions_to_cell,
        )

        # Convert pressure
        target_pressure_evA3 = pressure * 6.2415e-7  # bar to eV/Å³

        # Use initial temperature for barostat mass calculation (computed on device)
        velocities_wp = wp.from_torch(self.torch_velocities, dtype=self.wp_vec_dtype)
        vel_torch = wp.to_torch(velocities_wp)
        ke = 0.5 * (self.torch_masses.unsqueeze(1) * (vel_torch**2)).sum()
        initial_temp = 2.0 * ke / (3.0 * self.total_atoms * KB_EV)
        kT = initial_temp * KB_EV

        # Cell velocity (initially zero)
        cell_velocity = torch.zeros(
            self.num_systems, 3, 3, device=self.device, dtype=self.dtype
        )

        # Compute barostat mass
        cell_mass_single = compute_barostat_mass(
            target_temperature=float(kT.item()),
            tau_p=float(tau_p),
            num_atoms=self.num_atoms,
            dtype=self.wp_dtype,
            device=self.wp_device,
        )
        if self.is_batched:
            # For batched, replicate the single mass to all systems
            cell_mass_torch = (
                wp.to_torch(cell_mass_single).expand(self.num_systems).clone()
            )
            cell_mass = wp.from_torch(cell_mass_torch, dtype=self.wp_dtype)
        else:
            cell_mass = cell_mass_single

        # vec9 type
        vec_type = vec9f if self.dtype == torch.float32 else vec9d

        # Convert to warp arrays
        wp_positions = wp.from_torch(
            self.torch_positions.clone(), dtype=self.wp_vec_dtype
        )
        wp_velocities = wp.from_torch(
            self.torch_velocities.clone(), dtype=self.wp_vec_dtype
        )
        wp_cells = wp.from_torch(self.torch_cell.clone(), dtype=self.wp_mat_dtype)
        wp_cell_velocities = wp.from_torch(cell_velocity, dtype=self.wp_mat_dtype)
        wp_target_pressure = wp.array(
            [target_pressure_evA3] * self.num_systems,
            dtype=self.wp_dtype,
            device=self.wp_device,
        )
        wp_virial_vec9 = wp.zeros(
            self.num_systems, dtype=vec_type, device=self.wp_device
        )

        # Initial forces with virial
        _, wp_forces, wp_virial_flat = self._compute_forces(
            wp_positions, compute_virial=True
        )

        # Convert flat virial to vec9 without synchronization (negate for sign convention)
        convert_flat_virial_to_vec9(
            wp_virial_flat, wp_virial_vec9, negate=True, device=self.wp_device
        )

        # For batched mode, need num_atoms_per_system array
        if self.is_batched:
            num_atoms_per_system = wp.array(
                [self.num_atoms] * self.num_systems,
                dtype=wp.int32,
                device=self.wp_device,
            )
        else:
            num_atoms_per_system = None

        # Force computation callback for NPH
        def compute_forces_callback(pos, cells, forces_out, virial_out):
            # Update cell and rebuild neighbors
            self.torch_positions = wp.to_torch(pos)
            self.torch_cell = wp.to_torch(cells)
            self.wp_cell = cells
            self._rebuild_neighbors()

            # Compute forces and virial
            _, new_forces, new_virial = self._compute_forces(pos, compute_virial=True)
            wp.copy(forces_out, new_forces)

            # Convert flat virial to vec9 without synchronization (negate for sign convention)
            convert_flat_virial_to_vec9(
                new_virial, virial_out, negate=True, device=self.wp_device
            )

        # Warmup
        def warmup_step():
            run_nph_step(
                wp_positions,
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_cells,
                wp_cell_velocities,
                wp_virial_vec9,
                cell_mass,
                wp_target_pressure,
                self.num_atoms,
                dt,
                compute_forces_fn=compute_forces_callback,
                batch_idx=self.wp_batch_idx,
                num_atoms_per_system=num_atoms_per_system,
                device=self.wp_device,
            )
            # Wrap positions
            wp_cell_inv = compute_cell_inverse(wp_cells, device=self.wp_device)
            wrap_positions_to_cell(
                wp_positions,
                cells=wp_cells,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )

        self._run_warmup(warmup_step, warmup_steps, "NPH warmup")

        # Timed loop
        def run_step(step):
            run_nph_step(
                wp_positions,
                wp_velocities,
                wp_forces,
                self.wp_masses,
                wp_cells,
                wp_cell_velocities,
                wp_virial_vec9,
                cell_mass,
                wp_target_pressure,
                self.num_atoms,
                dt,
                compute_forces_fn=compute_forces_callback,
                batch_idx=self.wp_batch_idx,
                num_atoms_per_system=num_atoms_per_system,
                device=self.wp_device,
            )
            # Wrap positions
            wp_cell_inv = compute_cell_inverse(wp_cells, device=self.wp_device)
            wrap_positions_to_cell(
                wp_positions,
                cells=wp_cells,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )

        total_time, step_times = self._run_timed_loop(run_step, num_steps, log_interval)

        return BenchmarkResult(
            name="nph",
            backend="nvalchemiops",
            ensemble="NPH",
            num_atoms=self.num_atoms,
            num_steps=num_steps,
            dt=dt,
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.num_systems if self.is_batched else None,
        )

    # ========================================================================
    # Optimization
    # ========================================================================

    def run_fire(
        self,
        max_steps: int = 1000,
        force_tolerance: float = 0.01,
        dt_start: float = 1.0,
        dt_max: float = 10.0,
        dt_min: float = 0.001,
        alpha_start: float = 0.1,
        n_min: int = 5,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        f_alpha: float = 0.99,
        maxstep: float = 0.2,
        warmup_steps: int = 0,
        log_interval: int = 100,
        check_interval: int = 20,
    ) -> BenchmarkResult:
        """Run FIRE (Fast Inertial Relaxation Engine) geometry optimization.

        Parameters
        ----------
        max_steps : int, optional
            Maximum number of optimization steps.
        force_tolerance : float, optional
            Convergence criterion for maximum force magnitude (eV/Å).
        dt_start : float, optional
            Initial timestep in fs.
        dt_max : float, optional
            Maximum timestep in fs.
        dt_min : float, optional
            Minimum timestep in fs.
        alpha_start : float, optional
            Initial mixing parameter.
        n_min : int, optional
            Minimum steps before increasing dt.
        f_inc : float, optional
            Factor to increase dt.
        f_dec : float, optional
            Factor to decrease dt.
        f_alpha : float, optional
            Factor to decrease alpha.
        maxstep : float, optional
            Maximum position change per step (Å).
        warmup_steps : int, optional
            Number of warmup steps (usually 0 for optimization).
        log_interval : int, optional
            Logging interval.
        check_interval : int, optional
            Interval to check convergence.

        Returns
        -------
        BenchmarkResult
            Benchmark result with timing and metadata.

        Notes
        -----
        - FIRE does not require velocities to be initialized.
        - Convergence is checked based on maximum force magnitude.
        - Supports both single-system and batched modes.
        - Uses optimized nvalchemiops FIRE kernels.
        """
        from nvalchemiops.dynamics.optimizers.fire import fire_step
        from nvalchemiops.dynamics.utils import (
            compute_cell_inverse,
            wrap_positions_to_cell,
        )

        # Initialize velocities to zero
        velocities = torch.zeros_like(self.torch_positions)

        # Convert to warp
        wp_positions = wp.from_torch(
            self.torch_positions.clone(), dtype=self.wp_vec_dtype
        )
        wp_velocities = wp.from_torch(velocities, dtype=self.wp_vec_dtype)

        # Pre-compute cell inverse
        wp_cell_inv = compute_cell_inverse(self.wp_cell, device=self.wp_device)

        # Initialize FIRE control parameters as warp arrays
        wp_alpha = wp.array(
            [alpha_start] * self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )
        wp_dt = wp.array(
            [dt_start] * self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )
        wp_alpha_start = wp.array(
            [alpha_start] * self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )
        wp_f_alpha = wp.array(
            [f_alpha] * self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )
        wp_dt_min = wp.array(
            [dt_min] * self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )
        wp_dt_max = wp.array(
            [dt_max] * self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )
        wp_maxstep = wp.array(
            [maxstep] * self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )
        wp_n_steps_positive = wp.zeros(
            self.num_systems, dtype=wp.int32, device=self.wp_device
        )
        wp_n_min = wp.array(
            [n_min] * self.num_systems, dtype=wp.int32, device=self.wp_device
        )
        wp_f_dec = wp.array(
            [f_dec] * self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )
        wp_f_inc = wp.array(
            [f_inc] * self.num_systems, dtype=self.wp_dtype, device=self.wp_device
        )

        # Accumulators (required for single/batch_idx modes)
        wp_vf = wp.zeros(self.num_systems, dtype=self.wp_dtype, device=self.wp_device)
        wp_vv = wp.zeros(self.num_systems, dtype=self.wp_dtype, device=self.wp_device)
        wp_ff = wp.zeros(self.num_systems, dtype=self.wp_dtype, device=self.wp_device)

        # Initial forces
        _, wp_forces, _ = self._compute_forces(wp_positions)

        # Warmup (usually not needed for optimization, but included for API consistency)
        if warmup_steps > 0:

            def warmup_step():
                pass  # No warmup needed for FIRE

            self._run_warmup(warmup_step, warmup_steps, "FIRE warmup")

        # Timed loop
        import time

        wp.synchronize()
        start_time = time.perf_counter()

        actual_steps = 0
        for step in range(max_steps):
            actual_steps += 1
            # Check convergence periodically
            if step % check_interval == 0:
                forces_torch = wp.to_torch(wp_forces)
                max_force = torch.abs(forces_torch).max().item()
                if max_force < force_tolerance:
                    break

            # Zero accumulators before FIRE step
            wp_vf.zero_()
            wp_vv.zero_()
            wp_ff.zero_()

            # FIRE step (includes MD integration + parameter update)
            fire_step(
                positions=wp_positions,
                velocities=wp_velocities,
                forces=wp_forces,
                masses=self.wp_masses,
                alpha=wp_alpha,
                dt=wp_dt,
                alpha_start=wp_alpha_start,
                f_alpha=wp_f_alpha,
                dt_min=wp_dt_min,
                dt_max=wp_dt_max,
                maxstep=wp_maxstep,
                n_steps_positive=wp_n_steps_positive,
                n_min=wp_n_min,
                f_dec=wp_f_dec,
                f_inc=wp_f_inc,
                vf=wp_vf,
                vv=wp_vv,
                ff=wp_ff,
                atom_ptr=self.wp_atom_ptr,
                device=self.wp_device,
            )

            # Wrap positions back into cell
            wrap_positions_to_cell(
                wp_positions,
                cells=self.wp_cell,
                cells_inv=wp_cell_inv,
                device=self.wp_device,
            )

            # Compute new forces
            _, wp_forces, _ = self._compute_forces(wp_positions)

            # Rebuild neighbors if needed
            if self.neighbor_rebuild_interval > 0:
                self._steps_since_rebuild += 1
                if self._steps_since_rebuild >= self.neighbor_rebuild_interval:
                    self.torch_positions = wp.to_torch(wp_positions)
                    self._rebuild_neighbors()

        wp.synchronize()
        total_time = time.perf_counter() - start_time
        step_times = [total_time / actual_steps] * actual_steps

        actual_steps = len(step_times)

        return BenchmarkResult(
            name="fire",
            backend="nvalchemiops",
            ensemble="optimization",
            num_atoms=self.num_atoms,
            num_steps=actual_steps,
            dt=dt_start,  # Report initial dt
            warmup_steps=warmup_steps,
            total_time=total_time,
            step_times=step_times,
            batch_size=self.num_systems if self.is_batched else None,
        )
