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
Utility classes for Langevin dynamics examples.

This module provides helper classes for:
- Neighbor list management with cell list algorithm
- System creation (FCC lattice)
- Statistical analysis utilities
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np
import torch
import warp as wp

from nvalchemiops.dynamics.integrators import (
    langevin_baoab_finalize,
    langevin_baoab_half_step,
)
from nvalchemiops.dynamics.utils import (
    compute_cell_inverse,
    compute_cell_volume,
    compute_kinetic_energy,
    compute_temperature,
    initialize_velocities,
    wrap_positions_to_cell,
)
from nvalchemiops.interactions import lj_energy_forces, lj_energy_forces_virial
from nvalchemiops.torch.neighbors import cell_list
from nvalchemiops.torch.neighbors.rebuild_detection import neighbor_list_needs_rebuild

# ==============================================================================
# Physical Constants
# ==============================================================================

# Boltzmann constant in eV/K
KB_EV = 8.617333262e-5

# ------------------------------------------------------------------------------
# Unit conventions used by this example
# ------------------------------------------------------------------------------
#
# The nvalchemiops dynamics kernels are *unitless*: they assume the caller uses a
# self-consistent unit system for (x, v, t, m, E, F).
#
# In this example we choose:
# - length:        Angstrom (Å)
# - time:          femtosecond (fs)
# - energy:        electron-volt (eV)
# - mass (internal): eV * fs^2 / Å^2   (so that KE = 0.5 * m * v^2 is in eV when
#                                     v is in Å/fs)
#
# We accept user-facing masses in amu and convert them to the internal mass unit.
#
# Derivation:
#   1 [eV * fs^2 / Å^2] in SI is:
#     (eV→J) * (fs→s)^2 / (Å→m)^2 = kg
#   so:
#     1 amu = amu_kg / (eV*fs^2/Å^2)_kg
#
EV_TO_J = 1.602176634e-19
AMU_TO_KG = 1.66053906660e-27
FS_TO_S = 1.0e-15
ANGSTROM_TO_M = 1.0e-10

_MASS_UNIT_KG = EV_TO_J * (FS_TO_S**2) / (ANGSTROM_TO_M**2)  # kg per (eV*fs^2/Å^2)
AMU_TO_EV_FS2_PER_A2 = AMU_TO_KG / _MASS_UNIT_KG  # ~103.642691


def mass_amu_to_internal(mass_amu: np.ndarray) -> np.ndarray:
    """Convert masses from amu to internal units (eV*fs^2/Å^2)."""

    return mass_amu * AMU_TO_EV_FS2_PER_A2


# Argon LJ parameters (commonly used for liquid argon simulations)
EPSILON_AR = 0.0104  # eV
SIGMA_AR = 3.40  # Angstrom
MASS_AR = 39.948  # amu

# Default cutoff (2.5*sigma is typical for LJ)
DEFAULT_CUTOFF = 2.5 * SIGMA_AR  # ~8.5 Angstrom
DEFAULT_SKIN = 0.5  # Angstrom

# Pressure unit conversion (GPa)
_EV_PER_A3_TO_PA = EV_TO_J / (ANGSTROM_TO_M**3)  # ~1.602e11 Pa


def pressure_gpa_to_ev_per_a3(p_gpa: float) -> float:
    """Convert pressure from GPa to eV/Å³."""
    return p_gpa * 1e9 / _EV_PER_A3_TO_PA


def pressure_ev_per_a3_to_gpa(p_ev: float) -> float:
    """Convert pressure from eV/Å³ to GPa."""
    return p_ev * _EV_PER_A3_TO_PA / 1e9


# ==============================================================================
# Virial to Stress Conversion
# ==============================================================================


@wp.kernel
def _virial_to_stress_kernel(
    virial_flat: wp.array(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    external_pressure: wp.float64,
    stress: wp.array(dtype=wp.mat33d),
):
    """Convert virial to stress tensor for cell optimization.

    Computes: stress = P_ext - P_internal

    where P_internal = -virial/V (negated because LJ virial has W = -Σ r ⊗ F).

    At equilibrium: P_internal = P_ext, so stress = 0.
    When P_internal < P_ext: stress > 0, cell contracts (via stress_to_cell_force).
    When P_internal > P_ext: stress < 0, cell expands.
    """
    sys = wp.tid()
    V = volume[sys]
    inv_V = wp.float64(1.0) / V

    # Virial components (row-major: xx, xy, xz, yx, yy, yz, zx, zy, zz)
    # Negate because LJ virial uses convention W = -Σ r ⊗ F
    # After negation: positive = compression (repulsive forces)
    vxx = -virial_flat[0]
    vxy = -virial_flat[1]
    vxz = -virial_flat[2]
    vyx = -virial_flat[3]
    vyy = -virial_flat[4]
    vyz = -virial_flat[5]
    vzx = -virial_flat[6]
    vzy = -virial_flat[7]
    vzz = -virial_flat[8]

    # Internal pressure (diagonal): P_int = virial/V (positive = compression)
    # Stress for optimization: σ = P_ext - P_int
    # This ensures σ = 0 at equilibrium, and correct sign for cell force
    stress[sys] = wp.mat33d(
        external_pressure - vxx * inv_V,
        -vxy * inv_V,
        -vxz * inv_V,
        -vyx * inv_V,
        external_pressure - vyy * inv_V,
        -vyz * inv_V,
        -vzx * inv_V,
        -vzy * inv_V,
        external_pressure - vzz * inv_V,
    )


def virial_to_stress(
    virial_flat: wp.array,
    cell: wp.array,
    external_pressure: float,
    device: str,
) -> wp.array:
    """Convert virial tensor to stress with external pressure.

    Parameters
    ----------
    virial_flat : wp.array, shape (9,)
        Flat virial tensor from LJ computation.
    cell : wp.array, shape (1, 3, 3)
        Cell matrix.
    external_pressure : float
        External pressure in eV/Å³.
    device : str
        Warp device.

    Returns
    -------
    stress : wp.array, shape (1,), dtype=mat33d
        Stress tensor for cell optimization.
    """
    volume = wp.empty(1, dtype=wp.float64, device=device)
    compute_cell_volume(cell, volumes=volume, device=device)
    stress = wp.zeros(1, dtype=wp.mat33d, device=device)
    wp.launch(
        _virial_to_stress_kernel,
        dim=1,
        inputs=[virial_flat, volume, wp.float64(external_pressure)],
        outputs=[stress],
        device=device,
    )
    return stress


# ==============================================================================
# FCC Lattice Creation
# ==============================================================================


def create_fcc_lattice(n_cells: int, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Create FCC lattice with n_cells unit cells per dimension.

    Parameters
    ----------
    n_cells : int
        Number of unit cells in each dimension.
    a : float
        Lattice constant (Å).

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Atomic positions.
    cell : np.ndarray, shape (3, 3)
        Cell matrix.
    """
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )

    positions = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for b in basis:
                    pos = (np.array([i, j, k]) + b) * a
                    positions.append(pos)

    positions = np.array(positions, dtype=np.float64)
    cell = np.eye(3, dtype=np.float64) * (n_cells * a)

    return positions, cell


# ==============================================================================
# Data Structures
# ==============================================================================


class SimulationStats(NamedTuple):
    """Statistics for a simulation step."""

    step: int
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    temperature: float
    num_neighbors: int
    min_neighbor_distance: float
    max_force: float
    time_per_step_ms: float


def neighbor_distance_stats(
    positions: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    fill_value: int,
) -> float:
    """Compute minimum neighbor distance using the current neighbor matrix.

    Notes
    -----
    - Assumes neighbor shifts are integer cell shifts for the *neighbor* atom.
    - Computes distances only over valid neighbor entries (j < fill_value).
    - Uses the same shift convention as the LJ kernel: shift_vec = shift @ cell.
    """

    device = positions.device
    n, max_neighbors = neighbor_matrix.shape

    k = torch.arange(max_neighbors, device=device).view(1, -1).expand(n, -1)
    valid = k < num_neighbors.view(-1, 1)
    j = neighbor_matrix
    valid = valid & (j < fill_value)

    if not torch.any(valid):
        return float("inf")

    ii, kk = torch.where(valid)
    jj = j[ii, kk]

    ri = positions[ii]
    rj = positions[jj]
    shifts = neighbor_matrix_shifts[ii, kk].to(dtype=positions.dtype)

    # shift_vec = shift @ cell  (matches: cell_t * shift in Warp kernel)
    shift_vec = shifts @ cell
    rij = ri - rj - shift_vec
    r = torch.linalg.norm(rij, dim=1)
    return float(r.min().item())


# ==============================================================================
# Neighbor List Management
# ==============================================================================


class NeighborListManager:
    """Manages neighbor list construction and updates.

    Uses the cell list algorithm for O(N) neighbor finding with periodic
    boundary conditions.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the system.
    cutoff : float
        Cutoff distance for neighbor detection (Angstrom).
    skin : float
        Neighbor list skin distance (Angstrom). Rebuild when any atom
        moves more than skin/2.
    max_neighbors : int, optional
        Maximum number of neighbors per atom.
    device : str, optional
        Device for warp arrays (e.g., "cuda:0", "cpu").
    """

    def __init__(
        self,
        num_atoms: int,
        cutoff: float,
        skin: float,
        max_neighbors: int = 100,
        half_fill: bool = True,
        device: str = "cuda:0",
    ):
        self.num_atoms = num_atoms
        self.cutoff = cutoff
        self.skin = skin
        self.max_neighbors = max_neighbors
        self.half_fill = half_fill
        self.device = device

        # Track positions at last rebuild
        self.positions_at_rebuild: torch.Tensor | None = None

        # Torch tensors (from cell_list)
        self.torch_neighbor_matrix: torch.Tensor | None = None
        self.torch_neighbor_shifts: torch.Tensor | None = None
        self.torch_num_neighbors: torch.Tensor | None = None

        # Warp arrays (converted from torch)
        self.wp_neighbor_matrix: wp.array | None = None
        self.wp_neighbor_shifts: wp.array | None = None
        self.wp_num_neighbors: wp.array | None = None

    def needs_rebuild(self, positions: torch.Tensor) -> bool:
        """Check if neighbor list needs rebuilding.

        Parameters
        ----------
        positions : torch.Tensor, shape (N, 3)
            Current atomic positions.

        Returns
        -------
        bool
            True if rebuild needed (any atom moved > skin/2).
        """
        if self.positions_at_rebuild is None:
            return True

        # Use the package's GPU rebuild detection kernel (early-termination).
        # Note: the returned tensor is on-device; calling .item() syncs, but this
        # is still cheaper than a full max-reduction in Python.
        rebuild = neighbor_list_needs_rebuild(
            reference_positions=self.positions_at_rebuild,
            current_positions=positions,
            skin_distance_threshold=float(self.skin / 2.0),
        )
        return bool(rebuild.item())

    def build(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
    ) -> None:
        """Build neighbor list from positions.

        Parameters
        ----------
        positions : torch.Tensor, shape (N, 3)
            Atomic positions.
        cell : torch.Tensor, shape (3, 3)
            Unit cell matrix.
        pbc : torch.Tensor, shape (3,), dtype=bool
            Periodic boundary conditions in each dimension.
        """
        # Use cell_list for efficient neighbor finding
        neighbor_matrix, num_neighbors, neighbor_shifts = cell_list(
            positions=positions,
            cutoff=self.cutoff + self.skin,
            cell=cell,
            pbc=pbc,
            max_neighbors=self.max_neighbors,
            half_fill=self.half_fill,
            fill_value=self.num_atoms,
        )

        self.torch_neighbor_matrix = neighbor_matrix
        self.torch_neighbor_shifts = neighbor_shifts
        self.torch_num_neighbors = num_neighbors
        self.positions_at_rebuild = positions.clone()

        # Convert to warp arrays
        self._convert_to_warp()

    def _convert_to_warp(self) -> None:
        """Convert torch tensors to warp arrays."""
        # Neighbor matrix (int32)
        self.wp_neighbor_matrix = wp.from_torch(
            self.torch_neighbor_matrix, dtype=wp.int32
        )

        # Shifts (vec3i)
        self.wp_neighbor_shifts = wp.from_torch(
            self.torch_neighbor_shifts, dtype=wp.vec3i
        )

        # Num neighbors (int32)
        self.wp_num_neighbors = wp.from_torch(self.torch_num_neighbors, dtype=wp.int32)

    def total_neighbors(self) -> int:
        """Get total number of neighbors across all atoms."""
        if self.torch_num_neighbors is None:
            return 0
        return int(self.torch_num_neighbors.sum().item())


# ==============================================================================
# System Creation
# ==============================================================================


def create_fcc_argon(
    num_unit_cells: int = 4, a: float = 5.26
) -> tuple[np.ndarray, np.ndarray]:
    """Create FCC argon lattice.

    Creates a face-centered cubic (FCC) lattice with 4 atoms per unit cell,
    typical for noble gas crystals near the triple point.

    Parameters
    ----------
    num_unit_cells : int
        Number of unit cells in each dimension. Total atoms = 4 * n^3.
    a : float
        Lattice constant in Angstrom. For argon at 94K: ~5.26 Å

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Atomic positions in Angstrom.
    cell : np.ndarray, shape (3, 3)
        Unit cell matrix (diagonal).
    """
    # FCC basis (4 atoms per unit cell)
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )

    # Generate all positions
    positions = []
    for i in range(num_unit_cells):
        for j in range(num_unit_cells):
            for k in range(num_unit_cells):
                for b in basis:
                    pos = (np.array([i, j, k]) + b) * a
                    positions.append(pos)

    positions = np.array(positions, dtype=np.float64)

    # Create cell matrix (cubic)
    L = num_unit_cells * a
    cell = np.eye(3, dtype=np.float64) * L

    return positions, cell


# ==============================================================================
# Core MD System (integrator-agnostic)
# ==============================================================================


class MDSystem:
    """Integrator-agnostic molecular dynamics system.

    This class owns the *state* needed for many MD algorithms:
    - positions / velocities / forces / masses (Warp arrays)
    - cell + inverse (Warp arrays)
    - neighbor list manager (Torch → Warp conversion)
    - LJ force evaluation (via :func:`nvalchemiops.interactions.lj_energy_forces`)

    Integrators (Langevin, Velocity Verlet, NPT, NPH, ...) should be implemented
    as separate "runner" functions operating on this system.
    """

    def __init__(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        masses: np.ndarray | None = None,
        epsilon: float = EPSILON_AR,
        sigma: float = SIGMA_AR,
        cutoff: float = DEFAULT_CUTOFF,
        skin: float = DEFAULT_SKIN,
        switch_width: float = 0.0,
        half_neighbor_list: bool = True,
        device: str = "cuda:0",
        dtype: np.dtype = np.float64,
    ):
        self.num_atoms = len(positions)
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.switch_width = float(switch_width)
        self.half_neighbor_list = bool(half_neighbor_list)
        self.device = device

        # Determine types
        self.dtype = dtype
        self.wp_dtype = wp.float64 if dtype == np.float64 else wp.float32
        self.wp_vec_dtype = wp.vec3d if dtype == np.float64 else wp.vec3f
        self.wp_mat_dtype = wp.mat33d if dtype == np.float64 else wp.mat33f
        self.torch_dtype = torch.float64 if dtype == np.float64 else torch.float32

        # Set up masses
        if masses is None:
            masses = np.full(self.num_atoms, MASS_AR, dtype=dtype)
        else:
            masses = masses.astype(dtype)

        # Convert masses to internal MD units (so KE is in eV when v is Å/fs)
        masses = mass_amu_to_internal(masses)

        # Create torch tensors (for neighbor list - uses PyTorch interface)
        self.torch_positions = torch.tensor(
            positions, dtype=self.torch_dtype, device=device
        )
        self.torch_cell = torch.tensor(cell, dtype=self.torch_dtype, device=device)
        self.torch_pbc = torch.tensor([True, True, True], device=device)

        # Create warp arrays for dynamics
        self.wp_positions = wp.array(
            positions.astype(dtype), dtype=self.wp_vec_dtype, device=device
        )
        self.wp_velocities = wp.zeros(
            self.num_atoms, dtype=self.wp_vec_dtype, device=device
        )
        self.wp_forces = wp.zeros(
            self.num_atoms, dtype=self.wp_vec_dtype, device=device
        )
        self.wp_masses = wp.array(masses, dtype=self.wp_dtype, device=device)

        # Cell matrix (shape (1,) for single system)
        cell_reshaped = cell.reshape(1, 3, 3).astype(dtype)
        self.wp_cell = wp.array(cell_reshaped, dtype=self.wp_mat_dtype, device=device)

        # Compute cell inverse for position wrapping
        self.wp_cell_inv = wp.empty_like(self.wp_cell)
        compute_cell_inverse(self.wp_cell, self.wp_cell_inv, device=device)

        # Set up neighbor list manager
        self.neighbor_manager = NeighborListManager(
            num_atoms=self.num_atoms,
            cutoff=cutoff,
            skin=skin,
            device=device,
            half_fill=self.half_neighbor_list,
        )

        # Build initial neighbor list
        self._rebuild_neighbors()

        print(f"Initialized MD system with {self.num_atoms} atoms")
        print(f"  Cell: {cell[0, 0]:.2f} x {cell[1, 1]:.2f} x {cell[2, 2]:.2f} Å")
        print(f"  Cutoff: {cutoff:.2f} Å (+ {skin:.2f} Å skin)")
        print(f"  LJ: ε = {epsilon:.4f} eV, σ = {sigma:.2f} Å")
        print(f"  Device: {device}, dtype: {dtype}")
        print("  Units: x [Å], t [fs], E [eV], m [eV·fs²/Å²] (from amu), v [Å/fs]")

    def _rebuild_neighbors(self) -> None:
        """Rebuild neighbor list from current positions."""
        # Sync warp positions to torch
        self.torch_positions = wp.to_torch(self.wp_positions)

        self.neighbor_manager.build(
            positions=self.torch_positions,
            cell=self.torch_cell,
            pbc=self.torch_pbc,
        )

    def _check_rebuild(self) -> bool:
        """Check if neighbor list needs rebuilding and rebuild if so."""
        self.torch_positions = wp.to_torch(self.wp_positions)

        if self.neighbor_manager.needs_rebuild(self.torch_positions):
            self._rebuild_neighbors()
            return True
        return False

    def compute_forces(self) -> wp.array:
        """Compute LJ forces and return per-atom potential energies (device array).

        Notes
        -----
        This function intentionally does **not** synchronize or pull data back to
        the host. Host-side reductions (e.g., PE sum) should be done only at
        logging / analysis points.
        """
        # Check neighbor list
        self._check_rebuild()

        # Compute LJ energy and forces using the interactions module
        wp_energies, wp_forces = lj_energy_forces(
            positions=self.wp_positions,
            cell=self.wp_cell,
            epsilon=self.epsilon,
            sigma=self.sigma,
            cutoff=self.cutoff,
            switch_width=self.switch_width,
            half_neighbor_list=self.half_neighbor_list,
            neighbor_matrix=self.neighbor_manager.wp_neighbor_matrix,
            neighbor_matrix_shifts=self.neighbor_manager.wp_neighbor_shifts,
            num_neighbors=self.neighbor_manager.wp_num_neighbors,
            fill_value=self.num_atoms,
            device=self.device,
        )

        # Copy computed forces to our force array
        wp.copy(self.wp_forces, wp_forces)
        return wp_energies

    def compute_forces_virial(self) -> tuple[wp.array, wp.array, wp.array]:
        """Compute LJ forces, energies, and virial tensor.

        Returns
        -------
        energies : wp.array, shape (num_atoms,)
            Per-atom potential energies.
        forces : wp.array, shape (num_atoms,), dtype=vec3d
            Forces on atoms.
        virial : wp.array, shape (9,)
            Flat virial tensor (row-major).
        """
        # Check neighbor list
        self._check_rebuild()

        # Compute LJ energy, forces, and virial using the interactions module
        wp_energies, wp_forces, wp_virial = lj_energy_forces_virial(
            positions=self.wp_positions,
            cell=self.wp_cell,
            epsilon=self.epsilon,
            sigma=self.sigma,
            cutoff=self.cutoff,
            switch_width=self.switch_width,
            half_neighbor_list=self.half_neighbor_list,
            neighbor_matrix=self.neighbor_manager.wp_neighbor_matrix,
            neighbor_matrix_shifts=self.neighbor_manager.wp_neighbor_shifts,
            num_neighbors=self.neighbor_manager.wp_num_neighbors,
            fill_value=self.num_atoms,
            device=self.device,
        )

        # Copy computed forces to our force array
        wp.copy(self.wp_forces, wp_forces)
        return wp_energies, wp_forces, wp_virial

    def update_cell(self, cell: wp.array) -> None:
        """Update cell matrix and recompute cell inverse.

        Parameters
        ----------
        cell : wp.array, shape (1, 3, 3)
            New cell matrix.
        """
        wp.copy(self.wp_cell, cell)
        compute_cell_inverse(self.wp_cell, self.wp_cell_inv, device=self.device)
        # Update torch cell for neighbor list
        self.torch_cell = wp.to_torch(self.wp_cell).squeeze(0)
        # Force neighbor rebuild on next force computation
        self.neighbor_manager.positions_at_rebuild = None

    def kinetic_energy(self) -> wp.array:
        """Compute kinetic energy on device (shape (1,), in eV)."""
        ke = wp.zeros(1, dtype=self.wp_dtype, device=self.device)
        compute_kinetic_energy(
            velocities=self.wp_velocities,
            masses=self.wp_masses,
            kinetic_energy=ke,
            device=self.device,
        )
        return ke

    def temperature_kT(self) -> wp.array:
        """Compute instantaneous temperature on device (kB*T in eV, shape (1,))."""
        ke = self.kinetic_energy()
        temp = wp.zeros(1, dtype=self.wp_dtype, device=self.device)
        compute_temperature(
            kinetic_energy=ke,
            temperature=temp,
            num_atoms=self.num_atoms,
            device=self.device,
        )
        return temp

    def initialize_temperature(self, temperature: float, seed: int = 42) -> None:
        """Initialize velocities to target temperature.

        Parameters
        ----------
        temperature : float
            Target temperature in Kelvin.
        seed : int
            Random seed for reproducibility.
        """
        # Convert to kT in eV (our energy units)
        kT = temperature * KB_EV

        # Create temperature array (shape (1,) for single system)
        wp_temperature = wp.array([kT], dtype=self.wp_dtype, device=self.device)

        # Scratch arrays for COM removal
        wp_total_momentum = wp.zeros(1, dtype=self.wp_vec_dtype, device=self.device)
        wp_total_mass = wp.zeros(1, dtype=self.wp_dtype, device=self.device)
        wp_com_velocities = wp.zeros(1, dtype=self.wp_vec_dtype, device=self.device)

        # Initialize velocities from Maxwell-Boltzmann distribution
        initialize_velocities(
            velocities=self.wp_velocities,
            masses=self.wp_masses,
            temperature=wp_temperature,
            total_momentum=wp_total_momentum,
            total_mass=wp_total_mass,
            com_velocities=wp_com_velocities,
            random_seed=seed,
            remove_com=True,
            device=self.device,
        )

        # Verify temperature (one-time host read for user feedback)
        temp_arr = self.temperature_kT()
        actual_kT = float(temp_arr.numpy()[0])
        actual_temp = actual_kT / KB_EV

        print(
            f"Initialized velocities: target={temperature:.1f} K, actual={actual_temp:.1f} K"
        )


def run_langevin_baoab(
    system: MDSystem,
    num_steps: int,
    dt_fs: float,
    temperature_K: float,
    friction_per_fs: float,
    log_interval: int = 100,
    seed: int = 42,
) -> list[SimulationStats]:
    """Run Langevin (BAOAB) dynamics on an :class:`MDSystem`."""
    kT = temperature_K * KB_EV

    wp_dt = wp.array([dt_fs], dtype=system.wp_dtype, device=system.device)
    wp_temperature = wp.array([kT], dtype=system.wp_dtype, device=system.device)
    wp_friction = wp.array(
        [friction_per_fs], dtype=system.wp_dtype, device=system.device
    )

    wp_energies = system.compute_forces()
    stats_history: list[SimulationStats] = []

    print(f"\nRunning {num_steps} Langevin steps at T={temperature_K:.1f} K")
    print(f"  dt = {dt_fs:.3f} fs, friction = {friction_per_fs:.4f} 1/fs")
    print_header()

    step_start = time.perf_counter()

    for step in range(num_steps):
        langevin_baoab_half_step(
            positions=system.wp_positions,
            velocities=system.wp_velocities,
            forces=system.wp_forces,
            masses=system.wp_masses,
            dt=wp_dt,
            temperature=wp_temperature,
            friction=wp_friction,
            random_seed=seed + step,
            device=system.device,
        )

        wrap_positions_to_cell(
            positions=system.wp_positions,
            cells=system.wp_cell,
            cells_inv=system.wp_cell_inv,
            device=system.device,
        )

        wp_energies = system.compute_forces()
        ke_arr = system.kinetic_energy()

        langevin_baoab_finalize(
            velocities=system.wp_velocities,
            forces_new=system.wp_forces,
            masses=system.wp_masses,
            dt=wp_dt,
            device=system.device,
        )

        if step % log_interval == 0 or step == num_steps - 1:
            elapsed = time.perf_counter() - step_start
            ms_per_step = elapsed * 1000 / max(1, log_interval)

            temp_arr = system.temperature_kT()

            temp_K = float(temp_arr.numpy()[0]) / KB_EV
            pe = float(wp_energies.numpy().sum())
            ke = float(ke_arr.numpy()[0])

            max_force = float(
                torch.linalg.norm(wp.to_torch(system.wp_forces), dim=1).max().item()
            )
            min_r = neighbor_distance_stats(
                positions=wp.to_torch(system.wp_positions),
                cell=system.torch_cell,
                neighbor_matrix=system.neighbor_manager.torch_neighbor_matrix,
                neighbor_matrix_shifts=system.neighbor_manager.torch_neighbor_shifts,
                num_neighbors=system.neighbor_manager.torch_num_neighbors,
                fill_value=system.num_atoms,
            )

            stats = SimulationStats(
                step=step,
                kinetic_energy=ke,
                potential_energy=pe,
                total_energy=ke + pe,
                temperature=temp_K,
                num_neighbors=system.neighbor_manager.total_neighbors(),
                min_neighbor_distance=min_r,
                max_force=max_force,
                time_per_step_ms=ms_per_step,
            )
            stats_history.append(stats)
            print_stats(stats)
            step_start = time.perf_counter()

    return stats_history


# ==============================================================================
# Printing Utilities
# ==============================================================================


def print_header() -> None:
    """Print simulation statistics header."""
    print("=" * 95)
    print(
        f"{'Step':>8} {'KE (eV)':>12} {'PE (eV)':>12} {'Total (eV)':>12} "
        f"{'T (K)':>10} {'Neighbors':>10} {'min r (Å)':>10} {'max|F|':>10}"
    )
    print("=" * 95)


def print_stats(stats: SimulationStats) -> None:
    """Print statistics for a simulation step."""
    print(
        f"{stats.step:>8d} {stats.kinetic_energy:>12.4f} {stats.potential_energy:>12.4f} "
        f"{stats.total_energy:>12.4f} {stats.temperature:>10.2f} "
        f"{stats.num_neighbors:>10d} {stats.min_neighbor_distance:>10.3f} {stats.max_force:>10.3e}"
    )
