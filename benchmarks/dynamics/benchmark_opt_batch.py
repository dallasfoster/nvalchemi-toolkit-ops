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
Batched Geometry Optimization Benchmarks (nvalchemiops only)
============================================================

Benchmark batched optimization using nvalchemiops GPU-accelerated FIRE optimizer.

Usage
-----
    python benchmark_opt_batch.py --config benchmark_config.yaml

Output
------
CSV file with batched schema (14 columns):
- dynamics_opt_batch_nvalchemiops_<gpu_sku>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmarks.dynamics.shared_utils import (
    NvalchemiOpsBenchmark,
    create_fcc_argon,
    get_gpu_sku,
    load_config,
    print_batch_benchmark_footer,
    print_batch_benchmark_header,
    print_batch_benchmark_result,
    write_results_csv,
)


def create_batched_system(
    num_atoms_per_system: int,
    batch_size: int,
    lattice_constant: float = 5.26,
    temperature: float = 300.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create batched FCC argon systems for optimization.

    Systems are perturbed from equilibrium to create non-trivial optimization tasks.

    Returns
    -------
    positions : torch.Tensor
        Batched positions, shape (total_atoms, 3).
    cell : torch.Tensor
        Cell matrix, shape (batch_size, 3, 3).
    masses : torch.Tensor
        Batched masses, shape (total_atoms,).
    batch_idx : torch.Tensor
        Batch index for each atom, shape (total_atoms,).
    atom_ptr : torch.Tensor
        Pointer array, shape (batch_size + 1,).
    """
    # Create template system
    num_cells = int((num_atoms_per_system // 4 + 1) ** (1 / 3))
    positions, cell = create_fcc_argon(
        num_unit_cells=num_cells,
        a=5.26,
    )

    positions = torch.as_tensor(positions, dtype=dtype, device=device)
    cell = torch.as_tensor(cell, dtype=dtype, device=device)
    actual_num_atoms = positions.shape[0]

    # Replicate for batch with perturbations
    positions_list = []
    cell_list = []

    for i in range(batch_size):
        # Perturb positions for optimization task
        pos_perturbed = (
            positions + torch.randn_like(positions) * 0.1
        )  # Larger perturbation

        positions_list.append(pos_perturbed)
        cell_list.append(cell)

    # Stack into batched tensors
    positions = torch.cat(positions_list, dim=0)
    cell = torch.cat(cell_list, dim=0).reshape(batch_size, 3, 3)

    # Create batch_idx
    batch_idx = torch.repeat_interleave(
        torch.arange(batch_size, device=device), actual_num_atoms
    ).to(dtype=torch.int32)

    # Create atom_ptr
    atom_ptr = torch.arange(
        0,
        (batch_size + 1) * actual_num_atoms,
        actual_num_atoms,
        device=device,
        dtype=torch.int32,
    )

    return positions, cell, batch_idx, atom_ptr


def run_benchmarks(config: dict, output_dir: Path) -> None:
    """Run batched optimization benchmarks.

    Parameters
    ----------
    config : dict
        Benchmark configuration.
    output_dir : Path
        Output directory for CSV files.
    """
    batch_config = config.get("opt_batch", {})
    if not batch_config.get("enabled", False):
        print("Batched optimization benchmarks disabled in config")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = batch_config.get("dtype", torch.float32)
    system_sizes = batch_config.get("system_sizes", [256, 512])
    batch_sizes = batch_config.get("batch_sizes", [1, 2, 4, 8, 16])
    optimizers = batch_config.get("optimizers", {})

    # Extract potential config parameters
    potential_config = config.get("potential", {})
    epsilon = potential_config.get("epsilon", 0.0104)
    sigma = potential_config.get("sigma", 3.40)
    cutoff = potential_config.get("cutoff", 8.5)
    skin = potential_config.get("skin", 1.0)
    neighbor_rebuild_interval = potential_config.get("neighbor_rebuild_interval", 10)

    gpu_sku = get_gpu_sku()
    results = []

    # Print header with title and GPU info
    print("\nRunning Batched Optimization Benchmarks (nvalchemiops)")
    print(f"GPU: {gpu_sku}")
    print_batch_benchmark_header()

    for num_atoms in system_sizes:
        for batch_size in batch_sizes:
            # Create batched system
            batch_positions, batch_cells, batch_idx, atom_ptr = create_batched_system(
                num_atoms_per_system=num_atoms,
                batch_size=batch_size,
                lattice_constant=5.26,
                temperature=300.0,
                device=device,
                dtype=dtype,
            )

            pbc = torch.tensor([True, True, True], device=batch_positions.device)

            # Run nvalchemiops benchmarks
            nv_bench = NvalchemiOpsBenchmark(
                positions=batch_positions,
                cell=batch_cells,
                pbc=pbc,
                epsilon=epsilon,
                sigma=sigma,
                cutoff=cutoff,
                skin=skin,
                neighbor_rebuild_interval=neighbor_rebuild_interval,
                batch_idx=batch_idx,
            )

            # FIRE
            if optimizers.get("fire", {}).get("enabled", False):
                fire_config = optimizers["fire"]
                result = nv_bench.run_fire(
                    max_steps=fire_config.get("max_steps", 1000),
                    force_tolerance=fire_config.get("force_tolerance", 0.01),
                )
                results.append(result)
                print_batch_benchmark_result(result, is_md=False)

            nv_bench = NvalchemiOpsBenchmark(
                positions=batch_positions.clone(),
                cell=batch_cells.clone(),
                pbc=pbc,
                epsilon=epsilon,
                sigma=sigma,
                cutoff=cutoff,
                neighbor_rebuild_interval=neighbor_rebuild_interval,
                batch_idx=batch_idx,
            )

            if optimizers.get("fire2", {}).get("enabled", False):
                fire2_config = optimizers["fire2"]
                result = nv_bench.run_fire2(
                    max_steps=fire2_config.get("max_steps", 1000),
                    force_tolerance=fire2_config.get("force_tolerance", 0.01),
                )
                results.append(result)
                print_batch_benchmark_result(result, is_md=False)
    print_batch_benchmark_footer()

    # Write CSV results
    if results:
        output_path = output_dir / f"dynamics_opt_batch_nvalchemiops_{gpu_sku}.csv"
        write_results_csv(results, output_path)
        print(f"\nWrote results to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batched optimization benchmarks for nvalchemiops"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for CSV files",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_benchmarks(config, output_dir)


if __name__ == "__main__":
    main()
