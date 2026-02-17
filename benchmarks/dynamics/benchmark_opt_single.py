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
Single-System Geometry Optimization Benchmarks
==============================================

Benchmark optimization methods using Lennard-Jones systems.

Usage
-----
    python benchmark_opt_single.py --config benchmark_config.yaml

Backends
--------
- nvalchemiops: GPU-accelerated FIRE optimizer

Output
------
CSV file with single-system schema (11 columns):
- dynamics_opt_single_nvalchemiops_<gpu_sku>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .shared_utils import (
    NvalchemiOpsBenchmark,
    NvalchemiopsLJModel,
    create_lj_system,
    get_gpu_sku,
    load_config,
    print_benchmark_footer,
    print_benchmark_header,
    print_benchmark_result,
    write_results_csv,
)


def run_benchmarks(config: dict, output_dir: Path) -> None:
    """Run single-system optimization benchmarks.

    Parameters
    ----------
    config : dict
        Benchmark configuration.
    output_dir : Path
        Output directory for CSV files.
    """
    opt_config = config.get("opt_single", {})
    if not opt_config.get("enabled", True):
        print("Optimization single-system benchmarks disabled in config")
        return

    system_sizes = opt_config.get("system_sizes", [256, 512, 1024, 2048])
    optimizers = opt_config.get("optimizers", {})

    # Get potential parameters
    potential_config = config.get("potential", {})
    epsilon = potential_config.get("epsilon", 0.0104)
    sigma = potential_config.get("sigma", 3.40)
    cutoff = potential_config.get("cutoff", 8.5)
    skin = potential_config.get("skin", 1.0)
    neighbor_rebuild_interval = potential_config.get("neighbor_rebuild_interval", 10)

    gpu_sku = get_gpu_sku()
    results_nvalchemiops = []

    print("\nRunning Single-System Optimization Benchmarks (nvalchemiops)")
    print(f"GPU: {gpu_sku}")
    print_benchmark_header("Optimization")

    for num_atoms in system_sizes:
        # Create system (perturbed from equilibrium for optimization)
        positions, cell, masses, velocities = create_lj_system(
            num_atoms=num_atoms,
            lattice_constant=5.26,
            temperature=300.0,
            device="cuda",
            dtype=torch.float64,
        )

        # Perturb positions slightly to create a non-equilibrium structure
        positions += torch.randn_like(positions) * 0.1

        pbc = torch.tensor([True, True, True], device=positions.device)

        # Create LJ model
        lj_model = NvalchemiopsLJModel(
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
            cell=cell,
            batch_idx=None,  # Single-system mode
            device="cuda",
            dtype=torch.float64,
        )

        # Run nvalchemiops benchmarks
        nv_bench = NvalchemiOpsBenchmark(
            positions=positions,
            cell=cell,
            masses=masses,
            pbc=pbc,
            model=lj_model,
            skin=skin,
            neighbor_rebuild_interval=neighbor_rebuild_interval,
            velocities=velocities,  # Note: FIRE doesn't need initial velocities but we pass them anyway
        )

        # FIRE
        if "fire" in optimizers:
            fire_config = optimizers["fire"]
            result = nv_bench.run_fire(
                max_steps=fire_config.get("max_steps", 1000),
                force_tolerance=fire_config.get("force_tolerance", 0.01),
            )
            results_nvalchemiops.append(result)
            print_benchmark_result(result, is_md=False)

    print_benchmark_footer()

    # Write CSV results
    if results_nvalchemiops:
        output_path = output_dir / f"dynamics_opt_single_nvalchemiops_{gpu_sku}.csv"
        write_results_csv(results_nvalchemiops, output_path)
        print(f"\nWrote nvalchemiops results to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Single-system optimization benchmarks for nvalchemiops"
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
