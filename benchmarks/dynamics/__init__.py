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
Dynamics Benchmark Suite
========================

Benchmarks for comparing nvalchemiops dynamics integrators against ASE baselines.
"""

from .ase_calculator import ASELJCalculator, WarpLJCalculator, get_calculator
from .benchmark_dynamics import (
    ASEBenchmark,
    BenchmarkResult,
    NvalchemiOpsBenchmark,
    create_lj_system,
    run_benchmarks,
)
from .lj_calculator import lj_energy_forces, lj_energy_forces_virial

__all__ = [
    # LJ Calculator (pure Warp)
    "lj_energy_forces",
    "lj_energy_forces_virial",
    # ASE Calculators
    "WarpLJCalculator",
    "ASELJCalculator",
    "get_calculator",
    # Benchmarks
    "BenchmarkResult",
    "NvalchemiOpsBenchmark",
    "ASEBenchmark",
    "create_lj_system",
    "run_benchmarks",
]
