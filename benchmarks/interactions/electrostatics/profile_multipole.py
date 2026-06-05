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

"""Single-API profiling harness for the multipole electrostatics torch APIs.

Standalone (no ``benchmarks.systems`` dependency): builds its own periodic
system. Profiles ONE public API at a time across {eager, compiled} x
{forward, forward+backward}, wrapping the timed region in an NVTX range named
``prof`` so an ``nsys`` capture can be filtered to the steady-state work.

APIs (``--api``):
    directk    -> multipole_electrostatic_energy   (full Coulomb in k-space)
    features   -> multipole_electrostatic_features  (reciprocal projection)
    realspace  -> multipole_real_space_energy       (erfc-damped pair sum, CSR)
    reciprocal -> multipole_reciprocal_space_energy  (Ewald reciprocal)
    ewald      -> multipole_ewald_summation          (real + reciprocal)
    pme        -> multipole_particle_mesh_ewald       (real + mesh)

Run directly for a wall-clock summary, or under nsys for a kernel trace::

    python benchmarks/interactions/electrostatics/profile_multipole.py \\
        --api ewald --n 2000 --mode eager --grad bwd

    nsys profile --stats=true --force-overwrite=true -o /tmp/claude/prof_ewald \\
        --capture-range=nvtx --nvtx-capture=prof \\
        python benchmarks/interactions/electrostatics/profile_multipole.py \\
            --api ewald --n 2000 --mode eager --grad bwd --iters 50
"""

from __future__ import annotations

import argparse
import contextlib
import time

import torch

from nvalchemiops.torch.interactions.electrostatics import (
    multipole_electrostatic_energy,
    multipole_electrostatic_features,
    multipole_ewald_summation,
    multipole_real_space_energy,
    multipole_reciprocal_space_energy,
    pack_multipole_moments,
)
from nvalchemiops.torch.interactions.electrostatics.pme_multipole import (
    multipole_particle_mesh_ewald,
)
from nvalchemiops.torch.neighbors import neighbor_list

# Fixed physical knobs — representative, not accuracy-tuned (this is a timing harness).
SIGMA = 1.0
ALPHA = 0.3
REAL_CUTOFF = 6.0
KSPACE_CUTOFF = 4.0
NUMBER_DENSITY = 0.1  # atoms / A^3 -> cubic box side L = (N / rho)^(1/3)
SEED = 20260605


def build_system(n: int, l_max: int, dtype: torch.dtype, device: torch.device) -> dict:
    """Build a random neutral periodic system with l_max moments + a CSR neighbor list."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    box = float((n / NUMBER_DENSITY) ** (1.0 / 3.0))
    positions = (torch.rand(n, 3, generator=g, dtype=torch.float64) * box).to(
        device, dtype
    )
    cell = (torch.eye(3, dtype=torch.float64) * box).to(device, dtype)
    pbc = torch.ones(3, dtype=torch.bool, device=device)

    charges = torch.randn(n, generator=g, dtype=torch.float64).to(device, dtype)
    charges = charges - charges.mean()  # neutralize
    dipoles = quads = None
    if l_max >= 1:
        dipoles = (0.3 * torch.randn(n, 3, generator=g, dtype=torch.float64)).to(
            device, dtype
        )
    if l_max >= 2:
        q = (0.2 * torch.randn(n, 3, 3, generator=g, dtype=torch.float64)).to(
            device, dtype
        )
        q = 0.5 * (q + q.transpose(-1, -2))
        tr = q.diagonal(dim1=-2, dim2=-1).sum(-1)
        q = q - (tr / 3.0)[:, None, None] * torch.eye(3, dtype=dtype, device=device)
        quads = q
    moments = pack_multipole_moments(charges, dipoles, quads)

    cutoff = min(REAL_CUTOFF, 0.49 * box)
    pairs, nptr, shifts = neighbor_list(
        positions, cutoff, cell=cell, pbc=pbc, return_neighbor_list=True
    )
    idx_j = (pairs[1] if pairs.dim() == 2 else pairs).to(torch.int32).contiguous()
    return {
        "positions": positions,
        "moments": moments,
        "cell": cell,
        "idx_j": idx_j,
        "neighbor_ptr": nptr.to(torch.int32).contiguous(),
        "unit_shifts": shifts.to(torch.int32).contiguous(),
        "n_pairs": int(idx_j.numel()),
        "box": box,
    }


def make_callable(api: str, s: dict):
    """Return a 0-arg closure that runs ``api`` on system ``s`` and yields a scalar."""
    pos, mom, cell = s["positions"], s["moments"], s["cell"]
    idx_j, nptr, shifts = s["idx_j"], s["neighbor_ptr"], s["unit_shifts"]

    if api == "directk":
        return lambda: multipole_electrostatic_energy(
            pos, mom, cell, sigma=SIGMA, kspace_cutoff=KSPACE_CUTOFF
        )
    if api == "features":
        return lambda: multipole_electrostatic_features(
            pos,
            mom,
            cell,
            sigma=SIGMA,
            receiver_sigmas=[0.7, 1.0, 1.5],
            kspace_cutoff=KSPACE_CUTOFF,
            feature_max_l=1,
        ).sum()
    if api == "realspace":
        # NB: the low-level real-space entry takes sigma/alpha as (1,) tensors,
        # unlike the composites (ewald/pme) which accept plain floats.
        sigma_t = torch.tensor([SIGMA], dtype=pos.dtype, device=pos.device)
        alpha_t = torch.tensor([ALPHA], dtype=pos.dtype, device=pos.device)
        return lambda: multipole_real_space_energy(
            pos, mom, cell, idx_j, nptr, shifts, sigma_t, alpha_t
        )
    if api == "reciprocal":
        return lambda: multipole_reciprocal_space_energy(
            pos, mom, cell, sigma=SIGMA, alpha=ALPHA, kspace_cutoff=KSPACE_CUTOFF
        )
    if api == "ewald":
        return lambda: multipole_ewald_summation(
            pos,
            mom,
            cell,
            idx_j,
            nptr,
            shifts,
            sigma=SIGMA,
            alpha=ALPHA,
            kspace_cutoff=KSPACE_CUTOFF,
        )
    if api == "pme":
        return lambda: multipole_particle_mesh_ewald(
            pos, mom, cell, idx_j, nptr, shifts, sigma=SIGMA, alpha=ALPHA
        )
    raise ValueError(f"unknown api {api!r}")


def main() -> int:
    """Parse CLI args, build the system, and time the chosen API (GPU + wall ms/iter)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--api",
        required=True,
        choices=["directk", "features", "realspace", "reciprocal", "ewald", "pme"],
    )
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--l-max", type=int, choices=[0, 1, 2], default=1)
    p.add_argument("--mode", choices=["eager", "compiled"], default="eager")
    p.add_argument("--grad", choices=["none", "bwd"], default="none")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this harness targets GPU profiling.")
        return 1
    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)

    s = build_system(args.n, args.l_max, dtype, device)
    s["positions"].requires_grad_(args.grad == "bwd")
    fn = make_callable(args.api, s)
    if args.mode == "compiled":
        fn = torch.compile(fn)

    def step():
        if args.grad == "bwd":
            if s["positions"].grad is not None:
                s["positions"].grad = None
            out = fn()
            # Some entries (real-space) return per-atom energies; reduce to a
            # scalar so backward has a well-defined seed.
            (out.sum() if out.ndim > 0 else out).backward()
            return out
        with torch.no_grad():
            return fn()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    rng = (
        torch.cuda.nvtx.range("prof")
        if hasattr(torch.cuda, "nvtx")
        else contextlib.nullcontext()
    )
    t0 = time.perf_counter()
    start.record()
    with rng:
        for _ in range(args.iters):
            step()
    end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3 / args.iters
    gpu = start.elapsed_time(end) / args.iters

    label = f"{args.api} | n={args.n} l_max={args.l_max} {args.mode} grad={args.grad} {args.dtype}"
    print(
        f"{label}\n  pairs={s['n_pairs']}  box={s['box']:.2f} A"
        f"  | GPU {gpu:.3f} ms/iter  wall {wall:.3f} ms/iter"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
