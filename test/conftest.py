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
import os
import tempfile

import pytest

# Point Warp's kernel cache at $TMPDIR via env-var BEFORE any module
# imports Warp; the default ~/.cache/warp is read-only in some sandboxes.
_warp_cache_default = os.path.join(
    os.environ.get("TMPDIR", tempfile.gettempdir()), "warp_test_cache"
)
os.environ.setdefault("WARP_CACHE_PATH", _warp_cache_default)
os.makedirs(os.environ["WARP_CACHE_PATH"], exist_ok=True)


@pytest.fixture(scope="module", autouse=True)
def set_env_vars():
    """Set JAX specific environment variables"""
    if "XLA_PYTHON_CLIENT_PREALLOCATE" in os.environ:
        old_preallocate = os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]
    else:
        old_preallocate = ""
    if "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ:
        old_allocator = os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]
    else:
        old_allocator = ""
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    yield
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = old_allocator
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = old_preallocate


@pytest.fixture(autouse=True)
def _release_gpu_memory():
    """Drop JAX program cache and CUDA caching allocator pools after each test.

    On unified-memory hardware (e.g. GB10) the platform allocator's per-test
    allocations otherwise accumulate across a long pytest session until the
    OOM killer fires. Forcing GC + cache release between tests trades a few
    ms of per-test overhead (and occasional JIT recompilation if the same
    shape recurs later in the suite) for stable memory across the run.
    """
    yield
    import gc

    gc.collect()
    try:
        import jax

        jax.clear_caches()
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
