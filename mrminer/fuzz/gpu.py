"""Pick the freest GPU from an allowed pool, for per-test subprocess placement.

Each test runs in its own isolated subprocess; before spawning it we route it to the
GPU (from an allowed pool) that currently has the most free memory. This balances load
and avoids OOM when several lanes share the machine. The choice is per-test and
ephemeral (it lives only in that subprocess's `CUDA_VISIBLE_DEVICES`), so it does not
change any finding (eager-vs-compiled is GPU-agnostic) or affect the next run.

Pool: MRMINER_GPU_POOL (comma-separated absolute indices). Empty by default, so
the worker inherits the caller's device visibility unless a pool is configured.
"""
from __future__ import annotations

import os
import subprocess
import time

_DEFAULT_POOL = ""
_cache: dict = {"t": 0.0, "gpu": None}
_CACHE_TTL = 1.5   # seconds; avoid hammering nvidia-smi when tests spawn back-to-back


def _pool() -> list[str]:
    raw = os.environ.get("MRMINER_GPU_POOL", _DEFAULT_POOL)
    return [x.strip() for x in raw.split(",") if x.strip() != ""]


def pick_free_gpu() -> str | None:
    """Absolute index (str) of the freest GPU in the pool, or None if unavailable."""
    pool = _pool()
    if not pool:
        return None
    if len(pool) == 1:
        return pool[0]
    now = time.time()
    if _cache["gpu"] is not None and now - _cache["t"] < _CACHE_TTL:
        return _cache["gpu"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits", "-i", ",".join(pool)],
            capture_output=True, text=True, timeout=5,
        ).stdout
        best, best_free = None, -1
        for line in out.strip().splitlines():
            idx, free = (p.strip() for p in line.split(","))
            if int(free) > best_free:
                best, best_free = idx, int(free)
        if best is not None:
            _cache.update(t=now, gpu=best)
            return best
    except Exception:
        pass
    return pool[0]


def place_env(env: dict) -> dict:
    """Set CUDA_VISIBLE_DEVICES on a subprocess env to the freest pooled GPU (in place)."""
    g = pick_free_gpu()
    if g is not None:
        env["CUDA_VISIBLE_DEVICES"] = g
    return env
