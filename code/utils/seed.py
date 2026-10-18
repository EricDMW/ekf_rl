"""Deterministic seeding across numpy and Python ``random``.

The simulation does not depend on torch or gymnasium, so we keep this small.
"""
from __future__ import annotations

import os
import random
import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed every RNG we use and return a fresh numpy Generator.

    A single ``Generator`` is returned so callers can use a *local* RNG
    when they care about determinism in the face of library calls that
    might consume the global state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def git_sha() -> str:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "no-git"
