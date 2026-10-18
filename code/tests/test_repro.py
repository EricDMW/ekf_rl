"""Reproducibility — two runs of seed 0 give identical results (cheap)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from algorithms.common import run_method


def _digest(out: dict):
    return (
        round(out["deploy_return_mean"], 8),
        round(out["deploy_grad_norm_sq"], 8),
        round(out["deploy_exit_prob"], 8),
        tuple(round(v, 8) for v in out["learning_curve"][:5]),
    )


def main():
    out_a = run_method("Ours", "pendulum_partial",
                       seed=0, n_id=100, T=20, K=30, eval_rollouts=3)
    out_b = run_method("Ours", "pendulum_partial",
                       seed=0, n_id=100, T=20, K=30, eval_rollouts=3)
    a, b = _digest(out_a), _digest(out_b)
    print("a:", a)
    print("b:", b)
    assert a == b, f"non-deterministic: {a} vs {b}"
    print("OK")


if __name__ == "__main__":
    main()
