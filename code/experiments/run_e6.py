"""E6 — Safety-margin tail.

Vary the inner safety margin r (i.e., margin used inside in_safe(x, margin)).
Plot exit probability vs. r on semi-log; overlay Gaussian-tail reference.

Implementation: rather than retrain a policy per r, we take a trained policy
from E1 (Ours, seed 0) and evaluate it under varying r at deployment.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.common import run_method  # noqa: E402
from utils.logging import write_json, ensure_dir  # noqa: E402
from utils.seed import git_sha  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--rs", type=float, nargs="+",
                    default=[0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--out-root",
                    default=str(Path(__file__).resolve().parents[1] / "results_raw" / "E6"))
    args = ap.parse_args()

    sha = git_sha()
    env = "pendulum_partial"
    for r in args.rs:
        for seed in range(args.seeds):
            fn = Path(args.out_root) / f"r_{r:.2f}" / f"seed_{seed:03d}"
            if (fn.with_suffix(".json")).exists():
                print(f"[skip] r={r} seed={seed}")
                continue
            ensure_dir(fn.parent)
            t0 = time.time()
            out = run_method("Ours", env, seed=seed, n_id=1000,
                             T=200, K=100, eval_rollouts=200, lr=3e-3,
                             r_margin_override=r)
            dt = time.time() - t0
            out["wallclock_sec"] = float(dt)
            out["git_sha"] = sha
            out["r_margin"] = float(r)
            payload = {k: v for k, v in out.items()
                       if k not in ("learning_curve", "grad_norms_sq", "deploy_returns")}
            write_json(fn.with_suffix(".json"), payload)
            print(f"[done] r={r} seed={seed}: exit_prob={out['deploy_exit_prob']:.3f}, "
                  f"t={dt:.1f}s")


if __name__ == "__main__":
    main()
