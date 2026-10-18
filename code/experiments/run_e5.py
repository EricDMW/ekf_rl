"""E5 — Filter-vs-true-state execution gap.

Take a final policy from E1 (Ours, seed 0) and re-evaluate it on both envs
with varying R_v inflation. Plot the gap |J_filter − J_true| vs. R_v factor
(a proxy for the EKF steady-state floor B_e).
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
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--R-factors", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--envs", nargs="+",
                    default=["pendulum_partial"])
    ap.add_argument("--out-root",
                    default=str(Path(__file__).resolve().parents[1] / "results_raw" / "E5"))
    args = ap.parse_args()

    sha = git_sha()
    for env in args.envs:
        T = 200
        K = 100
        lr = 3e-3
        eval_rollouts = 30
        for factor in args.R_factors:
            for seed in range(args.seeds):
                fn = Path(args.out_root) / env / f"R_{factor:.2f}" / f"seed_{seed:03d}"
                if (fn.with_suffix(".json")).exists():
                    print(f"[skip] {env} R={factor} seed={seed}")
                    continue
                ensure_dir(fn.parent)
                t0 = time.time()
                out = run_method("Ours", env, seed=seed, n_id=1000,
                                 T=T, K=K, eval_rollouts=eval_rollouts, lr=lr,
                                 R_v_inflate=factor)
                dt = time.time() - t0
                out["wallclock_sec"] = float(dt)
                out["git_sha"] = sha
                out["R_v_inflate"] = float(factor)
                payload = {k: v for k, v in out.items()
                           if k not in ("learning_curve", "grad_norms_sq", "deploy_returns")}
                write_json(fn.with_suffix(".json"), payload)
                print(f"[done] {env} R={factor} seed={seed}: "
                      f"exec_gap={out['exec_gap']:.3f}, t={dt:.1f}s")


if __name__ == "__main__":
    main()
