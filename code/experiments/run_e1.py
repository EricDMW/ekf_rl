"""E1 — main comparison: 6 methods x 2 envs x N seeds.

Writes per-seed results to ``code/results_raw/E1/<env>/<method>/seed_<i>.json``
plus learning curves to ``.npz``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.common import run_method  # noqa: E402
from utils.logging import write_json, write_npz, ensure_dir  # noqa: E402
from utils.seed import git_sha  # noqa: E402


# These are defaults; the CLI overrides them.
DEFAULTS = {
    "pendulum_partial": dict(n_id=1000, T=120, K=100, eval_rollouts=20, lr=3e-3),
    "planar_quadrotor": dict(n_id=1500, T=120, K=60, eval_rollouts=10, lr=2e-3),
}


def run_one(method, env, seed, cfg):
    return run_method(
        method, env, seed=seed,
        n_id=cfg["n_id"], T=cfg["T"], K=cfg["K"],
        eval_rollouts=cfg["eval_rollouts"],
        lr=cfg["lr"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--methods", nargs="+",
                    default=["Ours", "B1", "B2", "B3"])
    ap.add_argument("--envs", nargs="+",
                    default=["pendulum_partial"])
    ap.add_argument("--out-root",
                    default=str(Path(__file__).resolve().parents[1] / "results_raw" / "E1"))
    ap.add_argument("--T-override", type=int, default=None)
    args = ap.parse_args()

    sha = git_sha()
    for env in args.envs:
        cfg = DEFAULTS[env].copy()
        if args.T_override:
            cfg["T"] = args.T_override
        for method in args.methods:
            for seed in range(args.seeds):
                out_dir = Path(args.out_root) / env / method
                ensure_dir(out_dir)
                fn = out_dir / f"seed_{seed:03d}"
                if (fn.with_suffix(".json")).exists():
                    print(f"[skip] {env}/{method}/seed{seed} — already exists")
                    continue
                t0 = time.time()
                out = run_one(method, env, seed, cfg)
                dt = time.time() - t0
                out["wallclock_sec"] = float(dt)
                out["git_sha"] = sha
                payload = {k: v for k, v in out.items()
                           if not isinstance(v, np.ndarray)}
                lc = np.array(out["learning_curve"], dtype=float)
                gns = np.array(out["grad_norms_sq"], dtype=float)
                deploy_returns = np.array(out["deploy_returns"], dtype=float)
                dc = np.array(out.get("deploy_curve", []), dtype=float)
                di = np.array(out.get("deploy_iter", []), dtype=float)
                write_npz(fn.with_suffix(".npz"),
                          learning_curve=lc,
                          grad_norms_sq=gns,
                          deploy_returns=deploy_returns,
                          deploy_curve=dc,
                          deploy_iter=di)
                for k in ("learning_curve", "grad_norms_sq", "deploy_returns",
                          "deploy_curve", "deploy_iter"):
                    payload.pop(k, None)
                write_json(fn.with_suffix(".json"), payload)
                print(f"[done] {env}/{method}/seed{seed}: "
                      f"deploy={out['deploy_return_mean']:.1f}, "
                      f"exit={out['deploy_exit_prob']:.2f}, "
                      f"t={dt:.1f}s")


if __name__ == "__main__":
    main()
