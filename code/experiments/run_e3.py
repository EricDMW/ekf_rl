"""E3 — optimization-rate verification.

Sweep T ∈ {500, 1000, 3000, 8000, 20000}. n, K fixed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.common import run_method  # noqa: E402
from utils.logging import write_json, write_npz, ensure_dir  # noqa: E402
from utils.seed import git_sha  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--Ts", type=int, nargs="+",
                    default=[50, 100, 200, 400, 800])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out-root",
                    default=str(Path(__file__).resolve().parents[1] / "results_raw" / "E3"))
    args = ap.parse_args()

    sha = git_sha()
    env = "pendulum_partial"
    for T in args.Ts:
        for seed in range(args.seeds):
            fn = Path(args.out_root) / f"T_{T:06d}" / f"seed_{seed:03d}"
            if (fn.with_suffix(".json")).exists():
                print(f"[skip] T={T} seed={seed}")
                continue
            ensure_dir(fn.parent)
            t0 = time.time()
            out = run_method("Ours", env, seed=seed, n_id=args.n,
                             T=T, K=args.K, eval_rollouts=15, lr=args.lr)
            dt = time.time() - t0
            out["wallclock_sec"] = float(dt)
            out["git_sha"] = sha
            lc = np.array(out["learning_curve"], dtype=float)
            gns = np.array(out["grad_norms_sq"], dtype=float)
            dr = np.array(out["deploy_returns"], dtype=float)
            write_npz(fn.with_suffix(".npz"),
                      learning_curve=lc, grad_norms_sq=gns,
                      deploy_returns=dr)
            payload = {k: v for k, v in out.items()
                       if k not in ("learning_curve", "grad_norms_sq", "deploy_returns")}
            write_json(fn.with_suffix(".json"), payload)
            print(f"[done] T={T} seed={seed}: grad_sq={out['deploy_grad_norm_sq']:.2e}, "
                  f"t={dt:.1f}s")


if __name__ == "__main__":
    main()
