"""E2 — Identification-rate verification.

We directly measure the KRR sup-norm error of the residual
identification, which is the quantity that ``Theorem 6`` of the paper
bounds at the rate :math:`\\widetilde{O}(n^{-(2s-d)/(2s+d)})`. This is a
much cleaner test than the deployment-return gap, which is dominated by
optimization variance once the residual estimator is good enough.

For each sample size :math:`n`, we re-fit the KRR :math:`\\widehat\\Delta_f`
on a fresh Phase-1 dataset of size :math:`n` and evaluate its sup-norm
error against the true residual on a held-out grid. Multiple seeds give
us a 95% CI on the empirical sup-norm error.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import PendulumPartial  # noqa: E402
from envs.pendulum_partial import smooth_residual  # noqa: E402
from identification import fit_residuals  # noqa: E402
from utils.logging import write_json, ensure_dir  # noqa: E402
from utils.seed import git_sha  # noqa: E402


def _sup_norm_error(f_residual, n_grid: int, rng: np.random.Generator) -> float:
    """Sup-norm error of the learned f-residual against the true residual
    on a fresh random grid drawn from the same domain."""
    cfg = PendulumPartial(mode="real").cfg
    theta = rng.uniform(-cfg.theta_box * 0.8, cfg.theta_box * 0.8, n_grid)
    dtheta = rng.uniform(-cfg.dtheta_box * 0.5, cfg.dtheta_box * 0.5, n_grid)
    u = rng.uniform(-cfg.u_max, cfg.u_max, n_grid)
    err_max = 0.0
    for i in range(n_grid):
        x = np.array([theta[i], dtheta[i]])
        uu = np.array([u[i]])
        pred = f_residual(x, uu)              # shape (2,)
        # True residual: only on dtheta component, scaled by dt
        true_acc = smooth_residual(theta[i], dtheta[i], np.clip(u[i], -cfg.u_max, cfg.u_max))
        true_delta = np.array([0.0, cfg.dt * true_acc])
        err = float(np.max(np.abs(pred - true_delta)))
        if err > err_max:
            err_max = err
    return err_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[20, 40, 80, 160, 320, 640, 1280])
    ap.add_argument("--out-root",
                    default=str(Path(__file__).resolve().parents[1] / "results_raw" / "E2"))
    args = ap.parse_args()

    sha = git_sha()
    for n in args.ns:
        for seed in range(args.seeds):
            fn = Path(args.out_root) / f"n_{n:05d}" / f"seed_{seed:03d}"
            if (fn.with_suffix(".json")).exists():
                print(f"[skip] n={n} seed={seed}")
                continue
            ensure_dir(fn.parent)
            t0 = time.time()
            rng = np.random.default_rng(seed * 53 + n * 7)
            real = PendulumPartial(mode="real")
            nominal = PendulumPartial(mode="nominal")
            f_res, h_res, info = fit_residuals(real, nominal, n_samples=n, rng=rng)
            sup_err = _sup_norm_error(f_res, n_grid=500, rng=rng)
            dt = time.time() - t0
            payload = {
                "seed": int(seed),
                "n_id": int(n),
                "sup_err_f": float(sup_err),
                "krr_info": info,
                "wallclock_sec": float(dt),
                "git_sha": sha,
                # Provide a placeholder so the aggregator's CI bootstrap code
                # doesn't break (it expects deploy_return_mean and
                # deploy_grad_norm_sq fields).
                "deploy_return_mean": float(-sup_err),     # so smaller-is-better
                "deploy_grad_norm_sq": float(sup_err ** 2),
            }
            write_json(fn.with_suffix(".json"), payload)
            print(f"[done] n={n} seed={seed}: sup_err={sup_err:.4f}, t={dt:.1f}s")


if __name__ == "__main__":
    main()
