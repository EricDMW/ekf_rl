"""E4 — Rollout-truncation rate verification.

We directly measure the K-truncation bias of the PG return estimator,
\(\\eps_K \\approx |J^{H}(\\pi)-J^{K}(\\pi)|\\), which is the quantity that
``Assumption 8`` of the paper bounds at the rate
\(\\gamma^K R_{\\max}/(1-\\gamma)^{2}\). This is much cleaner than the
deployment-return gap (which is variance-floor-limited).

For each rollout length :math:`K`, we evaluate a fixed reference policy
on the real environment by:
* simulating a long :math:`H`-step real-env rollout (with :math:`H\\gg K`);
* computing the truncated return \\(J^{K}(\\pi)\\);
* computing the long-horizon return \\(J^{H}(\\pi)\\);
* recording the absolute bias \\(|J^{H}-J^{K}|\\), averaged over many
  rollouts to suppress trajectory noise.

The reference policy is obtained from a short PG training run; the same
policy is reused for every :math:`K`-evaluation so that the only thing
that changes between conditions is the truncation length.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import PendulumPartial  # noqa: E402
from filters import EKF  # noqa: E402
from identification import fit_residuals  # noqa: E402
from training import SquashedGaussianPolicy, ValueBaseline, train_pg  # noqa: E402
from training.pg import PGConfig  # noqa: E402
from algorithms.common import (  # noqa: E402
    build_policy_and_baseline, make_default_P0,
)
from utils.logging import write_json, ensure_dir  # noqa: E402
from utils.seed import git_sha  # noqa: E402


def _trajectory(env, surrogate, policy, rng, H, P0):
    """Run a single H-step real-env trajectory; return per-step rewards."""
    x = env.sample_initial(rng)
    ekf = EKF(surrogate, env.Q, env.R_v, x0_mean=x.copy(), P0=P0)
    rewards = []
    for k in range(H):
        u, _ = policy.act(ekf.x_hat, rng, deterministic=False)
        x_next, y, r = env.sample_step(x, u, rng)
        ekf.step(u, y)
        rewards.append(float(r))
        x = x_next
    return np.asarray(rewards)


def _discounted(rewards: np.ndarray, gamma: float, K: int) -> float:
    K = min(K, rewards.size)
    g = gamma ** np.arange(K)
    return float(np.sum(g * rewards[:K]))


def _train_reference_policy(rng_master: np.random.Generator, T: int = 150):
    """Train a reference Ours-style policy once; reuse for every K."""
    real = PendulumPartial(mode="real")
    nominal = PendulumPartial(mode="nominal")
    f_res, h_res, _ = fit_residuals(real, nominal, n_samples=1000, rng=rng_master)
    augmented = PendulumPartial(mode="augmented", residual_fn=f_res, h_residual_fn=h_res)
    surrogate = PendulumPartial(mode="augmented", residual_fn=f_res, h_residual_fn=h_res)
    policy, baseline = build_policy_and_baseline(real, rng_master)
    P0 = make_default_P0(real)
    cfg = PGConfig(
        lr=3e-3, K=100, gamma=real.cfg.gamma, gae_lambda=0.95,
        A_max=2 * real.cfg.R_max / (1 - real.cfg.gamma),
        n_iters=T, batch_rollouts=8, baseline_steps_per_iter=5,
        inner_margin=0.05, baseline_lr=1e-2,
    )
    train_pg(augmented, surrogate, policy, baseline, rng_master, cfg, P0,
             real_env=real, eval_every=max(1, T // 20), eval_rollouts=8)
    return real, surrogate, policy, P0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--Ks", type=int, nargs="+",
                    default=[5, 10, 20, 40, 80, 160])
    ap.add_argument("--H-long", type=int, default=400)
    ap.add_argument("--n-rollouts", type=int, default=30)
    ap.add_argument("--out-root",
                    default=str(Path(__file__).resolve().parents[1] / "results_raw" / "E4"))
    args = ap.parse_args()

    sha = git_sha()
    gamma = 0.99

    for K in args.Ks:
        for seed in range(args.seeds):
            fn = Path(args.out_root) / f"K_{K:04d}" / f"seed_{seed:03d}"
            if (fn.with_suffix(".json")).exists():
                print(f"[skip] K={K} seed={seed}")
                continue
            ensure_dir(fn.parent)
            t0 = time.time()
            rng = np.random.default_rng(seed * 31 + 17)
            real, surrogate, policy, P0 = _train_reference_policy(rng)
            biases = []
            for _ in range(args.n_rollouts):
                rewards = _trajectory(real, surrogate, policy, rng,
                                      H=args.H_long, P0=P0)
                J_K = _discounted(rewards, gamma, K)
                J_H = _discounted(rewards, gamma, args.H_long)
                biases.append(abs(J_H - J_K))
            bias_mean = float(np.mean(biases))
            bias_std = float(np.std(biases))
            dt = time.time() - t0
            payload = {
                "seed": int(seed), "K": int(K),
                "trunc_bias_mean": bias_mean,
                "trunc_bias_std": bias_std,
                "n_rollouts": int(args.n_rollouts),
                "H_long": int(args.H_long),
                "wallclock_sec": float(dt),
                "git_sha": sha,
                # Aggregator compat (treat smaller-is-better gap).
                "deploy_return_mean": -bias_mean,
                "deploy_grad_norm_sq": bias_mean ** 2,
            }
            write_json(fn.with_suffix(".json"), payload)
            print(f"[done] K={K} seed={seed}: bias={bias_mean:.4f}, t={dt:.1f}s")


if __name__ == "__main__":
    main()
