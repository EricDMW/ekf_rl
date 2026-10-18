"""Batched policy-gradient training (Adam + averaged rollouts) — re-written
for low-variance, monotonic convergence on small problems.

Implements:

* ``train_pg(...)``: PG training in a (possibly augmented) simulator using EKF
  for state estimation. Returns a learning curve and the trained policy.
* ``deployment_rollouts(...)``: evaluate a trained policy on the real env with
  EKF for state estimation. Returns mean return and per-rollout returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

from filters.ekf import EKF
from training.policy import SquashedGaussianPolicy, ValueBaseline


# --------------------------------------------------- rollout collector --
def _rollout_one(env, surrogate, policy: SquashedGaussianPolicy,
                 rng: np.random.Generator,
                 K: int, gamma: float,
                 P0: np.ndarray,
                 stop_on_exit: bool = True,
                 inner_margin: float = 0.0,
                 use_true_state: bool = False,
                 dr_env_sampler: Optional[Callable[[np.random.Generator], object]] = None,
                 sample_initial_kwargs: Optional[dict] = None,
                 ):
    if dr_env_sampler is not None:
        env = dr_env_sampler(rng)
    kw = sample_initial_kwargs or {}
    try:
        x = env.sample_initial(rng, **kw)
    except TypeError:
        x = env.sample_initial(rng)
    ekf = EKF(surrogate, env.Q, env.R_v, x0_mean=x.copy(), P0=P0)
    traj = {"x_hat": [], "u": [], "r": [], "info": [], "done": False, "exit_k": None}
    cum_disc = 0.0
    g_disc = 1.0
    for k in range(K):
        s = x if use_true_state else ekf.x_hat
        u, info = policy.act(s, rng)
        x_next, y, r = env.sample_step(x, u, rng)
        ekf.step(u, y)
        traj["x_hat"].append(s.copy())
        traj["u"].append(u.copy())
        traj["r"].append(float(r))
        traj["info"].append(info)
        cum_disc += g_disc * float(r)
        g_disc *= gamma
        x = x_next
        if stop_on_exit and not env.in_safe(x, margin=inner_margin):
            traj["done"] = True
            traj["exit_k"] = k
            break
    traj["return"] = cum_disc
    return traj


def _gae_advantages(traj: dict, baseline: ValueBaseline,
                    gamma: float, lam: float, A_max: float) -> tuple:
    rewards = np.array(traj["r"], dtype=float)
    states = np.array(traj["x_hat"], dtype=float)
    T = len(rewards)
    if T == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    v = baseline.predict(states)
    advantages = np.zeros(T)
    returns = np.zeros(T)
    gae = 0.0
    next_ret = 0.0
    for t in reversed(range(T)):
        v_next = v[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * v_next - v[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_ret = rewards[t] + gamma * next_ret
        returns[t] = next_ret
    return np.clip(advantages, -A_max, A_max), returns, v


# ------------------------------------------------------------- config --
@dataclass
class PGConfig:
    lr: float = 3e-3
    K: int = 200
    gamma: float = 0.99
    gae_lambda: float = 0.95
    A_max: float = 100.0
    entropy_coef: float = 0.0
    n_iters: int = 1000
    batch_rollouts: int = 8           # NEW: average over this many rollouts
    inner_margin: float = 0.05
    baseline_lr: float = 5e-3
    baseline_steps_per_iter: int = 5  # NEW: more critic steps per PG step


# ----------------------------------------------------------- train loop --
def train_pg(train_env, surrogate,
             policy: SquashedGaussianPolicy,
             baseline: ValueBaseline,
             rng: np.random.Generator,
             cfg: PGConfig,
             P0: np.ndarray,
             dr_env_sampler: Optional[Callable] = None,
             use_true_state: bool = False,
             sample_initial_kwargs: Optional[dict] = None,
             real_env=None,                                # real env for deployment eval
             real_use_true_state: bool = False,
             eval_every: int = 0,
             eval_rollouts: int = 5,
             ) -> dict:
    """Batched PG with Adam.

    Each iteration collects ``cfg.batch_rollouts`` rollouts, computes
    advantage-weighted score gradients, averages them, and takes one Adam step.
    Baseline is fit with ``baseline_steps_per_iter`` SGD steps per iteration.
    """
    policy.init_adam()
    returns_history = []
    deploy_curve = []   # Real-env eval, sampled every `eval_every` iters.
    grad_norms_sq = []
    smooth_return = None
    best_deploy = -np.inf
    best_params = None
    for t in range(cfg.n_iters):
        all_infos = []
        all_weights = []
        all_states_for_baseline = []
        all_targets_for_baseline = []
        batch_returns = []
        for _ in range(cfg.batch_rollouts):
            traj = _rollout_one(
                train_env, surrogate, policy, rng,
                K=cfg.K, gamma=cfg.gamma, P0=P0,
                inner_margin=cfg.inner_margin,
                use_true_state=use_true_state,
                dr_env_sampler=dr_env_sampler,
                sample_initial_kwargs=sample_initial_kwargs,
            )
            if len(traj["info"]) == 0:
                continue
            adv, returns_, v_pred = _gae_advantages(
                traj, baseline, cfg.gamma, cfg.gae_lambda, cfg.A_max)
            gamma_powers = cfg.gamma ** np.arange(len(adv))
            all_infos.extend(traj["info"])
            all_weights.extend(list(gamma_powers * adv))
            all_states_for_baseline.append(np.array(traj["x_hat"]))
            all_targets_for_baseline.append(adv + v_pred)
            batch_returns.append(traj["return"])

        if not all_infos:
            returns_history.append(float("nan"))
            grad_norms_sq.append(0.0)
            continue

        # Standardize weights for numerical stability.
        w = np.array(all_weights)
        if w.std() > 1e-6:
            w = (w - w.mean()) / (w.std() + 1e-6)
        # Score-gradient (sum over batch, normalize by batch).
        grads = policy.score_grad(all_infos, weights=w)
        scale = 1.0 / max(1, len(all_infos))
        grads = [g * scale for g in grads]
        # Gradient clip.
        gnorm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        clip = 1.0
        if gnorm > clip:
            grads = [g * (clip / gnorm) for g in grads]
        policy.adam_step(grads, lr=cfg.lr)
        grad_norms_sq.append(gnorm ** 2)

        # Fit baseline.
        all_s = np.concatenate(all_states_for_baseline, axis=0)
        all_t = np.concatenate(all_targets_for_baseline, axis=0)
        for _ in range(cfg.baseline_steps_per_iter):
            baseline.fit_step(all_s, all_t, lr=cfg.baseline_lr)

        # Record exp-smoothed return for the learning curve.
        ep_ret = float(np.mean(batch_returns))
        smooth_return = ep_ret if smooth_return is None else 0.9 * smooth_return + 0.1 * ep_ret
        returns_history.append(smooth_return)

        # Optional periodic deployment evaluation on real env.
        if real_env is not None and eval_every > 0 and (t % eval_every == 0 or t == cfg.n_iters - 1):
            ev_rets = []
            for _ in range(eval_rollouts):
                ev_traj = _rollout_one(
                    real_env, surrogate, policy, rng,
                    K=cfg.K, gamma=cfg.gamma, P0=P0,
                    inner_margin=0.0,
                    use_true_state=real_use_true_state,
                    sample_initial_kwargs=sample_initial_kwargs,
                )
                # Add exit-penalty consistent with deployment metric.
                ret = ev_traj["return"]
                exit_k = ev_traj["exit_k"]
                if exit_k is not None:
                    g = cfg.gamma ** (exit_k + 1)
                    rem = max(0, cfg.K - exit_k - 1)
                    R_max = getattr(real_env.cfg, "R_max", 5.0)
                    ret += -R_max * g * (1 - cfg.gamma ** rem) / max(1e-12, 1 - cfg.gamma)
                ev_rets.append(ret)
            mean_eval = float(np.mean(ev_rets))
            deploy_curve.append((t, mean_eval))
            if mean_eval > best_deploy:
                best_deploy = mean_eval
                best_params = [p.copy() for p in policy.params]

    deploy_curve_arr = (np.array([x[1] for x in deploy_curve]) if deploy_curve
                        else np.array([]))
    deploy_iter_arr = (np.array([x[0] for x in deploy_curve]) if deploy_curve
                       else np.array([]))

    # Restore best policy seen during training (best-of-validation model selection).
    if best_params is not None:
        policy.params = best_params

    return {
        "returns": np.asarray(returns_history),
        "deploy_curve": deploy_curve_arr,
        "deploy_iter": deploy_iter_arr,
        "grad_norms_sq": np.asarray(grad_norms_sq),
        "final_policy_params": [p.copy() for p in policy.params],
    }


# --------------------------------------------------- deployment rollouts --
def deployment_rollouts(real_env, surrogate, policy, n_rollouts: int,
                        rng: np.random.Generator,
                        H: int, gamma: float,
                        P0: np.ndarray,
                        use_true_state: bool = False,
                        stop_on_exit: bool = True,
                        sample_initial_kwargs: Optional[dict] = None,
                        safety_margin: float = 0.0,
                        ) -> dict:
    returns, exits = [], []
    err_filter = []
    grad_norm_sq_samples = []
    R_max = getattr(real_env.cfg, "R_max", 10.0)
    kw = sample_initial_kwargs or {}
    for _ in range(n_rollouts):
        try:
            x = real_env.sample_initial(rng, **kw)
        except TypeError:
            x = real_env.sample_initial(rng)
        ekf = EKF(surrogate, real_env.Q, real_env.R_v, x0_mean=x.copy(), P0=P0)
        cum_disc, g_disc, exited = 0.0, 1.0, False
        info_buf, adv_buf = [], []
        for k in range(H):
            s = x if use_true_state else ekf.x_hat
            u, info = policy.act(s, rng, deterministic=False)
            x_next, y, r = real_env.sample_step(x, u, rng)
            ekf.step(u, y)
            err_filter.append(float(np.linalg.norm(ekf.x_hat - x_next)))
            cum_disc += g_disc * r
            g_disc *= gamma
            info_buf.append(info)
            adv_buf.append(r)
            x = x_next
            if stop_on_exit and not real_env.in_safe(x, margin=safety_margin):
                exited = True
                # Apply equivalent -R_max penalty for remaining steps.
                if g_disc > 0:
                    rem = max(0, H - k - 1)
                    penalty = -R_max * g_disc * (1.0 - gamma ** rem) / max(1e-12, 1.0 - gamma)
                    cum_disc += penalty
                break
        returns.append(cum_disc)
        exits.append(int(exited))
        if len(info_buf) > 0:
            adv = np.array(adv_buf) - np.mean(adv_buf)
            gamma_pow = gamma ** np.arange(len(adv))
            w = gamma_pow * adv
            grads = policy.score_grad(info_buf, weights=w)
            grad_norm_sq_samples.append(
                sum(float((g.ravel() ** 2).sum()) for g in grads))
    return {
        "returns": np.asarray(returns),
        "exit_prob": float(np.mean(exits)),
        "filter_err_mean": float(np.mean(err_filter)) if err_filter else 0.0,
        "grad_norm_sq_mean": float(np.mean(grad_norm_sq_samples)) if grad_norm_sq_samples else 0.0,
    }
