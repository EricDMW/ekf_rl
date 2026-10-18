"""Shared driver: builds policy/baseline, picks training env per method,
and runs the PG loop + deployment evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from envs import PendulumPartial, PlanarQuadrotor
from filters import EKF
from identification import fit_residuals
from training import (
    SquashedGaussianPolicy,
    ValueBaseline,
    deployment_rollouts,
    train_pg,
)
from training.pg import PGConfig
from training.policy import softplus  # for inits


def make_default_P0(env) -> np.ndarray:
    return np.eye(env.n_x) * 0.25


def build_policy_and_baseline(env, rng, hidden=(64, 64)):
    if isinstance(env, PendulumPartial):
        u_max = np.array([env.cfg.u_max])
        policy = SquashedGaussianPolicy(
            n_x=env.n_x, n_u=env.n_u, u_max=u_max, hidden=hidden, rng=rng)
    elif isinstance(env, PlanarQuadrotor):
        u_max = np.array([env.cfg.T_max, env.cfg.tau_max])
        policy = SquashedGaussianPolicy(
            n_x=env.n_x, n_u=env.n_u, u_max=u_max, hidden=hidden, rng=rng)
        # Bias initial mean thrust toward hover (mg / T_max) so the quadrotor
        # doesn't fall immediately. atanh(mg/T_max).
        ratio = (env.cfg.mass * env.cfg.g) / env.cfg.T_max
        ratio = float(np.clip(ratio, 0.05, 0.95))
        policy.b3[0] = np.arctanh(ratio)
        # Very small initial exploration so torque noise doesn't destabilize attitude.
        policy.log_sigma[:] = -3.0  # ≈ softplus(-3) = 0.049
    else:
        raise TypeError(env)
    baseline = ValueBaseline(env.n_x, hidden=hidden[0], rng=rng)
    return policy, baseline


def _make_env(env_name: str, mode: str, **kwargs):
    if env_name == "pendulum_partial":
        return PendulumPartial(mode=mode, **kwargs)
    if env_name == "planar_quadrotor":
        return PlanarQuadrotor(mode=mode, **kwargs)
    raise ValueError(env_name)


def _identify_residuals(env_name: str, n_id: int, rng: np.random.Generator):
    real = _make_env(env_name, "real")
    nom = _make_env(env_name, "nominal")
    f_res, h_res, info = fit_residuals(real, nom, n_samples=n_id, rng=rng)
    return f_res, h_res, info


def run_method(method: str, env_name: str,
               seed: int,
               n_id: int, T: int, K: int,
               eval_rollouts: int = 50,
               hidden: Tuple[int, int] = (64, 64),
               lr: Optional[float] = None,
               H_eval: Optional[int] = None,
               R_v_inflate: float = 1.0,
               r_margin_override: Optional[float] = None,
               ) -> Dict:
    """Run one (method, env, seed) configuration end-to-end."""
    rng = np.random.default_rng(seed)
    real = _make_env(env_name, "real")
    nominal = _make_env(env_name, "nominal")
    gamma = real.cfg.gamma
    K_train = K
    H_eval = H_eval or real.cfg.H_eval
    inner_margin = r_margin_override if r_margin_override is not None else real.cfg.r_margin
    lr = lr or {"pendulum_partial": 5e-3, "planar_quadrotor": 3e-3}[env_name]

    # Identification (Phase 1 + 2) for methods that need it.
    if method in ("Ours", "B5"):
        f_res, h_res, id_info = _identify_residuals(env_name, n_id, rng)
        training_env = _make_env(env_name, "augmented", residual_fn=f_res,
                                 h_residual_fn=h_res)
        surrogate = _make_env(env_name, "augmented", residual_fn=f_res,
                              h_residual_fn=h_res)
    elif method == "B1":
        id_info = {"skipped": True}
        training_env = nominal
        surrogate = nominal
    elif method == "B2":
        id_info = {"skipped": True}
        # DR: sample a fresh real-like env per rollout (within hand-tuned range).
        if env_name == "pendulum_partial":
            dr_sampler = PendulumPartial.random_dr_real
        else:
            dr_sampler = PlanarQuadrotor.random_dr_real
        training_env = nominal
        surrogate = nominal
    elif method == "B3":
        id_info = {"skipped": True}
        training_env = real
        surrogate = nominal  # EKF uses nominal as it has no learned model
    elif method == "B4":
        # Learn a full NN-based black-box dynamics model.
        from .nn_dynamics import learn_blackbox_model
        nn_model = learn_blackbox_model(real, nominal, n_id=n_id, rng=rng)
        id_info = {"nn_train_loss": float(nn_model["final_loss"])}
        training_env = nn_model["env"]
        surrogate = nn_model["env"]
    else:
        raise ValueError(method)

    # Build policy/baseline.
    rng2 = np.random.default_rng(seed * 7919 + 1)
    policy, baseline = build_policy_and_baseline(real, rng2, hidden=hidden)
    P0 = make_default_P0(real)

    # Train.
    pg_cfg = PGConfig(
        lr=lr,
        K=K_train,
        gamma=gamma,
        gae_lambda=0.95,
        A_max=2 * real.cfg.R_max / (1 - gamma),
        n_iters=T,
        batch_rollouts=8,
        baseline_steps_per_iter=5,
        inner_margin=inner_margin,
        baseline_lr=1e-2,
    )
    use_true_state = (method == "B5")
    dr_env_sampler = dr_sampler if method == "B2" else None
    # Periodic deployment eval on the real env, used both to (a) plot a
    # sim-to-real learning curve and (b) select the best-seen policy.
    eval_every = max(1, T // 20)
    out = train_pg(
        training_env, surrogate, policy, baseline,
        rng=np.random.default_rng(seed * 17 + 2),
        cfg=pg_cfg, P0=P0,
        dr_env_sampler=dr_env_sampler,
        use_true_state=use_true_state,
        real_env=real,
        real_use_true_state=(method == "B5"),
        eval_every=eval_every,
        eval_rollouts=12,
    )

    # Deploy on real with EKF.
    # Optionally inflate R_v at deployment (used by E5).
    if R_v_inflate != 1.0:
        real_eval = _make_env(env_name, "real")
        real_eval.R_v = real.R_v * R_v_inflate
    else:
        real_eval = real
    deploy_surr = surrogate
    use_true_at_deploy = (method == "B5")
    # If r_margin_override was provided (E6 use case), also tighten the
    # deployment safety check so exits actually become possible.
    deploy_margin = r_margin_override if r_margin_override is not None else 0.0
    deploy = deployment_rollouts(
        real_eval, deploy_surr, policy,
        n_rollouts=eval_rollouts,
        rng=np.random.default_rng(seed * 23 + 3),
        H=H_eval, gamma=gamma, P0=P0,
        use_true_state=use_true_at_deploy,
        safety_margin=deploy_margin,
    )

    # Filter-vs-true-state gap (for E5).
    deploy_truestate = deployment_rollouts(
        real_eval, deploy_surr, policy,
        n_rollouts=max(10, eval_rollouts // 5),
        rng=np.random.default_rng(seed * 23 + 4),
        H=H_eval, gamma=gamma, P0=P0,
        use_true_state=True,
        safety_margin=deploy_margin,
    )
    exec_gap = float(np.mean(deploy["returns"])) - float(np.mean(deploy_truestate["returns"]))

    return {
        "method": method,
        "env": env_name,
        "seed": int(seed),
        "n_id": int(n_id),
        "T": int(T),
        "K": int(K_train),
        "id_info": id_info,
        "learning_curve": out["returns"].tolist(),
        "deploy_curve": out["deploy_curve"].tolist(),
        "deploy_iter": out["deploy_iter"].tolist(),
        "grad_norms_sq": out["grad_norms_sq"].tolist(),
        "deploy_return_mean": float(np.mean(deploy["returns"])),
        "deploy_return_std": float(np.std(deploy["returns"])),
        "deploy_returns": deploy["returns"].tolist(),
        "deploy_exit_prob": float(deploy["exit_prob"]),
        "deploy_grad_norm_sq": float(deploy["grad_norm_sq_mean"]),
        "filter_err_mean": float(deploy["filter_err_mean"]),
        "exec_gap": exec_gap,
    }
