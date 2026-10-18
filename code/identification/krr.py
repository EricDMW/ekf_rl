"""Constrained kernel-ridge regression for residual identification.

We use a Matérn-5/2 kernel, which corresponds to Sobolev smoothness
``s = 5/2 + d/2`` in the sense that its RKHS is norm-equivalent to W^{s,2}.
The Sobolev norm constraint of `root.tex` Theorem 6 is enforced softly via
the ridge parameter, chosen on a held-out 20 % split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


# ---------------------------------------------------------------- kernels --
def matern52(X1: np.ndarray, X2: np.ndarray, ell: float) -> np.ndarray:
    """Matern 5/2 kernel: k(r) = (1 + sqrt(5)r/ell + 5 r^2 / (3 ell^2)) exp(-sqrt(5)r/ell)."""
    diff = X1[:, None, :] - X2[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    sqrt5_r = np.sqrt(5.0) * r / max(ell, 1e-8)
    return (1.0 + sqrt5_r + sqrt5_r ** 2 / 3.0) * np.exp(-sqrt5_r)


def median_heuristic_bandwidth(X: np.ndarray, sample: int = 200,
                               rng: np.random.Generator | None = None) -> float:
    n = X.shape[0]
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    Xs = X[idx]
    diff = Xs[:, None, :] - Xs[None, :, :]
    d2 = np.sum(diff ** 2, axis=-1)
    return float(np.sqrt(np.median(d2[d2 > 0]) + 1e-9))


# ------------------------------------------------------------ regressor --
@dataclass
class MaternKRR:
    """Vector-valued KRR with Matern 5/2 kernel (d-input -> p-output).

    Trained via ``fit(X, Y)``; ``predict(X)`` returns the mean.
    Ridge is set by cross-validation on a small grid.
    """
    ell: float = 1.0
    ridge: float = 1e-2

    def __post_init__(self):
        self._X = None
        self._alpha = None  # shape (n, p)

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray,
            ridge_grid: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
            seed: int = 0) -> "MaternKRR":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        ell = median_heuristic_bandwidth(X, rng=rng)
        # Train/val split.
        idx = rng.permutation(n)
        n_val = max(1, n // 5)
        val, tr = idx[:n_val], idx[n_val:]
        K_tr = matern52(X[tr], X[tr], ell)
        K_val = matern52(X[val], X[tr], ell)
        I_tr = np.eye(K_tr.shape[0])
        best_ridge, best_err = ridge_grid[0], np.inf
        for r in ridge_grid:
            try:
                A = np.linalg.solve(K_tr + r * I_tr, Y[tr])
            except np.linalg.LinAlgError:
                continue
            pred = K_val @ A
            err = float(np.mean((pred - Y[val]) ** 2))
            if err < best_err:
                best_err = err
                best_ridge = r
        # Refit on full data.
        K_full = matern52(X, X, ell)
        alpha = np.linalg.solve(K_full + best_ridge * np.eye(n), Y)
        obj = cls(ell=ell, ridge=best_ridge)
        obj._X = X
        obj._alpha = alpha
        return obj

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._X is None:
            raise RuntimeError("not fitted")
        K = matern52(X, self._X, self.ell)
        out = K @ self._alpha
        if out.shape[-1] == 1:
            return out[..., 0]
        return out


# --------------------------------------------------- residual fitting --
def fit_residuals(real_env, nominal_env, n_samples: int,
                  rng: np.random.Generator,
                  domain_box_x: Tuple[Tuple[float, float], ...] = None,
                  domain_box_u: Tuple[Tuple[float, float], ...] = None):
    """Phase-1 reset-sampled identification.

    Returns (f_residual_callable, h_residual_callable, raw_data_dict).

    The residual callables have signature ``Δ_f(x, u)`` → np.ndarray of same shape
    as ``x``, and ``Δ_h(x)`` → np.ndarray of same shape as ``h(x)``.
    """
    real = real_env
    nom = nominal_env
    n_x, n_u, n_y = real.n_x, real.n_u, real.n_y

    # Domain boxes: default to env safe set for x, full control box for u.
    if domain_box_x is None:
        domain_box_x = _default_x_box(real)
    if domain_box_u is None:
        domain_box_u = _default_u_box(real)
    x_lo, x_hi = np.array([b[0] for b in domain_box_x]), np.array([b[1] for b in domain_box_x])
    u_lo, u_hi = np.array([b[0] for b in domain_box_u]), np.array([b[1] for b in domain_box_u])

    # Sample i.i.d. (x_i, u_i).
    X_state = rng.uniform(low=x_lo, high=x_hi, size=(n_samples, n_x))
    U_ctrl = rng.uniform(low=u_lo, high=u_hi, size=(n_samples, n_u))

    # Per Assumption A6: z_i = (x_i, u_i); x_i^+ = f_real(x_i,u_i) + w_i;
    # y_i = h_real(x_i) + v_i (observation at x_i, NOT at x_next).
    XU = np.concatenate([X_state, U_ctrl], axis=1)
    x_next_real = np.zeros_like(X_state)
    y_obs = np.zeros((n_samples, n_y))
    for i in range(n_samples):
        w = rng.multivariate_normal(np.zeros(n_x), real.Q)
        v = rng.multivariate_normal(np.zeros(n_y), real.R_v)
        x_next_real[i] = real.f(X_state[i], U_ctrl[i]) + w
        y_obs[i] = np.atleast_1d(real.h(X_state[i])).ravel() + v

    x_next_nom = np.zeros_like(X_state)
    y_nom = np.zeros((n_samples, n_y))
    for i in range(n_samples):
        x_next_nom[i] = nom.f(X_state[i], U_ctrl[i])
        y_nom[i] = np.atleast_1d(nom.h(X_state[i])).ravel()

    delta_f_target = x_next_real - x_next_nom
    delta_h_target = y_obs - y_nom

    # Fit f residual on (x, u).
    f_krr = MaternKRR.fit(XU, delta_f_target)
    # Fit h residual on x only.
    h_krr = MaternKRR.fit(X_state, delta_h_target)

    def f_residual(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        if x.ndim == 1:
            xu = np.concatenate([x, np.atleast_1d(u).reshape(-1)])
            pred = f_krr.predict(xu[None, :])[0]
            # Convert "delta x_next" back into a "delta f" by dividing dt — but
            # we treat this as a one-step *update* residual, which is what
            # the augmented env expects (it adds dt * residual). Actually we
            # want a residual that, when added to f_nominal(x,u), gives
            # f_real(x,u). Since delta_f_target = x_next_real - x_next_nom is
            # already an "added next-state delta", we return it as-is and the
            # augmented env should add it without an extra dt scale.
            return pred
        # batch case
        u_b = u.reshape(x.shape[0], -1) if u.ndim > 0 else np.broadcast_to(u, (x.shape[0], 1))
        xu = np.concatenate([x, u_b], axis=-1)
        return f_krr.predict(xu)

    def h_residual(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return h_krr.predict(x[None, :])[0]
        return h_krr.predict(x)

    info = {
        "n_samples": int(n_samples),
        "ell_f": float(f_krr.ell),
        "ell_h": float(h_krr.ell),
        "ridge_f": float(f_krr.ridge),
        "ridge_h": float(h_krr.ridge),
        # Training MSE (cheap sanity).
        "mse_f": float(np.mean((f_krr.predict(XU) - delta_f_target) ** 2)),
        "mse_h": float(np.mean((h_krr.predict(X_state) - delta_h_target) ** 2)),
    }
    return f_residual, h_residual, info


def _default_x_box(env):
    if hasattr(env, "cfg") and hasattr(env.cfg, "theta_box"):
        return ((-env.cfg.theta_box, env.cfg.theta_box),
                (-env.cfg.dtheta_box, env.cfg.dtheta_box))
    if hasattr(env, "cfg") and hasattr(env.cfg, "pos_box"):
        c = env.cfg
        return ((-c.pos_box, c.pos_box),
                (-c.pos_box, c.pos_box),
                (-c.phi_box, c.phi_box),
                (-c.vel_box, c.vel_box),
                (-c.vel_box, c.vel_box),
                (-c.vel_box, c.vel_box))
    raise ValueError("unknown env type")


def _default_u_box(env):
    if hasattr(env, "cfg") and hasattr(env.cfg, "u_max"):
        return ((-env.cfg.u_max, env.cfg.u_max),)
    if hasattr(env, "cfg") and hasattr(env.cfg, "T_max"):
        return ((0.0, env.cfg.T_max), (-env.cfg.tau_max, env.cfg.tau_max))
    raise ValueError("unknown env type")
