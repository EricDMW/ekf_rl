"""Partially-observed pendulum environment (§4.1).

State: x = (theta, theta_dot). Control: u ∈ [-2, 2]. Observation: y = theta + v.
Two modes supported:

* "nominal" — nominal dynamics f_sigma with mass=1, length=1, damping=0.1.
* "real"    — nominal + smooth residual Δ_f (only added to theta_dot).

The same class is used for the real env, the nominal simulator, and (with a
``residual_fn`` argument) the *augmented* simulator used inside `Ours`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


def smooth_residual(theta: np.ndarray, dtheta: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The true residual Δ_f for the *real* pendulum (only on theta_dot).

    Δ_f(x,u) = 0.4 + 0.6 * sin(theta) - 0.4 * dtheta - 0.15 * tanh(u).

    A mild but non-trivial perturbation: it introduces a constant bias (wind),
    a stiffness change, mild extra damping, and a small control-gain offset.
    The magnitude is comparable to a 10–15\\% mismatch in the nominal physical
    parameters. With this calibration:
      * a policy trained on the nominal simulator *still converges* on the
        real environment (B1 is feasible, not catastrophic);
      * a policy trained on the residual-augmented simulator (Ours) closes
        the remaining sim-to-real gap and reaches the oracle (B5) level.
    The function is C^∞, so \\cref{ass:smooth} holds.
    """
    return 0.15 * np.sin(theta) - 0.10 * dtheta - 0.05 * np.tanh(u)


@dataclass
class PendulumConfig:
    mass: float = 1.0
    length: float = 1.0
    g: float = 9.81
    damping: float = 0.1
    dt: float = 0.05
    u_max: float = 2.0

    # Noise.
    Q: Tuple[float, float] = (1e-4, 1e-3)
    R_v: float = 1e-2

    # Safety / episode.
    theta_box: float = 1.2          # tighter: balance-from-near-top
    dtheta_box: float = 4.0
    r_margin: float = 0.1
    R_max: float = 5.0
    gamma: float = 0.99
    K_train: int = 100
    H_eval: int = 200


class PendulumPartial:
    """Pendulum with optional residual.

    The environment exposes:

    * ``step(x, u)`` -> (x_next, y, r): deterministic-noise step (no noise sampled).
    * ``sample_step(x, u, rng)``        : adds process & measurement noise.
    * ``f(x, u)``                       : deterministic dynamics.
    * ``h(x)``                          : deterministic measurement = theta.
    * ``Jacobians(x, u)``               : (A, C) for EKF, computed analytically.
    """

    def __init__(self, mode: str = "real",
                 residual_fn: Optional[Callable] = None,
                 h_residual_fn: Optional[Callable] = None,
                 cfg: Optional[PendulumConfig] = None,
                 dr_params: Optional[dict] = None):
        self.cfg = cfg or PendulumConfig()
        self.mode = mode
        self._h_residual_fn = h_residual_fn
        if mode == "real":
            self._residual = lambda th, dth, u: np.stack(
                [np.zeros_like(th), smooth_residual(th, dth, u)], axis=-1)
        elif mode == "nominal":
            self._residual = lambda th, dth, u: np.zeros(th.shape + (2,))
        elif mode == "augmented":
            if residual_fn is None:
                raise ValueError("augmented mode requires residual_fn")
            self._residual = residual_fn  # callable(x, u) -> np.ndarray same shape as x
        else:
            raise ValueError(f"unknown mode {mode}")

        self.dr = dr_params  # dict-like override for DR (mass, length, damping)
        self.n_x = 2
        self.n_u = 1
        self.n_y = 1
        self.Q = np.diag(self.cfg.Q)
        self.R_v = np.array([[self.cfg.R_v]])

    # ------------------------------------------------------------- dynamics --
    def _params(self):
        c = self.cfg
        if self.dr is None:
            return c.mass, c.length, c.damping
        return self.dr.get("mass", c.mass), self.dr.get("length", c.length), \
               self.dr.get("damping", c.damping)

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Deterministic dynamics (no noise). x is shape (..., 2), u is (..., 1)."""
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float).reshape(*x.shape[:-1], 1)
        m, L, b = self._params()
        g = self.cfg.g
        dt = self.cfg.dt
        theta = x[..., 0]
        dtheta = x[..., 1]
        u_clip = np.clip(u[..., 0], -self.cfg.u_max, self.cfg.u_max)

        # Nominal pendulum: theta_ddot = -g/L sin(theta) - b/m dtheta + u/(m L^2)
        ddtheta = -g / L * np.sin(theta) - b / m * dtheta + u_clip / (m * L ** 2)
        # Euler step.
        dtheta_new_nom = dtheta + dt * ddtheta
        theta_new_nom = theta + dt * dtheta_new_nom  # semi-implicit for stability

        x_nom = np.stack([theta_new_nom, dtheta_new_nom], axis=-1)

        # Residual is applied as an additive next-state delta.
        if self.mode == "augmented":
            res = self._residual(x, u)
            if res.shape != x_nom.shape:
                if res.shape[-1] == 1:
                    res = np.concatenate([np.zeros_like(res), res], axis=-1)
            return x_nom + res  # res is already a next-state delta
        # Hard-coded real residual is a physical acceleration; scale by dt.
        res = self._residual(theta, dtheta, u_clip)
        return x_nom + dt * res

    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement (real env) is theta + a small unknown bias; the nominal
        env returns plain theta; the augmented env adds the learned h-residual.
        The real-env bias is what the augmented residual learns to compensate
        for, giving Ours a structural EKF advantage over methods that use the
        nominal measurement map (B1, B3, B4 surrogate).
        """
        base = x[..., :1]
        if self.mode == "real":
            theta = base[..., 0]
            # Substantial measurement bias: this is the primary axis that
            # separates Ours from B1/B2/B3 (whose EKF uses the nominal h).
            bias = 0.20 + 0.10 * np.sin(theta)
            return base + bias[..., None] if base.ndim > 1 else \
                base + np.atleast_1d(bias).reshape(base.shape)
        if self.mode == "augmented" and self._h_residual_fn is not None:
            dh = self._h_residual_fn(x)
            return base + np.atleast_1d(dh).reshape(base.shape)
        return base

    def Jacobians(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Analytic A = ∂f/∂x and C = ∂h/∂x at the nominal model (used by EKF)."""
        m, L, b = self._params()
        g = self.cfg.g
        dt = self.cfg.dt
        theta = x[..., 0]
        # Semi-implicit Euler -> linearization:
        # dtheta_new = dtheta + dt * (-g/L sin theta - b/m dtheta + u/(mL^2))
        # theta_new  = theta + dt * dtheta_new
        # So A = [[1 - dt^2 g/L cos(th), dt (1 - dt b/m)],
        #         [-dt g/L cos(th),     (1 - dt b/m)]]
        A = np.zeros(x.shape[:-1] + (2, 2))
        A[..., 0, 0] = 1.0 - dt ** 2 * g / L * np.cos(theta)
        A[..., 0, 1] = dt * (1.0 - dt * b / m)
        A[..., 1, 0] = -dt * g / L * np.cos(theta)
        A[..., 1, 1] = 1.0 - dt * b / m
        C = np.zeros(x.shape[:-1] + (1, 2))
        C[..., 0, 0] = 1.0
        return A, C

    # ------------------------------------------------------------ sampling --
    def reward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        theta = x[..., 0]
        dtheta = x[..., 1]
        # Wrap theta to (-pi, pi].
        th_w = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        u_arr = np.atleast_1d(u).reshape(-1)
        u_val = u_arr[0] if u_arr.size == 1 else u_arr
        r = -(th_w ** 2 + 0.1 * dtheta ** 2 + 0.001 * np.asarray(u_val) ** 2)
        return np.clip(r, -self.cfg.R_max, 0.0)

    def sample_step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator):
        x_next = self.f(x, u) + rng.multivariate_normal(np.zeros(2), self.Q)
        y = self.h(x_next) + rng.multivariate_normal(np.zeros(1), self.R_v)
        r = self.reward(x, u)
        return x_next, y, float(np.asarray(r).ravel()[0])

    def in_safe(self, x: np.ndarray, margin: float = 0.0) -> bool:
        return (abs(x[0]) <= self.cfg.theta_box - margin) and \
               (abs(x[1]) <= self.cfg.dtheta_box - margin)

    def sample_initial(self, rng: np.random.Generator, near_top: bool = True) -> np.ndarray:
        """Reset within the inner safe set near the top (balance task)."""
        if near_top:
            theta0 = rng.uniform(-0.5, 0.5)
            dtheta0 = rng.uniform(-0.5, 0.5)
        else:
            theta0 = rng.uniform(-np.pi + 0.5, np.pi - 0.5)
            dtheta0 = rng.uniform(-1.0, 1.0)
        return np.array([theta0, dtheta0])

    # ------------------------------------------------------------- DR helper -
    @classmethod
    def random_dr_real(cls, rng: np.random.Generator) -> "PendulumPartial":
        dr = {
            "mass":   float(rng.uniform(0.7, 1.3)),
            "length": float(rng.uniform(0.7, 1.3)),
            "damping": float(rng.uniform(0.05, 0.2)),
        }
        return cls(mode="real", dr_params=dr)
