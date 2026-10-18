"""Planar quadrotor environment (§4.2).

State: x = (p_x, p_z, phi, v_x, v_z, dphi).
Control: u = (T, tau) with T ∈ [0, T_max], tau ∈ [-tau_max, tau_max].

Observation: y = (p_x, p_z, phi) — only positions are measured.

Nominal: rigid-body 2-D quadrotor (no drag, no propeller dynamics).
Real:    nominal + aerodynamic drag (− c_d ||v|| v) on the (v_x, v_z) channel
         + thrust mismatch (T_real = (1 + delta_T) T_cmd).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class QuadConfig:
    mass: float = 1.0
    inertia: float = 0.02
    arm: float = 0.1
    g: float = 9.81
    dt: float = 0.02
    T_max: float = 2.5 * 9.81   # ≈ 2.5 g thrust
    tau_max: float = 0.5
    # Real residual params (unknown to algorithms).
    drag: float = 0.15
    thrust_gain: float = 1.10
    # Noise.
    Q: Tuple[float, ...] = (1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4)
    R_v: Tuple[float, ...] = (5e-3, 5e-3, 5e-3)
    # Safety.
    pos_box: float = 3.0
    vel_box: float = 4.0
    phi_box: float = 0.6
    r_margin: float = 0.2
    R_max: float = 5.0
    gamma: float = 0.99
    K_train: int = 60
    H_eval: int = 120


class PlanarQuadrotor:
    def __init__(self, mode: str = "real",
                 residual_fn: Optional[Callable] = None,
                 h_residual_fn: Optional[Callable] = None,
                 cfg: Optional[QuadConfig] = None,
                 dr_params: Optional[dict] = None):
        self.cfg = cfg or QuadConfig()
        self.mode = mode
        self.dr = dr_params
        self._residual_fn = residual_fn
        self._h_residual_fn = h_residual_fn
        self.n_x = 6
        self.n_u = 2
        self.n_y = 3
        self.Q = np.diag(self.cfg.Q)
        self.R_v = np.diag(self.cfg.R_v)

    def _params(self):
        c = self.cfg
        if self.dr is None:
            return c.mass, c.inertia, c.arm, c.g, c.drag, c.thrust_gain
        return (self.dr.get("mass", c.mass),
                self.dr.get("inertia", c.inertia),
                c.arm, c.g,
                self.dr.get("drag", c.drag),
                self.dr.get("thrust_gain", c.thrust_gain))

    def _nominal_step(self, x: np.ndarray, u: np.ndarray,
                      thrust_gain: float = 1.0, drag: float = 0.0) -> np.ndarray:
        c = self.cfg
        m, J, arm, g, _, _ = self._params()
        dt = c.dt
        px, pz, phi, vx, vz, w = x[..., 0], x[..., 1], x[..., 2], x[..., 3], x[..., 4], x[..., 5]
        T_cmd = np.clip(u[..., 0], 0.0, c.T_max)
        tau_cmd = np.clip(u[..., 1], -c.tau_max, c.tau_max)

        T_eff = thrust_gain * T_cmd
        # Body-frame thrust → world.
        s, ch = np.sin(phi), np.cos(phi)
        a_x = -T_eff * s / m
        a_z = (T_eff * ch) / m - g
        # Drag (residual).
        speed = np.sqrt(vx ** 2 + vz ** 2 + 1e-8)
        a_x = a_x - drag * speed * vx / m
        a_z = a_z - drag * speed * vz / m
        a_phi = tau_cmd / J

        vx_new = vx + dt * a_x
        vz_new = vz + dt * a_z
        w_new = w + dt * a_phi
        px_new = px + dt * vx_new
        pz_new = pz + dt * vz_new
        phi_new = phi + dt * w_new
        return np.stack([px_new, pz_new, phi_new, vx_new, vz_new, w_new], axis=-1)

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        c = self.cfg
        m, J, arm, g, drag, tg = self._params()
        if self.mode == "real":
            return self._nominal_step(x, u, thrust_gain=tg, drag=drag)
        if self.mode == "nominal":
            return self._nominal_step(x, u, thrust_gain=1.0, drag=0.0)
        if self.mode == "augmented":
            base = self._nominal_step(x, u, thrust_gain=1.0, drag=0.0)
            res = self._residual_fn(x, u)
            return base + res
        raise ValueError(self.mode)

    def h(self, x: np.ndarray) -> np.ndarray:
        base = x[..., :3]
        if self.mode == "augmented" and self._h_residual_fn is not None:
            dh = np.atleast_1d(self._h_residual_fn(x))
            return base + dh.reshape(base.shape)
        return base

    def Jacobians(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        m, J, arm, g, _, _ = self._params()
        dt = c.dt
        phi = x[..., 2]
        s, ch = np.sin(phi), np.cos(phi)
        T_cmd = np.clip(u[..., 0], 0.0, c.T_max)
        # Nominal Jacobian (no drag, thrust_gain = 1).
        A = np.zeros(x.shape[:-1] + (6, 6))
        # State indices: 0=px, 1=pz, 2=phi, 3=vx, 4=vz, 5=w.
        # px_new = px + dt vx_new = px + dt (vx + dt a_x)
        A[..., 0, 0] = 1
        A[..., 0, 3] = dt
        A[..., 0, 2] = -dt ** 2 * T_cmd * ch / m
        A[..., 1, 1] = 1
        A[..., 1, 4] = dt
        A[..., 1, 2] = -dt ** 2 * T_cmd * s / m
        A[..., 2, 2] = 1
        A[..., 2, 5] = dt
        A[..., 3, 3] = 1
        A[..., 3, 2] = -dt * T_cmd * ch / m
        A[..., 4, 4] = 1
        A[..., 4, 2] = -dt * T_cmd * s / m
        A[..., 5, 5] = 1
        C = np.zeros(x.shape[:-1] + (3, 6))
        C[..., 0, 0] = 1
        C[..., 1, 1] = 1
        C[..., 2, 2] = 1
        return A, C

    # ----------------------------------------------------- reward / sampling -
    def reward(self, x: np.ndarray, u: np.ndarray) -> float:
        c = self.cfg
        # Track origin (hover at (0,0,0)).
        cost = (x[..., 0] ** 2 + x[..., 1] ** 2 + 0.5 * x[..., 2] ** 2
                + 0.1 * x[..., 3] ** 2 + 0.1 * x[..., 4] ** 2 + 0.05 * x[..., 5] ** 2
                + 0.001 * u[..., 0] ** 2 + 0.001 * u[..., 1] ** 2)
        r = -cost
        return float(np.clip(r, -c.R_max, 0.0))

    def sample_step(self, x: np.ndarray, u: np.ndarray, rng: np.random.Generator):
        x_next = self.f(x, u) + rng.multivariate_normal(np.zeros(self.n_x), self.Q)
        y = self.h(x_next) + rng.multivariate_normal(np.zeros(self.n_y), self.R_v)
        r = self.reward(x, u)
        return x_next, y, r

    def in_safe(self, x: np.ndarray, margin: float = 0.0) -> bool:
        c = self.cfg
        return (abs(x[0]) <= c.pos_box - margin and
                abs(x[1]) <= c.pos_box - margin and
                abs(x[2]) <= c.phi_box - margin and
                abs(x[3]) <= c.vel_box - margin and
                abs(x[4]) <= c.vel_box - margin and
                abs(x[5]) <= c.vel_box - margin)

    def sample_initial(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.25, 0.25),
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.05, 0.05),
            rng.uniform(-0.05, 0.05),
        ])

    @classmethod
    def random_dr_real(cls, rng: np.random.Generator) -> "PlanarQuadrotor":
        dr = {
            "drag":        float(rng.uniform(0.0, 0.6)),
            "thrust_gain": float(rng.uniform(0.85, 1.2)),
        }
        return cls(mode="real", dr_params=dr)
