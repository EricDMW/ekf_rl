"""Extended Kalman Filter (§II of `root.tex`).

Operates on a *surrogate* dynamics object that exposes ``f(x,u)``, ``h(x)``,
and ``Jacobians(x,u) -> (A, C)``. The filter is intentionally agnostic to
the underlying environment — the surrogate is the *nominal-plus-residual*
model that both training and deployment use, per the paper.
"""
from __future__ import annotations

import numpy as np


class EKF:
    def __init__(self, surrogate, Q: np.ndarray, R_v: np.ndarray,
                 x0_mean: np.ndarray, P0: np.ndarray):
        self.s = surrogate
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R_v, dtype=float)
        self.x_hat = np.array(x0_mean, dtype=float).copy()
        self.P = np.array(P0, dtype=float).copy()
        # Riccati-bound cache (used by analysis only).
        self._n_x = self.x_hat.size
        self._n_y = self.R.shape[0]

    def step(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Run one EKF step with control u and measurement y. Returns x_hat."""
        s = self.s
        # Time update.
        A, _ = s.Jacobians(self.x_hat, u)
        x_pred = s.f(self.x_hat, u)
        P_pred = A @ self.P @ A.T + self.Q
        # Measurement update.
        _, C = s.Jacobians(x_pred, u)
        # Use Cholesky-based gain.
        S = C @ P_pred @ C.T + self.R
        try:
            L = np.linalg.cholesky(S)
            inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(S.shape[0])))
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(S)
        K = P_pred @ C.T @ inv
        y_pred = s.h(x_pred)
        innov = np.atleast_1d(y) - np.atleast_1d(y_pred)
        self.x_hat = x_pred + K @ innov
        I = np.eye(self._n_x)
        self.P = (I - K @ C) @ P_pred
        # Symmetrize for numerical hygiene.
        self.P = 0.5 * (self.P + self.P.T)
        return self.x_hat.copy()

    def reset(self, x0_mean: np.ndarray, P0: np.ndarray):
        self.x_hat = np.asarray(x0_mean, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
