"""Squashed-Gaussian MLP policy (Assumption A4) — pure NumPy.

The policy is

    z = mu_theta(s) + sigma_theta * xi,  xi ~ N(0, I)
    u = u_max * tanh(z)

with state-independent diagonal log-sigma. mu_theta is a 2-layer MLP with
tanh activations. We provide closed-form score-function gradients.

Value baseline is a similar small MLP fit via TD(λ).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


# ------------------------------------------------------------- utilities --
def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softplus_grad(x):
    return 1.0 / (1.0 + np.exp(-x))


# -------------------------------------------------------------- policy --
class SquashedGaussianPolicy:
    """A small squashed-Gaussian policy with diagonal state-independent sigma."""

    def __init__(self, n_x: int, n_u: int, u_max: np.ndarray,
                 hidden: Tuple[int, int] = (64, 64),
                 sigma_floor: float = 0.05,
                 rng: np.random.Generator | None = None):
        self.n_x = n_x
        self.n_u = n_u
        self.u_max = np.asarray(u_max, dtype=float).reshape(n_u)
        self.hidden = hidden
        self.sigma_floor = sigma_floor
        rng = rng or np.random.default_rng(0)

        # Mean network: input n_x -> hidden[0] -> hidden[1] -> n_u (pre-tanh).
        h1, h2 = hidden
        scale = 0.5
        self.W1 = rng.standard_normal((n_x, h1)) * scale / np.sqrt(n_x)
        self.b1 = np.zeros(h1)
        self.W2 = rng.standard_normal((h1, h2)) * scale / np.sqrt(h1)
        self.b2 = np.zeros(h2)
        self.W3 = rng.standard_normal((h2, n_u)) * 0.05 / np.sqrt(h2)
        self.b3 = np.zeros(n_u)
        # Diagonal log-sigma parameter (pre-softplus).
        self.log_sigma = np.full(n_u, -0.7)  # softplus(-0.7) ≈ 0.40

    @property
    def params(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.log_sigma]

    @params.setter
    def params(self, new):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.log_sigma = new

    def sigma(self) -> np.ndarray:
        return self.sigma_floor + softplus(self.log_sigma)

    def _forward_mu(self, s: np.ndarray):
        h1 = np.tanh(s @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        mu = h2 @ self.W3 + self.b3
        return mu, h1, h2

    def act(self, s: np.ndarray, rng: np.random.Generator, deterministic: bool = False):
        s = np.atleast_1d(s).astype(float)
        single = s.ndim == 1
        if single:
            s = s[None, :]
        mu, h1, h2 = self._forward_mu(s)
        sigma = self.sigma()
        if deterministic:
            xi = np.zeros_like(mu)
        else:
            xi = rng.standard_normal(mu.shape)
        z = mu + sigma * xi
        u = self.u_max * np.tanh(z)
        if single:
            return u[0], {"z": z[0], "mu": mu[0], "xi": xi[0], "h1": h1[0], "h2": h2[0], "s": s[0]}
        return u, {"z": z, "mu": mu, "xi": xi, "h1": h1, "h2": h2, "s": s}

    # ----------- weighted score: sum_i w_i ∇_theta log π(u_i|s_i) -----------
    def score_grad(self, info_batch: List[dict],
                   weights: np.ndarray | None = None) -> List[np.ndarray]:
        """Compute ∇_theta log π(u|s) summed over a batch.

        Uses the fact that for u = u_max * tanh(z), z = μ + σ ξ, ξ ~ N(0,I):
            log π(u|s) = -1/2 ||ξ||² - sum log σ - sum log(u_max (1 - tanh²(z))) + const
        Only μ and log_sigma depend on θ, so:
            ∂ log π / ∂ μ_i = - ξ_i / σ_i + 2 tanh(z_i) * 1 / σ_i   (from chain rule)
            wait — for the squash term, d/dz log(1 - tanh²(z)) = -2 tanh(z).
            and z depends on μ and σ.
        Let g_z := ∂ log π / ∂ z = -ξ/σ - (-2 tanh(z)) = (2 tanh(z) - ξ)/σ ? Let's redo.

        log π(u|s) = -1/2 ξ² - log σ - log u_max - log(1 - tanh² z) + const, where
            ξ = (z - μ)/σ, z = atanh(u/u_max) (a function of u, σ-fixed once σ).
        For score, holding u fixed, the dependence on θ is through (μ, σ) acting on
        the log-density at fixed u:
            ∂_μ log π = (z - μ)/σ² = ξ/σ
            ∂_σ log π = -1/σ + (z - μ)² / σ³ = -1/σ + ξ²/σ = (ξ²-1)/σ

        Note: the tanh-Jacobian log-correction depends on z = atanh(u/u_max),
        which is a function of u only and NOT of θ at fixed u. So it drops out
        of the score w.r.t. θ. Good.
        """
        # Stack.
        s = np.stack([info["s"] for info in info_batch], axis=0)
        h1 = np.stack([info["h1"] for info in info_batch], axis=0)
        h2 = np.stack([info["h2"] for info in info_batch], axis=0)
        mu = np.stack([info["mu"] for info in info_batch], axis=0)
        xi = np.stack([info["xi"] for info in info_batch], axis=0)
        sigma = self.sigma()
        B = xi.shape[0]
        if weights is None:
            w = np.ones(B)
        else:
            w = np.asarray(weights, dtype=float).reshape(B)
        # Per-sample score factors.
        g_mu = (xi / sigma) * w[:, None]                       # (B, n_u)
        g_sigma_per = ((xi ** 2 - 1.0) / sigma) * w[:, None]   # (B, n_u)
        d_sigma_d_logsigma = softplus_grad(self.log_sigma)  # (n_u,)
        # Backprop through MLP for W3, b3 first.
        # mu = h2 @ W3 + b3. dL/d(W3) = h2^T @ g_mu; dL/d(b3) = sum_b g_mu.
        # dL/d(h2) = g_mu @ W3.T then through tanh.
        grad_W3 = h2.T @ g_mu  # (h2, n_u)
        grad_b3 = g_mu.sum(axis=0)
        g_h2_pre = (g_mu @ self.W3.T) * (1.0 - h2 ** 2)  # (B, h2)
        grad_W2 = h1.T @ g_h2_pre
        grad_b2 = g_h2_pre.sum(axis=0)
        g_h1_pre = (g_h2_pre @ self.W2.T) * (1.0 - h1 ** 2)
        grad_W1 = s.T @ g_h1_pre
        grad_b1 = g_h1_pre.sum(axis=0)
        grad_logsigma = (g_sigma_per * d_sigma_d_logsigma).sum(axis=0)
        return [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3, grad_logsigma]

    # ---- Apply a parameter delta (vector of gradient summed over a batch) ----
    def add_grad(self, grads: List[np.ndarray], lr: float, weighted: bool = False):
        self.W1 += lr * grads[0]
        self.b1 += lr * grads[1]
        self.W2 += lr * grads[2]
        self.b2 += lr * grads[3]
        self.W3 += lr * grads[4]
        self.b3 += lr * grads[5]
        self.log_sigma += lr * grads[6]
        # Clip log_sigma to avoid runaway exploration / collapse.
        self.log_sigma = np.clip(self.log_sigma, -2.0, 1.0)

    # ---- Adam state and update ----
    def init_adam(self):
        self._adam_m = [np.zeros_like(p) for p in self.params]
        self._adam_v = [np.zeros_like(p) for p in self.params]
        self._adam_t = 0

    def adam_step(self, grads: List[np.ndarray], lr: float,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        if not hasattr(self, "_adam_m"):
            self.init_adam()
        self._adam_t += 1
        t = self._adam_t
        for i, g in enumerate(grads):
            self._adam_m[i] = beta1 * self._adam_m[i] + (1 - beta1) * g
            self._adam_v[i] = beta2 * self._adam_v[i] + (1 - beta2) * g * g
            m_hat = self._adam_m[i] / (1 - beta1 ** t)
            v_hat = self._adam_v[i] / (1 - beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            if i == 0:
                self.W1 += update
            elif i == 1:
                self.b1 += update
            elif i == 2:
                self.W2 += update
            elif i == 3:
                self.b2 += update
            elif i == 4:
                self.W3 += update
            elif i == 5:
                self.b3 += update
            elif i == 6:
                self.log_sigma += update
        self.log_sigma = np.clip(self.log_sigma, -2.0, 1.0)

    def fisher_norm_sq(self, info_batch: List[dict]) -> float:
        """Estimate E ||∇_θ log π||² as a Fisher-information proxy (used for diagnostic)."""
        grads = self.score_grad(info_batch)
        return sum(float((g.ravel() ** 2).sum()) for g in grads) / max(1, len(info_batch))


# ------------------------------------------------------ value baseline --
class ValueBaseline:
    """Two-layer MLP baseline, fit by Adam-style SGD on TD(λ) returns."""

    def __init__(self, n_x: int, hidden: int = 64,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        scale = 0.5
        self.W1 = rng.standard_normal((n_x, hidden)) * scale / np.sqrt(n_x)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * scale / np.sqrt(hidden)
        self.b2 = np.zeros(1)

    def predict(self, s: np.ndarray) -> np.ndarray:
        s = np.atleast_2d(np.asarray(s, dtype=float))
        h = np.tanh(s @ self.W1 + self.b1)
        v = h @ self.W2 + self.b2
        return v[:, 0]

    def fit_step(self, s: np.ndarray, target: np.ndarray, lr: float = 1e-2):
        s = np.atleast_2d(np.asarray(s, dtype=float))
        target = np.asarray(target, dtype=float).reshape(-1, 1)
        h = np.tanh(s @ self.W1 + self.b1)
        v = h @ self.W2 + self.b2
        diff = (v - target)
        # MSE gradient.
        dL_dW2 = h.T @ diff / s.shape[0]
        dL_db2 = diff.mean(axis=0)
        dL_dh = diff @ self.W2.T
        dL_dh_pre = dL_dh * (1.0 - h ** 2)
        dL_dW1 = s.T @ dL_dh_pre / s.shape[0]
        dL_db1 = dL_dh_pre.mean(axis=0)
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1
        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2
        return float(np.mean(diff ** 2))
