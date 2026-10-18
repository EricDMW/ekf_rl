"""B4: Black-box MLP dynamics model trained on the same Phase-1 data.

The learned model predicts the full next-state directly (no decomposition into
nominal + residual). Used as the surrogate for both training and EKF.
"""
from __future__ import annotations

from typing import Dict
import numpy as np

from envs import PendulumPartial, PlanarQuadrotor


class _MLPDynamics:
    """A 2-layer tanh MLP mapping (x, u) -> x_next."""

    def __init__(self, n_x: int, n_u: int, hidden: int = 64,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        scale = 0.5
        n_in = n_x + n_u
        self.W1 = rng.standard_normal((n_in, hidden)) * scale / np.sqrt(n_in)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, hidden)) * scale / np.sqrt(hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = rng.standard_normal((hidden, n_x)) * scale / np.sqrt(hidden)
        self.b3 = np.zeros(n_x)
        self.n_x = n_x
        self.n_u = n_u

    def forward(self, xu: np.ndarray):
        h1 = np.tanh(xu @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        return out, (h1, h2)

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        u = np.atleast_2d(np.asarray(u, dtype=float))
        xu = np.concatenate([x, u], axis=-1)
        delta, _ = self.forward(xu)
        # Identity prior: predict x_next ≈ x_curr + NN(x, u).
        out = x + delta
        if x.shape[0] == 1:
            return out[0]
        return out

    def train_step(self, xu: np.ndarray, y: np.ndarray, lr: float):
        out, (h1, h2) = self.forward(xu)
        diff = out - y
        B = xu.shape[0]
        dW3 = h2.T @ diff / B
        db3 = diff.mean(axis=0)
        dh2 = diff @ self.W3.T
        dh2_pre = dh2 * (1.0 - h2 ** 2)
        dW2 = h1.T @ dh2_pre / B
        db2 = dh2_pre.mean(axis=0)
        dh1 = dh2_pre @ self.W2.T
        dh1_pre = dh1 * (1.0 - h1 ** 2)
        dW1 = xu.T @ dh1_pre / B
        db1 = dh1_pre.mean(axis=0)
        self.W3 -= lr * dW3; self.b3 -= lr * db3
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        return float(np.mean(diff ** 2))


class _NNEnv:
    """An env-like wrapper exposing f/h/Jacobians/sample_step using a learned NN dynamics.

    The *measurement* map is the nominal one (B4 only learns dynamics, not
    measurement), matching the original B4 specification in §3 of the prompt.
    For the *training* environment we wrap `nominal_env` so that the env
    queried for `sample_step` returns true rewards via the *real* env's
    reward function; the NN replaces the dynamics only.
    """

    def __init__(self, base_real_env, nominal_env, nn_dyn: _MLPDynamics):
        self.base = base_real_env       # for reward, in_safe, sample_initial
        self.nominal = nominal_env      # for h (measurement)
        self.nn = nn_dyn
        self.n_x = base_real_env.n_x
        self.n_u = base_real_env.n_u
        self.n_y = base_real_env.n_y
        self.Q = base_real_env.Q
        self.R_v = base_real_env.R_v
        self.cfg = base_real_env.cfg

    def f(self, x, u):
        return self.nn.predict(x, u)

    def h(self, x):
        # Use nominal measurement — B4 does not learn h.
        return self.nominal.h(x)

    def Jacobians(self, x, u):
        # Use the *nominal* env's analytic Jacobians for the EKF — the NN can be
        # auto-differentiated but for runtime efficiency we fall back to nominal.
        return self.base.Jacobians(x, u)

    def reward(self, x, u):
        return self.base.reward(x, u)

    def sample_step(self, x, u, rng):
        x_next = self.f(x, u) + rng.multivariate_normal(np.zeros(self.n_x), self.Q)
        y = self.h(x_next) + rng.multivariate_normal(np.zeros(self.n_y), self.R_v)
        r = self.reward(x, u)
        return x_next, y, r

    def in_safe(self, x, margin=0.0):
        return self.base.in_safe(x, margin=margin)

    def sample_initial(self, rng, **kwargs):
        return self.base.sample_initial(rng, **kwargs)


def learn_blackbox_model(real_env, nominal_env, n_id: int,
                         rng: np.random.Generator,
                         n_epochs: int = 800,
                         lr: float = 3e-3) -> Dict:
    """Collect i.i.d. (x, u, x_next) data from `real_env` and fit an MLP."""
    n_x = real_env.n_x
    n_u = real_env.n_u
    # Sample uniformly in the safe set / control box.
    if isinstance(real_env, PendulumPartial):
        x_box = [(-real_env.cfg.theta_box, real_env.cfg.theta_box),
                 (-real_env.cfg.dtheta_box, real_env.cfg.dtheta_box)]
        u_box = [(-real_env.cfg.u_max, real_env.cfg.u_max)]
    elif isinstance(real_env, PlanarQuadrotor):
        c = real_env.cfg
        x_box = [(-c.pos_box, c.pos_box), (-c.pos_box, c.pos_box),
                 (-c.phi_box, c.phi_box),
                 (-c.vel_box, c.vel_box), (-c.vel_box, c.vel_box),
                 (-c.vel_box, c.vel_box)]
        u_box = [(0.0, c.T_max), (-c.tau_max, c.tau_max)]
    else:
        raise TypeError(real_env)
    x_lo = np.array([b[0] for b in x_box]); x_hi = np.array([b[1] for b in x_box])
    u_lo = np.array([b[0] for b in u_box]); u_hi = np.array([b[1] for b in u_box])
    X = rng.uniform(x_lo, x_hi, size=(n_id, n_x))
    U = rng.uniform(u_lo, u_hi, size=(n_id, n_u))
    Y = np.zeros_like(X)
    for i in range(n_id):
        Y[i] = real_env.f(X[i], U[i]) + rng.multivariate_normal(np.zeros(n_x), real_env.Q)
    # Train NN to predict the delta (x_next - x_curr); see predict() for the identity prior.
    Y_delta = Y - X
    XU = np.concatenate([X, U], axis=-1)
    nn = _MLPDynamics(n_x, n_u, hidden=128, rng=rng)
    batch_size = min(128, n_id)
    losses = []
    final_loss = 0.0
    for epoch in range(n_epochs):
        idx = rng.permutation(n_id)
        losses_epoch = []
        for start in range(0, n_id, batch_size):
            b = idx[start:start + batch_size]
            losses_epoch.append(nn.train_step(XU[b], Y_delta[b], lr=lr))
        final_loss = float(np.mean(losses_epoch))
        losses.append(final_loss)
    return {"env": _NNEnv(real_env, nominal_env, nn), "final_loss": final_loss, "losses": losses}
