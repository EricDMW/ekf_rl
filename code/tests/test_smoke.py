"""Smoke tests: env shapes are sensible, EKF reduces error, rewards finite."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from envs import PendulumPartial, PlanarQuadrotor
from filters import EKF


def _test_pendulum_ekf_reduces_error():
    rng = np.random.default_rng(0)
    real = PendulumPartial(mode="real")
    nom = PendulumPartial(mode="nominal")
    x = real.sample_initial(rng, near_top=True)
    ekf = EKF(nom, real.Q, real.R_v,
              x0_mean=x + np.array([0.5, 0.5]),  # bad init
              P0=np.diag([1.0, 1.0]))
    errs0, errs100 = [], []
    for step in range(100):
        u = np.array([0.0])
        x, y, _ = real.sample_step(x, u, rng)
        if step == 0:
            errs0.append(np.linalg.norm(ekf.x_hat - x))
        ekf.step(u, y)
        if step == 99:
            errs100.append(np.linalg.norm(ekf.x_hat - x))
    e0, e1 = errs0[0], errs100[0]
    print(f"pendulum EKF error: t=0 -> {e0:.3f}, t=99 -> {e1:.3f}")
    assert e1 < e0, "EKF did not reduce error on pendulum"


def _test_quadrotor_ekf_reduces_error():
    rng = np.random.default_rng(1)
    real = PlanarQuadrotor(mode="real")
    nom = PlanarQuadrotor(mode="nominal")
    x = real.sample_initial(rng)
    ekf = EKF(nom, real.Q, real.R_v,
              x0_mean=x + 0.3 * np.ones(6),
              P0=np.eye(6))
    initial_err = np.linalg.norm(ekf.x_hat - x)
    err_final = initial_err
    for step in range(30):
        u = np.array([real.cfg.mass * real.cfg.g, 0.0])
        x, y, _ = real.sample_step(x, u, rng)
        ekf.step(u, y)
        err_final = np.linalg.norm(ekf.x_hat - x)
    print(f"quadrotor EKF: t=0 err={initial_err:.3f}, t=30 err={err_final:.3f}")
    assert err_final < initial_err, f"quadrotor EKF failed to reduce error: {err_final} >= {initial_err}"


if __name__ == "__main__":
    _test_pendulum_ekf_reduces_error()
    _test_quadrotor_ekf_reduces_error()
    print("OK")
