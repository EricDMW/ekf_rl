from .pendulum_partial import PendulumPartial
from .planar_quadrotor import PlanarQuadrotor

ENVS = {
    "pendulum_partial": PendulumPartial,
    "planar_quadrotor": PlanarQuadrotor,
}
