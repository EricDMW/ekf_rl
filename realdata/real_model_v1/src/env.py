"""Single-agent Go1 go-to-goal environment.

One robot. First-order kinematic body-velocity dynamics (the exact inverse of the
finite-difference used to extract actions from the mocap data, so the offline
data and this env are consistent):

    x'   = x + (vx cos yaw - vy sin yaw) dt
    y'   = y + (vx sin yaw + vy cos yaw) dt
    yaw' = yaw + omega dt

Constraints enforced by the env:
  * maximum speed: |(vx,vy)| <= V_MAX  and  |omega| <= W_MAX  (action clipped),
  * map boundary: the robot is clamped inside [MAP_X/Y MIN,MAX]; trying to cross
    a wall is penalised (R_BOUND).

Observation (position-aware, 9-D):
    [x, y, cos yaw, sin yaw, gx_body, gy_body, vx, vy, omega]
"""
import numpy as np
from . import config as C

GOAL = np.array(C.GOAL, dtype=np.float64)


def clip_speed(a, v_max=C.V_MAX, w_max=C.W_MAX):
    """Clip a body twist to the admissible speed set (linear disc + yaw box)."""
    a = np.asarray(a, dtype=np.float64).copy()
    v = a[:2]
    sp = np.hypot(v[0], v[1])
    if sp > v_max:
        v *= v_max / (sp + 1e-9)
    a[0], a[1] = v
    a[2] = np.clip(a[2], -w_max, w_max)
    return a


class Go1GoalEnv:
    def __init__(self, rng=None, start_sampler=None, max_steps=C.MAX_EP_STEPS):
        self.dt = C.CTRL_DT
        self.goal = GOAL
        self.max_steps = max_steps
        self.rng = rng or np.random.default_rng(0)
        self.start_sampler = start_sampler   # callable(rng)->(x,y,yaw) or None
        self.bounds = (C.MAP_XMIN, C.MAP_XMAX, C.MAP_YMIN, C.MAP_YMAX)
        self.obs_dim = C.OBS_DIM
        self.act_dim = C.ACT_DIM
        self.reset()

    # -- observation ---------------------------------------------------------
    def _obs(self):
        gx = self.goal[0] - self.x
        gy = self.goal[1] - self.y
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        gxb = c * gx + s * gy
        gyb = -s * gx + c * gy
        return np.array([self.x, self.y, c, s, gxb, gyb,
                         self.v[0], self.v[1], self.v[2]], dtype=np.float32)

    def _sample_start(self):
        xmin, xmax, ymin, ymax = self.bounds
        for _ in range(100):
            x = self.rng.uniform(xmin + 0.1, xmax - 0.1)
            y = self.rng.uniform(ymin + 0.1, ymax - 0.1)
            if np.hypot(x - self.goal[0], y - self.goal[1]) > 3 * C.GOAL_RADIUS:
                return x, y, self.rng.uniform(-np.pi, np.pi)
        return x, y, self.rng.uniform(-np.pi, np.pi)

    def reset(self, pose=None):
        if pose is None:
            pose = (self.start_sampler(self.rng) if self.start_sampler
                    else self._sample_start())
        self.x, self.y, self.yaw = float(pose[0]), float(pose[1]), float(pose[2])
        self.v = np.zeros(3)
        self.t = 0
        self.reached = False
        return self._obs()

    def step(self, action):
        a = clip_speed(action)                       # hard max-speed limit
        d_prev = np.hypot(self.goal[0] - self.x, self.goal[1] - self.y)

        c, s = np.cos(self.yaw), np.sin(self.yaw)
        nx = self.x + (a[0] * c - a[1] * s) * self.dt
        ny = self.y + (a[0] * s + a[1] * c) * self.dt
        xmin, xmax, ymin, ymax = self.bounds

        # boundary: clamp inside the map, penalise the attempted crossing
        hit = (nx < xmin) or (nx > xmax) or (ny < ymin) or (ny > ymax)
        nx = min(max(nx, xmin), xmax)
        ny = min(max(ny, ymin), ymax)

        self.x, self.y = nx, ny
        self.yaw += a[2] * self.dt
        self.v = a
        self.t += 1

        d_new = np.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        reached = d_new < C.GOAL_RADIUS
        r = (C.R_PROGRESS * (d_prev - d_new)
             - C.R_CTRL * float(a @ a)
             - C.R_TIME
             + C.R_GOAL * float(reached)
             - C.R_BOUND * float(hit))
        self.reached = bool(reached)
        terminated = reached                          # TRUE terminal (goal)
        truncated = self.t >= self.max_steps          # timeout (not terminal)
        done = terminated or truncated
        info = dict(dist=d_new, reached=reached, hit=hit,
                    x=self.x, y=self.y, yaw=self.yaw,
                    terminated=terminated, truncated=truncated)
        return self._obs(), r, done, info


# --------------------------------------------------------------------------
# rollout / evaluation helpers
# --------------------------------------------------------------------------
def rollout(policy_fn, env, pose=None):
    o = env.reset(pose=pose)
    xs, ys, yaws, acts = [env.x], [env.y], [env.yaw], []
    tot_r, hits, far_hits, done = 0.0, 0, 0, False
    info = {"dist": np.hypot(env.x - env.goal[0], env.y - env.goal[1])}
    while not done:
        a = policy_fn(o)
        o, r, done, info = env.step(a)
        tot_r += r
        hits += int(info["hit"])
        # a "meaningful" boundary violation is wall contact away from the
        # (cornered) goal -- grazing the wall right at the goal is expected.
        if info["hit"] and info["dist"] > 0.6:
            far_hits += 1
        xs.append(info["x"]); ys.append(info["y"]); yaws.append(info["yaw"])
        acts.append(clip_speed(a))
    xs, ys = np.array(xs), np.array(ys)
    path_len = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    return dict(x=xs, y=ys, yaw=np.array(yaws), acts=np.array(acts),
                reached=env.reached, steps=env.t, ret=tot_r,
                final_dist=info["dist"], wall_hits=hits, far_hits=far_hits,
                path_len=path_len)


def evaluate(policy_fn, env, poses):
    succ, steps, rets, dists, far, opt = [], [], [], [], [], []
    trajs = []
    for p in poses:
        tr = rollout(policy_fn, env, pose=p)
        succ.append(tr["reached"]); steps.append(tr["steps"]); rets.append(tr["ret"])
        dists.append(tr["final_dist"]); far.append(tr["far_hits"])
        straight = np.hypot(p[0] - env.goal[0], p[1] - env.goal[1])
        if tr["reached"] and straight > 1e-6:
            opt.append(tr["path_len"] / straight)     # path efficiency (>=1)
        trajs.append(tr)
    return dict(success_rate=float(np.mean(succ)),
                mean_steps=float(np.mean(steps)),
                mean_return=float(np.mean(rets)),
                mean_final_dist=float(np.mean(dists)),
                wall_hit_rate=float(np.mean([f > 0 for f in far])),  # meaningful violations
                mean_path_ratio=float(np.mean(opt)) if opt else float("nan")), trajs
