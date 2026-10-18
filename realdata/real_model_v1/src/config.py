"""Global configuration for the single-agent Go1 go-to-goal project.

Task: drive ONE Go1 quadruped from ANY start position inside the arena to the
fixed target g = (-2.0, -2.5) m, under a maximum-speed limit, without leaving
the map.  Everything is grounded in the real mocap data.
"""
import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REALDATA  = os.path.abspath(os.path.join(ROOT, ".."))
DATA_DIR  = os.path.join(REALDATA, "data")
CACHE_DIR  = os.path.join(ROOT, "cache")
POLICY_DIR = os.path.join(ROOT, "policies")
FIG_DIR    = os.path.join(ROOT, "figures")
ANIM_DIR   = os.path.join(ROOT, "animations")
REPORT_DIR = os.path.join(ROOT, "report")
for d in (CACHE_DIR, POLICY_DIR, FIG_DIR, ANIM_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------------
# Task / world definition  (single agent)
# ----------------------------------------------------------------------------
GOAL         = (-2.0, -2.5)      # fixed target in the mocap world frame (m)
GOAL_RADIUS  = 0.30              # success threshold (m)
CTRL_DT      = 0.10             # control timestep (s) -> 10 Hz policy
MOCAP_HZ     = 120.0

# Map (arena) rectangle.  Derived from the data extent (x in [-1.94,1.84],
# y in [-2.54,2.26]) expanded by a small margin and guaranteed to contain the
# goal corner.  The policy must keep the robot inside this rectangle.
MAP_XMIN, MAP_XMAX = -2.15, 2.00
MAP_YMIN, MAP_YMAX = -2.70, 2.40

# Kinematic limits (max-speed-limited running).  From data percentiles.
V_MAX     = 1.5                 # max body linear speed |(vx,vy)| (m/s)
W_MAX     = 2.5                 # max yaw rate |omega| (rad/s)

# Action = body-frame velocity twist [vx, vy, omega]; obs is position-aware.
ACT_DIM = 3
OBS_DIM = 9   # [x, y, cos_yaw, sin_yaw, gx_body, gy_body, vx, vy, omega]

# ----------------------------------------------------------------------------
# Reward weights (designed)
# ----------------------------------------------------------------------------
R_PROGRESS = 5.0     # reward per metre of progress toward the goal
R_CTRL     = 0.01    # penalty on squared control effort
R_TIME     = 0.02    # per-step time penalty (reach quickly)
R_GOAL     = 20.0    # one-time bonus for entering the goal radius
R_BOUND    = 5.0     # penalty for pushing against / trying to cross a wall

# ----------------------------------------------------------------------------
# Data handling
# ----------------------------------------------------------------------------
# Trajectories are truncated at first goal entry (post-goal drift removed).
# A run is "non-reaching" if it never comes within this distance of the goal.
NONREACH_MIN_DIST = 0.60
# Keep non-reaching runs in the offline buffer (value-based RL handles
# suboptimal data via correct reward labels); set False to drop them.
KEEP_NONREACHING  = True

# ----------------------------------------------------------------------------
# Shared training hyper-parameters
# ----------------------------------------------------------------------------
SEED       = 0
HIDDEN     = 256
GAMMA      = 0.99
TAU        = 0.005
LR_ACTOR   = 3e-4
LR_CRITIC  = 3e-4
BATCH_SIZE = 256
N_STEPS    = 75_000            # gradient steps per offline algorithm
EVAL_EVERY = 5_000
MAX_EP_STEPS = 400            # 40 s at 10 Hz

# Offline-algorithm specifics
TD3BC_ALPHA      = 2.5
TD3_POLICY_NOISE = 0.2
TD3_NOISE_CLIP   = 0.5
TD3_POLICY_FREQ  = 2
IQL_EXPECTILE    = 0.7
IQL_BETA         = 3.0
IQL_MAX_WEIGHT   = 100.0
CQL_ALPHA        = 1.0
CQL_N_ACTIONS    = 10

# PPO (online, in the data-grounded env)
PPO_TOTAL_STEPS   = 1_000_000
PPO_ROLLOUT       = 4096
PPO_EPOCHS        = 10
PPO_MINIBATCH     = 512
PPO_CLIP          = 0.2
PPO_LR            = 3e-4
PPO_GAE_LAMBDA    = 0.95
PPO_ENT_COEF      = 0.005
PPO_VF_COEF       = 0.5
PPO_MAX_GRAD_NORM = 0.5

# Evaluation from ANY position
N_EVAL_STARTS = 200          # random start poses across the whole map
