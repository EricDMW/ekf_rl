# Model card — IQL (Implicit Q-Learning)

**Role:** offline baseline (best offline model)
**Type:** offline off-policy RL on the real transitions

## Task
Drive a **single** Unitree Go1 quadruped from **any** start position inside the
arena to the fixed goal **(-2.0, -2.5) m**, within 0.3 m, under a
maximum-speed limit, **without leaving the map**.

- Map (arena): x in [-2.15, 2.0], y in [-2.7, 2.4] m
- Max speed: |(vx,vy)| <= 1.5 m/s ; |omega| <= 2.5 rad/s
- Control rate: 10 Hz

## Performance (evaluated from 200 random start positions across the whole map)
| metric | value |
|--------|-------|
| success rate            | **1.000** |
| wall-violation rate     | **0.140** (contact away from the cornered goal) |
| mean steps to goal      | 66  (~6.6 s) |
| mean final distance (m) | 0.275 |
| path ratio (1=optimal)  | 1.39 |

## Observation (9-D, position-aware)
| idx | name | meaning |
|-----|------|---------|
| 0 | `x` | robot x position (m), world frame |
| 1 | `y` | robot y position (m), world frame |
| 2 | `cos_yaw` | cosine of heading |
| 3 | `sin_yaw` | sine of heading |
| 4 | `gx_body` | goal x in the robot body frame (m) |
| 5 | `gy_body` | goal y in the robot body frame (m) |
| 6 | `vx` | current body forward velocity (m/s) |
| 7 | `vy` | current body lateral velocity (m/s) |
| 8 | `omega` | current yaw rate (rad/s) |

## Action (3-D, body-frame twist)
| idx | name | meaning |
|-----|------|---------|
| 0 | `vx` | body forward velocity command, |v|<= 1.5 m/s (disc cap on (vx,vy)) |
| 1 | `vy` | body lateral velocity command |
| 2 | `omega` | yaw-rate command, |omega|<= 2.5 rad/s |

## Architecture
Actor: Gaussian MLP [9-256-256-3] (state-dependent std). Twin Q critics [12-256-256-1]. State-value MLP [9-256-256-1]. Obs normalised by fixed dataset mean/std.

## Training
75k gradient steps on 4,705 real transitions. Expectile 0.7 for the value, advantage-weighted regression (temperature 3.0) for the actor. Never queries out-of-distribution actions.

**Note:** Most stable offline method; reaches the goal but occasionally grazes walls.

## Dataset
23 real Go1 mocap runs -> 4,705 transitions at 10 Hz. Careful handling:
trajectories truncated at first goal entry; terminal only at genuine goal entry;
runs classified (17 REACH / 5 approach / 1 NON-REACH); actions = body-frame twist
clipped to the speed limit. (Offline models train on these transitions; PPO
trains online in the env whose dynamics/limits are taken from this data.)

## Files
- `IQL.pt` — checkpoint: `state_dict`, obs mean/std, action scale, goal, map
- `metrics.json` — evaluation summary + full training history

## Usage
```python
from src.policy_io import load_policy
from src.env import Go1GoalEnv, rollout
act, meta = load_policy("models/iql_offline/IQL.pt")
tr = rollout(act, Go1GoalEnv(), pose=(1.8, 2.0, 0.0))   # any (x, y, yaw)
print(tr["reached"], tr["steps"], tr["final_dist"], "wall_hits:", tr["far_hits"])
```
