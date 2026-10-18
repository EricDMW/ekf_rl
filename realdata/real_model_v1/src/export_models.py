"""Export the trained policies into a detailed, well-labeled `models/` folder.

For each trained model we create a subfolder containing:
  * the weights (.pt),
  * metrics.json (evaluation on 200 random map positions + training history),
  * model_card.md (full, human-readable label: task, I/O spec, architecture,
    training config, dataset, performance, usage).
The best model is additionally copied into `models/best/`.
A top-level `models/MODELS.md` indexes everything with a comparison table.
"""
import os, json, shutil
import numpy as np

from . import config as C

MODELS_DIR = os.path.join(C.ROOT, "models")

# per-algorithm descriptive metadata
ALGO_INFO = {
    "PPO": dict(
        full="Proximal Policy Optimization",
        kind="online (on-policy) RL in the data-grounded simulator",
        role="PRIMARY / deployed policy",
        arch=("Actor: Gaussian MLP [9-256-256-3] with global log-std, tanh-squashed, "
              "scaled to the twist bounds. Critic: Value MLP [9-256-256-1]. "
              "Observation RunningMeanStd normaliser (stats stored in checkpoint)."),
        train=("Trained ONLINE in Go1GoalEnv for 1.0M env steps; episodes reset to "
               "random poses across the whole map so it directly optimises the "
               "'reach from any position, stay in-map, speed-limited' objective. "
               "PPO clip 0.2, GAE lambda 0.95, 10 epochs/rollout, rollout 4096."),
        note=("Sees the boundary penalty during training, so it learns to stay off "
              "the walls -> zero wall-violations at evaluation.")),
    "IQL": dict(
        full="Implicit Q-Learning",
        kind="offline off-policy RL on the real transitions",
        role="offline baseline (best offline model)",
        arch=("Actor: Gaussian MLP [9-256-256-3] (state-dependent std). "
              "Twin Q critics [12-256-256-1]. State-value MLP [9-256-256-1]. "
              "Obs normalised by fixed dataset mean/std."),
        train=("75k gradient steps on 4,705 real transitions. Expectile 0.7 for the "
               "value, advantage-weighted regression (temperature 3.0) for the actor. "
               "Never queries out-of-distribution actions."),
        note="Most stable offline method; reaches the goal but occasionally grazes walls."),
    "TD3BC": dict(
        full="TD3 + Behavior Cloning",
        kind="offline off-policy RL on the real transitions",
        role="offline baseline",
        arch=("Actor: deterministic MLP [9-256-256-3], tanh-scaled to twist bounds. "
              "Twin Q critics [12-256-256-1]. Obs normalised by dataset mean/std."),
        train=("75k gradient steps on 4,705 real transitions. TD3 with target-policy "
               "smoothing plus a normalised behavior-cloning term (alpha 2.5)."),
        note="Fast, direct paths, but training diverges after its early peak (best checkpoint kept)."),
    "CQL": dict(
        full="Conservative Q-Learning",
        kind="offline off-policy RL on the real transitions",
        role="offline baseline",
        arch=("Actor: Gaussian MLP [9-256-256-3] (SAC-style). Twin Q critics "
              "[12-256-256-1] with a conservative log-sum-exp penalty over random & "
              "policy actions (alpha 1.0). Obs normalised by dataset mean/std."),
        train="75k gradient steps on 4,705 real transitions; SAC backbone + CQL penalty.",
        note="Reaches the goal but hugs walls heavily and is the least stable here."),
}

OBS_SPEC = [
    ("x", "robot x position (m), world frame"),
    ("y", "robot y position (m), world frame"),
    ("cos_yaw", "cosine of heading"),
    ("sin_yaw", "sine of heading"),
    ("gx_body", "goal x in the robot body frame (m)"),
    ("gy_body", "goal y in the robot body frame (m)"),
    ("vx", "current body forward velocity (m/s)"),
    ("vy", "current body lateral velocity (m/s)"),
    ("omega", "current yaw rate (rad/s)"),
]
ACT_SPEC = [
    ("vx", f"body forward velocity command, |v|<= {C.V_MAX} m/s (disc cap on (vx,vy))"),
    ("vy", "body lateral velocity command"),
    ("omega", f"yaw-rate command, |omega|<= {C.W_MAX} rad/s"),
]


def _card(name, metrics, comp, folder_role):
    info = ALGO_INFO[name]
    b = metrics["best"]
    obs_rows = "\n".join(f"| {i} | `{n}` | {d} |" for i, (n, d) in enumerate(OBS_SPEC))
    act_rows = "\n".join(f"| {i} | `{n}` | {d} |" for i, (n, d) in enumerate(ACT_SPEC))
    is_best = (comp["winner"] == name)
    md = f"""# Model card — {name} ({info['full']})

**Role:** {info['role']}{"  ✅ **BEST / deployed as `best/`**" if is_best else ""}
**Type:** {info['kind']}

## Task
Drive a **single** Unitree Go1 quadruped from **any** start position inside the
arena to the fixed goal **{tuple(C.GOAL)} m**, within {C.GOAL_RADIUS} m, under a
maximum-speed limit, **without leaving the map**.

- Map (arena): x in [{C.MAP_XMIN}, {C.MAP_XMAX}], y in [{C.MAP_YMIN}, {C.MAP_YMAX}] m
- Max speed: |(vx,vy)| <= {C.V_MAX} m/s ; |omega| <= {C.W_MAX} rad/s
- Control rate: {int(1/C.CTRL_DT)} Hz

## Performance (evaluated from 200 random start positions across the whole map)
| metric | value |
|--------|-------|
| success rate            | **{b['success_rate']:.3f}** |
| wall-violation rate     | **{b['wall_hit_rate']:.3f}** (contact away from the cornered goal) |
| mean steps to goal      | {b['mean_steps']:.0f}  (~{b['mean_steps']*C.CTRL_DT:.1f} s) |
| mean final distance (m) | {b['mean_final_dist']:.3f} |
| path ratio (1=optimal)  | {b.get('mean_path_ratio', float('nan')):.2f} |

## Observation (9-D, position-aware)
| idx | name | meaning |
|-----|------|---------|
{obs_rows}

## Action (3-D, body-frame twist)
| idx | name | meaning |
|-----|------|---------|
{act_rows}

## Architecture
{info['arch']}

## Training
{info['train']}

**Note:** {info['note']}

## Dataset
23 real Go1 mocap runs -> 4,705 transitions at 10 Hz. Careful handling:
trajectories truncated at first goal entry; terminal only at genuine goal entry;
runs classified (17 REACH / 5 approach / 1 NON-REACH); actions = body-frame twist
clipped to the speed limit. (Offline models train on these transitions; PPO
trains online in the env whose dynamics/limits are taken from this data.)

## Files
- `{name}.pt` — checkpoint: `state_dict`, obs mean/std, action scale, goal, map{', obs RunningNorm stats' if name=='PPO' else ''}
- `metrics.json` — evaluation summary + full training history

## Usage
```python
from src.policy_io import load_policy
from src.env import Go1GoalEnv, rollout
act, meta = load_policy("models/{folder_role}/{name}.pt")
tr = rollout(act, Go1GoalEnv(), pose=(1.8, 2.0, 0.0))   # any (x, y, yaw)
print(tr["reached"], tr["steps"], tr["final_dist"], "wall_hits:", tr["far_hits"])
```
"""
    return md


def export():
    comp = json.load(open(os.path.join(C.POLICY_DIR, "comparison.json")))
    if os.path.exists(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR)

    # rank by (success, -wall, -dist) for the index
    order = sorted(ALGO_INFO.keys(),
                   key=lambda n: (comp[n]["success_rate"], -comp[n]["wall_hit_rate"],
                                  -comp[n]["mean_final_dist"]), reverse=True)
    role_dir = {"PPO": "ppo_online", "IQL": "iql_offline",
                "TD3BC": "td3bc_offline", "CQL": "cql_offline"}

    for name in order:
        folder = role_dir[name]
        d = os.path.join(MODELS_DIR, folder)
        os.makedirs(d)
        shutil.copy(os.path.join(C.POLICY_DIR, f"{name}.pt"), os.path.join(d, f"{name}.pt"))
        metrics = json.load(open(os.path.join(C.POLICY_DIR, f"{name}_metrics.json")))
        json.dump(metrics, open(os.path.join(d, "metrics.json"), "w"), indent=2)
        open(os.path.join(d, "model_card.md"), "w").write(_card(name, metrics, comp, folder))

    # best/
    best = comp["winner"]
    bd = os.path.join(MODELS_DIR, "best")
    os.makedirs(bd)
    shutil.copy(os.path.join(C.POLICY_DIR, f"{best}.pt"), os.path.join(bd, f"{best}.pt"))
    metrics = json.load(open(os.path.join(C.POLICY_DIR, f"{best}_metrics.json")))
    json.dump(metrics, open(os.path.join(bd, "metrics.json"), "w"), indent=2)
    open(os.path.join(bd, "model_card.md"), "w").write(_card(best, metrics, comp, "best"))

    # index
    rows = []
    for rank, name in enumerate(order, 1):
        b = comp[name]
        tag = " ✅ **BEST**" if name == best else ""
        rows.append(f"| {rank} | **{name}**{tag} | {ALGO_INFO[name]['full']} | "
                    f"{ALGO_INFO[name]['kind'].split(' ')[0]} | {b['success_rate']:.2f} | "
                    f"{b['wall_hit_rate']:.2f} | {b['mean_steps']:.0f} | "
                    f"{b['mean_final_dist']:.2f} | `{role_dir[name]}/` |")
    index = f"""# Trained models — Go1 single-agent go-to-goal

Task: drive one Go1 from any position to **{tuple(C.GOAL)} m**, speed-limited,
without leaving the map [{C.MAP_XMIN},{C.MAP_XMAX}] x [{C.MAP_YMIN},{C.MAP_YMAX}].
All models evaluated from the **same 200 random start positions** across the map.

## Ranking (success, then fewest wall-violations, then final distance)
| rank | model | algorithm | type | success | wall-viol | steps | final dist (m) | folder |
|------|-------|-----------|------|---------|-----------|-------|----------------|--------|
{chr(10).join(rows)}

**Deployed model:** `{best}` — copied to [`best/`](best/model_card.md).
Each folder has the weights, `metrics.json`, and a detailed `model_card.md`.

## Load any model
```python
from src.policy_io import load_policy
act, meta = load_policy("models/best/{best}.pt")   # -> callable act(obs)->action
```
"""
    open(os.path.join(MODELS_DIR, "MODELS.md"), "w").write(index)
    print(f"[models] wrote {MODELS_DIR}")
    for root, _, files in os.walk(MODELS_DIR):
        for f in sorted(files):
            print("  ", os.path.relpath(os.path.join(root, f), C.ROOT))


if __name__ == "__main__":
    export()
