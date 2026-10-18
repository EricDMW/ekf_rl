# Single-Agent Go1 Go-to-Goal — Offline RL + PPO

Drive **one** Unitree Go1 from **any** position inside the arena to the fixed
target **(-2.0, -2.5) m**, under a **maximum-speed limit**, **without leaving the
map**. Grounded in the real mocap data in `../data/`.

## Results (200 random start positions across the map)

| Method | Success | Wall-violation | Steps | Path ratio | Type |
|--------|:-------:|:--------------:|:-----:|:----------:|------|
| **PPO** (winner) | **1.00** | **0.00** | 21 | 0.93 | online in sim |
| IQL    | 1.00 | 0.17 | 66 | 1.34 | offline |
| TD3+BC | 1.00 | 0.47 | 34 | 1.01 | offline |
| CQL    | 1.00 | 0.94 | 66 | 2.41 | offline |

All reach the goal from any position; **PPO** additionally never touches the map
boundary and is near time-optimal. Winner saved to `policies/best_policy.pt`.
Offline methods never see boundary violations in the real data, so they reach
the goal but graze the walls — hence PPO (online, sees the boundary penalty) is
the right tool for the "stay in map" constraint.

## Careful data handling (single agent)
- Resample each run to 10 Hz; yaw from quaternion.
- **Classify** each run: 17 REACH / 5 approach / **1 NON-REACH** (the wandering
  run `20260624_153602`, which never approaches the goal — never treated as a
  success).
- **Truncate at first goal entry** (drop post-goal drift).
- **Terminal flag TRUE only at a genuine goal entry** (a run that merely stops is
  a truncation, not a terminal — treating it as terminal corrupts value learning).
- Extract action = body-frame twist, **clipped to the speed limit**.
- Result: 4,705 transitions, 17 true terminals.

## MDP
- **Obs (9-D, position-aware):** `[x, y, cos yaw, sin yaw, gx_body, gy_body, vx, vy, omega]`
  (absolute `x,y` lets the policy reason about walls).
- **Action:** body twist `(vx, vy, omega)`, capped at `|v|<=1.5 m/s`, `|omega|<=2.5 rad/s`.
- **Reward:** `5*progress - 0.01*|a|^2 - 0.02 + 20*[reached] - 5*[hit wall]`.
- **Env:** kinematic integrator (inverse of action extraction); robot hard-clamped
  inside the map `[-2.15,2.00]x[-2.70,2.40]`; random-position resets.

## Trained models (labeled `models/` folder)
```
models/
  MODELS.md               # index + ranked comparison table + how to load
  best/                   # the deployed winner (PPO): weights + metrics + model_card.md
  ppo_online/             # each folder: <ALGO>.pt, metrics.json, model_card.md
  iql_offline/
  td3bc_offline/
  cql_offline/
```
Each `model_card.md` fully labels the model: task, observation/action spec,
architecture, training config, dataset, and evaluation metrics.

## Run
```bash
PY=/home/dongmingwang/.conda/envs/marl_env/bin/python
$PY -m src.data_pipeline    # build cache/offline_dataset.npz
$PY -m src.train            # train TD3BC, IQL, CQL (offline) + PPO (online); save best
$PY -m src.export_models    # write the labeled models/ folder
$PY -m src.visualize        # figures/ + single-agent animation
$PY -m src.image_analysis   # onboard-camera analysis
cd report && pdflatex report.tex && pdflatex report.tex   # -> report/report.pdf
```

## Layout
```
src/  config, data_pipeline, env, networks, offline_algos, ppo, train,
      policy_io, visualize, image_analysis, export_models
models/     labeled model folder (see above)   <-- deliverable
policies/   raw training checkpoints + comparison.json + train_log.txt
figures/    learning_curves, comparison_bar, dataset_classified, rollouts_*, img_*
animations/ single_agent_multistart.gif   (ONE robot, sequential start positions)
report/     report.tex -> report.pdf
```

## Use the trained model
```python
from src.policy_io import load_policy
from src.env import Go1GoalEnv, rollout
act, meta = load_policy("models/best/PPO.pt")          # deployed winner
tr = rollout(act, Go1GoalEnv(), pose=(1.8, 2.0, 0.0))  # any (x, y, yaw)
print(tr["reached"], tr["steps"], tr["final_dist"], "wall_hits:", tr["far_hits"])
```

## Extension (report §6)
15,263 unused onboard fisheye frames are rich in structure (obstacle boxes,
doorway, floor). Proposed: vision-conditioned goal-reaching, obstacle-aware
navigation, visual-inertial estimation, self-supervised visual pretraining.
