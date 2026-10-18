# Trained models — Go1 single-agent go-to-goal

Task: drive one Go1 from any position to **(-2.0, -2.5) m**, speed-limited,
without leaving the map [-2.15,2.0] x [-2.7,2.4].
All models evaluated from the **same 200 random start positions** across the map.

## Ranking (success, then fewest wall-violations, then final distance)
| rank | model | algorithm | type | success | wall-viol | steps | final dist (m) | folder |
|------|-------|-----------|------|---------|-----------|-------|----------------|--------|
| 1 | **PPO** ✅ **BEST** | Proximal Policy Optimization | online | 1.00 | 0.00 | 21 | 0.22 | `ppo_online/` |
| 2 | **IQL** | Implicit Q-Learning | offline | 1.00 | 0.17 | 66 | 0.27 | `iql_offline/` |
| 3 | **TD3BC** | TD3 + Behavior Cloning | offline | 1.00 | 0.47 | 34 | 0.27 | `td3bc_offline/` |
| 4 | **CQL** | Conservative Q-Learning | offline | 1.00 | 0.94 | 66 | 0.26 | `cql_offline/` |

**Deployed model:** `PPO` — copied to [`best/`](best/model_card.md).
Each folder has the weights, `metrics.json`, and a detailed `model_card.md`.

## Load any model
```python
from src.policy_io import load_policy
act, meta = load_policy("models/best/PPO.pt")   # -> callable act(obs)->action
```
