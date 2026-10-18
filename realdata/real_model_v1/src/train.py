"""Train the offline algorithms (TD3+BC, IQL, CQL) on the real data and PPO
online in the data-grounded env, evaluate them all from RANDOM start positions
across the whole map, compare and save.

Run:  python -m src.train
"""
import os, json, time
import numpy as np
import torch

from . import config as C
from .data_pipeline import build_dataset
from .env import Go1GoalEnv, evaluate
from .offline_algos import OFFLINE_ALGOS
from .ppo import PPO


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def eval_poses(n=C.N_EVAL_STARTS, seed=12345):
    """Random start poses across the whole map (the 'from any position' test)."""
    rng = np.random.default_rng(seed)
    goal = np.array(C.GOAL)
    poses = []
    while len(poses) < n:
        x = rng.uniform(C.MAP_XMIN + 0.1, C.MAP_XMAX - 0.1)
        y = rng.uniform(C.MAP_YMIN + 0.1, C.MAP_YMAX - 0.1)
        if np.hypot(x - goal[0], y - goal[1]) > 3 * C.GOAL_RADIUS:
            poses.append((x, y, rng.uniform(-np.pi, np.pi)))
    return poses


class Buffer:
    def __init__(self, z, device):
        self.o = torch.as_tensor(z["obs"], dtype=torch.float32, device=device)
        self.a = torch.as_tensor(z["act"], dtype=torch.float32, device=device)
        self.r = torch.as_tensor(z["rew"], dtype=torch.float32, device=device)
        self.o2 = torch.as_tensor(z["nobs"], dtype=torch.float32, device=device)
        self.d = torch.as_tensor(z["term"], dtype=torch.float32, device=device)
        self.mean = torch.as_tensor(z["obs_mean"], dtype=torch.float32, device=device)
        self.std = torch.as_tensor(z["obs_std"], dtype=torch.float32, device=device)
        self.n = self.o.shape[0]; self.device = device

    def sample(self, bs):
        i = torch.randint(0, self.n, (bs,), device=self.device)
        nz = lambda x: (x - self.mean) / self.std
        return nz(self.o[i]), self.a[i], self.r[i], nz(self.o2[i]), self.d[i]


A_SCALE = [C.V_MAX, C.V_MAX, C.W_MAX]


def better(a, b):
    """Rank key: higher success, then fewer wall hits, then closer final dist."""
    return (a["success_rate"], -a["wall_hit_rate"], -a["mean_final_dist"]) > \
           (b["success_rate"], -b["wall_hit_rate"], -b["mean_final_dist"])


def train_offline(name, z, buf, device, poses, log):
    set_seed(C.SEED)
    algo = OFFLINE_ALGOS[name](C.OBS_DIM, C.ACT_DIM, A_SCALE,
                               z["obs_mean"], z["obs_std"], device)
    env = Go1GoalEnv(rng=np.random.default_rng(C.SEED))
    hist, best, best_sd = [], None, None
    t0 = time.time()
    for step in range(1, C.N_STEPS + 1):
        losses = algo.update(buf.sample(C.BATCH_SIZE))
        if step % C.EVAL_EVERY == 0 or step == 1:
            summ, _ = evaluate(lambda o: algo.act(o, True), env, poses)
            summ["step"] = step; summ.update({k: round(v, 4) for k, v in losses.items()})
            hist.append(summ)
            if best is None or better(summ, best):
                best = summ
                best_sd = {k: {kk: vv.detach().cpu().clone() for kk, vv in sd.items()}
                           for k, sd in algo.state_dict().items()}
            log(f"[{name}] {step:>7d} | succ {summ['success_rate']:.2f} "
                f"| final_dist {summ['mean_final_dist']:.3f} "
                f"| wall {summ['wall_hit_rate']:.2f} | steps {summ['mean_steps']:.0f}")
    log(f"[{name}] done {time.time()-t0:.0f}s | BEST succ {best['success_rate']:.2f} "
        f"wall {best['wall_hit_rate']:.2f} dist {best['mean_final_dist']:.3f}")
    if best_sd is not None:
        algo.load_state_dict(best_sd)
    save_policy(name, algo.state_dict(), z, best, hist, extra={})
    return dict(name=name, best=best, history=hist)


def _clone_sd(sd):
    """Deep-copy a possibly-nested state dict of tensors / numpy arrays."""
    out = {}
    for k, v in sd.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.detach().cpu().clone() for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = np.copy(v)
    return out


def train_ppo(z, device, poses, log):
    set_seed(C.SEED)
    ppo = PPO(C.OBS_DIM, C.ACT_DIM, A_SCALE, device)
    eval_env = Go1GoalEnv(rng=np.random.default_rng(999))
    state = dict(best=None, best_sd=None)

    def cb(agent):
        summ, _ = evaluate(lambda o: agent.act(o, True), eval_env, poses)
        if state["best"] is None or better(summ, state["best"]):
            state["best"] = summ
            state["best_sd"] = _clone_sd(agent.state_dict())
        return summ

    t0 = time.time()
    hist = ppo.train(C.PPO_TOTAL_STEPS, log=log, eval_cb=cb)
    log(f"[PPO] done {time.time()-t0:.0f}s | BEST succ {state['best']['success_rate']:.2f} "
        f"wall {state['best']['wall_hit_rate']:.2f} dist {state['best']['mean_final_dist']:.3f}")
    if state["best_sd"] is not None:
        ppo.load_state_dict(state["best_sd"])
    save_policy("PPO", ppo.state_dict(), z, state["best"], hist,
                extra=dict(norm_mean=ppo.norm.mean.tolist(), norm_var=ppo.norm.var.tolist()))
    return dict(name="PPO", best=state["best"], history=hist)


def save_policy(name, sd, z, best, history, extra):
    # numpy-ify tensors for portability
    def cpu(d):
        return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
    sd_cpu = {k: cpu(v) if isinstance(v, dict) else (v.cpu() if torch.is_tensor(v) else v)
              for k, v in sd.items()}
    path = os.path.join(C.POLICY_DIR, f"{name}.pt")
    torch.save(dict(algo=name, state_dict=sd_cpu,
                    obs_mean=np.asarray(z["obs_mean"]), obs_std=np.asarray(z["obs_std"]),
                    a_scale=A_SCALE, obs_dim=C.OBS_DIM, act_dim=C.ACT_DIM,
                    goal=list(C.GOAL), goal_radius=C.GOAL_RADIUS,
                    v_max=C.V_MAX, w_max=C.W_MAX,
                    map=[C.MAP_XMIN, C.MAP_XMAX, C.MAP_YMIN, C.MAP_YMAX],
                    best=best, history=history, **extra), path)
    json.dump(dict(best=best, history=history),
              open(os.path.join(C.POLICY_DIR, f"{name}_metrics.json"), "w"), indent=2)
    print(f"[save] {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    z, meta = build_dataset(force=False)
    buf = Buffer(z, device)
    poses_train = eval_poses(50, seed=777)       # lighter set for checkpoint selection
    poses_final = eval_poses(C.N_EVAL_STARTS)    # full set for the comparison table
    print(f"eval poses: {len(poses_train)} (train) / {len(poses_final)} (final)")

    logf = open(os.path.join(C.POLICY_DIR, "train_log.txt"), "w")
    def log(m): print(m); logf.write(m + "\n"); logf.flush()

    results = []
    for name in ["TD3BC", "IQL", "CQL"]:
        results.append(train_offline(name, z, buf, device, poses_train, log))
    results.append(train_ppo(z, device, poses_train, log))

    # final full-resolution evaluation of every saved best checkpoint
    log("\n---- final evaluation on 200 random map positions ----")
    from .policy_io import load_policy
    env = Go1GoalEnv(rng=np.random.default_rng(2024))
    for r in results:
        act, _ = load_policy(os.path.join(C.POLICY_DIR, f"{r['name']}.pt"),
                             device=str(device))
        summ, _ = evaluate(act, env, poses_final)
        r["best"] = summ
        log(f"  {r['name']:6s} succ={summ['success_rate']:.3f} wall={summ['wall_hit_rate']:.3f} "
            f"dist={summ['mean_final_dist']:.3f} steps={summ['mean_steps']:.0f}")

    results.sort(key=lambda r: (r["best"]["success_rate"], -r["best"]["wall_hit_rate"],
                                -r["best"]["mean_final_dist"]), reverse=True)
    winner = results[0]["name"]
    comp = {r["name"]: r["best"] for r in results}
    comp["winner"] = winner
    json.dump(comp, open(os.path.join(C.POLICY_DIR, "comparison.json"), "w"), indent=2)
    import shutil
    shutil.copy(os.path.join(C.POLICY_DIR, f"{winner}.pt"),
                os.path.join(C.POLICY_DIR, "best_policy.pt"))
    log("\n==== COMPARISON (best checkpoint; eval from random map positions) ====")
    for r in results:
        b = r["best"]
        log(f"  {r['name']:6s} succ={b['success_rate']:.3f} wall={b['wall_hit_rate']:.3f} "
            f"final_dist={b['mean_final_dist']:.3f} steps={b['mean_steps']:.0f} "
            f"path_ratio={b.get('mean_path_ratio', float('nan')):.2f}")
    log(f"WINNER: {winner} -> policies/best_policy.pt")

    # write the labeled models/ folder (weights + metrics + model cards)
    from .export_models import export
    export()
    logf.close()


if __name__ == "__main__":
    main()
