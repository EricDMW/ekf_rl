"""Figures and animations for the single-agent Go1 go-to-goal project.

figures/  learning_curves, comparison_bar, dataset_classified, rollouts_<best>
animations/ best_policy_anypos.gif  (dog to goal from many random starts),
            algo_comparison.gif      (4 methods from one hard start)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation

from . import config as C
from .data_pipeline import build_dataset
from .env import Go1GoalEnv, rollout
from .policy_io import load_policy

GOAL = np.array(C.GOAL)
BOUNDS = (C.MAP_XMIN, C.MAP_XMAX, C.MAP_YMIN, C.MAP_YMAX)
COLORS = {"TD3BC": "#E45756", "IQL": "#4C78A8", "CQL": "#54A24B", "PPO": "#B279A2"}
ORDER = ["TD3BC", "IQL", "CQL", "PPO"]


def _arena(ax):
    xmin, xmax, ymin, ymax = BOUNDS
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           fill=False, ec="k", lw=1.5, ls="-"))
    ax.add_patch(Circle(GOAL, C.GOAL_RADIUS, color="gold", alpha=.4, zorder=1))
    ax.plot(*GOAL, "*", color="#B8860B", ms=18, zorder=5, label="goal (-2,-2.5)")
    ax.set_xlim(xmin - 0.25, xmax + 0.25); ax.set_ylim(ymin - 0.25, ymax + 0.25)
    ax.set_aspect("equal"); ax.grid(alpha=.2)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def fig_learning_curves():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name in ORDER:
        p = os.path.join(C.POLICY_DIR, f"{name}_metrics.json")
        if not os.path.exists(p):
            continue
        h = json.load(open(p))["history"]
        xs = [d["step"] for d in h]
        ax[0].plot(xs, [d["success_rate"] for d in h], "-o", ms=2.5, color=COLORS[name], label=name)
        ax[1].plot(xs, [d["mean_final_dist"] for d in h], "-o", ms=2.5, color=COLORS[name], label=name)
        ax[2].plot(xs, [d["wall_hit_rate"] for d in h], "-o", ms=2.5, color=COLORS[name], label=name)
    ax[0].set_title("Success rate (from random map positions)"); ax[0].set_ylim(-.03, 1.03)
    ax[1].set_title("Mean final distance (m)"); ax[1].set_yscale("log")
    ax[1].axhline(C.GOAL_RADIUS, ls="--", c="k", lw=1)
    ax[2].set_title("Wall-violation rate (away from goal)"); ax[2].set_ylim(-.03, 1.03)
    for a in ax:
        a.set_xlabel("training step"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle("Learning curves — offline (TD3BC/IQL/CQL) vs online PPO", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "learning_curves.png"), dpi=140); plt.close(fig)


def fig_comparison_bar():
    comp = json.load(open(os.path.join(C.POLICY_DIR, "comparison.json")))
    names = [n for n in ORDER if n in comp]
    metrics = [("success_rate", "Success rate", "frac"),
               ("wall_hit_rate", "Wall-violation rate", "frac"),
               ("mean_final_dist", "Final distance (m)", "m")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, (k, title, yl) in zip(ax, metrics):
        vals = [comp[n][k] for n in names]
        a.bar(names, vals, color=[COLORS[n] for n in names])
        a.set_title(title); a.set_ylabel(yl); a.grid(alpha=.3, axis="y")
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Final comparison (winner = {comp['winner']})", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "comparison_bar.png"), dpi=140); plt.close(fig)


def fig_dataset_classified(z):
    fig, ax = plt.subplots(figsize=(6.5, 7))
    _arena(ax)
    cmap = {"REACH": "#4C78A8", "approach": "#F58518", "NON-REACH": "#E45756"}
    seen = set()
    for tr in z["traj"]:
        c = cmap[tr["cls"]]
        lab = tr["cls"] if tr["cls"] not in seen else None
        seen.add(tr["cls"])
        ax.plot(tr["x"], tr["y"], "-", color=c, alpha=.6, lw=1.1, label=lab)
        ax.plot(tr["x"][0], tr["y"][0], "o", color=c, ms=4)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Dataset trajectories by class\n(truncated at goal entry)")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "dataset_classified.png"), dpi=140); plt.close(fig)


def fig_rollouts(name, poses):
    act, _ = load_policy(os.path.join(C.POLICY_DIR, f"{name}.pt"))
    env = Go1GoalEnv(rng=np.random.default_rng(7))
    fig, ax = plt.subplots(figsize=(6.5, 7))
    _arena(ax)
    reached = 0
    for p in poses:
        tr = rollout(act, env, pose=p)
        reached += tr["reached"]
        col = COLORS[name] if tr["reached"] else "#bbbbbb"
        ax.plot(tr["x"], tr["y"], "-", color=col, alpha=.8, lw=1.0)
        ax.plot(tr["x"][0], tr["y"][0], ".", color="k", ms=3)
    ax.set_title(f"{name} rollouts from random positions  ({reached}/{len(poses)} reach)")
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, f"rollouts_{name}.png"), dpi=140); plt.close(fig)


# ---------------- animations ----------------
def _robot(ax, color):
    body = Circle((0, 0), 0.11, color=color, zorder=6); ax.add_patch(body)
    head = ax.plot([], [], "-", color="k", lw=2, zorder=7)[0]
    return body, head


def _place(body, head, x, y, yaw):
    body.center = (x, y)
    head.set_data([x, x + 0.26*np.cos(yaw)], [y, y + 0.26*np.sin(yaw)])


def _pick_starts(poses, n):
    """Pick n well-spread start positions (farthest-point sampling)."""
    P = np.array([[p[0], p[1]] for p in poses])
    idx = [int(np.argmax(np.hypot(P[:, 0] - GOAL[0], P[:, 1] - GOAL[1])))]
    while len(idx) < min(n, len(poses)):
        d = np.min([np.hypot(P[:, 0] - P[j, 0], P[:, 1] - P[j, 1]) for j in idx], axis=0)
        idx.append(int(np.argmax(d)))
    return [poses[i] for i in idx]


def anim_single_agent(name, poses, fname="single_agent_multistart.gif", n=6, hold=8):
    """ONE robot, driving to the goal from several different start positions in
    sequence (one episode after another). Only a single agent is ever on screen;
    finished paths stay faded so the coverage from different starts is visible."""
    act, _ = load_policy(os.path.join(C.POLICY_DIR, f"{name}.pt"))
    env = Go1GoalEnv(rng=np.random.default_rng(3))
    starts = _pick_starts(poses, n)
    trajs = [rollout(act, env, pose=p) for p in starts]

    fig, ax = plt.subplots(figsize=(6.5, 7)); _arena(ax)
    past = ax.plot([], [], "-", color="#bbbbbb", lw=1.0, alpha=.7, zorder=2)[0]  # finished paths
    cur = ax.plot([], [], "-", color="#4C78A8", lw=2.0, zorder=4)[0]             # active path
    start_mark = ax.plot([], [], "ks", ms=7, zorder=5)[0]
    body, head = _robot(ax, "#4C78A8")
    title = ax.set_title("")
    txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=10)

    # build a flat frame list: (episode_idx, step_idx) with a hold at each goal
    frames = []
    for ei, tr in enumerate(trajs):
        step = 2
        for i in range(0, len(tr["x"]), step):
            frames.append((ei, i))
        frames += [(ei, len(tr["x"]) - 1)] * hold          # pause on the goal

    past_x, past_y = [], []

    def upd(fr):
        ei, i = frames[fr]
        tr = trajs[ei]
        # accumulate finished episodes into the faded layer
        if ei > 0:
            fx, fy = [], []
            for pe in range(ei):
                fx += list(trajs[pe]["x"]) + [np.nan]
                fy += list(trajs[pe]["y"]) + [np.nan]
            past.set_data(fx, fy)
        cur.set_data(tr["x"][:i + 1], tr["y"][:i + 1])
        start_mark.set_data([tr["x"][0]], [tr["y"][0]])
        _place(body, head, tr["x"][i], tr["y"][i], tr["yaw"][i])
        d = np.hypot(tr["x"][i] - GOAL[0], tr["y"][i] - GOAL[1])
        title.set_text(f"{name}: single Go1, start {ei+1}/{len(trajs)}  "
                       f"= ({tr['x'][0]:+.1f}, {tr['y'][0]:+.1f})")
        txt.set_text(f"t = {i*C.CTRL_DT:4.1f}s   dist to goal = {d:.2f} m"
                     + ("   REACHED" if d < C.GOAL_RADIUS else ""))
        return [past, cur, body, head]

    ani = animation.FuncAnimation(fig, upd, frames=len(frames), interval=60, blit=False)
    out = os.path.join(C.ANIM_DIR, fname); ani.save(out, writer=animation.PillowWriter(fps=20))
    plt.close(fig); print("[anim]", out)


def main():
    from .train import eval_poses
    z, _ = build_dataset()
    comp = json.load(open(os.path.join(C.POLICY_DIR, "comparison.json")))
    winner = comp["winner"]
    poses = eval_poses(60, seed=4242)
    print("figures ...")
    fig_learning_curves(); fig_comparison_bar(); fig_dataset_classified(z)
    fig_rollouts(winner, poses)
    if winner != "PPO" and os.path.exists(os.path.join(C.POLICY_DIR, "PPO.pt")):
        fig_rollouts("PPO", poses)
    print("animations ...")
    anim_single_agent(winner, poses)          # single agent, different start positions
    print("done.")


if __name__ == "__main__":
    main()
