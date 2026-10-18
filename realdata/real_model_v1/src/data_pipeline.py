"""Build the offline transition dataset from the real Go1 mocap runs.

Careful data handling (single agent):
  * resample each run's pose to a uniform 10 Hz grid,
  * derive the action = body-frame twist (vx, vy, omega) by finite differencing,
    then CLIP it to the admissible max-speed set (denoises mocap spikes),
  * TRUNCATE each trajectory at the first goal entry (drop post-goal drift),
  * mark the terminal flag TRUE only at a real goal entry — a run that merely
    stops without reaching the goal is a *truncation*, not a terminal state
    (this keeps value bootstrapping correct),
  * classify every run as REACH / non-reaching and (optionally) keep the
    non-reaching ones for coverage, always with correct reward labels.

Observation (9-D): [x, y, cos_yaw, sin_yaw, gx_body, gy_body, vx, vy, omega]
"""
import os, glob, json
import numpy as np
import pandas as pd

from . import config as C
from .env import clip_speed, GOAL


def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_run_pose(mocap_csv):
    d = pd.read_csv(mocap_csv)
    if "rigid_body_id" in d.columns:
        rid = d["rigid_body_id"].mode().iloc[0]
        d = d[d["rigid_body_id"] == rid]
    d = d.sort_values("timestamp_ns").reset_index(drop=True)
    if len(d) < 20:
        return None
    t = d["timestamp_ns"].values.astype(np.float64) / 1e9
    t -= t[0]
    x, y = d["x"].values.astype(np.float64), d["y"].values.astype(np.float64)
    yaw = np.unwrap(quat_to_yaw(d["qx"].values, d["qy"].values,
                                d["qz"].values, d["qw"].values))
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, x, y, yaw = t[keep], x[keep], y[keep], yaw[keep]
    if len(t) < 20:
        return None
    tg = np.arange(0.0, t[-1], C.CTRL_DT)
    if len(tg) < 5:
        return None
    return tg, np.interp(tg, t, x), np.interp(tg, t, y), np.interp(tg, t, yaw)


def make_obs(x, y, yaw, twist):
    """Vectorised 9-D observation."""
    gx, gy = GOAL[0] - x, GOAL[1] - y
    c, s = np.cos(yaw), np.sin(yaw)
    gxb = c * gx + s * gy
    gyb = -s * gx + c * gy
    return np.stack([x, y, c, s, gxb, gyb,
                     twist[:, 0], twist[:, 1], twist[:, 2]], axis=1)


def build_dataset(force=False, verbose=True):
    cache = os.path.join(C.CACHE_DIR, "offline_dataset.npz")
    meta_f = os.path.join(C.CACHE_DIR, "dataset_meta.json")
    if os.path.exists(cache) and not force:
        z = np.load(cache, allow_pickle=True)
        meta = json.load(open(meta_f))
        if verbose:
            print(f"[data] cache: {z['obs'].shape[0]} transitions, "
                  f"{meta['n_runs_used']} runs used")
        return z, meta

    runs = sorted(glob.glob(os.path.join(C.DATA_DIR, "*", "Go1_dataset", "run_*")))
    obs, act, rew, nobs, term = [], [], [], [], []
    traj_list, classification = [], []
    n_used = 0

    for r in runs:
        mocap = os.path.join(r, "mocap_data.csv")
        if not os.path.exists(mocap):
            continue
        pose = load_run_pose(mocap)
        if pose is None:
            continue
        tg, xg, yg, yawg = pose
        dist = np.hypot(xg - GOAL[0], yg - GOAL[1])
        dmin = float(dist.min())
        reaches = dmin < C.GOAL_RADIUS
        approaches = (dist[0] - dist[-1]) > 0.5
        if reaches:
            cls = "REACH"
        elif approaches:
            cls = "approach"
        else:
            cls = "NON-REACH"

        # truncate at first goal entry (keep the entering step)
        if reaches:
            first = int(np.argmax(dist < C.GOAL_RADIUS))
            xg, yg, yawg = xg[:first + 1], yg[:first + 1], yawg[:first + 1]

        classification.append(dict(run=os.path.basename(r), n=len(xg),
                                   d0=float(dist[0]), dend=float(dist[-1]),
                                   dmin=dmin, cls=cls))
        traj_list.append(dict(run=os.path.basename(r), x=xg, y=yg, yaw=yawg,
                              cls=cls, reaches=bool(reaches)))

        if cls == "NON-REACH" and not C.KEEP_NONREACHING:
            continue
        if len(xg) < 3:
            continue

        # actions: body twist by finite difference, clipped to speed limits
        dt = C.CTRL_DT
        vxw = np.diff(xg) / dt
        vyw = np.diff(yg) / dt
        omega = np.diff(yawg) / dt
        c, s = np.cos(yawg[:-1]), np.sin(yawg[:-1])
        vbx = c * vxw + s * vyw
        vby = -s * vxw + c * vyw
        raw = np.stack([vbx, vby, omega], axis=1)
        a = np.stack([clip_speed(row) for row in raw])       # (N-1,3)

        in_twist = np.vstack([a[:1], a[:-1]])                # incoming twist
        s_obs = make_obs(xg[:-1], yg[:-1], yawg[:-1], in_twist)
        s2_obs = make_obs(xg[1:], yg[1:], yawg[1:], a)

        pos_t = np.stack([xg[:-1], yg[:-1]], 1)
        pos_tp = np.stack([xg[1:], yg[1:]], 1)
        d_t = np.hypot(pos_t[:, 0] - GOAL[0], pos_t[:, 1] - GOAL[1])
        d_tp = np.hypot(pos_tp[:, 0] - GOAL[0], pos_tp[:, 1] - GOAL[1])
        reached_step = d_tp < C.GOAL_RADIUS
        r_arr = (C.R_PROGRESS * (d_t - d_tp)
                 - C.R_CTRL * np.sum(a ** 2, axis=1)
                 - C.R_TIME
                 + C.R_GOAL * reached_step.astype(np.float64))
        # terminal ONLY at a genuine goal entry (never at a mere log end)
        term_arr = reached_step.astype(np.float64)

        obs.append(s_obs); act.append(a); rew.append(r_arr)
        nobs.append(s2_obs); term.append(term_arr)
        n_used += 1

    obs = np.concatenate(obs).astype(np.float32)
    act = np.concatenate(act).astype(np.float32)
    rew = np.concatenate(rew).astype(np.float32)
    nobs = np.concatenate(nobs).astype(np.float32)
    term = np.concatenate(term).astype(np.float32)
    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-6

    np.savez(cache, obs=obs, act=act, rew=rew, nobs=nobs, term=term,
             obs_mean=obs_mean, obs_std=obs_std,
             traj=np.array(traj_list, dtype=object),
             classification=np.array(classification, dtype=object))
    meta = dict(n_runs_total=len(classification), n_runs_used=n_used,
                n_transitions=int(obs.shape[0]),
                classes={c: sum(k["cls"] == c for k in classification)
                         for c in ("REACH", "approach", "NON-REACH")},
                goal=list(C.GOAL), goal_radius=C.GOAL_RADIUS, ctrl_dt=C.CTRL_DT,
                v_max=C.V_MAX, w_max=C.W_MAX,
                map=[C.MAP_XMIN, C.MAP_XMAX, C.MAP_YMIN, C.MAP_YMAX],
                keep_nonreaching=C.KEEP_NONREACHING,
                obs_mean=obs_mean.tolist(), obs_std=obs_std.tolist(),
                reward=dict(progress=C.R_PROGRESS, ctrl=C.R_CTRL, time=C.R_TIME,
                            goal=C.R_GOAL, bound=C.R_BOUND),
                classification=classification)
    json.dump(meta, open(meta_f, "w"), indent=2)
    if verbose:
        print(f"[data] {obs.shape[0]} transitions from {n_used} runs "
              f"(classes {meta['classes']})")
        print(f"[data] terminals (goal entries): {int(term.sum())} | "
              f"reward mean {rew.mean():.3f} [{rew.min():.2f},{rew.max():.2f}]")
    z = np.load(cache, allow_pickle=True)
    return z, meta


if __name__ == "__main__":
    build_dataset(force=True)
