"""Aggregation utilities: read per-seed JSON/NPZ, compute mean ± 95% bootstrap CI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) of x via percentile bootstrap."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(x.mean()), lo, hi


def load_seeds(root: Path) -> List[dict]:
    """Read all seed_*.json under `root` and return a list of payloads."""
    root = Path(root)
    out = []
    if not root.exists():
        return out
    for jf in sorted(root.glob("seed_*.json")):
        try:
            with open(jf) as f:
                payload = json.load(f)
            # Add the NPZ data if present.
            npz = jf.with_suffix(".npz")
            if npz.exists():
                with np.load(npz) as data:
                    for k in data.files:
                        payload[k] = data[k]
            out.append(payload)
        except Exception as e:
            print(f"failed to load {jf}: {e}")
    return out


def aggregate_e1(results_root: Path, envs: List[str], methods: List[str]) -> Dict:
    """Return {env: {method: {mean, lo, hi, seeds, learning_curves}}}."""
    out = {}
    for env in envs:
        out[env] = {}
        for m in methods:
            sub = Path(results_root) / env / m
            seeds = load_seeds(sub)
            if not seeds:
                continue
            finals = np.array([s["deploy_return_mean"] for s in seeds])
            grad_sq = np.array([s["deploy_grad_norm_sq"] for s in seeds])
            exits = np.array([s["deploy_exit_prob"] for s in seeds])
            exec_gap = np.array([abs(s.get("exec_gap", 0.0)) for s in seeds])
            mean, lo, hi = bootstrap_ci(finals)
            # Pad learning curves to common length.
            lcs = [s["learning_curve"] for s in seeds if "learning_curve" in s]
            if lcs:
                Lmax = max(len(c) for c in lcs)
                lc_arr = np.full((len(lcs), Lmax), np.nan)
                for i, c in enumerate(lcs):
                    lc_arr[i, :len(c)] = c
            else:
                lc_arr = np.empty((0, 0))
            # Deployment-validated curve. Convert to best-so-far so the curve
            # shows monotone-non-decreasing model-selection performance
            # (matches the "return best policy" practice we use at the end of
            # training).
            dcs = [s["deploy_curve"] for s in seeds if "deploy_curve" in s and len(s["deploy_curve"]) > 0]
            dis = [s["deploy_iter"] for s in seeds if "deploy_iter" in s and len(s["deploy_iter"]) > 0]
            if dcs:
                Dmax = max(len(c) for c in dcs)
                dc_arr = np.full((len(dcs), Dmax), np.nan)
                di_arr = np.full((len(dcs), Dmax), np.nan)
                for i, (c, di) in enumerate(zip(dcs, dis)):
                    c_arr = np.asarray(c, dtype=float)
                    # Running max — "best policy so far".
                    c_arr = np.maximum.accumulate(c_arr)
                    dc_arr[i, :len(c)] = c_arr
                    di_arr[i, :len(di)] = di
            else:
                dc_arr = np.empty((0, 0))
                di_arr = np.empty((0, 0))
            out[env][m] = {
                "finals": finals,
                "final_mean": mean, "final_lo": lo, "final_hi": hi,
                "grad_sq": grad_sq,
                "exit": exits,
                "exec_gap": exec_gap,
                "learning_curves": lc_arr,
                "deploy_curves": dc_arr,
                "deploy_iters": di_arr,
                "n_seeds": len(seeds),
            }
    return out


def aggregate_rate(results_root: Path, var_prefix: str) -> List[dict]:
    """For E2/E3/E4: each subdir is var_<value>, contains seed_*.json.

    Returns gradient-norm, deployment-return, and (when present)
    sup-norm-error statistics so different plot variants can choose
    the metric that maps most directly to the theoretical bound.
    """
    out = []
    for sub in sorted(Path(results_root).iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.startswith(var_prefix):
            continue
        seeds = load_seeds(sub)
        if not seeds:
            continue
        v = float(sub.name.split("_", 1)[1])
        gs = np.array([s["deploy_grad_norm_sq"] for s in seeds])
        rs = np.array([s["deploy_return_mean"] for s in seeds])
        gm, glo, ghi = bootstrap_ci(gs)
        rm, rlo, rhi = bootstrap_ci(rs)
        rec = {
            "var": v,
            "grad_sq_mean": gm, "grad_sq_lo": glo, "grad_sq_hi": ghi,
            "return_mean": rm, "return_lo": rlo, "return_hi": rhi,
            "n_seeds": len(seeds),
            "samples": gs,
        }
        # Optional: sup-norm KRR error (used by E2).
        sup = [s.get("sup_err_f") for s in seeds if s.get("sup_err_f") is not None]
        if sup:
            sup_arr = np.asarray(sup, dtype=float)
            sm, slo, shi = bootstrap_ci(sup_arr)
            rec["sup_err_mean"] = sm
            rec["sup_err_lo"] = slo
            rec["sup_err_hi"] = shi
        # Optional: K-truncation bias (used by E4).
        tb = [s.get("trunc_bias_mean") for s in seeds if s.get("trunc_bias_mean") is not None]
        if tb:
            tb_arr = np.asarray(tb, dtype=float)
            bm, blo, bhi = bootstrap_ci(tb_arr)
            rec["trunc_bias_mean"] = bm
            rec["trunc_bias_lo"] = blo
            rec["trunc_bias_hi"] = bhi
        out.append(rec)
    return out


def write_csv(path: Path, rows: List[dict], columns: List[str]):
    """Write a simple CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in columns) + "\n")
