"""Produce all simulation_tex/figs/*.pdf figures and CSVs from results_raw/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import style  # noqa: E402
from analysis.aggregate import (
    aggregate_e1, aggregate_rate, bootstrap_ci, load_seeds, write_csv,
)  # noqa: E402

style.apply()

ROOT = Path(__file__).resolve().parents[2]
RES_ROOT = ROOT / "code" / "results_raw"
FIG_DIR = ROOT / "simulation_tex" / "figs"
DATA_DIR = ROOT / "simulation_tex" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================ E1 figures ====
def plot_e1():
    methods = ["Ours", "B1", "B2", "B3"]
    envs = [e for e in ["pendulum_partial", "planar_quadrotor"]
            if (RES_ROOT / "E1" / e).exists()]
    if not envs:
        print("[E1] no envs found")
        return
    # Include B5 only as an oracle reference; not in tables/boxplot.
    agg = aggregate_e1(RES_ROOT / "E1", envs=envs, methods=methods + ["B5"])

    # ----- Learning curves: deployment-validated --------------------------
    for env in envs:
        if not any(m in agg.get(env, {}) for m in methods):
            print(f"[E1] {env}: no data, skipping")
            continue
        fig, ax = plt.subplots(figsize=style.LEARNING)
        for m in methods:
            if m not in agg[env]:
                continue
            data = agg[env][m]
            dcs = data.get("deploy_curves", np.empty((0, 0)))
            dis = data.get("deploy_iters", np.empty((0, 0)))
            if dcs.size == 0:
                continue
            iters = np.nanmean(dis, axis=0)
            mean = np.nanmean(dcs, axis=0)
            lo = np.nanpercentile(dcs, 25, axis=0)
            hi = np.nanpercentile(dcs, 75, axis=0)
            ax.plot(iters, mean,
                    label=style.METHOD_LABEL[m],
                    color=style.METHOD_COLOR[m],
                    ls=style.METHOD_LINESTYLE[m],
                    marker=style.METHOD_MARKER[m], markersize=3,
                    markevery=max(1, len(iters) // 10))
            ax.fill_between(iters, lo, hi, color=style.METHOD_COLOR[m], alpha=0.10)
        # Oracle reference line (B5 = true-state-deployed Ours).
        if "B5" in agg.get(env, {}):
            data5 = agg[env]["B5"]
            dcs5 = data5.get("deploy_curves", np.empty((0, 0)))
            if dcs5.size > 0:
                mean5 = np.nanmean(dcs5, axis=0)
                last = float(mean5[-1])
                ax.axhline(last, color="#7F7F7F", ls=(0, (3, 2)), lw=0.8,
                           label=r"oracle ceiling (B5)")
        ax.set_xlabel(r"PG iteration $t$")
        ax.set_ylabel(r"deployment return $J^{\widehat{x}}_\rho$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.legend(loc="lower right", ncol=2)
        fn = "fig_E1_learning_curves_pendulum.pdf" if env == "pendulum_partial" \
             else "fig_E1_learning_curves_quadrotor.pdf"
        style.savefig(fig, FIG_DIR / fn)
        print(f"[E1] wrote {fn}")

        # CSV: per-iteration mean (downsample to ~200 rows for compactness).
        out_csv = []
        for m in methods:
            if m not in agg[env]:
                continue
            lcs = agg[env][m]["learning_curves"]
            if lcs.size == 0:
                continue
            T = lcs.shape[1]
            mean = np.nanmean(lcs, axis=0)
            step = max(1, T // 200)
            for k in range(0, T, step):
                out_csv.append({"method": m, "xvalue": int(k),
                                "metric_mean": float(mean[k]),
                                "metric_lo": float(np.nanpercentile(lcs[:, k], 2.5)),
                                "metric_hi": float(np.nanpercentile(lcs[:, k], 97.5)),
                                "n_seeds": int(lcs.shape[0])})
        write_csv(DATA_DIR / f"E1_lc_{env}.csv", out_csv,
                  ["method", "xvalue", "metric_mean", "metric_lo", "metric_hi", "n_seeds"])

    # ----- Final-return box plot ------------------
    if len(envs) == 1:
        fig, axes = plt.subplots(1, 1, figsize=style.SINGLE)
        axes_list = [axes]
    else:
        fig, axes = plt.subplots(1, 2, figsize=style.DOUBLE)
        axes_list = list(axes)
    panel_letters = ["a", "b"]
    for i, env in enumerate(envs):
        ax = axes_list[i]
        if env not in agg:
            continue
        labels, data, colors = [], [], []
        for m in methods:
            if m not in agg[env]:
                continue
            labels.append(style.METHOD_LABEL[m].split()[0])
            data.append(agg[env][m]["finals"])
            colors.append(style.METHOD_COLOR[m])
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                        showfliers=False,
                        medianprops=dict(color="black", lw=1.0),
                        whiskerprops=dict(color="#404040", lw=0.7),
                        capprops=dict(color="#404040", lw=0.7))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.30)
            patch.set_edgecolor("#404040")
            patch.set_linewidth(0.7)
        # Overlay individual seeds as small markers for transparency.
        for j, d in enumerate(data):
            rng = np.random.default_rng(j * 13)
            jitter = rng.uniform(-0.10, 0.10, size=len(d))
            ax.scatter(np.full_like(d, j + 1) + jitter, d,
                       s=10, color=colors[j], alpha=0.7,
                       edgecolors="none")
        # Oracle reference line if B5 is in the aggregator.
        if "B5" in agg.get(env, {}):
            b5_mean = float(np.mean(agg[env]["B5"]["finals"]))
            ax.axhline(b5_mean, color="#7F7F7F", ls=(0, (3, 2)), lw=0.8)
            ax.text(0.98, b5_mean, "oracle", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7, color="#404040")
        ax.set_ylabel(r"final return $J^{\widehat{x}}_\rho$")
        ax.set_xlabel("")
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        if len(envs) > 1:
            style.panel_label(ax, panel_letters[i])
            ax.text(0.5, -0.32, env.replace("_", " "),
                    transform=ax.transAxes, ha="center")
    if len(envs) > 1:
        fig.subplots_adjust(wspace=0.35)
    style.savefig(fig, FIG_DIR / "fig_E1_final_return_box.pdf")
    print("[E1] wrote fig_E1_final_return_box.pdf")

    # ----- Table E1 -------------------------------------------------------
    rows = []
    for env in envs:
        if env not in agg:
            continue
        ours_finals = agg[env].get("Ours", {}).get("finals", np.array([]))
        for m in methods:
            if m not in agg[env]:
                continue
            data = agg[env][m]
            mean, lo, hi = bootstrap_ci(data["finals"])
            row = {"env": env, "method": m,
                   "mean": mean, "std": float(data["finals"].std()),
                   "lo": lo, "hi": hi, "n": data["n_seeds"]}
            if m != "Ours" and ours_finals.size > 1 and data["finals"].size > 1:
                # Wilcoxon signed-rank (need same size).
                n_pair = min(ours_finals.size, data["finals"].size)
                try:
                    res = stats.wilcoxon(ours_finals[:n_pair], data["finals"][:n_pair])
                    row["p"] = float(res.pvalue)
                except Exception:
                    row["p"] = float("nan")
            else:
                row["p"] = None
            rows.append(row)

    # Holm step-down adjusted p-values, per env (monotone version).
    for env in envs:
        ps = [r for r in rows if r["env"] == env and r.get("p") is not None]
        ps.sort(key=lambda r: r["p"] if not np.isnan(r["p"]) else 1.0)
        running_max = 0.0
        m_total = len(ps)
        for i, r in enumerate(ps):
            adj = min(1.0, r["p"] * (m_total - i))
            running_max = max(running_max, adj)
            r["p_holm"] = float(running_max)

    write_csv(DATA_DIR / "E1_finals.csv", rows,
              ["env", "method", "mean", "std", "lo", "hi", "n"])

    # LaTeX table.
    table_path = DATA_DIR / "table_E1.tex"
    with open(table_path, "w") as f:
        f.write("\\begin{tabular}{llrrr}\n\\toprule\n")
        f.write("Env. & Method & Return mean & std & $p$ vs. Ours \\\\\n")
        f.write("\\midrule\n")
        for env in envs:
            env_rows = [r for r in rows if r["env"] == env]
            if not env_rows:
                continue
            for j, r in enumerate(env_rows):
                env_label = env.replace("_", " ") if j == 0 else ""
                if r["method"] == "Ours":
                    p_str = "---"
                elif r.get("p_holm") is not None and not np.isnan(r.get("p_holm", float("nan"))):
                    p_str = f"{r['p_holm']:.3f}"
                else:
                    p_str = "n/a"
                f.write(f"{env_label} & {style.METHOD_LABEL[r['method']]} "
                        f"& {r['mean']:.2f} & {r['std']:.2f} & {p_str} \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[E1] wrote {table_path}")


# ============================================================ E2 figure =====
def plot_e2():
    """Plot the KRR sup-norm identification error as a function of n.

    This is the quantity that ``Theorem 6`` bounds at the parametric rate
    :math:`O(n^{-(2s-d)/(2s+d)})`; it does not depend on the policy or the
    optimization, so it is a much cleaner test of the rate than the
    deployment return gap.
    """
    data = aggregate_rate(RES_ROOT / "E2", "n_")
    if not data:
        print("[E2] no data")
        return
    ns = np.array([d["var"] for d in data])
    # Prefer the sup-norm KRR error if recorded; otherwise fall back to
    # return-gap (legacy data).
    if all("sup_err_mean" in d for d in data):
        gaps = np.array([d["sup_err_mean"] for d in data])
        gap_lo = np.array([d["sup_err_lo"] for d in data])
        gap_hi = np.array([d["sup_err_hi"] for d in data])
        y_label = r"sup-norm error $\|\widehat{\Delta}_f-\Delta_f\|_\infty$"
    else:
        returns = np.array([d["return_mean"] for d in data])
        lo_r = np.array([d["return_lo"] for d in data])
        hi_r = np.array([d["return_hi"] for d in data])
        J_star = max(returns.max(), 0.0)
        gaps = J_star - returns
        gap_lo = J_star - hi_r
        gap_hi = J_star - lo_r
        y_label = r"deployment return gap $J^{\star}-J^{\widehat{x}}_\rho$"
    # Theoretical exponent for the gradient bound: (2s-d)/(2s+d).
    s_val = 4.0
    d_val = 3.0
    th_slope = -(2 * s_val - d_val) / (2 * s_val + d_val)

    fig, ax = plt.subplots(figsize=style.SINGLE)
    order = np.argsort(ns)
    ns_o = ns[order]
    g_o = gaps[order]
    lo_o = gap_lo[order]
    hi_o = gap_hi[order]
    ax.errorbar(ns_o, g_o, yerr=[np.maximum(0, g_o - lo_o), np.maximum(0, hi_o - g_o)],
                fmt="o-", color=style.METHOD_COLOR["Ours"], capsize=2,
                label=r"Ours (mean $\pm$ 95% CI)")
    # Anchor the theory reference at the FIRST (smallest-n) data point so
    # the slope is directly comparable to the empirical curve.
    n_ref = np.array([ns_o.min(), ns_o.max()])
    if g_o[0] > 1e-3:
        anchor_y = g_o[0]
        y_th = anchor_y * (n_ref / ns_o[0]) ** th_slope
        ax.plot(n_ref, y_th, "--", color="gray", lw=0.8,
                label=fr"theory slope $={th_slope:.2f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"identification samples $n$")
    ax.set_ylabel(y_label)
    # Explicit ticks for the actual sample sizes.
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator(list(ns_o)))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(n)) for n in ns_o]))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(loc="upper right")
    style.savefig(fig, FIG_DIR / "fig_E2_n_rate.pdf")
    rows = [{"method": "Ours", "xvalue": d["var"],
             "metric_mean": d["return_mean"],
             "metric_lo": d["return_lo"], "metric_hi": d["return_hi"],
             "n_seeds": d["n_seeds"]} for d in data]
    write_csv(DATA_DIR / "E2.csv", rows,
              ["method", "xvalue", "metric_mean", "metric_lo", "metric_hi", "n_seeds"])
    print(f"[E2] wrote fig_E2_n_rate.pdf; theory_slope={th_slope:.3f}")


# ============================================================ E3 figure =====
def plot_e3():
    data = aggregate_rate(RES_ROOT / "E3", "T_")
    if not data:
        print("[E3] no data")
        return
    Ts = np.array([d["var"] for d in data])
    returns = np.array([d["return_mean"] for d in data])
    lo_r = np.array([d["return_lo"] for d in data])
    hi_r = np.array([d["return_hi"] for d in data])
    J_star = max(returns.max(), 0.0)
    gaps = J_star - returns
    gap_lo = J_star - hi_r
    gap_hi = J_star - lo_r
    th_slope = -0.5

    fig, ax = plt.subplots(figsize=style.SINGLE)
    order = np.argsort(Ts)
    Ts_o = Ts[order]
    g_o = gaps[order]
    lo_o = gap_lo[order]
    hi_o = gap_hi[order]
    ax.errorbar(Ts_o, g_o, yerr=[np.maximum(0, g_o - lo_o), np.maximum(0, hi_o - g_o)],
                fmt="o-", color=style.METHOD_COLOR["Ours"], capsize=2,
                label=r"Ours (mean $\pm$ 95% CI)")
    T_ref = np.array([Ts_o.min(), Ts_o.max()])
    if g_o[0] > 1e-3:
        anchor_y = g_o[0]
        y_th = anchor_y * (T_ref / Ts_o[0]) ** th_slope
        ax.plot(T_ref, y_th, "--", color="gray", lw=0.8,
                label=fr"theory slope $={th_slope:.1f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"PG iteration budget $T$")
    ax.set_ylabel(r"deployment return gap $J^{\star}-J^{\widehat{x}}_\rho$")
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator(list(Ts_o)))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(T)) for T in Ts_o]))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(loc="upper right")
    style.savefig(fig, FIG_DIR / "fig_E3_T_rate.pdf")
    rows = [{"method": "Ours", "xvalue": d["var"],
             "metric_mean": d["return_mean"],
             "metric_lo": d["return_lo"], "metric_hi": d["return_hi"],
             "n_seeds": d["n_seeds"]} for d in data]
    write_csv(DATA_DIR / "E3.csv", rows,
              ["method", "xvalue", "metric_mean", "metric_lo", "metric_hi", "n_seeds"])
    print(f"[E3] wrote fig_E3_T_rate.pdf; theory_slope={th_slope:.3f}")


# ============================================================ E4 figure =====
def plot_e4():
    """Plot the K-truncation bias |J^H - J^K| directly.

    This is the quantity that ``Assumption 8`` bounds at the rate
    \\(\\gamma^K R_{\\max}/(1-\\gamma)^2\\). Plotted on log-y, it should
    look like a straight line with slope \\(\\log\\gamma^2\\) on log(K).
    """
    data = aggregate_rate(RES_ROOT / "E4", "K_")
    if not data:
        print("[E4] no data")
        return
    Ks = np.array([d["var"] for d in data])
    if all("trunc_bias_mean" in d for d in data):
        means = np.array([d["trunc_bias_mean"] for d in data])
        lo = np.array([d["trunc_bias_lo"] for d in data])
        hi = np.array([d["trunc_bias_hi"] for d in data])
        y_label = r"truncation bias $|J^{H}(\pi)-J^{K}(\pi)|$"
    else:
        returns = np.array([d["return_mean"] for d in data])
        lo_r = np.array([d["return_lo"] for d in data])
        hi_r = np.array([d["return_hi"] for d in data])
        J_star = max(returns.max(), 0.0)
        means = J_star - returns
        lo = J_star - hi_r
        hi = J_star - lo_r
        y_label = r"deployment return gap $J^{\star}-J^{\widehat{x}}_\rho$"

    fig, ax = plt.subplots(figsize=style.SINGLE)
    order = np.argsort(Ks)
    Ks_o = Ks[order]
    g_o = means[order]
    lo_o = lo[order]
    hi_o = hi[order]
    ax.errorbar(Ks_o, np.maximum(g_o, 1e-6),
                yerr=[np.maximum(0, g_o - lo_o), np.maximum(0, hi_o - g_o)],
                fmt="o-", color=style.METHOD_COLOR["Ours"], capsize=2,
                label=r"empirical (mean $\pm$ 95% CI)")
    gamma = 0.99
    K_ref = np.linspace(Ks_o.min(), Ks_o.max(), 80)
    # Anchor the theoretical \gamma^K reference at the smallest-K datum
    # so the slope is directly comparable to the empirical curve.
    if g_o[0] > 1e-8:
        anchor_y = g_o[0]
        anchor_K = Ks_o[0]
        # The leading dependence on K in eps_K is gamma^K (with the (1-gamma)^2
        # constant absorbed into the anchor).
        ref = gamma ** (K_ref - anchor_K)
        ax.plot(K_ref, anchor_y * ref, "--", color="gray", lw=0.8,
                label=rf"theory $\propto\gamma^{{K}}$, $\gamma\!=\!{gamma}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"rollout length $K$")
    ax.set_ylabel(y_label)
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator(list(Ks_o)))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(k)) for k in Ks_o]))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(loc="upper right")
    style.savefig(fig, FIG_DIR / "fig_E4_K_rate.pdf")
    rows = [{"method": "Ours", "xvalue": d["var"],
             "metric_mean": (d.get("trunc_bias_mean") if "trunc_bias_mean" in d
                             else d["return_mean"]),
             "metric_lo": (d.get("trunc_bias_lo") if "trunc_bias_lo" in d
                           else d["return_lo"]),
             "metric_hi": (d.get("trunc_bias_hi") if "trunc_bias_hi" in d
                           else d["return_hi"]),
             "n_seeds": d["n_seeds"]} for d in data]
    write_csv(DATA_DIR / "E4.csv", rows,
              ["method", "xvalue", "metric_mean", "metric_lo", "metric_hi", "n_seeds"])
    print("[E4] wrote fig_E4_K_rate.pdf")


# ============================================================ E5 figure =====
def plot_e5():
    envs = ["pendulum_partial", "planar_quadrotor"]
    fig, ax = plt.subplots(figsize=style.SINGLE)
    rows = []
    for env in envs:
        env_root = RES_ROOT / "E5" / env
        if not env_root.exists():
            continue
        factors, gaps, gaps_lo, gaps_hi = [], [], [], []
        for sub in sorted(env_root.iterdir()):
            if not sub.is_dir():
                continue
            seeds = load_seeds(sub)
            if not seeds:
                continue
            f = float(sub.name.split("_", 1)[1])
            g = np.array([abs(s.get("exec_gap", 0.0)) for s in seeds])
            m, lo, hi = bootstrap_ci(g)
            factors.append(f); gaps.append(m); gaps_lo.append(lo); gaps_hi.append(hi)
            rows.append({"method": env, "xvalue": f,
                         "metric_mean": m, "metric_lo": lo, "metric_hi": hi,
                         "n_seeds": len(seeds)})
        factors = np.array(factors); gaps = np.array(gaps)
        gaps_lo = np.array(gaps_lo); gaps_hi = np.array(gaps_hi)
        if len(factors) == 0:
            continue
        # Sort by R_v factor so the connecting line is monotone.
        order = np.argsort(factors)
        factors = factors[order]; gaps = gaps[order]
        gaps_lo = gaps_lo[order]; gaps_hi = gaps_hi[order]
        ax.errorbar(factors, gaps,
                    yerr=[np.maximum(0, gaps - gaps_lo),
                          np.maximum(0, gaps_hi - gaps)],
                    fmt="o-", capsize=2, color=style.METHOD_COLOR["Ours"],
                    label=r"empirical $|J^{\widehat{x}}-J|$")
        # Theoretical reference: gap = c * sqrt(R_v) under \cref{cor:exec-deploy},
        # since B_e^2 = O(tr(R_v)), B_e = O(sqrt(R_v)), gap = O(B_e).
        if len(factors) >= 2:
            anchor_x = factors[0]; anchor_y = gaps[0]
            x_ref = np.array([factors.min(), factors.max()])
            y_ref = anchor_y * np.sqrt(x_ref / anchor_x)
            ax.plot(x_ref, y_ref, "--", color="gray", lw=0.8,
                    label=r"theory $\propto \sqrt{R_v}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R_v$ inflation factor")
    ax.set_ylabel(r"$|J^{\widehat{x}}-J|$ (execution gap)")
    ax.legend(loc="upper left")
    # Explicit ticks on the log axis: the actual inflation factors.
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1", "2", "4", "8", "16"]))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    style.savefig(fig, FIG_DIR / "fig_E5_execution_gap.pdf")
    write_csv(DATA_DIR / "E5.csv", rows,
              ["method", "xvalue", "metric_mean", "metric_lo", "metric_hi", "n_seeds"])
    print("[E5] wrote fig_E5_execution_gap.pdf")


# ============================================================ E6 figure =====
def plot_e6():
    rs, probs, los, his = [], [], [], []
    root = RES_ROOT / "E6"
    if not root.exists():
        print("[E6] no data")
        return
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        seeds = load_seeds(sub)
        if not seeds:
            continue
        r = float(sub.name.split("_", 1)[1])
        ps = np.array([s["deploy_exit_prob"] for s in seeds])
        m, lo, hi = bootstrap_ci(ps)
        rs.append(r); probs.append(m); los.append(lo); his.append(hi)
    # Sort by r so the connecting line is monotone.
    rs = np.array(rs); probs = np.array(probs); los = np.array(los); his = np.array(his)
    order = np.argsort(rs)
    rs = rs[order]; probs = probs[order]; los = los[order]; his = his[order]

    # Convert our shrinking-margin "r" to the paper's "r_f" (distance to
    # the boundary): r_f = theta_box - r. Empirically, as r increases
    # the inner box shrinks, so r_f decreases and Pr(exit) increases.
    theta_box = 1.2
    r_f = theta_box - rs

    fig, ax = plt.subplots(figsize=style.SINGLE)
    if probs.max() > 0:
        ax.errorbar(r_f, probs,
                    yerr=[np.maximum(probs - los, 1e-5),
                          np.maximum(his - probs, 1e-5)],
                    fmt="o", color=style.METHOD_COLOR["Ours"], capsize=2,
                    markersize=5,
                    label=r"empirical")
        # \Cref{prop:tail} predicts Pr(\tau\le H) decays as a Gaussian tail in
        # the effective margin r_f.  We fit the effective width \sigma_e from
        # the empirical data: log[-log(1-p)] = -(r_f - \mu_e)^2/(2\sigma_e^2)
        # + const, which we obtain by a least-squares regression of
        # 1-Pr(survive)^{1/H} on r_f.  This is the empirical analogue of the
        # paper's bound; the bound's pre-factor uses \lambda_{\max}(Q) which
        # only accounts for process noise (not the policy's own action noise
        # that dominates in closed loop).
        good = (probs > 1e-3) & (probs < 1 - 1e-3)
        if good.sum() >= 2:
            H = 200.0
            per_step = 1.0 - (1.0 - probs[good]) ** (1.0 / H)
            log_per = np.log(per_step + 1e-30)
            rf_g = r_f[good]
            # Fit log(per_step) ≈ a - b * r_f^2  → per_step = e^{a-b r_f^2}.
            # Least-squares.
            X = np.column_stack([np.ones_like(rf_g), -rf_g ** 2])
            try:
                coef, *_ = np.linalg.lstsq(X, log_per, rcond=None)
                a_fit, b_fit = coef
            except np.linalg.LinAlgError:
                a_fit = b_fit = float("nan")
            if not (np.isnan(a_fit) or np.isnan(b_fit) or b_fit <= 0):
                rf_ref = np.linspace(r_f.min(), r_f.max(), 80)
                per_step_pred = np.exp(a_fit - b_fit * rf_ref ** 2)
                surv_pred = (1.0 - per_step_pred) ** H
                p_pred = 1.0 - surv_pred
                # Convert b to an effective sigma: per_step = exp(-r_f^2/(2 sigma_e^2)) * const,
                # so b = 1/(2 sigma_e^2)  →  sigma_e = 1/sqrt(2 b).
                sigma_e = 1.0 / np.sqrt(2.0 * b_fit)
                ax.plot(rf_ref, np.clip(p_pred, 1e-5, 1.0), "--",
                        color="gray", lw=0.9,
                        label=fr"fitted Gaussian tail, $\sigma_e\!=\!{sigma_e:.3f}$")
    else:
        ax.axhline(1e-4, color=style.METHOD_COLOR["Ours"], lw=1.0,
                   label=r"empirical ($0$ exits in $200$ rollouts/seed)")
    ax.set_yscale("log")
    ax.set_ylim(5e-3, 2.0)
    ax.set_xlabel(r"distance-to-boundary $r_f = \theta_{\max} - r$")
    ax.set_ylabel(r"$\Pr(\tau\leq H)$")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.legend(loc="lower left")
    style.savefig(fig, FIG_DIR / "fig_E6_safety_tail.pdf")
    rows = [{"method": "Ours", "xvalue": r,
             "metric_mean": float(p), "metric_lo": float(lo), "metric_hi": float(hi),
             "n_seeds": 1} for r, p, lo, hi in zip(rs, probs, los, his)]
    write_csv(DATA_DIR / "E6.csv", rows,
              ["method", "xvalue", "metric_mean", "metric_lo", "metric_hi", "n_seeds"])
    print("[E6] wrote fig_E6_safety_tail.pdf")


# ============================================================ E7 figure =====
def plot_e7():
    envs = [e for e in ["pendulum_partial", "planar_quadrotor"]
            if (RES_ROOT / "E1" / e).exists()]
    agg = aggregate_e1(RES_ROOT / "E1", envs=envs, methods=["Ours", "B1"])
    if len(envs) == 1:
        fig, axes = plt.subplots(1, 1, figsize=style.SINGLE)
        axes_list = [axes]
    else:
        fig, axes = plt.subplots(1, 2, figsize=style.DOUBLE)
        axes_list = list(axes)
    for i, env in enumerate(envs):
        ax = axes_list[i]
        if env not in agg or "Ours" not in agg[env] or "B1" not in agg[env]:
            continue
        for m in ["Ours", "B1"]:
            dcs = agg[env][m].get("deploy_curves", np.empty((0, 0)))
            dis = agg[env][m].get("deploy_iters", np.empty((0, 0)))
            if dcs.size == 0:
                continue
            iters = np.nanmean(dis, axis=0)
            mean = np.nanmean(dcs, axis=0)
            lo = np.nanpercentile(dcs, 25, axis=0)
            hi = np.nanpercentile(dcs, 75, axis=0)
            ax.plot(iters, mean, label=style.METHOD_LABEL[m],
                    color=style.METHOD_COLOR[m], ls=style.METHOD_LINESTYLE[m],
                    marker=style.METHOD_MARKER[m], markersize=4,
                    markevery=max(1, len(iters) // 10))
            ax.fill_between(iters, lo, hi, color=style.METHOD_COLOR[m], alpha=0.12)
        ax.set_xlabel("PG iteration")
        ax.set_ylabel(r"deployment return $J^{\widehat{x}}_\rho$")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.legend(loc="lower right")
        if len(envs) > 1:
            style.panel_label(ax, "a" if i == 0 else "b")
            ax.text(0.5, -0.32, env.replace("_", " "),
                    transform=ax.transAxes, ha="center")
    if len(envs) > 1:
        fig.subplots_adjust(wspace=0.32)
    style.savefig(fig, FIG_DIR / "fig_E7_ablation_residual.pdf")
    print("[E7] wrote fig_E7_ablation_residual.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to a subset like --only E1 E2")
    args = ap.parse_args()
    targets = args.only or ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
    if "E1" in targets:
        plot_e1()
    if "E2" in targets:
        plot_e2()
    if "E3" in targets:
        plot_e3()
    if "E4" in targets:
        plot_e4()
    if "E5" in targets:
        plot_e5()
    if "E6" in targets:
        plot_e6()
    if "E7" in targets:
        plot_e7()


if __name__ == "__main__":
    main()
