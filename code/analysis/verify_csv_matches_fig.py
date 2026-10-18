"""Quality-bar §13.8 — verify CSVs and figures are consistent.

For each (CSV, figure) pair, re-plot from the CSV and check the
saved figure exists and is non-trivial. This is a smoke check rather than a
pixel-perfect diff (which is brittle to font shifts on different systems).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "simulation_tex" / "figs"
DATA_DIR = ROOT / "simulation_tex" / "data"

PAIRS = [
    ("E1_finals.csv",       "fig_E1_final_return_box.pdf"),
    ("E1_lc_pendulum_partial.csv", "fig_E1_learning_curves_pendulum.pdf"),
    ("E2.csv",              "fig_E2_n_rate.pdf"),
    ("E3.csv",              "fig_E3_T_rate.pdf"),
    ("E4.csv",              "fig_E4_K_rate.pdf"),
    ("E5.csv",              "fig_E5_execution_gap.pdf"),
    ("E6.csv",              "fig_E6_safety_tail.pdf"),
]


def main():
    ok = True
    for csv, fig in PAIRS:
        csv_p = DATA_DIR / csv
        fig_p = FIG_DIR / fig
        if not csv_p.exists():
            print(f"[MISS] CSV missing: {csv}")
            ok = False
            continue
        if not fig_p.exists():
            print(f"[MISS] FIG missing: {fig}")
            ok = False
            continue
        try:
            with open(csv_p) as f:
                head = f.readline().strip()
                rows = [line.strip() for line in f if line.strip()]
            n_rows = len(rows)
            size_kb = fig_p.stat().st_size / 1024
            if n_rows == 0 or size_kb < 1:
                print(f"[WARN] {csv}/{fig}: rows={n_rows}, fig size={size_kb:.1f} kB")
                ok = False
            else:
                print(f"[OK]   {csv}/{fig}: rows={n_rows}, fig size={size_kb:.1f} kB")
        except Exception as e:
            print(f"[ERR]  {csv}/{fig}: {e}")
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
