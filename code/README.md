# Simulation Code — TAC paper companion

## Reproducing every figure

All commands are run from `code/`. Each experiment is a thin entry-point
under `experiments/`; results land in `code/results_raw/<exp>/`, and the
analysis script reads from there and writes figures/CSVs to `simulation_tex/`.

```bash
# Pre-flight: smoke test of envs and EKF.
python tests/test_smoke.py

# E2 — identification-rate sweep (n).
python experiments/run_e2.py --seeds 3 --ns 50 100 200 500 1000 2000 --T 800 --K 120

# E3 — PG-budget sweep (T).
python experiments/run_e3.py --seeds 3 --Ts 500 1000 3000 8000 --n 2000 --K 100

# E4 — rollout-length sweep (K).
python experiments/run_e4.py --seeds 3 --Ks 10 25 50 100 200 --T 1500 --n 2000

# E1 — main comparison (6 methods × 2 envs × 5 seeds).
python experiments/run_e1.py --seeds 5

# E5 — filter-vs-true-state execution gap (R_v inflation).
python experiments/run_e5.py --seeds 3 --R-factors 1 2 4 8 16

# E6 — safety-margin tail.
python experiments/run_e6.py --seeds 2 --rs 0.05 0.1 0.2 0.4 0.8

# All figures + CSVs + table.
python analysis/make_plots.py
```

## Layout

```
code/
├── envs/             # pendulum_partial, planar_quadrotor
├── filters/          # EKF
├── identification/   # Matérn-5/2 KRR residual fitter
├── training/         # policy, value-baseline, PG loop
├── algorithms/       # one driver shared by Ours and B1–B5 + B4's NN dynamics
├── experiments/      # one entry-point script per experiment
├── analysis/         # aggregation, plotting, table generation
├── utils/            # seeding, JSON/NPZ I/O
├── configs/          # YAML configs (one per experiment)
├── results_raw/      # per-run pickles/JSON/NPZ (never edited by hand)
├── tests/            # unit and smoke tests
└── DECISIONS.md      # design log (read this if anything surprises you)
```

## Compute budget

The full pipeline as documented above runs on a single 8-core CPU machine in
about 60–90 minutes. We deliberately scaled down from the §11 budget of
`CLAUDE_CODE_PROMPT.md` per §14 escape clause — see `DECISIONS.md` for the
exact cuts.

## Requirements

Pure NumPy + SciPy + Matplotlib; no GPU, no PyTorch. Versions used:

* numpy 1.24
* scipy 1.11
* matplotlib 3.7

See `requirements.txt`.
