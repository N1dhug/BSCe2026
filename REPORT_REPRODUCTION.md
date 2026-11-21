# Reproducing report Chapters 3–5

The maintained implementation contains only the BM1 fixed-beta and WM4/ARM workflows described by the report. Other research prototypes were removed from the current branch; Git history retains their revisions. The raw input spreadsheets and historical output artifacts remain untouched.

## Data and descriptive statistics (Chapter 3)

```bash
python -m src.exe.data.baseline_diagnostics
python -m src.exe.data.season_overview
python -m src.exe.data.weeks_overview
```

## Moving-horizon construction (Chapter 4)

```bash
python -m src.exe.run_model.wm_4.plot_corresponding_interval_fit --help
python -m src.exe.run_model.wm_4.plot_moving_horizon_predictions --help
```

The reusable scheduling, baseline estimation, and window construction routines live in `src/method/moving_horizon.py`. The stochastic variance routines live in `src/method/arm_variance.py` and `src/method/moving_horizon_uncertainty.py`.

## Quantitative results (Chapter 5)

```bash
python -m src.exe.run_model.wm_4.sweep_weekday_horizon --help
python -m src.exe.run_model.wm_4.plot_baseline_moving_horizon_quality --help
python -m src.exe.run_model.wm_4.plot_moving_horizon_quality --help
python -m src.exe.run_model.wm_4.fit_moving_horizon_diffusion --help
python -m src.exe.run_model.wm_4.plot_diffusion_demand_std --help
python -m src.exe.run_model.bm_1.evaluate_fixed_beta_samples --help
python -m src.exe.run_model.wm_4.selected_sample_nll --help
python -m src.exe.run_model.wm_4.plot_selected_sample_oracle_nll --help
```

The fixed-beta Fokker–Planck solver and its parameter estimator are together in `src/params/bm_1_fixed_beta.py`. The WM4 joint convex QP and target-state optimizer are in `src/params/wm_4.py`.

The selected ARM target initial states are conditioned on the target day's observations (retrospective oracles), **not** causal day-ahead forecasts. Historical stochastic variance calibration does not use target outcomes.

To check the maintained tests:

```bash
python -m pytest -q tests/test_report_imports.py tests/test_moving_horizon_method.py tests/test_bm_1_fixed_beta.py tests/test_wm4_weekday_horizon_sweep.py
```

Full numerical reproductions require the original input data in `files/input/data`. The historical output archives under `files/output` are preserved rather than treated as unused source code.
