# Reproducing report Chapters 3–5

The maintained implementation contains only the BM1 fixed-beta and WM4/ARM workflows described by the report. Other research prototypes were removed from the current branch; Git history retains their revisions. The original input spreadsheets remain untouched. Obsolete generated results were removed from `files/output/`; the active report procedures recreate their outputs when run.

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

Full numerical reproductions require the original input data in `files/input/data`. The tracked output archives for retired models and exploratory experiments were removed, including older descriptive overview plots that are not used as report figures. Report-specific result folders are created automatically by the retained executables. Historical artifacts remain recoverable through Git history.


## Single-day reproducible fits (both models)

These entrypoints accept the same five primary settings:
- `--day` is the prediction-origin calendar date (YYYY-MM-DD).
- `--origin-point` is the origin hour of that date, 0 through 23.
- `--k` is the k-hour fitting/prediction interval, 2 through 24.
- `--N` is the number of earlier corresponding weekdays.
- `--baseline` is one of `weekly_average`, `weekly_temperature`, `statistical_output`.

```bash
python -m src.exe.run_model.bm_1.fit_single_day --day 2022-08-20 --origin-point 8 --k 12 --N 6 --baseline weekly_temperature
python -m src.exe.run_model.wm_4.fit_single_day --day 2022-08-20 --origin-point 8 --k 12 --N 6 --baseline weekly_temperature
```

The programs use the existing 2022 dataset, train their baseline and model
parameters only on N matching historical k-hour intervals, and produce
historical and target plots, hourly CSVs, and summary JSON under
`files/output/<model>/single_day/`. Both use all consumers by default; add
`--group MED` for the electrical-heating subset. BM1 converts normalized
prices from the signed ARM range [-1,1] to [0,1]; WM4 retains signed prices.

The BM1 historical scoring uses *filtered one-step-ahead conditional*
likelihoods, but its target uses *open-loop marginal* likelihoods. WM4 fits a
joint constrained drift QP and historical Gaussian noise parameters. Target
initial conditions in **both** programs are retrospectively optimized using
target observations (oracle analysis, not forward-looking validation).
