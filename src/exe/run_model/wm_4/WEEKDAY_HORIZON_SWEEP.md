# WM4 three-baseline × historical-weekdays × horizon sweep

Run from repository root on branch thomas_main:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon

**Default comparison: all three established WM4 baselines** from
src/tools/baseline.py:

1. statistical_output — Huber-ridge Fourier/weekday/season/temperature baseline.
2. weekly_average — weekday-and-hour historical mean.
3. weekly_temperature — weekday-and-hour mean with fitted heating response.

Every baseline is fitted to exactly the same N preceding **corresponding
weekdays**, not all earlier days. To allow N=2,...,10, their pre-existing
minimum seven-day training requirement is lowered **only for this sweep**
using the new optional min_samples=48 constructor argument; the regular
baseline implementations retain their former 168-sample default.

For each forecast day, N = 2,...,10 and H = 6,...,24:

- Fit the selected baseline on exactly N historical matching weekdays,
  using both historical demand and temperatures where supported.
- Fit shared WM4 [theta_F,theta_U,theta_R] through its joint convex QP
  on the same N historical 24-hour trajectories, separately fitting
  each historical F0 (bounded preferred domain).
- Choose forecast R0 as the median of N historical midnight deviations
  I(00:00)-B(00:00) using that baseline; never target-day demand.
- For each H, compute oracle F0 in [-1,1] using only the first H future
  target demand observations, holding baseline, theta and R0 fixed.
  H-specific F0 is solved by analytic linear sensitivity, not by an
  additional parameter QP.
- Evaluate WM4 absolute prediction RMSE against the same baseline's
  RMSE. Baseline-relative skill is diagnostic, NEVER a ranking criterion.

There are **3 * 9 * 19 = 513** distinct baseline/N/H configurations per
forecast date. A single QP parameter fit is reused for all 19 H choices,
so each date needs 27 fits rather than 513 fits.

The same available forecast days are used for the ranking across all
methods and N counts (earliest day for N=10 is 2022-03-12).
All target observations are retrospective: **oracle F0 is not a deployable
forecast estimate**, and temperature and price use observed target-day
series. The normalization is fitted to the historical full-year range.

Ranking uses the **LOWEST ABSOLUTE pooled oracle WM4 prediction RMSE**:

    RMSE(B,N,H) = sqrt(mean_over_matched_days(MSE(day,B,N,H)))

The top three rank across all three baselines and all N/H settings.
Tie-breaking is by baseline name, then N, then H. The matched cohort
requires every baseline/N combination to have all requested H results
on each included day. If none is available, the program explicitly
warns and uses unmatched data as a fallback. Compare N *at fixed H* in
best_by_horizon.csv because differing H are different prediction tasks.

## Commands

Full 3-baseline MED study:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon

Quick smoke test in a separate output folder:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon --max-days 3 --output files/output/wm_4/weekday_horizon_3baseline_smoke

All-consumer dataset:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon --group ALL

Resume interrupted run with identical settings:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon --resume

Rebuild report and figures after successful data CSV saving:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon --report-only

Optionally use --baselines weekly_average weekly_temperature to run
only those two methods. Specify the same --baselines in a resumed or
report-only run. Other controls: --steps-per-hour (default 4),
--min-weekdays / --max-weekdays, --min-horizon / --max-horizon,
--start-day / --end-day and --progress-every.

## Output

Default output directory:

    files/output/wm_4/weekday_horizon_sweep_three_baselines/med/

- parameter_estimates.csv: (forecast day, baseline, N), fitted theta,
  damping, fit loss, rank, runtime, exact N historical dates, fitted
  baseline coefficients and preprocessing metadata (JSON-valued CSV
  fields), 24 target-day baseline values, and historical state moments.
- historical_initial_states.csv: every historical [F0,R0] per
  baseline/day/N.
- oracle_horizon_scores.csv: (forecast day, baseline, N, H), oracle F0,
  absolute RMSE/MSE/MAE/bias and baseline comparison.
- forecast_components.csv: exact hourly observed demand, baseline,
  price, neutral-F0 forecast and sensitivity per baseline/day/N.
  Reconstruct an H-specific prediction by adding oracle F0 times
  the sensitivity.
- skipped_windows.csv: failed fit with the baseline and N.
- ranking_all_available.csv, ranking_matched.csv, top_three.csv,
  best_by_horizon.csv, parameter_summary.csv.
- rmse_heatmap_<baseline>.png, baseline_skill_heatmap_<baseline>.png
  (separate figures for all three methods).
- rmse_by_horizon.png (three method panels with 9 N curves each).
- best_baseline_by_horizon.png (best N within each baseline for each H).
- top_three_daily_rmse.png and top_three_rmse_distribution.png
  (top three across ALL methods).
- configuration.json: settings used for checkpoint compatibility.

Unlike the old single-average-baseline sweep, this comparison fits the
three **existing** baseline families on the same N histories. It writes
to a different output directory to preserve historical one-baseline
results.

Synthetic/regression checks:

    python -m unittest discover -s tests -p "test_wm4_weekday_horizon_sweep.py"
