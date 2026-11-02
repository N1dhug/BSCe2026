# WM4 three-baseline × historical-weekdays × horizon sweep

Run from repository root on branch thomas_main:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon

**Default comparison: all three established WM4 baselines** from
src/tools/baseline.py:

1. statistical_output — Huber-ridge Fourier/weekday/season/temperature baseline.
2. weekly_average — weekday-and-hour historical mean.
3. weekly_temperature — weekday-and-hour mean with fitted heating response.

Every baseline is fitted to exactly the same N preceding **corresponding
weekdays**, not all earlier days. The sweep examines N=2,...,14; the pre-existing
minimum seven-day training requirement is lowered **only for this sweep**
using the new optional min_samples=48 constructor argument; the regular
baseline implementations retain their former 168-sample default.

## Hourly-gap imputation and full missing days

All three native baselines now use the shared
`src/tools/impute_hourly.py` historical-data preprocessing.
For each *observed historical calendar day* the original hourly
index is reconstructed and isolated missing hours are filled strictly
within that day. Demand and temperature use time-linear interpolation
(interior gaps) and the nearest available same-day endpoint (edge gaps).
The price signal uses same-day forward fill and, at a missing first
hour, same-day backward fill because hourly spot prices are stepwise.

**Whole missing days are never synthesized.** If a selected historical
corresponding weekday has no observations for even one required
signal (I, U or Temp), the sweep skips that (forecast day, baseline, N)
combination with a `Missing full day` error in `skipped_windows.csv`.
No values are borrowed from preceding/following days or from
future forecast dates. There is no carry-over of a baseline across
an entirely missing historical day.

Baseline fitting and historical WM4 parameter fitting use the *same
hourly-imputed historical rows*. The target prediction day's observed
demand is **never imputed** or treated as a ground-truth score when
missing: if a target day is incomplete, that configuration is skipped.
This avoids manufacturing a better full-state oracle prediction score from
fabricated future demand. Baseline metadata and
`parameter_estimates.csv` record the count of imputed historical
hourly slots and imputation counts separately for I, U and Temp.

The imputation update changes the experiment. Its configuration
version is now `3_baseline_matched_training_hourly_imputation_full_oracle_v4_N14_H1`.
Do not combine these results with earlier, non-imputed CSVs.
By default, the expanded sweep writes to the separate
`weekday_horizon_sweep_expanded` output directory, preserving previous
2–10 weekday / 6–24 hour outputs. Choose an explicit `--output`
path for further experimental variants. After the updated
run, `--report-only` can regenerate summary plots without refitting.

For each forecast day, N = 2,...,14 and H = 1,...,24:

- Fit the selected baseline on exactly N historical matching weekdays,
  using both historical demand and temperatures where supported.
- Fit shared WM4 [theta_F,theta_U,theta_R] through its joint convex QP
  on the same N historical 24-hour trajectories, separately fitting
  each historical F0 (bounded preferred domain).
- Compute the historical median R0 from the N preceding weekday midnight
  residuals ONLY as a comparison diagnostic; do not fix it in the oracle.
- For every forecast horizon H, jointly optimize BOTH oracle F0 and R0
  over [-1,1]^2 against its H target demand observations. The trajectory
  is affine in these initial states for fixed theta, so this separate
  two-variable fit is a globally solvable convex QP using the existing
  model._least_squares_qp/active_set solver. The historical theta fit
  is unchanged and reused across all H.
- Evaluate WM4 absolute prediction RMSE against the same baseline's
  RMSE. Baseline-relative skill is diagnostic, NEVER a ranking criterion.

There are **3 * 13 * 24 = 936** distinct baseline/N/H configurations per
forecast date. A single historical QP parameter fit is reused for all 24 H choices,
so each date requires up to 39 historical QP fits (one per baseline and N),
plus horizon-specific initial-state QPs.

The same available forecast days are used for the ranking across all
methods and N counts (the N=14 warm-up is 14 weeks = 98 calendar days;
starting from 2022-01-01, the earliest possible target is 2022-04-09).
All target observations are retrospective: **oracle [F0,R0] is not a deployable
forecast initial-state estimate**, and temperature and price use observed target-day
series. The normalization is fitted to the historical full-year range.

Ranking uses the **LOWEST ABSOLUTE pooled oracle WM4 prediction RMSE**:

    RMSE(B,N,H) = sqrt(mean_over_matched_days(MSE(day,B,N,H)))

The top three rank across all three baselines and all N/H settings.
Tie-breaking is by baseline name, then N, then H.

**Important H=1–2 caveat:** The same target observations are used to
optimize BOTH oracle initial states. At H=1, only R0 influences the
initial demand sample: the oracle can normally match this first
observation exactly by choosing R0=observed demand minus baseline.
F0 is not identified from that single sample. At H=2, two states
can often match the two target observations nearly exactly.
Consequently, a near-zero H=1 or H=2 retrospective RMSE is
essentially a fitted-state interpolation result, NOT evidence
of superior predictive performance. Cross-horizon absolute RMSE
ranking now tends to favor these trivial short horizons. Interpret
N and baseline differences **at a fixed substantive H** instead,
and separately evaluate forecasts with strictly historical initial
states. The CSV records each horizon's initial-state design rank.

The matched cohort
requires every baseline/N combination to have all requested H results
on each included day. If none is available, the program explicitly
warns and uses unmatched data as a fallback. Compare N *at fixed H* in
best_by_horizon.csv because differing H are different prediction tasks.

## Commands

Full 3-baseline MED study:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon

Quick smoke test in a separate output folder:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon --max-days 3 --output files/output/wm_4/weekday_horizon_3baseline_smoke

Compare the previously studied horizon range while retaining 14-day backlog:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon --min-horizon 6 --max-horizon 24 --output files/output/wm_4/weekday_horizon_sweep_h6_24

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

    files/output/wm_4/weekday_horizon_sweep_expanded/med/

- parameter_estimates.csv: (forecast day, baseline, N), fitted theta,
  damping, fit loss, rank, runtime, exact N historical dates, fitted
  baseline coefficients and preprocessing metadata (JSON-valued CSV
  fields), 24 target-day baseline values, historical state moments,
  and numbers of hourly training samples imputed for each signal.
- historical_initial_states.csv: every historical [F0,R0] per
  baseline/day/N.
- oracle_horizon_scores.csv: (forecast day, baseline, N, H), BOTH oracle F0 and R0,
  absolute RMSE/MSE/MAE/bias, baseline comparison, and initial-state design rank.
- forecast_components.csv: exact hourly observed demand, baseline,
  price, zero-initial-state forecast and F0/R0 sensitivities per baseline/day/N.
  Reconstruct each prediction from the zero-initial-state trajectory by adding
  oracle F0 and R0 times their respective sensitivities.
- skipped_windows.csv: failed fit with the baseline and N.
- ranking_all_available.csv, ranking_matched.csv, top_three.csv,
  top_three_h6plus.csv (excludes H=1–5), best_by_horizon.csv,
  parameter_summary.csv.
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
