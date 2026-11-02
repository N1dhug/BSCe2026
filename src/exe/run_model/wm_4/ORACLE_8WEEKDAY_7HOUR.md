# WM4 eight-corresponding-weekday seven-hour oracle diagnostics

Run from the repository root on thomas_main:

    python -m src.exe.run_model.wm_4.analyze_oracle_8weekday

Defaults: MED consumer group, weekly_temperature baseline, 8 preceding
matching weekdays (eight weeks, 56 calendar days of history),
7-hour forecast (hours 00:00–06:00), and 4 Euler substeps per hour.
The full eligible January–October 2022 prediction period is used.

Each day fits its own baseline to the N=8 preceding corresponding
weekday 24-hour records, uses a single shared convex WM4 joint QP
with separately estimated historical F0 values, and holds these theta
coefficients fixed during the forecast. Historical median R0 is
retained only as a diagnostic. BOTH prediction F0 AND R0 are jointly
optimized over [-1,1]^2 using the seven target observations in a
separate globally solved convex two-variable QP (fixed theta).
Because both initial states use future demand, this is a
retrospective full-state oracle, not an operational prediction.

Historical missing hours are imputed within observed calendar days;
whole missing days are NEVER imputed and missing target-day demand
labels remain excluded. This script uses the current three native
baseline families and the current shared WM4 fitter.

## Commands

Fresh experiment:

    python -m src.exe.run_model.wm_4.analyze_oracle_8weekday

Quick three-day test in a separate directory:

    python -m src.exe.run_model.wm_4.analyze_oracle_8weekday --max-days 3 --output files/output/wm_4/oracle_8weekday_smoke

Resume an interrupted full fit:

    python -m src.exe.run_model.wm_4.analyze_oracle_8weekday --resume

Reuse previously saved three-baseline sweep CSV files (no QP refits):

    python -m src.exe.run_model.wm_4.analyze_oracle_8weekday --from-sweep files/output/wm_4/weekday_horizon_sweep_three_baselines/med

Reuse this dedicated study's own CSV files and regenerate the entire
parameter report and quantile plots without refitting:

    python -m src.exe.run_model.wm_4.analyze_oracle_8weekday --report-only

Optionally choose --group ALL, --baseline weekly_average or
--baseline statistical_output. Output paths automatically separate
groups and baselines. With --from-sweep, resolution must match the
original source --steps-per-hour; the source configuration is saved
to the output directory. Old sweep data is INCOMPATIBLE when it has only an F0 oracle;
rerun the weekday/horizon sweep with the full-state implementation.

## What gets tested

- Absolute pooled seven-hour oracle RMSE versus the baseline RMSE.
- Quantiles of individual forecast-hour absolute errors and daily
  cumulative seven-hour RMSE (20/40/50/60/80 percentiles).
- Daily error quantiles, parameter distributions and trends for
  theta_F, theta_R, theta_U and the fitted baseline heating coefficient
  (beta_C for weekly_temperature).
- Oracle F0, oracle R0, and historical R0 empirical quintiles.
- Exploratory Bartlett HAC tests of nonzero mean parameters,
  Spearman rank tests of coefficient/error association, and
  Kruskal–Wallis weekday comparisons. These are descriptive checks,
  not independent tests of causal response or parameter identification.
- Actual model contributions to R_dot (not merely raw coefficient sizes):
      restoring = theta_F * F
      damping   = theta_R * R
      price     = theta_U * U
  with signed quantile bands and per-day RMS/mean magnitudes.
- Structural sign diagnostics and regressor-design rank.

The contribution trajectories are reconstructed using the exact
deterministic WM4 Euler recurrence and the recorded horizon-specific
oracle [F0,R0]. Reconstructed RMSE is checked against the original
fitted prediction score; mismatches cause an explicit error.

## Saved results

Default folder:

    files/output/wm_4/oracle_8weekday_7hour/med/weekly_temperature/

Raw tables: daily_metrics.csv, hourly_predictions.csv,
historical_initial_states.csv, skipped_days.csv, configuration.json.

Statistics: parameter_distribution_and_hac_tests.csv,
parameter_spearman_tests.csv, weekday_kruskal_tests.csv,
hourly_error_quintiles.csv, cumulative_rmse_quintiles.csv.

Figures: forecast_error_quintiles.png, daily_rmse_quintiles.png,
rmse_distribution.png, theta_time_quantiles.png,
theta_distributions.png, theta_quintiles.png,
baseline_heating_coefficient_quintiles.png,
initial_states_and_damping_quintiles.png,
prediction_demand_quintiles.png, contribution_rms_quintiles.png,
contribution_hourly_quantiles.png, contribution_rms_time.png,
and demand_and_residual_horizon.png.

Run the regression tests:

    python -m unittest discover -s tests -p "test_wm4_oracle_8weekday.py"
