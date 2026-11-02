# WM4 weekly_temperature N=9, H=12: cumulative RMSE

The new executable
`src/exe/run_model/wm_4/analyze_weekly_temperature_9weekday_12hour.py`
fits (or reuses) WM4 using nine preceding **corresponding weekdays**,
the weekly_temperature baseline, and one fixed 12-hour midnight-start
prediction interval [00:00,12:00).

For historical fitting, each of the nine corresponding weekdays has
its own fitted F0 and *observed* R0, while sharing the same
three dynamical coefficients. Parameter fitting uses the existing
`joint_qp_params.fit_many` convex QP via
`sweep_weekday_horizon.evaluate_configuration`.

For each target day, optimize BOTH F0 and R0 against all 12
target-hour observations using the existing convex QP, **once**.
Then record predicted and baseline errors at hourly observations
00:00,01:00,...,11:00. For each prefix ending at hour h=1,...,12:

    WM4_prefix_RMSE(h)
      = sqrt(sum_{days} sum_{j=0}^{h-1} error(day,j)^2 / (n_days*h))

    baseline_prefix_RMSE(h)
      = sqrt(sum_{days} sum_{j=0}^{h-1} baseline_error(day,j)^2
             / (n_days*h))

The initial states are not reoptimized as h increases.
This gives the *cumulative mean-square error* of the SAME
12-hour oracle trajectory. The prefix RMSE can rise or fall
because the denominator increases with h. The total cumulative
**squared error** cannot decrease.

The 12th prefix equals the pooled full-horizon N9/H12 RMSE.
The baseline uses the same target-day observations and dates.

## Commands

Compute from source data (potentially many historical QP fits):

    python -m src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour

Reuse an already completed expanded sweep with zero refitting,
preserving its cohort (recommended if available):

    python -m src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour --from-sweep files/output/wm_4/weekday_horizon_sweep_expanded/med

Quick two-day fresh fit:

    python -m src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour --max-days 2

Regenerate figures from this experiment's saved data:

    python -m src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour --report-only

The default output folder is

    files/output/wm_4/weekly_temperature_9weekday_12hour/med/

Change the output directory with `--output` and group with `--group ALL`.

## Outputs

- `daily_metrics.csv`: daily N9/H12 QP oracle states F0/R0,
  parameters, 12-hour WM4 RMSE, and baseline RMSE.
- `hourly_predictions.csv`: one error and baseline error for each
  of the twelve hours on each successful day.
- `cumulative_rmse.csv`: exactly twelve cumulative prefix rows
  with pooled RMSE, baseline RMSE, improvement, and per-hour RMSE.
- `cumulative_rmse_vs_baseline.png`: requested 00–12
  cumulative WM4 versus baseline RMSE trajectories.
- `hourly_rmse_vs_baseline.png`: independent lead-hour errors,
  avoiding interpretation of cumulative averaging as model recovery.
- `cumulative_improvement_pct.png`: reduction in prefix RMSE
  relative to the same baseline.
- `skipped_days.csv`: failed daily historical fits, where relevant.
- `configuration.json`: input configuration and oracle version.

A report-only command does not change the oracle states. If the
source sweep used an old F0-only oracle, the executable rejects it
rather than silently treating the historical R0 as optimized.

Validation:

    python -m unittest discover -s tests -p "test_wm4_weekly_temperature_9weekday_12hour.py"

**Interpretation limitation:** Both prediction initial states are
estimated retrospectively from the *same 12 observations being scored*.
These prefix curves assess how a single 12-hour retrospective fitted
trajectory matches the target observations; they are not deployable
historical-only forecasting errors.
