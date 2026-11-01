# WM4 corresponding-weekday × forecast-horizon oracle sweep

Run from the repository root on thomas_main:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon

This evaluates all 171 combinations of:

- N = 2,...,10 previous corresponding weekdays (nine selections);
- H = 6,...,24 hourly target observations (19 forecast horizons).

Defaults: MED group, January–October 2022 source data, hourly origins at
midnight, 4 deterministic Euler substeps/hour. For the full N=10 comparison
the first valid date is 2022-03-12 (70 days of warm-up); **the identical
calendar forecast cohort is used for all N**. The historical matching days
are exactly 7,14,...,7N days before each forecast origin.

Run a small empirical smoke test in its own output directory:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon --max-days 3 --output files/output/wm_4/weekday_horizon_smoke

Use --group ALL for the aggregate of all consumers, or --steps-per-hour 60
for one-minute deterministic numerical steps. Running N=2,...,10 across
all eligible dates may require many constrained QP solves: one shared
coefficient fit per (day,N), not a separate fit for every horizon H.

The run continuously appends completed per-day/per-N results. After an
interruption, use the identical arguments plus --resume to skip complete
pairs; duplicate rows from an interrupted write are deduplicated in the
final CSV export. Configuration mismatches are rejected. After all fits
(or after a partial run), use --report-only to reconstruct rankings and
plots directly from saved CSVs without rerunning parameter estimation.

## Empirical baseline (same N for baseline and WM4)

For each target weekday, the statistical baseline B(h) is the **hour-by-hour
arithmetic mean** of the N immediately preceding matching weekdays' actual
demand. Its 24 values are linearly interpolated on the fine simulation
grid, with wrap-around from hour 23 to hour 0. The *same baseline* is
applied to the N historical trajectories used for parameter estimation
and to the target prediction. No target-day demand enters baseline
construction; temperature is not additionally regressed in this
specific empirical comparison. Thus, both baseline and WM4 use the
same N corresponding weekdays.

WM4 fits the **updated joint convex QP** over the full N historical
24-hour trajectories, with one shared [theta_F,theta_U,theta_R] and one
fitted F0 for each historical weekday. The rate at the start of each
historical day is the observed I(0)-B(0). All coefficients obey the
existing WM4 restoring, damping, price-response and Euler-stability
conditions, and the reconstructed historical F trajectories remain
in their preferred [-1,1] domain.

The forecast R0 is the median of the N historical initial demand
deviations I(0)-B(0); it uses no target-day observation.

## Oracle F0 versus horizon

Once theta is fixed from historical data, the demand trajectory is
affine in the forecast F0. For every candidate horizon H, its oracle F0
is optimized on *only the first H hourly target observations* and
clipped to [-1,1]. Therefore F0 is reoptimized separately for
H=6,...,24, but theta, the baseline and R0 are the same for every H
with the same (day,N). A linear sensitivity calculation and cumulative
dot-products recover every H-specific optimum without repeated dynamic
optimization.

**The oracle uses the future H observations, so these are retrospective
best-state reconstruction scores, not deployable predictions.**
Price and temperature are retrospectively observed exogenous signals,
and the full-year 2022 data normalization is likewise retrospective.

## Ranking and saved results

The main ranking uses **pooled RMSE**:
sqrt(mean(per-day MSE)) across a strictly matched set of forecast days
for which *every requested N* has all H scores. All combinations for
the same H use exactly the same dates. If there is no such cohort, the
tool warns and temporarily ranks the available data on unmatched dates
as a fallback. Saved all-available and matched rankings remain separate.

Across *different H*, the objective includes different numbers of
forecast hours. A six-hour optimum need not be better over 24 hours.
Consult **best_by_horizon.csv** to compare weekday counts at *fixed H*.
Also inspect the signed comparison to the N-specific baseline.

Saved under files/output/wm_4/weekday_horizon_sweep/med/ by default:

- parameter_estimates.csv: day,N,all three fitted theta, damping ratio,
  fit loss/runtime/rank, training dates and all 24 **exact empirical
  baseline values** for that day,N.
- historical_initial_states.csv: **every training-day F0 and R0**.
- oracle_horizon_scores.csv: day,N,H,**all three theta**, optimal
  H-specific F0, fixed historical R0, RMSE/MSE/MAE/bias, baseline
  metrics and sensitivity-identifiability diagnostic.
- forecast_components.csv: day,N,hour,observed demand, baseline,
  price, prediction with F0=0, and sensitivity to F0.
  Any full H-specific predicted hourly demand can be reconstructed as
  prediction_at_F0_zero + oracle_F0 * sensitivity_F0.
- skipped_windows.csv: any unsuccessful historical QP window.
- ranking_all_available.csv, ranking_matched.csv, top_three.csv,
  best_by_horizon.csv, parameter_summary.csv.
- rmse_heatmap.png (all N versus H; best three highlighted).
- baseline_skill_heatmap.png (percentage gain over the N-day baseline).
- rmse_by_horizon.png (one curve per N).
- top_three_daily_rmse.png, top_three_rmse_distribution.png.
- configuration.json: the full experimental configuration.

The terminal prints **the three best N/H combinations**, their
aggregate oracle RMSE, corresponding baseline RMSE, improvement
percentages, daily RMSE quantiles and mean/SD fitted theta.
No results are invented: the ranking is computed from the CSV data
that the completed run actually produced.

Tests:

    python -m unittest discover -s tests -p "test_wm4_weekday_horizon_sweep.py"

To inspect an existing completed run without rerunning:

    python -m src.exe.run_model.wm_4.sweep_weekday_horizon --report-only

For a run with non-default min/max counts/horizons, pass those same
limits to --report-only so that the cohort is interpreted correctly.
