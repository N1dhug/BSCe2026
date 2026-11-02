# WM4 seven-hour oracle prediction error by forecast origin

This executable evaluates the optimal fixed configuration from the
weekday/horizon sweep: **eight preceding corresponding weekdays** with
the **weekly_temperature** baseline and a **seven-hour** forecast horizon.

The forecast origin is moved through each evaluation day, one hour
at a time, from **00:00 to 17:00**:

    origin 00 -> hours 00, 01, 02, 03, 04, 05, 06
    origin 01 -> hours 01, 02, 03, 04, 05, 06, 07
    ...
    origin 17 -> hours 17, 18, 19, 20, 21, 22, 23

Every origin yields a seven-observation prediction and a daily RMSE.
For each origin, aggregate RMSE is calculated over all matched days
and all seven hours:

    aggregate_RMSE(origin) =
        sqrt(sum_over_days_and_leads(prediction - observed)^2 / (days * 7))

This is NOT the mean of the daily RMSE values. The mean and median
daily RMSE are also recorded separately.

## Cumulative seven-hour evolution in 18 full-day-aligned panels

There are **18**, rather than 17, distinct integer starting origins
0,...,17 inclusive. Each has seven observations at hours o,...,o+6,
so the physical prediction interval is [o,o+7), with the last
interval [17,24). The single 6-by-3 figure
origin_cumulative_rmse_panels.png shows **one panel per origin**.

For origin o and elapsed observation count h=1,...,7:

    cumulative_RMSE(o,h) =
        sqrt(sum_{days d} sum_{j=0}^{h-1}
             (prediction[d,o+j] - observed[d,o+j])^2 / (N*h))

This uses the SAME seven-hour oracle F0 for all seven cumulative
prefixes of a given (day, origin); it does not refit F0 at
h=1,2,...,7. Otherwise, each point would represent a different
optimized forecast rather than accumulated errors from one forecast.
The final h=7 point equals the original aggregate RMSE for origin o.

Each panel displays the full 00–24 clock, highlights and fills
the corresponding seven-hour interval from start o to end o+7,
and uses the SAME global y-axis limits, computed from all WM4
and baseline cumulative RMSE values across all origins.

Within each one-hour bin the step height is the pooled cumulative
RMSE after the corresponding observed hour; the final step ends
at o+7. Because this is an average of squared errors, the
cumulative RMSE need not increase monotonically with h.

## Parameter fitting and initial states

For each prediction day:

1. Fit the weekly_temperature baseline on exactly eight preceding
   matching weekdays, with historical within-day hourly imputation.
2. Fit the three WM4 parameters theta_F, theta_U and theta_R jointly
   on those eight complete historical 24-hour profiles, using the
   existing joint_qp_params.fit_many() and active_set solver.
   This joint parameter fit is performed **once per prediction day**,
   not once per origin hour.
3. At each target origin hour, set R0 to the historical median of
   I(origin_hour) - O(origin_hour) across those eight historical days.
   This is origin-specific, but not fitted to target-day observations.
4. Optimize that origin's F0 over [-1,1] analytically using its
   seven target-hour observations (retrospective seven-hour oracle).
5. Integrate the WM4 Euler dynamics from the new origin and score the
   resulting seven-hour prediction, alongside the statistical baseline.

The experimental Euler time step is exactly the same as the original
24-hour evaluation: dt=1/(24 * steps_per_hour) normalized days.

Using the target's seven hours for the oracle F0 means the evaluated
numbers are a **retrospective representational fit**, not a deployable
out-of-sample forecasting score.

## Run

From repository root:

    python -m src.exe.run_model.wm_4.analyze_origin_7hour_rmse

Reuse historical parameter estimates from the already completed N8/H7
91-day oracle experiment, preserving precisely the same days and
theta values and checking that the new origin 00 result matches the
saved seven-hour RMSE:

    python -m src.exe.run_model.wm_4.analyze_origin_7hour_rmse --from-oracle files/output/wm_4/oracle_8weekday_7hour/med/weekly_temperature

A two-day smoke experiment (fresh QP fits):

    python -m src.exe.run_model.wm_4.analyze_origin_7hour_rmse --max-days 2 --output files/output/wm_4/origin_7hour_smoke

Resume an interrupted run with identical settings:

    python -m src.exe.run_model.wm_4.analyze_origin_7hour_rmse --resume

Regenerate statistics and figures from stored CSVs:

    python -m src.exe.run_model.wm_4.analyze_origin_7hour_rmse --report-only

## Outputs

By default:

    files/output/wm_4/origin_7hour_rmse/med/weekly_temperature/

- origin_daily_metrics.csv: 18 origin-specific score rows per
  evaluation day, fitted parameters, historical R0 and oracle F0.
- origin_hourly_predictions.csv: individual prediction, observation,
  baseline, and residual at every (day, origin, lead hour).
- origin_aggregate_rmse.csv: the requested **18-point aggregate RMSE
  series**, baseline RMSE, gain, bias, and daily RMSE quantiles.
- origin_lead_rmse.csv: pooled lead-hour RMSE by origin, an 18 x 7 grid.
- origin_cumulative_rmse.csv: exact 18 x 7 pooled cumulative RMSE
  trajectories after 1,...,7 observed hours, baseline included,
  with each point mapped to its absolute end-of-hour clock time.
- origin_aggregate_rmse.png: principal WM4-versus-baseline RMSE plot,
  x-axis origin hours 00–17.
- origin_rmse_quantiles.png: aggregate RMSE and cross-day 20–80% spread.
- origin_lead_rmse_heatmap.png: pooled RMSE for each origin/lead-hour pair.
- origin_cumulative_rmse_panels.png: 18 panels across a common 24-hour
  x-axis, with the respective seven-hour interval shaded under
  the cumulative RMSE curve and identical y-axis limits.
- skipped_days.csv: dates excluded for incomplete target or historical
  input, infeasible fitting, or failed consistency checks.
- configuration.json: settings used for resume verification.

All plotted origins and their seven cumulative steps share the same set of complete evaluation dates;
incomplete target days are excluded as a whole. For additional confidence,
use --from-oracle: origin 00 must reproduce the original 91-day
seven-hour RMSE (approximately 0.07896767) if the source input files
are unchanged.

Tests:

    python -m unittest discover -s tests -p "test_wm4_origin_7hour_rmse.py"
