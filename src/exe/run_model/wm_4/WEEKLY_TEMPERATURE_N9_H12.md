# WM4 weekly_temperature: N=9, H=12, TWELVE rolling origins

`src/exe/run_model/wm_4/analyze_weekly_temperature_9weekday_12hour.py`
implements twelve separate 12-hour prediction windows for EACH
eligible target day, not twelve cumulative prefixes from a single
midnight-origin forecast.

## Twelve origins and one common 24-hour clock

There are exactly 12 plots on a 4 × 3 figure, one per starting hour:

    origin 00:00 -> [00:00,12:00)
    origin 01:00 -> [01:00,13:00)
    origin 02:00 -> [02:00,14:00)
    ...
    origin 11:00 -> [11:00,23:00)

Each plot's horizontal axis is always 00–24 hours and its vertical
axis always uses the SAME global RMSE scale. Its 12-hour interval
is highlighted, and the area below the cumulative WM4 RMSE is
shaded. A dashed curve shows the weekly-temperature baseline RMSE
for the identical origin-specific target window.

The twelve origins cover hourly observations o,...,o+11 and an
underlying physical window [o,o+12). A window starting at 12:00
would be a thirteenth origin and is deliberately excluded to
satisfy the requested EXACTLY twelve plots.

## Historical parameter estimation and prediction state

For each prediction day:

1. Use 9 preceding corresponding weekdays to fit the
   weekly_temperature baseline.
2. Fit the shared WM4 parameters [theta_F,theta_U,theta_R]
   using the existing historical joint QP. Every historical day
   retains its own optimized F0 and observed R0.
3. Reuse ONE shared theta vector for all twelve target origins.
4. Separately for each origin o=0,...,11, jointly optimize BOTH
   F0 and R0 in [-1,1]^2 using ALL twelve target-hour
   observations at that origin; the existing globally solvable
   state-only convex QP is used.
5. Retain the SAME origin-specific optimized [F0,R0] for each
   accumulated prefix h=1,...,12 (no repeated oracle fitting
   at the progressively increasing prefixes).

For origin o, cumulative pooled prefix RMSE at elapsed hour h is

    RMSE_o(h) =
      sqrt(sum_day sum_(j=0,...,h-1) (prediction[day,o+j]
           - observed[day,o+j])^2 / (N_days*h)),

with the same formula for the baseline residuals.
All twelve origins are compared over an identical cohort of
complete target days. Each h=12 endpoint is the pooled
full-window RMSE for that specific origin.

The pooled cumulative RMSE is an average and may decrease.
Cumulative squared error cannot decrease.

## Run the new twelve-origin experiment

Use your expanded sweep's already fitted historical theta
and eligible day cohort. ONLY theta is reused: midnight
oracle [F0,R0] CANNOT be shifted to another origin, so the
new script refits BOTH states for each of its twelve origins.

    python -m src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour --from-sweep files/output/wm_4/weekday_horizon_sweep_expanded/med

Or fit historical theta afresh:

    python -m src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour

Smoke test on two prediction days:

    python -m src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour --max-days 2

Regenerate the 12-panel figure and summaries from this script's
NEW rolling-origin CSVs:

    python -m src.exe.run_model.wm_4.analyze_weekly_temperature_9weekday_12hour --report-only

Report-only refuses the prior midnight-only experiment output.
The default output folder remains:

    files/output/wm_4/weekly_temperature_9weekday_12hour/med/

Files:
- `origin_cumulative_rmse_panels.png`: requested 4×3 panel figure.
- `origin_cumulative_rmse.csv`: 12 origins × 12 prefix hours.
- `origin_aggregate_rmse.csv`: pooled 12-hour RMSE per origin.
- `origin_daily_metrics.csv`: full-state oracle and scores per day/origin.
- `origin_hourly_predictions.csv`: hourly observations and residuals.
- `skipped_days.csv`: failed/incomplete full days; excluded as a whole.
- `configuration.json`: experiment version and selected parameters.

Run focused tests with:

    python -m unittest discover -s tests -p "test_wm4_weekly_temperature_9weekday_12hour.py"

**Important:** Every oracle uses the future twelve observed target
hours to fit both initial states. This is retrospective agreement,
not historical-only forecasting performance.
