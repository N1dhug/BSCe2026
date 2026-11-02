# WM4 Jan–End moving-horizon full-state oracle [F0,R0] study

Run from the repository root:

    python -m src.exe.run_model.wm_4.predict_all_moving_horizon

The default is **MED** (electrically heated consumers), the complete available
2022 January–October daily range, one prediction per complete day, a
24-hour horizon and four model steps per physical hour. Optionally run
all consumers with --group ALL.

Quick smoke test:

    python -m src.exe.run_model.wm_4.predict_all_moving_horizon --max-days 3 --progress-every 1

Use --start-day 2022-04-01 --end-day 2022-05-31 to restrict the calendar,
or --steps-per-hour 60 for minute-resolution Euler integration.

If the evaluation already produced daily_metrics.csv and hourly_predictions.csv,
reprint the theta/RMSE terminal report **without rerunning any fits** using:

    python -m src.exe.run_model.wm_4.predict_all_moving_horizon --report-only

The same --output directory is used when reading saved results.

## Protocol

Each prediction date requires **exactly five complete preceding matching
weekdays** (dates -35, -28, -21, -14 and -7 days). These historical
24-hour trajectories share a **single convex quadratic-program parameter fit**
via joint_qp_params.fit_many. The joint vector is

    [theta_F*F0_day1, ..., theta_F*F0_day5, theta_F, theta_U, theta_R].

Each historical day receives its own fitted F0; all days share
[theta_F,theta_U,theta_R]. The initial historical rate equals
observed I(0)-B(0). Only 24 hourly observations enter each
historical rate-transition fit; unobserved endpoint transitions
are excluded. Preferred reconstructed F-domain and structural
stability constraints are retained.

The statistical baseline is estimated from **strictly before** the
prediction day. Price and temperature over the target horizon are
the retrospectively observed exogenous series. Note that the 2022
min/max input-price normalization uses the full period:
this is a *retrospective* experiment.

The observed residual from the last pre-forecast hour is recorded
only as the historical **R0 comparison diagnostic**.

For fixed historical theta, prediction demand is affine in **both**
forecast initial-state components [F0,R0]. The full target-day
retrospective oracle fits both components in [-1,1]^2 jointly
using the existing globally solvable convex two-variable QP
(src.params.wm_4.optimized_qp_params.optimize_prediction_initial_state).
The historical theta QP remains unchanged, and no nonlinear
optimization solver is introduced.

The oracle uses all 24 target observations and is NOT a deployable
forecast. The first-hour target observation can directly determine
the fitted R0, which will lower the apparent first-hour error.

Previously cached CSVs from F0-only evaluations must be recomputed;
--report-only rejects stale output lacking oracle_R0.

The earliest 35 calendar days cannot satisfy the five-prior-weekday
restriction and appear as warm-up skips in skipped_days.csv.
Missing or incomplete later days have explicit skip reasons.

## Results

Under files/output/wm_4/predict_all_moving_horizon/:

- daily_metrics.csv: daily scores, baseline comparison, theta,
  damping ratios, oracle F0/R0, historical R0, training dates, rank and runtime.
- hourly_predictions.csv: 24-hour prediction, baseline, observations,
  errors, model states and prices for each eligible day.
- horizon_rmse_quintiles.csv: pooled per-hour RMSE, pooled cumulative
  RMSE, hour-wise absolute-error quintiles and across-day
  cumulative-RMSE quintiles.
- rmse_horizon_quintiles.png: pooled horizon RMSE and cumulative
  RMSE with quintile boundaries.
- daily_rmse_quintiles.png: per-day RMSE distribution and its quintile markers.
- skipped_days.csv: early warm-up and other failed origin reasons.

Absolute error at each individual forecast hour is the
one-observation root-mean-square error. For genuine per-day
RMSE through forecast hour h, the second panel calculates
sqrt(mean(error[j]**2 for j<=h)) for each day, then the
cross-day empirical quintile boundaries. Pooled RMSE takes
the square root *after* averaging squared errors.

Terminal output contains pooled RMSE and bias, baseline comparison,
daily RMSE quintiles, mean/std/min/quintiles/max for all estimated
theta coefficients, theta correlations, structural checks,
near-zero price share, design rank deficiency, damping regimes
and weekday-specific results.
