# Twenty ordered fixed-response pairs in WM4/WM5 dynamics

Run on branch thomas_main from the repository root:

    python -m src.exe.run_model.wm_4.sweep_fixed_responses

This is a nonlinear-response *generalization of the WM4 rate dynamics*
using the existing WM5 basis and admissibility machinery:

    F_dot = R
    R_dot = theta_F F + theta_R g(R) + theta_U h(U)
    D = B + R

Here theta_F < 0, theta_R < 0 and theta_U <= 0. g/h are not fitted:
they are selected from an experimental **fixed catalog defined directly
in src/exe/run_model/wm_4/sweep_fixed_responses.py**, not a new model
module. All functions are strictly increasing odd functions with
g(0)=h(0)=0 and g(-1)=h(-1)=-1, g(1)=h(1)=1:

    exponential:       sign(x)*expm1(2*abs(x))/expm1(2)
    logarithmic:       sign(x)*log1p(4*abs(x))/log1p(4)
    polynomial_steep:             1.15*x - 0.15*x**3
    polynomial_very_steep:        1.35*x - 0.35*x**3
    polynomial_extremely_steep:   1.49*x - 0.49*x**3

The exponential and logarithmic curves are centered/normalized
versions of the requested families. A raw e^x is not admissible
because it does not map [-1,1] to [-1,1] or pass through the origin.
For cubic a*x-b*x**3 with p(±1)=±1, a=1+b; demanding p'(x)>=0
on the full [-1,1] forces b<=0.5. Thus the maximum possible
central slope is 1.5, so the three cubics cannot become arbitrarily
steep while retaining this form and monotonicity. The selected
central slopes are 1.15, 1.35 and 1.49.

The 20 cases are the 5*4 **ordered distinct pairs**
(g,h), excluding g=h. Swapping functions changes the model.

## Historical weekday and forecast horizon search

Defaults:
- MED group, weekly_temperature statistical baseline (the best
  previously evaluated baseline), 2022 Jan-Oct data.
- N=2,...,10 preceding occurrences of the same weekday for **both**
  baseline estimation and shared model parameter estimation.
- H=6,...,24 future hourly observations from the same target date.
- A distinct joint convex historical QP for each (day,N,g,h),
  called through **src.params.wm_5.joint_qp_params.fit_many**.
  The QP optimizes [theta_F*F0_day1, ..., theta_F*F0_dayN,
  theta_F,theta_R,theta_U] with one historical F0 per day.
  The WM5 parameter layer validates and fits the fixed nonlinear
  responses and produces all historical deterministic states.
  The pre-existing src.solver.joint_drift_qp.solve_many assembles
  the multi-day QP with optional precomputed nonlinear feature
  matrices, and calls the existing src.solver.active_set.active_set.solve
  numerical optimizer. No additional QP solver is introduced.
  All derivative-response features g(R),h(U) are fixed, and the
  structural, equilibrium, Euler stability, and reconstructed
  historical F-path constraints are linear. The QP remains convex.
- Historical median R0 is retained ONLY as a diagnostic,
  never fixed during an oracle prediction.
- Both forecast F0 AND R0 are optimized *jointly for every H* against
  the first H target observations. Nonlinear g makes the nonlinear
  initial-state problem potentially nonconvex, even though the
  historical theta fit remains a convex QP.
  The numerical oracle evaluates a 2D grid in [-1,1]^2, then performs
  bounded multi-start L-BFGS-B refinement from the best grid state,
  the historical-median seed, and the neutral seed. The final result
  is never worse than the best grid candidate, but GLOBAL OPTIMALITY
  IS NOT CERTIFIED. The output stores both oracle_F0 and oracle_R0.
- Baseline/parameter fitting imputes only missing historical hours
  within existing days; fully missing historical days and missing
  target observation hours remain excluded.

There are 20*9*19=3420 (g,h,N,H) combinations for each eligible
calendar day, but only 20*9=180 historical QP fits per day because
the 19 horizons reuse theta. Expect a computationally significant
full-period experiment.

The first eligible forecast day for N=10 is 2022-03-12.
Using the same start date for all N prevents a varying
warm-up period from artificially favoring shorter windows.

## Rankings

For each ordered pair (g,h), search all its historical-weekday
counts N and forecast horizons H. Select its best combination using:

    RMSE(g,h,N,H) = sqrt(mean_over_days(MSE_day(g,h,N,H)))

Then rank the **20 response permutations by the lowest absolute
aggregate RMSE at each permutation's best N/H**. Baseline percentage
improvement is NOT the ranking objective. Save the full grid
in all_weekday_horizon_splits.csv.

The ranking prioritizes a **fully matched cohort** of calendar
dates for which all 20 pairs, all requested N, and all requested H
have valid results. If none exists it uses an explicitly flagged
unmatched fallback; in that case examine the number of days per
candidate and do not interpret the order as a perfectly controlled
head-to-head comparison.

Across different forecast horizons the prediction tasks differ:
a seven-hour oracle result should not be called a 24-hour forecast.
Also, the forecast full [F0,R0] oracle uses future target observations and
cannot be deployed as a real-time initial-state estimator.

## Commands

Full experiment:

    python -m src.exe.run_model.wm_4.sweep_fixed_responses

Smoke test, two dates and only N=8, with a separate output directory:

    python -m src.exe.run_model.wm_4.sweep_fixed_responses --max-days 2 --min-weekdays 8 --max-weekdays 8 --output files/output/wm_4/fixed_responses_smoke

Try a different baseline:

    python -m src.exe.run_model.wm_4.sweep_fixed_responses --baseline weekly_average

Old F0-only sweep output is NOT compatible with the full-state
version. Regenerate the sweep or write to a new output directory.

Resume an interrupted run using identical settings:

    python -m src.exe.run_model.wm_4.sweep_fixed_responses --resume

Regenerate rankings and figures from saved CSVs:

    python -m src.exe.run_model.wm_4.sweep_fixed_responses --report-only

For nondefault settings, repeat the same CLI options when using
--resume/--report-only (the saved configuration is validated).

## Results

By default all results are stored under:

    files/output/wm_4/fixed_response_sweep/med/weekly_temperature/

- response_library.csv — every fixed formula and exact
  derivative bounds, center slope, and endpoint values.
- parameter_fits.csv — the full historical joint-QP theta, fit
  loss/rank/time, baseline coefficients/metadata, training dates,
  fitted historical F0 moments and imputation counts.
- historical_states.csv — all per-training-day F0 and R0 estimates.
- horizon_scores.csv — daily absolute scores, baseline comparison,
  chosen oracle F0/R0, optimizer method, and error metrics for every (g,h,N,H).
- skipped_fits.csv — unavailable days or other errors, with cause.
- all_weekday_horizon_splits.csv — absolute aggregate RMSE and
  per-day RMSE quintiles for all 3420 configurations.
- response_pair_ranking.csv — **all 20 permutations ranked from
  best to worst**, each using its best N,H combination and
  corresponding sample count and baseline error.
- response_functions.png — five admissible response shapes.
- response_pair_ranking.png — ordered pair ranking bar chart.
- top_five_horizon_curves.png — lowest RMSE by horizon for
  the strongest five response pairs.
- twenty_response_heatmaps.png — common-scale 4x5 panel with
  one RMSE heatmap per ordered pair.
- configuration.json — complete experiment settings.

These experiments retain the previous WM4 sweep/analysis source code.
They do not overwrite the original linear WM4 dynamics.

Run focused tests:

    python -m unittest discover -s tests -p "test_wm4_fixed_responses.py"
