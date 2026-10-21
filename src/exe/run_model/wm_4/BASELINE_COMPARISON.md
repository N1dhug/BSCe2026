# WM4 moving-horizon baseline comparison

The executable `src/exe/run_model/wm_4/compare_baselines.py` evaluates three
existing implementations of the expected consumption `O(t)` on the ALL and
MED consumption aggregates:

| Name | Class in `src/tools/baseline.py` | Features |
|---|---|---|
| `statistical_output` | `statistical_output` | Robust Huber/ridge regression with hour-of-day harmonics, weekday interactions, seasonal Fourier harmonics and nonlinear chill-temperature features. |
| `weekly_average` | `weekly_average` | Historical mean consumption for 168 weekday/hour cells. |
| `weekly_temperature` | `weekly_temperature` | Weekday/hour means and one fitted global linear heating-degree adjustment. |

## Fair fitting procedure

For every origin and baseline:

1. Fit the baseline independently for each group, using observations
   with timestamps **strictly before the origin's calendar day**
   (identical look-back cutoff to the existing WM4 rolling experiment).
   The fitted baseline is reused for all origins within that day.
2. Use the preceding 24 hours of observed consumption, prices and
   temperature to reconstruct `R_obs=I_obs-O`; initialize
   `[F0,R0]=[0,R_obs[0]]` by default. Fit WM4 with the existing
   `optimized_qp_params.fit(...,fit_initial_state=True)`: alternate
   the constrained rate-transition QP with the convex initial-state
   QP. **Both F0 and R0 are fitted in the historical window**, each
   restricted to [-1,1]. The fitted WM4 drift has no intercept.
3. Evaluate training free-run RMSE and the fitted transition-QP loss.
4. Predict the next 24 hours in two ways:
   - **standard**: continue from the historically fitted trajectory's
     terminal `[F,R]` at the forecast origin. There is **no fitting
     to future consumption** in this branch, and no clipping or
     artificial reset of the propagated initial state.
   - **oracle**: freeze those same historical rate coefficients and
     optimize *future* `[F0,R0]` in [-1,1] against all 24 future
     consumption observations, exactly as in WM4's prior oracle test.
     This is a retrospective *optimistic benchmark*, **not a real
     forecast**.
5. Also score the **baseline-only** prediction, `I_hat=O`.

All three forecasts use the same future observed price and temperature
series as exogenous inputs. This is an *ex-post* price/weather scenario
evaluation: standard WM4 does **not** use future consumption, but it
does use realized future prices and temperatures. A prospective forecast
would instead require price and weather forecasts/scenarios.

The number of historical samples, time integration grid, training
window, 24-hour forecast horizon, normalization and observation masks
are shared. Results are reported separately for ALL and MED because
those aggregates have different consumption data.

## Usage

From the repository root:

```bash
python -m src.exe.run_model.wm_4.compare_baselines --max-origins 3 --fail-fast
python -m src.exe.run_model.wm_4.compare_baselines --stride 24
python -m src.exe.run_model.wm_4.compare_baselines --stride 1
```

The experiment defaults to 2022-02-28 through 2022-08-22 23:00,
`ALL,MED`, 4 substeps per hour, and 50 alternating iterations. Narrow
it with `--groups MED`, `--baselines statistical_output,weekly_temperature`,
`--start`, `--end`, or `--max-origins`. `--seed-rate zero` changes
the R0 **initial guess**, not whether R0 is subsequently optimized.

Run basic regression checks:

```bash
python -m unittest discover -s tests -p 'test_wm_4_compare_baselines.py'
```

## Results

Output directory: `files/output/wm_4/baseline_comparison/`.

- `matched_summary.csv`: baseline-only, standard, and oracle pooled
  RMSE, windowwise RMSE, biases, fitting quality, runtime, convergence.
- `matched_horizon.csv`: hourly RMSE, MAE and bias for all three
  prediction types on **the same origin cohort**.
- `matched_window_metrics.csv`: one row per group / origin / baseline,
  with fitted WM4 coefficients and initial/terminal states.
- `matched_hourly_errors_wide.csv`: a row per window with the 24
  forecast-hour errors for baseline-only, standard and oracle modes.
- `all_available_* `: unpaired results from each successful window.
  **Do not use these for direct comparison if any baseline fails
  disproportionately.** The matched variants use only origins where
  *every requested baseline* succeeds, per group.
- `paired_baseline_gains.csv`: matched pairwise MSE and RMSE differences\n  and descriptive win percentages for baseline-only, standard and oracle.\n  Rolling windows overlap, so percentages are **not** independent-sample\n  significance tests.\n- `skipped.csv`: the failed group/origin/baseline windows with reasons.
- Group-specific comparison plots: aggregate RMSE, horizon RMSE, and
  example 24-hour trajectories.

The oracle is allowed to look at the future 24-hour consumption to
set the initial state. The standard forecast is not. Comparing the
baselines' oracle RMSE measures the quality of the baseline plus fitted
rate dynamics with a favorable initial state; the standard forecast
additionally tests practical state propagation.
