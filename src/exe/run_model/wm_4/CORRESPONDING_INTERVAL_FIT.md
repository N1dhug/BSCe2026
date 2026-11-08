# Historical fitting figure: N corresponding k-hour intervals

Executable: `src/exe/run_model/wm_4/plot_corresponding_interval_fit.py`

## Example

```bash
python -m src.exe.run_model.wm_4.plot_corresponding_interval_fit --day 2022-08-20 --hour 8 --k 12 --N 4 --B weekly_temperature --group ALL
```

Equivalent Python call:

```python
from src.exe.run_model.wm_4.plot_corresponding_interval_fit import run

result = run(day="2022-08-20", hour=8, k=12, N=4, B="weekly_temperature", group="ALL")
```

Allowed baseline selectors:
`weekly_average`, `weekly_temperature`, `statistical_output`.

## What is fitted

For origin `(day,hour)`, select exactly N preceding weekly-offset
corresponding intervals with k hourly observations each. For example,
N=4, k=12, hour=8 selects the four preceding same-weekday intervals
from 08:00 through 20:00.

1. Impute missing **historical** hourly data within its original calendar
   day using the project's standard historical imputer.
2. Train the selected baseline on only the N × k corresponding
   observations. Unlike the usual baseline `.fit` methods, this code
   **does not add other calendar-day hours to the baseline fit**.
   It reuses each baseline class's same feature and estimation formulas.
3. Compute R = D - B for each corresponding interval.
4. Run the existing `joint_qp_params.fit_many` solver once:
   shared `[theta_F,theta_U,theta_R]`; independent F0 for
   each historical interval; R0 observed as the interval's D0-B0.
5. Estimate process diffusion and observation variance using nonnegative least
   squares of squared hourly residuals from the N historical fitted trajectories.
   Compute the Euler-Maruyama rate-variance kernel and the model-implied
   pointwise 95% Gaussian demand interval. Target observations are not used
   to estimate sigma or tau.
6. Plot the historical demand with the fitted baseline, WM4 trajectory,
   and blue 95% uncertainty shading inside each selected k-hour interval.
7. Hold the baseline, the single shared theta and both noise parameters fixed, then optimize both
   WM4 prediction initial-state components [F0,R0] from **target-window**
   observations using the existing convex oracle QP. Plot this retrospective
   oracle prediction in a separate PNG with a **red** WM4 line.

Only the N historical intervals contribute to fitting the baseline and
shared theta. Target-day demand is used only for the oracle initial-state
optimization and retrospective scoring; the oracle is **not** a deployable
forecast that knows its initial state before the target observations.

## Plot layout

The historical-fit PNG is a grid containing exactly N panels.
Each panel uses the same x- and y-axis scales. An additional, single-panel
PNG shows the selected prediction interval on its own.

- Black: raw observed demand across a 24–25-hour context.
- Orange dashed: historical fitted baseline, **only inside** the
  k-hour fitting interval.
- Blue: deterministic WM4 historical fit, **only inside** the
  k-hour fitting interval.
- Transparent blue band: 95% pointwise Gaussian demand uncertainty,
  estimated from all N historical trajectories.
- Pale shading: actual fitting interval [origin,origin+k).
- X-axis: actual hour of day (00–23), with the fitting interval starting at the selected `--hour`. For intervals crossing midnight, the axis continues across the day boundary while the displayed tick labels wrap back to 00.

The separate oracle PNG has the same 24–25-hour context:
- Black: observed prediction-day demand across the full context.
- Orange dashed: the **historically fitted** baseline, only within the
  prediction interval.
- Red: WM4 oracle predicted demand, only within the prediction interval.
- Transparent red band: 95% oracle-conditioned pointwise demand interval
  using historical sigma and tau, with no target-period noise refit.
- Light red shading: prediction interval.

This second PNG is named `<grid_stem>_oracle_prediction.png`, and its
companion CSV contains observed, baseline, oracle, demand standard deviation,
95% lower bound and 95% upper bound for each of
the k hourly prediction samples. The grid PNG itself remains unchanged.

The historical context is selected symmetrically:
`edge=ceil((24-k)/2)` hourly observations on each side.
Thus k=12 yields 6+12+6=24 hours; k=7 yields 9+7+9=25 hours.
Demand values outside the fitting interval are displayed only; they
are neither fitted nor imputed for plotting.

The program also writes a historical per-interval CSV with initial
states, RMSE, and historical noise estimates, and a JSON with the shared fitted
parameters, sigma, tau, process identifiability, historical and oracle
interval coverage, plus the prediction oracle state and error metrics.

The bands are pointwise marginal Gaussian intervals conditional on fitted
initial states and parameters. They are not simultaneous confidence bands
or estimates of uncertainty in the fitted baseline or drift parameters.

Default output directory:
`files/output/wm_4/corresponding_interval_fit/all/`.

## Tests

```bash
python -m unittest discover -s tests -p "test_wm4_corresponding_interval_fit.py"
```

The historical grid illustrates the common-parameter fit. The separate
prediction PNG illustrates the **retrospective** oracle initial-state fit
for the target day. It must not be interpreted as out-of-sample forecasting
accuracy because the oracle uses target-window demand.
