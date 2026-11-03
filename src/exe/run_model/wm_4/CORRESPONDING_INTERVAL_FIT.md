# Historical fitting figure: N corresponding k-hour intervals

Executable: `src/exe/run_model/wm_4/plot_corresponding_interval_fit.py`

## Example

```bash
python -m src.exe.run_model.wm_4.plot_corresponding_interval_fit --day 2022-08-20 --hour 8 --k 12 --N 4 --B weekly_temperature
```

Equivalent Python call:

```python
from src.exe.run_model.wm_4.plot_corresponding_interval_fit import run

result = run(day="2022-08-20", hour=8, k=12, N=4, B="weekly_temperature", group="MED")
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
5. Plot the historical demand with the fitted baseline and WM4 trajectory
   inside each selected k-hour interval.

Only the N historical intervals contribute to fitting; the target
prediction interval is not used.

## Plot layout

The single output PNG is a grid containing exactly N panels.
Each panel uses the same x- and y-axis scales.

- Black: raw observed demand across a 24–25-hour context.
- Orange dashed: historical fitted baseline, **only inside** the
  k-hour fitting interval.
- Blue: deterministic WM4 historical fit, **only inside** the
  k-hour fitting interval.
- Pale shading: actual fitting interval [origin,origin+k).
- X-axis: hours relative to that interval's own origin (hour 0).

The historical context is selected symmetrically:
`edge=ceil((24-k)/2)` hourly observations on each side.
Thus k=12 yields 6+12+6=24 hours; k=7 yields 9+7+9=25 hours.
Demand values outside the fitting interval are displayed only; they
are neither fitted nor imputed for plotting.

For convenience, the program also writes a per-interval CSV with
initial states and RMSE, and a JSON with the shared fitted parameters.

Default output directory:
`files/output/wm_4/corresponding_interval_fit/med/`.

## Tests

```bash
python -m unittest discover -s tests -p "test_wm4_corresponding_interval_fit.py"
```

These plots illustrate **historical fitting**, not oracle prediction:
no future target-day initial state is optimized.
