# WM4 moving-horizon oracle prediction grid

Executable: `src/exe/run_model/wm_4/plot_moving_horizon_predictions.py`

## Run

```bash
python -m src.exe.run_model.wm_4.plot_moving_horizon_predictions --day 2022-08-20 --hour 6 --k 12 --N 4 --M 9 --B weekly_temperature
```

Equivalent Python kwargs:

```python
from src.exe.run_model.wm_4.plot_moving_horizon_predictions import run

result = run(day="2022-08-20", hour=6, k=12, N=4, M=9,
             B="weekly_temperature", group="MED", steps_per_hour=4)
```

`day` and `hour` define the initial origin `t0`. `M` is the **number of
prediction intervals**, so for `M=9` the origins start at 06:00 and end
at 14:00 on the same day. The shift is exactly one hour; a sequence that
crosses midnight continues into the next calendar day.

`k` is the prediction horizon in hours, `N` is the number of preceding
weekly-offset historical intervals used for each refit, and `B` is one of
`weekly_average`, `weekly_temperature`, or `statistical_output`.
`group` may be `MED` (default) or `ALL`.

## Every one-hour step

For each origin `t_m = t0 + m hours` for `m = 0, ..., M-1`:

1. Select exactly the `N` preceding weekly-offset, corresponding `k`-hour intervals.
2. Fit the selected baseline to exactly `N*k` historical hourly observations.
3. Run WM4 `joint_qp_params.fit_many` with one shared parameter vector
   and independent historical initial states. **Refit at every origin.**
4. With baseline and parameters fixed, use the model's convex
   `optimize_prediction_initial_state` oracle to fit the target origin's
   `[F0,R0]` from the target interval.
5. Evaluate that single target interval and append its results.

No target-period demand contributes to the baseline or the shared WM4
parameter vector. The target observations are used to determine the oracle
initial state and to evaluate errors. These are *retrospective* results,
not true out-of-sample performance.

## One figure, M subplots

The executable saves exactly **one** PNG containing M prediction-only panels;
the historical parameter-fitting days are never plotted.

Every panel uses the previous single-interval plotter's display layout:
a 24- or 25-hour demand context (black), historical baseline (orange dashed),
and WM4 oracle prediction (red). The baseline and red model line are drawn
**only** inside the k-hour prediction interval. The hour-of-day x-axis uses
actual clock hours, wrapping to 00 at midnight. All panels share a y-axis
scale, but each uses its origin's correct clock labels.

For a k=12 prediction, the context includes 6 hours before and 6 hours
after. For k=7, it includes 9 hours before and 9 hours after.

## Outputs

Default:
`files/output/wm_4/moving_horizon_predictions/med/<origin>_M<M>_k<k>_N<N>_<B>.png`

Companions with the same PNG stem:
- `.csv`: origin-level WM4 theta, oracle state, prediction error and
  baseline error.
- `_hourly.csv`: all `M*k` hourly prediction/observation pairs,
  with origin and lead-hour labels. Overlapping horizons remain separate
  rows; no deduplication.
- `.json`: aggregate pooled errors (and mean per-origin RMSE), with
  settings and methodology metadata.

Pooled RMSE is calculated over all `M*k` prediction errors:
`sqrt(mean((predicted-observed)^2))`. Overlap between hourly-shifted
prediction horizons means these errors are generally dependent.

## Tests

```bash
python -m unittest discover -s tests -p "test_wm4_moving_horizon_plot.py"
```
