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

Every panel now uses **the exact same x-axis limits, tick positions,
tick labels, y-axis limits, and observed-demand background**. The shared
timeline spans the union of the individual prediction contexts, including
the earliest left margin and the latest right margin. On that common axis:

- Black: the *same* observed-demand series, from the shared timeline.
- Orange dashed: the baseline independently fitted at this origin, only
  within its k-hour prediction interval.
- Red: the WM4 oracle prediction, only within that same interval.
- Light red: the current interval, shifted one hour right per panel.

This makes the motion of the prediction origin and the re-estimated
parameters visible without any axis recentering. Clock hours still use
00–23 labels, with the calendar date printed at midnight so a timeline
crossing days remains unambiguous.

The union extends the individual plot windows that normally have six
hours on each side for k=12, or nine hours on each side for k=7. For
example, k=12 with nine hourly origins from 06:00 through 14:00 uses
the same 00:00–08:00 (next day) axis in *every* panel.

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
