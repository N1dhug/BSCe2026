# WM4 N-k oracle RMSE heatmaps (three baselines)

This plotter reads the **saved estimates** from `sweep_weekday_horizon.py`; it does not refit baselines, model parameters, or oracle initial states. In the repository root, first run the full sweep (or use an already completed sweep):

```bash
python -m src.exe.run_model.wm_4.sweep_weekday_horizon --group MED
```

Plot the full set of prediction hours (1 through each selected k):

```bash
python -m src.exe.run_model.wm_4.plot_weekday_horizon_heatmaps --group MED
```

Score only hours **7–12**, with zero-based, half-open bounds `[6,12)`:

```bash
python -m src.exe.run_model.wm_4.plot_weekday_horizon_heatmaps --group MED --hour-start 6 --hour-end 12
```

For shorter horizons, available hours in the requested range are scored. Add `--require-full-window` to leave those cells blank instead. Score the **last four hours of each horizon** using:

```bash
python -m src.exe.run_model.wm_4.plot_weekday_horizon_heatmaps --group MED --last-hours 4
```

Optional `--min-horizon`, `--max-horizon`, `--min-weekdays`, and `--max-weekdays` select axes; `--vmax` sets a common upper color-scale limit. Pass `--input PATH` to read an alternate completed sweep directory and `--output PATH.png` to choose a figure path.

A Python caller can also use the reusable function:

```python
from src.exe.run_model.wm_4.plot_weekday_horizon_heatmaps import plot_heatmaps

summary = plot_heatmaps(hour_start=6, hour_end=12, cohort="global")
summary = plot_heatmaps(last_hours=4, cohort="cell")
```

The PNG has one pane per baseline, a **shared RMSE color normalization** and a **single colorbar**. A same-name CSV contains every cell's baseline, N, k, scored hour range, matched prediction days, number of hourly observations, and pooled RMSE.

## Scoring and cohort

For each historical N, horizon k and forecast date, the saved `oracle_F0` and `oracle_R0` **fitted specifically for k** are combined with saved 24-hour zero-state predictions and state sensitivities:

```
prediction = prediction_at_initial_zero + sensitivity_F0 * oracle_F0
                                        + sensitivity_R0 * oracle_R0
```

The heatmap value is **sqrt(mean(all selected squared hourly prediction errors))**, pooled across matched days. It is *not* the arithmetic mean of daily RMSEs. A sliced score uses the existing full-k oracle state; the initial state is **not reoptimized to fit the sub-window**.

By default, `cohort="global"` retains only prediction dates complete across **all three baselines and all selected N,k combinations**, making all heatmap cells comparable on the same dates. `cohort="cell"` instead pairs dates only across the three baselines at each (N,k); it retains more dates but cells may have different day counts.

The model uses two oracle state variables. Values at k=1 and k=2 are inherently optimistic as they fit at most two target observations. Results remain retrospective oracle diagnostics, not deployable forecasts.

## Files

By default, the input directory is `files/output/wm_4/weekday_horizon_sweep_expanded/med/` (or `all/` with `--group ALL`):

- `oracle_horizon_scores.csv` — per-date, per-baseline, per-(N,k) estimated two-state oracle.
- `forecast_components.csv` — observed demand, zero-state trajectory, and both state sensitivities at each forecast hour.

Both files must be present. The saved-sweep CSVs are typically generated locally and are not checked into GitHub.
