# Baseline-only N,k demand quality (Chapter 5)

The existing `sweep_weekday_horizon.py` already saved **out-of-sample baseline
RMSE/MSE** for each successful ARM (N,k,origin,baseline) fit. However, it did
not generate the full **RMSE / MAE / signed bias** baseline-only heatmaps for
**both historical (in-sample) and target (out-of-sample)** intervals.

Run `plot_baseline_moving_horizon_quality.py` to produce this direct comparison.

## Methodology

The executable **does not refit** the baselines or the ARM. It reads the saved
`parameter_estimates.csv` from the same completed ARM hourly-origin sweep
used by `plot_moving_horizon_quality.py`, restores each N,k-specific baseline
using the saved coefficients, and evaluates:

- **Fit quality:** the exact N preceding corresponding weekday/hour windows
  used to estimate the baseline and ARM drift, at hourly offsets 0..k-1.
  The same within-day historical imputation is used as the ARM heatmap.
- **Fit validation:** the following target interval, exactly k hourly demand
  observations at offsets 0..k-1. Only the historically fitted baseline is
  used; there is **no oracle state** or use of target demand for estimation.

The historical and target errors for a given cell come from **the same
successful configurations and origin set**. Both baseline formulations are
`weekly_average` and `weekly_temperature`; the default group is `ALL`.

At each (N,k,baseline), errors are pooled **per observation** before computing
RMSE, MAE and bias (baseline minus observed demand). Each fit origin contributes
N*k historical observations and k target observations. Overlapping hourly
origins remain separate evaluation instances, as in the ARM figures.

The figures use the existing 3-row by 2-column demand-quality renderer,
with the **same color limits across both baselines and both the fit and
validation figures** for each metric.

## Run

From repository root, after the original ARM sweep has completed:

```bash
python -m src.exe.run_model.wm_4.plot_baseline_moving_horizon_quality --group ALL
```

For a strict shared-origin comparison across both baselines and all selected
(N,k) configurations:

```bash
python -m src.exe.run_model.wm_4.plot_baseline_moving_horizon_quality \
  --group ALL --cohort matched
```

The `--source` option specifies an alternative completed ARM sweep directory,
and `--output` selects a different destination. Optional N/k bounds and
`--start-day`/`--end-day` select a subset. To align the ARM comparison,
use the **same** bounds and `--cohort` setting with:

```bash
python -m src.exe.run_model.wm_4.plot_moving_horizon_quality \
  --group ALL --mode both --cohort all
```

Results default to:

```text
files/output/wm_4/baseline_moving_horizon_quality/all/
    fit_cumulative_quality.png
    prediction_cumulative_quality.png
    fit_cumulative_quality.csv
    prediction_cumulative_quality.csv
    configuration.json
```

Each CSV records baseline, N, k, scored origin count, observation count,
pooled RMSE, pooled MAE and signed bias.

## Interpretation

A baseline is evaluated **without** the dynamic ARM correction, so these
metrics can be compared to corresponding ARM fit/oracle metrics on the same
windows. However, ARM target figures are **oracle-conditioned**, while the
baseline predictions use only historical fitted coefficients and target
exogenous temperature (the same realized temperature input as the ARM).
The comparison therefore measures achievable conditional model performance,
not a comparison of two fully operational future forecasts.

The two heatmaps' colors are directly comparable by metric. Short k values
score fewer future observations. As in the ARM evaluation, the N,k sweeps
represent independently fitted configurations; horizon variation does not
measure the forecast drift of one fixed parameter set.

Tests:

```bash
python -m unittest tests.test_wm4_baseline_moving_horizon_quality
```
