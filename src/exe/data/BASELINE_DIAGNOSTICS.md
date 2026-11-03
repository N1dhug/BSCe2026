# Preliminary statistical tests and baseline diagnostics

Executable: `src/exe/data/baseline_diagnostics.py`

## Run

```bash
python -m src.exe.data.baseline_diagnostics --groups MED ALL --N 4 --k 12
```

To compare several backlogs and horizons:

```bash
python -m src.exe.data.baseline_diagnostics --groups MED --N 2 4 8 --k 6 12 24 --stride-hours 24 --max-origins 80
```

Optional arguments: `--start 2022-03-01`, `--end 2022-11-01`
(end is exclusive), `--base-temperature 18`, `--max-week-lag 8`,
`--max-acf-lag 336`, `--hac-lags 24`, and `--output /some/folder`.

Python entrypoint:

```python
from src.exe.data.baseline_diagnostics import run
result=run(groups=("MED","ALL"),backlogs=(4,),horizons=(12,),
           stride_hours=24,max_origins=64)
```

The executable loads the same aligned, normalized `I`, `U`, and
physical-temperature `Temp` signals as WM4 via `group_frame` and
creates separate outputs for MED and ALL.

## Preliminary statistical analyses

1. **Temporal patterns**. Compute mean, standard deviation and sample
   count for each weekday/hour; plot weekday hourly profiles. Evaluate
   pairwise raw-demand ACF using actual timestamps, including lags 24 h
   and 168 h. Observations separated by gaps are never misidentified as
   adjacent hours.
2. **Corresponding historical similarity**. For weekly offsets
   `n = 1..max_week_lag`, compute matched-sample RMSE for `168*n`
   hourly separation. Compare against `168*n +/- 24` hour lags
   (the adjacent weekday at roughly the same historical distance).
   All three comparisons use exactly the same valid target timestamps;
   improvement is descriptive rather than proof of causal relevance.
3. **Temperature and price dependence**. Remove 168 weekday/hour
   fixed effects and regress residual demand first on the residual
   heating-degree factor `max(0,base_temperature-Temp)`, then jointly
   on the residual heating-degree factor and price. Report coefficients,
   reduced MSEs, incremental MSE improvement, and approximate
   Bartlett/Newey-West HAC normal p-values. Plot daily demand versus
   heating-degree factor and *partial* temperature and price
   relationships. Price coefficients are associations, **not causal
   estimates of flexibility response**.
4. **Residual diagnostics**. Save hourly residuals and matched-hour
   ACFs after the weekday/hour adjustment, after temperature, and after
   temperature and price. Use residual ACF to identify temporal
   structure not captured by the candidate nominal baseline variables.

## Rolling-origin baseline comparison

For each origin in a common, evenly sampled set and for each choice of
`N` and `k`:

- Use **only** the `N` corresponding historical `k`-hour intervals
  to estimate each baseline using the *same internal estimation logic*
  as the WM4 single-interval plotter.
- Evaluate `weekly_average`, `weekly_temperature` and
  `statistical_output` against the next `k` hourly demand values.
- Report per-origin RMSE, MSE, MAE and bias; aggregate errors across
  the matched target hours and plot pooled RMSE/bias.

All three baselines use the **same training and evaluation timestamps**
at each origin. All evaluation origins are selected from the period
that can accommodate the **largest** requested backlog/horizon.
By default the candidate origin sequence uses a 24-hour stride and is
thinned evenly across the full admissible period to at most 64 origins,
so the default analysis does not concentrate only on the first weeks.

**No missing target demand is imputed.** In this diagnostic, incomplete
historical `N*k` intervals are skipped rather than being filled from
hours outside the selected intervals. This is deliberately stricter
than the WM4 plotting script's historical-day imputation: the baseline
comparison isolates exactly the chosen intervals. Skipped windows are
counted in the JSON output.

`Temp` is the **observed target-period temperature**. Thus, baseline
parameter estimation is out-of-sample but the evaluation is
retrospective with known exogenous conditions. A deployable forecast
would replace `Temp` with an available weather forecast. Likewise,
the use of historical prices here is exploratory and not a causal
estimate. Repeated targets across overlapping horizons are *not*
independent observations; RMSE is descriptive, not a confidence
interval.

Lower standalone baseline RMSE does **not** automatically imply better
WM4 model performance: a flexible baseline may absorb dynamics that
the WM4 residual model is intended to identify.

## Outputs

Default directory:
`files/output/data/baseline_diagnostics/{med,all}/`

Key tables:
- `weekday_hour_profile.csv`
- `consumption_acf.csv`
- `corresponding_similarity.csv`
- `dependence.json`
- `adjusted_residuals.csv`
- `residual_{time,temperature,temperature_price}_acf.csv`
- `baseline_origin_scores.csv`
- `baseline_hourly_predictions.csv`
- `baseline_summary.csv`
- `summary.json`

Figures:
`weekday_profiles.png`, `consumption_acf.png`,
`corresponding_similarity.png`, `daily_temperature.png`,
`temperature_partial.png`, `price_partial.png`,
`adjusted_residual_acf.png`, `baseline_rmse.png`, and
`baseline_bias.png`.

## Tests

```bash
python -m unittest discover -s tests -p "test_baseline_diagnostics.py"
```
