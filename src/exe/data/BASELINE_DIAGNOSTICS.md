# Preliminary statistical analyses

Executable: `src/exe/data/baseline_diagnostics.py`.

The filename is retained for existing commands, but the executable now
runs **only four exploratory statistical analyses**. It does not fit or
compare baselines, evaluate rolling forecasts, or optimize WM4 states.

## Usage

```bash
python -m src.exe.data.baseline_diagnostics --groups MED ALL
```

To focus on one group and a selected date range:

```bash
python -m src.exe.data.baseline_diagnostics --groups MED --start 2022-03-01 --end 2022-11-01 --max-week-lag 8 --max-acf-lag 336
```

`--start` is inclusive; `--end` is exclusive. Optional arguments are
`--base-temperature 18`, `--hac-lags 24`, `--max-week-lag 8`,
`--max-acf-lag 336`, and `--output path`.

Python usage:

```python
from src.exe.data.baseline_diagnostics import run
result=run(groups=("MED","ALL"),max_weeks=8,max_acf_lag=336)
```

The script uses the same aligned normalized aggregate demand `I`,
normalized spot price `U` and physical weather temperature `Temp`
as the existing WM4 executable. No additional normalization or
full-day imputation is introduced by this script.

## Analyses

1. **Temporal consumption patterns.** Plot the average hourly
   consumption profile for every weekday, and compute the raw demand
   autocorrelation at exact hourly lags (including 24 and 168 hours).
   The code uses explicit named `weekday` and `hour` columns rather
   than inherited pandas group names, avoiding collisions when the
   source DatetimeIndex is called `Datetime`.

2. **Corresponding-interval similarity.** For weekly offsets
   `n=1,...,max_week_lag`, calculate the mean-squared differences
   between demand at the same weekday/hour, compared with historical
   observations one weekday earlier/later. Each reported triplet uses
   an identical set of available timestamps.

3. **Temperature dependence.** Remove weekday/hour fixed effects
   from normalized demand and from the heating-degree variable
   `max(0,base_temperature-Temp)`. Estimate their conditional
   association using least squares. Record coefficients, residual MSE,
   incremental MSE improvement and approximate Bartlett/Newey-West
   HAC normal p-values. Plot daily chill/consumption and the hourly
   fixed-effect-adjusted partial relationship.

4. **Price dependence.** Jointly regress fixed-effect-adjusted demand
   on chill and normalized price. The price partial-regression plot
   first adjusts for weekday/hour and chill effects. Record coefficient,
   incremental MSE improvement and HAC approximate p-value.

Residual ACFs after removing weekday/hour effects, then temperature,
then price, are saved as **diagnostics supporting these four analyses**.

Price coefficients are *conditional associations*, not causal
estimates of consumer flexibility. Statistical significance alone
does not demonstrate that either input improves out-of-sample
prediction. Baseline performance comparisons are intentionally
outside the scope of this script.

## Terminal summaries

Each run now prints the 24-hour and 168-hour raw and adjusted
autocorrelations, matched-sample historical RMSE for each weekly lag,
and the goodness-of-fit of **all three nested descriptive regressions**:

- Weekday/hour fixed effects only.
- Weekday/hour fixed effects plus heating-degree factor.
- Weekday/hour fixed effects plus heating-degree factor and price.

For each fit it reports the sample count, MSE, RMSE, MAE and R²
relative to the unconditional demand mean. It also prints the
temperature coefficient (with and without price adjustment),
the conditional price coefficient, approximate HAC p-values and
incremental in-sample MSE improvements.

The same regression fit metrics are written to
`dependence.json` and `summary.json` under `regression.fit_quality`.
These are **in-sample explanatory fits**, not operational forecasts
or out-of-sample prediction quality.

## Outputs

Default directory:
`files/output/data/preliminary_statistics/med/` and
`files/output/data/preliminary_statistics/all/`.

Main outputs:

- `weekday_hour_profile.csv`, `weekday_profiles.png`
- `consumption_acf.csv`, `consumption_acf.png`
- `corresponding_similarity.csv`, `corresponding_similarity.png`
- `daily_temperature.csv`, `daily_temperature.png`
- `dependence.json`
- `temperature_partial.png`, `price_partial.png`
- `adjusted_residuals.csv`
- `residual_time_acf.csv`, `residual_temperature_acf.csv`,
  `residual_temperature_price_acf.csv`,
  `adjusted_residual_acf.png`
- `summary.json` with statistical settings and regression results

All results describe observed 2022 data (or the provided selected
period), without using oracle predictions.

## Tests

```bash
python -m unittest discover -s tests -p "test_baseline_diagnostics.py"
```

Regression tests include a DatetimeIndex named `Datetime` to cover
the prior pandas `ValueError: cannot insert Datetime, already exists`.
