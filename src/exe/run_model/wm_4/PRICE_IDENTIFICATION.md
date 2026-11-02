# When does WM4 identify a price coefficient?

This experiment investigates the observation that WM4 N=8/H=7 oracle
forecasts tend to fit better when theta_U is nonzero. It tests whether
historical price levels or price variation predict a detectable
negative fitted theta_U.

Run after the existing N=8/H=7 study. No parameter refitting is done:

    python -m src.exe.run_model.wm_4.analyze_price_identification

The tool reads previously saved daily_metrics.csv and
hourly_predictions.csv from the default study location:

    files/output/wm_4/oracle_8weekday_7hour/med/weekly_temperature/

The study also reads historical normalized hourly prices through
group_frame("MED"). It outputs five figures and four CSV tables to
the price_identification subfolder. Alternative source and output
directories are available with --input and --output; choose --group ALL
when analyzing the ALL consumer group.

## Separation of historical and target price signals

theta_U is estimated from eight **preceding matching weekdays**.
Only the normalized U data from those historical dates are used in
the correlation tests, never the prices from the seven-hour
prediction day. Single missing historical hours are imputed within
their original day. Entire missing historical days are not imputed.

For each training window, calculate signed price mean, mean absolute
normalized price, pooled standard deviation and range, within-day
standard deviations, within-day mean absolute hourly differences,
and the fraction of nonzero intra-day price changes. Within-day
changes do NOT jump between different historical dates.

Seven-hour target price level and volatility are shown separately
to contextualize prediction error and realized price forcing.
Comparing realized RMS(theta_U * U) with target |U| is partly
mechanical and is not evidence of a price response.

## Tests and outputs

A detectable nonzero coefficient is defined as theta_U < -1e-6.
Near-zero solutions (|theta_U| <= 1e-6) are explicitly distinguished
from positive coefficients beyond numerical tolerance.

Ten prespecified Spearman associations test five training-price
variables against (a) binary price-active status and (b) the
magnitude of the negative fitted theta_U.

Overlapping eight-week historical fitting windows create serial
dependence. Results include paired moving-block bootstrap 95%
confidence intervals, approximate bootstrap-standard-error
p-values, and Benjamini-Hochberg adjusted q-values across the
10 tests. Default block size is 7 successive eligible evaluation
records with 1000 bootstrap draws. Since evaluation dates can
have gaps and price dynamics can be nonstationary, these
intervals and p/q-values are exploratory approximations, not
definitive hypothesis tests.

Saved tables:
- daily_price_identification.csv
- historical_price_associations.csv
- price_volatility_quartiles.csv
- error_quartiles_price_activity.csv

Saved plots:
- historical_price_vs_coefficient.png
- price_activation_by_quantiles.png
- historical_vs_forecast_price_volatility.png
- price_identification_and_error.png
- price_activity_association_intervals.png

Run focused tests:

    python -m unittest discover -s tests -p "test_wm4_price_identification.py"

A correlation between fitted theta_U and training-price volatility
would support the hypothesis that sufficiently informative price
variation helps identify price response, but would NOT establish
that prices causally change consumption. A stronger follow-up
would fit an otherwise comparable nested model with theta_U=0,
then assess any incremental held-out prediction improvement.

All price levels reported here are normalized U, not spot prices
in physical currency units. Low signed U must not be confused
with an independently established low DKK spot-price threshold.
