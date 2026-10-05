"""G0 data contract: raw aggregates, train-only scales, common WM4 baseline."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.exe.run_model.wm_3.fit_weekly import consumption_series, price_series, temperature_series
from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.import_data.import_weather import weather_data


def load_raw_frame(data_directory, start, end, tag='MED'):
    directory = Path(data_directory)
    # observed_data retains missing observations: never use the importer's
    # season-wide imputation or WM's season-wide min/max for this experiment.
    demand = consumption_series(input_data(directory/'input.xlsx'), tag)
    price = price_series(spot_price_data(directory/'prices.xlsx', price_area='DK1'))
    history_start = max(demand.index.min(),price.index.min())
    temp = temperature_series(weather_data(cache_dir=directory/'weather_cache'), history_start, end)
    return pd.concat((demand.rename('Demand'), price.rename('Price'), temp.rename('Temp')), axis=1).sort_index()


def raw_csv(path):
    frame = pd.read_csv(path, index_col='Datetime', parse_dates=['Datetime'])
    if set(('Demand', 'Price', 'Temp')) - set(frame):
        raise ValueError('CSV needs Datetime,Demand,Price,Temp in raw units.')
    return frame


def prepare_frame(raw, train_start, train_end, *, price_bounds=None):
    """No future values affect scales or baseline training; no clipping."""
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError('Require sorted unique DatetimeIndex.')
    if raw.index.tz is not None:
        raise ValueError('Declare local-naive Europe/Copenhagen timestamps for this January adapter.')
    if any(name not in raw for name in ('Demand', 'Price', 'Temp')):
        raise ValueError('Missing Demand/Price/Temp columns.')
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    if end <= start or any(v.minute or v.second or v.microsecond for v in (start, end)):
        raise ValueError('Training window needs ordered hour edges.')
    train = raw.reindex(pd.date_range(start, end, freq='h', inclusive='left'))
    if len(train) < 168 or not np.all(np.isfinite(train[['Demand','Price','Temp']].to_numpy(dtype=float))):
        raise ValueError('At least 168 complete hourly training observations are required; gaps are not imputed.')
    scales = {'demand_min': float(train.Demand.min()), 'demand_scale': float(train.Demand.max()-train.Demand.min()),
              'price_min': float(train.Price.min()), 'price_scale': float(train.Price.max()-train.Price.min())}
    if price_bounds is not None:
        lo,hi = map(float,price_bounds)
        if not np.isfinite(lo+hi) or lo >= hi or train.Price.min() < lo or train.Price.max() > hi:
            raise ValueError('Declared ex-ante price bounds must be finite, ordered and contain training prices.')
        scales['price_min'],scales['price_scale'] = lo,hi-lo
    scales['price_scale_source'] = 'training' if price_bounds is None else 'declared-ex-ante'
    if scales['demand_scale'] <= 0 or scales['price_scale'] <= 0:
        raise ValueError('Training demand/price must vary.')
    frame = pd.DataFrame(index=raw.index)
    frame['I'] = (raw.Demand-scales['demand_min'])/scales['demand_scale']
    frame['U'] = 2*(raw.Price-scales['price_min'])/scales['price_scale']-1
    frame['Temp'] = raw.Temp
    return frame, scales


def common_inputs(frame, baseline_model, start, end, *, observations=False, steps_per_hour=60):
    """Common fine baseline and held prices, without requiring future I."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hours = (end-start).total_seconds()/3600
    if hours < 1 or int(hours) != hours or int(steps_per_hour) != steps_per_hour or steps_per_hour < 2:
        raise ValueError('Require integer hours and at least two fine samples/hour.')
    hours, steps_per_hour = int(hours), int(steps_per_hour)
    index = pd.date_range(start, end, freq='h', inclusive='left')
    index.tz_localize('Europe/Copenhagen',ambiguous='raise',nonexistent='raise')
    if start.tz_localize('Europe/Copenhagen').utcoffset() != end.tz_localize('Europe/Copenhagen').utcoffset():
        raise ValueError('This January adapter does not yet support a DST-transition window.')
    columns = ['U','Temp'] + (['I'] if observations else [])
    source = frame.reindex(index)[columns]
    if not np.all(np.isfinite(source.to_numpy(dtype=float))):
        raise ValueError(f'Incomplete declared inputs/observations for {start} to {end}.')
    if np.any(np.abs(source.U.to_numpy()) > 1):
        raise ValueError('Price outside training domain: declare a common ex-ante scale; never clip or fit it on test data.')
    elapsed = np.arange(hours*steps_per_hour+1, dtype=float)/steps_per_hour
    fine_index = start + pd.to_timedelta(elapsed, unit='h')
    # Endpoint: hold the final known sample. Training never reads a test demand.
    temp = np.interp(elapsed, np.arange(hours), source.Temp.to_numpy(dtype=float))
    raw_baseline = baseline_model._standardize(baseline_model._features(fine_index, temp)) @ baseline_model.coef
    baseline = baseline_model.predict(fine_index, temp)
    price_hour = source.U.to_numpy(dtype=float)
    price = price_hour[np.minimum(np.floor(elapsed+1e-12).astype(int), hours-1)]
    result = {'index': index, 'fine_index': fine_index, 'time': elapsed, 'baseline': baseline,
              'price_hour': price_hour, 'price': price,
              'baseline_clipped_samples': int(np.count_nonzero((raw_baseline < 0) | (raw_baseline > 1)))}
    if observations:
        result['observed'] = source.I.to_numpy(dtype=float)
        result['fine_observed'] = np.interp(elapsed, np.arange(hours), result['observed'])
    return result
