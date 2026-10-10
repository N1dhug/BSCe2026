"""BM hourly data preparation, using WM's shared consumption/price importers.

Consumption is the mean of observed meters in the selected group. By default,
baseline is fitted by WM4's statistical_output; a raw baseline CSV is optional.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.import_data.import_weather import weather_data
from src.exe.run_model.wm_4.output_model import statistical_output


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'files' / 'input' / 'data'
ZONE = 'Europe/Copenhagen'


def _timestamp(value):
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(ZONE) if stamp.tzinfo is None else stamp.tz_convert(ZONE)
    if pd.isna(stamp) or stamp != stamp.floor('h', ambiguous=bool(stamp.dst())):
        raise ValueError('Start/end must be valid hour edges.')
    return stamp


def _window(series, start, end, *, include_end=False):
    """Localize only the requested window; never merge repeated local hours."""
    series = series.copy()
    if series.index.tz is None:
        left, right = start.tz_localize(None), end.tz_localize(None)
        selected = (series.index >= left) & (series.index <= right if include_end else series.index < right)
        series = series.loc[selected]
        series.index = series.index.tz_localize(ZONE, ambiguous='raise', nonexistent='raise')
    else:
        series.index = series.index.tz_convert(ZONE)
        selected = (series.index >= start) & (series.index <= end if include_end else series.index < end)
        series = series.loc[selected]
    if series.index.has_duplicates:
        raise ValueError('Duplicate timestamps in the selected BM data.')
    return series.sort_index()


def _observed_demand(input_path, tag):
    source = input_data(input_path)
    columns = [c for c in source.measurement_columns if source.measurement_tag(c).upper() == tag.strip().upper()]
    if not columns:
        raise ValueError(f'No consumption measurements for group {tag}.')
    return source.observed_data[columns].mean(axis=1)


def aligned_bm_data(start, end, *, tag='MED', input_path=DATA/'input.xlsx',
                    prices_path=DATA/'prices.xlsx', price_area='DK1', demand_label='hour_start'):
    """Return raw Demand/Price on every physical hour in [start,end).

    Follows WM's group-mean convention using observed_data, not imputed meters.
    Missing demand stays missing; all hourly prices are required. End-labelled
    demand is shifted back one physical hour before alignment.
    """
    start, end = _timestamp(start), _timestamp(end)
    if end <= start or demand_label not in ('hour_start', 'hour_end'):
        raise ValueError('Use ordered start/end and hour_start or hour_end demand labels.')
    demand = _observed_demand(input_path, tag)
    shift = pd.Timedelta(hours=1) if demand_label == 'hour_end' else pd.Timedelta(0)
    demand = _window(demand, start+shift, end+shift)
    demand.index = demand.index-shift
    prices = spot_price_data(prices_path, price_area=price_area).prices('DKK')
    prices = _window(prices, start, end)
    index = pd.date_range(start, end, freq='h', inclusive='left')
    frame = pd.DataFrame({'Demand': demand.reindex(index), 'Price': prices.reindex(index)})
    if not np.isfinite(frame.Price.to_numpy()).all():
        raise ValueError('The selected BM window needs complete hourly prices.')
    frame.attrs.update(group=tag, price_area=price_area, price_currency='DKK',
                       demand_aggregation='mean of available observed meters', demand_label=demand_label,
                       input_workbook=str(input_path))
    return frame


def fit_inputs(frame, baseline_path=None, *, demand_offset=None, demand_scale=None,
               price_offset=None, price_scale=None, steps_per_hour=12):
    """Prepare normalized inputs and baseline, without fitting BM parameters.

    Without a CSV, fit WM4's unchanged statistical_output on observed demand
    and temperature before the window end, then evaluate on a fine grid.
    Automatic demand scales use that history; price scales use the fit window.
    A supplied raw CSV retains the window-based normalization and hourly edges.
    Demand/baseline always share a scale; explicit scales override defaults.
    """
    start, end = frame.index[0], frame.index[-1]+pd.Timedelta(hours=1)
    edges = pd.date_range(start, end, freq='h')
    demand = frame.Demand.to_numpy(dtype=float)
    price = frame.Price.to_numpy(dtype=float)
    measured = np.isfinite(demand)
    if not measured.any():
        raise ValueError('Measured demand is required.')
    if baseline_path is None:
        history = _observed_demand(frame.attrs['input_workbook'], frame.attrs['group']).dropna()
        shift = pd.Timedelta(hours=1) if frame.attrs['demand_label'] == 'hour_end' else pd.Timedelta(0)
        history = _window(history, _timestamp(history.index.min()), end+shift)
        history.index = history.index-shift
        reference = history.to_numpy(dtype=float)
    else:
        table = pd.read_csv(baseline_path, index_col='Datetime', parse_dates=['Datetime'])
        baseline = _window(pd.to_numeric(table['baseline'], errors='coerce'), start, end, include_end=True)
        baseline = baseline.reindex(edges).to_numpy(dtype=float)
        if not np.isfinite(baseline).all():
            raise ValueError('A finite baseline at every hour edge is required.')
        reference = np.r_[demand[measured], baseline]
    d0 = float(reference.min()) if demand_offset is None else float(demand_offset)
    ds = float(reference.max()-d0) if demand_scale is None else float(demand_scale)
    p0 = float(price.min()) if price_offset is None else float(price_offset)
    ps = float(price.max()-p0) if price_scale is None else float(price_scale)
    if not np.isfinite([d0, ds, p0, ps]).all() or ds <= 0 or ps <= 0:
        raise ValueError('Normalization needs finite offsets and positive scales; supply scales for constant signals.')
    demand, price = (demand-d0)/ds, (price-p0)/ps
    baseline_metadata = {}
    if baseline_path is None:
        if isinstance(steps_per_hour, bool) or int(steps_per_hour) != steps_per_hour or steps_per_hour < 1:
            raise ValueError('steps_per_hour must be a positive integer.')
        # Fetch one extra boundary temperature; no demand at/after end is fitted.
        temperature = weather_data().fetch_hourly(history.index[0], end+pd.Timedelta(hours=1))['Temperature']
        train = pd.DataFrame({'I': (history-d0)/ds, 'Temp': temperature.reindex(history.index)})
        model = statistical_output().fit(train, end)
        baseline_time = np.arange(len(frame)*int(steps_per_hour)+1, dtype=float)/int(steps_per_hour)
        fine_index = start+pd.to_timedelta(baseline_time, unit='h')
        hourly_temp = temperature.reindex(edges).to_numpy(dtype=float)
        baseline = model.predict(fine_index, np.interp(baseline_time, np.arange(len(edges)), hourly_temp))
        baseline_metadata = {
            'baseline_fit': {key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                             for key, value in model.metadata.items()},
            'baseline_cutoff': end.isoformat(), 'baseline_steps_per_hour': int(steps_per_hour),
        }
    else:
        baseline_time = np.arange(len(edges), dtype=float)
        baseline = (baseline-d0)/ds
    if any(np.any((values < 0) | (values > 1)) for values in (demand[measured], baseline, price)):
        raise ValueError('Normalized BM inputs must lie in [0,1]; no clipping.')
    return {
        'baseline_time': baseline_time, 'baseline': baseline,
        'prices': price, 'demand': demand, 'observation_mask': measured,
        'normalization': {'demand_offset': d0, 'demand_scale': ds,
                          'price_offset': p0, 'price_scale': ps},
        'metadata': {**frame.attrs, 'start': start.isoformat(), 'end_exclusive': end.isoformat(),
                     'hours': len(frame), 'observed_hours': int(measured.sum()),
                     'baseline_source': str(baseline_path) if baseline_path is not None else 'wm_4.output_model.statistical_output',
                     'baseline_fitted': baseline_path is None, **baseline_metadata,
                     'normalization_policy': 'selected window or explicit scales' if baseline_path is not None else
                                             'demand history before end; selected-window price; or explicit scales',
                     'time_unit': 'physical elapsed hours'},
    }
