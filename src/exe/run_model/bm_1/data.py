"""BM hourly data preparation, using WM's shared consumption/price importers.

Consumption is the mean of observed meters in the selected group. Baseline
is supplied separately in the same raw units. No temperature or baseline fit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data


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
    source = input_data(input_path)
    columns = [c for c in source.measurement_columns if source.measurement_tag(c).upper() == tag.strip().upper()]
    if not columns:
        raise ValueError(f'No consumption measurements for group {tag}.')
    demand = source.observed_data[columns].mean(axis=1)
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
                       demand_aggregation='mean of available observed meters', demand_label=demand_label)
    return frame


def fit_inputs(frame, baseline_path, *, demand_offset=None, demand_scale=None,
               price_offset=None, price_scale=None):
    """Prepare normalized arguments for fit_known_inputs, without fitting.

    Baseline CSV has Datetime,baseline at every hour edge including the final
    endpoint. Baseline is piecewise linear. Automatic scales use this window
    only; demand/baseline share a scale. Explicit scales override these defaults.
    """
    start, end = frame.index[0], frame.index[-1]+pd.Timedelta(hours=1)
    edges = pd.date_range(start, end, freq='h')
    table = pd.read_csv(baseline_path, index_col='Datetime', parse_dates=['Datetime'])
    baseline = _window(pd.to_numeric(table['baseline'], errors='coerce'), start, end, include_end=True)
    baseline = baseline.reindex(edges).to_numpy(dtype=float)
    demand = frame.Demand.to_numpy(dtype=float)
    price = frame.Price.to_numpy(dtype=float)
    measured = np.isfinite(demand)
    if not measured.any() or not np.isfinite(baseline).all():
        raise ValueError('Measured demand and a finite baseline at every hour edge are required.')
    reference = np.r_[demand[measured], baseline]
    d0 = float(reference.min()) if demand_offset is None else float(demand_offset)
    ds = float(reference.max()-d0) if demand_scale is None else float(demand_scale)
    p0 = float(price.min()) if price_offset is None else float(price_offset)
    ps = float(price.max()-p0) if price_scale is None else float(price_scale)
    if not np.isfinite([d0, ds, p0, ps]).all() or ds <= 0 or ps <= 0:
        raise ValueError('Normalization needs finite offsets and positive scales; supply scales for constant signals.')
    demand, baseline, price = (demand-d0)/ds, (baseline-d0)/ds, (price-p0)/ps
    if any(np.any((values < 0) | (values > 1)) for values in (demand[measured], baseline, price)):
        raise ValueError('Normalized BM inputs must lie in [0,1]; no clipping.')
    return {
        'baseline_time': np.arange(len(edges), dtype=float), 'baseline': baseline,
        'prices': price, 'demand': demand, 'observation_mask': measured,
        'normalization': {'demand_offset': d0, 'demand_scale': ds,
                          'price_offset': p0, 'price_scale': ps},
        'metadata': {**frame.attrs, 'start': start.isoformat(), 'end_exclusive': end.isoformat(),
                     'hours': len(frame), 'observed_hours': int(measured.sum()),
                     'baseline_source': str(baseline_path), 'baseline_fitted': False,
                     'normalization_policy': 'selected window or explicit scales',
                     'time_unit': 'physical elapsed hours'},
    }
