"""Raw G CSV input and historical-only normalization; no benchmark or workbook driver."""
import numpy as np
import pandas as pd
from src.tools.import_data.import_input import interpolate_time_gaps

def raw_csv(path, *, require_demand=True):
    frame = pd.read_csv(path, index_col='Datetime', parse_dates=['Datetime'])
    required = ('Demand', 'Price', 'Temp') if require_demand else ('Price', 'Temp')
    if set(required) - set(frame):
        raise ValueError('CSV needs Datetime,'+','.join(required)+' in raw units.')
    return frame

def training_data_audit(raw, train_start, train_end):
    """Explain raw coverage before validation, scaling or optimization."""
    start,end = pd.Timestamp(train_start),pd.Timestamp(train_end)
    if not isinstance(raw.index,pd.DatetimeIndex) or raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError('Require sorted unique DatetimeIndex.')
    if raw.index.tz is not None:
        raise ValueError('Declare local-naive Europe/Copenhagen timestamps for this January adapter.')
    names = ('Demand','Price','Temp')
    if any(name not in raw for name in names):
        raise ValueError('Missing Demand/Price/Temp columns.')
    if end <= start or any(v.minute or v.second or v.microsecond for v in (start,end)):
        raise ValueError('Training window needs ordered hour edges.')
    index = pd.date_range(start,end,freq='h',inclusive='left')
    data = raw.loc[:,list(names)].apply(pd.to_numeric,errors='coerce')
    train = data.reindex(index)
    finite = np.isfinite(train.to_numpy(dtype=float))
    columns = {}
    for i,name in enumerate(names):
        available = data.index[np.isfinite(data[name].to_numpy(dtype=float))]
        missing = index[~finite[:,i]]
        columns[name] = {'valid_hours':int(finite[:,i].sum()), 'missing_hours':len(missing),
                         'first_missing':[str(t) for t in missing[:8]],
                         'available_start':str(available.min()) if len(available) else None,
                         'available_end':str(available.max()) if len(available) else None}
    off_grid = raw.index[(raw.index >= start)&(raw.index < end)&(raw.index != raw.index.floor('h'))]
    return {'training_start':str(start),'training_end_exclusive':str(end),
            'expected_hours':len(index),'complete_hours':int(finite.all(axis=1).sum()),
            'raw_complete':bool(finite.all()),
            'fit_ready':bool(finite[:,0].sum()>=168 and finite[:,1].all() and finite[:,2].any() and not len(off_grid)),
            'gap_policy':'Training-only time interpolation and boundary fill; demand fit/scoring use measured hours only; prices must be complete.',
            'columns':columns,
            'off_hour_timestamps':len(off_grid),'first_off_hour':[str(t) for t in off_grid[:8]]}

def training_audit_text(audit):
    parts = [f"TRAIN data: {audit['complete_hours']}/{audit['expected_hours']} complete hours; fit_ready={audit['fit_ready']}"]
    for name,info in audit['columns'].items():
        parts.append(f"{name}: missing={info['missing_hours']}; first={info['first_missing']}; available={info['available_start']} -> {info['available_end']}")
    if audit['off_hour_timestamps']:
        parts.append(f"Off-hour timestamps={audit['off_hour_timestamps']}; first={audit['first_off_hour']}")
    if not audit['raw_complete']:
        parts.append('TRAIN gap policy: interpolate Demand/Temp within training only; fit/score measured demand; no price filling here.')
    return '\n'.join(parts)

def prepare_frame(raw, train_start, train_end, *, price_bounds=None):
    """No future values affect scales or baseline training; no clipping."""
    audit = training_data_audit(raw,train_start,train_end)
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    train = raw.reindex(pd.date_range(start, end, freq='h', inclusive='left')).copy()
    if not audit['fit_ready']:
        raise ValueError('Training needs at least 168 measured demand hours, complete prices, some temperature data and hourly timestamps.\n'+training_audit_text(audit))
    train = train.apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    temperature_missing = train.Temp.isna()
    train['Temp'] = interpolate_time_gaps(train.Temp)
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
    # Keep missing observations missing for baseline fitting and measured-only scores.
    # Include absent training timestamps so both forward models propagate every hour.
    frame = pd.DataFrame(index=raw.index.union(train.index).sort_values())
    frame['I'] = (raw.Demand-scales['demand_min'])/scales['demand_scale']
    frame['U'] = 2*(raw.Price-scales['price_min'])/scales['price_scale']-1
    frame['Temp'] = raw.Temp
    frame.loc[train.index,'I'] = (train.Demand-scales['demand_min'])/scales['demand_scale']
    frame.loc[train.index,'Temp'] = train.Temp
    frame['temperature_imputed'] = False
    frame.loc[train.index,'temperature_imputed'] = temperature_missing
    frame.attrs['gap_audit'] = audit
    return frame, scales
