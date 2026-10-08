"""Standalone G-S3: fit 35 histories, save seven starts, forecast without refit.

The response/RK4/optimizer kernels are the original S3 implementation.
Forecast days are independent, frozen at a Monday cutoff, not a continuous
168-hour state trajectory. The supported observation operator is hourly-point.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.deterministic_strategies import calendar_inputs, ZONE
from src.exe.run_model.bm_1.deterministic_weekday import metrics
from src.exe.run_model.wm_4.output_model import statistical_output
from src.method.bm_1_deterministic_fit import _parameters, BudgetStop
from src.method.bm_1_deterministic_strategy import fit_strategy
from src.method.bm_1_deterministic_weekday import BM1PooledInputs
from src.params.bm_1_response import BM1PiecewiseResponse
from src.tools.bm_1_deterministic_data import prepare_frame
from src.tools.bm_1_deterministic_snapshot import baseline_settings, restore_baseline


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'G-S3-five-weeks-Monday-frozen-v1'
TOLERANCE = 1e-5
SOURCES = (
    'src/method/bm_1_s3_forecast.py',
    'src/method/bm_1_deterministic_fit.py',
    'src/method/bm_1_deterministic_response.py',
    'src/method/bm_1_deterministic_weekday.py',
    'src/method/bm_1_deterministic_strategy.py',
    'src/exe/run_model/bm_1/deterministic_strategies.py',
    'src/exe/run_model/wm_4/output_model.py',
    'src/tools/bm_1_deterministic_data.py',
    'src/tools/bm_1_deterministic_snapshot.py',
    'src/params/bm_1_response.py',
)


def source_hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for p in SOURCES}


def _plain(value):
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, pd.Timestamp): return value.isoformat()
    return value


def _origin(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tz is not None or t != t.normalize() or t.weekday() != 0:
        raise ValueError('S3 cutoff must be a local-naive Monday midnight.')
    return t


def _frame(raw, columns):
    if (not isinstance(raw, pd.DataFrame) or not isinstance(raw.index, pd.DatetimeIndex)
            or raw.index.tz is not None or raw.index.has_duplicates
            or not raw.index.is_monotonic_increasing or raw.index.hasnans
            or not set(columns).issubset(raw)):
        raise ValueError('Require sorted unique local-naive DatetimeIndex and '+','.join(columns)+'.')
    if np.any(raw.index != raw.index.floor('h')):
        raise ValueError('S3 inputs must be hourly.')


def _day_inputs(frame, baseline, day, *, history=False, steps=12):
    # calendar_inputs uses physical hours, but returns naive local timestamps.
    # Fall-back duplication cannot be represented unambiguously in that contract.
    begin = day.tz_localize(ZONE)
    if history: begin -= pd.Timedelta(hours=24)
    end = (day+pd.Timedelta(days=1)).tz_localize(ZONE)
    local = pd.date_range(begin, end, freq='h', inclusive='left').tz_localize(None)
    if local.has_duplicates:
        raise ValueError('Autumn DST with repeated local hours is unsupported by S3 v1; supply another window.')
    return calendar_inputs(frame, baseline, day, history=history, steps=steps)


def _fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def validate_snapshot(snapshot):
    """Fail closed on incompatible fitted data; no fits or source migrations."""
    try:
        s = deepcopy(snapshot)
        if (type(s['schema_version']) is not int or s['schema_version'] != 1
                or s['kind'] != 'G-S3-fit' or s['protocol'] != PROTOCOL):
            raise ValueError('Unsupported S3 snapshot version/protocol.')
        origin = _origin(s['origin'])
        if s['source_sha256'] != source_hashes():
            raise ValueError('S3 numerical sources changed; an explicit reviewed migration is required.')
        if (s['time_unit'] != 'hours' or s['timezone'] != ZONE
                or s['observation_contract'] != 'hourly-point-development-approximation'
                or s['lambda'] != 0 or s['fit']['status'] != 'converged'):
            raise ValueError('Snapshot must be a converged S3 lambda=0 fit with declared units.')
        p = _parameters(s['parameters'])
        if p[3] != .5 or not 1 <= p[0] <= 744 or not .05 <= p[1] <= 10:
            raise ValueError('Snapshot parameters disagree with S3 fit bounds/neutral history start.')
        curve = BM1PiecewiseResponse.from_record(s['price_response'])
        if not np.array_equal(curve.knots, [0., .5, 1.]):
            raise ValueError('S3 needs the selected two-piece price response.')
        best = s['fit']['best_converged']
        if not np.array_equal(p, best['parameters']) or curve.values[1] != best['g_mid']:
            raise ValueError('Saved fit and response parameters disagree.')
        n = s['numerics']
        steps = n['steps_per_hour']
        if type(steps) is not int or steps not in (12, 24, 48) or n['fine_steps_per_hour'] != 2*steps:
            raise ValueError('Invalid S3 coarse/fine grids.')
        if n['tolerance'] != TOLERANCE or not n['passed'] or not 0 <= n['history_max_demand_error'] <= TOLERANCE:
            raise ValueError('Snapshot requires the passed S3 historical refinement check.')
        for name in ('forecast_states', 'fine_forecast_states'):
            states = np.asarray(s[name], float)
            if states.shape != (7,) or not np.isfinite(states).all() or np.any((states < 0) | (states > 1)):
                raise ValueError('Snapshot requires seven own finite daily starts in [0,1].')
        history = s['history']
        expected = [[str((origin+pd.Timedelta(days=d)-pd.Timedelta(weeks=w)).date())
                     for w in range(1, 6)] for d in range(7)]
        if (history['days_by_weekday'] != expected or history['warmup_physical_hours'] != 24
                or history['start_state'] != .5 or history['forecast_start_rule'] != 'median of five own states after 24 physical hours'):
            raise ValueError('Snapshot history/initialization contract disagrees with S3.')
        masks = history['target_measured_masks']
        if len(masks) != 35 or any(len(mask) not in (23, 24) or any(type(v) is not bool for v in mask) for mask in masks):
            raise ValueError('Snapshot needs the 35 historical target masks.')
        if sum(sum(mask) for mask in masks) != s['fit']['observed_samples']:
            raise ValueError('Fit sample count and historical masks disagree.')
        scales = s['scales']
        v = np.array([scales[k] for k in ('demand_min', 'demand_scale', 'price_min', 'price_scale')], float)
        if not np.isfinite(v).all() or v[1] <= 0 or v[3] <= 0:
            raise ValueError('Invalid frozen scales.')
        a, b = map(pd.Timestamp, s['normalization_window'])
        if a.tz is not None or b.tz is not None or not a < b <= origin:
            raise ValueError('Normalization must use historical data only.')
        baseline = restore_baseline(s['baseline'])
        if pd.Timestamp(baseline.metadata['fit_end']) >= origin:
            raise ValueError('Baseline metadata crosses forecast cutoff.')
        if not isinstance(s['training_fingerprint'], str) or len(s['training_fingerprint']) != 64:
            raise ValueError('Missing historical input fingerprint.')
        return s
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Incomplete or malformed S3 snapshot.') from exc


def load_snapshot(path):
    with Path(path).open(encoding='utf-8') as f:
        return validate_snapshot(json.load(f))


def save_snapshot(path, snapshot):
    from src.tools.bm_1_reports import _atomic_report
    _atomic_report(Path(path), _plain(validate_snapshot(snapshot)))


def fit_s3(raw, origin, *, reference=(12., 2., .5, .5),
           calibration_start=None, calibration_end=None, price_bounds=(-500., 6000.),
           seconds=600., max_calls=3000, checkpoint=None,
           population='declared raw aggregate', demand_unit='original raw units; physical unit unverified'):
    """Fit one selected model. Future signals/observations are not required.

    All refits/grids share one budget. Historical demand is never imputed;
    baseline and temperature handling reuse G's existing data contract.
    """
    started = perf_counter()
    origin = _origin(origin)
    reference = _parameters(reference)
    if (not np.isfinite(seconds) or not 0 < seconds <= 600 or type(max_calls) is not int
            or not 1 <= max_calls <= 3000):
        raise ValueError('Declare a positive S3 budget <=600 seconds and <=3000 residual calls.')
    _frame(raw, ('Demand', 'Price', 'Temp'))
    train = raw.loc[raw.index < origin, ['Demand', 'Price', 'Temp']].copy()
    _frame(train, ('Demand', 'Price', 'Temp'))
    a = origin-pd.Timedelta(weeks=5) if calibration_start is None else pd.Timestamp(calibration_start)
    b = origin if calibration_end is None else pd.Timestamp(calibration_end)
    if not a < b <= origin: raise ValueError('Normalization must end at/before cutoff.')
    frame, scales = prepare_frame(train, a, b, price_bounds=price_bounds)
    # prepare_frame only fills Temp inside the declared normalization window.
    # Warm-up/earlier baseline inputs must therefore be present in the raw data.
    baseline = statistical_output().fit(frame[['I', 'Temp']], origin)
    histories, days = [], []
    for weekday in range(7):
        group = []
        for lag in range(1, 6):
            day = origin+pd.Timedelta(days=weekday)-pd.Timedelta(weeks=lag)
            inp, _ = _day_inputs(frame, baseline, day, history=True)
            histories.append(inp); group.append(str(day.date()))
        days.append(group)
    attempts, remaining, previous = [], max_calls, None
    for steps in (12, 24, 48):
        if remaining < 1 or perf_counter()-started >= seconds: break
        while histories[0].steps < steps:
            histories = [h.refined() for h in histories]
        pool = BM1PooledInputs(histories)
        fitted = fit_strategy(pool, reference, 'joint', 0., max_calls=remaining,
            calls_per_start=min(750, remaining), seconds=seconds-(perf_counter()-started),
            checkpoint=checkpoint, initial_records=[previous] if previous is not None else None)
        remaining -= fitted['counts']['residual_calls']
        attempts.append({'steps_per_hour': steps, 'fit': fitted})
        best = fitted['best_converged']
        if best is None: break
        curve = BM1PiecewiseResponse.from_inner_values([best['g_mid']], pieces=2)
        try:
            coarse = pool.solve(best['parameters'], price_response=curve, deadline=started+seconds)
            fine = pool.refined().solve(best['parameters'], price_response=curve, deadline=started+seconds)
        except BudgetStop:
            attempts[-1]['numerical_check_status'] = 'budget-stop'
            break
        error = float(np.max(np.abs(coarse['demand']-fine['demand'])))
        attempts[-1]['history_max_demand_error'] = error
        if error > TOLERANCE:
            previous = best
            continue
        starts = [float(np.median(coarse['historical_start_states'][5*d:5*d+5])) for d in range(7)]
        fine_starts = [float(np.median(fine['historical_start_states'][5*d:5*d+5])) for d in range(7)]
        baseline_record = _plain({'implementation':'statistical_output', 'settings':baseline_settings(baseline),
            'coef':baseline.coef, 'feature_mean':baseline.feature_mean,
            'feature_scale':baseline.feature_scale, 'metadata':baseline.metadata})
        snapshot = dict(schema_version=1, kind='G-S3-fit', protocol=PROTOCOL,
            origin=str(origin), time_unit='hours', timezone=ZONE, lambda_=0,
            observation_contract='hourly-point-development-approximation',
            parameters=best['parameters'], price_response=curve.to_record(),
            forecast_states=starts, fine_forecast_states=fine_starts, baseline=baseline_record,
            scales=scales, normalization_window=[str(a),str(b)],
            population=population, demand_unit=demand_unit,
            training_fingerprint=_fingerprint(train), source_sha256=source_hashes(),
            history={'days_by_weekday':days, 'warmup_physical_hours':24, 'start_state':.5,
                'forecast_start_rule':'median of five own states after 24 physical hours',
                'target_measured_masks':[h.observation_mask[24:].tolist() for h in histories]},
            fit=fitted, numerics={'steps_per_hour':steps, 'fine_steps_per_hour':2*steps,
                'tolerance':TOLERANCE, 'history_max_demand_error':error, 'passed':True})
        snapshot['lambda'] = snapshot.pop('lambda_')
        return {'status':'complete', 'snapshot':validate_snapshot(snapshot), 'attempts':attempts,
            'seconds':perf_counter()-started, 'budget':{'seconds':seconds,'max_calls':max_calls},
            'calls_used':max_calls-remaining}
    return {'status':'unfinished', 'snapshot':None, 'attempts':attempts,
        'seconds':perf_counter()-started, 'budget':{'seconds':seconds,'max_calls':max_calls},
        'calls_used':max_calls-remaining}


def predict_s3(snapshot, raw_inputs):
    """One week of seven independent frozen daily forecasts. No optimizer calls."""
    started = perf_counter()
    s = validate_snapshot(snapshot)
    origin = _origin(s['origin']); end = origin+pd.Timedelta(days=7)
    _frame(raw_inputs, ('Price','Temp'))
    scenario = raw_inputs.loc[(raw_inputs.index >= origin)&(raw_inputs.index < end), ['Price','Temp']].copy()
    _frame(scenario, ('Price','Temp'))
    if not np.isfinite(scenario.to_numpy(dtype=float)).all():
        raise ValueError('Scenario price and temperature must be finite.')
    scales = s['scales']
    frame = pd.DataFrame({'U':2*(scenario.Price-scales['price_min'])/scales['price_scale']-1,
                          'Temp':scenario.Temp}, index=scenario.index)
    baseline = restore_baseline(s['baseline'])
    curve = BM1PiecewiseResponse.from_record(s['price_response'])
    rows, errors = [], []
    for d in range(7):
        day = origin+pd.Timedelta(days=d)
        inp, index = _day_inputs(frame, baseline, day, steps=s['numerics']['steps_per_hour'])
        p = list(s['parameters']); p[3] = s['forecast_states'][d]
        q = p.copy(); q[3] = s['fine_forecast_states'][d]
        coarse = inp.solve(p, price_response=curve)
        fine = inp.refined().solve(q, price_response=curve)
        errors.append(float(np.max(np.abs(coarse['demand']-fine['demand']))))
        rows.append(pd.DataFrame({'prediction':coarse['demand'],
            'baseline':np.interp(np.arange(len(index)), inp.time, inp.baseline),
            'state':coarse['state'][:-1], 'price_normalized':inp.prices,
            'price':scenario.reindex(index).Price.to_numpy(), 'weekday':d,
            'horizon_hours':np.arange(len(index)), 'cutoff':str(origin), 'protocol':PROTOCOL}, index=index))
    normalized = pd.concat(rows)
    normalized['elapsed_hours'] = np.arange(len(normalized))
    normalized['time_utc'] = normalized.index.tz_localize(ZONE).tz_convert('UTC').astype(str)
    values = normalized.copy()
    for col in ('prediction','baseline'):
        values[col] = scales['demand_min']+scales['demand_scale']*values[col]
    passed = max(errors) <= TOLERANCE
    return {'status':'complete' if passed else 'numerics-failed', 'predictions':values,
        'normalized_predictions':normalized, 'manifest':{'protocol':PROTOCOL, 'cutoff':str(origin),
            'end_exclusive':str(end), 'timezone':ZONE, 'physical_hours':len(values),
            'scales':deepcopy(scales), 'population':s['population'], 'demand_unit':s['demand_unit'],
            'observation_contract':s['observation_contract'], 'scenario_fingerprint':_fingerprint(scenario),
            'demand_used':False, 'baseline_refitted':False, 'model_refitted':False,
            'scenario':'conditional on declared future price/temperature, not automatically a weather forecast',
            'state_path':'seven independent daily paths frozen Monday', 'stochastic_status':'not-implemented'},
        'numerics':{'passed':passed,'tolerance':TOLERANCE,'daily_max_demand_errors':errors},
        'scoring':{'status':'not-scored'}, 'seconds':perf_counter()-started}


def score_s3(forecast, observations):
    """Measured-only raw-unit scores, attached after forecasting; no mutation."""
    if not isinstance(observations, pd.Series) or observations.index.has_duplicates:
        raise ValueError('Scoring requires a uniquely indexed raw-demand Series.')
    result = deepcopy(forecast)
    table = result['predictions']
    actual = pd.to_numeric(observations.reindex(table.index), errors='coerce').to_numpy(dtype=float)
    actual[~np.isfinite(actual)] = np.nan
    mask = np.isfinite(actual)
    table['observed'] = actual; table['scored'] = mask
    scales = result['manifest']['scales']
    result['normalized_predictions']['observed'] = (actual-scales['demand_min'])/scales['demand_scale']
    result['normalized_predictions']['scored'] = mask
    count = int(mask.sum())
    result['scoring'] = {'status':'complete' if count == len(mask) else 'partial' if count else 'not-scored',
        'expected_samples':len(mask),'scored_samples':count,
        'metrics':{name:metrics(actual, table[name].to_numpy()) for name in ('prediction','baseline')},
        'policy':'raw units; same finite measured-demand mask; no demand filling'}
    return result
