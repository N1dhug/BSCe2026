"""Declared-date G/WM development bands using the preserved finite-rank kernels.

This new protocol does not modify or relabel the fixed 2022 study. A record
declares all calibration/evaluation weeks before evaluation targets are read.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.method import g_prequential_bands as original
from src.tools import weekday_backtest as calendar
from src.tools.weekday_backtest import WeekWindow

PROTOCOL = 'BM-G-declared-date-prequential-bands-v1'
MODELS = original.MODELS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    hashes = original.source_hashes()
    hashes['src/method/bm_1_bands.py'] = hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return hashes


def dates(values):
    values = [str(WeekWindow(value).origin.date()) for value in values]
    if not values or values != sorted(set(values)):
        raise ValueError('Declare nonempty, unique chronological Monday origins.')
    if any(pd.Timestamp(b) < pd.Timestamp(a)+pd.Timedelta(days=7) for a, b in zip(values, values[1:])):
        raise ValueError('Forecast weeks must not overlap.')
    return values


def load_forecast(directory):
    """Verify old-study, shared-backtest and saved-run receipts without rewriting."""
    directory = Path(directory)
    receipt = json.loads((directory/'forecast_freeze.json').read_text(encoding='utf-8'))
    flags = [receipt[key] for key in ('target_observation_file_opened', 'scoring_targets_opened') if key in receipt]
    if receipt.get('status') != 'frozen-input-only' or not flags or any(flag is not False for flag in flags):
        raise ValueError('Require a verified input-only forecast freeze receipt.')
    hashes = receipt['artifact_sha256']
    if not {'forecast.json', 'forecast.csv', 'forecast_normalized.csv'} <= set(hashes):
        raise ValueError('Receipt must bind both forecast tables and manifest.')
    calendar._verify_artifacts(directory, hashes)
    forecast = json.loads((directory/'forecast.json').read_text(encoding='utf-8'))
    for key, name in (('predictions', 'forecast.csv'), ('normalized_predictions', 'forecast_normalized.csv')):
        forecast[key] = pd.read_csv(directory/name, index_col=0, parse_dates=[0], float_precision='round_trip')
    origin, _ = original.validate_forecast(forecast)
    if origin != pd.Timestamp(receipt['origin']):
        raise ValueError('Receipt origin differs from forecast cutoff.')
    return forecast


def load_calibration(path):
    """Require the saved freeze receipt in addition to mathematical consistency."""
    path = Path(path)
    if path.name != 'calibration_snapshot.json':
        raise ValueError('Use calibration_snapshot.json beside its calibration_freeze.json receipt.')
    receipt = json.loads((path.parent/'calibration_freeze.json').read_text(encoding='utf-8'))
    if receipt.get('evaluation_observations_used') is not False:
        raise ValueError('Calibration receipt must declare no evaluation observations used.')
    hashes = receipt['artifact_sha256']
    if not {'calibration_snapshot.json', 'calibration_scores.csv'} <= set(hashes):
        raise ValueError('Calibration receipt must bind the snapshot and scores.')
    calendar._verify_artifacts(path.parent, hashes)
    return validate(json.loads(path.read_text(encoding='utf-8')))


def contract(forecast):
    manifest = forecast['manifest']
    return {key:manifest[key] for key in ('population', 'demand_unit', 'demand_timestamp_label',
                                         'timezone', 'scenario', 'baseline_policy', 'fit_policy')}


def observations_for_forecast(observations, forecast):
    """Validate targets against physical hours; NaN stays missing, never imputed."""
    frame = observations.rename('Demand').to_frame()
    calendar._frame(frame, ('Demand',))
    frame['Demand'] = pd.to_numeric(frame['Demand'], errors='raise')
    if np.iscomplexobj(frame.to_numpy()) or np.isinf(frame.to_numpy(float)).any():
        raise ValueError('Observation targets must be real finite values or NaN, never infinite.')
    manifest = forecast['manifest']
    if manifest['demand_timestamp_label'] != 'hour_start':
        raise ValueError('This weekday-export protocol requires hour-start Demand labels.')
    aligned = original.interval.align_demand(frame.Demand, 'hour_start').rename('Demand').to_frame()
    physical = forecast['predictions'].index.tz_localize('Europe/Copenhagen')
    return calendar._grid(aligned, physical).Demand


def calibration_plan(forecasts, evaluation_origins):
    """Preflight metadata only, before any calibration observation is opened."""
    evaluation = dates(evaluation_origins); origins = []; contracts = []
    for forecast in forecasts:
        origin, end = original.validate_forecast(forecast)
        origins.append(str(origin.date())); contracts.append(contract(forecast))
        if end > pd.Timestamp(evaluation[0]):
            raise ValueError('Calibration must end before the declared evaluation cutoff.')
    origins = dates(origins)
    if any(value != contracts[0] for value in contracts):
        raise ValueError('Calibration forecasts must share population, units and information protocol.')
    return origins, evaluation, contracts[0]


def calibrate(packets, evaluation_origins):
    origins, evaluation, common_contract = calibration_plan([p['forecast'] for p in packets], evaluation_origins)
    references = []; rows = []; banks = {model:[] for model in MODELS}
    pairs = {model:[[], []] for model in MODELS}
    for packet in packets:
        forecast = packet['forecast']; origin, end = original.validate_forecast(forecast)
        table = forecast['predictions']
        y = observations_for_forecast(packet['observations'], forecast).to_numpy(float)
        mask = np.isfinite(y)
        physical = table.index.tz_localize('Europe/Copenhagen').tz_convert('UTC')
        adjacent = mask[:-1] & mask[1:] & (np.diff(physical.asi8) == pd.Timedelta(hours=1).value)
        references.append({'origin':str(origin), 'end_exclusive':str(end), 'samples':int(mask.sum()),
            'forecast_sha256':original.forecast_fingerprint(forecast),
            'observation_sha256':original.interval.point._fingerprint(pd.DataFrame({'Demand':y}, index=table.index))})
        for model in MODELS:
            mean = table[model+'_output_mean'].to_numpy(float)
            scale = np.sqrt(table[model+'_output_variance'].to_numpy(float))
            z = np.abs(y-mean)/scale; banks[model].extend(z[mask])
            pairs[model][0].extend(z[:-1][adjacent]); pairs[model][1].extend(z[1:][adjacent])
            rows.append(pd.DataFrame({'origin':str(origin), 'model':model, 'observed':y[mask],
                'mean':mean[mask], 'historical_sd':scale[mask], 'score':z[mask]}, index=table.index[mask]))
    models = {}
    for model, values in banks.items():
        scores = sorted(values); estimate = original.finite_rank(scores)
        if len(scores) < 300 or not estimate['deployable']:
            raise ValueError('Require at least 300 finite deployable scores per model.')
        left, right = pairs[model]
        correlation = float(np.corrcoef(left, right)[0, 1]) if len(left) >= 2 and np.std(left) > 0 and np.std(right) > 0 else None
        models[model] = {**estimate, 'scores':scores, 'within_week_adjacent_pairs':len(left),
                         'score_lag1_correlation':correlation}
    record = {'kind':'BM-G-declared-date-bands', 'schema_version':1, 'protocol':PROTOCOL,
        'calibration_origins':origins, 'evaluation_origins':evaluation, 'freeze_cutoff':evaluation[0],
        'confidence':.95, 'models':models, 'references':references, 'forecast_contract':common_contract,
        'source_sha256':source_hashes(), 'evaluation_observations_used':False,
        'coverage_status':'chronological development; temporal exchangeability and 95% guarantee not established',
        'mean_policy':'unchanged empirical output mean', 'scale_policy':'unchanged historical residual SD',
        'variance_status':'q is only an interval multiplier; no new full predictive distribution'}
    return {'snapshot':validate(record), 'scores':pd.concat(rows)}


def validate(record):
    r = deepcopy(record)
    if (r['kind'] != 'BM-G-declared-date-bands' or r['schema_version'] != 1 or r['protocol'] != PROTOCOL
            or r['source_sha256'] != source_hashes() or r['confidence'] != .95 or r['evaluation_observations_used']):
        raise ValueError('Require an own compatible declared-date band record.')
    calibration = dates(r['calibration_origins']); evaluation = dates(r['evaluation_origins'])
    if r['freeze_cutoff'] != evaluation[0] or pd.Timestamp(calibration[-1])+pd.Timedelta(days=7) > pd.Timestamp(evaluation[0]):
        raise ValueError('Calibration/evaluation chronology is invalid.')
    refs = r['references']
    if [str(pd.Timestamp(ref['origin']).date()) for ref in refs] != calibration:
        raise ValueError('Calibration references differ from declared origins.')
    for ref in refs:
        if pd.Timestamp(ref['end_exclusive']) != pd.Timestamp(ref['origin'])+pd.Timedelta(days=7) or type(ref['samples']) is not int or ref['samples'] < 0:
            raise ValueError('Invalid calibration reference window/count.')
    for model in MODELS:
        item = r['models'][model]; calculated = original.finite_rank(item['scores'])
        if (len(item['scores']) < 300 or np.any(np.diff(item['scores']) < 0)
                or not calculated['deployable'] or sum(ref['samples'] for ref in refs) != calculated['samples']
                or any(item[key] != calculated[key] for key in calculated)):
            raise ValueError('Calibration rank/count differs from the preserved score bank.')
    return r


def apply(forecast, record):
    r = validate(record); origin, _ = original.validate_forecast(forecast)
    if str(origin.date()) not in r['evaluation_origins'] or contract(forecast) != r['forecast_contract']:
        raise ValueError('Forecast must match the declared evaluation dates and information contract.')
    result = deepcopy(forecast)
    for key in ('predictions', 'normalized_predictions'):
        table = result[key]
        for model in MODELS:
            width = r['models'][model]['q']*np.sqrt(table[model+'_output_variance'])
            table[model+'_calibrated_lower'] = table[model+'_output_mean']-width
            table[model+'_calibrated_upper'] = table[model+'_output_mean']+width
    result['manifest']['calibrated_bands'] = {'protocol':PROTOCOL,
        'snapshot_sha256':original.residual.snapshot_fingerprint(r), 'freeze_cutoff':r['freeze_cutoff'],
        'confidence':.95, 'coverage_status':r['coverage_status'], 'evaluation_observations_used':False,
        'clipping':False, 'band_refitted':False}
    return result
