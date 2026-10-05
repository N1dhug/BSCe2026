"""Versioned G fit/origin data; restoration never calls an optimizer.

Legacy imports are explicit: verify the old numerical sources before recovering
baseline settings and the exact endpoint from the original fine-grid CSV.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.output_model import statistical_output
from src.method.bm_1_deterministic_fit import _parameters
from src.validation.model.wm_4 import wm_4_validation


ROOT = Path(__file__).resolve().parents[2]
BASELINE_SETTINGS = ('base_temperature', 'hour_harmonics', 'season_harmonics',
                     'ridge', 'huber_delta', 'max_iter')
# These are the numerical dependencies used by G's forward paths. Documentation,
# runners and optimizer-only changes do not invalidate an already fitted origin.
FORWARD_SOURCES = (
    'src/method/bm_1_deterministic_fit.py',
    'src/params/bm_1_deterministic.py',
    'src/model/wm_4.py', 'src/model/base/wm.py',
    'src/params/wm_4.py',
    'src/validation/model/wm_4.py', 'src/validation/model/wm_1.py',
    'src/exe/run_model/wm_4/output_model.py',
)
LEGACY_SOURCES = (
    'src/method/bm_1_deterministic_fit.py', 'src/params/bm_1_deterministic.py',
    'src/model/wm_4.py', 'src/params/wm_4.py',
    'src/exe/run_model/wm_4/output_model.py',
)
# Reviewed source pair: WM's H5 diffusion addition leaves the deterministic
# constructor, drift and forward solver unchanged. No general version bypass.
LEGACY_WM4_HASHES = {
    '38853b1b0992c76039c6c17e941aa2d1b35ee030fc63e5edeb05abd3d6abd6d6',
    '8e92c6bca6d1de2dcbf971bf2d752b4992f236171fa3209e6da1b408684a2f71',
}
REVIEWED_WM4_H5_HASH = '291187b1fc2ce83f64032f79c9f1f497fb62cc83441d50da211fcd853554bfca'


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in FORWARD_SOURCES}


def baseline_settings(model):
    return {name: getattr(model, name) for name in BASELINE_SETTINGS}


def _finite_vector(value, name, size=None):
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or (size is not None and len(vector) != size) or not np.isfinite(vector).all():
        raise ValueError(f'{name} must be a finite vector of the expected length.')
    return vector.copy()


def restore_baseline(record):
    if record.get('implementation') != 'statistical_output':
        raise ValueError('Unsupported fitted baseline implementation.')
    settings = record.get('settings', {})
    if set(settings) != set(BASELINE_SETTINGS):
        raise ValueError('Snapshot requires all fitted baseline settings.')
    for name in ('hour_harmonics', 'season_harmonics', 'max_iter'):
        value = settings[name]
        if isinstance(value, bool) or not np.isfinite(value) or int(value) != value:
            raise ValueError(f'Invalid baseline setting {name}.')
    if not all(np.isfinite(value) for value in settings.values()):
        raise ValueError('Baseline settings must be finite.')
    model = statistical_output(**settings)
    # Check feature layout using a fixture, without raw observations or .fit().
    count = model._features(pd.DatetimeIndex(['2022-01-01']), [5.]).shape[1]
    model.coef = _finite_vector(record['coef'], 'baseline coefficients', count)
    model.feature_mean = _finite_vector(record['feature_mean'], 'baseline feature mean', count)
    model.feature_scale = _finite_vector(record['feature_scale'], 'baseline feature scale', count)
    if np.any(model.feature_scale <= 0):
        raise ValueError('Baseline feature scales must be positive.')
    model.metadata = deepcopy(record['metadata'])
    return model


def validate_snapshot(snapshot):
    data = deepcopy(snapshot)
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1 or data.get('kind') != 'G-deterministic-fit':
        raise ValueError('Unsupported G fit snapshot schema.')
    if data.get('forward_source_sha256') != source_hashes():
        raise ValueError('Snapshot numerical sources are incompatible; use an explicit reviewed migration.')
    manifest = data['manifest']
    start, end = (pd.Timestamp(manifest[key]) for key in ('training_start', 'training_end_exclusive'))
    if start.tz is not None or end.tz is not None or end <= start or end != end.floor('h') or start != start.floor('h'):
        raise ValueError('Snapshot requires ordered local-naive hourly training edges.')
    if manifest['time_unit'] != 'hours' or not manifest['observation_contract'].startswith('hourly-point'):
        raise ValueError('This G snapshot path supports the original hourly-point development contract only.')
    if manifest['baseline'].get('interpolation') != 'linear between shared fine samples':
        raise ValueError('Unsupported baseline input interpolation.')
    scales = manifest['scales']
    for name in ('demand_min', 'demand_scale', 'price_min', 'price_scale'):
        if not np.isfinite(scales[name]):
            raise ValueError('Snapshot scales must be finite.')
    if scales['demand_scale'] <= 0 or scales['price_scale'] <= 0:
        raise ValueError('Snapshot scales must be positive.')
    p = _parameters(data['fit']['best_converged']['parameters'])
    if manifest['config']['steps_per_hour'] != data['numerics']['bm_steps_per_hour'] or manifest['wm_steps_per_hour'] != data['numerics']['wm_steps_per_hour']:
        raise ValueError('Snapshot and original manifest numerical settings disagree.')
    fixed_x0 = data['fit'].get('fixed_x0')
    if fixed_x0 is not None and p[3] != fixed_x0:
        raise ValueError('Snapshot fixed initial state disagrees with fitted parameters.')
    if data['bm']['parameter_order'] != ['C_hours', 'k', 'Xi', 'x0'] or data['bm']['response'] != 'linear-1-minus-2-argument':
        raise ValueError('Unsupported BM parameter order or response curves.')
    _parameters([*p[:3], data['bm']['x_origin']])
    if pd.Timestamp(data['origin']) != end:
        raise ValueError('Snapshot origin must equal the exact training end.')
    for name in ('bm_steps_per_hour', 'wm_steps_per_hour'):
        value = data['numerics'][name]
        if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < 2:
            raise ValueError('Snapshot step counts must be integers >= 2.')
    tolerance = data['numerics']['tolerance']
    error = data['numerics']['training_refinement_error']
    if not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(error) or error < 0 or error > tolerance:
        raise ValueError('Snapshot must have a passed finite training numerical check.')
    if data['wm'] is not None:
        wm_4_validation.r_coefficients(data['wm']['r'])
        wm_4_validation.initial_condition(data['wm']['state_origin'])
        if data['wm']['state_order'] != ['F', 'R'] or data['wm']['model_time_unit'] != 'days':
            raise ValueError('Unsupported WM state order or model time unit.')
        if data.get('wm_fit') is None or not np.array_equal(data['wm_fit']['r'],data['wm']['r']) or not np.array_equal(data['wm_fit']['state_forecast_origin'],data['wm']['state_origin']):
            raise ValueError('Snapshot WM origin and original fit disagree.')
    baseline = restore_baseline(manifest['baseline'])
    if pd.Timestamp(baseline.metadata['fit_end']) >= end:
        raise ValueError('Baseline fitting metadata crosses the forecast cutoff.')
    return data


def export_fit_snapshot(result):
    if result.get('fit', {}).get('best_converged') is None or not result.get('numerics', {}).get('passed'):
        raise ValueError('Only a converged, numerically checked fit can be exported.')
    manifest = deepcopy(result['manifest'])
    wm = None
    if 'wm_fit' in result:
        wm = {'r': deepcopy(result['wm_fit']['r']),
              'state_origin': deepcopy(result['wm_fit']['state_forecast_origin']),
              'state_order': ['F', 'R'], 'model_time_unit': 'days'}
    snapshot = dict(schema_version=1, kind='G-deterministic-fit',
                    manifest=manifest, origin=manifest['training_end_exclusive'],
                    fit=deepcopy(result['fit']), wm_fit=deepcopy(result.get('wm_fit')),
                    bm={'parameter_order': ['C_hours', 'k', 'Xi', 'x0'],
                        'response': 'linear-1-minus-2-argument', 'x_origin': result['x_forecast_origin']},
                    wm=wm, forward_source_sha256=source_hashes(),
                    numerics={'bm_steps_per_hour': manifest['config']['steps_per_hour'],
                              'wm_steps_per_hour': manifest['wm_steps_per_hour'],
                              'tolerance': result['numerics']['tolerance'],
                              'training_refinement_error': result['numerics']['max_demand_refinement_error']},
                    provenance={'mode': 'exported-fit', 'data_source': manifest.get('data_source')})
    return validate_snapshot(snapshot)


def _read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def migrate_legacy_fit(directory):
    """Read original artifacts without modifying them or replaying a fit."""
    directory = Path(directory)
    names = ('manifest.json', 'fit.json', 'result.json', 'fit_signals_normalized.csv')
    hashes = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names}
    manifest, fit, result = (_read_json(directory/name) for name in names[:3])
    if result['manifest'] != manifest or result['fit'] != fit:
        raise ValueError('Legacy manifest/fit/result artifacts disagree.')
    if result.get('status') not in ('complete', 'wm-rejected'):
        raise ValueError('Legacy result is not a completed checked fit.')
    available = manifest.get('source_sha256', {})
    expected = source_hashes()
    def verified_legacy_source(name):
        content = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
        variants = {expected[name], hashlib.sha256(content.replace(b'\n', b'\r\n')).hexdigest()}
        if name == 'src/model/wm_4.py' and expected[name] == REVIEWED_WM4_H5_HASH:
            variants.update(LEGACY_WM4_HASHES)
        return available.get(name) in variants
    if any(not verified_legacy_source(name) for name in LEGACY_SOURCES):
        raise ValueError('Cannot verify legacy numerical/baseline implementation; no defaults or refit substituted.')
    fine = pd.read_csv(directory/names[3], index_col='Datetime', parse_dates=['Datetime'])
    cutoff = pd.Timestamp(manifest['training_end_exclusive'])
    start = pd.Timestamp(manifest['training_start'])
    if fine.empty or fine.index.has_duplicates or not fine.index.is_monotonic_increasing or fine.index[-1] != cutoff or fine.index[0] != start:
        raise ValueError('Legacy fine trajectory must include the exact training start and cutoff.')
    elapsed = np.asarray((fine.index-start).total_seconds(), dtype=float)
    step = elapsed[1]-elapsed[0] if len(elapsed) > 1 else 0.
    steps = int(round(3600/step)) if step > 0 else 0
    if steps < 2 or not np.allclose(np.diff(elapsed), 3600/steps, rtol=0, atol=1e-6):
        raise ValueError('Legacy WM fine-grid sampling is inconsistent.')
    last = fine.iloc[-1]
    if not np.isfinite(last.x) or not np.isclose(last.x, result['x_forecast_origin'], rtol=0, atol=1e-12):
        raise ValueError('Legacy BM endpoint disagrees with saved forecast origin.')
    if 'settings' not in manifest['baseline']:
        # Source verification above establishes these exact original defaults.
        manifest['baseline']['settings'] = baseline_settings(statistical_output())
    manifest['wm_steps_per_hour'] = steps
    result['manifest'] = manifest
    if 'wm_fit' in result:
        if not {'F', 'R'}.issubset(fine):
            raise ValueError('Legacy WM endpoint fields are missing.')
        result['wm_fit']['state_forecast_origin'] = [float(last.F), float(last.R)]
    snapshot = export_fit_snapshot(result)
    snapshot['provenance'] = {'mode': 'verified-legacy-import', 'original_directory': str(directory),
                              'original_artifact_sha256': hashes,
                              'verified_source_sha256': {name: available[name] for name in LEGACY_SOURCES},
                              'compatibility': 'LF/CRLF normalization; reviewed WM4 H5-only change if present'}
    if hashes != {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names}:
        raise ValueError('Legacy artifacts changed during import.')
    return snapshot


def load_fit_snapshot(path):
    path = Path(path)
    if path.is_dir():
        candidate = path/'fit_snapshot.json'
        return validate_snapshot(_read_json(candidate)) if candidate.exists() else migrate_legacy_fit(path)
    return validate_snapshot(_read_json(path))
