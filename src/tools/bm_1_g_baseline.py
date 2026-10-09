"""Frozen baseline restoration for G. No legacy model migration or refitting."""
from copy import deepcopy
import numpy as np
import pandas as pd
from src.exe.run_model.wm_4.output_model import statistical_output

BASELINE_SETTINGS=('base_temperature','hour_harmonics','season_harmonics','ridge','huber_delta','max_iter')

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
