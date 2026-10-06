"""Pre-origin F4-H2 data preparation; no filter, optimizer or final Y reads.

The historical loader and fixed-population missing rule are reused. The new
baseline is an explicitly declared OLS weekday/hour profile, fitted only to
complete MED group hours. No consumption observation is imputed. All model
parameters remain declared conditional assumptions, not historical estimates.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.params.bm_1_history import BM1ChronologicalSplit, BM1HistoricalCurveContract
from src.tools.bm_1_history_io import read_raw_power_window, read_raw_spot_window
from src.tools.bm_1_reports import _atomic_report

MED_COLUMNS = tuple(x+'_MED' for x in ('223774', '223811', '223868', '223891',
    '223982', '5080166', '5099164', '5099500', '5099508', '5100327',
    '5100459', '5101629', '5102623'))


def validate_policy(policy):
    """Reject silent population, chronology, physical or exposure revisions."""
    if policy.get('schema') != 'f4-h2-compact-data-policy-v1':
        raise ValueError('Unknown compact data policy.')
    if tuple(policy['population']['measurement_columns']) != MED_COLUMNS:
        raise ValueError('Freeze exactly the 13 audited MED columns, in audited order.')
    split = policy['split']
    expected = dict(prior_origin='2022-02-17', train_start='2022-02-18',
        selection_start='2022-02-20', holdout_start='2022-02-21', holdout_end='2022-02-28')
    if split['timezone'] != 'Europe/Copenhagen' or any(
            pd.Timestamp(split[k]) != pd.Timestamp(v, tz=split['timezone']) for k,v in expected.items()):
        raise ValueError('Preserve the approved 24/48/24/168-hour chronology.')
    if any(pd.Timestamp(policy['history'][k]) != pd.Timestamp(v, tz=split['timezone'])
           for k,v in dict(start='2022-01-01', information_cutoff='2022-02-16').items()):
        raise ValueError('Freeze the declared pre-origin history and information cutoff.')
    if any(policy['physical'][k] != v for k,v in dict(C=1., k=2., Xi=1., sigma_x=.5, sigma_y=.04).items()):
        raise ValueError('A physical/noise-condition revision requires a new reviewed policy.')
    if (policy['prior']['alpha'], policy['prior']['beta']) != (20.,20.):
        raise ValueError('Preserve the declared F Beta(20,20) prior.')
    if policy['baseline']['minimum_complete_samples_per_bin'] != 3 or policy['power_scale']['history_maximum_multiplier'] != 1.1:
        raise ValueError('No automatic missing-data or scale retuning.')
    if (policy['price']['area'],policy['price']['unit'],policy['price']['role'],policy['price']['history_range_padding']) != ('DK1','DKK/MWh','spot_proxy',.1):
        raise ValueError('Freeze the declared F spot-proxy normalization.')
    if policy['exposure']['blind_holdout_verified'] is not False:
        raise ValueError('This development packet cannot establish historical blindness.')
    numerics = policy['numerics']
    if any(numerics[k] != v for k,v in dict(base_cells=1921, fine_cells=2561,
        base_max_step=1/32768, fine_max_step=1/65536, frequencies=64, span=2).items()):
        raise ValueError('Preserve the declared conditional time-grid revision.')
    conditional_time_settings(policy['physical'], numerics)


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def weekday_hour_profile(measurements, *, cutoff, targets, minimum_count=3):
    """168 separate OLS intercepts, on complete fixed-population group hours.

    The closed-form minimizer per weekday/hour is its available sample mean.
    Missing members mask the entire observed group hour before any fitting.
    This is a baseline forecast, not reconstruction of missing measurements.
    """
    cutoff = pd.Timestamp(cutoff)
    if (not isinstance(measurements, pd.DataFrame) or measurements.empty
            or measurements.index.tz is None or cutoff.tzinfo is None
            or not measurements.index.is_unique or not measurements.index.is_monotonic_increasing
            or np.any(measurements.index+pd.Timedelta(hours=1) > cutoff)):
        raise ValueError('Baseline history must end by its information cutoff.')
    targets = pd.DatetimeIndex(targets)
    if (targets.tz is None or targets.hasnans or not targets.is_unique
            or not targets.is_monotonic_increasing or np.any(targets < cutoff)):
        raise ValueError('Baseline targets must follow the frozen history cutoff.')
    if not isinstance(minimum_count, int) or minimum_count < 3:
        raise ValueError('Declare at least three complete prior samples per weekday/hour.')
    values = measurements.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('History contains infinite measurement values.')
    complete = np.isfinite(values).all(axis=1)
    observed = measurements.mean(axis=1).where(complete)
    if np.any(observed.dropna().to_numpy() < 0.):
        raise ValueError('Negative complete group energy needs a documented meter correction policy.')
    ids = measurements.index.dayofweek*24+measurements.index.hour
    means, counts = np.empty(168), np.empty(168, dtype=int)
    for key in range(168):
        samples = observed[(ids == key) & complete].to_numpy()
        counts[key] = len(samples)
        if len(samples) < minimum_count:
            raise ValueError(f'Baseline weekday={key//24}/hour={key%24} has only {len(samples)} complete samples.')
        means[key] = float(samples.mean())
    local_targets = targets.tz_convert(measurements.index.tz)
    predicted = means[local_targets.dayofweek*24+local_targets.hour]
    return pd.Series(predicted, index=targets, name='baseline'), dict(
        coefficients=means, counts=counts, observed=observed,
        history_hours=len(observed), complete_group_hours=int(complete.sum()),
        missing_group_hours=int((~complete).sum()))


def conditional_time_settings(physical, numerics):
    """Conservative positive-CN bound for uniform fitted FV with bounded BM1 D.

    D and B are in [0,1], so |drift| <=1/C. Each fitted face coefficient is
    bounded by diffusion/dx² plus its one-sided drift/dx. Hence outgoing rate
    <= sigma_x²*Nx²/16 + 2*Nx/C. Keep Nx and every precision gate unchanged.
    """
    result = {}
    for grid in ('base', 'fine'):
        cells = numerics[grid+'_cells']
        step = numerics[grid+'_max_step']
        rate = physical['sigma_x']**2*cells**2/16.+2*cells/physical['C']
        ratio = step*rate/2.
        if not 0. < ratio < 1.:
            raise ValueError('Declared F condition does not meet conservative positive-CN bound.')
        result[grid] = dict(cells=cells, max_step=step,
            conservative_outgoing_rate=rate, cn_explicit_load_bound=ratio)
    return result


def prepare_compact_packet(policy, *, input_path, price_path, output_dir):
    """Freeze only history/inputs; no TRAIN/selection/final observations loaded."""
    validate_policy(policy)
    output_dir = Path(output_dir)
    if any((output_dir/name).exists() for name in ('packet.json', 'baseline.csv', 'inputs.npz')):
        raise FileExistsError('Preserve the frozen packet; no automatic replacement or retuning.')
    source_before = dict(input=file_sha256(input_path), prices=file_sha256(price_path))
    expected = policy['header_evidence']
    if source_before != dict(input=expected['input_workbook_sha256'], prices=expected['price_workbook_sha256']):
        raise ValueError('Workbook hashes differ from the audited header evidence.')
    columns = tuple(policy['population']['measurement_columns'])
    start = pd.Timestamp(policy['history']['start'])
    cutoff = pd.Timestamp(policy['history']['information_cutoff'])
    split = policy['split']
    prior = pd.Timestamp(split['prior_origin'])
    selection_origin = pd.Timestamp(split['holdout_start'])
    end = pd.Timestamp(split['holdout_end'])
    if not start < cutoff < prior < pd.Timestamp(split['train_start']):
        raise ValueError('History/scale cutoff must be strictly before the prior origin.')
    history = read_raw_power_window(input_path, columns=columns, start=start, end=cutoff,
        timezone=split['timezone'], sheet=expected['input_sheet'])
    targets = pd.date_range(prior, end, freq='h', inclusive='left')
    baseline, profile = weekday_hour_profile(history, cutoff=cutoff, targets=targets,
        minimum_count=policy['baseline']['minimum_complete_samples_per_bin'])
    observed = profile['observed'].dropna().to_numpy()
    power = float(policy['power_scale']['history_maximum_multiplier']*observed.max())
    if not np.isfinite(power) or power <= 0.:
        raise ValueError('Pre-origin complete MED history cannot establish a positive scale.')
    # Price covariates up to forecast origin are allowed; final realized prices
    # and every target-period demand value remain unopened by this adapter.
    prices = read_raw_spot_window(price_path, start=start, end=selection_origin,
        price_area=policy['price']['area'], sheet=expected['price_sheet'])
    calibration = prices[prices.index < cutoff]
    if calibration.isna().any() or len(calibration) == 0 or prices.isna().any():
        raise ValueError('Price history/pre-final inputs must be complete; no interpolation.')
    margin = policy['price']['history_range_padding']*float(calibration.max()-calibration.min())
    low, high = float(calibration.min()-margin), float(calibration.max()+margin)
    if not low < high:
        raise ValueError('Earlier price history has no nonzero normalization range.')
    declared_split = BM1ChronologicalSplit(**{k: split[k] for k in
        ('train_start', 'selection_start', 'holdout_start', 'holdout_end')},
        exposure_intervals=((selection_origin, end, 'Unknown prior exposure; development evaluation only'),),
        holdout_audit_reference=policy['exposure']['audit_reference'])
    contract = BM1HistoricalCurveContract(policy['population']['id'], columns, declared_split,
        power_unit='kW', power_scale=power, price_low=low, price_high=high,
        price_unit=policy['price']['unit'], price_role=policy['price']['role'],
        observation_kind='interval_energy', measurement_unit='kWh', timestamp_label='hour_start',
        metadata_status='working_assumptions', metadata_reference=policy['header_evidence']['report'],
        baseline_reference='pre-origin MED weekday/hour OLS profile in this immutable packet',
        scale_fitted_before=cutoff)
    contract.normalize_prices(prices.to_numpy())
    if np.any(baseline.to_numpy() < 0.) or np.any(baseline.to_numpy() > power):
        raise ValueError('Frozen baseline lies outside the declared power scale.')
    open_history_start = selection_origin-pd.Timedelta(days=7)
    previous_prices = prices.reindex(pd.date_range(open_history_start, selection_origin, freq='h', inclusive='left'))
    if previous_prices.isna().any() or len(previous_prices) != 168:
        raise ValueError('Seasonal-naive open forecast requires the complete previous price week.')
    prefinal_index = pd.date_range(prior, selection_origin, freq='h', inclusive='left')
    open_index = pd.date_range(selection_origin, end, freq='h', inclusive='left')
    input_prices = np.r_[prices.reindex(prefinal_index).to_numpy(), previous_prices.to_numpy()]
    if len(input_prices) != len(targets) or len(open_index) != 168:
        raise ValueError('Input/forecast indices disagree with the declared 24/48/24/168-hour study.')
    normalized_input_prices = contract.normalize_prices(input_prices)
    cn = conditional_time_settings(policy['physical'], policy['numerics'])
    source_after = dict(input=file_sha256(input_path), prices=file_sha256(price_path))
    if source_after != source_before:
        raise ValueError('Workbooks changed while freezing the F packet.')
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(timestamp=[t.isoformat() for t in targets],
                      baseline=baseline.to_numpy())).to_csv(output_dir/'baseline.csv', index=False)
    rows = [dict(weekday=i//24, hour=i%24, mean_kW=float(profile['coefficients'][i]),
                 count=int(profile['counts'][i])) for i in range(168)]
    _atomic_report(output_dir/'inputs.npz', dict(input_edges=np.arange(len(targets)+1)/24.,
        baseline=baseline.to_numpy()/power, price=normalized_input_prices,
        prior_prediction_hours=24, train_hours=48, selection_hours=24, open_hours=168,
        observation_values=None, policy='prefinal realized prices then frozen previous-week open prices'))
    packet = dict(schema='f4-h2-prepared-compact-data-v1', policy=policy,
        conditional_data_ready=True, historical_fit_ready=False, runtime_admitted=False,
        numerical_likelihood_ready=False, blind_h3_acceptance=False, scientific_acceptance=False,
        input_sha256=source_before, contract=contract.to_record(),
        prior=policy['prior'], physical=policy['physical'], numerics=policy['numerics'],
        baseline_profile=rows, baseline_history=dict(start=start.isoformat(),
            information_cutoff=cutoff.isoformat(), history_hours=profile['history_hours'],
            complete_group_hours=profile['complete_group_hours'], missing_group_hours=profile['missing_group_hours']),
        frozen_scales=dict(power_kW=power, price_low_DKK_MWh=low, price_high_DKK_MWh=high),
        exposure=policy['exposure'], positive_cn_bound=cn,
        price_history_decoded=dict(start=start.isoformat(), end=selection_origin.isoformat(), area=policy['price']['area']),
        target_measurement_values_decoded=0, final_realized_prices_decoded=0,
        prediction_times=[t.isoformat() for t in targets],
        outputs_sha256={name:file_sha256(output_dir/name) for name in ('baseline.csv','inputs.npz')})
    packet['identity'] = hashlib.sha256(json.dumps(packet, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
    _atomic_report(output_dir/'packet.json', packet)
    return packet
