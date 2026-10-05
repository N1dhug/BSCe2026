"""G4 archive fit and frozen forecasts; no new baseline/WM fit or solver family."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.method.bm_1_deterministic_response import BM1ResponseInputs, fit_response, is_linear
from src.params.bm_1_deterministic_response import BM1ResponseModel
from src.params.bm_1_response import BM1PiecewiseResponse
from src.tools.bm_1_deterministic_snapshot import ROOT, load_fit_snapshot, validate_snapshot, restore_baseline
from src.method.bm_1_deterministic_benchmark import (forecast_from_snapshot, score_forecast,
    demand_units, model_metrics, performance_rows, deviation_metrics)
from src.tools.bm_1_deterministic_data import prepare_forecast_frame, common_inputs


SOURCES = ('src/params/bm_1_response.py', 'src/params/bm_1_deterministic_response.py',
           'src/method/bm_1_deterministic_response.py', 'src/tools/bm_1_deterministic_response.py',
           'src/tools/bm_1_deterministic_data.py', 'src/tools/bm_1_deterministic_snapshot.py',
           'src/method/bm_1_deterministic_benchmark.py')

DEVELOPMENT_POLICY = {'primary_metric':'pooled measured-demand MSE',
    'reference':'linear_bm1', 'practical_relative_tolerance':.01,
    'automatic_selection':False,
    'rule':'prefer linear within 1% of best checked development MSE; include numerical error and diagnose identification',
    'evidence':'development choice only; no statistical significance or causal-response claim'}


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for name in SOURCES}


def read_training_archive(directory, *, deadline=None):
    """Verify saved measured training data against its original linear replay."""
    directory = Path(directory)
    reference = load_fit_snapshot(directory)
    manifest = reference['manifest']
    if manifest['warmup_hours'] != 0:
        raise ValueError('The first archive pilot requires a zero-warmup fit; cropped warmup is not reconstructed.')
    names = ('fit_predictions_normalized.csv', 'fit_signals_normalized.csv')
    hashes = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names}
    hourly, fine = [pd.read_csv(directory/name, index_col='Datetime', parse_dates=['Datetime']) for name in names]
    start, end = (pd.Timestamp(manifest[k]) for k in ('training_start', 'training_end_exclusive'))
    expected = pd.date_range(start, end, freq='h', inclusive='left')
    wm_steps = reference['numerics']['wm_steps_per_hour']
    canonical_time = np.arange(len(expected)*wm_steps+1,dtype=float)/wm_steps
    elapsed = np.asarray((fine.index-start).total_seconds(),dtype=float)
    # The original float-hours -> timedelta conversion may differ by a few ns.
    # Verify its grid within the legacy import's 1 microsecond tolerance, then
    # reconstruct the original numerical grid rather than those display offsets.
    if (not hourly.index.equals(expected) or len(fine) != len(canonical_time)
            or fine.index.has_duplicates or not fine.index.is_monotonic_increasing
            or fine.index[0] != start or fine.index[-1] != end
            or not np.allclose(elapsed,3600*canonical_time,rtol=0,atol=1e-6)):
        raise ValueError('Archive must contain the exact complete hourly/fine TRAIN grids and cutoff.')
    required = ['O','U','bm1','x','demand_input','Price_DKK_MWh']
    if not set(required+['observed','demand_imputed','temperature_imputed']).issubset(hourly) or not {'O','U','x','bm1'}.issubset(fine):
        raise ValueError('Training archive fields are missing.')
    if not np.isfinite(hourly[required].to_numpy(dtype=float)).all() or not np.isfinite(fine[['O','U','x']].to_numpy(dtype=float)).all():
        raise ValueError('Archived model inputs/states must be finite.')
    if not all(hourly[name].dtype == bool for name in ('demand_imputed','temperature_imputed')):
        raise ValueError('Archive imputation masks must be explicit booleans.')
    mask = np.isfinite(hourly.observed.to_numpy(dtype=float))
    if not np.array_equal(mask, ~hourly.demand_imputed.to_numpy()) or int(mask.sum()) < 168:
        raise ValueError('Archive requires the original measured mask and at least 168 measured TRAIN hours.')
    times = canonical_time
    np.testing.assert_allclose(fine.U.to_numpy(), np.r_[np.repeat(hourly.U.to_numpy(),wm_steps),hourly.U.iloc[-1]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(hourly.O, fine.O.iloc[:-1:wm_steps], rtol=0, atol=1e-12)
    inputs = BM1ResponseInputs(times, fine.O.to_numpy(), (hourly.U.to_numpy()+1)/2,
        hourly.observed.to_numpy(), observation_mask=mask, steps_per_hour=reference['numerics']['bm_steps_per_hour'])
    p = reference['fit']['best_converged']['parameters']
    replay = inputs.solve(p, deadline=deadline)
    if not np.allclose(replay['demand'],hourly.bm1,rtol=0,atol=1e-11) or not np.allclose(replay['state'][:-1],hourly.x,rtol=0,atol=1e-11) or not np.isclose(replay['state'][-1],reference['bm']['x_origin'],rtol=0,atol=1e-11):
        raise ValueError('Archived inputs do not reproduce the original linear fit and exact cutoff state.')
    saved_mse = reference['fit']['best_converged'].get('mse')
    replay_mse = float(np.mean((replay['demand'][mask]-inputs.observations[mask])**2))
    if saved_mse is not None and not np.isclose(saved_mse,replay_mse,rtol=0,atol=1e-11):
        raise ValueError('Archived measured TRAIN observations disagree with the original saved MSE.')
    saved_count = reference['fit'].get('observed_samples')
    if saved_count is not None and saved_count != int(mask.sum()):
        raise ValueError('Archived measured TRAIN count disagrees with the original fit.')
    if hashes != {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names}:
        raise ValueError('Training exports changed during import.')
    return reference, inputs, hourly, fine, hashes, replay['rhs_evaluations']


def validate_response_snapshot(snapshot):
    data = deepcopy(snapshot)
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1 or data.get('kind') != 'G4-deterministic-fit':
        raise ValueError('Unsupported G4 snapshot schema.')
    reference = validate_snapshot(data['reference'])
    if data['forward_source_sha256'] != source_hashes():
        raise ValueError('G4 numerical sources are incompatible; an explicit migration is required.')
    if data.get('development_policy') != DEVELOPMENT_POLICY:
        raise ValueError('The first G4 pilot must preserve its declared development-choice rule.')
    model = BM1ResponseModel.from_record(data['model'])
    best = data['fit']['best_converged']
    original = reference['fit']['best_converged']['parameters']
    p = np.asarray(best['parameters'],dtype=float)
    if p.shape != (4,) or not np.isfinite(p).all() or not 0 <= p[3] <= 1 or not best.get('converged'):
        raise ValueError('G4 requires a converged physical fit.')
    if not np.array_equal(p[:3],[model.C_hours,model.k,model.Xi]) or p[1] != original[1]:
        raise ValueError('G4 model and fit disagree, or fixed reference k changed.')
    config = reference['manifest']['config']
    if np.any(p < config['lower']) or np.any(p > config['upper']):
        raise ValueError('G4 parameters violate the original physical bounds.')
    if not is_linear(model.state_response) or not np.array_equal(model.price_response.knots,[0.,.5,1.]) or model.price_response.values[1] != best['g_mid']:
        raise ValueError('The first G4 pilot requires linear f and the declared two-piece g.')
    fixed = reference['fit'].get('fixed_x0')
    if data['fit'].get('fixed_x0') != fixed or (fixed is not None and p[3] != fixed):
        raise ValueError('G4 must preserve the reference initial-state policy.')
    if data['origin'] != reference['origin'] or not np.isfinite(data['x_origin']) or not 0 <= data['x_origin'] <= 1:
        raise ValueError('Invalid G4 cutoff/state.')
    if data['numerics']['steps_per_hour'] != reference['numerics']['bm_steps_per_hour'] or data['numerics']['tolerance'] != reference['numerics']['tolerance']:
        raise ValueError('G4 must preserve the declared grid/tolerance.')
    error = data['numerics']['training_refinement_error']
    if not np.isfinite(error) or not 0 <= error <= data['numerics']['tolerance']:
        raise ValueError('G4 training refinement check did not pass.')
    strength = data['fit']['regularization']
    if not np.isfinite(strength) or strength < 0 or not np.isclose(best['penalty'],strength*best['g_mid']**2,rtol=0,atol=1e-14):
        raise ValueError('G4 penalty record disagrees with fitted geometry.')
    if not np.isfinite(best['mse']) or best['mse'] < 0 or not np.isclose(best['regularized_objective'],best['mse']+best['penalty'],rtol=0,atol=1e-14):
        raise ValueError('G4 data MSE and regularized objective must be separate and consistent.')
    return data


def load_response_snapshot(path):
    path = Path(path)
    if path.is_dir(): path = path/'fit_snapshot.json'
    return validate_response_snapshot(json.loads(path.read_text(encoding='utf-8')))


def fit_training_archive(directory, *, seconds=600., checkpoint=None, progress=None, **options):
    """Return checked per-strength candidates; never select a strength on TRAIN."""
    if not np.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError('The entire G4 computation is capped at 600 seconds.')
    started = perf_counter(); deadline = started+seconds
    reference, inputs, hourly, fine, hashes, replay_rhs = read_training_archive(directory, deadline=deadline)
    config = reference['manifest']['config']
    # Reserve a bounded share for final solves/refinement, within the same wall cap.
    remaining = deadline-perf_counter()
    reserve = min(30.,remaining/4)
    if remaining <= 0: raise TimeoutError('G4 archive replay exhausted the wall budget.')
    report = fit_response(inputs, reference['fit']['best_converged']['parameters'],
        fixed_x0=reference['fit'].get('fixed_x0'), C_bounds=(config['lower'][0],config['upper'][0]),
        Xi_bounds=(config['lower'][2],config['upper'][2]), x0_bounds=(config['lower'][3],config['upper'][3]),
        seconds=remaining-reserve, checkpoint=checkpoint, progress=progress, **options)
    report['total_seconds_cap'] = seconds
    report['development_policy'] = deepcopy(DEVELOPMENT_POLICY)
    report['archive_sha256'] = hashes
    report['counts']['archive_replay_forward_solves'] = 1
    report['counts']['archive_replay_rhs_evaluations'] = replay_rhs
    report['counts']['final_forward_solves'] = 0
    report['counts']['final_rhs_evaluations'] = 0
    results = []
    for number,item in enumerate(report['candidates']):
        best = item['best_converged']
        if best is None: item['export_status'] = 'no-converged-fit'; continue
        p = best['parameters']
        model = BM1ResponseModel(*p[:3],price_response=BM1PiecewiseResponse.from_inner_values([best['g_mid']],pieces=2))
        inputs.price_response = model.price_response
        try:
            report['counts']['final_forward_solves'] += 1
            try: solved = inputs.solve(p,deadline=deadline,keep_fine=True)
            finally: report['counts']['final_rhs_evaluations'] += inputs.last_rhs_evaluations
            fine_inputs = inputs.refined()
            report['counts']['final_forward_solves'] += 1
            try: refined = fine_inputs.solve(p,deadline=deadline)
            finally: report['counts']['final_rhs_evaluations'] += fine_inputs.last_rhs_evaluations
        except RuntimeError as exc:
            item['export_status'] = 'budget-stopped'; item['export_failure'] = str(exc); continue
        error = float(np.max(np.abs(solved['demand']-refined['demand'])))
        tolerance = reference['numerics']['tolerance']
        item['refinement_error'] = error
        if error > tolerance: item['export_status'] = 'numerics-rejected'; continue
        fit = dict(best_converged=deepcopy(best),fixed_x0=report['fixed_x0'],fixed_k=report['fixed_k'],regularization=item['regularization'])
        snapshot = dict(schema_version=1,kind='G4-deterministic-fit',reference=reference,
            development_policy=deepcopy(DEVELOPMENT_POLICY),
            model=model.to_record(),fit=fit,origin=reference['origin'],x_origin=float(solved['state'][-1]),
            numerics={'steps_per_hour':inputs.steps,'tolerance':tolerance,'training_refinement_error':error},
            forward_source_sha256=source_hashes(), provenance={'training_artifact_sha256':hashes,
                'selection':'unselected DEVELOPMENT candidate; no final-test claim','reference_directory':str(directory)})
        snapshot = validate_response_snapshot(snapshot)
        normalized = hourly.copy(); normalized['linear_bm1'] = hourly.bm1
        normalized['linear_x'] = hourly.x; normalized['bm1'] = solved['demand']; normalized['x'] = solved['state'][:-1]
        signals = fine.copy(); signals['linear_bm1'] = fine.bm1; signals['linear_x'] = fine.x
        signals['bm1'] = np.interp(inputs.time,solved['fine_time'],solved['fine_demand'])
        signals['x'] = np.interp(inputs.time,solved['fine_time'],solved['fine_state'])
        signals['bm_correction'] = signals.bm1-signals.O
        signals['bm_x_dot_per_hour'] = signals.bm_correction/p[0]
        signals['bm_signed_state'] = 1-2*signals.x
        raw = demand_units(normalized,reference['manifest']['scales'])
        scales = reference['manifest']['scales']
        raw['linear_bm1'] = normalized.linear_bm1*scales['demand_scale']+scales['demand_min']
        daily, performance = performance_rows(raw)
        manifest = deepcopy(reference['manifest'])
        manifest.update(stage='G4-shared-price-response',fit_only=True,response_model=model.to_record(),
            response_regularization=item['regularization'],reference_fit_reused=True,
            evidence='TRAIN fit; unselected DEVELOPMENT candidate; original hourly-point contract')
        result = dict(status='complete' if reference['wm'] is not None else 'wm-rejected',manifest=manifest,
            fit=fit,fit_snapshot=snapshot,x_forecast_origin=snapshot['x_origin'],
            numerics={'passed':True,'tolerance':tolerance,'max_demand_refinement_error':error},
            fit_predictions=raw,fit_normalized=normalized,fit_signals=signals,
            fit_daily_metrics=daily,fit_performance=performance,fit_metrics=model_metrics(raw),
            fit_metrics_normalized=model_metrics(normalized),variant=f'lambda_{number:02d}')
        for table,key in ((raw,'fit_metrics'),(normalized,'fit_metrics_normalized')):
            mask = np.isfinite(table.observed)
            result[key]['linear_bm1'] = deviation_metrics(table.loc[mask,'observed'],table.loc[mask,'linear_bm1'])
        if reference['wm_fit'] is not None: result['wm_fit'] = deepcopy(reference['wm_fit'])
        item['export_status'] = 'complete'; item['variant'] = result['variant']
        results.append(result)
    report['total_seconds'] = perf_counter()-started
    report['counts']['all_forward_solves'] = report['counts'].get('forward_solves',0)+1+report['counts']['final_forward_solves']
    report['counts']['all_rhs_evaluations'] = report['counts'].get('rhs_evaluations',0)+replay_rhs+report['counts']['final_rhs_evaluations']
    if any(c.get('export_status') != 'complete' for c in report['candidates']) and report['status'] == 'complete':
        report['status'] = 'unfinished'
    if checkpoint is not None: checkpoint(report)
    return report, results


def forecast_response(snapshot, raw_inputs, *, forecast_hours=168):
    snapshot = validate_response_snapshot(snapshot)
    reference = snapshot['reference']
    result = forecast_from_snapshot(reference,raw_inputs,forecast_hours=forecast_hours)
    model = BM1ResponseModel.from_record(snapshot['model'])
    origin = pd.Timestamp(snapshot['origin']); end = origin+pd.Timedelta(hours=forecast_hours)
    frame, _ = prepare_forecast_frame(raw_inputs,reference['manifest']['scales'],origin,end)
    future = common_inputs(frame,restore_baseline(reference['manifest']['baseline']),origin,end,
        steps_per_hour=reference['numerics']['wm_steps_per_hour'])
    inputs = BM1ResponseInputs(future['time'],future['baseline'],(future['price_hour']+1)/2,
        state_response=model.state_response,price_response=model.price_response,
        steps_per_hour=snapshot['numerics']['steps_per_hour'])
    p = [model.C_hours,model.k,model.Xi,snapshot['x_origin']]
    deadline = perf_counter()+60.
    solved = inputs.solve(p,deadline=deadline); refined = inputs.refined().solve(p,deadline=deadline)
    error = float(np.max(np.abs(solved['demand']-refined['demand'])))
    normalized = result['normalized_predictions'].copy()
    normalized['linear_bm1'] = normalized.bm1; normalized['linear_x'] = normalized.x
    normalized['bm1'] = solved['demand']; normalized['x'] = solved['state'][:-1]
    raw = demand_units(normalized,reference['manifest']['scales'])
    raw['linear_bm1'] = result['predictions'].bm1.copy()
    result.update(predictions=raw,normalized_predictions=normalized,fit_snapshot=snapshot,
        fit=snapshot['fit'],x_forecast_origin=snapshot['x_origin'])
    result['manifest'].update(stage='G4-from-fit',response_model=model.to_record(),
        response_regularization=snapshot['fit']['regularization'],response_provenance=snapshot['provenance'],
        evidence='unselected conditional DEVELOPMENT forecast; original hourly-point contract')
    numerics = result['numerics']
    numerics['linear_reference_refinement_error'] = numerics['max_demand_refinement_error']
    numerics['response_refinement_error'] = error
    numerics['max_demand_refinement_error'] = max(error,numerics['max_demand_refinement_error'])
    numerics['forward_solves'] += 2
    numerics['rhs_evaluations'] += solved['rhs_evaluations']+refined['rhs_evaluations']
    numerics['passed'] = numerics['passed'] and error <= numerics['tolerance']
    if not numerics['passed']: result.update(status='numerics-rejected',forecast_status='numerics-rejected')
    return result


def score_response_forecast(result, observations):
    if not np.isfinite(result['predictions'].linear_bm1.to_numpy()).all():
        raise ValueError('The frozen linear reference must be finite on every forecast hour.')
    scored = score_forecast(result,observations)
    mask = scored['predictions'].scored
    if mask.any():
        table = scored['predictions'].loc[mask]
        metrics = deviation_metrics(table.observed,table.linear_bm1)
        metrics['mse_gain'] = scored['metrics']['O']['mse']-metrics['mse']
        scored['metrics']['linear_bm1'] = metrics
    scored['scoring']['models'].append('linear_bm1')
    return scored
