"""G0-G3 orchestration. The January variant changes both models' fit window."""
from dataclasses import asdict
from time import perf_counter
import numpy as np
import pandas as pd
from src.exe.run_model.wm_4.output_model import statistical_output, output_callable
from src.exe.run_model.wm_4.fit_daily import fitted_initial_condition
from src.exe.run_model.wm_4.predict_weekly import deviation_metrics
from src.exe.run_model.wm_4.analyze_performance import analyze, summary_table, weekday_summary
from src.model.wm_4 import wm_4
from src.params.wm_4 import qp_params
from src.validation.model.wm_4 import wm_4_validation
from src.params.bm_1_deterministic import BM1DeterministicConfig
from src.method.bm_1_deterministic_fit import BM1DeterministicInputs, fit_deterministic, BudgetStop
from src.tools.bm_1_deterministic_data import prepare_frame, common_inputs


def matched_wm_fit(train, forecast, warmup_hours, steps_per_hour):
    """Existing constrained WM QP and forward model; no oracle state fit."""
    n = len(train['time'])
    obs = np.zeros(n, dtype=bool)
    measured = train['observation_mask']
    obs[:-1:steps_per_hour] = measured
    obs[:warmup_hours*steps_per_hour] = False
    # WM needs a dense reconstructed state, but gaps must not create extra
    # measured transitions in its QP objective.
    adjacent = measured & np.r_[measured[1:],measured[-1]]
    transition = np.repeat(adjacent,steps_per_hour)
    transition[:warmup_hours*steps_per_hour] = False
    data = {'I': train['fine_observed'], 'U': train['price'], 'Time': train['time'],
            'O': train['baseline'], 'observation_mask': obs, 'transition_mask': transition}
    model = wm_4((0., train['time'][-1]/24), n, output_callable(train['fine_index'], train['baseline']))
    init = fitted_initial_condition(data)
    fit = qp_params(model).fit(data, init, verbose=False)
    if forecast is None:
        return fit, None
    last = fit['deterministic_state']
    next_init = np.array([last['F'][-1], last['R'][-1]])
    wm_4_validation.initial_condition(next_init)
    next_model = wm_4((0., forecast['time'][-1]/24), len(forecast['time']),
                      output_callable(forecast['fine_index'], forecast['baseline']))
    next_state = next_model._solve_deterministic(next_init, forecast['price'], forecast['price'],
                                               forecast['time'], forecast['baseline'], fit['r'])
    return fit, next_state


def performance_rows(predictions):
    """Reuse WM analysis by adapting its model prefix, then restore the name."""
    rows = []
    for day, group in predictions.groupby(predictions.index.normalize()):
        measured = group.loc[np.isfinite(group.observed)]
        row = {'day': day, 'weekday': day.day_name(), 'samples': len(measured),
               'missing_demand_hours':len(group)-len(measured)}
        for name in ('O','bm1','wm4'):
            if name not in predictions: continue
            metrics = deviation_metrics(measured.observed,measured[name]) if len(measured) else dict.fromkeys(('mse','rmse','mae','bias'),np.nan)
            row.update({f'{name}_{k}':v for k,v in metrics.items()})
        rows.append(row)
    daily = pd.DataFrame(rows)
    scored = daily.loc[daily.samples>0].copy()
    performance = {}
    if scored.empty:
        return daily, performance
    if 'wm4' in predictions:
        performance['wm4'] = analyze(scored.copy())
    bm_columns = {k: k.replace('bm1_', 'wm4_') for k in daily if k.startswith('bm1_')}
    bm = analyze(scored.drop(columns=[k for k in scored if k.startswith('wm4_')]).rename(columns=bm_columns))
    bm = bm.rename(columns=lambda k: k.replace('wm4_', 'bm1_'))
    performance['bm1'] = bm
    return daily, performance


def demand_units(normalized, scales):
    result = normalized.copy()
    for key in ('observed','demand_input','O','bm1','wm4','R'):
        if key not in result: continue
        result[key] *= scales['demand_scale']
        if key != 'R': result[key] += scales['demand_min']
    return result


def model_metrics(predictions):
    predictions = predictions.loc[np.isfinite(predictions.observed)]
    if predictions.empty:
        raise ValueError('Metrics require at least one measured demand sample.')
    metrics = {key: deviation_metrics(predictions.observed, predictions[key])
               for key in ('O','bm1','wm4') if key in predictions}
    for key in ('bm1','wm4'):
        if key in metrics: metrics[key]['mse_gain'] = metrics['O']['mse']-metrics[key]['mse']
    return metrics


def score_forecast(result, observations):
    """Attach measured-only scores to an already computed forecast.

    The caller can save the input-only forecast before supplying observations.
    Copy tables so scoring cannot mutate the frozen forecast or its origin.
    """
    scored = dict(result)
    predictions = result['predictions'].copy()
    if observations is None:
        actual = np.full(len(predictions), np.nan)
    else:
        if not isinstance(observations, pd.Series) or observations.index.has_duplicates:
            raise ValueError('Scoring observations require a uniquely indexed pandas Series.')
        actual = pd.to_numeric(observations.reindex(predictions.index), errors='coerce').to_numpy(dtype=float, copy=True)
        actual[~np.isfinite(actual)] = np.nan
    predictions['observed'] = actual
    models = [name for name in ('O','bm1','wm4') if name in predictions]
    mask = np.isfinite(actual) & np.isfinite(predictions[models].to_numpy(dtype=float)).all(axis=1)
    predictions['scored'] = mask
    count = int(mask.sum())
    scored['scoring'] = {'status': 'complete' if count == len(predictions) else 'partial' if count else 'not-scored',
                         'expected_samples': len(predictions), 'observed_samples': int(np.isfinite(actual).sum()),
                         'scored_samples': count, 'coverage': count/len(predictions),
                         'models': models, 'matched_wm': 'wm4' in models,
                         'policy': 'same finite measured-demand mask for all available models; no filling'}
    normalized = result['normalized_predictions'].copy()
    scales = result['manifest']['scales']
    normalized['observed'] = (actual-scales['demand_min'])/scales['demand_scale']
    normalized['scored'] = mask
    # Analysis helpers see the shared scoring mask, rather than a model-specific one.
    analysis = predictions.copy()
    analysis.loc[~mask, 'observed'] = np.nan
    daily, performance = performance_rows(analysis)
    scored.update(predictions=predictions, normalized_predictions=normalized,
                  metrics=model_metrics(analysis) if count else {},
                  daily_metrics=daily, performance=performance)
    return scored


def forecast_from_snapshot(snapshot, raw_inputs, *, forecast_hours=168):
    """G3 forward propagation from frozen fit data; no future demand or fit."""
    from copy import deepcopy
    from src.tools.bm_1_deterministic_snapshot import validate_snapshot, restore_baseline
    from src.tools.bm_1_deterministic_data import prepare_forecast_frame
    snapshot = validate_snapshot(snapshot)
    if isinstance(forecast_hours, bool) or int(forecast_hours) != forecast_hours or forecast_hours < 1:
        raise ValueError('Forecast hours must be a positive integer.')
    started = perf_counter()
    origin = pd.Timestamp(snapshot['origin'])
    end = origin+pd.Timedelta(hours=int(forecast_hours))
    manifest = deepcopy(snapshot['manifest'])
    manifest.update(stage='G3-from-fit', fit_only=False, forecast_start=str(origin), forecast_end_exclusive=str(end),
                    evidence='development conditional forecast from frozen fit; original observation contract',
                    fit_reused=True, snapshot_provenance=deepcopy(snapshot['provenance']))
    frame, audit = prepare_forecast_frame(raw_inputs, manifest['scales'], origin, end)
    baseline = restore_baseline(manifest['baseline'])
    wm_steps = snapshot['numerics']['wm_steps_per_hour']
    future = common_inputs(frame, baseline, origin, end, steps_per_hour=wm_steps)
    inputs = BM1DeterministicInputs(future['time'], future['baseline'], (future['price_hour']+1)/2,
                                    steps_per_hour=snapshot['numerics']['bm_steps_per_hour'])
    p = snapshot['fit']['best_converged']['parameters']
    future_p = [*p[:3], snapshot['bm']['x_origin']]
    deadline = perf_counter()+60.
    prediction = inputs.solve(future_p, deadline=deadline)
    refined = inputs.refined().solve(future_p, deadline=deadline)
    error = float(np.max(np.abs(prediction['demand']-refined['demand'])))
    tolerance = snapshot['numerics']['tolerance']
    manifest['forecast_input_audit'] = audit
    manifest['baseline']['forecast_clipped_samples'] = future['baseline_clipped_samples']
    result = dict(manifest=manifest, fit=snapshot['fit'], fit_snapshot=snapshot,
                  x_forecast_origin=snapshot['bm']['x_origin'],
                  numerics={'max_demand_refinement_error': error, 'tolerance': tolerance,
                            'forward_solves': 2, 'rhs_evaluations': prediction['rhs_evaluations']+refined['rhs_evaluations'],
                            'passed': error <= tolerance}, fit_status='reused', forecast_status='complete',
                  scoring={'status': 'pending'})
    hourly = np.arange(int(forecast_hours))*wm_steps
    normalized = pd.DataFrame({'observed': np.nan, 'O': future['baseline'][hourly],
                               'bm1': prediction['demand'], 'U': future['price_hour'],
                               'x': prediction['state'][:-1],
                               'Price_DKK_MWh': raw_inputs.reindex(future['index']).Price.to_numpy(dtype=float)}, index=future['index'])
    result['status'] = 'complete' if snapshot['wm'] is not None else 'wm-rejected'
    if snapshot['wm'] is not None:
        wm = snapshot['wm']
        model = wm_4((0., int(forecast_hours)/24), len(future['time']),
                     output_callable(future['fine_index'], future['baseline']))
        state = model._solve_deterministic(wm['state_origin'], future['price'], future['price'],
                                          future['time'], future['baseline'], wm['r'])
        for name in ('I','F','R'):
            normalized['wm4' if name == 'I' else name] = state[name][hourly]
        result['wm_fit'] = snapshot['wm_fit']
    if not result['numerics']['passed']:
        result['status'], result['forecast_status'] = 'numerics-rejected', 'numerics-rejected'
    result.update(predictions=demand_units(normalized, manifest['scales']), normalized_predictions=normalized,
                  seconds=perf_counter()-started)
    return result


def run_benchmark(raw, train_start, train_end, *, config=None, warmup_hours=0,
                  forecast_hours=168, wm_steps_per_hour=60, price_bounds=None, checkpoint=None, progress=None):
    config = config or BM1DeterministicConfig()
    if isinstance(forecast_hours,bool) or int(forecast_hours) != forecast_hours or forecast_hours < 0:
        raise ValueError('forecast_hours must be a nonnegative integer; zero is fit-only.')
    forecast_hours = int(forecast_hours)
    started = perf_counter()
    train_start, train_end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    origin = train_start-pd.Timedelta(hours=warmup_hours)
    forecast_end = train_end+pd.Timedelta(hours=forecast_hours)
    frame, scales = prepare_frame(raw, train_start, train_end, price_bounds=price_bounds)
    # Fit once with cutoff at forecast origin; exactly the same baseline object.
    baseline = statistical_output().fit(frame, train_end)
    train = common_inputs(frame, baseline, origin, train_end, observations=True, steps_per_hour=wm_steps_per_hour)
    forecast = common_inputs(frame, baseline, train_end, forecast_end, steps_per_hour=wm_steps_per_hour) if forecast_hours else None
    inputs = BM1DeterministicInputs(train['time'], train['baseline'], (train['price_hour']+1)/2,
                                    train['observed'], warmup_hours=warmup_hours,
                                    steps_per_hour=config.steps_per_hour, observation_mask=train['observation_mask'])
    manifest = {'stage': 'G0-G3', 'training_start': str(train_start), 'training_end_exclusive': str(train_end),
                'forecast_start': str(train_end), 'forecast_end_exclusive': str(forecast_end),
                'warmup_hours': warmup_hours, 'time_unit': 'hours', 'scales': scales,
                'normalization': 'train-only affine, shared BM/WM; no held-out rescaling',
                'observation_contract': 'hourly-point-approximation; raw interval semantics unresolved',
                'evidence': 'in-sample development fit with realized prices/temperature' if forecast is None else 'development conditional forecast with realized prices/temperature',
                'fit_only': forecast is None,
                'gap_handling': {'audit':frame.attrs['gap_audit'],
                                 'helper':'src.tools.import_data.import_input.interpolate_time_gaps',
                                 'scope':'training window only; aggregate demand and temperature',
                                 'scoring':'measured demand hours only; baseline, BM and WM share the same mask',
                                 'wm_reconstruction':'interpolated demand; QP transitions across missing hourly endpoints excluded',
                                 'wm_initial_demand_imputed':bool(not train['observation_mask'][0])},
                'initialization': {'bm': 'joint training x0 fit' if config.fixed_x0 is None else 'fixed state before parameter fit',
                                   'fixed_x0': config.fixed_x0,
                                   'wm': 'fit_daily.fitted_initial_condition: F0=INIT_F, R0=I0-O0',
                                   'forecast': 'carry training terminal state, no reset'},
                'baseline': {'implementation': 'statistical_output', 'metadata': baseline.metadata,
                             'coef': baseline.coef.tolist(), 'feature_mean': baseline.feature_mean.tolist(),
                             'feature_scale': baseline.feature_scale.tolist(),
                             'train_clipped_samples': train['baseline_clipped_samples'],
                             'forecast_clipped_samples': forecast['baseline_clipped_samples'] if forecast is not None else None,
                             'interpolation': 'linear between shared fine samples'},
                'bm_solver': 'scalar BM1 RK4, restart held price each hour, no state clipping',
                'wm_solver': 'existing qp_params and discrete deterministic WM4',
                'config': asdict(config)}
    from src.tools.bm_1_deterministic_snapshot import baseline_settings
    manifest['baseline']['settings'] = baseline_settings(baseline)
    manifest['wm_steps_per_hour'] = wm_steps_per_hour
    if checkpoint:
        checkpoint({'manifest': manifest, 'status': 'prepared'})
    # A three-start representative pilot is below the 30-call / 60-second cap.
    pilot_start = perf_counter()
    pilot_rhs = 0
    for p in config.starts[:3]:
        pilot_rhs += inputs.solve(p, deadline=pilot_start+60.)['rhs_evaluations']
    manifest['pilot'] = {'forward_solves': min(3,len(config.starts)), 'rhs_evaluations': pilot_rhs,
                         'seconds': perf_counter()-pilot_start}
    fit = fit_deterministic(inputs, config, checkpoint=(lambda state: checkpoint({'manifest': manifest, 'fit': state})) if checkpoint else None,
                            progress=progress)
    result = {'manifest': manifest, 'fit': fit, 'status': 'unfinished'}
    if fit['best_converged'] is None:
        return result
    p = np.asarray(fit['best_converged']['parameters'])
    # Final scoring and two time-refinement solves have a separate 60-second quota.
    final_deadline = perf_counter()+60.
    trajectory = inputs.solve(p, deadline=final_deadline, keep_fine=True)
    refined = inputs.refined().solve(p, deadline=final_deadline)
    error = float(np.max(np.abs(trajectory['demand']-refined['demand'])))
    final_solves = [trajectory,refined]
    if forecast is not None:
        future = BM1DeterministicInputs(forecast['time'], forecast['baseline'], (forecast['price_hour']+1)/2,
                                       steps_per_hour=config.steps_per_hour)
        future_p = np.r_[p[:3], trajectory['state'][-1]]
        prediction = future.solve(future_p, deadline=final_deadline)
        future_refined = future.refined().solve(future_p, deadline=final_deadline)
        error = max(error, float(np.max(np.abs(prediction['demand']-future_refined['demand']))))
        final_solves += [prediction,future_refined]
    result['numerics'] = {'max_demand_refinement_error': error, 'tolerance': config.numerical_tolerance,
                           'forward_solves': len(final_solves),
                           'rhs_evaluations': sum(v['rhs_evaluations'] for v in final_solves),
                           'passed': error <= config.numerical_tolerance}
    if not result['numerics']['passed']:
        result['status'] = 'numerics-rejected'
        return result
    wm_fit, wm_prediction = None, None
    try:
        wm_fit, wm_prediction = matched_wm_fit(train, forecast, warmup_hours, wm_steps_per_hour)
        result['wm_fit'] = {'observed_samples':int(wm_fit['observation_mask'].sum()),
                           'transition_samples':int(wm_fit['transition_mask'].sum()), 'r': wm_fit['r'].tolist(), 'qp_loss': wm_fit['qp_loss'],
                           'deterministic_loss': wm_fit['deterministic_loss'], 'seconds': wm_fit['qp_fit_time'],
                           'init_cond': wm_fit['init_cond'].tolist(),
                           'state_forecast_origin': [float(wm_fit['deterministic_state'][key][-1]) for key in ('F','R')],
                           'damping_branch': wm_fit['damping_branch'], 'damping_ratio': wm_fit['damping_ratio']}
    except (ValueError, RuntimeError) as exc:
        result['wm_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    # Export the selected training trajectory, not a second fitted model.
    fit_hourly = np.arange(len(train['index']))*wm_steps_per_hour
    fitted = pd.DataFrame({'observed': train['observed'], 'demand_input':train['demand_input'],
                           'demand_imputed':~train['observation_mask'],
                           'temperature_imputed':train['temperature_imputed'], 'O': train['baseline'][fit_hourly],
                           'bm1': trajectory['demand'], 'U': train['price_hour'],
                           'Price_DKK_MWh': raw.reindex(train['index']).Price.to_numpy(dtype=float),
                           'x': trajectory['state'][:-1]},index=train['index'])
    signals = pd.DataFrame({'O': train['baseline'], 'U': train['price'],
                            'bm1': np.interp(train['time'],trajectory['fine_time'],trajectory['fine_demand']),
                            'x': np.interp(train['time'],trajectory['fine_time'],trajectory['fine_state'])},index=train['fine_index'])
    signals['bm_correction'] = signals.bm1-signals.O
    signals['bm_x_dot_per_hour'] = signals.bm_correction/p[0]
    signals['bm_signed_state'] = 1-2*signals.x
    signals['Price_DKK_MWh'] = scales['price_min']+(signals.U+1)*scales['price_scale']/2
    if wm_fit is not None:
        state = wm_fit['deterministic_state']
        for key in ('I','F','R','R_dot'):
            name = 'wm4' if key == 'I' else key
            fitted[name] = state[key][fit_hourly]
            signals[name] = state[key]
    fitted = fitted.loc[fitted.index >= train_start]
    signals = signals.loc[signals.index >= train_start]
    fit_raw = demand_units(fitted, scales)
    fit_daily, fit_performance = performance_rows(fit_raw)
    result.update(fit_predictions=fit_raw, fit_normalized=fitted, fit_signals=signals,
                  fit_daily_metrics=fit_daily, fit_performance=fit_performance,
                  fit_metrics=model_metrics(fit_raw), fit_metrics_normalized=model_metrics(fitted),
                  training_metrics=model_metrics(fitted)['bm1'],
                  x_forecast_origin=float(trajectory['state'][-1]))
    if forecast is None:
        result.update(status='complete' if wm_fit is not None else 'wm-rejected', seconds=perf_counter()-started)
        return result
    idx = forecast['index']
    # Scoring is separate; no future observations were passed to either forward model.
    actual = frame.reindex(idx)['I'].to_numpy(dtype=float)
    hourly = np.arange(forecast_hours)*wm_steps_per_hour
    normalized = pd.DataFrame({'observed': actual, 'O': forecast['baseline'][hourly],
                               'bm1': prediction['demand'],
                               'U': forecast['price_hour'], 'x': prediction['state'][:-1],
                               'Price_DKK_MWh': raw.reindex(idx).Price.to_numpy(dtype=float)}, index=idx)
    if wm_prediction is not None:
        normalized['wm4'] = wm_prediction['I'][hourly]
        normalized['F'] = wm_prediction['F'][hourly]
        normalized['R'] = wm_prediction['R'][hourly]
    predictions = demand_units(normalized, scales)
    result.update(status='complete' if wm_prediction is not None else 'wm-rejected', predictions=predictions, normalized_predictions=normalized,
                  fit_status='complete', forecast_status='complete',
                  x_forecast_origin=float(trajectory['state'][-1]), seconds=perf_counter()-started)
    return score_forecast(result, raw.reindex(idx).Demand)
