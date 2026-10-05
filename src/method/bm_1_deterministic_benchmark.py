"""G0-G3 orchestration. The January variant changes both models' fit window."""
from dataclasses import asdict
from time import perf_counter
import numpy as np
import pandas as pd
from src.exe.run_model.wm_4.output_model import statistical_output, output_callable
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
    obs[warmup_hours*steps_per_hour: n-1:steps_per_hour] = True
    transition = np.zeros(n-1, dtype=bool)
    transition[warmup_hours*steps_per_hour:] = True
    data = {'I': train['fine_observed'], 'U': train['price'], 'Time': train['time'],
            'O': train['baseline'], 'observation_mask': obs, 'transition_mask': transition}
    model = wm_4((0., train['time'][-1]/24), n, output_callable(train['fine_index'], train['baseline']))
    init = np.array([0., data['I'][0]-data['O'][0]])
    fit = qp_params(model).fit(data, init)
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
        row = {'day': day, 'weekday': day.day_name(), 'samples': len(group)}
        for name in ('O','bm1','wm4'):
            if name not in predictions: continue
            row.update({f'{name}_{k}': v for k,v in deviation_metrics(group.observed, group[name]).items()})
        rows.append(row)
    daily = pd.DataFrame(rows)
    performance = {}
    if 'wm4' in predictions:
        performance['wm4'] = analyze(daily.copy())
    bm_columns = {k: k.replace('bm1_', 'wm4_') for k in daily if k.startswith('bm1_')}
    bm = analyze(daily.drop(columns=[k for k in daily if k.startswith('wm4_')]).rename(columns=bm_columns))
    bm = bm.rename(columns=lambda k: k.replace('wm4_', 'bm1_'))
    performance['bm1'] = bm
    return daily, performance


def run_benchmark(raw, train_start, train_end, *, config=None, warmup_hours=0,
                  forecast_hours=168, wm_steps_per_hour=60, price_bounds=None, checkpoint=None, progress=None):
    config = config or BM1DeterministicConfig()
    started = perf_counter()
    train_start, train_end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    origin = train_start-pd.Timedelta(hours=warmup_hours)
    forecast_end = train_end+pd.Timedelta(hours=forecast_hours)
    frame, scales = prepare_frame(raw, train_start, train_end, price_bounds=price_bounds)
    # Fit once with cutoff at forecast origin; exactly the same baseline object.
    baseline = statistical_output().fit(frame, train_end)
    train = common_inputs(frame, baseline, origin, train_end, observations=True, steps_per_hour=wm_steps_per_hour)
    forecast = common_inputs(frame, baseline, train_end, forecast_end, steps_per_hour=wm_steps_per_hour)
    inputs = BM1DeterministicInputs(train['time'], train['baseline'], (train['price_hour']+1)/2,
                                    train['observed'], warmup_hours=warmup_hours,
                                    steps_per_hour=config.steps_per_hour)
    manifest = {'stage': 'G0-G3', 'training_start': str(train_start), 'training_end_exclusive': str(train_end),
                'forecast_start': str(train_end), 'forecast_end_exclusive': str(forecast_end),
                'warmup_hours': warmup_hours, 'time_unit': 'hours', 'scales': scales,
                'normalization': 'train-only affine, shared BM/WM; no held-out rescaling',
                'observation_contract': 'hourly-point-approximation; raw interval semantics unresolved',
                'evidence': 'development conditional forecast with realized prices/temperature',
                'baseline': {'implementation': 'statistical_output', 'metadata': baseline.metadata,
                             'coef': baseline.coef.tolist(), 'feature_mean': baseline.feature_mean.tolist(),
                             'feature_scale': baseline.feature_scale.tolist(),
                             'train_clipped_samples': train['baseline_clipped_samples'],
                             'forecast_clipped_samples': forecast['baseline_clipped_samples'],
                             'interpolation': 'linear between shared fine samples'},
                'bm_solver': 'scalar BM1 RK4, restart held price each hour, no state clipping',
                'wm_solver': 'existing qp_params and discrete deterministic WM4',
                'config': asdict(config)}
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
    trajectory = inputs.solve(p, deadline=final_deadline)
    refined = inputs.refined().solve(p, deadline=final_deadline)
    error = float(np.max(np.abs(trajectory['demand']-refined['demand'])))
    future = BM1DeterministicInputs(forecast['time'], forecast['baseline'], (forecast['price_hour']+1)/2,
                                   steps_per_hour=config.steps_per_hour)
    future_p = np.r_[p[:3], trajectory['state'][-1]]
    prediction = future.solve(future_p, deadline=final_deadline)
    future_refined = future.refined().solve(future_p, deadline=final_deadline)
    error = max(error, float(np.max(np.abs(prediction['demand']-future_refined['demand']))))
    result['numerics'] = {'max_demand_refinement_error': error, 'tolerance': config.numerical_tolerance,
                           'forward_solves': 4,
                           'rhs_evaluations': sum(v['rhs_evaluations'] for v in (trajectory,refined,prediction,future_refined)),
                           'passed': error <= config.numerical_tolerance}
    if not result['numerics']['passed']:
        result['status'] = 'numerics-rejected'
        return result
    wm_prediction = None
    try:
        wm_fit, wm_prediction = matched_wm_fit(train, forecast, warmup_hours, wm_steps_per_hour)
        result['wm_fit'] = {'r': wm_fit['r'].tolist(), 'qp_loss': wm_fit['qp_loss'],
                           'deterministic_loss': wm_fit['deterministic_loss'], 'seconds': wm_fit['qp_fit_time']}
    except (ValueError, RuntimeError) as exc:
        result['wm_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    idx = forecast['index']
    # Scoring is separate; no future observations were passed to either forward model.
    actual = frame.reindex(idx)['I'].to_numpy(dtype=float)
    if not np.all(np.isfinite(actual)):
        raise ValueError('Forecast created, but scoring requires complete future observations.')
    hourly = np.arange(forecast_hours)*wm_steps_per_hour
    normalized = pd.DataFrame({'observed': actual, 'O': forecast['baseline'][hourly],
                               'bm1': prediction['demand'],
                               'U': forecast['price_hour'], 'x': prediction['state'][:-1]}, index=idx)
    if wm_prediction is not None:
        normalized['wm4'] = wm_prediction['I'][hourly]
        normalized['F'] = wm_prediction['F'][hourly]
        normalized['R'] = wm_prediction['R'][hourly]
    predictions = normalized.copy()
    for key in ('observed','O','bm1','wm4','R'):
        if key not in predictions: continue
        predictions[key] *= scales['demand_scale']
        if key != 'R': predictions[key] += scales['demand_min']
    metrics = {key: deviation_metrics(predictions.observed, predictions[key]) for key in ('O','bm1','wm4') if key in predictions}
    for key in ('bm1','wm4'):
        if key not in metrics: continue
        metrics[key]['mse_gain'] = metrics['O']['mse']-metrics[key]['mse']
    daily, performance = performance_rows(predictions)
    result.update(status='complete' if wm_prediction is not None else 'wm-rejected', predictions=predictions, normalized_predictions=normalized,
                  daily_metrics=daily, performance=performance, metrics=metrics,
                  training_metrics=deviation_metrics(inputs.observations[warmup_hours:], trajectory['demand'][warmup_hours:]),
                  x_forecast_origin=float(trajectory['state'][-1]), seconds=perf_counter()-started)
    return result
