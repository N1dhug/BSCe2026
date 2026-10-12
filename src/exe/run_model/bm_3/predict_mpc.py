"""BM3 rolling 1..8h prediction; same five historical 24h windows as BM1."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
from src.exe.run_model.bm_1.predict_daily import held_signal
from src.exe.run_model.bm_1.predict_mpc import posterior_mass
from src.exe.run_model.bm_3.predict_daily import add_model_options
from src.model.bm_3 import bm_3
from src.params.bm_3 import lsq_params, historical_diffusion_params
from src.tools.baseline import statistical_output
from src.tools.forecast_metrics import predictive_intervals, forecast_metrics
from src.validation.exe.run_model.bm_1.predict_mpc import bm_1_mpc_validation as checks

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'files'/'output'/'bm_3'/'prediction'/'mpc'


def distribution_forecast(data, fitted, initial_mass, sigma_x, *, fp_max_step=.25,
                          substeps=2, state_bounds=(-1., 2.), tail_tolerance=1e-4):
    model = bm_3(held_signal(data['B']), lambda t: .5,
        (0., float(data['Time'][-1])), len(data['Time']),
        restoring_strength=fitted['restoring_strength'], state_bounds=state_bounds,
        tail_tolerance=tail_tolerance)
    x = model.fokker_planc_reference.grid(len(initial_mass), state_bounds)[1]
    if sigma_x > 0:
        parameters = [held_signal(data['U']), fitted['f'], fitted['g'],
                      fitted['C'], fitted['k'], fitted['Xi']]
        _, x, density, mass, mean, variance, u, b, d, diagnostics = model.fokker_planc_reference_solver(
            initial_mass, parameters, sigma_x, t_eval=data['Time'], max_step=fp_max_step)
        supports = [np.asarray(model.eval(t, x, *[parameters[j] for j in (0,1,2,4,5)])[-1]) for t in data['Time']]
        return supports, list(mass), {'diagnostics': diagnostics, 'state_mean': mean,
            'state_variance': variance, 'outside_probability': np.asarray(diagnostics['outside_probability']),
            'auxiliary_flow_mean': np.asarray(diagnostics['auxiliary_flow_mean'])}
    # A zero-noise law is propagated deterministically, without FV artificial diffusion.
    paths = [lsq_params._predict(data, float(x0), fitted, substeps) for x0 in x]
    states = np.asarray([s for s, d in paths]); demands = np.asarray([d for s, d in paths])
    mean = initial_mass @ states
    return list(demands.T), [initial_mass]*len(data['Time']), {
        'diagnostics': {'method': 'deterministic grid trajectories', 'state_bounds': list(state_bounds)},
        'state_mean': mean, 'state_variance': np.maximum(initial_mass @ (states*states)-mean*mean, 0),
        'outside_probability': initial_mass @ ((states < 0) | (states > 1)),
        'auxiliary_flow_mean': fitted['C']*(initial_mass @ bm_3.restoring(states, fitted['restoring_strength']))}


def predict_origin(frame, origin, output_model, *, X0=.5, weeks=5, horizon=8,
    x_steps=301, max_iterations=3, max_nfev=120, state_maxiter=60, substeps=2,
    fp_max_step=.25, prior_bandwidth=.025, C=None, Xi=None, min_capacity=.05,
    fitted=None, joint_max_nfev=0, joint_max_seconds=20., joint_gtol=1e-5,
    restoring_strength=10., state_bounds=(-1., 2.), tail_tolerance=1e-4):
    origin = checks.origin(origin)
    checks.options(horizon, x_steps, fp_max_step, prior_bandwidth, substeps, weeks)
    if not np.isfinite(min_capacity) or not .001 <= min_capacity < 100:
        raise ValueError('min_capacity must be in [0.001,100).')
    if C is not None and C < min_capacity: raise ValueError('Fixed C is below min_capacity.')
    origins = [origin-pd.Timedelta(weeks=k) for k in range(weeks, 0, -1)]
    training = [checks.window(frame, t, output_model, hours=24) for t in origins]
    model = bm_3(held_signal(training[0]['B']), lambda t: .5, (0., 24.), 25,
        restoring_strength=restoring_strength, state_bounds=state_bounds, tail_tolerance=tail_tolerance)
    if fitted is None:
        fitted = lsq_params(model).fit_many(training, X0, C=C, Xi=Xi,
            max_iterations=max_iterations, max_nfev=max_nfev, state_maxiter=state_maxiter,
            substeps=substeps, fit_bounds={'C': (float(min_capacity), 100.)},
            joint_max_nfev=joint_max_nfev, joint_max_seconds=joint_max_seconds, joint_gtol=joint_gtol)
    else:
        # Recompute noise for BM3; never transfer BM1's fitted sigma_x unchanged.
        fitted = {**fitted, 'restoring_strength': model.restoring_strength,
                  'f': lsq_params.response(fitted['alpha']), 'g': lsq_params.response(fitted['beta'])}
    diffusion = historical_diffusion_params().fit_many(training, fitted, substeps=substeps)
    future = checks.window(frame, origin, output_model, hours=horizon+1, observations=False)
    future = {key: value[:horizon+1] for key, value in future.items()}
    observed = checks.observation(frame, origin)
    x = model.fokker_planc_reference.grid(x_steps, state_bounds)[1]
    output0 = np.array([lsq_params._demand(float(z), float(future['B'][0]),
                                        float(future['U'][0]), fitted) for z in x])
    posterior = posterior_mass(fitted['initial_states'], x, output0, observed,
        diffusion['observation_sd'], prior_bandwidth)
    supports, masses, fp = distribution_forecast(future, fitted, posterior, diffusion['sigma_x'],
        fp_max_step=fp_max_step, substeps=substeps, state_bounds=state_bounds, tail_tolerance=tail_tolerance)
    intervals = predictive_intervals(supports[1:], masses[1:])
    rows = []
    for lead in range(1, horizon+1):
        row = {'origin': origin, 'origin_day': origin.normalize(), 'origin_hour': origin.hour,
            'target_time': origin+pd.Timedelta(hours=lead), 'lead_hours': lead,
            'I_BM3': float(np.dot(supports[lead], masses[lead])), 'B': float(future['B'][lead]),
            'sigma_x': diffusion['sigma_x'], 'diffusion_q': diffusion['q'],
            'observation_sd': diffusion['observation_sd'], 'historical_fit_rmse': diffusion['historical_rmse'],
            'historical_diffusion_variance_rmse': diffusion['variance_rmse'],
            'prior_state_mean': float(np.mean(fitted['initial_states'])),
            'posterior_state_mean': float(x @ posterior),
            'posterior_state_std': float(np.sqrt(np.dot((x-x @ posterior)**2, posterior))),
            'X_predicted_mean': float(fp['state_mean'][lead]),
            'X_predicted_std': float(np.sqrt(fp['state_variance'][lead])),
            'outside_probability': float(fp['outside_probability'][lead]),
            'auxiliary_flow_mean': float(fp['auxiliary_flow_mean'][lead]),
            'restoring_strength': restoring_strength, 'pooled_training_mse': fitted['mse'],
            'fit_iterations': fitted['iterations'], 'fit_converged': bool(fitted['converged']),
            'fit_stop_reason': fitted['stop_reason'], 'fit_message': fitted['message'],
            'theta_nfev': fitted['theta_nfev']}
        for level, (lo, hi) in intervals.items():
            row[f'lower_{level}'], row[f'upper_{level}'] = float(lo[lead-1]), float(hi[lead-1])
        rows.append(row)
    return rows, {'fit': fitted, 'diffusion': diffusion, 'posterior': posterior, 'fp': fp,
        'training': training, 'training_origins': origins, 'future': future,
        'origin_observed_I': observed, 'state_bounds': state_bounds,
        'tail_tolerance': tail_tolerance, 'prior_bandwidth': prior_bandwidth}


def predict_day(frame, day, *, hours=range(24), skip_errors=False, **options):
    day = checks.origin(pd.Timestamp(day).normalize()); frame = checks.frame(frame)
    baseline = statistical_output().fit(frame, day)
    rows, diagnostics, skipped = [], [], []
    for hour in hours:
        origin = day+pd.Timedelta(hours=hour)
        try:
            predicted, info = predict_origin(frame, origin, baseline, **options)
            rows.extend(predicted)
            diagnostics.append({'origin': origin, **info['diffusion'],
                'fit_converged': info['fit']['converged'], 'fit_stop_reason': info['fit']['stop_reason'],
                'maximum_numerical_tail_mass': info['fp']['diagnostics'].get('maximum_numerical_tail_mass', 0.)})
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            if not skip_errors: raise
            skipped.append((origin, str(exc)))
    return pd.DataFrame(rows), pd.DataFrame(diagnostics), skipped


def main():
    from src.exe.run_model.wm_4.fit_daily import aligned_wm4_data
    from src.validation.exe.run_model.compare_model_predictions import model_comparison_validation as scale
    from src.exe.run_model.bm_1.compare_fg_week import write_json
    p = ArgumentParser(description='BM3 MPC: refit five historical windows each hour; predict 1..8h.')
    p.add_argument('--day', default='2022-08-22'); p.add_argument('--tag', default='MED')
    p.add_argument('--horizon', type=int, default=8); p.add_argument('--X0', type=float, default=.5)
    p.add_argument('--hours', nargs='+', type=int, choices=range(24), default=list(range(24)))
    p.add_argument('--x-steps', type=int, default=301); p.add_argument('--fp-max-step', type=float, default=.25)
    p.add_argument('--max-iterations', type=int, default=3); p.add_argument('--max-nfev', type=int, default=120)
    p.add_argument('--state-maxiter', type=int, default=60); p.add_argument('--prior-bandwidth', type=float, default=.025)
    p.add_argument('--min-capacity', type=float, default=.05)
    p.add_argument('--joint-max-nfev', type=int, default=0); p.add_argument('--joint-max-seconds', type=float, default=20.)
    p.add_argument('--joint-gtol', type=float, default=1e-5)
    p.add_argument('--skip-errors', action='store_true'); p.add_argument('--output', type=Path, default=OUTPUT)
    add_model_options(p); args = p.parse_args(); opts = vars(args).copy()
    output = opts.pop('output'); tag = opts.pop('tag')
    opts['state_bounds'] = (opts.pop('state_min'), opts.pop('state_max'))
    raw, _ = aligned_wm4_data(tag)
    normalized, calibration = scale.bm1_scale(raw, pd.Timestamp(args.day))
    forecasts, diagnostics, skipped = predict_day(normalized, **opts)
    if forecasts.empty: raise RuntimeError('BM3 MPC did not complete any origins.')
    for name in ('I_BM3','B','lower_68','upper_68','lower_95','upper_95'):
        forecasts[name] = scale.bm1_to_wm4(forecasts[name].to_numpy(), calibration)
    forecasts['I_observed'] = raw.I.reindex(pd.DatetimeIndex(forecasts.target_time)).to_numpy()
    folder = output/f'{pd.Timestamp(args.day):%Y-%m-%d}'; folder.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(folder/'bm3_mpc_forecasts.csv', index=False)
    diagnostics.to_csv(folder/'historical_diffusion.csv', index=False)
    pd.DataFrame(skipped, columns=['origin','reason']).to_csv(folder/'skipped_origins.csv', index=False)
    scores = []
    for lead, group in forecasts.groupby('lead_hours'):
        bounds = {p: (group[f'lower_{p}'].to_numpy(), group[f'upper_{p}'].to_numpy()) for p in (68,95)}
        scores.append({'lead_hours': lead, **forecast_metrics(group.I_observed, group.I_BM3, bounds)})
    pd.DataFrame(scores).to_csv(folder/'rmse_coverage_by_lead.csv', index=False)
    write_json(folder/'settings.json', {**{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'model': 'BM3', 'diffusion': 'additive', 'auxiliary_flow_units': 'BM normalized demand'})
    print(pd.DataFrame(scores).to_string(index=False)); print('Saved:', folder)
    return forecasts


if __name__ == '__main__': main()
