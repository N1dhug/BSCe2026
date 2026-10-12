"""BM3 day-ahead forecasts using BM1's data preparation, fitter and result writer."""
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.exe.run_model.bm_1 import predict_daily as shared
from src.model.bm_3 import bm_3
from src.params.bm_3 import lsq_params
from src.tools.forecast_metrics import predictive_intervals, weighted_quantiles

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'files'/'output'/'bm_3'/'prediction'


def forecast_from_distribution(data, fitted, *, sigma_x=0.0, x_steps=301,
                               fp_max_step=0.25, substeps=2,
                               state_bounds=(-1.0, 2.0), tail_tolerance=1e-4):
    states = np.asarray(fitted['initial_states'], float)
    parameters = {**fitted, 'restoring_strength': fitted.get('restoring_strength', 10.0)}
    model = bm_3(shared.held_signal(data['B']), lambda t: 0.5, (0., 24.), 25,
        restoring_strength=parameters['restoring_strength'], state_bounds=state_bounds,
        tail_tolerance=tail_tolerance)
    trajectories = [lsq_params._predict(data, float(x), parameters, substeps) for x in states]
    paths = np.asarray([d[:24] for x, d in trajectories])
    state_paths = np.asarray([x[:24] for x, d in trajectories])
    mass0 = model.initial_mass(states, x_steps)
    x_grid = model.fokker_planc_reference.grid(x_steps, state_bounds)[1]
    supports = list(paths.T); probabilities = [np.full(len(states), 1/len(states))]*24
    fp = None
    if sigma_x > 0:
        args = [shared.held_signal(data['U']), fitted['f'], fitted['g'],
                fitted['C'], fitted['k'], fitted['Xi']]
        result = model.fokker_planc_reference_solver(mass0, args, sigma_x,
            t_eval=data['Time'], max_step=fp_max_step)
        t, x, density, mass, ex, vx, u, b, expected, diagnostics = result
        supports = [model.eval(float(i), x, *[args[j] for j in (0, 1, 2, 4, 5)])[-1] for i in range(24)]
        probabilities = list(mass[:24])
        fp = {'x': x, 'cell_mass': mass, 'density': density, 'state_mean': ex,
              'state_variance': vx, 'diagnostics': diagnostics}
    forecast = np.array([np.dot(d, w) for d, w in zip(supports, probabilities)])
    limits = np.array([weighted_quantiles(d, w, [0.1, 0.9]) for d, w in zip(supports, probabilities)])
    return {'forecast': forecast, 'lower': limits[:, 0], 'upper': limits[:, 1],
        'intervals': predictive_intervals(supports, probabilities),
        'ensemble': paths, 'state_paths': state_paths, 'initial_mass': mass0,
        'initial_grid': x_grid, 'fp': fp, 'demand_supports': np.asarray(supports),
        'probabilities': np.asarray(probabilities)}


def predict_day(day='2022-08-22', *, restoring_strength=10.0,
                state_bounds=(-1.0, 2.0), tail_tolerance=1e-4, x_steps=301, **kwargs):
    factory = partial(bm_3, restoring_strength=restoring_strength,
                      state_bounds=state_bounds, tail_tolerance=tail_tolerance)
    result = shared.predict_day(day, x_steps=x_steps, _model_factory=factory,
        _fitter_factory=lsq_params,
        _forecast=partial(forecast_from_distribution, state_bounds=state_bounds,
                          tail_tolerance=tail_tolerance), _model_name='BM3', **kwargs)
    result['model_configuration'] = {'restoring_strength': restoring_strength,
        'restoring_strength_fitted': False, 'diffusion': 'additive', 'state_bounds': list(state_bounds),
        'tail_tolerance': tail_tolerance, 'demand_bounds': [0., 1.],
        'x_steps': x_steps, 'fp_max_step': kwargs.get('fp_max_step', .25),
        'energy_balance': 'C*dX/dt = D-B + C*R(X); auxiliary flow is not reported as demand'}
    fp = result['ensemble']['fp']
    hourly = result['hourly']
    if fp is not None:
        hourly['X_predicted_mean'] = fp['state_mean'][:24]
        hourly['X_predicted_std'] = np.sqrt(fp['state_variance'][:24])
        hourly['outside_probability'] = fp['diagnostics']['outside_probability'][:24]
        hourly['auxiliary_flow_mean'] = fp['diagnostics']['auxiliary_flow_mean'][:24]
        result['model_configuration']['fp_diagnostics'] = json.loads(json.dumps(
            fp['diagnostics'], default=lambda value: value.tolist(), allow_nan=False))
    else:
        states = result['ensemble']['state_paths']
        hourly['X_predicted_mean'] = states.mean(axis=0)
        hourly['X_predicted_std'] = states.std(axis=0)
        hourly['outside_probability'] = ((states < 0) | (states > 1)).mean(axis=0)
        hourly['auxiliary_flow_mean'] = result['fit']['C']*bm_3.restoring(states, restoring_strength).mean(axis=0)
    return result


def save_result(result, output_root=OUTPUT, plot=True):
    directory = shared.save_result(result, output_root=output_root, plot=plot)
    ensemble = result['ensemble']
    np.savez_compressed(directory/'distribution.npz', demand_supports=ensemble['demand_supports'],
        probabilities=ensemble['probabilities'], initial_grid=ensemble['initial_grid'],
        initial_mass=ensemble['initial_mass'])
    if plot:
        import matplotlib.pyplot as plt
        h = result['hourly']
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(h.index, h.X_predicted_mean, label='Mean X')
        axes[0].fill_between(h.index, h.X_predicted_mean-h.X_predicted_std,
            h.X_predicted_mean+h.X_predicted_std, alpha=.2, label='Mean +/- one SD')
        for boundary in (0, 1): axes[0].axhline(boundary, color='black', linestyle='--')
        axes[0].set_ylabel('State X'); axes[0].legend()
        axes[1].plot(h.index, h.outside_probability, label='P(X<0 or X>1)')
        axes[1].set_ylabel('Probability'); axes[1].legend()
        for ax in axes: ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(directory/'state_boundary.png', dpi=160); plt.close(fig)
        from src.exe.run_model.bm_3.plot_prediction import plot_saved
        plot_saved(directory)
    return directory


def add_model_options(parser):
    parser.add_argument('--restoring-strength', type=float, default=10.0, help='Fixed lambda, in inverse model hours.')
    parser.add_argument('--state-min', type=float, default=-1.0, help='Numerical FV truncation below 0.')
    parser.add_argument('--state-max', type=float, default=2.0, help='Numerical FV truncation above 1.')
    parser.add_argument('--tail-tolerance', type=float, default=1e-4)


def main():
    parser = ArgumentParser(description='BM3: five-weekday logistic fit and 24h forecast with soft state limits.')
    parser.add_argument('--day', default='2022-08-22'); parser.add_argument('--tag', default='MED')
    parser.add_argument('--X0', type=float, default=.5)
    parser.add_argument('--C', type=float); parser.add_argument('--Xi', type=float)
    parser.add_argument('--max-iterations', type=int, default=6)
    parser.add_argument('--max-nfev', type=int, default=150)
    parser.add_argument('--state-maxiter', type=int, default=100)
    parser.add_argument('--joint-max-nfev', type=int, default=0)
    parser.add_argument('--joint-max-seconds', type=float, default=20.)
    parser.add_argument('--joint-gtol', type=float, default=1e-5)
    parser.add_argument('--sigma-x', type=float, default=0., help='Additive noise amplitude; 0 is deterministic.')
    parser.add_argument('--x-steps', type=int, default=301)
    parser.add_argument('--fp-max-step', type=float, default=.25)
    parser.add_argument('--score', action='store_true'); parser.add_argument('--save', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    add_model_options(parser)
    args = parser.parse_args(); options = vars(args).copy()
    output = options.pop('output'); save = options.pop('save')
    options['state_bounds'] = (options.pop('state_min'), options.pop('state_max'))
    result = predict_day(**options)
    print('BM3 parameters:', {k: result['fit'][k] for k in ('alpha','beta','k','Xi','C','restoring_strength')})
    print('Fit converged:', result['fit']['converged'], result['fit']['stop_reason'])
    print('Held-out metrics:', result['metrics'])
    if save: print('Saved:', save_result(result, output))
    return result


if __name__ == '__main__': main()
