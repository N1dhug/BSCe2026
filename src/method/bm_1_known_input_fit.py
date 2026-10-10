"""Fit C/k/Xi on one continuous BM1 trajectory with known inputs and fixed X0.

f(X)=1-2X and g(u)=1-2u. Time is physical elapsed hours; model inputs
are normalized to [0,1]. Run this module with --help for the CSV interface.
"""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from src.method.bm_1_deterministic_fit import BM1DeterministicInputs


def fit_known_inputs(baseline_time, baseline, prices, demand, x0, *,
                     initial=(12., 2., .5), bounds=((1., .05, 0.), (744., 10., 1.)),
                     observation_contract='hourly-mean', steps_per_hour=12,
                     max_nfev=500, numerical_tolerance=1e-5):
    """Return parameters, fitted demand/state, residuals, RMSE and stop status.

    prices/demand have N hourly values. baseline_time/baseline describe a
    piecewise-linear baseline over [0,N], including every integer hour edge.
    Hourly prices are held on [i,i+1). X0 is fixed, without daily resets or
    warmup. Demand observations never enter state propagation. The objective
    is unweighted least squares; C and k are optimized in log coordinates.
    Bounds and initial contain [C_hours,k,Xi]. No baseline or curve fitting.
    """
    begun = perf_counter()
    initial = np.asarray(initial, float)
    lower, upper = np.asarray(bounds, float)
    if (initial.shape != (3,) or lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite(np.r_[initial, lower, upper]).all()
            or np.any(lower >= upper) or np.any(initial < lower) or np.any(initial > upper)
            or np.any(lower[:2] <= 0) or lower[2] < 0 or upper[2] > 1):
        raise ValueError('Use finite valid initial/bounds for [C_hours,k,Xi].')
    if not np.isfinite(x0) or not 0 <= x0 <= 1:
        raise ValueError('X0 must be a fixed finite value in [0,1].')
    if isinstance(max_nfev, bool) or int(max_nfev) != max_nfev or max_nfev < 1:
        raise ValueError('max_nfev must be a positive integer.')
    if not np.isfinite(numerical_tolerance) or numerical_tolerance <= 0:
        raise ValueError('numerical_tolerance must be positive and finite.')
    observed = np.asarray(demand, float)
    inputs = BM1DeterministicInputs(baseline_time, baseline, prices, observed,
        observation_contract=observation_contract, steps_per_hour=steps_per_hour)
    if inputs.n_hours < 3:
        raise ValueError('At least three demand observations are required for three parameters.')
    if not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 1)):
        raise ValueError('Demand must be finite and normalized to [0,1].')
    if upper[1] * upper[2] * inputs.h / lower[0] > 1:
        raise ValueError('Increase steps_per_hour for the requested parameter bounds.')

    def encode(p):
        return np.r_[np.log(p[:2]), p[2]]

    def decode(v):
        return np.r_[np.exp(v[:2]), v[2], x0]

    calls = 0

    def residual(v):
        nonlocal calls
        calls += 1
        return inputs.solve(decode(v))['demand'] - observed

    initial_prediction = inputs.solve(np.r_[initial, x0])['demand']
    fit = least_squares(residual, encode(initial), bounds=(encode(lower), encode(upper)),
        method='trf', jac='2-point', diff_step=1e-4, x_scale='jac', loss='linear',
        ftol=1e-7, xtol=1e-7, gtol=1e-7, max_nfev=int(max_nfev))
    p = decode(fit.x)
    solved = inputs.solve(p)
    fine = inputs.refined().solve(p)
    numerical_error = float(np.max(np.abs(solved['demand'] - fine['demand'])))
    errors = solved['demand'] - observed
    numerical_passed = numerical_error <= numerical_tolerance
    status = 'converged' if fit.success and numerical_passed else (
        'numerics-failed' if not numerical_passed else 'not-converged')
    return {
        'status': status, 'parameters': {'C_hours': float(p[0]), 'k': float(p[1]), 'Xi': float(p[2])},
        'x0': float(x0), 'response': {'f': '1-2X', 'g': '1-2u'},
        'observation_contract': observation_contract, 'time_unit': 'hours',
        'state_time': np.arange(inputs.n_hours + 1, dtype=float), 'state': solved['state'],
        'price': inputs.prices.copy(), 'observed_demand': observed.copy(),
        'baseline': inputs.b_mean.copy() if observation_contract == 'hourly-mean' else
                    np.interp(np.arange(inputs.n_hours), inputs.time, inputs.baseline),
        'demand': solved['demand'], 'residuals': errors,
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'initial_rmse': float(np.sqrt(np.mean((initial_prediction-observed)**2))),
        'fit_settings': {'initial': initial.tolist(), 'lower_bounds': lower.tolist(),
                         'upper_bounds': upper.tolist(), 'max_nfev': int(max_nfev)},
        'optimizer': {'success': bool(fit.success), 'status': int(fit.status),
                      'message': str(fit.message), 'nfev': int(fit.nfev), 'residual_calls': calls,
                      'active_bounds': fit.active_mask.tolist()},
        'numerics': {'passed': numerical_passed, 'steps_per_hour': inputs.steps,
                     'refined_steps_per_hour': 2*inputs.steps,
                     'max_demand_difference': numerical_error, 'tolerance': numerical_tolerance},
        'seconds': perf_counter() - begun,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path,
        help='CSV: time_hours,baseline,price,demand; last row is a baseline-only endpoint.')
    parser.add_argument('--x0', required=True, type=float)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--initial', nargs=3, type=float, default=(12., 2., .5), metavar=('C', 'K', 'XI'))
    parser.add_argument('--contract', choices=('hourly-mean', 'hourly-point'), default='hourly-mean')
    parser.add_argument('--steps-per-hour', type=int, default=12)
    parser.add_argument('--max-nfev', type=int, default=500)
    parser.add_argument('--demand-offset', type=float, default=0.)
    parser.add_argument('--demand-scale', type=float, default=1.)
    parser.add_argument('--price-offset', type=float, default=0.)
    parser.add_argument('--price-scale', type=float, default=1.)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Use a new output directory.')
    scales = np.array([args.demand_offset, args.demand_scale, args.price_offset, args.price_scale])
    if not np.isfinite(scales).all() or args.demand_scale <= 0 or args.price_scale <= 0:
        parser.error('Offsets must be finite and scales must be positive and finite.')
    with args.input.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'time_hours', 'baseline', 'price', 'demand'} <= set(reader.fieldnames or ()):
            parser.error('CSV requires time_hours,baseline,price,demand.')
        rows = list(reader)
    if len(rows) < 2 or rows[-1]['price'].strip() or rows[-1]['demand'].strip():
        parser.error('Include a final baseline endpoint with empty price and demand.')
    time = np.array([float(row['time_hours']) for row in rows])
    if not np.isfinite(time).all() or not np.allclose(np.diff(time), 1., rtol=0., atol=1e-12):
        parser.error('CSV rows must follow consecutive physical elapsed hours.')
    baseline = (np.array([float(row['baseline']) for row in rows])-args.demand_offset)/args.demand_scale
    price = (np.array([float(row['price']) for row in rows[:-1]])-args.price_offset)/args.price_scale
    demand = (np.array([float(row['demand']) for row in rows[:-1]])-args.demand_offset)/args.demand_scale
    result = fit_known_inputs(time-time[0], baseline, price, demand, args.x0,
        initial=args.initial, observation_contract=args.contract,
        steps_per_hour=args.steps_per_hour, max_nfev=args.max_nfev)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'trajectory.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_hours', 'observed_demand', 'fitted_demand', 'baseline', 'price', 'X', 'residual'])
        for i in range(len(demand)):
            writer.writerow([time[i], args.demand_offset+args.demand_scale*demand[i],
                args.demand_offset+args.demand_scale*result['demand'][i],
                args.demand_offset+args.demand_scale*result['baseline'][i],
                args.price_offset+args.price_scale*price[i], result['state'][i],
                args.demand_scale*result['residuals'][i]])
        writer.writerow([time[-1], '', '', '', '', result['state'][-1], ''])
    summary = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    summary.update(rmse_normalized=result['rmse'], initial_rmse_normalized=result['initial_rmse'],
        rmse=args.demand_scale*result['rmse'], initial_rmse=args.demand_scale*result['initial_rmse'],
        normalization={'demand_offset': args.demand_offset, 'demand_scale': args.demand_scale,
                       'price_offset': args.price_offset, 'price_scale': args.price_scale},
        input_time_start=float(time[0]), input_time_end=float(time[-1]),
        final_state=float(result['state'][-1]), initial_parameters=list(args.initial))
    with (args.output/'fit.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(f"{result['status']}: RMSE={summary['rmse']:.8g}; {args.output}")
    return 0 if result['status'] == 'converged' else 1


if __name__ == '__main__':
    raise SystemExit(main())
