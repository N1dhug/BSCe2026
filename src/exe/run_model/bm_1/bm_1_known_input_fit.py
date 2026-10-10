"""Fit C/k/Xi and unknown X0 on one continuous BM1 trajectory with known inputs.

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


def fit_known_inputs(baseline_time, baseline, prices, demand, x0=None, *,
                     initial=(12., 2., .5), bounds=((1., .05, 0.), (744., 10., 1.)),
                     initial_x0=.5, c=None,
                     observation_contract='hourly-mean', steps_per_hour=12,
                     max_nfev=500, numerical_tolerance=1e-5, observation_mask=None):
    """Return parameters, fitted demand/state, residuals, RMSE and stop status.

    prices/demand have N hourly values. baseline_time/baseline describe a
    piecewise-linear baseline over [0,N], including every integer hour edge.
    Hourly prices are held on [i,i+1). With x0=None, X0 is estimated jointly
    with C/k/Xi in [0,1], starting at initial_x0. An explicit x0 is held fixed.
    An explicit c fixes capacity in hours and fits only the other parameters.
    There are no daily resets or warmup. Demand observations enter the fit
    objective, not the state propagation. The objective
    is unweighted least squares; C and k are optimized in log coordinates.
    Bounds and initial contain [C_hours,k,Xi]. No baseline or curve fitting.
    observation_mask excludes unmeasured demand from the objective/RMSE while
    retaining every hourly input and the continuous state trajectory.
    """
    begun = perf_counter()
    initial = np.asarray(initial, float).copy()
    lower, upper = np.asarray(bounds, float)
    if (initial.shape != (3,) or lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite(np.r_[initial, lower, upper]).all()
            or np.any(lower >= upper) or np.any(initial < lower) or np.any(initial > upper)
            or np.any(lower[:2] <= 0) or lower[2] < 0 or upper[2] > 1):
        raise ValueError('Use finite valid initial/bounds for [C_hours,k,Xi].')
    if c is not None:
        if not np.isfinite(c) or not lower[0] <= c <= upper[0]:
            raise ValueError("Fixed C must be finite and within the capacity bounds.")
        initial[0] = c
    estimate_x0 = x0 is None
    start_x0 = initial_x0 if estimate_x0 else x0
    if not np.isfinite(start_x0) or not 0 <= start_x0 <= 1:
        raise ValueError('X0 or its initial guess must be finite and in [0,1].')
    free = np.array([c is None, True, True, estimate_x0])
    parameter_names = np.array(['C_hours', 'k', 'Xi', 'x0'])[free].tolist()
    full_initial = np.r_[initial, start_x0]
    if isinstance(max_nfev, bool) or int(max_nfev) != max_nfev or max_nfev < 1:
        raise ValueError('max_nfev must be a positive integer.')
    if not np.isfinite(numerical_tolerance) or numerical_tolerance <= 0:
        raise ValueError('numerical_tolerance must be positive and finite.')
    observed = np.asarray(demand, float)
    inputs = BM1DeterministicInputs(baseline_time, baseline, prices, observed,
        observation_contract=observation_contract, steps_per_hour=steps_per_hour,
        observation_mask=observation_mask)
    measured = inputs.observation_mask
    if np.count_nonzero(measured) < len(parameter_names):
        raise ValueError(f'At least {len(parameter_names)} demand observations are required '
                         f'for {len(parameter_names)} fitted parameters.')
    if not np.isfinite(observed[measured]).all() or np.any((observed[measured] < 0) | (observed[measured] > 1)):
        raise ValueError('Measured demand must be finite and normalized to [0,1].')
    if upper[1] * upper[2] * inputs.h / (lower[0] if c is None else c) > 1:
        raise ValueError('Increase steps_per_hour for the requested parameter bounds.')

    def encode(p):
        return np.r_[np.log(p[:2]), p[2:]][free]

    def decode(v):
        values = np.r_[np.log(full_initial[:2]), full_initial[2:]]
        values[free] = v
        p = np.r_[np.exp(values[:2]), values[2:]]
        if c is not None: p[0] = float(c)
        return p

    calls = 0

    def residual(v):
        nonlocal calls
        calls += 1
        return (inputs.solve(decode(v))['demand'] - observed)[measured]

    initial_prediction = inputs.solve(np.r_[initial, start_x0])['demand']
    fitted_initial = full_initial
    fitted_lower = np.r_[lower, 0.]
    fitted_upper = np.r_[upper, 1.]
    fit = least_squares(residual, encode(fitted_initial),
        bounds=(encode(fitted_lower), encode(fitted_upper)),
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
        'x0': float(p[3]), 'x0_estimated': estimate_x0, 'c_estimated': c is None,
        'response': {'f': '1-2X', 'g': '1-2u'},
        'observation_contract': observation_contract, 'time_unit': 'hours',
        'state_time': np.arange(inputs.n_hours + 1, dtype=float), 'state': solved['state'],
        'price': inputs.prices.copy(), 'observed_demand': observed.copy(),
        'observation_mask': measured.copy(),
        'baseline': inputs.b_mean.copy() if observation_contract == 'hourly-mean' else
                    np.interp(np.arange(inputs.n_hours), inputs.time, inputs.baseline),
        'demand': solved['demand'], 'residuals': errors,
        'rmse': float(np.sqrt(np.mean(errors[measured]**2))),
        'initial_rmse': float(np.sqrt(np.mean((initial_prediction[measured]-observed[measured])**2))),
        'fit_settings': {'parameter_names': parameter_names,
                         'initial': fitted_initial[free].tolist(), 'initial_x0': float(start_x0),
                         'lower_bounds': fitted_lower[free].tolist(),
                         'upper_bounds': fitted_upper[free].tolist(), 'max_nfev': int(max_nfev)},
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path,
        help='CSV: time_hours,baseline,price,demand; last row is a baseline-only endpoint.')
    source.add_argument('--start', help='Historical window start in Europe/Copenhagen local time.')
    parser.add_argument('--end', help='Exclusive historical window end.')
    parser.add_argument('--baseline-csv', type=Path,
        help='Optional raw baseline: Datetime,baseline including final edge; otherwise fit WM4 baseline.')
    parser.add_argument('--input-workbook', type=Path)
    parser.add_argument('--prices-workbook', type=Path)
    parser.add_argument('--tag', default='MED')
    parser.add_argument('--price-area', default='DK1')
    parser.add_argument('--demand-label', choices=('hour_start', 'hour_end'), default='hour_start',
        help='Declared timestamp convention for historical demand; default: hour_start.')
    parser.add_argument('--c', type=float,
        help='Hold capacity C fixed in hours (e.g. --c 5); omit to estimate C.')
    parser.add_argument('--x0', type=float,
        help='Hold a known X0 fixed; omit to estimate X0 jointly with C/k/Xi.')
    parser.add_argument('--initial-x0', type=float, default=.5,
        help='Initial guess for unknown X0 (default: 0.5; unused with --x0).')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--initial', nargs=3, type=float, default=(12., 2., .5), metavar=('C', 'K', 'XI'))
    parser.add_argument('--contract', choices=('hourly-mean', 'hourly-point'), default='hourly-mean')
    parser.add_argument('--steps-per-hour', type=int, default=12)
    parser.add_argument('--max-nfev', type=int, default=500)
    parser.add_argument('--demand-offset', type=float)
    parser.add_argument('--demand-scale', type=float)
    parser.add_argument('--price-offset', type=float)
    parser.add_argument('--price-scale', type=float)
    args = parser.parse_args(argv)
    if args.input is not None and not args.input.is_file():
        parser.error(f'Input CSV not found: {args.input}. Supply an existing CSV; see docs/bm/KNOWN_INPUT_FIT.md.')
    if args.output.exists():
        parser.error('Use a new output directory.')
    names = ('demand_offset', 'demand_scale', 'price_offset', 'price_scale')
    metadata, measured = None, None
    if args.start is not None:
        from src.exe.run_model.bm_1.data import aligned_bm_data, fit_inputs
        if args.end is None:
            parser.error('Historical import requires --end.')
        paths = {}
        if args.input_workbook is not None: paths['input_path'] = args.input_workbook
        if args.prices_workbook is not None: paths['prices_path'] = args.prices_workbook
        frame = aligned_bm_data(args.start, args.end, tag=args.tag,
            price_area=args.price_area, demand_label=args.demand_label, **paths)
        data = fit_inputs(frame, args.baseline_csv, steps_per_hour=args.steps_per_hour,
                          **{name: getattr(args, name) for name in names})
        time, baseline, price, demand = (data[name] for name in ('baseline_time', 'baseline', 'prices', 'demand'))
        metadata, measured = data['metadata'], data['observation_mask']
        for name, value in data['normalization'].items(): setattr(args, name, value)
    else:
        for name, default in zip(names, (0., 1., 0., 1.)):
            if getattr(args, name) is None: setattr(args, name, default)
        scales = np.array([getattr(args, name) for name in names])
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
    result = fit_known_inputs(time-time[0], baseline, price, demand, args.x0, observation_mask=measured,
        initial=args.initial, initial_x0=args.initial_x0, c=args.c, observation_contract=args.contract,
        steps_per_hour=args.steps_per_hour, max_nfev=args.max_nfev)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'trajectory.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_hours', 'observed_demand', 'fitted_demand', 'baseline', 'price', 'X', 'residual'])
        for i in range(len(demand)):
            observed = args.demand_offset+args.demand_scale*demand[i] if result['observation_mask'][i] else ''
            residual = args.demand_scale*result['residuals'][i] if result['observation_mask'][i] else ''
            writer.writerow([time[0]+i, observed,
                args.demand_offset+args.demand_scale*result['demand'][i],
                args.demand_offset+args.demand_scale*result['baseline'][i],
                args.price_offset+args.price_scale*price[i], result['state'][i],
                residual])
        writer.writerow([time[-1], '', '', '', '', result['state'][-1], ''])
    summary = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    summary.update(rmse_normalized=result['rmse'], initial_rmse_normalized=result['initial_rmse'],
        rmse=args.demand_scale*result['rmse'], initial_rmse=args.demand_scale*result['initial_rmse'],
        normalization={'demand_offset': args.demand_offset, 'demand_scale': args.demand_scale,
                       'price_offset': args.price_offset, 'price_scale': args.price_scale},
        input_time_start=float(time[0]), input_time_end=float(time[-1]),
        final_state=float(result['state'][-1]), initial_parameters=list(args.initial))
    if metadata is not None: summary['data_import'] = metadata
    with (args.output/'fit.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    from src.exe.run_model.bm_1.plot_fit import save_plots
    save_plots(args.output)
    print(f"{result['status']}: RMSE={summary['rmse']:.8g}; X0={result['x0']:.8g}; {args.output}")
    return 0 if result['status'] == 'converged' else 1


if __name__ == '__main__':
    raise SystemExit(main())
