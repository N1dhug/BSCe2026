"""Generate BM1 figures, or redraw stored I/figure outputs without estimation.

reference: existing F2 input scenario, one FV solve and deterministic comparison.
i-report: only the seven stored I-baseline horizons, without FV or optimizer calls.
redraw: reuse exported full joint distributions, without any numerical solve.
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.tools.bm_1_outputs import export_distribution_outputs


ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT / 'files/output/bm_1/runs/outputs'


def _destination(args):
    path = args.output_dir or OUTPUT_DIR / (args.case + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}.')
    return path


def run_reference(args):
    from src.control.bm_1 import beta_initial_cell_mass
    from src.validation.bm_1 import cases
    model, parameters = cases.build_reference_case()
    times = np.unique(np.r_[np.linspace(*cases.T, 193), cases.PRICE_BREAKPOINTS])
    edges, x, _ = model.fokker_planc_reference.grid(args.cells)
    initial = beta_initial_cell_mass(edges, alpha=cases.INITIAL_ALPHA, beta=cases.INITIAL_BETA)
    output = model.fokker_planc_reference_solver(initial, parameters, cases.SIGMA_X,
        t_eval=times, breakpoints=cases.PRICE_BREAKPOINTS, max_step=cases.MAX_STEP)
    deterministic = model.deterministic_solver(float(initial @ x), parameters,
        t_eval=times, breakpoints=cases.PRICE_BREAKPOINTS, max_step=cases.MAX_STEP,
        rtol=1e-9, atol=1e-11)
    demand = np.asarray(model.eval(times[:, None], x[None, :],
                                  *(parameters[:3] + parameters[4:]))[-1])
    metadata = dict(interpretation='F2 reference simulation; fixed parameters; not real-data estimation',
        demand_unit='Normalized demand', includes_parameter_uncertainty=False,
        includes_curve_uncertainty=False, observations_used=0,
        parameters=dict(C=cases.C, k=cases.K, Xi=cases.XI, sigma_x=cases.SIGMA_X),
        curves=dict(f_knots=parameters[1].x.tolist(), f_values=parameters[1](parameters[1].x).tolist(),
                    g_knots=parameters[2].x.tolist(), g_values=parameters[2](parameters[2].x).tolist()),
        inputs=dict(price_edges=cases.PRICE_EDGES.tolist(), prices=cases.PRICE_VALUES.tolist(),
                    baseline=cases.BASELINE, initial_beta=[cases.INITIAL_ALPHA, cases.INITIAL_BETA]),
        numerics=dict(cells=args.cells, max_step=cases.MAX_STEP), diagnostics=output[-1])
    # Keep diagnostic arrays out of JSON; the solver's scalar controls remain declared.
    metadata['diagnostics'] = {k: v.item() if isinstance(v, np.generic) else v
                              for k, v in output[-1].items() if np.isscalar(v)}
    metadata['source_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in ('src/model/bm_1.py', 'src/solver/finite_volume.py', 'src/validation/bm_1/cases.py')}
    return dict(times=times, x_edges=edges, joint_mass=output[3], demand_values=demand,
                baseline=np.asarray(model.B(times)), price=np.asarray(parameters[0](times)),
                mode='reference_simulation', metadata=metadata,
                price_edges=cases.PRICE_EDGES, price_levels=cases.PRICE_VALUES,
                deterministic_state=deterministic[1], deterministic_demand=deterministic[-1])


def run_i_report(args):
    from src.model.bm_1 import bm_1
    from src.tools.input_generator.input_builder import input_builder
    from src.validation.bm_1.cases import linear_state_response, linear_price_response
    raw = args.input.read_bytes()
    report = json.loads(gzip.decompress(raw) if args.input.suffix == '.gz' else raw)
    if report.get('stage') != 'I-baseline-development-v1':
        raise ValueError('Expected an I-baseline-development-v1 report; no guessed schema.')
    forecast = report['forecast'][args.belief]
    origin = report['origins'][args.belief]
    p = origin['parameters']
    if p['state_response'] != '1-2*x' or p['price_response'] != '1-2*u':
        raise ValueError('This report adapter supports only the declared linear reference curves.')
    times, edges = np.asarray(forecast['times']), np.asarray(forecast['x_edges'])
    x = (edges[:-1] + edges[1:]) / 2
    future = report['future_inputs']
    B = np.full(times.shape, future['baseline'])
    U = input_builder.price_intervals((np.asarray(future['price_edges']), np.asarray(future['prices'])))
    model = bm_1(lambda t: np.full(np.asarray(t).shape, future['baseline']),
                 lambda t: np.full(np.asarray(t).shape, future['baseline']),
                 (origin['time'], times[-1]), 2)
    demand = np.asarray(model.eval(times[:, None], x[None, :], U, linear_state_response,
                                  linear_price_response, p['k'], p['Xi'])[-1])
    metadata = dict(interpretation='I baseline: ' + args.belief + '; seven stored forecast horizons only',
        demand_unit='Normalized demand', input_sha256=hashlib.sha256(raw).hexdigest(),
        source_commit=report.get('source_commit'), information_cutoff=origin['time'],
        future_observations_assimilated=0, includes_parameter_uncertainty=False,
        includes_curve_uncertainty=False, forecast_scientific_acceptance=False,
        origin=origin, future_inputs=future, limitations=report['interpretation'])
    return dict(times=times, x_edges=edges, joint_mass=np.asarray(forecast['cell_mass']),
                demand_values=demand, baseline=B, price=U(times),
                time_unit=forecast['time_unit'], time_origin=origin['time'],
                price_edges=np.asarray(future['price_edges']), price_levels=np.asarray(future['prices']),
                mode='fixed_origin_forecast', metadata=metadata)


def run_redraw(args):
    summary_path = args.input / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('schema') != 'bm1-distribution-output-v1':
        raise ValueError('Expected bm1-distribution-output-v1; no guessed schema.')
    with np.load(args.input / 'distribution.npz', allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    return dict(**values, mode=summary['mode'], time_unit=summary['time_unit'], metadata=summary['metadata'])


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='case', required=True)
    ref = commands.add_parser('reference', help='Existing F2 scenario; one short FV solve.')
    ref.add_argument('--cells', type=int, default=201)
    ref.set_defaults(run=run_reference)
    stored = commands.add_parser('i-report', help='Render an existing I report, with no numerical replay.')
    stored.add_argument('--input', type=Path, required=True)
    stored.add_argument('--belief', choices=('known', 'rho_map'), default='rho_map')
    stored.set_defaults(run=run_i_report)
    redraw = commands.add_parser('redraw', help='Render exported raw distributions again; no simulation.')
    redraw.add_argument('--input', type=Path, required=True, help='Previous figure-output directory.')
    redraw.set_defaults(run=run_redraw)
    for command in (ref, stored, redraw):
        command.add_argument('--output-dir', type=Path, help='New directory; never overwritten.')
    args = parser.parse_args(argv)
    path = _destination(args)
    started = perf_counter()
    values = args.run(args)
    result = export_distribution_outputs(path, **values)
    print(f'Saved BM1 outputs to {path.resolve()} ({perf_counter() - started:.2f} s).')
    print('Figures: bm_state_density.png, bm_response.png, bm_demand.png')
    print('Raw data: moments.csv, distribution.npz, summary.json')
    print(result['metadata']['interpretation'])


if __name__ == '__main__':
    main()
