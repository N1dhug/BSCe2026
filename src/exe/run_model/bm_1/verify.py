"""Numerical BM1 reference experiments, with short reports by default.

Choose one case. --verbose adds diagnostic detail; --save creates local figures.
Model equations and experiment calculations live in model/method/validation.
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np

from src.control import bm_1 as control
from src.method.euler_maruyama import euler_maruyama
from src.validation.bm_1 import cases, control as control_cases, deterministic, mixture, stochastic


ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT / 'files' / 'output' / 'bm_1' / 'runs'


def _table(title, headings, rows):
    """Print a small aligned table; callers supply already formatted values."""
    rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(label), *(len(row[i]) for row in rows)) for i, label in enumerate(headings)]
    print(title)
    for row in [headings, *rows]:
        print('  '.join((value.ljust(width) for value, width in zip(row, widths))))


def _diagnostics(values, verbose):
    """Show scalar diagnostics only when requested; never hide raised errors."""
    if verbose:
        values = values if isinstance(values, dict) else vars(values)
        for name, value in values.items():
            if np.isscalar(value):
                print(f'  {name}: {value}')


def _save_lines(name, panels, *, loglog=False):
    """Save panels of (title, xlabel, ylabel, [(x, y, label), ...])."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.5 * len(panels)), squeeze=False)
    for axis, (title, xlabel, ylabel, lines) in zip(axes[:, 0], panels):
        for x, y, label in lines:
            axis.plot(x, y, label=label)
        if loglog:
            axis.set_xscale('log')
            axis.set_yscale('log')
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(loc='best')
    fig.tight_layout()
    folder = OUTPUT_DIR / 'graphs'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved graph to {path}')
    return path


def run_deterministic(args):
    """Compare ODE/Euler solutions with the analytic piecewise-price solution."""
    result = deterministic.run_deterministic_reference()
    _table(
        'F1 deterministic reference',
        ('solver', 'maximum state error'),
        [
            ('default ODE', f"{result['default_error']:.6e}"),
            ('segmented ODE', f"{result['segmented_error']:.6e}"),
            ('accurate ODE', f"{result['accurate_error']:.6e}"),
        ],
    )
    _table(
        'Euler refinement',
        ('step [days]', 'maximum state error', 'order'),
        [
            (f"{item['dt']:.6g}", f"{item['error']:.6e}", f'{order:.3f}')
             for item, order in zip(result['euler'], result['orders'])
        ],
    )
    _diagnostics(
        {
            'same_tolerance_improvement': result['same_tolerance_improvement'],
            'maximum_energy_residual': result['euler'][-1]['energy_error'],
        },
        args.verbose,
    )
    if args.save:
        t, state = result['accurate'][:2]
        _save_lines(
            'bm_1_deterministic_reference.png',
            [
                (
                    'Exact and computed state',
                    'Time [h]',
                    'X',
                    [
                        (24 * t, deterministic.exact_state(t), 'exact'),
                        (24 * t, state, 'segmented ODE'),
                        (24 * result['default'][0], result['default'][1], 'default ODE'),
                    ],
                ),
                (
                    'Euler error',
                    'step [days]',
                    'max state error',
                    [([r['dt'] for r in result['euler']], [r['error'] for r in result['euler']], 'Euler')],
                ),
            ],
        )
    return result


def run_feasibility(args):
    """Show why a target is unattainable before interpreting an optimizer residual."""
    result = deterministic.run_feasibility_reference()
    _table(
        'F1 feasibility: price bound versus capacity',
        ('case', 'working J', 'fine J'),
        [
            ('price bound', f"{result['price_objective']:.6e}", f"{result['price_fine_objective']:.6e}"),
            ('capacity', f"{result['capacity_objective']:.6e}", f"{result['capacity_fine_objective']:.6e}"),
        ],
    )
    print(
        (
            f"Required price range: [{np.min(result['required_price']):.6g}, "
            f"{np.max(result['required_price']):.6g}]; allowed range: [0, 1]"
        ),
    )
    _diagnostics(result['capacity_certificate'], args.verbose)
    if args.save:
        hours = 24 * result['t']
        _save_lines(
            'bm_1_optimization_feasibility.png',
            [
                (
                    'Price-limited tracking',
                    'Time [h]',
                    'Demand',
                    [
                        (hours, result['target_demand'], 'target'),
                        (hours, result['price_solution'][7], 'best bounded price'),
                    ],
                ),
                (
                    'Capacity-limited state',
                    'Time [h]',
                    'X',
                    [
                        (hours, result['capacity_required_state'], 'state required by target'),
                        (hours, result['capacity_solution'][1], 'computed state'),
                    ],
                ),
            ],
        )
    return result


def run_em_convergence(args):
    """Estimate strong error on coupled Brownian paths, not independent runs."""
    method = euler_maruyama()
    exact = stochastic.exact_logistic_benchmark(method)
    bm = stochastic.bm1_self_convergence(method)
    stochastic._require_no_boundary_violations('Exact benchmark', exact[4])
    stochastic._require_no_boundary_violations('BM1 self-convergence', bm[4])
    _table(
        'Coupled EM convergence',
        ('case', 'fitted order', 'finest measured error'),
        [
            ('analytic benchmark', f'{exact[3]:.3f}', f'{exact[1][-1]:.6e}'),
            ('BM1 self-convergence', f'{bm[3]:.3f}', f'{bm[1][-1]:.6e}'),
        ],
    )
    if args.verbose:
        for label, result in [('analytic', exact), ('BM1', bm)]:
            _table(
                label,
                ('step', 'RMS error', 'local order'),
                [
                    (f'{h:.6g}', f'{err:.6e}', '-' if index == 0 else f'{result[2][index - 1]:.3f}')
                    for index, (h, err) in enumerate(zip(result[0], result[1]))
                ],
            )
    if args.save:
        _save_lines(
            'bm_1_em_convergence.png',
            [
                (
                    'Coupled EM errors',
                    'step [days]',
                    'RMS state error',
                    [(exact[0], exact[1], 'analytic benchmark'), (bm[0], bm[1], 'BM1')],
                ),
            ],
            loglog=True,
        )
    return (exact, bm)


def run_em(args):
    """Propagate uncorrected SDE paths and report mean-state sampling uncertainty."""
    model, parameters = cases.build_reference_case()
    grid, _ = model.euler_maruyama.build_time_grid(
        cases.T_EVAL,
        breakpoints=cases.PRICE_BREAKPOINTS,
        max_step=cases.MAX_STEP,
    )
    increments = model.euler_maruyama.brownian_increments(grid, n_paths=args.mc_paths, seed=stochastic.SEED)
    result = model.euler_maruyama_solver(
        [cases.X0],
        parameters,
        cases.SIGMA_X,
        n_paths=args.mc_paths,
        boundary='none',
        t_eval=cases.T_EVAL,
        breakpoints=cases.PRICE_BREAKPOINTS,
        max_step=cases.MAX_STEP,
        brownian_increments=increments,
        return_diagnostics=True,
    )
    t, state, price, baseline, _, _, _, demand, diagnostics = result
    if diagnostics['total_violations']:
        raise RuntimeError('Uncorrected EM paths left [0, 1]; refine before using this reference.')
    mean_x, mean_d = (state.mean(axis=0), demand.mean(axis=0))
    _table(
        'EM terminal moments',
        ('quantity', 'mean', 'MC standard error'),
        [
            ('X(T)', f'{mean_x[-1]:.8f}', f'{state[:, -1].std(ddof=1) / np.sqrt(args.mc_paths):.3e}'),
            ('D(T)', f'{mean_d[-1]:.8f}', f'{demand[:, -1].std(ddof=1) / np.sqrt(args.mc_paths):.3e}'),
        ],
    )
    _diagnostics(diagnostics, args.verbose)
    if args.save:
        _save_lines(
            'bm_1_em_reference.png',
            [
                ('State and price', 'Time [h]', 'X / U', [(24 * t, mean_x, 'E[X]'), (24 * t, price, 'U')]),
                ('Expected demand', 'Time [h]', 'D', [(24 * t, mean_d, 'E[D]'), (24 * t, baseline, 'B')]),
            ],
        )
    return result


def run_fv(args):
    """Evolve probability masses and expose conservation diagnostics."""
    model, parameters = cases.build_reference_case()
    edges, _, _ = model.fokker_planc_reference.grid(args.cells)
    result = model.fokker_planc_reference_solver(
        control.beta_initial_cell_mass(edges),
        parameters,
        cases.SIGMA_X,
        t_eval=cases.T_EVAL,
        breakpoints=cases.PRICE_BREAKPOINTS,
        max_step=cases.MAX_STEP,
    )
    t, x, density, mass, mean, variance, price, baseline, demand, diagnostics = result
    print(f'FV: cells={args.cells}, E[X(T)]={mean[-1]:.8f}, E[D(T)]={demand[-1]:.8f}')
    print(
        (
            f"Mass error={diagnostics['maximum_abs_mass_error']:.3e}; minimum cell mass="
            f"{diagnostics['minimum_cell_mass']:.3e}"
        ),
    )
    _diagnostics(diagnostics, args.verbose)
    if args.save:
        _save_lines(
            'bm_1_fp_reference.png',
            [
                (
                    'State moments',
                    'Time [h]',
                    'X',
                    [(24 * t, mean, 'E[X]'), (24 * t, np.sqrt(variance), 'std[X]')],
                ),
                (
                    'Probability density',
                    'x',
                    'p(x)',
                    [(x, density[0], 'initial'), (x, density[-1], 'terminal')],
                ),
            ],
        )
    return result


def run_mc_fv(args):
    """Compare independent numerical descriptions of the same SDE/input law."""
    result = stochastic.run_comparison(n_paths=args.mc_paths, x_steps=args.cells)
    _table(
        'MC versus FV at T',
        ('quantity', 'MC', 'FV'),
        [
            ('E[X]', f"{result['mc_mean'][-1]:.8f}", f"{result['fp_mean'][-1]:.8f}"),
            ('E[D]', f"{result['mc_expected_demand'][-1]:.8f}", f"{result['fp_expected_demand'][-1]:.8f}"),
        ],
    )
    print(f"Terminal L1={result['l1'][-1]:.3e}, W1={result['wasserstein_1'][-1]:.3e}")
    _diagnostics(result['em_diagnostics'], args.verbose)
    _diagnostics(result['fp_diagnostics'], args.verbose)
    if args.save:
        _save_lines(
            'bm_1_mc_vs_fp_reference.png',
            [
                (
                    'Expected state',
                    'Time [h]',
                    'E[X]',
                    [
                        (24 * result['t'], result['mc_mean'], 'MC'),
                        (24 * result['t'], result['fp_mean'], 'FV'),
                    ],
                ),
                (
                    'Terminal density',
                    'x',
                    'p(x)',
                    [
                        (result['x'], result['mc_cell_mass'][-1] / result['dx'], 'MC'),
                        (result['x'], result['fp_density'][-1], 'FV'),
                    ],
                ),
            ],
        )
    return result


def run_projection(args):
    """Separate static representation error from errors in forward evolution."""
    result = mixture.run_projection_study()
    rows = []
    for degree, item in result['projection'].items():
        if (
            item['maximum_coefficient_sum_error'] > 1e-09
            or item['minimum_coefficient'] < -1e-09
            or item['maximum_mass_error'] > 1e-09
        ):
            raise RuntimeError('Static mixture projection violated probability constraints.')
        rows.append(
            (degree, f"{max(item['l2']):.3e}", f"{max(item['l1']):.3e}", f"{item['variance'][0]:.3e}"),
        )
        _diagnostics(item, args.verbose)
    _table('Static Beta projection', ('degree', 'max L2', 'max L1', 'initial variance'), rows)
    if args.save:
        _save_lines(
            'bm_1_beta_projection_reference.png',
            [
                (
                    'Static representation error',
                    'Time [h]',
                    'L2',
                    [
                        (24 * result['t'], item['l2'], f'M={degree}')
                         for degree, item in result['projection'].items()
                    ],
                ),
                (
                    'Terminal density',
                    'x',
                    'p(x)',
                    [(result['x'], result['fp_density'][-1], 'FV')] + [
                        (result['x'], item['density'][-1], f'M={degree}')
                         for degree, item in result['projection'].items()
                    ],
                ),
            ],
        )
    return result


def run_mixture(args):
    """Compare dynamic mixture error against the same-degree static approximation."""
    result = mixture.run_dynamic_mixture_study()
    _table(
        'Dynamic Beta mixtures',
        ('degree', 'max total L2', 'max static L2', 'max dynamic gap'),
        [
            (
                degree,
                f"{max(item['total_error']['l2']):.3e}",
                f"{max(result['projection'][degree]['l2']):.3e}",
                f"{max(item['dynamic_gap']['l2']):.3e}",
            )
             for degree, item in result['dynamic'].items()
        ],
    )
    for item in result['dynamic'].values():
        _diagnostics(item['diagnostics'], args.verbose)
    if args.save:
        _save_lines(
            'bm_1_beta_dynamic_reference.png',
            [
                (
                    'Dynamic error',
                    'Time [h]',
                    'L2',
                    [
                        (24 * result['t'], item['total_error']['l2'], f'M={degree}')
                         for degree, item in result['dynamic'].items()
                    ],
                ),
                (
                    'Terminal density',
                    'x',
                    'p(x)',
                    [(result['x'], result['fp_density'][-1], 'FV')] + [
                        (result['x'], item['density'][-1], f'M={degree}')
                         for degree, item in result['dynamic'].items()
                    ],
                ),
            ],
        )
    return result


def run_conditioning(args):
    """Measure coefficient and density sensitivity, not only a matrix condition number."""
    result = mixture.run_conditioning_study()
    _table(
        'Beta-mixture conditioning',
        ('degree', 'cond(H)', 'max local dynamic cond', 'QP start failures'),
        [
            (
                degree,
                f"{item['hessian_condition_number']:.3e}",
                f"{max(item['dynamic_local_condition_numbers']):.3e}",
                item['alternative_start']['failures'],
            )
             for degree, item in result['conditioning'].items()
        ],
    )
    if args.verbose:
        for degree, item in result['conditioning'].items():
            _table(
                f'M={degree}: mass-preserving perturbations',
                ('epsilon', 'density amplification', 'failures'),
                [
                    (f'{eps:.1e}', f"{row['maximum_density_amplification']:.3e}", row['failures'])
                     for eps, row in item['perturbations'].items()
                ],
            )
    if args.save:
        _save_lines(
            'bm_1_beta_conditioning_reference.png',
            [
                (
                    'Density sensitivity',
                    'perturbation epsilon',
                    'max L2 change',
                    [
                        (
                            result['epsilons'],
                            [
                                item['perturbations'][eps]['maximum_projected_density_l2']
                                 for eps in result['epsilons']
                            ],
                            f'M={degree}',
                        )
                         for degree, item in result['conditioning'].items()
                    ],
                ),
            ],
        )
    return result


def run_objective(args):
    """Compare tracking moments and MC uncertainty on the established F2-H case."""
    result = mixture.run_objective_reference(n_paths=args.mc_paths, x_steps=args.cells)
    mc, fv = (result['mc'], result['fv'])
    rows = [('MC', mc.objective), ('FV', fv)] + [
        (f'M={degree}', item)
         for degree, item in result['mixture'].items()
    ]
    _table(
        'F2-H expected tracking cost',
        ('method', 'J', 'bias contribution', 'variance contribution'),
        [(name, f'{item.total:.8e}', f'{item.bias:.8e}', f'{item.variance:.8e}') for name, item in rows],
    )
    gap = abs(mc.objective.total - fv.total)
    z = gap / mc.standard_error if mc.standard_error > 0 else np.nan
    print(f'MC SE={mc.standard_error:.3e}; |MC-FV|={gap:.3e} ({z:.3f} MC standard errors)')
    print(
        (
            f"Setup: {args.cells} cells, {args.mc_paths} paths, {result['widths'].size}"
            f" objective intervals; EM violations={result['em_diagnostics']['total_violations']}"
        ),
    )
    _diagnostics(
        {
            'direct_vs_decomposed_J': abs(mc.direct_total - mc.objective.total),
            'maximum_pointwise_decomposition_error': mc.maximum_decomposition_error,
        },
        args.verbose,
    )
    if args.save:
        hours = 24 * result['midpoints']
        _save_lines(
            'bm_1_stochastic_objective_reference.png',
            [
                (
                    'Tracking integrand',
                    'Time [h]',
                    'E[(D-A)^2]',
                    [(hours, item.total_integrand, name) for name, item in rows],
                ),
                (
                    'Expected demand',
                    'Time [h]',
                    'E[D]',
                    [(hours, result['target'], 'target')] + [
                        (hours, item.expected_demand, name)
                         for name, item in rows
                    ],
                ),
                (
                    'Cumulative expected cost',
                    'Time [h]',
                    'J(t)',
                    [
                        (hours, np.cumsum(result['widths'] * item.total_integrand), name)
                         for name, item in rows
                    ],
                ),
            ],
        )
    return result


def run_benchmark(args):
    """Time the two FV backends only after checking that their results agree."""
    result = control_cases.run_benchmark(
        args.cells,
        args.repeats,
        args.optimize,
        print if args.verbose else None,
    )
    _table(
        'FV timing (this machine/run)',
        ('cells', 'reference [s]', 'fast [s]', 'speedup'),
        [
            (
                row['cells'],
                f"{row['reference_seconds']:.6f}",
                f"{row['fast_seconds']:.6f}",
                f"{row['speedup']:.2f}x",
            )
             for row in result['fixed_price']
        ],
    )
    if args.optimize:
        _diagnostics(result['optimization'], True)
    if args.save or args.output:
        path = Path(args.output) if args.output else OUTPUT_DIR / 'fast_fv_benchmark.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f'Saved benchmark to {path}')
    return result


def main(argv=None):
    """Parse once, select one experiment, and leave numerical choices explicit."""
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='case', required=True)
    handlers = {
        'deterministic': run_deterministic,
        'feasibility': run_feasibility,
        'em-convergence': run_em_convergence,
        'em': run_em,
        'fv': run_fv,
        'mc-fv': run_mc_fv,
        'projection': run_projection,
        'mixture': run_mixture,
        'conditioning': run_conditioning,
        'objective': run_objective,
        'benchmark': run_benchmark,
    }
    for name, handler in handlers.items():
        command = commands.add_parser(name, help=handler.__doc__)
        command.set_defaults(run=handler)
        command.add_argument('--verbose', action='store_true', help='Show detailed diagnostics.')
        command.add_argument('--save', action='store_true', help='Save local plots/results under runs/.')
        if name in ('em', 'mc-fv', 'objective'):
            default_paths = stochastic.EM_REF_N_PATHS if name == 'em' else 50000
            command.add_argument('--mc-paths', type=int, default=default_paths)
        if name in ('fv', 'mc-fv', 'objective'):
            command.add_argument('--cells', type=int, default=cases.X_STEPS)
        if name == 'benchmark':
            command.add_argument('--cells', type=int, nargs='+', default=[201, 1001])
            command.add_argument('--repeats', type=int, default=7)
            command.add_argument('--optimize', action='store_true')
            command.add_argument('--output', type=Path, help='Explicit path for a saved benchmark JSON.')
    args = parser.parse_args(argv)
    if hasattr(args, 'mc_paths') and args.mc_paths < 2:
        parser.error('Monte Carlo comparisons require at least two paths.')
    args.run(args)


if __name__ == '__main__':
    main()
