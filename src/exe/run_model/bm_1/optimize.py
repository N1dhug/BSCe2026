"""BM1 price experiments: build a case, optimize, then compare on a common model.

Reports are short by default. --verbose shows solver progress and full prices;
--save writes figures under ignored runs/, never over the retained evidence.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from src.method.objective import tracking_error
from src.model.bm_1 import bm_1
from src.validation.bm_1 import control as cases, deterministic


OUTPUT_DIR = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'


def _prices(values):
    return np.array2string(np.asarray(values), precision=6, floatmode='fixed')


def _costs(evaluations):
    print('candidate                  J              bias           variance')
    for name, evaluation in evaluations.items():
        cost = evaluation.objective
        print(f'{name:26s} {cost.total:.8e} {cost.bias:.8e} {cost.variance:.8e}')


def _solver_summary(result):
    """Report solver status even in short mode; success is not a global-optimum proof."""
    if hasattr(result, 'results'):
        successes = sum((bool(item.success) for item in result.results))
        print(
            (
                f'Powell starts: {successes}/{len(result.results)} successful; solves/cache hits='
                f'{result.objective_evaluations}/{result.cache_hits}'
            ),
        )
    else:
        print(f'Solver success={result.success}; iterations/evaluations={result.nit}/{result.nfev}')
        if not result.success:
            print(f'Solver message: {result.message}')


def _save_comparison(name, target_t, target, price_plans, demand_plans, surface=None):
    """Plot a common target, candidate demands and held price schedules."""
    import matplotlib.pyplot as plt
    nrows = 3 if surface is not None else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 4 * nrows))
    if surface is not None:
        values, costs = surface
        grid_x, grid_y = np.meshgrid(values, values, indexing='ij')
        mesh = axes[0].contourf(grid_x, grid_y, costs, levels=15)
        fig.colorbar(mesh, ax=axes[0], label='Expected tracking cost')
        axes[0].set(xlabel='First-half price', ylabel='Second-half price', title='Coarse objective surface')
        for label, (_, prices) in price_plans.items():
            axes[0].scatter(prices[0], prices[1], label=label)
        axes[0].legend()
        axes = axes[1:]
    axes[0].plot(24 * target_t, target, label='Target A', linewidth=2)
    for label, (t, demand) in demand_plans.items():
        axes[0].plot(24 * t, demand, label=label)
    for label, (edges, prices) in price_plans.items():
        axes[1].step(24 * edges, np.r_[prices, prices[-1]], where='post', label=label)
    axes[0].set(ylabel='Demand', title='Tracking under candidate prices')
    axes[1].set(ylabel='Normalized price', xlabel='Time [h]', ylim=(0, 1))
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc='best')
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved graph to {path}')
    return path


def run_deterministic(args):
    """Recover the known two-price target with the original F1-D optimizer."""
    baseline = deterministic.build_baseline()
    generator = bm_1(baseline, baseline, deterministic.T, len(deterministic.OBJECTIVE_T))
    target_result = deterministic.solve_prices(generator, deterministic.KNOWN_PRICES)
    target_demand = target_result[7]
    target = interp1d(deterministic.OBJECTIVE_T, target_demand, kind='linear', bounds_error=True)
    model = bm_1(baseline, target, deterministic.T, len(deterministic.OBJECTIVE_T))
    initial = deterministic.solve_prices(model, deterministic.INITIAL_PRICES)
    fit = model.optimize(
        deterministic.INITIAL_PRICES,
        [deterministic.X0],
        deterministic.build_parameters(deterministic.INITIAL_PRICES),
        interval_edges=deterministic.INTERVAL_EDGES,
        t_eval=deterministic.OBJECTIVE_T,
        rtol=deterministic.RTOL,
        atol=deterministic.ATOL,
        options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 200},
    )
    solved = deterministic.solve_prices(model, fit.x)
    initial_cost = tracking_error(deterministic.OBJECTIVE_T, initial[7], target_demand)
    final_cost = tracking_error(deterministic.OBJECTIVE_T, solved[7], target_demand)
    print('F1-D deterministic price optimization')
    print(f'Initial J={initial_cost:.8e}; recomputed best J={final_cost:.8e}')
    print(f'Known prices: {_prices(deterministic.KNOWN_PRICES)}; best prices: {_prices(fit.x)}')
    _solver_summary(fit)
    if not fit.success:
        raise RuntimeError('F1-D optimizer did not converge; inspect its reported status.')
    if args.verbose:
        print(f'Max price recovery error={np.max(np.abs(fit.x - deterministic.KNOWN_PRICES)):.3e}')
        print(f'Reported J={fit.fun:.8e}; recomputation difference={fit.fun - final_cost:.3e}')
    if args.save:
        _save_comparison(
            'bm_1_optimization_reference.png',
            deterministic.OBJECTIVE_T,
            target_demand,
            {
                'known': (deterministic.INTERVAL_EDGES, deterministic.KNOWN_PRICES),
                'optimized': (deterministic.INTERVAL_EDGES, fit.x),
            },
            {'initial': (initial[0], initial[7]), 'optimized': (solved[0], solved[7])},
        )
    return fit


def run_contract(args):
    """Use one F3-A problem for a fixed external input or an optional new decision."""
    result = cases.run_reference(optimize=args.optimize, progress=print if args.verbose else None)
    problem = result['problem']
    print('F3-A common BM1 control problem')
    print(f'Fixed external/reference prices: {_prices(problem.observed_prices)}')
    _costs({'fixed': result['fixed']})
    fit = result['optimization']
    if fit is not None:
        print(f'Best prices: {_prices(fit.best.x)}; best J={fit.best.fun:.8e}')
        _solver_summary(fit)
    else:
        print('Fixed-price simulation only; no price optimization.')
    if args.save:
        evaluator = problem.build_evaluator(result['numerics'], fast=True)
        prices = problem.observed_prices if fit is None else fit.best.x
        evaluation = evaluator.evaluate(prices)
        _save_comparison(
            'bm_1_control_contract.png',
            evaluator.midpoints,
            evaluator.target,
            {'evaluated': (problem.price_edges, prices)},
            {'E[D]': (evaluator.midpoints, evaluation.objective.expected_demand)},
        )
    return result


def run_stochastic(args):
    """Design FV/mixture controls, then assess both on the same reference methods."""
    result = cases.run_optimization_reference(progress=print if args.verbose else None)
    print('F2-I stochastic price optimization: common FV reevaluation')
    _costs(result['fv_evaluations'])
    print(f"FV prices: {_prices(result['candidates']['fv_optimum'])}")
    print(f"Mixture prices: {_prices(result['candidates']['mixture_optimum'])}")
    _solver_summary(result['fv_optimization'])
    if result['mixture_optimization'] is not None:
        _solver_summary(result['mixture_optimization'])
    if result['mc'] is not None:
        print('Paired MC improvement over the deterministic reference')
        for name, (improvement, se) in result['paired_mc_improvement'].items():
            print(f'{name:26s} improvement={improvement:.6e}; SE={se:.3e}')
    if args.verbose:
        for degree, evaluations in result['final_mixture'].items():
            print(f'Mixture reevaluation: degree {degree}')
            _costs(evaluations)
    if args.save:
        evaluator = result['fv_evaluator']
        _save_comparison(
            'bm_1_stochastic_optimization_reference.png',
            evaluator.midpoints,
            evaluator.target,
            {name: (cases.INTERVAL_EDGES, prices) for name, prices in result['candidates'].items()},
            {
                name: (evaluator.midpoints, item.objective.expected_demand)
                 for name, item in result['fv_evaluations'].items()
            },
            surface=(result['coarse_values'], result['coarse_surface']),
        )
    return result


def run_resolution(args):
    """Compare price resolutions without hiding differences in the underlying grids."""
    run = cases.run_multi_price_experiment if args.case == 'multi' else cases.run_complex_target_experiment
    result = run(progress=print if args.verbose else None, fast_fv=args.fast_fv)
    resolved = result['cases']
    print(
        f"Price-resolution experiment: {args.case}; FV backend={('fast' if args.fast_fv else 'reference')}",
    )
    _costs({f'{count} intervals': case.evaluation for count, case in resolved.items()})
    for count, case in resolved.items():
        print(f'{count} intervals: ', end='')
        _solver_summary(case.optimization)
        if args.verbose:
            print(f'  prices={_prices(case.optimization.best.x)}')
    if result['mc'] is not None:
        first = min(resolved)
        print(f'Paired MC improvement relative to {first} intervals')
        for count, item in result['mc'].items():
            if count != first:
                improvement, se = cases._paired_improvement(result['mc'][first], item)
                print(f'{count:2d} intervals: improvement={improvement:.6e}; SE={se:.3e}')
    print('Interval counts 4 and 6 are not nested; improvement need not be monotone.')
    if args.save:
        evaluator = resolved[min(resolved)].evaluator
        _save_comparison(
            f'bm_1_{args.case}_price_optimization.png',
            evaluator.midpoints,
            evaluator.target,
            {f'N={n}': (case.edges, case.optimization.best.x) for n, case in resolved.items()},
            {
                f'N={n}': (case.evaluator.midpoints, case.evaluation.objective.expected_demand)
                 for n, case in resolved.items()
            },
        )
    return result


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='case', required=True)
    for name, run in [
        ('deterministic', run_deterministic),
        ('stochastic', run_stochastic),
        ('multi', run_resolution),
        ('complex', run_resolution),
        ('contract', run_contract),
    ]:
        command = commands.add_parser(name, help=run.__doc__)
        command.set_defaults(run=run)
        command.add_argument(
            '--verbose',
            action='store_true',
            help='Show prices, details and solver progress.',
        )
        command.add_argument('--save', action='store_true', help='Save a comparison figure under runs/.')
        if name == 'contract':
            command.add_argument('--optimize', action='store_true', help='Also optimize new prices.')
        if name in ('multi', 'complex'):
            command.add_argument(
                '--fast-fv',
                action='store_true',
                help='Use the equivalent LAPACK FV evaluator.',
            )
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == '__main__':
    main()
