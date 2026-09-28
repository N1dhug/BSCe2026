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
from src.validation.bm_1 import control as cases, deterministic, numerics as numerical_reference


OUTPUT_DIR = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'


def _prices(values):
    return np.array2string(np.asarray(values), precision=6, floatmode='fixed')


def _costs(evaluations):
    print('candidate                  J              bias           variance')
    for name, evaluation in evaluations.items():
        cost = evaluation.objective
        print(f'{name:26s} {cost.total:.8e} {cost.bias:.8e} {cost.variance:.8e}')


def _augmented_costs(evaluations):
    print('candidate                  J_total        tracking       terminal       smoothing')
    for name, evaluation in evaluations.items():
        print(
            f'{name:26s} {evaluation.total:.8e} {evaluation.objective.total:.8e} '
            f'{evaluation.penalties.terminal:.8e} {evaluation.penalties.smoothing:.8e}',
        )


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


def run_augmented(args):
    """F3-C terminal-state and price-jump regularized stochastic control."""
    result = cases.run_f3_c(
        terminal_reference=args.terminal_reference,
        terminal_weight=args.terminal_weight,
        smoothing_weight=args.smoothing_weight,
        progress=print if args.verbose else None,
    )
    problem = result['problem']
    fit = result['optimization']
    print('F3-C augmented stochastic control objective')
    print(
        f'x_ref={problem.terminal_reference:.6f}; '
        f'lambda_T={problem.terminal_weight:.6g}; '
        f'lambda_U={problem.price_smoothing_weight:.6g}',
    )
    _augmented_costs({'fixed': result['fixed'], 'optimized': result['optimized']})
    print(f'Best prices: {_prices(fit.best.x)}; best total J={fit.best.fun:.8e}')
    _solver_summary(fit)
    if args.verbose:
        for name, evaluation in (('fixed', result['fixed']), ('optimized', result['optimized'])):
            terminal_mse = evaluation.penalties.terminal_mse
            print(
                f'{name}: terminal MSE={terminal_mse:.8e}; '
                f'price roughness={evaluation.penalties.price_roughness:.8e}; '
                f'tracking bias/variance={evaluation.objective.bias:.8e}/{evaluation.objective.variance:.8e}',
            )
    if args.save:
        evaluator = result['evaluator']
        prices = fit.best.x
        _save_comparison(
            'bm_1_f3_c_augmented_control.png',
            evaluator.midpoints,
            evaluator.target,
            {
                'fixed': (problem.price_edges, problem.observed_prices),
                'optimized': (problem.price_edges, prices),
            },
            {
                'fixed E[D]': (evaluator.midpoints, result['fixed'].objective.expected_demand),
                'optimized E[D]': (evaluator.midpoints, result['optimized'].objective.expected_demand),
            },
        )
    return result


def _sensitivity_table(title, selected):
    print(title)
    print('lambda_T   lambda_U   tracking       terminal_MSE   roughness      J_total         prices')
    for case in selected:
        evaluation = case.evaluation
        print(
            f'{case.terminal_weight:<10.6g} {case.smoothing_weight:<10.6g} '
            f'{evaluation.objective.total:.8e} {evaluation.penalties.terminal_mse:.8e} '
            f'{evaluation.penalties.price_roughness:.8e} {evaluation.total:.8e} '
            f'{_prices(evaluation.prices)}',
        )


def run_sensitivity(args):
    """F3-C sensitivity of tracking, terminal state and price roughness to objective weights."""
    result = cases.run_f3_c_sensitivity(
        terminal_reference=args.terminal_reference,
        terminal_weights=args.terminal_weights,
        smoothing_weights=args.smoothing_weights,
        include_grid=args.grid,
        progress=print if args.verbose else None,
    )
    resolved = result['cases']
    terminal_axis = [case for case in resolved if case.smoothing_weight == 0.0]
    smoothing_axis = [case for case in resolved if case.terminal_weight == 0.0]
    interior = [case for case in resolved if case.terminal_weight > 0.0 and case.smoothing_weight > 0.0]
    print(f"F3-C weight sensitivity; x_ref={result['terminal_reference']:.6f}")
    print('Compare raw tracking, terminal_MSE and roughness across rows; J_total uses a different objective when weights change.')
    _sensitivity_table('Terminal-weight sweep (lambda_U=0)', terminal_axis)
    print()
    _sensitivity_table('Smoothing-weight sweep (lambda_T=0)', smoothing_axis)
    if interior:
        print()
        _sensitivity_table('Two-dimensional weight grid', interior)
    return result


def _decision_case(name, case):
    problem = case['problem']
    deterministic = case['deterministic_evaluations']
    stochastic = case['stochastic_evaluations']
    print(
        f"{name}: lambda_T={problem.terminal_weight:.6g}, "
        f"lambda_U={problem.price_smoothing_weight:.6g}",
    )
    print(f"  deterministic policy: {_prices(case['policies']['deterministic_policy'])}")
    print(f"  stochastic policy:    {_prices(case['policies']['stochastic_policy'])}")
    print(
        f"  policy distance: Linf={case['policy_linf']:.6e}; "
        f"L2={case['policy_l2']:.6e}",
    )
    print('  cross-evaluation totals')
    print('  policy                     deterministic       stochastic')
    for policy in ('deterministic_policy', 'stochastic_policy'):
        print(
            f"  {policy:26s} {deterministic[policy].total:.8e} "
            f"{stochastic[policy].total:.8e}",
        )
    print(
        f"  deterministic regret={case['deterministic_regret']:.6e}; "
        f"stochastic value={case['stochastic_value']:.6e}",
    )
    print('  deterministic optimizer: ', end='')
    _solver_summary(case['deterministic_optimization'])
    print('  stochastic optimizer:    ', end='')
    _solver_summary(case['stochastic_optimization'])
    if case['paired_mc'] is not None:
        paired = case['paired_mc']
        print(
            f"  paired MC stochastic value={paired['improvement']:.6e}; "
            f"SE={paired['standard_error']:.3e}; ratio={paired['ratio']:.3f}",
        )


def run_decision(args):
    """F3-D compare policies optimized with deterministic and stochastic BM1."""
    result = cases.run_f3_d(
        validate_mc=not args.no_mc,
        mc_paths=args.mc_paths,
        progress=print if args.verbose else None,
    )
    print('F3-D deterministic versus stochastic control decision')
    for name, case in result['cases'].items():
        _decision_case(name, case)
    return result


def run_uncertainty(args):
    """F3-E isolate initial-state and process-noise effects on the optimized policy."""
    result = cases.run_f3_e(progress=print if args.verbose else None)
    order = result['scenario_order']

    print('F3-E uncertainty-source decomposition')
    print('scenario        initial law   sigma_X    prices')
    for name in order:
        initial_law = 'Beta(20,20)' if name in ('initial_only', 'full') else 'point 0.5'
        sigma_x = result['problems'][name].parameters.sigma_x
        print(f'{name:15s} {initial_law:12s} {sigma_x:<10.6g} {_prices(result["policies"][name])}')

    print('Policy distance from deterministic optimum')
    for name in order:
        distance = result['policy_distances'][name]
        print(f'  {name:15s} Linf={distance["linf"]:.6e}; L2={distance["l2"]:.6e}')

    print('Cross-evaluation total cost; compare policies within each model row')
    print('model \\ policy      deterministic   initial_only    process_only    full')
    for model_name in order:
        values = result['evaluations'][model_name]
        totals = ' '.join(f'{values[policy_name].total:14.8e}' for policy_name in order)
        print(f'{model_name:18s} {totals}')

    print('Regret relative to each model row optimum')
    print('model \\ policy      deterministic   initial_only    process_only    full')
    for model_name in order:
        values = result['regrets'][model_name]
        regrets = ' '.join(f'{values[policy_name]:14.6e}' for policy_name in order)
        print(f'{model_name:18s} {regrets}')

    print('Optimizer status')
    for name in order:
        print(f'  {name:15s}', end='')
        _solver_summary(result['optimizations'][name])
    return result


def run_mixture_control(args):
    """F3-F compare FV-grid and dynamic-mixture effects on stochastic control."""
    result = cases.run_f3_f(
        validate_mc=not args.no_mc,
        mc_paths=args.mc_paths,
        progress=print if args.verbose else None,
    )
    policies = result['policies']

    print('F3-F FV versus dynamic-mixture control decision')
    print(
        f"reference FV cells={result['reference_numerics'].x_steps}; "
        f"design cells={result['design_numerics'].x_steps}; "
        f"mixture degree={cases.MIXTURE_OPT_DEGREE}; cfl={cases.DEFAULT_CFL:.3g}",
    )
    print(f"FV reference policy: {_prices(policies['fv_reference_policy'])}")
    print(f"FV 81-cell policy:   {_prices(policies['fv_design_policy'])}")
    print(f"Mixture policy:      {_prices(policies['mixture_policy'])}")

    print('Policy distances')
    for name, distance in result['policy_distances'].items():
        print(f'  {name:24s} Linf={distance["linf"]:.6e}; L2={distance["l2"]:.6e}')

    print('81-cell design-grid cross-evaluation')
    print('method             FV_ref_policy   FV_81_policy    mixture_policy')
    for name, evaluations in result['design_evaluations'].items():
        print(
            f'{name:18s} '
            f'{evaluations["fv_reference_policy"].total:.8e} '
            f'{evaluations["fv_design_policy"].total:.8e} '
            f'{evaluations["mixture_policy"].total:.8e}',
        )
    print(
        f"  FV-grid regret of mixture policy={result['fv_design_regret_of_mixture']:.6e}; "
        f"mixture-model regret of FV-81 policy={result['mixture_design_regret_of_fv']:.6e}",
    )

    print('Common 201-cell reevaluation')
    print('method             FV_ref_policy   FV_81_policy    mixture_policy')
    for name, evaluations in result['reference_evaluations'].items():
        print(
            f'{name:18s} '
            f'{evaluations["fv_reference_policy"].total:.8e} '
            f'{evaluations["fv_design_policy"].total:.8e} '
            f'{evaluations["mixture_policy"].total:.8e}',
        )
    print(
        f"FV-201 regret of FV-81 policy={result['fv_grid_regret']:.6e}; "
        f"FV-201 regret of mixture policy={result['fv_reference_regret_of_mixture']:.6e}",
    )

    print('Signed common-reference differences relative to FV-reference policy')
    for method, differences in result['reference_differences'].items():
        print(
            f'  {method:16s} '
            f'FV_81={differences["fv_design_policy"]:.6e}; '
            f'mixture={differences["mixture_policy"]:.6e}',
        )

    print('FV-201 optimizer:   ', end='')
    _solver_summary(result['fv_reference_optimization'])
    print('FV-81 optimizer:    ', end='')
    _solver_summary(result['fv_design_optimization'])
    print('Mixture optimizer:  ', end='')
    _solver_summary(result['mixture_optimization'])

    if result['paired_mc'] is not None:
        print('Paired MC differences: FV-reference minus candidate')
        for name, paired in result['paired_mc'].items():
            print(
                f'  {name:18s} difference={paired["reference_minus_candidate"]:.6e}; '
                f'SE={paired["standard_error"]:.3e}; ratio={paired["ratio"]:.3f}',
            )

    if args.verbose:
        for degree in result['mixture_reference']:
            evaluation = result['reference_evaluations'][f'mixture_{degree}_201']['mixture_policy']
            diagnostics = evaluation.diagnostics
            print(
                f'Mixture degree {degree}: steps={diagnostics["n_steps"]}; '
                f'projection fallbacks={diagnostics["projection_fallback_count"]}; '
                f'Hessian condition={diagnostics["hessian_condition_number"]:.3e}; '
                f'min raw prediction={diagnostics["minimum_raw_prediction"]:.3e}; '
                f'max projected mass error={diagnostics["maximum_abs_projected_mass_error"]:.3e}',
            )
    return result

def run_mixture_validation(args):
    """F3-F high-resolution validation of already-fixed FV/mixture policies."""
    problem, _ = cases.build_f3_a_problem()
    candidates = {
        'fv_reference_policy': np.asarray(args.fv_reference_prices, dtype=float),
        'fv_design_policy': np.asarray(args.fv_design_prices, dtype=float),
        'mixture_policy': np.asarray(args.mixture_prices, dtype=float),
    }
    result = numerical_reference.run_f3_f_high_resolution_validation(
        problem,
        candidates,
        mc_paths=args.mc_paths,
        mc_time_paths=args.mc_time_paths,
        progress=print if args.verbose else None,
    )
    numerics = result['numerics']
    print('F3-F fixed-policy high-resolution validation')
    print(
        f'FV reference numerics: cells={numerics.x_steps}; '
        f'objective max step={numerics.objective_max_step:.8f}; '
        f'solver max step={numerics.solver_max_step:.8f}',
    )
    for name, prices in result['candidates'].items():
        print(f'{name:20s} {_prices(prices)}')
    print('High-resolution FV optimizer: ', end='')
    _solver_summary(result['highres_fv_optimization'])

    print('High-resolution FV totals and differences from high-resolution FV optimum')
    reference_total = result['fv']['highres_fv_policy'].total
    print(f'  highres_fv_policy    total={reference_total:.8e}; difference=0.00000000e+00')
    for name in ('fv_reference_policy', 'fv_design_policy', 'mixture_policy'):
        print(
            f'  {name:20s} total={result["fv"][name].total:.8e}; '
            f'difference={result["fv_differences"][name]:.8e}',
        )

    print('High-resolution paired MC: high-resolution FV minus candidate')
    for name, paired in result['paired_mc'].items():
        print(
            f'  {name:20s} difference={paired["reference_minus_candidate"]:.8e}; '
            f'SE={paired["standard_error"]:.3e}; ratio={paired["ratio"]:.3f}',
        )

    if result['paired_mc_time'] is not None:
        print('Coupled MC time refinement: high-resolution FV minus candidate')
        print('  max_step        FV_201            FV_81             mixture')
        for row in result['paired_mc_time']:
            values = row['reference_minus_candidate']
            print(
                f'  {row["solver_max_step"]:<14.8f} '
                f'{values["fv_reference_policy"]: .8e} '
                f'{values["fv_design_policy"]: .8e} '
                f'{values["mixture_policy"]: .8e}',
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
        ('augmented', run_augmented),
        ('sensitivity', run_sensitivity),
        ('decision', run_decision),
        ('uncertainty', run_uncertainty),
        ('mixture-control', run_mixture_control),
        ('mixture-validate', run_mixture_validation),
    ]:
        command = commands.add_parser(name, help=run.__doc__)
        command.set_defaults(run=run)
        command.add_argument(
            '--verbose',
            action='store_true',
            help='Show prices, details and solver progress.',
        )
        if name not in ('sensitivity', 'decision', 'uncertainty', 'mixture-control', 'mixture-validate'):
            command.add_argument('--save', action='store_true', help='Save a comparison figure under runs/.')
        if name == 'mixture-validate':
            command.add_argument('--fv-reference-prices', nargs=2, type=float, required=True)
            command.add_argument('--fv-design-prices', nargs=2, type=float, required=True)
            command.add_argument('--mixture-prices', nargs=2, type=float, required=True)
            command.add_argument('--mc-paths', type=int, default=20000)
            command.add_argument('--mc-time-paths', type=int, default=5000)
        if name == 'contract':
            command.add_argument('--optimize', action='store_true', help='Also optimize new prices.')
        if name == 'augmented':
            command.add_argument(
                '--terminal-reference',
                type=float,
                default=cases.F3_C_TERMINAL_REFERENCE,
                help='Reference terminal state x_ref; default 0.5.',
            )
            command.add_argument(
                '--terminal-weight',
                type=float,
                default=0.0,
                help='Non-negative lambda_T multiplying E[(X_T-x_ref)^2].',
            )
            command.add_argument(
                '--smoothing-weight',
                type=float,
                default=0.0,
                help='Non-negative lambda_U multiplying squared adjacent price jumps.',
            )
        if name == 'sensitivity':
            command.add_argument(
                '--terminal-reference',
                type=float,
                default=cases.F3_C_TERMINAL_REFERENCE,
                help='Reference terminal state x_ref; default 0.5.',
            )
            command.add_argument(
                '--terminal-weights',
                type=float,
                nargs='+',
                default=list(cases.F3_C_TERMINAL_WEIGHTS),
                help='Strictly increasing non-negative lambda_T values.',
            )
            command.add_argument(
                '--smoothing-weights',
                type=float,
                nargs='+',
                default=list(cases.F3_C_SMOOTHING_WEIGHTS),
                help='Strictly increasing non-negative lambda_U values.',
            )
            command.add_argument(
                '--grid',
                action='store_true',
                help='Also optimize every positive lambda_T/lambda_U combination.',
            )
        if name == 'decision':
            command.add_argument(
                '--mc-paths',
                type=int,
                default=cases.F3_D_MC_PATHS,
                help='Paired Monte Carlo paths for stochastic policy validation.',
            )
            command.add_argument(
                '--no-mc',
                action='store_true',
                help='Skip paired Monte Carlo and run only deterministic/FV cross-evaluation.',
            )
        if name == 'mixture-control':
            command.add_argument(
                '--mc-paths',
                type=int,
                default=cases.F3_F_MC_PATHS,
                help='Paired Monte Carlo paths for FV-versus-mixture policy validation.',
            )
            command.add_argument(
                '--no-mc',
                action='store_true',
                help='Skip paired Monte Carlo and run only density-method cross-evaluation.',
            )
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
