"""BM1 executable: input selection, reporting and plots."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from src.method.objective import tracking_error
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder
from src.validation.bm_1.control import (HOURLY_EDGES, INTERVAL_COUNTS, KNOWN_DETERMINISTIC_PRICES,
    COMPLEX_MC_PATHS as MC_PATHS, MIXTURE_OPT_DEGREE, MULTI_INTERVAL_COUNTS, MULTI_MC_PATHS, REFERENCE_PRICES_24,
    _paired_improvement, INTERVAL_EDGES as control_INTERVAL_EDGES, T as control_T, multi_paired_improvement,
    run_complex_target_experiment, run_multi_price_experiment, run_optimization_reference, run_reference)
from src.validation.bm_1.deterministic import (ATOL, INITIAL_PRICES, INTERVAL_EDGES, KNOWN_PRICES, OBJECTIVE_T,
    OBJECTIVE_TOL, PRICE_TOL, RTOL, T, TARGET_TOL, X0, build_baseline, build_parameters, raises_value_error,
    solve_prices)

def save_figure(target_result, initial_result, optimized_result, optimized_prices):
    """Save the feasible target, recovered demand, state, and prices."""
    hours = 24.0 * OBJECTIVE_T
    target_demand = target_result[7]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(hours, target_demand, linewidth=2.5, label='Target A(t)')
    axes[0].plot(hours, initial_result[7], linestyle='--', label='Demand from initial prices')
    axes[0].plot(hours, optimized_result[7], linestyle=':', linewidth=2.5, label='Optimized demand')
    axes[0].set_ylabel('Normalized demand')
    axes[0].set_title('bm_1 deterministic price optimization (F1-D)')
    axes[0].grid(True)
    axes[0].legend(loc='best')
    axes[1].plot(hours, target_result[1], linewidth=2.5, label='Target state')
    axes[1].plot(hours, optimized_result[1], linestyle='--', label='Optimized state')
    axes[1].set_ylabel('State X')
    axes[1].grid(True)
    axes[1].legend(loc='best')
    known_price = input_builder.price_intervals((INTERVAL_EDGES, KNOWN_PRICES))
    initial_price = input_builder.price_intervals((INTERVAL_EDGES, INITIAL_PRICES))
    optimized_price = input_builder.price_intervals((INTERVAL_EDGES, optimized_prices))
    axes[2].step(hours, known_price(OBJECTIVE_T), where='post', linewidth=2.5, label='Known prices')
    axes[2].step(hours, initial_price(OBJECTIVE_T), where='post', linestyle='--', label='Initial prices')
    axes[2].step(hours, optimized_price(OBJECTIVE_T), where='post', linestyle=':', linewidth=2.5,
        label='Optimized prices')
    axes[2].set_xlabel('Time [hours]')
    axes[2].set_ylabel('Normalized price')
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True)
    axes[2].legend(loc='best')
    for axis in axes:
        for edge in INTERVAL_EDGES[1:-1]:
            axis.axvline(24.0 * edge, color='black', alpha=0.3, linestyle='--')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'graphs' / 'bm_1'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'BM1_optimization_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return graph_path

def run_deterministic():
    """Generate a feasible target and verify deterministic optimization."""
    baseline = build_baseline()
    generator = bm_1(baseline, baseline, T, OBJECTIVE_T.size)
    target_result = solve_prices(generator, KNOWN_PRICES)
    target_demand = target_result[7]
    target = interp1d(OBJECTIVE_T, target_demand, kind='linear', bounds_error=True)
    model = bm_1(baseline, target, T, OBJECTIVE_T.size)
    initial_result = solve_prices(model, INITIAL_PRICES)
    known_objective = tracking_error(OBJECTIVE_T, target_demand, target_demand)
    initial_objective = tracking_error(OBJECTIVE_T, initial_result[7], target_demand)
    result = model.optimize(INITIAL_PRICES, [X0], build_parameters(INITIAL_PRICES), interval_edges=INTERVAL_EDGES,
        t_eval=OBJECTIVE_T, rtol=RTOL, atol=ATOL, options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 200})
    optimized_result = solve_prices(model, result.x)
    optimized_objective = tracking_error(OBJECTIVE_T, optimized_result[7], target_demand)
    graph_path = save_figure(target_result, initial_result, optimized_result, result.x)
    checks = {'optimizer reports success': bool(result.success),
        'known feasible target has zero objective': known_objective <= TARGET_TOL,
        'optimized objective reaches the target tolerance': optimized_objective <= OBJECTIVE_TOL,
        'reported and recomputed objectives agree': np.isclose(result.fun, optimized_objective, rtol=1e-06, atol=1e-16),
        'optimization improves the initial objective': optimized_objective < initial_objective,
        'optimizer recovers the known price decisions': np.max(np.abs(result.x - KNOWN_PRICES)) <= PRICE_TOL,
        'optimized prices remain in [0, 1]': np.all(result.x >= 0.0) and np.all(result.x <= 1.0),
        'price decisions remain independent of the objective grid': result.x.size + 1 == INTERVAL_EDGES.size and result.x.size != OBJECTIVE_T.size,
        'mismatched interval counts are rejected': raises_value_error(lambda: model.optimize(INITIAL_PRICES, [X0], build_parameters(INITIAL_PRICES), interval_edges=np.array(T, dtype=float))),
        'interval edges must cover the model horizon': raises_value_error(lambda: model.optimize(INITIAL_PRICES, [X0], build_parameters(INITIAL_PRICES), interval_edges=np.array([0.1, 0.5, 1.0]))),
        'objective grid must cover the model horizon': raises_value_error(lambda: model.optimize(INITIAL_PRICES, [X0], build_parameters(INITIAL_PRICES), interval_edges=INTERVAL_EDGES, t_eval=OBJECTIVE_T[1:])),
        'reference figure is saved': graph_path.is_file() and graph_path.stat().st_size > 0}
    improvement = initial_objective / optimized_objective if optimized_objective > 0.0 else np.inf
    price_error = float(np.max(np.abs(result.x - KNOWN_PRICES)))
    print('bm_1 deterministic optimization F1-D verification')
    print('================================================')
    print()
    print(f'known prices             = {KNOWN_PRICES}')
    print(f'initial prices           = {INITIAL_PRICES}')
    print(f'optimized prices         = {result.x}')
    print(f'max price error          = {price_error:.6e}')
    print(f'known objective          = {known_objective:.6e}')
    print(f'initial objective        = {initial_objective:.6e}')
    print(f'optimized objective      = {optimized_objective:.6e}')
    print(f'objective improvement    = {improvement:.3e}')
    print(f'iterations / evaluations = {result.nit} / {result.nfev}')
    print()
    print('Checks')
    for name, passed in checks.items():
        print(f"{('PASS' if passed else 'FAIL')} {name}")
    print()
    print(f'Saved graph to {graph_path}')
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print()
        print('One or more F1-D checks failed:')
        for name in failed:
            print(f'- {name}')
        raise SystemExit(1)

def _format_prices(prices):
    return np.array2string(np.asarray(prices, dtype=float), precision=6, floatmode='fixed')

def run_stochastic():
    result = run_optimization_reference(progress=lambda message: print(message, flush=True))
    print('F2-I stochastic BM1 price optimization')
    print('--------------------------------------')
    print('Deterministic reference prices: ' + _format_prices(KNOWN_DETERMINISTIC_PRICES))
    print('FV stochastic optimum:          ' + _format_prices(result['candidates']['fv_optimum']))
    print('Mixture stochastic optimum:     ' + _format_prices(result['candidates']['mixture_optimum']))
    print()
    fv_opt = result['fv_optimization']
    print('FV bounded Powell multistart')
    print('----------------------------')
    print(f'starts: {len(fv_opt.results)}')
    print(f'successful: {sum((bool(item.success) for item in fv_opt.results))}')
    print(f'best objective: {float(fv_opt.best.fun):.8e}, iterations/evaluations: {fv_opt.best.nit}/{fv_opt.best.nfev}')
    successful_costs = [float(item.fun) for item in fv_opt.results if item.success]
    print(f'successful-start objective spread: {np.ptp(successful_costs):.3e}')
    print(f'actual solves/cache hits: {fv_opt.objective_evaluations}/{fv_opt.cache_hits}')
    print()
    if result['mixture_optimization'] is not None:
        mix_opt = result['mixture_optimization']
        print(f'M={MIXTURE_OPT_DEGREE} mixture Powell multistart')
        print('--------------------------------')
        print(f'starts: {len(mix_opt.results)}')
        print(f'successful: {sum((bool(item.success) for item in mix_opt.results))}')
        print(f'best surrogate objective: {float(mix_opt.best.fun):.8e}, iterations/evaluations: {mix_opt.best.nit}/{mix_opt.best.nfev}')
        print()
    print('Pure-FV reevaluation')
    print('--------------------')
    reference_fv = result['fv_evaluations']['deterministic_reference']
    for name, item in result['fv_evaluations'].items():
        improvement = reference_fv.objective.total - item.objective.total
        print(f'{name:24s} J={item.objective.total:.8e}, bias={item.objective.bias:.8e}, variance={item.objective.variance:.8e}, improvement={improvement:.6e}')
    print()
    print('High-resolution mixture reevaluation')
    print('------------------------------------')
    for degree, evaluations in result['final_mixture'].items():
        print(f'M={degree}')
        for name, item in evaluations.items():
            print(f'  {name:22s} J={item.objective.total:.8e}, bias={item.objective.bias:.8e}, variance={item.objective.variance:.8e}')
    if result['mc'] is not None:
        print()
        print('Monte Carlo common-random-number validation')
        print('-------------------------------------------')
        for name, item in result['mc'].items():
            objective = item.objective
            print(f'{name:24s} J={objective.objective.total:.8e}, SE={objective.standard_error:.3e}')
        print()
        print('Paired MC improvement over deterministic reference')
        print('--------------------------------------------------')
        for name, (improvement, standard_error) in result['paired_mc_improvement'].items():
            z = improvement / standard_error if standard_error > 0.0 else np.inf
            print(f'{name:24s} improvement={improvement:.6e}, SE={standard_error:.3e}, ratio={z:.3f}')
    fv = result['fv_evaluator']
    hours = 24.0 * fv.midpoints
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_surface, ax_demand, ax_price, ax_parts = axes.ravel()
    grid_first, grid_second = np.meshgrid(result['coarse_values'], result['coarse_values'], indexing='ij')
    contour = ax_surface.contourf(grid_first, grid_second, result['coarse_surface'], levels=15)
    fig.colorbar(contour, ax=ax_surface, label='FV stochastic objective')
    for name, marker in (('deterministic_reference', 'o'), ('fv_optimum', 'x'), ('mixture_optimum', '+')):
        prices = result['candidates'][name]
        ax_surface.scatter(prices[0], prices[1], marker=marker, s=90, label=name.replace('_', ' '))
    ax_surface.set_xlabel('First-half price')
    ax_surface.set_ylabel('Second-half price')
    ax_surface.set_title('Coarse stochastic objective surface')
    ax_surface.legend(loc='best')
    ax_demand.plot(hours, fv.target, linewidth=2.5, label='Target A(t)')
    for name, item in result['fv_evaluations'].items():
        ax_demand.plot(hours, item.objective.expected_demand, label=name.replace('_', ' '))
    ax_demand.set_xlabel('Time [h]')
    ax_demand.set_ylabel('Expected demand')
    ax_demand.set_title('Pure-FV expected demand')
    ax_demand.grid(True)
    ax_demand.legend(loc='best')
    for name, prices in result['candidates'].items():
        price = input_builder.price_intervals((control_INTERVAL_EDGES, prices))
        ax_price.step(hours, price(fv.midpoints), where='mid', label=name.replace('_', ' '))
    ax_price.set_xlabel('Time [h]')
    ax_price.set_ylabel('Normalized price')
    ax_price.set_ylim(0.0, 1.0)
    ax_price.set_title('Candidate dynamic prices')
    ax_price.grid(True)
    ax_price.legend(loc='best')
    labels = ['deterministic\nreference', 'FV\noptimum', 'mixture\noptimum']
    names = ['deterministic_reference', 'fv_optimum', 'mixture_optimum']
    bias = [result['fv_evaluations'][name].objective.bias for name in names]
    variance = [result['fv_evaluations'][name].objective.variance for name in names]
    positions = np.arange(len(names))
    ax_parts.bar(positions, bias, label='Bias-squared')
    ax_parts.bar(positions, variance, bottom=bias, label='Demand variance')
    ax_parts.set_xticks(positions, labels)
    ax_parts.set_ylabel('Pure-FV stochastic objective')
    ax_parts.set_title('Objective decomposition')
    ax_parts.grid(True, axis='y')
    ax_parts.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'graphs' / 'bm_1'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_stochastic_optimization_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print()
    print(f'Saved graph to {graph_path}')

def run_complex():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast-fv', action='store_true',
        help='Use the numerically equivalent LAPACK/blocked FV evaluator.')
    args = parser.parse_args()
    print('FV backend: ' + ('fast' if args.fast_fv else 'reference'), flush=True)
    result = run_complex_target_experiment(progress=lambda message: print(message, flush=True), fast_fv=args.fast_fv)
    cases = result['cases']
    print()
    print('BM1 complex-target multi-price stochastic optimization')
    print('------------------------------------------------------')
    print('Target-generating hourly reference prices:')
    print(np.array2string(REFERENCE_PRICES_24, precision=3, floatmode='fixed', max_line_width=120))
    print()
    print('intervals  changes  FV objective       bias               variance           actual evals  cache hits')
    for count, case in cases.items():
        objective = case.evaluation.objective
        optimization = case.optimization
        print(f'{count:9d}  {count - 1:7d}  {objective.total:.10e}  {objective.bias:.10e}  {objective.variance:.10e}  {optimization.objective_evaluations:12d}  {optimization.cache_hits:10d}')
        print(' ' * 20 + 'prices=' + np.array2string(np.asarray(optimization.best.x, dtype=float), precision=6,
            floatmode='fixed', max_line_width=120))
    print()
    print('Marginal pure-FV improvement')
    previous = None
    for count, case in cases.items():
        if previous is not None:
            improvement = previous.evaluation.objective.total - case.evaluation.objective.total
            relative = improvement / previous.evaluation.objective.total
            print(f'N={previous.n_intervals} -> N={count}: dJ={improvement:.6e} ({relative:.3%})')
        previous = case
    if result['mc'] is not None:
        print()
        print(f'Common-random-number Monte Carlo ({MC_PATHS} paths)')
        for count in INTERVAL_COUNTS:
            item = result['mc'][count]
            print(f'N={count}: J={item.objective.total:.10e}, SE={item.standard_error:.3e}')
        reference = result['mc'][INTERVAL_COUNTS[0]]
        print()
        print(f'Paired MC improvement relative to N={INTERVAL_COUNTS[0]}')
        for count in INTERVAL_COUNTS[1:]:
            mean, se = _paired_improvement(reference, result['mc'][count])
            ratio = mean / se if se > 0.0 else np.inf
            print(f'N={count}: improvement={mean:.6e}, SE={se:.3e}, ratio={ratio:.3f}')
    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    ax_price, ax_objective, ax_demand = axes
    plot_t = np.linspace(control_T[0], control_T[1], 1441, dtype=float)
    reference_price = input_builder.price_intervals((HOURLY_EDGES, REFERENCE_PRICES_24))
    ax_price.step(24.0 * plot_t, reference_price(plot_t), where='post', linewidth=2.5, linestyle='--',
        label='Target-generating N=24 reference')
    for count, case in cases.items():
        price = input_builder.price_intervals((case.edges, case.optimization.best.x))
        ax_price.step(24.0 * plot_t, price(plot_t), where='post', label=f'Optimized N={count}')
    ax_price.set_ylabel('Normalized price')
    ax_price.set_ylim(0.0, 1.0)
    ax_price.set_title('Reference and optimized stochastic price schedules')
    ax_price.grid(True)
    ax_price.legend(loc='best')
    counts = np.asarray(INTERVAL_COUNTS, dtype=int)
    totals = np.array([cases[count].evaluation.objective.total for count in counts])
    bias = np.array([cases[count].evaluation.objective.bias for count in counts])
    variance = np.array([cases[count].evaluation.objective.variance for count in counts])
    ax_objective.plot(counts, totals, marker='o', label='Total J')
    ax_objective.plot(counts, bias, marker='o', label='Bias contribution')
    ax_objective.plot(counts, variance, marker='o', label='Variance contribution')
    ax_objective.set_xlabel('Number of price intervals')
    ax_objective.set_ylabel('Pure-FV stochastic objective')
    ax_objective.set_title('Value of price resolution for a complex target')
    ax_objective.grid(True)
    ax_objective.legend(loc='best')
    reference_evaluator = cases[INTERVAL_COUNTS[0]].evaluator
    hours = 24.0 * reference_evaluator.midpoints
    ax_demand.plot(hours, reference_evaluator.target, linewidth=2.5, label='Target A(t)')
    for count, case in cases.items():
        ax_demand.plot(hours, case.evaluation.objective.expected_demand, label=f'E[D], N={count}')
    ax_demand.set_xlabel('Time [h]')
    ax_demand.set_ylabel('Expected demand')
    ax_demand.set_title('Expected demand under optimized price schedules')
    ax_demand.grid(True)
    ax_demand.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'graphs' / 'bm_1'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_complex_target_multi_price_experiment.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print()
    print(f'Saved graph to {graph_path}')

def run_multi():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fast-fv', action='store_true',
        help='Use the numerically equivalent LAPACK/blocked FV evaluator.')
    args = parser.parse_args()
    print('FV backend: ' + ('fast' if args.fast_fv else 'reference'), flush=True)
    result = run_multi_price_experiment(progress=lambda message: print(message, flush=True), fast_fv=args.fast_fv)
    cases = result['cases']
    print()
    print('BM1 multi-price stochastic optimization experiment')
    print('------------------------------------------------')
    print('All cases: same target, Beta(20,20), sigma=0.5, pure FV objective, common internal breakpoints.')
    print()
    print('intervals  changes  FV objective       bias               variance           actual evals  cache hits')
    for count, case in cases.items():
        objective = case.evaluation.objective
        optimization = case.optimization
        print(f'{count:9d}  {count - 1:7d}  {objective.total:.10e}  {objective.bias:.10e}  {objective.variance:.10e}  {optimization.objective_evaluations:12d}  {optimization.cache_hits:10d}')
        print(' ' * 20 + 'prices=' + np.array2string(np.asarray(optimization.best.x, dtype=float), precision=6,
            floatmode='fixed'))
    print()
    print('Marginal pure-FV improvement')
    previous = None
    for count, case in cases.items():
        if previous is not None:
            improvement = previous.evaluation.objective.total - case.evaluation.objective.total
            relative = improvement / previous.evaluation.objective.total
            print(f'N={previous.n_intervals} -> N={count}: dJ={improvement:.6e} ({relative:.3%})')
        previous = case
    if result['mc'] is not None:
        print()
        print(f'Common-random-number Monte Carlo ({MULTI_MC_PATHS} paths)')
        for count in MULTI_INTERVAL_COUNTS:
            item = result['mc'][count]
            print(f'N={count}: J={item.objective.total:.10e}, SE={item.standard_error:.3e}')
        reference = result['mc'][MULTI_INTERVAL_COUNTS[0]]
        print()
        print(f'Paired MC improvement relative to N={MULTI_INTERVAL_COUNTS[0]}')
        for count in MULTI_INTERVAL_COUNTS[1:]:
            mean, se = multi_paired_improvement(reference, result['mc'][count])
            ratio = mean / se if se > 0.0 else np.inf
            print(f'N={count}: improvement={mean:.6e}, SE={se:.3e}, ratio={ratio:.3f}')
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    ax_price, ax_objective, ax_demand = axes
    for count, case in cases.items():
        price = input_builder.price_intervals((case.edges, case.optimization.best.x))
        plot_t = np.linspace(control_T[0], control_T[1], 721)
        ax_price.step(24.0 * plot_t, price(plot_t), where='post', label=f'N={count}')
    ax_price.set_ylabel('Normalized price')
    ax_price.set_ylim(0.0, 1.0)
    ax_price.set_title('Optimized stochastic price schedules')
    ax_price.grid(True)
    ax_price.legend(loc='best')
    counts = np.asarray(MULTI_INTERVAL_COUNTS, dtype=int)
    totals = np.array([cases[count].evaluation.objective.total for count in counts])
    bias = np.array([cases[count].evaluation.objective.bias for count in counts])
    variance = np.array([cases[count].evaluation.objective.variance for count in counts])
    ax_objective.plot(counts, totals, marker='o', label='Total J')
    ax_objective.plot(counts, bias, marker='o', label='Bias contribution')
    ax_objective.plot(counts, variance, marker='o', label='Variance contribution')
    ax_objective.set_xlabel('Number of price intervals')
    ax_objective.set_ylabel('Pure-FV stochastic objective')
    ax_objective.set_title('Value of additional price decisions')
    ax_objective.grid(True)
    ax_objective.legend(loc='best')
    reference_evaluator = cases[MULTI_INTERVAL_COUNTS[0]].evaluator
    hours = 24.0 * reference_evaluator.midpoints
    ax_demand.plot(hours, reference_evaluator.target, linewidth=2.5, label='Target A(t)')
    for count, case in cases.items():
        ax_demand.plot(hours, case.evaluation.objective.expected_demand, label=f'E[D], N={count}')
    ax_demand.set_xlabel('Time [h]')
    ax_demand.set_ylabel('Expected demand')
    ax_demand.set_title('Expected demand under optimized price schedules')
    ax_demand.grid(True)
    ax_demand.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'graphs' / 'bm_1'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_multi_price_optimization_experiment.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print()
    print(f'Saved graph to {graph_path}')

def run_contract():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize', action='store_true', help='also solve the bounded open-loop price problem')
    args = parser.parse_args()
    result = run_reference(optimize=args.optimize, progress=lambda message: print(message, flush=True))
    problem = result['problem']
    fixed = result['fixed']
    print('F3-A common BM1 control problem')
    print('-------------------------------')
    print(f'price intervals: {problem.n_intervals}')
    print('fixed external/reference prices: ' + np.array2string(problem.observed_prices, precision=6))
    print(f'fixed-price J={fixed.objective.total:.8e}, bias={fixed.objective.bias:.8e}, variance={fixed.objective.variance:.8e}')
    print('The fixed path above was evaluated unchanged; no price optimization was performed.')
    if result['optimization'] is not None:
        optimization = result['optimization']
        print()
        print('Open-loop candidate optimization')
        print('--------------------------------')
        print('best prices: ' + np.array2string(np.asarray(optimization.best.x, dtype=float), precision=6))
        print(f'best J: {float(optimization.best.fun):.8e}')
        print(f'actual objective evaluations/cache hits: {optimization.objective_evaluations}/{optimization.cache_hits}')


def main():
    import sys
    commands = {'deterministic': 'run_deterministic', 'stochastic': 'run_stochastic', 'complex': 'run_complex',
        'multi': 'run_multi', 'contract': 'run_contract'}
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Cases: ' + ', '.join(commands))
        print('Run one case, then pass its options after the case name.')
        return
    case = sys.argv.pop(1)
    if case not in commands: raise SystemExit('Unknown case: ' + case)
    globals()[commands[case]]()


if __name__ == '__main__':
    main()
