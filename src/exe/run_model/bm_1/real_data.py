"""Run the F3-G historical-price application with fixed model parameters.

By default the control target is the history-only baseline, so current-day
observed consumption is not used as the counterfactual target.  The old
observed-consumption target remains available explicitly as a retrospective
diagnostic.  This is not parameter estimation.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.control import bm_1 as control
from src.control.objective import standard_error_ratio
from src.tools import bm_1_data as data
from src.validation.bm_1 import cases
from src.validation.bm_1 import control as scenario


OUTPUT_DIR = data.ROOT / 'files' / 'output' / 'bm_1' / 'runs' / 'real_data'
DEFAULT_CELLS = 201
DEFAULT_MC_PATHS = 20000
OBJECTIVE_MAX_STEP = 1.0 / 96.0
SOLVER_MAX_STEP = 1.0 / 96.0
MC_INITIAL_SEED = 2091
MC_BROWNIAN_SEED = 2092
TARGET_MODES = ('baseline', 'observed')


@dataclass(frozen=True)
class DayExperiment:
    signals: data.DaySignals
    target_mode: str
    problem: control.BM1ControlProblem
    numerics: control.BM1ControlNumerics
    evaluator: control.FastFVStochasticPriceEvaluator
    optimization: object | None
    evaluations: dict
    mc: dict | None
    fixed_paired_mc: dict | None
    paired_mc: dict | None
    plot_path: Path | None


def _initial_mass_builder(edges):
    return control.beta_initial_cell_mass(
        edges,
        alpha=cases.INITIAL_ALPHA,
        beta=cases.INITIAL_BETA,
    )


def _parameters():
    return control.BM1Parameters(
        state_response=cases.linear_state_response,
        price_response=cases.linear_price_response,
        C=scenario.C,
        k=scenario.K,
        Xi=scenario.XI,
        sigma_x=cases.SIGMA_X,
    )


def _validated_target_mode(target_mode):
    target_mode = str(target_mode)
    if target_mode not in TARGET_MODES:
        raise ValueError(f'target_mode must be one of {TARGET_MODES}.')
    return target_mode


def _target_values(signals, target_mode):
    target_mode = _validated_target_mode(target_mode)
    # "baseline" is the F3-G application target: it uses only preceding
    # matching weekdays and is independent of current-day observed consumption.
    return signals.baseline if target_mode == 'baseline' else signals.target


def build_problem(signals, *, target_mode='baseline'):
    """Attach one historical spot-price day to the common immutable F3 contract."""
    if not isinstance(signals, data.DaySignals):
        raise TypeError('signals must be a DaySignals instance.')
    target_mode = _validated_target_mode(target_mode)
    return control.BM1ControlProblem(
        baseline=data._hourly_hold(signals.baseline),
        target=data._hourly_hold(_target_values(signals, target_mode)),
        horizon=scenario.T,
        price_edges=data.HOURLY_EDGES,
        parameters=_parameters(),
        initial_mass_builder=_initial_mass_builder,
        deterministic_initial_state=scenario.X0_DETERMINISTIC,
        breakpoints=data.HOURLY_EDGES[1:-1],
        observed_prices=signals.observed_price,
    )


def build_numerics(*, cells=DEFAULT_CELLS):
    """Numerics for the historical day, using the established F3 working settings."""
    return control.BM1ControlNumerics(
        x_steps=cells,
        objective_max_step=OBJECTIVE_MAX_STEP,
        solver_max_step=SOLVER_MAX_STEP,
        model_t_steps=2,
    )


def save_day_plot(
    signals,
    evaluator,
    evaluations,
    optimized_prices=None,
    *,
    target_mode='baseline',
    output_dir=OUTPUT_DIR,
):
    """Plot one evaluated day only when explicitly requested."""
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    ax_demand, ax_price, ax_objective = axes
    hours = 24.0 * evaluator.midpoints
    plot_t = np.linspace(0.0, 1.0, 1441)
    plot_hours = 24.0 * plot_t
    target_mode = _validated_target_mode(target_mode)
    target = data._hourly_hold(_target_values(signals, target_mode))
    baseline = data._hourly_hold(signals.baseline)
    ax_demand.step(
        plot_hours,
        target(plot_t),
        where='post',
        linewidth=2.2,
        label=('History baseline = chosen target A(t)' if target_mode == 'baseline'
               else 'Observed consumption = retrospective target A(t)'),
    )
    if target_mode == 'observed':
        ax_demand.step(
            plot_hours,
            baseline(plot_t),
            where='post',
            linestyle='--',
            label='History baseline B(t)',
        )
    else:
        observed = data._hourly_hold(signals.target)
        ax_demand.step(
            plot_hours,
            observed(plot_t),
            where='post',
            linestyle=':',
            label='Observed consumption (diagnostic only)',
        )
    ax_demand.plot(
        hours,
        evaluations['observed_spot'].objective.expected_demand,
        label='E[D] with observed DK1 price',
    )
    ax_demand.plot(
        hours,
        evaluations['neutral'].objective.expected_demand,
        linestyle=':',
        label='E[D] with neutral 0.5 price',
    )
    if 'optimized' in evaluations:
        ax_demand.plot(
            hours,
            evaluations['optimized'].objective.expected_demand,
            label='E[D] with optimized price',
        )
    ax_demand.set_ylabel('Normalized consumption')
    ax_demand.set_title(f'Real-data BM1 tracking — {signals.day:%Y-%m-%d}')
    ax_demand.grid(True, alpha=0.25)
    ax_demand.legend(loc='best')
    edge_hours = np.arange(25, dtype=float)
    observed_plot = np.append(signals.observed_price, signals.observed_price[-1])
    neutral_plot = np.full(edge_hours.size, 0.5, dtype=float)
    ax_price.step(edge_hours, observed_plot, where='post', label='Observed normalized DK1 spot price')
    ax_price.step(
        edge_hours,
        neutral_plot,
        where='post',
        linestyle='--',
        label='Neutral 0.5 benchmark',
    )
    if optimized_prices is not None:
        optimized_prices = np.asarray(optimized_prices, dtype=float)
        optimized_plot = np.append(optimized_prices, optimized_prices[-1])
        ax_price.step(edge_hours, optimized_plot, where='post', label='BM1 retrospective optimized price')
    ax_price.set_xlim(0.0, 24.0)
    ax_price.set_ylim(0.0, 1.0)
    ax_price.set_ylabel('Normalized price')
    ax_price.set_xlabel('Hour')
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc='best')
    keys = ['observed_spot', 'neutral']
    names = ['Observed DK1', 'Neutral 0.5']
    if 'optimized' in evaluations:
        keys.append('optimized')
        names.append('Retrospective optimized')
    bias = np.array([evaluations[key].objective.bias for key in keys])
    variance = np.array([evaluations[key].objective.variance for key in keys])
    positions = np.arange(len(keys))
    ax_objective.bar(positions, bias, label='Bias contribution')
    ax_objective.bar(positions, variance, bottom=bias, label='Variance contribution')
    ax_objective.set_xticks(positions, names)
    ax_objective.set_ylabel('Stochastic tracking objective')
    ax_objective.set_title('Objective decomposition')
    ax_objective.grid(True, axis='y', alpha=0.25)
    ax_objective.legend(loc='best')
    fig.tight_layout()
    path = output_dir / f'bm_1_real_data_{signals.day:%Y-%m-%d}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def _result_record(result):
    signals = result.signals
    optimized = (
        np.asarray(result.optimization.best.x, dtype=float)
        if result.optimization is not None else None
    )
    record = {
        'day': signals.day,
        'target_mode': result.target_mode,
        'baseline_days': list(signals.baseline_days),
        'raw_consumption_mean': float(np.mean(signals.raw_target)),
        'raw_price_mean_dkk_per_mwh': float(np.mean(signals.raw_price)),
        'raw_price_min_dkk_per_mwh': float(np.min(signals.raw_price)),
        'raw_price_max_dkk_per_mwh': float(np.max(signals.raw_price)),
        'normalized_observed_consumption': signals.target,
        'normalized_control_target': _target_values(signals, result.target_mode),
        'normalized_baseline': signals.baseline,
        'normalized_observed_price': signals.observed_price,
        'optimized_price': optimized,
        'price_correlation': (
            data._price_correlation(signals.observed_price, optimized)
            if optimized is not None else None
        ),
        'objective': {
            name: data._objective_dict(evaluation)
             for name, evaluation in result.evaluations.items()
        },
        'optimization': (
            {
                'objective_evaluations': int(result.optimization.objective_evaluations),
                'cache_hits': int(result.optimization.cache_hits),
            }
            if result.optimization is not None else None
        ),
        'plot': _plot_location(result.plot_path),
    }
    if result.mc is not None:
        record['mc'] = {name: data._mc_summary(item) for name, item in result.mc.items()}
        record['fixed_paired_mc'] = result.fixed_paired_mc
        record['paired_mc'] = result.paired_mc
    return record


def build_evaluator(signals, *, cells=DEFAULT_CELLS, target_mode='baseline'):
    """Compatibility helper: construct the fast FV evaluator through BM1ControlProblem."""
    problem = build_problem(signals, target_mode=target_mode)
    return problem.build_evaluator(build_numerics(cells=cells), fast=True)


def run_day(
    signals,
    *,
    cells=DEFAULT_CELLS,
    mc_paths=DEFAULT_MC_PATHS,
    validate_mc=True,
    optimize=True,
    target_mode='baseline',
    progress=None,
    save=False,
    output_dir=OUTPUT_DIR,
):
    """Evaluate historical prices unchanged, then optionally optimize an alternative plan."""
    target_mode = _validated_target_mode(target_mode)
    problem = build_problem(signals, target_mode=target_mode)
    numerics = build_numerics(cells=cells)
    evaluator = problem.build_evaluator(numerics, fast=True)
    neutral = np.full(problem.n_intervals, 0.5, dtype=float)

    # F3-G core: historical spot prices are attached external inputs and are
    # evaluated unchanged through BM1ControlProblem, never used as a template.
    evaluations = {
        'observed_spot': problem.evaluate_observed(numerics, fast=True),
        'neutral': evaluator.evaluate(neutral),
    }

    optimization = None
    optimized = None
    if optimize:
        starts = np.vstack((problem.observed_prices, neutral))
        if progress is not None:
            progress(
                f'{signals.day:%Y-%m-%d}: 24-price optimization '
                f'(target_mode={target_mode}) from observed DK1 and neutral starts',
            )
        optimization = control.multistart_powell(
            evaluator,
            starts,
            progress=progress,
            options={'xtol': 5e-05, 'ftol': 1e-09, 'maxiter': 100},
        )
        optimized = np.asarray(optimization.best.x, dtype=float)
        evaluations['optimized'] = evaluator.evaluate(optimized)

    mc = None
    fixed_paired = None
    paired = None
    if validate_mc:
        candidates = {
            'observed_spot': problem.observed_prices,
            'neutral': neutral,
        }
        if optimized is not None:
            candidates['optimized'] = optimized
        mc = control.evaluate_mc_candidates_common_random_numbers(
            evaluator,
            candidates,
            n_paths=mc_paths,
            initial_seed=MC_INITIAL_SEED,
            brownian_seed=MC_BROWNIAN_SEED,
        )
        mean, standard_error = control.paired_mc_improvement(
            mc['neutral'], mc['observed_spot'],
        )
        fixed_paired = {
            'neutral_minus_observed_spot': float(mean),
            'standard_error': float(standard_error),
            'ratio': standard_error_ratio(mean, standard_error),
        }
        if optimized is not None:
            paired = {}
            for reference_name in ('observed_spot', 'neutral'):
                mean, standard_error = control.paired_mc_improvement(mc[reference_name], mc['optimized'])
                paired[reference_name] = {
                    'improvement': float(mean),
                    'standard_error': float(standard_error),
                    'ratio': standard_error_ratio(mean, standard_error),
                }

    path = (
        save_day_plot(
            signals,
            evaluator,
            evaluations,
            optimized,
            target_mode=target_mode,
            output_dir=output_dir,
        )
        if save else None
    )
    return DayExperiment(
        signals=signals,
        target_mode=target_mode,
        problem=problem,
        numerics=numerics,
        evaluator=evaluator,
        optimization=optimization,
        evaluations=evaluations,
        mc=mc,
        fixed_paired_mc=fixed_paired,
        paired_mc=paired,
        plot_path=path,
    )


def run_real_data_experiment(
    *,
    days=None,
    n_days=data.DEFAULT_DAYS,
    baseline_weeks=data.DEFAULT_BASELINE_WEEKS,
    price_area='DK1',
    cells=DEFAULT_CELLS,
    mc_paths=DEFAULT_MC_PATHS,
    validate_mc=True,
    optimize=True,
    target_mode='baseline',
    progress=None,
    save=False,
    output_dir=OUTPUT_DIR,
):
    """Load aligned historical inputs, choose complete days, and run them independently."""
    consumption_raw, price_raw, consumption, price, normalization = data.load_real_signals(
        price_area=price_area,
    )
    if days:
        selected = [pd.Timestamp(day).normalize() for day in days]
    else:
        selected = data.select_representative_days(
            consumption_raw,
            price_raw,
            consumption,
            price,
            n_days=n_days,
            baseline_weeks=baseline_weeks,
        )
    experiments = []
    for day in selected:
        signals = data.build_day_signals(
            consumption_raw,
            price_raw,
            consumption,
            price,
            day,
            baseline_weeks=baseline_weeks,
        )
        experiments.append(
            run_day(
                signals,
                cells=cells,
                mc_paths=mc_paths,
                validate_mc=validate_mc,
                optimize=optimize,
                target_mode=target_mode,
                progress=progress,
                save=save,
                output_dir=output_dir,
            ),
        )
    return (experiments, normalization)


def _plot_location(path):
    """Store repository-relative paths when possible, including Windows separators."""
    if path is None:
        return None
    path = Path(path).resolve()
    return path.relative_to(data.ROOT).as_posix() if path.is_relative_to(data.ROOT) else path.as_posix()


def report_day(result, *, verbose=False):
    """Distinguish retrospective tracking from an estimated causal demand response."""
    target_label = (
        'history-only baseline target'
        if result.target_mode == 'baseline'
        else 'observed-consumption retrospective target'
    )
    print(f'\n{result.signals.day:%Y-%m-%d}: fixed model parameters, {target_label}')
    for name, evaluation in result.evaluations.items():
        cost = evaluation.objective
        print(f'{name:14s} FV J={cost.total:.8e} bias={cost.bias:.3e} variance={cost.variance:.3e}')
    fit = result.optimization
    if fit is None:
        print('Alternative price optimization skipped; observed spot prices were evaluated unchanged.')
    else:
        successes = sum((bool(item.success) for item in fit.results))
        print(
            (
                f'Powell: {successes}/{len(fit.results)} successful starts; evaluations='
                f'{fit.objective_evaluations}; time={fit.elapsed_seconds:.2f}s'
            ),
        )
    if result.mc is not None:
        for name, item in result.mc.items():
            print(
                f'MC {name:14s} J={item.total:.8e} SE={item.standard_error:.3e}',
            )
    if result.fixed_paired_mc is not None:
        row = result.fixed_paired_mc
        print(
            (
                'Paired MC neutral - observed_spot: '
                f"difference={row['neutral_minus_observed_spot']:.6e}, "
                f"SE={row['standard_error']:.3e}; "
                'positive means observed spot has lower MC cost'
            ),
        )
    if result.paired_mc is not None:
        for name, row in result.paired_mc.items():
            print(
                (
                    f"Paired MC {name} -> optimized: improvement={row['improvement']:.6e}, SE="
                    f"{row['standard_error']:.3e}"
                ),
            )
    if verbose:
        print(
            'Baseline history:',
            ', '.join((day.strftime('%Y-%m-%d') for day in result.signals.baseline_days)),
        )
        print('Observed prices:', result.problem.observed_prices)
        if fit is not None:
            print('Optimized alternative prices:', np.asarray(fit.best.x))
            print('Price correlation:', data._price_correlation(result.problem.observed_prices, fit.best.x))
    if result.plot_path is not None:
        print('Saved graph to', result.plot_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--days',
        nargs='+',
        help='Explicit YYYY-MM-DD days; otherwise select representative days.',
    )
    parser.add_argument('--n-days', type=int, default=data.DEFAULT_DAYS)
    parser.add_argument('--baseline-weeks', type=int, default=data.DEFAULT_BASELINE_WEEKS)
    parser.add_argument('--price-area', default='DK1')
    parser.add_argument('--cells', type=int, default=DEFAULT_CELLS)
    parser.add_argument('--mc-paths', type=int, default=DEFAULT_MC_PATHS)
    parser.add_argument('--no-mc', action='store_true', help='Skip only the Monte Carlo comparison.')
    parser.add_argument(
        '--target-mode',
        choices=TARGET_MODES,
        default='baseline',
        help=(
            "Control target: 'baseline' (default, history-only F3-G application) "
            "or 'observed' (legacy retrospective diagnostic)."
        ),
    )
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='F3-G fixed-input mode: evaluate observed spot and neutral prices without retrospective optimization.',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show full prices, metadata and solver progress.',
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Write figures and JSON under ignored runs/real_data/.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help='Destination used only with --save.',
    )
    args = parser.parse_args(argv)
    print(
        'BM1 F3-G: observed prices are fixed external inputs; '
        'the control target is selected explicitly; parameters are NOT fitted.',
    )
    experiments, normalization = run_real_data_experiment(
        days=args.days,
        n_days=args.n_days,
        baseline_weeks=args.baseline_weeks,
        price_area=args.price_area,
        cells=args.cells,
        mc_paths=args.mc_paths,
        validate_mc=not args.no_mc,
        optimize=not args.no_optimize,
        target_mode=args.target_mode,
        progress=print if args.verbose else None,
        save=args.save,
        output_dir=args.output_dir,
    )
    for result in experiments:
        report_day(result, verbose=args.verbose)
    if args.save:
        payload = {
            'note': (
                'F3-G historical-price application; observed spot prices enter as fixed external inputs, '
                'the target is selected explicitly, and reference BM parameters are not fitted to real data.'
            ),
            'target_mode': args.target_mode,
            'normalization': normalization.__dict__,
            'baseline_weeks': int(args.baseline_weeks),
            'cells': int(args.cells),
            'mc_paths': 0 if args.no_mc else int(args.mc_paths),
            'retrospective_optimization': not args.no_optimize,
            'days': [_result_record(result) for result in experiments],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / 'bm_1_real_data_daily_results.json'
        path.write_text(json.dumps(data._json_safe(payload), indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'Saved summary to {path}')
    return experiments
if __name__ == '__main__':
    main()
