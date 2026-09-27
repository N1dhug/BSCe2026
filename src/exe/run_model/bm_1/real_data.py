"""BM1 executable: input selection, reporting and plots."""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from src.method.bm_1_control import (FastFVStochasticPriceEvaluator, beta_initial_cell_mass,
    evaluate_mc_candidates_common_random_numbers, multistart_powell, paired_mc_improvement)
from src.model.bm_1 import bm_1
from src.tools.bm_1_data import (DEFAULT_BASELINE_WEEKS, DEFAULT_DAYS, DaySignals, HOURLY_EDGES, ROOT, _hourly_hold,
    _json_safe, _mc_summary, _objective_dict, _price_correlation, build_day_signals, load_real_signals,
    select_representative_days)
from src.tools.input_generator.input_builder import input_builder
from src.validation.bm_1.cases import (INITIAL_ALPHA, INITIAL_BETA, SIGMA_X, linear_price_response,
    linear_state_response)
from src.validation.bm_1.control import (C, K, T, XI)

OUTPUT_DIR = ROOT / 'files' / 'output' / 'bm_1' / 'real_data'
DEFAULT_CELLS = 201
DEFAULT_MC_PATHS = 20000
OBJECTIVE_MAX_STEP = 1.0 / 96.0
SOLVER_MAX_STEP = 1.0 / 96.0
MC_INITIAL_SEED = 2091
MC_BROWNIAN_SEED = 2092

@dataclass(frozen=True)
class DayExperiment:
    signals: DaySignals
    evaluator: FastFVStochasticPriceEvaluator
    optimization: object
    evaluations: dict
    mc: dict | None
    paired_mc: dict | None
    plot_path: Path

def _parameter_template():
    neutral = input_builder.price_intervals((HOURLY_EDGES, np.full(24, 0.5, dtype=float)))
    return [neutral, linear_state_response, linear_price_response, C, K, XI]

def save_day_plot(signals, evaluator, evaluations, optimized_prices):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    ax_demand, ax_price, ax_objective = axes
    hours = 24.0 * evaluator.midpoints
    plot_t = np.linspace(0.0, 1.0, 1441)
    plot_hours = 24.0 * plot_t
    target = _hourly_hold(signals.target)
    baseline = _hourly_hold(signals.baseline)
    ax_demand.step(plot_hours, target(plot_t), where='post', linewidth=2.2, label='Observed target A(t)')
    ax_demand.step(plot_hours, baseline(plot_t), where='post', linestyle='--', label='History baseline B(t)')
    ax_demand.plot(hours, evaluations['observed_spot'].objective.expected_demand, label='E[D] with observed DK1 price')
    ax_demand.plot(hours, evaluations['optimized'].objective.expected_demand, label='E[D] with optimized price')
    ax_demand.set_ylabel('Normalized consumption')
    ax_demand.set_title(f'Real-data BM1 tracking — {signals.day:%Y-%m-%d}')
    ax_demand.grid(True, alpha=0.25)
    ax_demand.legend(loc='best')
    edge_hours = np.arange(25, dtype=float)
    observed_plot = np.append(signals.observed_price, signals.observed_price[-1])
    optimized_plot = np.append(optimized_prices, optimized_prices[-1])
    ax_price.step(edge_hours, observed_plot, where='post', label='Observed normalized DK1 spot price')
    ax_price.step(edge_hours, optimized_plot, where='post', label='BM1 optimized hourly price')
    ax_price.set_xlim(0.0, 24.0)
    ax_price.set_ylim(0.0, 1.0)
    ax_price.set_ylabel('Normalized price')
    ax_price.set_xlabel('Hour')
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc='best')
    names = ('Observed DK1', 'Neutral 0.5', 'Optimized')
    keys = ('observed_spot', 'neutral', 'optimized')
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
    path = OUTPUT_DIR / f'bm_1_real_data_{signals.day:%Y-%m-%d}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path

def _result_record(result):
    signals = result.signals
    optimized = np.asarray(result.optimization.best.x, dtype=float)
    record = {'day': signals.day, 'baseline_days': list(signals.baseline_days),
        'raw_consumption_mean': float(np.mean(signals.raw_target)),
        'raw_price_mean_dkk_per_mwh': float(np.mean(signals.raw_price)),
        'raw_price_min_dkk_per_mwh': float(np.min(signals.raw_price)),
        'raw_price_max_dkk_per_mwh': float(np.max(signals.raw_price)), 'normalized_target': signals.target,
        'normalized_baseline': signals.baseline, 'normalized_observed_price': signals.observed_price,
        'optimized_price': optimized, 'price_correlation': _price_correlation(signals.observed_price, optimized),
        'objective': {name: _objective_dict(evaluation) for name, evaluation in result.evaluations.items()},
        'optimization': {'objective_evaluations': int(result.optimization.objective_evaluations), 'cache_hits': int(result.optimization.cache_hits), 'elapsed_seconds': float(result.optimization.elapsed_seconds)},
        'plot': result.plot_path}
    if result.mc is not None:
        record['mc'] = {name: _mc_summary(item) for name, item in result.mc.items()}
        record['paired_mc'] = result.paired_mc
    return record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', nargs='+', default=None,
        help='Explicit YYYY-MM-DD days. Omit to auto-select representative days.')
    parser.add_argument('--n-days', type=int, default=DEFAULT_DAYS,
        help='Number of automatically selected low/typical/high-consumption days.')
    parser.add_argument('--baseline-weeks', type=int, default=DEFAULT_BASELINE_WEEKS,
        help='Previous matching weekdays used for the trimmed baseline.')
    parser.add_argument('--price-area', default='DK1')
    parser.add_argument('--cells', type=int, default=DEFAULT_CELLS)
    parser.add_argument('--mc-paths', type=int, default=DEFAULT_MC_PATHS)
    parser.add_argument('--no-mc', action='store_true')
    args = parser.parse_args()
    print('BM1 real-data retrospective diagnostic', flush=True)
    print('=====================================', flush=True)
    print('Target is observed consumption; baseline uses only previous matching weekdays. Reference BM parameters are not fitted to these days.',
        flush=True)
    experiments, normalization = run_real_data_experiment(days=args.days, n_days=args.n_days,
        baseline_weeks=args.baseline_weeks, price_area=args.price_area, cells=args.cells, mc_paths=args.mc_paths,
        validate_mc=not args.no_mc, progress=lambda message: print(message, flush=True))
    print()
    print(f'Season normalization: consumption=[{normalization.consumption_min:.6g}, {normalization.consumption_max:.6g}], DK1 price=[{normalization.price_min:.6g}, {normalization.price_max:.6g}] DKK/MWh')
    records = []
    for result in experiments:
        signals = result.signals
        optimized = np.asarray(result.optimization.best.x, dtype=float)
        observed = result.evaluations['observed_spot'].objective
        neutral = result.evaluations['neutral'].objective
        optimum = result.evaluations['optimized'].objective
        print()
        print(f'{signals.day:%Y-%m-%d}')
        print('-' * 32)
        print('baseline days: ' + ', '.join((day.strftime('%Y-%m-%d') for day in signals.baseline_days)))
        print(f'raw mean consumption={np.mean(signals.raw_target):.6g}; DK1 price mean/min/max={np.mean(signals.raw_price):.3f}/{np.min(signals.raw_price):.3f}/{np.max(signals.raw_price):.3f} DKK/MWh')
        print(f'FV J observed={observed.total:.10e}, neutral={neutral.total:.10e}, optimized={optimum.total:.10e}')
        print(f'optimized bias={optimum.bias:.10e}, variance={optimum.variance:.10e}')
        print(f'FV improvement vs observed DK1={(observed.total - optimum.total) / observed.total:.3%}; vs neutral={(neutral.total - optimum.total) / neutral.total:.3%}')
        print('observed normalized DK1=' + np.array2string(signals.observed_price, precision=4, floatmode='fixed',
            max_line_width=120))
        print('optimized hourly price=' + np.array2string(optimized, precision=4, floatmode='fixed',
            max_line_width=120))
        print(f'observed/optimized price correlation={_price_correlation(signals.observed_price, optimized):.4f}')
        print(f'Powell evaluations={result.optimization.objective_evaluations}, time={result.optimization.elapsed_seconds:.3f}s')
        if result.mc is not None:
            print(f"MC J observed={result.mc['observed_spot'].objective.objective.total:.10e} (SE={result.mc['observed_spot'].objective.standard_error:.3e}), optimized={result.mc['optimized'].objective.objective.total:.10e} (SE={result.mc['optimized'].objective.standard_error:.3e})")
            paired = result.paired_mc['observed_spot']
            print(f"paired MC observed -> optimized improvement={paired['improvement']:.6e}, SE={paired['standard_error']:.3e}, ratio={paired['ratio']:.3f}")
        print(f'plot={result.plot_path}')
        records.append(_result_record(result))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'bm_1_real_data_daily_results.json'
    payload = {'note': 'Retrospective tracking diagnostic; target day is observed and reference BM parameters are not fitted to real data.',
        'normalization': normalization.__dict__, 'baseline_weeks': int(args.baseline_weeks), 'cells': int(args.cells),
        'mc_paths': 0 if args.no_mc else int(args.mc_paths), 'days': records}
    output_path.write_text(json.dumps(payload, indent=2, default=_json_safe) + '\n', encoding='utf-8')
    print()
    print(f'Saved summary: {output_path}')

def build_evaluator(signals, *, cells=DEFAULT_CELLS):
    baseline = _hourly_hold(signals.baseline)
    target = _hourly_hold(signals.target)
    model = bm_1(baseline, target, T, 2)
    spatial_edges, _, _ = model.fokker_planc_reference.grid(int(cells))
    initial_mass = beta_initial_cell_mass(spatial_edges, alpha=INITIAL_ALPHA, beta=INITIAL_BETA)
    return FastFVStochasticPriceEvaluator(model, _parameter_template(), HOURLY_EDGES, SIGMA_X, initial_mass,
        objective_max_step=OBJECTIVE_MAX_STEP, solver_max_step=SOLVER_MAX_STEP, breakpoints=HOURLY_EDGES[1:-1])

def run_day(signals, *, cells=DEFAULT_CELLS, mc_paths=DEFAULT_MC_PATHS, validate_mc=True, progress=print):
    evaluator = build_evaluator(signals, cells=cells)
    neutral = np.full(24, 0.5, dtype=float)
    starts = np.vstack((signals.observed_price, neutral))
    if progress is not None:
        progress(f'{signals.day:%Y-%m-%d}: optimizing 24 hourly prices from observed DK1 and neutral starts')
    optimization = multistart_powell(evaluator, starts, progress=progress,
        options={'xtol': 5e-05, 'ftol': 1e-09, 'maxiter': 100})
    optimized = np.asarray(optimization.best.x, dtype=float)
    evaluations = {'observed_spot': evaluator.evaluate(signals.observed_price), 'neutral': evaluator.evaluate(neutral),
        'optimized': evaluator.evaluate(optimized)}
    mc = None
    paired = None
    if validate_mc:
        mc = evaluate_mc_candidates_common_random_numbers(evaluator,
            {'observed_spot': signals.observed_price, 'neutral': neutral, 'optimized': optimized},
            n_paths=int(mc_paths), initial_seed=MC_INITIAL_SEED, brownian_seed=MC_BROWNIAN_SEED)
        paired = {}
        for reference_name in ('observed_spot', 'neutral'):
            mean, standard_error = paired_mc_improvement(mc[reference_name], mc['optimized'])
            paired[reference_name] = {'improvement': float(mean), 'standard_error': float(standard_error),
                'ratio': float(mean / standard_error) if standard_error > 0.0 else float('inf')}
    path = save_day_plot(signals, evaluator, evaluations, optimized)
    return DayExperiment(signals=signals, evaluator=evaluator, optimization=optimization, evaluations=evaluations,
        mc=mc, paired_mc=paired, plot_path=path)

def run_real_data_experiment(*, days=None, n_days=DEFAULT_DAYS, baseline_weeks=DEFAULT_BASELINE_WEEKS,
    price_area='DK1', cells=DEFAULT_CELLS, mc_paths=DEFAULT_MC_PATHS, validate_mc=True, progress=print):
    consumption_raw, price_raw, consumption, price, normalization = load_real_signals(price_area=price_area)
    if days:
        selected = [pd.Timestamp(day).normalize() for day in days]
    else:
        selected = select_representative_days(consumption_raw, price_raw, consumption, price, n_days=n_days,
            baseline_weeks=baseline_weeks)
    experiments = []
    for day in selected:
        signals = build_day_signals(consumption_raw, price_raw, consumption, price, day, baseline_weeks=baseline_weeks)
        experiments.append(run_day(signals, cells=cells, mc_paths=mc_paths, validate_mc=validate_mc,
            progress=progress))
    return (experiments, normalization)


if __name__ == '__main__':
    main()
