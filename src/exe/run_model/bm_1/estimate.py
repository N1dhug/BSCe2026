"""F4 estimation entry point.

F4-B exposes the reproducible synthetic generator. F4-C adds the known-parameter
finite-volume state filter. F4-D begins with static mixture representation before
any recursive mixture filter is accepted. Parameter estimation remains later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.validation.bm_1 import estimation as synthetic
from src.validation.bm_1 import mixture_estimation as mixture_validation


ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT / 'files' / 'output' / 'bm_1' / 'runs' / 'estimation'


def _diagnostics_record(values):
    record = {}
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            record[key] = value
    return record


def _sequence_record(result):
    return {
        'sequence_id': result.sequence_id,
        'observation_times': result.observations.times.tolist(),
        'observations': result.observations.values.tolist(),
        'truth': {
            'times': result.truth.times.tolist(),
            'state': result.truth.state.tolist(),
            'price': result.truth.price.tolist(),
            'baseline': result.truth.baseline.tolist(),
            'demand': result.truth.demand.tolist(),
            'initial_state': result.truth.initial_state,
            'valid': result.truth.valid,
            'diagnostics': _diagnostics_record(result.truth.diagnostics),
        },
        'measurement_noise': result.measurement_noise.tolist(),
    }


def _print_sequence(result, label='base'):
    diagnostics = result.truth.diagnostics
    print(f'F4-B synthetic {label}: sequence {result.sequence_id}')
    print(
        f"  X0={result.truth.initial_state:.6f}; "
        f"state range=[{np.min(result.truth.state):.6f}, {np.max(result.truth.state):.6f}]"
    )
    print(
        f"  observations={result.observations.times.size}; "
        f"EM steps={diagnostics['n_steps']}; "
        f"raw boundary violations={diagnostics['total_violations']}"
    )
    print(
        f"  first Y={result.observations.values[0]:.6f}; "
        f"last Y={result.observations.values[-1]:.6f}"
    )


def run_synthetic(args):
    case = synthetic.build_f4_b_case()
    if args.refine:
        result = synthetic.generate_coupled_refinement(case, sequence_id=args.sequence_id)
        _print_sequence(result.coarse, 'coarse')
        _print_sequence(result.fine, 'fine')
        print(f'  max |X_coarse-X_fine|={result.maximum_abs_state_difference:.6e}')
        print(f'  max |D_coarse-D_fine|={result.maximum_abs_demand_difference:.6e}')
        payload = {
            'case': 'F4-A locked core',
            'refinement': {
                'coarse': _sequence_record(result.coarse),
                'fine': _sequence_record(result.fine),
                'maximum_abs_state_difference': result.maximum_abs_state_difference,
                'maximum_abs_demand_difference': result.maximum_abs_demand_difference,
            },
        }
        suffix = 'refinement'
    else:
        result = synthetic.generate_synthetic_sequence(case, sequence_id=args.sequence_id)
        _print_sequence(result)
        payload = {'case': 'F4-A locked core', 'sequence': _sequence_record(result)}
        suffix = 'sequence'
    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / f'f4_b_{suffix}_{args.sequence_id:04d}.json'
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'Saved {path}')
    return result



def run_filter(args):
    """Run one F4-C known-parameter filter sanity case."""
    result = synthetic.run_f4_c_reference(
        sequence_id=args.sequence_id,
        x_steps=args.cells,
        max_step=args.max_step,
        likelihood_quadrature_order=args.quadrature_order,
    )
    truth = result.sequence.truth.state[1:]
    open_mean = result.open_propagation.posterior_mean
    predictive = result.filtered.predicted_mean
    posterior = result.filtered.posterior_mean

    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values - truth))))

    print(f'F4-C FV filter: sequence {args.sequence_id}')
    print(
        f'  cells={result.filtered.x_grid.size}; '
        f'FP steps={result.filtered.integration_t.size - 1}; '
        f'actual max step={np.max(np.diff(result.filtered.integration_t)):.6e} day'
    )
    print(
        f'  RMSE open={rmse(open_mean):.6e}; '
        f'predictive={rmse(predictive):.6e}; '
        f'posterior={rmse(posterior):.6e}'
    )
    print(
        f'  log likelihood={result.filtered.total_log_likelihood:.6f}; '
        f'max mass error={result.filtered.maximum_abs_mass_error:.3e}; '
        f'min cell mass={result.filtered.minimum_cell_mass:.3e}'
    )
    for hour in (6, 12, 24):
        index = hour - 1
        print(
            f'  t={hour:02d}h: true={truth[index]:.6f}; '
            f'open={open_mean[index]:.6f}; pred={predictive[index]:.6f}; '
            f'post={posterior[index]:.6f}'
        )
    return result


def _print_refinement_axis(title, rows):
    print(title)
    print('  level          RMSE open    RMSE pred    RMSE post    logL        steps   actual h      mass err   min mass')
    for row in rows:
        level = row['level']
        level_text = f'{level:.6g}' if isinstance(level, float) else str(level)
        print(
            f"  {level_text:<12} "
            f"{row['rmse_open']:.6e}  {row['rmse_predictive']:.6e}  "
            f"{row['rmse_posterior']:.6e}  {row['total_log_likelihood']:.6f}  "
            f"{row['n_steps']:>5d}   {row['actual_max_step']:.6e}  "
            f"{row['maximum_abs_mass_error']:.1e}  {row['minimum_cell_mass']:.1e}"
        )


def run_filter_refinement(args):
    """Run the preregistered one-at-a-time F4-C numerical refinement study."""
    result = synthetic.run_f4_c_refinement(sequence_id=args.sequence_id)
    print(f'F4-C numerical refinement: sequence {args.sequence_id}')
    _print_refinement_axis('Spatial cells', result['spatial'])
    _print_refinement_axis('FV max step [day]', result['time'])
    _print_refinement_axis('Measurement quadrature order', result['quadrature'])
    for axis in ('spatial', 'time', 'quadrature'):
        summary = result['summary'][axis]
        ratio = summary['effect_to_assimilation_ratio']
        ratio_text = 'n/a' if not np.isfinite(ratio) else f'{ratio:.4f}'
        print(
            f"  {axis}: max posterior-RMSE refinement effect="
            f"{summary['maximum_posterior_rmse_effect']:.6e}; "
            f"effect/assimilation={ratio_text}; "
            f"max |delta logL|={summary['maximum_log_likelihood_effect']:.6e}"
        )
    return result



def run_filter_study(args):
    """Run the preregistered multi-sequence F4-C state-filter study."""
    print(
        f'F4-C population study: sequences {args.start}..'
        f'{args.start + args.sequences - 1} (n={args.sequences})'
    )
    result = synthetic.run_f4_c_population(
        n_sequences=args.sequences,
        sequence_start=args.start,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    rmse = result['rmse']
    score = result['predictive_log_score']
    print(
        f"  RMSE open={rmse['open']:.6e}; predictive={rmse['predictive']:.6e}; "
        f"posterior={rmse['posterior']:.6e}"
    )
    print(
        f"  RMSE-/RMSE0={rmse['predictive_over_open']:.6f} "
        f"[bootstrap95 {rmse['predictive_over_open_bootstrap95'][0]:.6f}, "
        f"{rmse['predictive_over_open_bootstrap95'][1]:.6f}]"
    )
    print(
        f"  RMSE+/RMSE0={rmse['posterior_over_open']:.6f} "
        f"[bootstrap95 {rmse['posterior_over_open_bootstrap95'][0]:.6f}, "
        f"{rmse['posterior_over_open_bootstrap95'][1]:.6f}]"
    )
    print(
        f"  predictive Y log score/obs: filter={score['filter_mean_per_observation']:.6f}; "
        f"open={score['open_mean_per_observation']:.6f}; "
        f"paired delta={score['paired_difference']:.6f} "
        f"[bootstrap95 {score['paired_difference_bootstrap95'][0]:.6f}, "
        f"{score['paired_difference_bootstrap95'][1]:.6f}]"
    )
    print(
        f"  first-observation score agreement max abs diff="
        f"{score['first_observation_max_abs_difference']:.3e}"
    )
    for hour in (6, 12, 24):
        item = result['coverage90_posterior'][hour]
        print(
            f"  posterior 90% coverage at {hour:02d}h: "
            f"{item['successes']}/{item['total']}={item['coverage']:.3f}; "
            f"Wilson95=[{item['wilson95'][0]:.3f}, {item['wilson95'][1]:.3f}]; "
            f"mean width={item['mean_width']:.6f}"
        )
    diagnostics = result['diagnostics']
    print(
        f"  invalid truth sequences={len(diagnostics['invalid_truth_sequences'])}; "
        f"open max mass error={diagnostics['open_maximum_abs_mass_error']:.3e}; "
        f"open min cell mass={diagnostics['open_minimum_cell_mass']:.3e}"
    )
    if args.sequences < 200:
        print('  diagnostic only: preregistered F4-C population study requires >=200 sequences')
    return result



def run_filter_calibration(args):
    """Run the post-hoc C4 numerical coverage sensitivity diagnostic."""
    result = synthetic.run_f4_c_calibration_sensitivity(
        n_sequences=args.sequences,
        sequence_start=args.start,
    )
    print(
        f'F4-C calibration sensitivity: sequences {args.start}..'
        f'{args.start + args.sequences - 1} (n={args.sequences})'
    )
    for name, variant in result['variants'].items():
        print(
            f"  {name}: cells={variant['x_steps']}; "
            f"hmax={variant['max_step']:.6e}; q={variant['quadrature_order']}"
        )
        for hour in (6, 12, 24):
            item = variant['horizons'][hour]
            print(
                f"    {hour:02d}h coverage={item['coverage']:.3f} "
                f"[Wilson95 {item['wilson95'][0]:.3f}, {item['wilson95'][1]:.3f}]; "
                f"width={item['mean_width']:.6f}; "
                f"RMSE={item['posterior_mean_rmse']:.6f}; "
                f"sqrt(E[var])={item['root_mean_posterior_variance']:.6f}; "
                f"RMSE/sd={item['rmse_over_posterior_sd']:.3f}"
            )
    print(f"  invalid truth sequences={len(result['invalid_truth_sequences'])}")
    return result

def run_filter_truth_refine(args):
    """Run the paired coarse/fine truth-EM calibration sensitivity diagnostic."""
    result = synthetic.run_f4_c_truth_refinement(
        n_sequences=args.sequences,
        sequence_start=args.start,
    )
    print(
        f'F4-C truth refinement: sequences {args.start}..'
        f'{args.start + args.sequences - 1} (n={args.sequences})'
    )
    for label in ('coarse', 'fine'):
        summary = result[label]
        print(
            f"  {label}: max mass error={summary['maximum_abs_mass_error']:.3e}; "
            f"min cell mass={summary['minimum_cell_mass']:.3e}"
        )
        for hour in (6, 12, 24):
            item = summary['horizons'][hour]
            print(
                f"    {hour:02d}h coverage={item['coverage']:.3f} "
                f"[Wilson95 {item['wilson95'][0]:.3f}, {item['wilson95'][1]:.3f}]; "
                f"width={item['mean_width']:.6f}; "
                f"RMSE={item['posterior_mean_rmse']:.6f}; "
                f"sqrt(E[var])={item['root_mean_posterior_variance']:.6f}; "
                f"RMSE/sd={item['rmse_over_posterior_sd']:.3f}"
            )
    diff = result['truth_difference']
    print(
        f"  coarse/fine truth differences: max |dX|={diff['maximum_abs_state']:.3e}; "
        f"RMS dX={diff['rms_state']:.3e}; max |dD|={diff['maximum_abs_demand']:.3e}; "
        f"RMS dD={diff['rms_demand']:.3e}; max |dY|={diff['maximum_abs_observation']:.3e}"
    )
    print(f"  invalid truth pairs={len(result['invalid_truth_pairs'])}")
    return result



def _print_mixture_representation_result(result, label):
    print(f"F4-D {label}: cells={result['x_steps']}")
    print(
        "  dictionary                 comp   max L1      max W1      "
        "max gap     cond(A)      fallback  eligible"
    )
    for name in result['dictionary_order']:
        item = result['results'][name]['summary']
        print(
            f"  {name:<26} {item['components']:>4d}   "
            f"{item['maximum_cell_mass_l1']:.3e}  "
            f"{item['maximum_wasserstein_1']:.3e}  "
            f"{item['maximum_optimality_gap']:.2e}  "
            f"{item['projection_matrix_condition_number']:.3e}  "
            f"{str(item['any_fallback']):>8}  {str(item['eligible']):>8}"
        )
        if item['failed_targets']:
            print(
                "    FAILED targets="
                + ", ".join(item['failed_targets'])
                + "; no clipping/repair was applied"
            )
            for target in item['failed_targets']:
                fit = result['results'][name]['fits'][target]
                print(
                    f"      {target}: {fit['failure_type']}: "
                    f"{fit['failure_message']}"
                )
        else:
            print(
                f"    worst L1={item['worst_l1_target']}; "
                f"worst W1={item['worst_w1_target']}; "
                f"condition alarm={item['conditioning_alarm']}"
            )
    selected = result['selected_dictionary']
    print(f"  selected dictionary={selected if selected is not None else 'NONE'}")
    if selected is not None:
        print("  selected per-target errors:")
        for target, fit in result['results'][selected]['fits'].items():
            print(
                f"    {target:<16} L1={fit['cell_mass_l1']:.3e}; "
                f"W1={fit['wasserstein_1']:.3e}; "
                f"L2={fit['density_l2']:.3e}; "
                f"dmean={fit['mean_error']:.3e}; "
                f"dvar={fit['variance_error']:.3e}; "
                f"active={fit['active_components']}"
            )
    return result


def run_mixture_static(args):
    """Run the original preregistered F4-D D1 representation study."""
    result = mixture_validation.run_f4_d_representation_study(x_steps=args.cells)
    return _print_mixture_representation_result(result, 'static mixture representation D1')


def run_mixture_static_adaptive(args):
    """Run the preregistered post-D1 adaptive D1b representation study."""
    result = mixture_validation.run_f4_d_adaptive_representation_study(x_steps=args.cells)
    return _print_mixture_representation_result(result, 'adaptive static representation D1b')

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    command = subparsers.add_parser(
        'synthetic',
        help='Generate one locked F4-A synthetic truth/observation sequence.',
    )
    command.add_argument('--sequence-id', type=int, default=0)
    command.add_argument(
        '--refine',
        action='store_true',
        help='Generate the coupled 1/1152 versus 1/2304 day EM pair.',
    )
    command.add_argument(
        '--save',
        action='store_true',
        help='Write the generated sequence/refinement JSON under files/output/bm_1/runs/estimation.',
    )
    command.set_defaults(handler=run_synthetic)

    filter_command = subparsers.add_parser(
        'filter',
        help='Run the F4-C known-parameter finite-volume state filter.',
    )
    filter_command.add_argument('--sequence-id', type=int, default=0)
    filter_command.add_argument('--cells', type=int, default=201)
    filter_command.add_argument('--max-step', type=float, default=1.0 / 192.0)
    filter_command.add_argument('--quadrature-order', type=int, default=1)
    filter_command.set_defaults(handler=run_filter)

    refinement = subparsers.add_parser(
        'filter-refine',
        help='Run F4-C spatial/time/measurement-quadrature refinement on fixed synthetic data.',
    )
    refinement.add_argument('--sequence-id', type=int, default=0)
    refinement.set_defaults(handler=run_filter_refinement)

    study = subparsers.add_parser(
        'filter-study',
        help='Run the preregistered F4-C multi-sequence state-filter evaluation.',
    )
    study.add_argument('--sequences', type=int, default=200)
    study.add_argument('--start', type=int, default=1)
    study.add_argument('--bootstrap-replicates', type=int, default=5000)
    study.set_defaults(handler=run_filter_study)

    calibration = subparsers.add_parser(
        'filter-calibration',
        help='Diagnose C4 coverage sensitivity to one-at-a-time numerical refinement.',
    )
    calibration.add_argument('--sequences', type=int, default=200)
    calibration.add_argument('--start', type=int, default=1)
    calibration.set_defaults(handler=run_filter_calibration)

    truth_refine = subparsers.add_parser(
        'filter-truth-refine',
        help='Compare F4-C calibration under the coupled coarse/fine F4-B truth grids.',
    )
    truth_refine.add_argument('--sequences', type=int, default=200)
    truth_refine.add_argument('--start', type=int, default=1)
    truth_refine.set_defaults(handler=run_filter_truth_refine)

    mixture_static = subparsers.add_parser(
        'mixture-static',
        help='Run the F4-D preregistered static mixture representation study.',
    )
    mixture_static.add_argument('--cells', type=int, default=801)
    mixture_static.set_defaults(handler=run_mixture_static)

    mixture_static_adaptive = subparsers.add_parser(
        'mixture-static-adaptive',
        help='Run the post-D1 F4-D adaptive multiresolution representation study.',
    )
    mixture_static_adaptive.add_argument('--cells', type=int, default=801)
    mixture_static_adaptive.set_defaults(handler=run_mixture_static_adaptive)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == '__main__':
    main()
