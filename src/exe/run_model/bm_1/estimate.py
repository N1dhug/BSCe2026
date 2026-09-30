"""F4 estimation entry point.

F4-B exposes the reproducible synthetic generator. F4-C adds the known-parameter
finite-volume state filter. Parameter estimation remains a later F4 milestone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.validation.bm_1 import estimation as synthetic


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
    print('  level          RMSE open    RMSE pred    RMSE post    logL        steps   actual h')
    for row in rows:
        level = row['level']
        level_text = f'{level:.6g}' if isinstance(level, float) else str(level)
        print(
            f"  {level_text:<12} "
            f"{row['rmse_open']:.6e}  {row['rmse_predictive']:.6e}  "
            f"{row['rmse_posterior']:.6e}  {row['total_log_likelihood']:.6f}  "
            f"{row['n_steps']:>5d}   {row['actual_max_step']:.6e}"
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
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == '__main__':
    main()
