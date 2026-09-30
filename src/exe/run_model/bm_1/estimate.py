"""F4 estimation entry point.

F4-B currently exposes only the reproducible synthetic generator.  Recursive
state filtering and parameter estimation are added in later F4 milestones.
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
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == '__main__':
    main()
