"""F3-B numerical refinement of frozen price plans (no re-optimization).

Default output shows decision gaps and reference costs. --verbose adds every
refinement row. --save writes local results; --output-dir explicitly selects
another destination, e.g. the retained evidence directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.validation.bm_1.numerics import REFERENCE_NUMERICS, run_f3_b_reference


ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT / 'files' / 'output' / 'bm_1' / 'runs' / 'f3_b'


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_json(result, output_dir=OUTPUT_DIR):
    """Serialize measured values without re-running any experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_fv = {
        name: {
            'prices': np.asarray(item.prices, dtype=float).tolist(),
            'total': float(item.objective.total),
            'bias': float(item.objective.bias),
            'variance': float(item.objective.variance),
            'diagnostics': {
                'n_steps': int(item.diagnostics.get('n_steps', -1)),
                'maximum_abs_mass_error': float(item.diagnostics.get('maximum_abs_mass_error', np.nan)),
                'minimum_cell_mass': float(item.diagnostics.get('minimum_cell_mass', np.nan)),
            },
        }
         for name, item in result['reference_fv'].items()
    }
    payload = {
        'note': 'F3-B fixed-candidate numerical refinement. No price vector is re-optimized during any refinement sweep.',
        'candidates': _json_safe(result['candidates']),
        'spatial': result['spatial'],
        'fv_time': result['fv_time'],
        'quadrature': result['quadrature'],
        'reference_numerics': {
            'x_steps': REFERENCE_NUMERICS.x_steps,
            'objective_max_step': REFERENCE_NUMERICS.objective_max_step,
            'solver_max_step': REFERENCE_NUMERICS.solver_max_step,
        },
        'reference_fv': reference_fv,
        'summary': result['summary'],
        'mc': result['mc'],
        'mc_time': result['mc_time'],
    }
    path = output_dir / 'f3_b_numerical_reference.json'
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding='utf-8')
    return path


def save_plot(result, output_dir=OUTPUT_DIR):
    """Plot each refinement axis at its fixed physical problem settings."""
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))
    studies = (
        (axes[0], result['spatial'], 'FV spatial refinement', 'FV cells', False),
        (axes[1], result['fv_time'], 'FV integration-time refinement', 'solver max step [day]', True),
        (
            axes[2],
            result['quadrature'],
            'Objective quadrature refinement (common FV solve)',
            'quadrature max step [day]',
            True,
        ),
    )
    for ax, rows, title, xlabel, invert in studies:
        names = list(result['candidates'])
        for name in names:
            selected = [row for row in rows if row['candidate'] == name]
            x = np.array([row['level'] for row in selected], dtype=float)
            y = np.array([row['total'] for row in selected], dtype=float)
            ax.plot(x, y, marker='o', label=name.replace('_', ' '))
        if invert:
            ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel('tracking objective J')
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc='best')
    fig.tight_layout()
    path = output_dir / 'bm_1_f3_b_numerical_reference.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def _print_sweep(title, rows):
    print(title)
    print('-' * len(title))
    for name in dict.fromkeys((row['candidate'] for row in rows)):
        print(name)
        for row in rows:
            if row['candidate'] != name:
                continue
            print(
                (
                    f"  level={row['level']:.9g} J={row['total']:.10e} |dJ finest|="
                    f"{row['abs_total_difference_to_finest']:.3e}"
                ),
            )
    print()


def report(result, verbose=False):
    """Keep numerical uncertainty visible without repeating every sweep by default."""
    print('F3-B: fixed-policy numerical reference')
    for name, item in result['reference_fv'].items():
        cost = item.objective
        print(f'{name:26s} J={cost.total:.10e} bias={cost.bias:.3e} variance={cost.variance:.3e}')
    print('Last refinement changes: absolute J and decision gap G = J_det - J_F3A')
    axes = [('space', 'spatial'), ('fv_time', 'fv_time'), ('quadrature', 'quadrature')]
    for axis, key in axes:
        row = result['summary']['decision_gap_refinement'][axis]
        absolute_change = result['summary'][key + '_last_change']
        print(
            (
                f"{axis:12s} |dJ|={absolute_change:.3e} G={row['finest_gap']:.6e} |dG|="
                f"{row['last_gap_change']:.3e} ({row['last_gap_change_over_finest_gap']:.3%})"
            ),
        )
    print('Finest grids are comparison references, not exact solutions.')
    if verbose:
        for name in ('spatial', 'fv_time', 'quadrature'):
            _print_sweep(name, result[name])
        print('Fixed candidates:', {name: prices.tolist() for name, prices in result['candidates'].items()})
    if result['mc'] is not None:
        print('Common-path MC comparison')
        for row in result['mc']['rows']:
            print(
                (
                    f"{row['candidate']:26s} FV={row['fv_total']:.8e} MC={row['mc_total']:.8e} SE="
                    f"{row['mc_standard_error']:.3e} raw violations="
                    f"{row['raw_boundary_violations']}"
                ),
            )
        print('Paired MC improvement over deterministic reference')
        for name, row in result['mc']['paired_improvement'].items():
            print(
                (
                    f"{name:26s} improvement={row['mean_reference_minus_candidate']:.6e} SE="
                    f"{row['standard_error']:.3e}"
                ),
            )
    if result['mc_time'] is not None:
        print('Coupled EM time refinement (zero finest difference is by construction)')
        for row in result['mc_time']:
            print(
                (
                    f"{row['candidate']:26s} hmax={row['solver_max_step']:.6g} dJ_fine="
                    f"{row['mean_path_cost_difference_to_finest']:.3e} SE(dJ)="
                    f"{row['se_path_cost_difference_to_finest']:.3e} RMS dX_T="
                    f"{row['rms_terminal_state_difference_to_finest']:.3e} viol="
                    f"{row['raw_boundary_violations']}"
                ),
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mc-paths', type=int, default=0, help='MC sample size; 0 skips this study.')
    parser.add_argument('--mc-time-paths', type=int, default=0, help='Coupled EM sample size; 0 skips.')
    parser.add_argument('--verbose', action='store_true', help='Show full refinement tables and progress.')
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save JSON and plot (default: ignored runs/f3_b/).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help='Destination used only with --save.',
    )
    args = parser.parse_args(argv)
    for value in (args.mc_paths, args.mc_time_paths):
        if value < 0 or value == 1:
            parser.error('MC path counts must be 0 (skip) or at least 2.')
    result = run_f3_b_reference(
        mc_paths=args.mc_paths,
        mc_time_paths=args.mc_time_paths,
        progress=print if args.verbose else None,
    )
    report(result, args.verbose)
    if args.save:
        print(f'Saved JSON to {save_json(result, args.output_dir)}')
        print(f'Saved graph to {save_plot(result, args.output_dir)}')
    return result
if __name__ == '__main__':
    main()
