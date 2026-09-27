"""BM1 executable: input selection, reporting and plots."""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.validation.bm_1.numerics import (REFERENCE_NUMERICS, run_f3_b_reference)

ROOT = Path(__file__).resolve().parents[4]

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

def save_json(result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_fv = {name: {'prices': np.asarray(item.prices, dtype=float).tolist(), 'total': float(item.objective.total), 'bias': float(item.objective.bias), 'variance': float(item.objective.variance), 'diagnostics': {'n_steps': int(item.diagnostics.get('n_steps', -1)), 'maximum_abs_mass_error': float(item.diagnostics.get('maximum_abs_mass_error', np.nan)), 'minimum_cell_mass': float(item.diagnostics.get('minimum_cell_mass', np.nan))}} for name,
        item in result['reference_fv'].items()}
    payload = {'note': 'F3-B fixed-candidate numerical refinement. No price vector is re-optimized during any refinement sweep.',
        'candidates': _json_safe(result['candidates']), 'spatial': result['spatial'], 'fv_time': result['fv_time'],
        'quadrature': result['quadrature'],
        'reference_numerics': {'x_steps': REFERENCE_NUMERICS.x_steps, 'objective_max_step': REFERENCE_NUMERICS.objective_max_step, 'solver_max_step': REFERENCE_NUMERICS.solver_max_step},
        'reference_fv': reference_fv, 'summary': result['summary'], 'mc': result['mc'], 'mc_time': result['mc_time']}
    path = OUTPUT_DIR / 'f3_b_numerical_reference.json'
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding='utf-8')
    return path

def save_plot(result):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))
    studies = ((axes[0], result['spatial'], 'FV spatial refinement', 'FV cells', False),
        (axes[1], result['fv_time'], 'FV integration-time refinement', 'solver max step [day]', True),
        (axes[2], result['quadrature'], 'Objective quadrature refinement (common FV solve)', 'quadrature max step [day]', True))
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
    path = GRAPH_DIR / 'bm_1_f3_b_numerical_reference.png'
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
            print(f"  level={row['level']:.9g} J={row['total']:.10e} |dJ finest|={row['abs_total_difference_to_finest']:.3e}")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mc-paths', type=int, default=0,
        help='run final FV-vs-MC comparison with this many paths (0 skips)')
    parser.add_argument('--mc-time-paths', type=int, default=0,
        help='run coupled EM time refinement with this many paths (0 skips)')
    parser.add_argument('--save', action='store_true', help='save JSON and convergence plot under files/output')
    args = parser.parse_args()
    result = run_f3_b_reference(mc_paths=args.mc_paths, mc_time_paths=args.mc_time_paths,
        progress=lambda message: print(message, flush=True))
    print()
    print('F3-B stochastic BM1 numerical reference')
    print('----------------------------------------')
    for name, prices in result['candidates'].items():
        print(f'{name:26s} ' + np.array2string(np.asarray(prices), precision=6))
    print()
    _print_sweep('FV spatial refinement', result['spatial'])
    _print_sweep('FV integration-time refinement', result['fv_time'])
    _print_sweep('Objective quadrature refinement', result['quadrature'])
    print('Provisional high-resolution FV')
    print('------------------------------')
    for name, item in result['reference_fv'].items():
        print(f'{name:26s} J={item.objective.total:.10e} bias={item.objective.bias:.3e} variance={item.objective.variance:.3e}')
    print()
    print('Last absolute-objective refinement changes')
    print('------------------------------------------')
    for key in ('decision_scale_reference_vs_f3_a', 'spatial_last_change', 'fv_time_last_change',
        'quadrature_last_change', 'spatial_last_change_over_decision_scale', 'fv_time_last_change_over_decision_scale',
        'quadrature_last_change_over_decision_scale'):
        print(f"{key:48s} {result['summary'][key]:.6e}")
    print()
    print('Decision-gap refinement: J_det - J_F3A')
    print('--------------------------------------')
    for name, row in result['summary']['decision_gap_refinement'].items():
        print(f"{name:12s} gap={row['finest_gap']:.6e} last change={row['last_gap_change']:.3e} relative={row['last_gap_change_over_finest_gap']:.3%}")
    print()
    if result['mc'] is not None:
        print('FV versus Monte Carlo')
        print('---------------------')
        for row in result['mc']['rows']:
            print(f"{row['candidate']:26s} FV={row['fv_total']:.10e} MC={row['mc_total']:.10e} SE={row['mc_standard_error']:.3e} (MC-FV)/SE={row['mc_minus_fv'] / row['mc_standard_error']:.3f}")
        print()
        print('Paired MC improvement over deterministic reference')
        print('--------------------------------------------------')
        for name, row in result['mc']['paired_improvement'].items():
            print(f"{name:26s} improvement={row['mean_reference_minus_candidate']:.6e} SE={row['standard_error']:.3e} ratio={row['ratio']:.3f}")
        print()
    if result['mc_time'] is not None:
        print('Coupled Euler-Maruyama time refinement')
        print('--------------------------------------')
        for row in result['mc_time']:
            print(f"{row['candidate']:26s} hmax={row['solver_max_step']:.9g} J={row['total']:.10e} dJ_fine={row['mean_path_cost_difference_to_finest']:.3e} SE(dJ)={row['se_path_cost_difference_to_finest']:.3e} RMS dX_T={row['rms_terminal_state_difference_to_finest']:.3e} viol={row['raw_boundary_violations']}")
        print()
    if args.save:
        json_path = save_json(result)
        graph_path = save_plot(result)
        print(f'Saved JSON to {json_path}')
        print(f'Saved graph to {graph_path}')
    else:
        print('Results were not written to files; add --save for JSON/plot output.')
OUTPUT_DIR = ROOT / 'files' / 'output' / 'bm_1' / 'f3_b'
GRAPH_DIR = OUTPUT_DIR


if __name__ == '__main__':
    main()
