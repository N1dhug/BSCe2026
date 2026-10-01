"""F4 estimation entry point.

F4-B exposes the reproducible synthetic generator. F4-C adds the known-parameter
finite-volume state filter. F4-D validates the recursive mixture representation. F4-E screens fixed-parameter
predictive likelihoods. F4-F adds a recursive FV rho bank and prefix identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

def _configure_blas_startup(argv):
    """Opt-in process-local BLAS limits, before importing NumPy/SciPy.

    Spawned workers inherit these settings. Importing this module as a library
    never changes the environment; notebook users can use threadpoolctl instead.
    """
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--blas-threads', type=int)
    requested, _ = probe.parse_known_args(argv)
    if requested.blas_threads is not None:
        if requested.blas_threads < 1:
            probe.error('--blas-threads must be positive')
        for variable in ('OPENBLAS_NUM_THREADS', 'OPENBLAS_DEFAULT_NUM_THREADS',
                         'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS'):
            os.environ[variable] = str(requested.blas_threads)


if __name__ == '__main__':
    _configure_blas_startup(sys.argv[1:])

# Numerical libraries are deliberately imported after the optional CLI limits.
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


def run_mixture_static_shape_aware(args):
    """Run the preregistered post-D1b shape-aware D1c representation study."""
    result = mixture_validation.run_f4_d_shape_aware_representation_study(x_steps=args.cells)
    return _print_mixture_representation_result(result, 'shape-aware static representation D1c')


def run_mixture_holdout(args):
    """Run the preregistered F4-D D2 FV-snapshot holdout."""
    result = mixture_validation.run_f4_d_snapshot_holdout(
        sequence_start=args.start,
        n_sequences=args.sequences,
        x_steps=args.cells,
        max_step=args.max_step,
        likelihood_quadrature_order=args.quadrature_order,
    )
    print(
        f"F4-D D2 snapshot holdout: dictionary={result['dictionary']}; "
        f"sequences={result['sequence_ids'][0]}..{result['sequence_ids'][-1]}; "
        f"cells={result['x_steps']}"
    )
    for label in ('prediction', 'posterior'):
        item = result[label]
        print(
            f"  {label}: max L1={item['maximum_cell_mass_l1']:.6e}; "
            f"max W1={item['maximum_wasserstein_1']:.6e}; "
            f"max CDF={item['maximum_cdf_sup']:.6e}; "
            f"max L2={item['maximum_density_l2']:.6e}"
        )
        print(
            f"    max |dmean|={item['maximum_abs_mean_error']:.6e}; "
            f"max |dvar|={item['maximum_abs_variance_error']:.6e}; "
            f"max gap={item['maximum_optimality_gap']:.3e}; "
            f"max mass error={item['maximum_mass_error']:.3e}; "
            f"fallbacks={item['fallback_count']}; "
            f"passes={item['passes_representation_gate']}"
        )
        if item['failed_snapshot_indices']:
            print(
                f"    failed snapshot indices={item['failed_snapshot_indices']}"
            )
    score = result['predictive_log_evidence_error']
    print(
        f"  predictive log-evidence error: complete={score['complete']}; "
        f"count={score['count']}/{score['expected_count']}; "
        f"mean abs/obs={score['mean_absolute_per_observation']:.6e}; "
        f"max abs/obs={score['maximum_absolute_per_observation']:.6e}; "
        f"mean signed/obs={score['mean_signed_per_observation']:.6e}; "
        f"max |daily sum diff|={score['maximum_absolute_daily_sum_difference']:.6e}"
    )
    print(
        f"  cond(A)={result['projection_matrix_condition_number']:.3e}; "
        f"cond(H)={result['hessian_condition_number']:.3e}; "
        f"invalid truth sequences={len(result['invalid_truth_sequences'])}; "
        f"passes holdout={result['passes_holdout']}"
    )
    return result



def run_mixture_filter_sanity(args):
    """Run the preregistered F4-D D3a recursive mixture filter sanity."""
    result = mixture_validation.run_f4_d_recursive_sanity(
        sequence_id=args.sequence_id,
        x_steps=args.cells,
        max_step=args.max_step,
        cfl=args.cfl,
        likelihood_quadrature_order=args.quadrature_order,
    )
    print(
        f"F4-D D3a recursive mixture filter: sequence={result['sequence_id']}; "
        f"dictionary={result['dictionary']}; cells={result['x_steps']}; "
        f"cfl={result['cfl']}; max_step={result['max_step']:.6e}"
    )
    initial = result['initial_projection_error']
    print(
        f"  initial projection: L1={initial['cell_mass_l1']:.6e}; "
        f"W1={initial['wasserstein_1']:.6e}; L2={initial['density_l2']:.6e}"
    )
    for family, title in (
        ('static_best_fit', 'static best-fit FV->mixture'),
        ('dynamic_gap', 'dynamic mixture->static best-fit'),
        ('total_fv_gap', 'total mixture->FV'),
    ):
        print(f"  {title}:")
        for label in ('prediction', 'posterior'):
            item = result[family][label]
            print(
                f"    {label}: max L1={item['maximum_cell_mass_l1']:.6e}; "
                f"max W1={item['maximum_wasserstein_1']:.6e}; "
                f"max CDF={item['maximum_cdf_sup']:.6e}; "
                f"max L2={item['maximum_density_l2']:.6e}; "
                f"max |dmean|={item['maximum_abs_mean_error']:.6e}; "
                f"max |dvar|={item['maximum_abs_variance_error']:.6e}"
            )
    score = result['log_evidence_error']
    print(
        f"  log-evidence error: mean abs/obs={score['mean_absolute_per_observation']:.6e}; "
        f"max abs/obs={score['maximum_absolute_per_observation']:.6e}; "
        f"mean signed/obs={score['mean_signed_per_observation']:.6e}; "
        f"|total diff|={score['absolute_total_log_likelihood_difference']:.6e}"
    )
    rmse = result['state_rmse']
    print(
        f"  state RMSE: FV pred={rmse['fv_predictive']:.6e}; "
        f"FV post={rmse['fv_posterior']:.6e}; "
        f"mixture pred={rmse['mixture_predictive']:.6e}; "
        f"mixture post={rmse['mixture_posterior']:.6e}"
    )
    num = result['numerical']
    print(
        f"  numerics: internal steps={num['total_internal_steps']}; "
        f"h range=[{num['minimum_internal_step']:.3e}, {num['maximum_internal_step']:.3e}]; "
        f"prediction fallbacks={num['prediction_projection_fallback_count']}; "
        f"posterior fallbacks={num['posterior_projection_fallback_count']}"
    )
    print(
        f"    min raw prediction={num['maximum_raw_prediction_negativity']:.3e}; "
        f"max raw mass error={num['maximum_abs_raw_prediction_mass_error']:.3e}; "
        f"max projected mass error={num['maximum_abs_projected_prediction_mass_error']:.3e}; "
        f"max Bayes->projected posterior L1={num['maximum_bayes_to_projected_posterior_l1']:.3e}"
    )
    print(
        f"    cond(A)={num['projection_matrix_condition_number']:.3e}; "
        f"cond(H)={num['hessian_condition_number']:.3e}"
    )
    return result



def run_mixture_filter_refinement(args):
    """Run the preregistered F4-D D3b numerical refinement study."""
    result = mixture_validation.run_f4_d_recursive_refinement(
        sequence_id=args.sequence_id,
    )
    print(f"F4-D D3b recursive mixture refinement: sequence={result['sequence_id']}")
    print(
        "  axis/level          cells  req h        actual h     steps   "
        "total post L1  evidence |err|/obs  mix post RMSE  fallbacks"
    )
    for axis in ('time', 'spatial'):
        for row in result[axis]:
            level = row['level']
            level_text = f"{level:.6g}" if isinstance(level, float) else str(level)
            fallbacks = row['prediction_fallbacks'] + row['posterior_fallbacks']
            print(
                f"  {axis}/{level_text:<11} {row['x_steps']:>5d}  "
                f"{row['requested_max_step']:.3e}  {row['actual_max_step']:.3e}  "
                f"{row['internal_steps']:>5d}   {row['total_posterior_l1']:.6e}   "
                f"{row['evidence_mean_abs']:.6e}       "
                f"{row['mixture_posterior_rmse']:.6e}   {fallbacks}"
            )
        summary = result['summary'][axis]
        p = summary['posterior_l1']
        e = summary['evidence_mean_abs']
        print(
            f"  {axis} summary: posterior-L1 max effect={p['maximum_effect']:.6e}; "
            f"effect/reference={p['effect_to_reference_ratio']:.4f}; "
            f"evidence max effect={e['maximum_effect']:.6e}; "
            f"effect/reference={e['effect_to_reference_ratio']:.4f}; "
            f"passes={summary['passes_stability_gate']}"
        )
    print(
        f"  unique configurations={len(result['unique_configurations'])}; "
        f"stability ratio gate={result['stability_ratio']:.2f}"
    )
    return result



def run_mixture_filter_convergence(args):
    """Run the preregistered F4-D D3c convergence confirmation."""
    result = mixture_validation.run_f4_d_recursive_convergence(
        sequence_id=args.sequence_id,
    )
    print(f"F4-D D3c convergence confirmation: sequence={result['sequence_id']}")
    print(
        "  axis/level          cells  req h        actual h     steps   "
        "total post L1  evidence |err|/obs  mix post RMSE  fallbacks"
    )
    for axis in ('time', 'spatial'):
        for row in result[axis]:
            level = row['level']
            level_text = f"{level:.6g}" if isinstance(level, float) else str(level)
            fallbacks = row['prediction_fallbacks'] + row['posterior_fallbacks']
            print(
                f"  {axis}/{level_text:<11} {row['x_steps']:>5d}  "
                f"{row['requested_max_step']:.3e}  {row['actual_max_step']:.3e}  "
                f"{row['internal_steps']:>5d}   {row['total_posterior_l1']:.6e}   "
                f"{row['evidence_mean_abs']:.6e}       "
                f"{row['mixture_posterior_rmse']:.6e}   {fallbacks}"
            )
    t = result['summary']['time']
    for label in ('posterior_l1', 'evidence_mean_abs'):
        item = t[label]
        print(
            f"  time {label}: E0={item['extrapolated_limit']:.6e}; "
            f"pred fine={item['predicted_fine']:.6e}; "
            f"obs fine={item['observed_fine']:.6e}; "
            f"pred rel err={item['relative_prediction_error']:.4f}; "
            f"|E0|/mid={item['limit_to_middle_ratio']:.4f}; "
            f"monotone={item['monotone_decrease']}; "
            f"step-bound={item['step_limit_binding']}; "
            f"passes={item['passes']}"
        )
    sp = result['summary']['spatial']
    for label in ('posterior_l1', 'evidence_mean_abs'):
        item = sp[label]
        changes = item['successive_relative_changes']
        print(
            f"  spatial {label}: successive rel changes="
            f"({changes[0]:.4f}, {changes[1]:.4f}); "
            f"max={item['maximum_relative_change']:.4f}; "
            f"step-bound={item['step_limit_binding']}; "
            f"passes={item['passes']}"
        )
    print(
        f"  time passes={t['passes']}; spatial passes={sp['passes']}; "
        f"overall={result['summary']['passes_convergence']}; "
        f"unique configurations={len(result['unique_configurations'])}"
    )
    setting = result['summary']['population_setting']
    if setting is not None:
        print(
            f"  locked population setting: cells={setting['x_steps']}; "
            f"hmax={setting['max_step']:.6e}; cfl={setting['cfl']}; "
            f"q={setting['likelihood_quadrature_order']}"
        )
    return result



def run_mixture_filter_population(args):
    """Run the final preregistered F4-D D3d paired population study."""
    print(
        f"F4-D D3d population: sequences {args.start}.."
        f"{args.start + args.sequences - 1}; cells={args.cells}; "
        f"hmax={args.max_step:.6e}; workers={args.workers}"
    )
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; choose a new result path.")
    started = perf_counter()

    def progress(completed, total, row):
        if not args.quiet and (completed == 1 or completed % max(1, total // 20) == 0):
            elapsed = perf_counter() - started
            remaining = elapsed * (total - completed) / completed
            print(f"  completed {completed}/{total}; sequence={row['sequence_id']}; "
                  f"status={row['status']}; elapsed={elapsed:.1f}s; ETA~{remaining:.1f}s",
                  flush=True)

    result = mixture_validation.run_f4_d_recursive_population(
        n_sequences=args.sequences,
        sequence_start=args.start,
        x_steps=args.cells,
        max_step=args.max_step,
        cfl=args.cfl,
        likelihood_quadrature_order=args.quadrature_order,
        bootstrap_replicates=args.bootstrap_replicates,
        workers=args.workers,
        progress=progress,
    )
    if args.output is not None:
        try:
            revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                      text=True, capture_output=True, check=True).stdout.strip()
            dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                                        text=True, capture_output=True, check=True).stdout)
        except (OSError, subprocess.SubprocessError):
            revision, dirty = None, None
        payload = {**result, 'source_commit': revision, 'working_tree_dirty': dirty,
                   'elapsed_seconds': perf_counter() - started,
                   'blas_threads_requested': args.blas_threads}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')
        print(f'Saved {args.output}')
    if result.get('successful_sequences', 0) == 0:
        print(
            f"  no successful sequences; invalid={result['invalid_truth_sequences']}; "
            f"failed={result['failed_sequences']}"
        )
        return result
    print(
        f"  successful={result['successful_sequences']}/{args.sequences}; "
        f"invalid truth={len(result['invalid_truth_sequences'])}; "
        f"failed={len(result['failed_sequences'])}"
    )
    for label in ('predictive', 'posterior'):
        item = result['rmse'][label]
        ci = item['ratio_bootstrap95']
        dci = item['difference_bootstrap95']
        print(
            f"  {label} RMSE: FV={item['fv']:.6e}; mixture={item['mixture']:.6e}; "
            f"ratio={item['ratio']:.6f} [bootstrap95 {ci[0]:.6f}, {ci[1]:.6f}]; "
            f"diff={item['difference']:.6e} "
            f"[bootstrap95 {dci[0]:.6e}, {dci[1]:.6e}]; "
            f"equivalent={item['passes_equivalence']}"
        )
    score = result['log_evidence_difference']
    print(
        f"  log-evidence: mean abs/obs={score['mean_absolute_per_observation']:.6e} "
        f"[bootstrap95 {score['mean_absolute_bootstrap95'][0]:.6e}, "
        f"{score['mean_absolute_bootstrap95'][1]:.6e}]; "
        f"max abs/obs={score['maximum_absolute_per_observation']:.6e}"
    )
    print(
        f"    mean signed/obs={score['mean_signed_per_observation']:.6e} "
        f"[bootstrap95 {score['mean_signed_bootstrap95'][0]:.6e}, "
        f"{score['mean_signed_bootstrap95'][1]:.6e}]; "
        f"daily abs mean={score['daily_absolute_mean']:.6e}; "
        f"q95={score['daily_absolute_q95']:.6e}; "
        f"max={score['daily_absolute_maximum']:.6e}; passes={score['passes']}"
    )
    for name, item in result['distribution_gap'].items():
        print(
            f"  {name}: mean={item['mean']:.6e}; q95={item['q95']:.6e}; "
            f"max={item['maximum']:.6e}; passes={item['passes']}"
        )
    num = result['numerical']
    print(
        f"  numerics: internal steps={num['total_internal_steps']}; "
        f"prediction fallbacks={num['prediction_fallbacks']}; "
        f"posterior fallbacks={num['posterior_fallbacks']}; "
        f"max mass error={num['maximum_mass_error']:.3e}; passes={num['passes']}"
    )
    if result['failed_sequences']:
        print(f"  failed sequences={result['failed_sequences']}")
    print(f"  passes population={result['passes_population']}")
    return result



def _write_likelihood_report(args, result, started):
    """Write estimation evidence with common source metadata, never overwrite a run."""
    if args.output is None:
        return
    try:
        revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True,
            capture_output=True, check=True,
        ).stdout)
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    payload = {
        **result,
        'source_commit': revision,
        'working_tree_dirty': dirty,
        'elapsed_seconds': perf_counter() - started,
        'blas_threads_requested': args.blas_threads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == '.gz':
        import gzip
        handle = gzip.open(args.output, 'xt', encoding='utf-8')
    else:
        handle = args.output.open('x', encoding='utf-8')
    with handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f"Saved {args.output}")


def run_likelihood_refinement(args):
    """Run preregistered F4-E E1 rho-likelihood numerical refinement."""
    if args.output is not None and args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; choose a new result path."
        )
    print(
        f"F4-E E1 likelihood refinement: sequence={args.sequence_id}; "
        f"rho nodes={len(synthetic.F4_E_RHO_GRID)}; workers={args.workers}"
    )
    started = perf_counter()

    def progress(completed, total, key, curve):
        if not args.quiet:
            elapsed = perf_counter() - started
            remaining = elapsed * (total - completed) / completed
            cells, hmax, order = key
            print(
                f"  completed {completed}/{total}; cells={cells}; "
                f"hmax={hmax:.6e}; q={order}; "
                f"MLE rho={curve['maximizer_rho']}; "
                f"elapsed={elapsed:.1f}s; ETA~{remaining:.1f}s",
                flush=True,
            )

    result = synthetic.run_f4_e_likelihood_refinement(
        sequence_id=args.sequence_id,
        workers=args.workers,
        progress=progress,
    )
    for axis in ('time', 'spatial', 'quadrature'):
        print(f"  {axis}:")
        for row in result[axis]:
            if axis == 'time':
                level = f"h={row['requested_max_step']:.6e}"
            elif axis == 'spatial':
                level = f"N={row['x_steps']}"
            else:
                level = f"q={row['likelihood_quadrature_order']}"
            print(
                f"    {level}: MLE rho={row['maximizer_rho']}; "
                f"actual h={row['actual_max_step']:.6e}; "
                f"steps={row['n_steps']}; "
                f"mass err={row['maximum_abs_mass_error']:.3e}"
            )
        item = result['summary'][axis]
        print(
            f"    fine/middle: same unique MLE={item['unique_same_maximizer']}; "
            f"max |delta centered logL|={item['maximum_centered_log_likelihood_change']:.6e}; "
            f"rho window={item['comparison_rho_range']}; passes={item['passes']}"
        )
    print(
        f"  unique configurations={len(result['unique_configurations'])}; "
        f"registered={result['matches_registered_design']}; "
        f"passes E1={result['passes_numerical_refinement']}"
    )

    _write_likelihood_report(args, result, started)
    return result


def run_parameter_screening(args):
    """Run locked E2 after rechecking the supplied registered E1 artifact."""
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}; choose a new result path.")
    raw = args.e1_report.read_bytes()
    e1 = json.loads(raw.decode('utf-8'))
    settings = synthetic.f4_e_screen_numerics_from_refinement(e1)
    print(
        f"F4-E E2 parameter screening: sequences=1212..1261; "
        f"5 parameters x 41 nodes; cells={settings['x_steps']}; "
        f"hmax={settings['max_step']:.6e}; q={settings['likelihood_quadrature_order']}; "
        f"workers={args.workers}"
    )
    started = perf_counter()

    def progress(completed, total, sequence_id, row):
        if not args.quiet:
            elapsed = perf_counter() - started
            remaining = elapsed * (total - completed) / completed
            mles = ', '.join(f"{name}={s['grid_mle']}" for name, s in row['slices'].items())
            print(f"  completed {completed}/{total}; sequence={sequence_id}; {mles}; "
                  f"elapsed={elapsed:.1f}s; ETA~{remaining:.1f}s", flush=True)

    result = synthetic.run_f4_e_parameter_screening(
        e1_refinement=e1, workers=args.workers, progress=progress,
    )
    result['e1_report_path'] = str(args.e1_report)
    result['e1_report_sha256'] = hashlib.sha256(raw).hexdigest()
    for name, item in result['summary'].items():
        print(
            f"  {name}: median MLE={item['unique_mle_median']}; "
            f"q05/q95={item['unique_mle_q05_q95']}; "
            f"boundary={item['boundary_mle_frequency']:.1%}; "
            f"ties={item['nonunique_mle_count']}; "
            f"median truth loss={item['median_log_likelihood_loss_at_truth']:.6f}; "
            f"median LR span={item['median_lr_span_width']:.6f}; "
            f"aggregate MLE={item['aggregate_slice']['grid_mle']}"
        )
    print(f"  mass err={result['maximum_abs_mass_error']:.3e}; "
          f"min mass={result['minimum_cell_mass']:.3e}; "
          f"registered={result['matches_registered_design']}")
    print('  LR sets are shape diagnostics; nuisance parameters held at truth. '
          'E3/E4 and parameter-specific refinement still required before F4-F.')
    _write_likelihood_report(args, result, started)
    return result


def _pair_surface_figure_paths(prefix):
    if prefix is None:
        return ()
    return tuple(prefix.parent / f"{prefix.name}_{'_'.join(pair)}.png"
                 for pair in synthetic.F4_E_PAIRS)


def _save_pair_surface_figures(result, paths):
    """Plot exact E3 matrices; color clipping changes only the display."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    labels = {'rho': r'$\rho$', 'Xi': r'$\Xi$', 'k': r'$k$',
              'sigma_x': r'$\sigma_X$', 'sigma_y': r'$\sigma_Y$'}
    for pair, path in zip(synthetic.F4_E_PAIRS, paths):
        items = result['pair_results']['__'.join(pair)]
        surfaces = (items['aggregate_surface'], *items['sequence_surfaces'])
        rows = (len(surfaces) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows + 1),
                                 squeeze=False, constrained_layout=True)
        try:
            for index, (axis, surface) in enumerate(zip(axes.flat, surfaces)):
                a, b = np.asarray(surface['first_values']), np.asarray(surface['second_values'])
                z = np.asarray(surface['centered_log_likelihood'])
                mesh = axis.pcolormesh(a, b, z.T, shading='auto', cmap='viridis',
                                       norm=Normalize(vmin=-10, vmax=0, clip=True))
                levels = [v for v in (-10, -5, -2, -.5) if np.min(z) < v < np.max(z)]
                if levels:
                    contours = axis.contour(a, b, z.T, levels=levels, colors='white', linewidths=.65)
                    axis.clabel(contours, fontsize=7, fmt='%g')
                ta, tb = surface['truth_values']
                axis.plot(ta, tb, '*', color='red', markersize=8, clip_on=False)
                maxima = np.asarray(surface['maximizer_values'])
                axis.plot(maxima[:, 0], maxima[:, 1], 'x', color='orange', markersize=6,
                          markeredgewidth=1.4, clip_on=False)
                for name, set_scale, set_ticks, values in (
                    (pair[0], axis.set_xscale, axis.set_xticks, a),
                    (pair[1], axis.set_yscale, axis.set_yticks, b),
                ):
                    if name.startswith('sigma_'):
                        set_scale('log', base=2)
                        ticks = values[::max(1, (len(values) - 1) // 4)]
                        set_ticks(ticks, labels=[f'{v:.3g}' for v in ticks])
                axis.set_xlim(a[0], a[-1])
                axis.set_ylim(b[0], b[-1])
                axis.set_xlabel(labels[pair[0]])
                axis.set_ylabel(labels[pair[1]])
                title = f"Aggregate ({result['n_sequences']} records)" if index == 0 else \
                        f"ID {surface['sequence_id']}"
                axis.set_title(title, fontsize=10)
                axis.tick_params(labelsize=8)
            for axis in list(axes.flat)[len(surfaces):]:
                axis.set_visible(False)
            fig.colorbar(mesh, ax=list(axes.flat)[:len(surfaces)], shrink=.8,
                         label='Centered log-likelihood (display floor -10)')
            fig.suptitle(f"F4-E E3: {pair[0]} / {pair[1]}\n"
                         "Other parameters fixed at truth; star = truth; x = grid maximum\n"
                         "Contours are likelihood levels, not confidence regions", fontsize=12)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                fig.savefig(handle, format='png', dpi=150)
        finally:
            plt.close(fig)
        print(f"Saved {path}")


def run_pair_surfaces(args):
    """Replay registered E3 from saved E2 observations, with no new draws."""
    figures = _pair_surface_figure_paths(args.figure_prefix)
    targets = ((args.output,) if args.output is not None else ()) + figures
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError('JSON and figure paths must be distinct.')
    for path in targets:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new result path.")
    raw = args.e2_report.read_bytes()
    e2 = json.loads(raw.decode('utf-8'))
    settings = synthetic.f4_e_screening_inputs(e2)
    print(f"F4-E E3 pair surfaces: sequences=1212..1221; 3 pairs x 21x21 nodes; "
          f"cells={settings['x_steps']}; hmax={settings['max_step']:.6e}; "
          f"q={settings['likelihood_quadrature_order']}; workers={args.workers}")
    started = perf_counter()

    def progress(completed, total, key, surface):
        if not args.quiet:
            elapsed = perf_counter() - started
            remaining = elapsed * (total - completed) / completed
            sequence_id, pair = key
            print(f"  completed {completed}/{total}; sequence={sequence_id}; pair={pair}; "
                  f"MLE={surface['grid_mle']}; elapsed={elapsed:.1f}s; ETA~{remaining:.1f}s",
                  flush=True)

    result = synthetic.run_f4_e_pair_surfaces(e2_screening=e2, workers=args.workers, progress=progress)
    result['e2_report_path'] = str(args.e2_report)
    result['e2_report_sha256'] = hashlib.sha256(raw).hexdigest()
    result['figure_paths'] = tuple(str(path) for path in figures)
    for pair in synthetic.F4_E_PAIRS:
        item = result['pair_results']['__'.join(pair)]
        surface = item['aggregate_surface']
        print(f"  {pair}: aggregate MLE={surface['grid_mle']}; "
              f"aggregate boundary={surface['boundary_dimensions']}; "
              f"sequence boundary frequency={item['boundary_mle_frequency']}; "
              f"ties={item['nonunique_mle_count']}")
    print(f"  mass err={result['maximum_abs_mass_error']:.3e}; "
          f"min mass={result['minimum_cell_mass']:.3e}; "
          f"max E2 truth-cut difference={result['maximum_e2_truth_cut_difference']:.3e}; "
          f"registered={result['matches_registered_design']}")
    print('  Aggregate uses the same first ten E2 records. Surfaces and grid profiles are '
          'diagnostics; parameter selection remains E4.')
    if figures:
        _save_pair_surface_figures(result, figures)
    _write_likelihood_report(args, result, started)
    return result


def run_bank_identity(args):
    if args.output is not None and args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}.')
    started = perf_counter()
    print('F4-F F0 FV rho bank: sequence=1262; hypotheses=41; cells=401; '
          f'hmax={1/768:.6e}; q=3; workers={args.workers}')

    def progress(row):
        if not args.quiet:
            print(f"  {row['phase']}: completed {row['completed']}/{row['total']}; "
                  f"elapsed={perf_counter() - started:.1f}s", flush=True)

    result = synthetic.run_f4_f_bank_identity(workers=args.workers, progress=progress)
    for name, error in result['maximum_errors'].items():
        print(f'  max |delta {name}|={error:.6e}')
    print(f"  mass err={result['maximum_abs_mass_error']:.3e}; "
          f"min mass={result['minimum_cell_mass']:.3e}; "
          f"actual h={result['actual_max_step']:.6e}")
    print(f"  final rho mean={result['rho_posterior_mean'][-1]:.6g}; "
          f"MAP rho={result['map_rho'][-1]}; "
          f"effective hypotheses={result['effective_hypotheses'][-1]:.6g}; "
          f"endpoint weight={result['endpoint_weight'][-1]:.3%}")
    print(f"  registered={result['registered_design']}; passes F0={result['passes_f0']}")
    _write_likelihood_report(args, result, started)
    if not result['passes_f0']:
        raise RuntimeError('F0 bank identity/probability gates failed; report retained.')


def _read_bank_dependency(path, validator):
    import gzip
    raw = path.read_bytes()
    decoded = gzip.decompress(raw) if path.suffix == '.gz' else raw
    report = json.loads(decoded.decode('utf-8'))
    validator(report, require_source=True)
    return report, hashlib.sha256(raw).hexdigest()


def run_bank_study(args):
    """Staged remaining F4-F studies; source-linked reports and bounded progress."""
    if args.output is not None and args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}.')
    started = perf_counter()
    dependencies = {}
    refinement = mixture_check = None
    if args.study != 'refine':
        refinement, digest = _read_bank_dependency(args.refinement_report, synthetic.f4_f_validated_refinement)
        dependencies['refinement'] = dict(path=str(args.refinement_report), sha256=digest,
                                          source_commit=refinement['source_commit'])
    if args.study == 'population':
        mixture_check, digest = _read_bank_dependency(args.mixture_report, synthetic.f4_f_validated_mixture_check)
        dependencies['mixture'] = dict(path=str(args.mixture_report), sha256=digest,
                                       source_commit=mixture_check['source_commit'])
    label = ('rho-mixture, sigma_x and sigma_y known' if args.study == 'mixture-check'
             else 'rho and sigma_x separately, sigma_y known')
    print(f'F4-F bank-{args.study}: workers={args.workers}; {label}', flush=True)
    last_printed = started

    def progress(row):
        nonlocal last_printed
        now = perf_counter()
        if not args.quiet and (row['completed'] in (1, row['total']) or now-last_printed >= 10.):
            last_printed = now
            elapsed = now-started
            # Startup and unequal candidate costs make this an approximate ETA.
            eta = (f"~{elapsed * (row['total']-row['completed']) / row['completed']:.1f}s"
                   if row['completed'] >= args.workers else 'startup')
            print(f"  candidates {row['completed']}/{row['total']}; ID{row['sequence_id']}; "
                  f"{row['coordinate']}; {row['label']}; elapsed={elapsed:.1f}s; ETA={eta}", flush=True)

    if args.study == 'refine':
        result = synthetic.run_f4_f_bank_refinement(workers=args.workers, progress=progress)
    elif args.study == 'sensitivity':
        result = synthetic.run_f4_f_bank_sensitivity(refinement, workers=args.workers, progress=progress)
    elif args.study == 'mixture-check':
        result = synthetic.run_f4_f_bank_mixture_check(refinement, workers=args.workers, progress=progress)
    else:
        result = synthetic.run_f4_f_bank_population(refinement, mixture_check, workers=args.workers, progress=progress)
    result['dependencies'] = dependencies
    print(f"  stage={result['stage']}; banks={len(result['rows'])}; registered={result['registered_design']}; "
          f"passes={result['passes']}")
    for row in result.get('comparisons', []):
        print(f"  {row['coordinate']} ID{row['sequence_id']} {row.get('axis', row.get('comparison', 'paired'))}: "
              f"log-odds={row['maximum_log_odds_error']:.4g}; TV={row['maximum_weight_tv']:.4g}; "
              f"evidence={row['maximum_cumulative_evidence_error']:.4g}; passes={row['passes']}; "
              f"selected={row.get('selected', True)}")
    _write_likelihood_report(args, result, started)
    if not result['passes']:
        raise RuntimeError('F4-F study gates failed; report retained for review before adapting the next experiment.')


def run_preparation(args):
    """Reproduce independent F4-0 quadrature through the common entry point."""
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}.")
    report = synthetic.run_f4_preparation_checks()
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(text + '\n')
    print(text)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    preparation = subparsers.add_parser(
        'preparation', help='Independent continuous F4-0 quadrature and Bayes checks.',
    )
    preparation.add_argument('--output', type=Path, help='Optional new JSON file.')
    preparation.set_defaults(handler=run_preparation)
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

    likelihood_refinement = subparsers.add_parser(
        'likelihood-refine',
        help='Run preregistered F4-E E1 rho likelihood numerical refinement.',
    )
    likelihood_refinement.add_argument(
        '--sequence-id', type=int, default=synthetic.F4_E_NUMERICAL_SEQUENCE_ID,
    )
    likelihood_refinement.add_argument(
        '--workers', type=int, default=1,
        help='Independent numerical-configuration workers; runtime only.',
    )
    likelihood_refinement.add_argument(
        '--output', type=Path,
        help='Optional new JSON file with curves, gates, settings and source revision.',
    )
    likelihood_refinement.add_argument(
        '--quiet', action='store_true',
        help='Suppress per-configuration progress, not the final summary.',
    )
    likelihood_refinement.set_defaults(handler=run_likelihood_refinement)

    parameter_screening = subparsers.add_parser(
        'parameter-screen', help='Run locked F4-E E2 one-parameter screening on IDs 1212..1261.',
    )
    parameter_screening.add_argument(
        '--e1-report', type=Path, required=True,
        help='Passing registered E1 JSON; curves and numerical settings are rechecked.',
    )
    parameter_screening.add_argument('--workers', type=int, default=1,
                                    help='Independent sequence workers; runtime only.')
    parameter_screening.add_argument('--output', type=Path,
                                    help='Optional new JSON with every curve, sequence and source revision.')
    parameter_screening.add_argument('--quiet', action='store_true',
                                    help='Suppress per-sequence progress, not the final summary.')
    parameter_screening.set_defaults(handler=run_parameter_screening)

    pair_surfaces = subparsers.add_parser(
        'pair-surfaces', help='Run registered F4-E E3 pair surfaces on the first ten E2 records.',
    )
    pair_surfaces.add_argument('--e2-report', type=Path, required=True,
                               help='Complete registered E2 JSON; saved observations are reused.')
    pair_surfaces.add_argument('--workers', type=int, default=1,
                               help='Independent (sequence,pair) workers; runtime only.')
    pair_surfaces.add_argument('--output', type=Path,
                               help='Optional new JSON with raw surfaces, ridges and evidence.')
    pair_surfaces.add_argument('--figure-prefix', type=Path,
                               help='Optional prefix for three new PNGs with aggregate and per-record surfaces.')
    pair_surfaces.add_argument('--quiet', action='store_true',
                               help='Suppress per-surface progress, not the final summary.')
    pair_surfaces.set_defaults(handler=run_pair_surfaces)

    bank_identity = subparsers.add_parser(
        'bank-identity', help='Run registered F4-F F0 FV rho bank identities on new ID1262.',
    )
    bank_identity.add_argument('--workers', type=int, default=7,
                               help='Persistent process workers over independent hypotheses.')
    bank_identity.add_argument('--output', type=Path,
                               help='Optional new JSON with prefix identities and final conditional states.')
    bank_identity.add_argument('--quiet', action='store_true',
                               help='Suppress progress, not the final summary.')
    bank_identity.set_defaults(handler=run_bank_identity)

    for command, study, description in (
            ('bank-refine', 'refine', 'F1: independent rho/sigma_x FV bank numerical refinement.'),
            ('bank-sensitivity', 'sensitivity', 'F2: parameter grid, support and prior on new data.'),
            ('bank-mixture-check', 'mixture-check', 'F3: paired recursive mixture-bank pilot and time refinement.'),
            ('bank-population', 'population', 'F4: paired one-parameter FV/mixture population.')):
        bank = subparsers.add_parser(command, help=description)
        bank.add_argument('--workers', type=int, default=7,
                          help='Persistent spawn-process workers over independent conditional replays.')
        bank.add_argument('--output', type=Path, required=True,
                          help='New evidence file; use .json.gz for large losslessly compressed reports.')
        bank.add_argument('--quiet', action='store_true')
        if study != 'refine':
            bank.add_argument('--refinement-report', type=Path, required=True)
        if study == 'population':
            bank.add_argument('--mixture-report', type=Path, required=True)
        bank.set_defaults(handler=run_bank_study, study=study)

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

    mixture_static_shape = subparsers.add_parser(
        'mixture-static-shape',
        help='Run the post-D1b F4-D shape-aware multiresolution representation study.',
    )
    mixture_static_shape.add_argument('--cells', type=int, default=801)
    mixture_static_shape.set_defaults(handler=run_mixture_static_shape_aware)

    mixture_holdout = subparsers.add_parser(
        'mixture-holdout',
        help='Run the preregistered F4-D D2 static holdout on FV filter snapshots.',
    )
    mixture_holdout.add_argument('--start', type=int, default=1001)
    mixture_holdout.add_argument('--sequences', type=int, default=10)
    mixture_holdout.add_argument('--cells', type=int, default=801)
    mixture_holdout.add_argument('--max-step', type=float, default=1.0 / 192.0)
    mixture_holdout.add_argument('--quadrature-order', type=int, default=1)
    mixture_holdout.set_defaults(handler=run_mixture_holdout)

    mixture_filter = subparsers.add_parser(
        'mixture-filter',
        help='Run the preregistered F4-D D3a recursive mixture-filter sanity.',
    )
    mixture_filter.add_argument('--sequence-id', type=int, default=0)
    mixture_filter.add_argument('--cells', type=int, default=101)
    mixture_filter.add_argument('--max-step', type=float, default=1.0 / 384.0)
    mixture_filter.add_argument('--cfl', type=float, default=0.4)
    mixture_filter.add_argument('--quadrature-order', type=int, default=1)
    mixture_filter.set_defaults(handler=run_mixture_filter_sanity)

    mixture_filter_refine = subparsers.add_parser(
        'mixture-filter-refine',
        help='Run the preregistered F4-D D3b time/spatial refinement study.',
    )
    mixture_filter_refine.add_argument('--sequence-id', type=int, default=0)
    mixture_filter_refine.set_defaults(handler=run_mixture_filter_refinement)

    mixture_filter_converge = subparsers.add_parser(
        'mixture-filter-converge',
        help='Run the preregistered F4-D D3c convergence confirmation.',
    )
    mixture_filter_converge.add_argument('--sequence-id', type=int, default=0)
    mixture_filter_converge.set_defaults(handler=run_mixture_filter_convergence)

    mixture_filter_population = subparsers.add_parser(
        'mixture-filter-population',
        help='Run the final preregistered F4-D D3d paired population study.',
    )
    mixture_filter_population.add_argument('--start', type=int, default=1011)
    mixture_filter_population.add_argument('--sequences', type=int, default=200)
    mixture_filter_population.add_argument('--cells', type=int, default=151)
    mixture_filter_population.add_argument('--max-step', type=float, default=1.0 / 3072.0)
    mixture_filter_population.add_argument('--cfl', type=float, default=0.4)
    mixture_filter_population.add_argument('--quadrature-order', type=int, default=1)
    mixture_filter_population.add_argument('--bootstrap-replicates', type=int, default=5000)
    mixture_filter_population.add_argument(
        '--workers', type=int, default=1,
        help='Independent process workers; changes runtime only, not the scientific design.',
    )
    mixture_filter_population.add_argument('--output', type=Path,
        help='Optional new JSON file, including every sequence and source revision.')
    mixture_filter_population.add_argument('--quiet', action='store_true',
        help='Suppress intermediate progress rows, not the final summary.')
    mixture_filter_population.set_defaults(handler=run_mixture_filter_population)
    for command_parser in subparsers.choices.values():
        command_parser.add_argument('--blas-threads', type=int,
            help='Opt-in BLAS/OpenMP thread limit per process, applied at CLI startup.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == '__main__':
    main()
