"""Review all uploaded G2a-v2 units; independent arithmetic, zero FV replays."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.exe.run_model.bm_1.estimate import _read_likelihood_report, _rectangular_identity
from src.validation.bm_1 import joint_pilot as pilot, joint_grid as grid

LD = np.longdouble
started = time.perf_counter()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--units-dir', type=Path)
parser.add_argument('--summary', type=Path)
parser.add_argument('--v1-report', type=Path, required=True)
parser.add_argument('--log', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f'Preserve existing review: {args.output}')
upload = args.run_dir
units_dir = args.units_dir or upload / 'units'
summary_path = args.summary or upload / 'summary.json'
manifest = json.loads((upload / 'manifest.json').read_text())
saved = json.loads(summary_path.read_text())
errors = {}
checked = 0

def eq(got, want, category, atol=1e-10):
    global checked
    a, b = np.asarray(got), np.asarray(want)
    assert a.shape == b.shape, (category, a.shape, b.shape)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), category
    difference = float(np.max(abs(a.astype(LD) - b.astype(LD)))) if a.size else 0.
    errors[category] = max(errors.get(category, 0.), difference)
    assert difference <= atol, (category, difference, atol)
    checked += 1

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

grid._check_recorded(manifest['design'], pilot.design_header(), 'registered design')
grid._check_recorded(saved['source'], manifest['source'], 'original source')
grid._check_recorded(saved['dependencies'], manifest['dependencies'], 'original dependencies')
assert saved['source']['source_commit'] == '01c7c1532ea1b686d3ba47a2fca725d082ea4ca2'
assert saved['source']['working_tree_dirty'] is False
v1_digest = digest(args.v1_report)
assert v1_digest == manifest['dependencies']['g2a_v1']['sha256']
assert saved['completed_units'] == saved['expected_units'] == 8
assert saved['conditional_runs'] == 42312 and saved['partial_unit'] is None
assert set(saved['unit_sha256']) == set(manifest['design']['unit_names'])

truth, rows, units = {}, [], []
for unit in pilot.units():
    path = units_dir / (pilot.unit_name(unit) + '.npz')
    assert path.is_file(), path
    if unit[0] not in truth:
        truth[unit[0]] = grid._g2a_truth_record(unit[0], unit[1], pilot.DAYS)
    report, sha = _read_likelihood_report(path)
    assert sha == saved['unit_sha256'][pilot.unit_name(unit)]
    for key, value in _rectangular_identity(manifest, unit).items():
        grid._check_recorded(report[key], value, 'unit ' + key)
    pilot.validate_rows(report['rows'], pilot.make_group(unit, truth[unit[0]]))
    assert report['raw_validation'] == {'passes': True}
    rows.extend(report['rows'])
    units.append(dict(name=report['unit_name'], sha256=sha, bytes=path.stat().st_size))
    print('Validated raw unit:', report['unit_name'], flush=True)

recomputed = pilot.summarize(rows)
for key, value in recomputed.items():
    grid._check_recorded(saved[key], value, 'rebuilt summary ' + key)
assert recomputed['completed'] and recomputed['passes']
assert not recomputed['releases_existing_g2b_g2c']
print('All 16 banks and all 14 saved controls rebuilt.', flush=True)

lookup, raw, mass_errors = {}, {}, []
count = 168
for row in rows:
    key = tuple(row['coordinates']), row['label']
    assert key not in lookup
    lookup[key] = row
    shape = tuple(row['parameter_grid_shape'])
    axes = [np.linspace(0, 1, n, dtype=LD) for n in shape]
    widths = [np.diff(np.r_[LD(0), (u[:-1] + u[1:]) / 2, LD(1)]) for u in axes]
    prior = np.outer(*widths).ravel()
    eq(row['initial_log_weights'], np.log(prior), 'prior')
    scores = np.asarray(row['conditional_log_evidence'], dtype=LD)
    joint = np.log(prior)[None, :] + np.cumsum(scores, axis=0, dtype=LD)
    shift = joint.max(axis=1)
    evidence = shift + np.log(np.exp(joint - shift[:, None]).sum(axis=1, dtype=LD))
    logs = joint - evidence[:, None]
    weights = np.exp(logs)
    eq(row['log_weights'], logs, 'Bayes_log_weights')
    eq(row['cumulative_log_evidence'], evidence, 'Bayes_evidence')
    eq(row['bank_log_evidence'], np.diff(np.r_[LD(0), evidence]), 'Bayes_increments')
    eq(weights.sum(axis=1), np.ones(count), 'weight_normalization')
    hypotheses = np.asarray(row['hypotheses'], dtype=LD)
    mean = (weights[:, :, None] * hypotheses).sum(axis=1, dtype=LD)
    centered = hypotheses[None, :, :] - mean[:, None, :]
    covariance = np.einsum('kh,khi,khj->kij', weights, centered, centered)
    eq(row['parameter_mean'], mean, 'physical_moments')
    eq(row['parameter_covariance'], covariance, 'physical_moments')
    eq(row['parameter_sd'], np.sqrt(np.diagonal(covariance, axis1=1, axis2=2)), 'physical_moments')
    boundary = np.any((hypotheses == hypotheses.min(axis=0)) | (hypotheses == hypotheses.max(axis=0)), axis=1)
    eq(row['boundary_union_weight'], weights[:, boundary].sum(axis=1), 'boundary_weights')
    eq(row['effective_hypotheses'], np.exp(-(weights * logs).sum(axis=1)), 'effective_hypotheses')
    assert row['map_hypotheses'] == [hypotheses[abs(l - l.max()) <= LD('1e-12')].astype(float).tolist() for l in logs]
    for axis, name in enumerate(row['coordinates']):
        values = np.unique(hypotheses[:, axis])
        marginal = np.column_stack([weights[:, hypotheses[:, axis] == v].sum(axis=1) for v in values])
        recorded = row['marginal_parameters'][name]
        eq(recorded['values'], values, 'marginal_values')
        eq(recorded['weights'], marginal, 'marginal_weights')
        eq(recorded['endpoint_weight'], marginal[:, 0] + marginal[:, -1], 'marginal_endpoints')
        intervals = [[float(values[min(np.searchsorted(np.cumsum(m), LD(p)), len(values) - 1)]) for p in (.025, .975)] for m in marginal]
        eq(recorded['equal_tail_95'], intervals, 'discrete_intervals', 0.)
    for phase in ('predicted', 'posterior'):
        mass = np.asarray(row['marginal_' + phase + '_mass'])
        assert mass.shape == (count, row['settings']['x_steps'])
        assert np.all(np.isfinite(mass)) and mass.min() >= 0
        mass_errors.append(float(abs(mass.sum(axis=1) - 1).max()))
    assert row['maximum_abs_mass_error'] <= 1e-10 and row['minimum_cell_mass'] >= 0
    raw[key] = dict(weights=weights.astype(float), evidence=evidence.astype(float))
print('Independent long-double Bayes arithmetic passed for all banks.', flush=True)

def state_prefix(left, right):
    output = {}
    for phase in ('predicted', 'posterior'):
        mass = np.asarray(right['marginal_' + phase + '_mass'])
        cdf = np.column_stack([np.zeros(count), np.cumsum(mass, axis=1)])
        transferred = np.array([np.diff(np.interp(left['x_edges'], right['x_edges'], c)) for c in cdf])
        output[phase + '_l1'] = abs(transferred - np.asarray(left['marginal_' + phase + '_mass'])).sum(axis=1)
    return output

def check_prefix(comparison, prefix, category):
    assert set(comparison['prefix_errors']) == set(prefix)
    for name, values in prefix.items():
        eq(comparison['prefix_errors'][name], values, category)
        eq(comparison['maximum_' + name], np.max(values), category)

def accepts(prefix, kind):
    limits = (dict(log_odds_error=.1, weight_tv=.01, cumulative_evidence_error=.1,
                   predicted_l1=.02, posterior_l1=.02) if kind.startswith('FV_') else
              dict(parameter_tv=.02, unit_mean_error=.01, unit_sd_error=.02,
                   cumulative_evidence_error=.1, predicted_l1=.02, posterior_l1=.02))
    return all(float(np.max(prefix[name])) <= limit for name, limit in limits.items())

for comparison in saved['comparisons']:
    if not comparison['kind'].startswith('FV_'):
        continue
    pair = tuple(comparison['coordinates'])
    left, right = [lookup[pair, comparison[k]] for k in ('left', 'right')]
    a, b = [raw[pair, comparison[k]] for k in ('left', 'right')]
    ac, bc = [np.cumsum(np.asarray(r['conditional_log_evidence'], dtype=LD), axis=0, dtype=LD) for r in (left, right)]
    window = (ac >= ac.max(axis=1)[:, None] - 5) | (bc >= bc.max(axis=1)[:, None] - 5)
    delta = bc - ac
    prefix = dict(cumulative_evidence_error=abs(b['evidence'] - a['evidence']),
                  log_odds_error=np.array([float(d[m].max() - d[m].min()) for d, m in zip(delta, window)]),
                  all_node_log_odds_error=np.ptp(delta, axis=1).astype(float),
                  weight_tv=.5 * abs(b['weights'] - a['weights']).sum(axis=1), **state_prefix(left, right))
    check_prefix(comparison, prefix, 'independent_FV_controls')
    accepted = accepts(prefix, comparison['kind'])
    if comparison['kind'] == 'FV_time':
        refined = bool(np.all(np.asarray(right['conditional_actual_max_steps']) < np.asarray(left['conditional_actual_max_steps']) * (1 - 1e-12)))
        assert comparison['actual_step_refined'] is refined
        accepted = accepted and refined
    assert comparison['passes'] is bool(accepted)
print('12 independent FV comparisons passed on all 168 prefixes.', flush=True)

def moment_vectors(nodes):
    g, w = np.polynomial.legendre.leggauss(3)
    g, w = (g + 1) / 2, w / 2
    vectors = np.zeros((3, len(nodes)))
    for i, (left, right) in enumerate(zip(nodes[:-1], nodes[1:])):
        x = left + (right - left) * g
        for degree in range(3):
            vectors[degree, i] += (right - left) * np.sum(w * (1 - g) * x**degree)
            vectors[degree, i + 1] += (right - left) * np.sum(w * g * x**degree)
    return vectors

laws = {}
for key, row in lookup.items():
    if row['fv_label'] != 'middle':
        continue
    axes = [np.asarray(v) for v in row['parameter_unit_nodes']]
    qa, ma, sa = moment_vectors(axes[0])
    qb, mb, sb = moment_vectors(axes[1])
    density = raw[key]['weights'].reshape(count, *row['parameter_grid_shape']) / np.outer(qa, qb)
    integrate = lambda x, y: np.einsum('kij,i,j->k', density, x, y)
    mass = integrate(qa, qb)
    mean = np.column_stack((integrate(ma, qb), integrate(qa, mb)))
    second = np.column_stack((integrate(sa, qb), integrate(qa, sb)))
    sd = np.sqrt(np.maximum(second - mean**2, 0))
    covariance = integrate(ma, mb) - mean[:, 0] * mean[:, 1]
    eq(mass, np.ones(count), 'independent_hat_mass')
    law = pilot.hats.tensor_hat_law(row['conditional_log_evidence'], row['parameter_grid_shape'])
    for name, value in [('density', density), ('mass', mass), ('mean', mean), ('sd', sd), ('covariance', covariance)]:
        eq(law[name], value, 'independent_hat_moments', 1e-8 if name == 'density' else 1e-10)
    laws[key] = dict(unit=axes, density=density, mean=mean, sd=sd, covariance=covariance)

def adaptive_tv(a, b):
    axes = [np.unique(np.r_[x, y]) for x, y in zip(a['unit'], b['unit'])]
    u, v = np.meshgrid(*axes, indexing='ij')
    points = np.column_stack((u.ravel(), v.ravel()))
    area = np.outer(*[np.diff(x) for x in axes])
    values, estimates = [], []
    for k in range(count):
        deltas = [RegularGridInterpolator(law['unit'], law['density'][k])(points).reshape(u.shape) for law in (a, b)]
        d = deltas[1] - deltas[0]
        corners = [d[:-1, :-1], d[:-1, 1:], d[1:, :-1], d[1:, 1:]]
        same = (np.minimum.reduce(corners) >= 0) | (np.maximum.reduce(corners) <= 0)
        total = np.sum(area[same] * np.mean(np.abs(corners), axis=0)[same])
        error = 0.
        for i, j in zip(*np.where(~same)):
            a0, a1, b0, b1 = [c[i, j] for c in corners]
            def inner(t):
                lo, hi = a0 + (a1 - a0) * t, b0 + (b1 - b0) * t
                if lo * hi >= 0:
                    return (abs(lo) + abs(hi)) / 2
                root = -lo / (hi - lo)
                return (abs(lo) * root + abs(hi) * (1 - root)) / 2
            cuts = [-x / (y - x) for x, y in ((a0, a1), (b0, b1)) if y != x and 0 < -x / (y - x) < 1]
            integral, estimate = quad(inner, 0, 1, points=cuts, epsabs=1e-12, epsrel=1e-12, limit=100)
            total += area[i, j] * integral
            error += area[i, j] * estimate
        values.append(.5 * total)
        estimates.append(.5 * error)
    return np.array(values), max(estimates)

adaptive = []
for comparison in saved['comparisons']:
    if comparison['kind'] != 'parameter_hat':
        continue
    pair = tuple(comparison['coordinates'])
    keys = [(pair, comparison[k]) for k in ('left', 'right')]
    left, right = [lookup[k] for k in keys]
    a, b = [laws[k] for k in keys]
    tv, estimated_error = adaptive_tv(a, b)
    prefix = dict(parameter_tv=tv, unit_mean_error=abs(b['mean'] - a['mean']).max(axis=1),
                  unit_sd_error=abs(b['sd'] - a['sd']).max(axis=1),
                  cumulative_evidence_error=abs(raw[keys[1]]['evidence'] - raw[keys[0]]['evidence']),
                  **state_prefix(left, right))
    for name, values in prefix.items():
        eq(comparison['prefix_errors'][name], values, 'independent_hat_controls', 1e-8 if name == 'parameter_tv' else 1e-10)
        eq(comparison['maximum_' + name], np.max(values), 'independent_hat_controls', 1e-8 if name == 'parameter_tv' else 1e-10)
    refinement = np.asarray(comparison['prefix_errors']['tv_quadrature_refinement'])
    assert comparison['passes'] is bool(accepts(prefix, 'hat') and tv.max() + 1e-8 <= .02 and refinement.max() <= 1e-8)
    adaptive.append(dict(coordinates=list(pair), maximum_parameter_tv=float(tv.max()), quadrature_error_estimate=estimated_error))
    print('Independent adaptive hat TV:', pair, float(tv.max()), flush=True)

for checkpoint in saved['checkpoints']:
    pair = tuple(checkpoint['coordinates'])
    shape = pilot.GRIDS['__'.join(pair)][0]
    key = pair, f'{shape[0]}x{shape[1]}/middle/quadrature'
    k = 24 * checkpoint['days'] - 1
    eq(checkpoint['cumulative_log_evidence'], raw[key]['evidence'][k], 'independent_checkpoints')
    eq(checkpoint['mean_predictive_y_log_score'], raw[key]['evidence'][k] / (k + 1), 'independent_checkpoints')
    for field, name in [('hat_unit_mean', 'mean'), ('hat_unit_sd', 'sd'), ('hat_unit_covariance', 'covariance')]:
        eq(checkpoint[field], laws[key][name][k], 'independent_checkpoints')

logpath = args.log
log_bytes = logpath.read_bytes()
log = log_bytes.decode('utf-16' if log_bytes[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig')
assert 'Rectangular G2a: 8/8 final units; complete=True; passes=True' in log
assert '144.75 minutter; exit=0' in log
output = dict(stage='G2a-rectangular-v2-uploaded-raw-review', review_date='2026-10-03',
    completed=True, passes=True, completed_units=8, bank_reductions=16, conditional_runs=42312,
    new_fv_replays=0, review_script_sha256=digest(Path(__file__)), original_source=manifest['source'], dependencies=manifest['dependencies'],
    files=dict(manifest=dict(sha256=digest(upload/'manifest.json')),
               final_summary=dict(uploaded_name=summary_path.name, sha256=digest(summary_path)),
               terminal_log=dict(name=logpath.name, sha256=digest(logpath)),
               g2a_v1=dict(sha256=v1_digest)), raw_units=units,
    checks=dict(production_raw_validation=True, source_dependency_design_identity=True,
                exact_unit_sha256=True, independent_long_double_bayes=True,
                independent_fv_comparisons=12, independent_adaptive_hat_comparisons=2,
                observation_prefixes_per_comparison=count, independent_day_checkpoints=6,
                rebuilt_summary_matches=True, terminal_exit=0),
    maximum_abs_saved_state_mass_error=max(mass_errors),
    maximum_conditional_mass_error=max(float(r['maximum_abs_mass_error']) for r in rows),
    minimum_conditional_cell_mass=min(float(r['minimum_cell_mass']) for r in rows),
    independent_array_checks=checked, maximum_absolute_arithmetic_errors=errors,
    adaptive_parameter_tv=adaptive, comparisons=[{k:v for k,v in c.items() if k != 'prefix_errors'} for c in saved['comparisons']], checkpoints=saved['checkpoints'],
    final_invocation_seconds=saved['invocation_seconds'], final_invocation_external_minutes=144.75,
    all_invocations_total_seconds=None, releases_existing_g2b_g2c=False,
    interpretation='Same-record numerical development acceptance only; original negative v1 and blocked square G2b/G2c are preserved.',
    limitations=['Conditional state histories are not stored; individual state-mixture numerators cannot be independently reconstructed without new FV replays.',
                 'The separately uploaded old partial checkpoint is superseded by eight complete units and is not needed for final acceptance.',
                 'No held-out recovery, robustness population, historic-data fit or forecast acceptance.'],
    review_seconds=time.perf_counter()-started)
path = args.output
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(json.dumps({k:output[k] for k in ('completed','passes','new_fv_replays','independent_array_checks','maximum_abs_saved_state_mass_error','review_seconds')}, indent=2), flush=True)
