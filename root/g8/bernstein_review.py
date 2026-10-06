"""Static G8 review of one supervisor-basis candidate; no filter replay.

Run on the saved FV257/FV513 development laws. The registration is written
before loading arrays or building the new candidate. Original files are read
only. This script is a scientific diagnostic, not a production filter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
from time import perf_counter


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def elevate(coefficients, target_degree, np):
    """Exact normalized-Bernstein degree elevation, up to floating point."""
    c = np.array(coefficients, dtype=float, copy=True)
    for n in range(c.size - 1, target_degree):
        i = np.arange(n + 1, dtype=float)
        higher = np.zeros(n + 2)
        higher[:-1] += c * (n + 1 - i) / (n + 2)
        higher[1:] += c * (i + 1) / (n + 2)
        c = higher
    return c


def main():
    start = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--fv257', type=Path, required=True)
    parser.add_argument('--fv513', type=Path, required=True)
    parser.add_argument('--legacy78', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    degree = 254
    inputs = {key: {'path': str(getattr(args, key).resolve()),
                    'sha256': sha256(getattr(args, key))}
              for key in ('fv257', 'fv513', 'legacy78')}
    registration = {
        'study': 'G8 supervisor Bernstein static degree review',
        'degree': degree,
        'components': degree + 1,
        'basis': 'Beta(m+1,M-m+1), all m=0,...,M; exact cell averages',
        'choice_reason': 'Same concentration 256 as the retained reference; '
                        'central variance floor below half the saved FV513 minimum.',
        'target_selection': 'For each of five nodes choose its saved FV513 '
                            'posterior with largest mu*(1-mu)/variance; '
                            'descending node order. Also the globally hardest FV257 law.',
        'projection_count_limit': 7,
        'projection_detail': 'One exact-prior check, five FV513 posteriors, '
                             'one FV257 posterior, all at M254.',
        'posterior_solver_start': 'Existing uniform default, independently per target',
        'cache_size': 32,
        'diagnostic_cpu_wall_limit_seconds': 60,
        'alarm_seconds': 55,
        'blas_threads_required': 1,
        'gates': {'cell_mass_l1': .05, 'state_mean': .005, 'state_cdf': .03,
                  'state_variance': .002, 'probability_mass': 1e-9},
        'solver_optimality': 'Existing unchanged primary/fallback gap checks',
        'scope': 'Static exposed-development representation only; no propagation, '
                 'filter, likelihood-score or price-optimization acceptance',
        'input_files': inputs,
        'script_sha256': sha256(__file__),
    }
    write_json(args.output / 'registration.json', registration)
    report = {'registration': registration, 'records': [], 'failures': [],
              'full_filter_accept': False, 'historical_accept': False}
    arrays = {}

    def deadline_handler(_signal, _frame):
        raise TimeoutError('Static G8 review exhausted its registered budget.')

    signal.signal(signal.SIGALRM, deadline_handler)
    signal.setitimer(signal.ITIMER_REAL, max(.01, 55. - (perf_counter() - start)))
    try:
        import numpy as np
        sys.path.insert(0, str(args.repo))
        from src.method.beta_mixture import PreparedBetaProjector, beta_cell_average_basis
        from src.method.fokker_planc import bm_1_fokker_planc

        source_hashes = {}
        for module in list(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                if path.is_relative_to(args.repo) and path.suffix == '.py':
                    source_hashes[str(path.relative_to(args.repo))] = sha256(path)
        report['source_sha256'] = source_hashes
        write_json(args.output / 'source_manifest.json', source_hashes)

        references = {}
        for key in ('fv257', 'fv513'):
            with np.load(getattr(args, key), allow_pickle=False) as saved:
                z = {name: saved[name].copy() for name in saved.files}
            p, edges = z['conditional_mass'], z['edges']
            x = (edges[1:] + edges[:-1]) / 2
            dx = 1. / x.size
            if (p.shape != (24, 5, x.size) or not np.all(np.isfinite(p))
                    or np.any(p < 0.) or np.max(abs(p.sum(-1) - 1.)) > 1e-9):
                raise ValueError('Invalid saved conditional laws.')
            mu = p @ x
            variance_midpoint = np.sum(p * x * x, axis=-1) - mu**2
            # Exact variance of the piecewise-constant FV reconstruction.
            variance = variance_midpoint + dx**2 / 12.
            ratio = mu * (1. - mu) / variance
            row, node = np.unravel_index(np.argmax(ratio), ratio.shape)
            references[key] = (z, x, dx, mu, variance, ratio)
            report[key + '_moments'] = {
                'mean_range': [float(mu.min()), float(mu.max())],
                'minimum_variance_midpoint': float(variance_midpoint.min()),
                'minimum_variance_piecewise_constant': float(variance.min()),
                'maximum_ratio': float(ratio[row, node]),
                'necessary_M_from_continuous_moment_bound': int(np.ceil(ratio[row, node] - 3)),
                'hardest_time': float(z['times'][row]), 'hardest_node': int(node),
                'hardest_mean': float(mu[row, node]),
                'hardest_variance': float(variance[row, node]),
                'per_node_selected_times': [float(z['times'][np.argmax(ratio[:, j])])
                                            for j in range(5)],
                'interpretation': 'Continuum design screen applied to the FV '
                                  'piecewise-constant law; not an exact '
                                  'moment constraint for the discretized mixture.',
            }
            arrays[key + '_mean'] = mu
            arrays[key + '_variance'] = variance

        if .25 / (degree + 3) >= .5 * report['fv513_moments']['minimum_variance_piecewise_constant']:
            raise RuntimeError('Declared M254 design margin is absent.')

        with np.load(args.legacy78, allow_pickle=False) as saved:
            old_matrix = saved['basis'].T / np.sqrt(saved['basis'].shape[1])
        old_s = np.linalg.svd(old_matrix, compute_uv=False)
        report['saved_M78_geometry'] = {
            'matrix_shape': list(old_matrix.shape),
            'matrix_condition_number': float(old_s[0] / old_s[-1]),
            'numerical_rank': int(np.count_nonzero(old_s > max(old_matrix.shape) * np.finfo(float).eps * old_s[0])),
            'source': inputs['legacy78'],
        }

        prior38 = np.zeros(39)
        prior38[19] = 1.
        elevated = elevate(prior38, degree, np)
        report['degree_elevation'] = {
            'initial_degree': 38, 'target_degree': degree,
            'coefficient_sum_error': abs(float(elevated.sum()) - 1.),
            'minimum_coefficient': float(elevated.min()),
            'nonzero_coefficients': int(np.count_nonzero(elevated)),
        }
        arrays['prior_coefficients'] = elevated
        report['geometry'] = {}
        for key in ('fv513', 'fv257'):
            z, x, dx, mu, variance, ratio = references[key]
            before = perf_counter()
            indices = np.arange(degree + 1, dtype=float)
            basis = beta_cell_average_basis(z['edges'], indices + 1, degree - indices + 1)
            projector = PreparedBetaProjector(basis, dx, cache_size=32)
            matrix = projector.projection_data[0]
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            raw_sampled = bm_1_fokker_planc.beta_basis(x, degree)
            sampled = bm_1_fokker_planc._normalise_basis_discretely(raw_sampled, dx)
            delta_columns = dx * np.sum(np.abs(basis - sampled), axis=1)
            prior_density = beta_cell_average_basis(z['edges'], [20.], [20.])[0]
            prior_l1 = float(dx * np.sum(np.abs(elevated @ basis - prior_density)))
            if prior_l1 > 1e-10:
                raise RuntimeError('Degree elevation failed its analytic prior identity.')
            report['geometry'][key] = {
                'seconds': perf_counter() - before,
                'matrix_shape': list(matrix.shape),
                'matrix_condition_number': projector.projection_solver.matrix_condition_number,
                'hessian_condition_number': projector.projection_data[2],
                'numerical_rank': int(np.count_nonzero(singular_values > max(matrix.shape) * np.finfo(float).eps * singular_values[0])),
                'column_mass_error': float(np.max(np.abs(basis.sum(axis=1) * dx - 1.))),
                'sampled_vs_integrated_column_L1_max': float(delta_columns.max()),
                'sampled_vs_integrated_worst_column': int(delta_columns.argmax()),
                'elevated_prior_cell_L1': prior_l1,
            }
            arrays[key + '_basis'] = basis
            arrays[key + '_singular_values'] = singular_values
            targets = []
            if key == 'fv513':
                targets.append(('prior', None, None, prior_density * dx, elevated))
                for j in reversed(range(5)):
                    row = int(np.argmax(ratio[:, j]))
                    targets.append(('posterior', row, j, z['conditional_mass'][row, j], None))
            else:
                row, j = np.unravel_index(np.argmax(ratio), ratio.shape)
                targets.append(('posterior', int(row), int(j), z['conditional_mass'][row, j], None))
            for kind, row, node, mass, warm in targets:
                before = perf_counter()
                result = projector.project(mass / dx, initial_coefficients=warm)
                elapsed = perf_counter() - before
                fitted = result['density'] * dx
                c = result['coefficients']
                diff = fitted - mass
                expected = float(mass @ x)
                fitted_mean = float(fitted @ x)
                variance_error = abs(float(fitted @ (x*x) - fitted_mean**2)
                                     - float(mass @ (x*x) - expected**2))
                gradient_gap, gradient = projector.projection_solver.optimality_gap(mass / np.sqrt(dx), c)
                gap_scale = max(1., float(np.max(np.abs(gradient))))
                record = {
                    'grid': key, 'kind': kind, 'node': node,
                    'time': None if row is None else float(z['times'][row]),
                    'seconds': elapsed,
                    'diagnostics': result['diagnostics'],
                    'mean_error': abs(fitted_mean - expected),
                    'variance_error': variance_error,
                    'cdf_error': float(np.max(np.abs(np.cumsum(diff)))),
                    'active_coefficients': int(np.count_nonzero(c)),
                    'scaled_gap': gradient_gap / gap_scale,
                    'sampled_same_coefficients_density_L1': float(dx * np.sum(np.abs(c @ sampled - result['density']))),
                }
                tol = registration['gates']
                record['passed'] = bool(
                    result['diagnostics']['cell_mass_l1_error'] <= tol['cell_mass_l1']
                    and record['mean_error'] <= tol['state_mean']
                    and variance_error <= tol['state_variance']
                    and record['cdf_error'] <= tol['state_cdf'])
                label = key + '_' + kind + '_' + str(node)
                arrays[label + '_target_mass'] = mass
                arrays[label + '_fitted_mass'] = fitted
                arrays[label + '_coefficients'] = c
                report['records'].append(record)
                write_json(args.output / 'partial.json', report)
                print(json.dumps({'case': label, 'seconds': elapsed,
                                  'l1': result['diagnostics']['cell_mass_l1_error'],
                                  'fallback': result['diagnostics']['used_fallback'],
                                  'passed': record['passed']}), flush=True)
                if not record['passed']:
                    raise RuntimeError('Registered static representation gate failed.')
    except Exception as exc:
        report['failures'].append({'type': type(exc).__name__, 'message': str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        report['static_review_passed'] = bool(len(report['records']) == 7 and not report['failures'])
        report['invocation_seconds'] = perf_counter() - start
        if arrays:
            import numpy as np
            np.savez_compressed(args.output / 'arrays.npz', **arrays)
        write_json(args.output / 'report.json', report)
        print(json.dumps({'static_review_passed': report['static_review_passed'],
                          'invocation_seconds': report['invocation_seconds'],
                          'failures': report['failures']}), flush=True)


if __name__ == '__main__':
    main()
