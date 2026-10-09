"""H1 fixed-curve numerical screening; no optimizer or historical-data access."""

import numpy as np

from src.method.bm_1_response import replay_curve_record
from src.params.bm_1 import gaussian_log_likelihood
from src.validation import estimation as checks
from src.validation.bm_1.response_cases import (
    H1_NUMERICS, H1_TOLERANCES, h1_training_records, h1_candidate_panel)


def density_l1(left_edges, left_mass, right_edges, right_mass):
    """Exact L1 distance of two piecewise constant density histories.

    Integrate on the union of cell edges. Projecting both laws onto a coarser
    grid could hide differences within a cell and is deliberately avoided.
    """
    a = checks.increasing_grid(left_edges, 'left edges')
    b = checks.increasing_grid(right_edges, 'right edges')
    p = checks.probability_rows(left_mass, 'left mass')
    q = checks.probability_rows(right_mass, 'right mass')
    if a[0] != b[0] or a[-1] != b[-1] or p.shape[0] != q.shape[0]:
        raise ValueError('Density histories must have common support and row count.')
    if p.shape[1] != len(a)-1 or q.shape[1] != len(b)-1:
        raise ValueError('One mass is required per cell.')
    edges = np.union1d(a, b)
    middle = .5*(edges[:-1]+edges[1:])
    ia = np.searchsorted(a, middle, side='right')-1
    ib = np.searchsorted(b, middle, side='right')-1
    return np.sum(np.abs(p[:, ia]/np.diff(a)[ia] - q[:, ib]/np.diff(b)[ib])
                  *np.diff(edges), axis=1)


def filter_arrays(result):
    """Keep the raw laws/scores needed for an independent non-FV review."""
    return dict(x_edges=result.x_edges, times=result.times,
        predicted_mass=result.predicted_mass, posterior_mass=result.posterior_mass,
        log_evidence=result.log_evidence, observed=result.observed,
        maximum_abs_mass_error=result.maximum_abs_mass_error,
        minimum_cell_mass=result.minimum_cell_mass)


def validate_filter_arrays(row, record, *, cells):
    edges = checks.increasing_grid(row['x_edges'], 'x_edges')
    if not np.array_equal(edges, np.linspace(0., 1., cells+1)):
        raise ValueError('Saved H1 grid differs from its declared resolution.')
    if not np.array_equal(row['times'], record.observations.times):
        raise ValueError('Saved H1 observation times changed.')
    if not np.array_equal(row['observed'], record.observations.observed):
        raise ValueError('Saved H1 missing mask changed.')
    for key in ('predicted_mass', 'posterior_mass'):
        mass = checks.probability_rows(row[key], key)
        if mass.shape != (len(record.observations.times), cells):
            raise ValueError('Saved H1 mass history has the wrong shape.')
    scores = checks.finite_vector(row['log_evidence'], 'log evidence')
    if scores.shape != record.observations.times.shape:
        raise ValueError('Saved H1 scores have the wrong shape.')
    error = checks.finite_scalar(row['maximum_abs_mass_error'], 'mass error')
    minimum = checks.finite_scalar(row['minimum_cell_mass'], 'minimum mass')
    if error < 0. or error > 1e-10 or minimum < 0.:
        raise ValueError('Saved H1 probability guards failed.')


def run_numerical_unit(record, f, g, *, numerics=None, backend='upwind'):
    """Four full replays of one candidate/record, preserving every prefix."""
    numerics = H1_NUMERICS if numerics is None else numerics
    rows = {name: filter_arrays(replay_curve_record(record, f, g, keep_history=True, backend=backend, **settings))
            for name, settings in numerics.items()}
    for name, settings in numerics.items():
        validate_filter_arrays(rows[name], record, cells=settings['cells'])
    return rows


def compare_numerical_unit(rows, linear_rows):
    """Recompute gates from saved arrays; no trust in a cached passes flag."""
    result = {}
    base = rows['base']
    for name in rows:
        if name == 'base':
            continue
        fine = rows[name]
        error = np.cumsum(fine['log_evidence']-base['log_evidence'])
        linear_error = np.cumsum(linear_rows[name]['log_evidence']-linear_rows['base']['log_evidence'])
        centered = error-linear_error
        l1 = density_l1(base['x_edges'], base['posterior_mass'], fine['x_edges'], fine['posterior_mass'])
        maxima = dict(prefix_log_evidence=float(np.max(np.abs(error))),
                      centered_prefix_log_evidence=float(np.max(np.abs(centered))),
                      posterior_l1=float(np.max(l1)))
        result[name] = dict(maxima, passes=all(v <= H1_TOLERANCES[k] for k, v in maxima.items()),
            prefix_evidence_difference=error.tolist(),
            centered_prefix_evidence_difference=centered.tolist(), posterior_l1_by_prefix=l1.tolist())
    return result


def truth_diagnostic(pair):
    state = pair.maximum_abs_state_difference
    demand = pair.maximum_abs_demand_difference
    return dict(maximum_state_difference=state, maximum_demand_difference=demand,
        coarse_valid=pair.coarse.truth.valid, fine_valid=pair.fine.truth.valid,
        passes=bool(pair.coarse.truth.valid and pair.fine.truth.valid
                    and state <= H1_TOLERANCES['truth_state']
                    and demand <= H1_TOLERANCES['truth_demand']))


def negative_control_error(rows, record):
    # Each observed Y has its own Gaussian density; Xi=0 implies D=B exactly.
    expected = np.array([gaussian_log_likelihood(y, [float(record.baseline(t))], record.sigma_y)[0]
                         for t, y in zip(record.observations.times, record.observations.values)])
    return max(float(np.max(np.abs(row['log_evidence']-expected))) for row in rows.values())


def run_h1_job(spec):
    records, _ = h1_training_records(spec['case'])
    record = records[spec['record_index']]
    candidates = {name: (f, g) for name, f, g in h1_candidate_panel()}
    f, g = candidates[spec['candidate']]
    if spec['unit'] != f"{spec['case']}_{record.record_id}_{spec['candidate']}":
        raise ValueError('H1 task identity mismatch.')
    return run_numerical_unit(record, f, g, numerics=spec['numerics'], backend=spec['backend'])
