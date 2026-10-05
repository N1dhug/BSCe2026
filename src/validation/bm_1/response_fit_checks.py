"""Raw validation and selected-curve diagnostics for conditional H1b fits.

No training gain chooses a model. Full histories are recomputed from each
record's own prior for the exact selected curves; the fixed panel is no
certificate for these optimizer-selected candidates.
"""
import numpy as np
from src.method.bm_1_response import curve_pair_from_logits
from src.params.bm_1_response import BM1PiecewiseResponse
from src.validation import estimation as checks
from src.validation.bm_1.response_cases import H1_V2_NUMERICS, build_h1_case
from src.validation.bm_1.response_numerics import run_numerical_unit, compare_numerical_unit


def validate_fit_result(result, trace, *, case, pieces, record_ids):
    """Check finished fit algebra and posterior geometry without new FV work."""
    if (result.get('case') != case or result.get('pieces') != pieces
            or result.get('paused') is not False or result.get('trace') != trace
            or result.get('distinct_evaluations') != len(trace) or not trace
            or len(trace) > 250 or result.get('max_evaluations') != 250
            or result.get('regularization') != .1 or result.get('backend') != 'fitted'):
        raise ValueError('Finished H1b fit metadata/trace differs from its registered design.')
    seen = set()
    for row in trace:
        z = checks.finite_vector(row['logits'], 'candidate logits')
        if np.any(np.abs(z) > 4.) or tuple(z) in seen:
            raise ValueError('Duplicate or out-of-bounds H1b candidate.')
        seen.add(tuple(z))
        f, g = curve_pair_from_logits(z, pieces)
        penalty = .1*sum(float(np.sum((c.values[1:-1]-(1-2*c.knots[1:-1]))**2)) for c in (f,g))
        scores = row['record_log_likelihoods']
        if set(scores) != set(record_ids):
            raise ValueError('Candidate training population changed.')
        total = float(sum(checks.finite_scalar(scores[k], 'record score') for k in record_ids))
        for actual, expected in ((row['penalty'], penalty), (row['log_likelihood'], total),
                                  (row['objective'], -total+penalty)):
            if abs(checks.finite_scalar(actual, 'candidate scalar')-expected) > 1e-10:
                raise ValueError('Candidate objective algebra is inconsistent.')
        error = checks.finite_scalar(row['maximum_abs_mass_error'], 'candidate mass error')
        if not 0. <= error <= 1e-10:
            raise ValueError('Candidate probability guard failed.')
    best = min(trace, key=lambda r:r['objective'])
    f,g = curve_pair_from_logits(best['logits'], pieces)
    if (result['logits'] != best['logits'] or result['f'] != f.to_record()
            or result['g'] != g.to_record() or result['objective'] != best['objective']
            or result['log_likelihood'] != best['log_likelihood']):
        raise ValueError('Selected curve differs from best evaluated candidate.')
    replay = result['replay']
    rows = replay['records']
    if [r['record_id'] for r in rows] != list(record_ids):
        raise ValueError('Selected replay population/order changed.')
    pooled = 0.
    for row in rows:
        observed = np.asarray(row['observed'], dtype=bool)
        scores = np.asarray([v if v is not None else np.nan for v in row['log_evidence']])
        if scores.shape != observed.shape or np.any(~np.isfinite(scores[observed])):
            raise ValueError('Selected replay observations/scores are invalid.')
        total = float(scores[observed].sum())
        if abs(total-row['log_likelihood']) > 1e-10 or abs(total-best['record_log_likelihoods'][row['record_id']]) > 1e-10:
            raise ValueError('Selected replay does not reproduce its candidate score.')
        edges=checks.increasing_grid(row['x_edges'],'selected edges')
        mass=checks.probability_mass(row['final_posterior_mass'],'selected posterior')
        if len(mass)+1 != len(edges) or not np.array_equal(edges,np.linspace(0,1,H1_V2_NUMERICS['base']['cells']+1)):
            raise ValueError('Selected posterior grid differs from registered numerics.')
        pooled += total
    if abs(pooled-replay['log_likelihood']) > 1e-10 or abs(pooled-best['log_likelihood']) > 1e-10:
        raise ValueError('Selected pooled score differs from record scores.')
    return f,g


def selected_curve_unit(record, f, g, linear_rows, selected_replay):
    rows=run_numerical_unit(record,f,g,numerics=H1_V2_NUMERICS,backend='fitted')
    base=rows['base']
    if (not np.allclose(base['log_evidence'],selected_replay['log_evidence'],rtol=0,atol=1e-10)
            or not np.allclose(base['posterior_mass'][-1],selected_replay['final_posterior_mass'],rtol=0,atol=1e-12)):
        raise ValueError('Fresh selected-curve replay differs from stored fit result.')
    controls=compare_numerical_unit(rows,linear_rows)
    # Posterior mass in each quarter is descriptive support, never observed truth.
    edges=base['x_edges']; widths=np.diff(edges)
    overlap=np.maximum(0,np.minimum(edges[1:,None],np.arange(1,5)[None,:]/4)-np.maximum(edges[:-1,None],np.arange(4)[None,:]/4))/widths[:,None]
    occupancy=base['posterior_mass']@overlap
    return dict(rows=rows,controls=controls,numerical_passes=all(r['passes'] for r in controls.values()),
                posterior_quarter_mass=occupancy,minimum_mean_quarter_mass=float(occupancy.mean(axis=0).min()))


def curve_recovery_description(case, f, g):
    truth=build_h1_case(case,initial_index=0)
    x=np.linspace(0,1,1001)
    return dict(f_grid_rmse=float(np.sqrt(np.mean((f(x)-truth.state_response(x))**2))),
                g_grid_rmse=float(np.sqrt(np.mean((g(x)-truth.price_response(x))**2))),
                interpretation='Descriptive error on two development records; no recovery acceptance. Xi=0 curves are unidentifiable.')
