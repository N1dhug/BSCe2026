"""Shared-curve cases, numerical acceptance checks and selected-fit validation."""

import numpy as np
from src.control.bm_1 import beta_initial_cell_mass
from src.method.bm_1_response import CurveTrainingRecord, curve_pair_from_logits, replay_curve_record
from src.params.bm_1_response import BM1PiecewiseResponse
from dataclasses import dataclass, field
from functools import partial
from src.validation.bm_1.estimation import BM1SyntheticCase, generate_coupled_refinement
from src.method.bm_1_observation import gaussian_log_likelihood
from src.validation import estimation as checks


H1_IDS = {'linear': (1762, 1763), 'nonlinear': (1764, 1765), 'no_response': (1766, 1767)}

H1_NUMERICS = {
    'base': dict(cells=801, max_step=1/1536, quadrature_order=3),
    'time': dict(cells=801, max_step=1/3072, quadrature_order=3),
    'spatial': dict(cells=1601, max_step=1/1536, quadrature_order=3),
    'quadrature': dict(cells=801, max_step=1/1536, quadrature_order=5),
}

H1_V2_NUMERICS = {
    'base': dict(cells=3201, max_step=1/12288, quadrature_order=3),
    'time': dict(cells=3201, max_step=1/24576, quadrature_order=3),
    'spatial': dict(cells=6401, max_step=1/12288, quadrature_order=3),
    'quadrature': dict(cells=3201, max_step=1/12288, quadrature_order=5),
}

H1_TOLERANCES = dict(prefix_log_evidence=.1, centered_prefix_log_evidence=.1,
                     posterior_l1=.02, truth_state=.002, truth_demand=.003)

@dataclass(frozen=True)
class CurveSyntheticCase(BM1SyntheticCase):
    """The existing generator with explicit shared response geometry."""
    state_response: BM1PiecewiseResponse = field(default_factory=BM1PiecewiseResponse.linear)
    price_response: BM1PiecewiseResponse = field(default_factory=BM1PiecewiseResponse.linear)

    def __post_init__(self):
        super().__post_init__()
        if not all(isinstance(c, BM1PiecewiseResponse)
                   for c in (self.state_response, self.price_response)):
            raise TypeError('H1 truth requires explicit piecewise response geometry.')

    def build_model(self):
        model, parameters = super().build_model()
        parameters[1:3] = self.state_response, self.price_response
        return model, parameters

def build_h1_case(name, *, initial_index):
    if name not in H1_IDS or initial_index not in (0, 1):
        raise ValueError('Declare an H1 case and low/high initial-prior index.')
    linear = BM1PiecewiseResponse.linear()
    f = BM1PiecewiseResponse.from_inner_values([.8, .2, -.7], pieces=4)
    g = BM1PiecewiseResponse.from_inner_values([.7, -.15, -.8], pieces=4)
    if name == 'linear':
        f = g = linear
    alpha, beta = ((2., 8.), (8., 2.))[initial_index]
    return CurveSyntheticCase(horizon=(0., 2.), C=.5, Xi=0. if name == 'no_response' else .75,
        price_edges=np.arange(9)/4.,
        price_values=[.125, .875, .375, .625, .875, .125, .625, .375],
        observation_times=np.arange(1, 49)/24., initial_alpha=alpha, initial_beta=beta,
        truth_max_step=1/2304, truth_fine_max_step=1/4608,
        state_response=f, price_response=g)

def h1_candidate_panel():
    """Fixed geometry probe, NOT an optimizer box certificate or model choice."""
    linear = BM1PiecewiseResponse.linear()
    truth = build_h1_case('nonlinear', initial_index=0)
    panel = [('linear', linear, linear), ('nonlinear', truth.state_response, truth.price_response)]
    for coordinate in range(6):
        for sign in (-1., 1.):
            logits = np.zeros(6)
            logits[coordinate] = sign
            panel.append((f'axis_{coordinate}_{int(sign):+d}', *curve_pair_from_logits(logits, 4)))
    for sign in (-1., 1.):
        logits = sign*np.array([4., -4., 4., -4., 4., -4.])
        panel.append((f'steep_{int(sign):+d}', *curve_pair_from_logits(logits, 4)))
    return tuple(panel)

def response_price_support(prices, *, pieces=4):
    """Rank of inner-knot interpolation: necessary, never sufficient for recovery.

    The known endpoint contribution is removed; repeated prices do not add rank.
    SVD null vectors describe exact shape directions invisible at these inputs.
    """
    curve = BM1PiecewiseResponse.linear(pieces=pieces)
    prices = np.asarray(prices)
    curve(prices)  # Shared real/domain validation before conversion.
    if prices.ndim != 1 or prices.size == 0:
        raise ValueError('Price support requires a nonempty vector.')
    unique = np.unique(prices.astype(float))
    matrix = np.column_stack([np.interp(unique, curve.knots, np.eye(pieces+1)[i])
                              for i in range(1, pieces)])
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    tolerance = max(matrix.shape)*np.finfo(float).eps*(singular[0] if singular.size else 0.)
    rank = int(np.count_nonzero(singular > tolerance))
    return dict(prices=unique.tolist(), basis_matrix=matrix.tolist(), rank=rank,
                inner_coordinates=pieces-1, full_rank=rank == pieces-1,
                singular_values=singular.tolist(), null_directions=vh[rank:].tolist(),
                interpretation='Input support only; not practical f/g identification.')

def h1_design(version=1):
    """JSON-safe design frozen before generating any new observations."""
    if version not in (1, 2):
        raise ValueError('H1 design version must be 1 or 2.')
    cases = {}
    for name, ids in H1_IDS.items():
        case = build_h1_case(name, initial_index=0)
        cases[name] = dict(sequence_ids=list(ids), horizon=list(case.horizon),
            price_edges=case.price_edges.tolist(), prices=case.price_values.tolist(),
            observation_times=case.observation_times.tolist(), baseline=case.baseline,
            C=case.C, k=case.k, Xi=case.Xi, sigma_x=case.sigma_x, sigma_y=case.sigma_y,
            initial_beta=[[2., 8.], [8., 2.]], independent_initializations=True,
            f=case.state_response.to_record(), g=case.price_response.to_record(),
            truth_steps=[case.truth_max_step, case.truth_fine_max_step],
            seed_roots=[case.initial_seed_root, case.brownian_seed_root, case.measurement_seed_root])
    return dict(schema=f'bm1-h1-development-v{version}', cases=cases,
        candidates=[dict(name=n, f=f.to_record(), g=g.to_record()) for n, f, g in h1_candidate_panel()],
        numerics=H1_NUMERICS if version == 1 else H1_V2_NUMERICS, tolerances=H1_TOLERANCES,
        backend='upwind' if version == 1 else 'fitted',
        selection_basis='preregistered v1' if version == 1 else 'posthoc follow-up to rejected v1; same development records',
        observation_kind='point', time_unit='day', price_side='right',
        role='DEVELOPMENT', final_holdout_used=False,
        interpretation='Fixed-candidate numerical screen; no population/recovery/forecast acceptance.')

def h1_training_records(name):
    """Two independent complete two-day streams, no midnight reset or truth state."""
    records, truths = [], []
    for index, sid in enumerate(H1_IDS[name]):
        case = build_h1_case(name, initial_index=index)
        pair = generate_coupled_refinement(case, sequence_id=sid)
        if not pair.coarse.truth.valid or not pair.fine.truth.valid:
            raise RuntimeError('H1 truth left its domain; no seed replacement or path repair.')
        model, parameters = case.build_model()
        initial = partial(beta_initial_cell_mass, alpha=case.initial_alpha, beta=case.initial_beta)
        records.append(CurveTrainingRecord(str(sid), 'TRAIN', model.B, case.horizon,
            parameters, case.sigma_x, case.sigma_y, initial, pair.coarse.observations, case.breakpoints))
        truths.append(pair)
    return tuple(records), tuple(truths)

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

def _replay_scores(row):
    """Read an explicit boolean mask and JSON scores without truthy coercion."""
    observed = np.asarray(row['observed'])
    if observed.ndim != 1 or observed.size == 0 or observed.dtype.kind != 'b':
        raise ValueError('Selected replay observed mask must be a boolean vector.')
    values = row['log_evidence']
    if np.iscomplexobj(values):
        raise ValueError('Selected replay scores must be real.')
    scores = np.asarray(values, dtype=float)
    if (scores.shape != observed.shape or not np.any(observed)
            or np.any(~np.isfinite(scores[observed]))
            or np.any(~np.isnan(scores[~observed]))):
        raise ValueError('Selected replay observations/scores are invalid.')
    return observed, scores

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
        observed, scores = _replay_scores(row)
        total = float(scores[observed].sum())
        saved_total = checks.finite_scalar(row['log_likelihood'], 'selected record likelihood')
        if abs(total-saved_total) > 1e-10 or abs(total-best['record_log_likelihoods'][row['record_id']]) > 1e-10:
            raise ValueError('Selected replay does not reproduce its candidate score.')
        error = checks.finite_scalar(row['maximum_abs_mass_error'], 'selected mass error')
        minimum = checks.finite_scalar(row['minimum_cell_mass'], 'selected minimum mass')
        if not 0. <= error <= 1e-10 or minimum < 0.:
            raise ValueError('Selected replay probability guard failed.')
        edges=checks.increasing_grid(row['x_edges'],'selected edges')
        mass=checks.probability_mass(row['final_posterior_mass'],'selected posterior')
        if len(mass)+1 != len(edges) or not np.array_equal(edges,np.linspace(0,1,H1_V2_NUMERICS['base']['cells']+1)):
            raise ValueError('Selected posterior grid differs from registered numerics.')
        pooled += total
    saved_pooled = checks.finite_scalar(replay['log_likelihood'], 'selected pooled likelihood')
    if abs(pooled-saved_pooled) > 1e-10 or abs(pooled-best['log_likelihood']) > 1e-10:
        raise ValueError('Selected pooled score differs from record scores.')
    return f,g

def validate_selected_replay(rows, record, selected_replay):
    """Bind fresh or recovered raw laws to the selected fit, with no FV work.

    A crash can leave valid atomically written units without a summary hash.
    Refinement differences alone cannot establish which fit those units test.
    Recheck the base scores and final posterior on every recovery path.
    """
    if set(rows) != set(H1_V2_NUMERICS):
        raise ValueError('Selected raw unit differs from the registered numerical levels.')
    for level, settings in H1_V2_NUMERICS.items():
        validate_filter_arrays(rows[level], record, cells=settings['cells'])
    base = rows['base']
    observed, scores = _replay_scores(selected_replay)
    mass = checks.probability_mass(selected_replay['final_posterior_mass'], 'selected posterior')
    if (selected_replay['record_id'] != record.record_id
            or not np.array_equal(observed, record.observations.observed)
            or scores.shape != np.asarray(base['log_evidence']).shape
            or not np.array_equal(selected_replay['x_edges'], base['x_edges'])
            or mass.shape != np.asarray(base['posterior_mass'][-1]).shape
            or not np.allclose(np.asarray(base['log_evidence'])[observed], scores[observed], rtol=0, atol=1e-10)
            or not np.allclose(base['posterior_mass'][-1], mass, rtol=0, atol=1e-12)):
        raise ValueError('Selected-curve replay differs from stored fit result.')

def selected_curve_unit(record, f, g, linear_rows, selected_replay):
    rows=run_numerical_unit(record,f,g,numerics=H1_V2_NUMERICS,backend='fitted')
    validate_selected_replay(rows, record, selected_replay)
    base=rows['base']
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
