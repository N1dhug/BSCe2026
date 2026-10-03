"""Posthoc G2a grid diagnostics on old records; never a production acceptance.

Rectangular tensor hats preserve the original continuous prior. Their node
weights are tensor trapezoidal quadrature in normalized prior coordinates.
The diagnostic compares axes separately, using only already recorded scores.
"""

import numpy as np
from scipy.special import logsumexp

from src.validation import estimation as checks
from src.validation.bm_1 import joint_grid as grid
from src.validation.bm_1 import joint_estimation as fv


def rectangular_hat_law(scores, counts):
    """Normalize a nested rectangular hat law from a square score grid.

    First-axis nodes vary slowest, exactly as in the existing square banks.
    Physical coordinates and continuous prior remain unchanged.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not scores.shape[0] or not np.all(np.isfinite(scores)):
        raise ValueError('Scores must be a finite nonempty two-dimensional array.')
    finest = int(np.sqrt(scores.shape[1]))
    if finest < 2 or finest**2 != scores.shape[1]:
        raise ValueError('Source scores require a complete square parameter grid.')
    counts = tuple(checks.positive_integer(n, 'axis nodes', minimum=2) for n in counts)
    if len(counts) != 2 or any(n > finest or (finest-1) % (n-1) for n in counts):
        raise ValueError('Both rectangular axes must be nested in the source grid.')
    indices = [np.arange(0, finest, (finest-1) // (n - 1)) for n in counts]
    selected = scores.reshape(len(scores), finest, finest)[:, indices[0], :][:, :, indices[1]]
    return tensor_hat_law(selected.reshape(len(scores), -1), counts)


def tensor_hat_law(scores, counts):
    """Positive normalized tensor hats from a complete rectangular score grid."""
    counts = tuple(checks.positive_integer(n, 'axis nodes', minimum=2) for n in counts)
    if len(counts) != 2:
        raise ValueError('A tensor hat requires two axes.')
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 2 or not len(scores) or scores.shape[1] != np.prod(counts)
            or not np.all(np.isfinite(scores))):
        raise ValueError('Scores must contain the complete finite rectangular grid.')
    unit = [np.linspace(0, 1, n) for n in counts]
    quadrature = [np.r_[.5, np.ones(n - 2), .5] / (n - 1) for n in counts]
    likelihood = np.cumsum(scores.reshape(len(scores), *counts), axis=0)
    log_prior = np.log(np.outer(*quadrature))
    evidence = logsumexp((likelihood + log_prior).reshape(len(scores), -1), axis=1)
    density = np.exp(likelihood - evidence[:, None, None])
    weights = density * np.outer(*quadrature)
    mass = weights.sum(axis=(1, 2))
    if np.max(abs(mass - 1)) >= 1e-10:
        raise ValueError('Tensor hat mass is not normalized.')
    mean, sd = [], []
    for axis, n in enumerate(counts):
        step = 1 / (n - 1)
        centers = unit[axis].copy()
        centers[[0, -1]] = [step / 3, 1 - step / 3]
        second = unit[axis] ** 2 + step ** 2 / 6
        second[[0, -1]] = [step ** 2 / 6, 1 - 2 * step / 3 + step ** 2 / 6]
        marginal = weights.sum(axis=2 if axis == 0 else 1)
        m = marginal @ centers
        mean.append(m)
        sd.append(np.sqrt(np.maximum(marginal @ second - m ** 2, 0)))
    centers = []
    for nodes in unit:
        c = nodes.copy(); h = 1 / (len(nodes) - 1)
        c[[0, -1]] = [h / 3, 1 - h / 3]
        centers.append(c)
    mean = np.column_stack(mean)
    cross = np.einsum('kij,i,j->k', weights, *centers, optimize=True)
    return dict(unit=unit, density=density, evidence=evidence,
                mean=mean, sd=np.column_stack(sd), mass=mass,
                covariance=cross - mean[:, 0] * mean[:, 1])


def _interpolate(values, source, target):
    for axis in (0, 1):
        j = np.clip(np.searchsorted(source[axis], target[axis], side='right') - 1,
                    0, len(source[axis]) - 2)
        t = (target[axis] - source[axis][j]) / np.diff(source[axis])[j]
        if axis == 0:
            values = values[j, :] * (1 - t[:, None]) + values[j + 1, :] * t[:, None]
        else:
            values = values[:, j] * (1 - t[None, :]) + values[:, j + 1] * t[None, :]
    return values


def compare_rectangular_hats(left, right):
    """Compare parameter law and evidence only; no state-mixture claim."""
    union = [np.unique(np.r_[a, b]) for a, b in zip(left['unit'], right['unit'])]
    area = np.outer(*[np.diff(u) for u in union])
    tv = []
    for a, b in zip(left['density'], right['density']):
        delta = _interpolate(b, right['unit'], union) - _interpolate(a, left['unit'], union)
        corners = delta[:-1, :-1], delta[1:, :-1], delta[:-1, 1:], delta[1:, 1:]
        tv.append([.5 * float(np.sum(area * grid._bilinear_abs_integral(corners, order)))
                   for order in grid.G1C0_TV_ORDERS])
    tv = np.asarray(tv)
    values = dict(parameter_tv=tv[:, -1],
                  unit_mean_error=np.max(abs(left['mean'] - right['mean']), axis=1),
                  unit_sd_error=np.max(abs(left['sd'] - right['sd']), axis=1),
                  cumulative_evidence_error=abs(left['evidence'] - right['evidence']),
                  tv_quadrature_refinement=np.max(abs(np.diff(tv, axis=1)), axis=1))
    maxima = {name: float(v.max()) for name, v in values.items()}
    return dict(maximum_errors=maxima, prefix_errors={k: v.tolist() for k, v in values.items()},
                parameter_only_thresholds_pass=bool(maxima['parameter_tv'] + 1e-8 <= .02
                    and maxima['unit_mean_error'] <= .01 and maxima['unit_sd_error'] <= .02
                    and maxima['cumulative_evidence_error'] <= .1
                    and maxima['tv_quadrature_refinement'] <= 1e-8))


def select_spatial_panel(report):
    """Deterministic posthoc support, truth, peak and worst old-odds nodes."""
    rows = {(tuple(r['coordinates']), r['fv_label']): r for r in report['rows']
            if r['parameter_nodes_per_axis'] == 65}
    panel = []
    for pair in grid.core.F4_G_PAIRS:
        middle, fine = rows[pair, 'middle'], rows[pair, 'spatial_fine']
        cumulative = np.cumsum(middle['conditional_log_evidence'], axis=0)
        selected = {j: ['support corner'] for j in (0, 64, 4160, 4224)}
        selected[2112] = ['support center']
        def add(index, reason):
            selected.setdefault(int(index), []).append(reason)
        hypotheses = np.asarray(middle['hypotheses'])
        truth = np.asarray(middle['truth'])
        distance = ((hypotheses-truth)/np.ptp(hypotheses, axis=0))**2
        add(np.argmin(distance.sum(axis=1)), 'nearest truth node')
        for day in (1, 3, 7):
            add(np.argmax(cumulative[24*day-1]), f'maximum likelihood at day {day}')
        refined = np.cumsum(fine['conditional_log_evidence'], axis=0)
        window = ((cumulative >= cumulative.max(axis=1)[:, None]-5)
                  | (refined >= refined.max(axis=1)[:, None]-5))
        delta = refined-cumulative
        span = np.max(np.where(window, delta, -np.inf), axis=1)-np.min(np.where(window, delta, np.inf), axis=1)
        k = int(np.argmax(span))
        add(np.argmax(np.where(window[k], delta[k], -np.inf)), f'old spatial odds maximum at observation {k+1}')
        add(np.argmin(np.where(window[k], delta[k], np.inf)), f'old spatial odds minimum at observation {k+1}')
        indices = sorted(selected)
        panel.append(dict(coordinates=list(pair), sequence_id=middle['sequence_id'],
            selected_indices=indices, hypotheses=hypotheses[indices].tolist(),
            reasons={str(j): selected[j] for j in indices},
            old_spatial_maximum_log_odds_error=float(span.max())))
    return panel


def diagnose_parameter_axes(report):
    """Coarsen one parameter axis at a time on each recorded FV setting.

    The caller validates the original G2a report. Completed negative reports
    are deliberately usable; this cannot change their acceptance status.
    """
    rows = []
    for row in report['rows']:
        if row['parameter_nodes_per_axis'] != 65:
            continue
        scores = np.asarray(row['conditional_log_evidence'])
        shapes = [(n, 65) for n in (5, 9, 17, 33, 65)]+[(65, n) for n in (17, 33)]
        laws = {shape: rectangular_hat_law(scores, shape) for shape in shapes}
        square = grid._hat_law(row)
        fv._equal_array(laws[65,65]['density'], square['density'], 'square hat reproduction')
        fv._equal_array(laws[65,65]['mean'], square['mean'], 'square hat moment reproduction')
        pairs = [((a,65),(b,65)) for a,b in ((5,9),(9,17),(17,33),(33,65),(9,65),(17,65))]
        pairs += [((65,17),(65,33)), ((65,33),(65,65))]
        for left, right in pairs:
            comparison = compare_rectangular_hats(laws[left], laws[right])
            comparison['day_1_3_7'] = [dict(days=d,
                **{name: values[24*d-1] for name,values in comparison['prefix_errors'].items()})
                for d in (1, 3, 7)]
            rows.append(dict(coordinates=row['coordinates'], fv_label=row['fv_label'],
                left_counts=list(left), right_counts=list(right), **comparison))
    return rows


def _state_difference(left, right):
    """Conservatively remap for comparison only; never feed this to a filter."""
    target = np.linspace(0, 1, left['predicted'].shape[1]+1)
    source = np.linspace(0, 1, right['predicted'].shape[1]+1)
    result = {}
    for phase in ('predicted', 'posterior'):
        cdf = np.column_stack([np.zeros(len(right[phase])), np.cumsum(right[phase], axis=1)])
        remapped = np.array([np.diff(np.interp(target, source, row)) for row in cdf])
        result[phase+'_conditional_l1'] = np.sum(abs(remapped-left[phase]), axis=1).tolist()
    return result


def run_spatial_panel(report, panel, *, rho_levels=4, progress=None):
    """Serial conditional panel with exact cached FV and timing.

    Raw scores are retained. Conditional state histories are used to compute
    L1 diagnostics, then discarded to bound memory/output size. A subset can
    reveal a failure, but cannot certify a complete parameter bank.
    """
    rho_levels = checks.positive_integer(rho_levels, 'rho spatial levels')
    if rho_levels not in (3,4):
        raise ValueError('Rho spatial levels must be 3 or 4.')
    rows, comparisons = [], []
    total = sum((rho_levels if p['coordinates'][0]=='rho' else 3)*len(p['selected_indices']) for p in panel)
    completed = 0
    for selection in panel:
        pair, sid = tuple(selection['coordinates']), selection['sequence_id']
        case, sequence, truth = grid._g2a_truth_record(pair, sid, 7)
        if not truth['passes']:
            raise ValueError('Old-record truth reproduction failed.')
        originals = {r['fv_label']: r for r in report['rows']
                     if tuple(r['coordinates']) == pair and r['parameter_nodes_per_axis'] == 65}
        base_cells = originals['middle']['settings']['x_steps']
        level_count = rho_levels if pair[0]=='rho' else 3
        levels = tuple(2**level*(base_cells-1)+1 for level in range(level_count))
        scores_by_level = [[] for _ in levels]
        state_errors = [[] for _ in range(level_count-1)]
        for original_index, hypothesis in zip(selection['selected_indices'], selection['hypotheses']):
            previous = None
            for level, cells in enumerate(levels):
                settings = dict(originals['middle']['settings'], x_steps=cells)
                group = dict(stage='G2a-grid-diagnosis', case=case, observations=sequence.observations,
                    pair=pair, sequence_id=sid, hypotheses=np.asarray([hypothesis]),
                    label=f'spatial/{cells}', settings=settings, backend='cached')
                result = fv._candidate_replay((0,0,group))[2]
                if not fv._probability_gate([result]):
                    raise ValueError('Spatial panel probability gate failed.')
                reproduction = None
                if level < 2:
                    old = np.asarray(originals[('middle','spatial_fine')[level]]['conditional_log_evidence'])[:, original_index]
                    reproduction = float(np.max(abs(result['scores']-old)))
                    if reproduction > 1e-10:
                        raise ValueError('Spatial panel does not reproduce old conditional scores.')
                scores_by_level[level].append(result['scores'])
                rows.append(dict(coordinates=list(pair), sequence_id=sid, source_node_index=original_index,
                    hypothesis=hypothesis, settings=settings, conditional_log_evidence=result['scores'].tolist(),
                    old_score_reproduction_error=reproduction,
                    **{name: float(result[name]) for name in ('elapsed_seconds','actual_max_step',
                        'maximum_abs_mass_error','minimum_cell_mass')}))
                if previous is not None:
                    state_errors[level-1].append(_state_difference(previous, result))
                previous = result
                completed += 1
                if progress is not None:
                    progress(completed, total)
        for level in range(level_count-1):
            a, b = [np.cumsum(np.column_stack(s), axis=0) for s in scores_by_level[level:level+2]]
            window = (a >= a.max(axis=1)[:,None]-5) | (b >= b.max(axis=1)[:,None]-5)
            delta = b-a
            span = np.max(np.where(window,delta,-np.inf),axis=1)-np.min(np.where(window,delta,np.inf),axis=1)
            differences = dict(panel_log_odds_span=span.tolist(),
                maximum_conditional_cumulative_score_difference=np.max(abs(delta),axis=1).tolist())
            for phase in ('predicted','posterior'):
                errors = np.array([r[phase+'_conditional_l1'] for r in state_errors[level]]).T
                differences['maximum_'+phase+'_conditional_l1'] = np.max(errors,axis=1).tolist()
                differences['relevant_'+phase+'_conditional_l1'] = np.max(np.where(window,errors,0.),axis=1).tolist()
            comparisons.append(dict(coordinates=list(pair), left_cells=levels[level], right_cells=levels[level+1],
                prefix_errors=differences, maximum_errors={k: float(np.max(v)) for k,v in differences.items()},
                conditional_runs=len(selection['selected_indices']), certifies_full_bank=False))
    return dict(rows=rows, refinement_comparisons=comparisons, conditional_runs=completed,
        stored_conditional_states=False, serial_seconds_sum=sum(r['elapsed_seconds'] for r in rows))
