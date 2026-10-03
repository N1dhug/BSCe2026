"""Paired robustness experiments and resumable population units for F4-G.

Scientific design and raw validation live here; file checkpoints and source
provenance belong to the estimation CLI. No experiment runs at import time.
"""
from copy import deepcopy
from dataclasses import replace
import re

import numpy as np

from src.params.bm_1 import BM1ObservationSeries
from src.validation import estimation as checks
from src.validation.bm_1 import estimation as core, joint_estimation as fv, joint_grid as grid


RECORD_DAYS = 7
COUNTS = (33, 65)
PILOT_IDS = (1360, 1361)
POPULATION_STARTS = (1362, 1562)
POPULATION_SIZE = 200
ROBUSTNESS_SIZE = 20
SCENARIOS = ('matched', 'sampling_2h', 'sampling_4h', 'baseline_055', 'initial_beta_16_24')
BACKEND = 'cached-lapack; exact BM1 input keys and unrounded step lengths'


def design_units(stage):
    """Stable complete job order; subsets are execution chunks, never a new study."""
    if stage == 'G2b':
        return [(pair, sid, scenario) for pair, sid in zip(core.F4_G_PAIRS, PILOT_IDS)
                for scenario in SCENARIOS]
    if stage == 'G2c':
        return [(pair, sid, scenario) for pair, start in zip(core.F4_G_PAIRS, POPULATION_STARTS)
                for sid in range(start, start+POPULATION_SIZE)
                for scenario in (SCENARIOS if sid < start+ROBUSTNESS_SIZE else ('matched',))]
    raise ValueError('Robustness stage must be G2b or G2c.')


def unit_name(unit):
    pair, sid, scenario = unit
    return f"{'_'.join(pair)}_{sid}_{scenario}"


def design_header(stage):
    units = design_units(stage)
    return dict(stage=stage, design_version=1, record_days=RECORD_DAYS,
        parameter_grid_counts=list(COUNTS if stage=='G2b' else COUNTS[:1]),
        parameter_grid_anchor=COUNTS[-1], numerical_levels=deepcopy(fv.G1A_NUMERICS),
        raw_unit_format='bm1-array-checkpoint-v1; lossless float64 NPY arrays and JSON metadata in ZIP, no pickle',
        truth_criteria=dict(grid.G2A_TRUTH_CRITERIA), conditional_backend=BACKEND,
        unit_names=[unit_name(u) for u in units],
        observation_contract=dict(time_unit='day', observation_kind='point', price_side='right'),
        scenarios=dict(matched=dict(sampling_hours=1, baseline=.5, initial_beta=[20.,20.]),
            sampling_2h=dict(sampling_hours=2, baseline=.5, initial_beta=[20.,20.]),
            sampling_4h=dict(sampling_hours=4, baseline=.5, initial_beta=[20.,20.]),
            baseline_055=dict(sampling_hours=1, baseline=.55, initial_beta=[20.,20.]),
            initial_beta_16_24=dict(sampling_hours=1, baseline=.5, initial_beta=[16.,24.])),
        fv_criteria=grid._g2a_header(RECORD_DAYS, PILOT_IDS, COUNTS, fv.G1A_NUMERICS,
                                    registered=True)['fv_criteria'],
        parameter_criteria=dict(grid.G1B0_CRITERIA),
        demand_mean_rule='cell Gauss quadrature at each hypothesis; old/new parameter weights for prediction/posterior',
        population_size_per_pair=POPULATION_SIZE if stage=='G2c' else 1,
        paired_robustness_size_per_pair=ROBUSTNESS_SIZE if stage=='G2c' else 1,
        interpretation=('New-record numerical robustness pilot; no population/recovery/forecast acceptance.'
                        if stage=='G2b' else 'Pilot-supported population numerics; no per-record convergence proof or recovery pass target.'))


def require_g2a(report):
    grid.validated_g2a(report, require_source=True)
    if report['record_days'] != RECORD_DAYS or not report['completed'] or not report['passes']:
        raise ValueError('G2b requires the complete passing seven-day G2a raw report.')


def _scenario(case, observations, name):
    if name not in SCENARIOS:
        raise ValueError('Unknown registered robustness scenario.')
    hours = 2 if name=='sampling_2h' else 4 if name=='sampling_4h' else 1
    indices = np.arange(hours-1, len(observations.times), hours)
    chosen = BM1ObservationSeries(observations.times[indices], observations.values[indices],
        observations.time_unit, observations.observed[indices], observations.observation_kind, observations.price_side)
    return replace(case, observation_times=chosen.times,
        baseline=.55 if name=='baseline_055' else .5,
        initial_alpha=16. if name=='initial_beta_16_24' else 20.,
        initial_beta=24. if name=='initial_beta_16_24' else 20.), chosen, indices


def _configurations(stage, pair):
    configs = grid._grid_configurations(fv.G1A_NUMERICS)['__'.join(pair)]
    return configs if stage=='G2b' else [c for c in configs if c[0]=='middle']


def _comparisons(rows):
    lookup = {(r['parameter_nodes_per_axis'], r['fv_label']): r for r in rows}
    comparisons = []
    for n in COUNTS:
        middle = lookup[n, 'middle']
        for axis in ('time', 'spatial', 'quadrature'):
            fine = lookup[n, axis+'_fine']
            comparison = fv.compare_banks(middle, fine)
            if axis == 'time':
                refined = bool(np.all(np.asarray(fine['conditional_actual_max_steps']) <
                                      np.asarray(middle['conditional_actual_max_steps'])*(1-1e-12)))
                comparison.update(actual_step_refined=refined, passes=bool(comparison['passes'] and refined))
            comparisons.append(dict(kind='FV_'+axis, parameter_nodes_per_axis=n,
                left=middle['label'], right=fine['label'], selected=True, **comparison))
    a, b = [lookup[n, 'middle'] for n in COUNTS]
    comparisons.append(dict(kind='parameter_hat', left=a['label'], right=b['label'],
                            selected=True, **grid.compare_hat_laws(a, b)))
    dual = dict(kind='original_dual_cell', selected=False, **grid.compare_parameter_banks(a, b))
    return comparisons, dual


def _linear_quantile(nodes, density, probability):
    """Invert the exact integral of a nonnegative piecewise-linear density."""
    if probability == 0: return float(nodes[0])
    if probability == 1: return float(nodes[-1])
    widths = np.diff(nodes)
    cumulative = np.r_[0., np.cumsum(widths*(density[:-1]+density[1:])/2)]
    j = min(int(np.searchsorted(cumulative, probability, side='right')-1), len(widths)-1)
    remaining = probability-cumulative[j]
    slope = (density[j+1]-density[j])/widths[j]
    if slope == 0:
        distance = remaining/density[j]
    else:
        discriminant = density[j]**2 + 2*slope*remaining
        if discriminant < -1e-13:
            raise ValueError('Hat quantile encountered a negative discriminant.')
        distance = 2*remaining/(density[j]+np.sqrt(max(0., discriminant)))
    return float(nodes[j]+distance)


def _physical_hat_moments(nodes, density, coordinate):
    """Integrate physical moments; log-uniform axes are not linear parameters."""
    bounds = np.asarray(core.F4_E_PARAMETER_GRIDS[coordinate])[[0,-1]]
    def transform(unit):
        if coordinate.startswith('sigma'):
            return np.exp(np.log(bounds[0]) + unit*np.log(bounds[1]/bounds[0]))
        return bounds[0] + unit*(bounds[1]-bounds[0])
    moments = []
    for order in (8, 16):
        z, w = np.polynomial.legendre.leggauss(order)
        s = (z+1)/2
        points = nodes[:-1,None] + np.diff(nodes)[:,None]*s
        values = density[:-1,None]*(1-s) + density[1:,None]*s
        weights = values*np.diff(nodes)[:,None]*w/2
        physical = transform(points)
        moments.append([float(np.sum(weights*physical)), float(np.sum(weights*physical**2))])
    if np.max(abs(np.subtract(*moments))) > 1e-10:
        raise ValueError('Physical hat moment quadrature did not converge.')
    mean, second = moments[-1]
    variance = second-mean**2
    if variance < -1e-12:
        raise ValueError('Physical hat variance is negative.')
    interval = [float(transform(_linear_quantile(nodes, density, p))) for p in (.025,.975)]
    return dict(mean=mean, sd=float(np.sqrt(max(variance,0.))), interval_95=interval)


def checkpoints(row, truth_record, indices):
    law = grid._hat_law(row)
    nodes = law['unit']; q = np.full(len(nodes), 1/(len(nodes)-1)); q[[0,-1]] *= .5
    state = np.asarray(truth_record['fine']['state'])[1:][indices]
    demand = np.asarray(truth_record['fine']['demand'])[1:][indices]
    centers = (np.asarray(row['x_edges'])[:-1]+np.asarray(row['x_edges'])[1:])/2
    result = []
    for day in grid.G2A_DAYS:
        if day > RECORD_DAYS: continue
        count = int(np.searchsorted(row['observation_times'], day, side='right'))
        density = law['density'][count-1]
        marginals = (density @ q, q @ density)
        parameters = {name:_physical_hat_moments(nodes, marginal, name)
                      for name, marginal in zip(row['coordinates'], marginals)}
        for name, value in zip(row['coordinates'], row['truth']):
            a, b = parameters[name]['interval_95']
            parameters[name]['truth'] = value
            parameters[name]['interval_contains_truth'] = bool(a<=value<=b)
        scores = np.asarray(row['bank_log_evidence'])[:count]
        metrics = {}
        for phase in ('predicted','posterior'):
            means = np.asarray(row['marginal_'+phase+'_mass'])[:count] @ centers
            dm = np.asarray(row['marginal_'+phase+'_demand_mean'])[:count]
            metrics[phase+'_state_rmse'] = float(np.sqrt(np.mean((means-state[:count])**2)))
            metrics[phase+'_demand_rmse'] = float(np.sqrt(np.mean((dm-demand[:count])**2)))
        result.append(dict(days=day, observation_count=count, parameters=parameters,
            mean_predictive_y_log_score=float(np.mean(scores)),
            cumulative_log_evidence=float(np.sum(scores)), **metrics))
    return result


def run_unit(stage, unit, *, workers=8, progress=None):
    """Run one complete preregistered record/scenario; the CLI gates dependencies."""
    workers = checks.positive_integer(workers, 'workers')
    pair, sid, scenario = tuple(unit[0]), unit[1], unit[2]
    unit = (pair, sid, scenario)
    if unit not in design_units(stage):
        raise ValueError('Unregistered robustness unit.')
    truth_case, sequence, record = grid._g2a_truth_record(pair, sid, RECORD_DAYS)
    case, observations, indices = _scenario(truth_case, sequence.observations, scenario)
    counts = COUNTS if stage=='G2b' else COUNTS[:1]
    result = dict(stage=stage, design_version=1, unit_name=unit_name(unit),
        coordinates=list(pair), sequence_id=sid, scenario=scenario, record_days=RECORD_DAYS,
        truth_record=deepcopy(record), observation_indices=indices.tolist(), completed=False, passes=False,
        conditional_runs=0, rows=[], comparisons=[], original_dual_cell_comparison=None, checkpoints=[])
    if not record['passes']: return result
    hypotheses = grid._parameter_grid(pair, counts[-1], COUNTS[-1])[0]
    groups = [dict(stage=stage, pair=pair, sequence_id=sid, case=case, observations=observations,
        hypotheses=hypotheses, counts=counts, parameter_grid_finest=COUNTS[-1], label=label,
        settings=settings, prior_bridge=False, backend='cached', keep_demand=True, array_output=True)
        for label, settings in _configurations(stage, pair)]
    rows = [r for bank in fv._run_groups_streaming(groups, workers, progress, grid._StreamingGridReduction) for r in bank]
    comparisons, dual = _comparisons(rows) if stage=='G2b' else ([], None)
    candidate = next(r for r in rows if r['parameter_nodes_per_axis']==COUNTS[0] and r['fv_label']=='middle')
    result.update(completed=True, conditional_runs=sum(len(g['hypotheses']) for g in groups),
        rows=rows, comparisons=comparisons, original_dual_cell_comparison=dual,
        checkpoints=checkpoints(candidate, record, indices),
        passes=bool(fv._probability_gate(rows) and all(c['passes'] for c in comparisons)))
    return result


def validate_unit(report, stage, unit, *, require_source=False, truth_cache=None):
    """Recompute identities and gates from raw values, including negative results."""
    pair, sid, scenario = unit
    if unit not in design_units(stage): raise ValueError('Unregistered robustness unit.')
    if require_source and (report.get('working_tree_dirty') is not False or
            not re.fullmatch('[0-9a-f]{40}', str(report.get('source_commit')))):
        raise ValueError('Robustness evidence requires clean source metadata.')
    for name, value in dict(stage=stage, design_version=1, unit_name=unit_name(unit),
        coordinates=list(pair), sequence_id=sid, scenario=scenario, record_days=RECORD_DAYS).items():
        grid._check_recorded(report.get(name), value, stage+' '+name)
    key = (pair, sid, RECORD_DAYS)
    if truth_cache is not None and key in truth_cache:
        truth_case, sequence, expected = truth_cache[key]
    else:
        truth_case, sequence, expected = grid._g2a_truth_record(*key)
        if truth_cache is not None: truth_cache[key] = (truth_case, sequence, expected)
    grid._check_recorded(report.get('truth_record'), expected, stage+' truth')
    case, observations, indices = _scenario(truth_case, sequence.observations, scenario)
    grid._check_recorded(report.get('observation_indices'), indices, stage+' indices')
    if not expected['passes']:
        for name, value in dict(completed=False, passes=False, conditional_runs=0, rows=[],
            comparisons=[], original_dual_cell_comparison=None, checkpoints=[]).items():
            grid._check_recorded(report.get(name), value, stage+' stopped '+name)
        return True
    counts = COUNTS if stage=='G2b' else COUNTS[:1]
    expected_rows = {(n,label):settings for label, settings in _configurations(stage,pair) for n in counts}
    rows = report['rows']; seen = set()
    if len(rows) != len(expected_rows): raise ValueError(stage+' incomplete banks.')
    for row in rows:
        key = (row['parameter_nodes_per_axis'], row['fv_label'])
        if key not in expected_rows or key in seen: raise ValueError(stage+' duplicate or unregistered bank.')
        seen.add(key); n, label = key
        grid._validate_grid_row(row, pair=pair, sid=sid, n=n, label=label, case=case,
            observations=observations, settings=expected_rows[key], finest=COUNTS[-1], stage=stage, keep_demand=True)
    comparisons, dual = _comparisons(rows) if stage=='G2b' else ([],None)
    candidate = next(r for r in rows if r['parameter_nodes_per_axis']==COUNTS[0] and r['fv_label']=='middle')
    for name, value in dict(completed=True, comparisons=comparisons, original_dual_cell_comparison=dual,
        conditional_runs=len(_configurations(stage,pair))*counts[-1]**2,
        checkpoints=checkpoints(candidate,expected,indices),
        passes=bool(fv._probability_gate(rows) and all(c['passes'] for c in comparisons))).items():
        grid._check_recorded(report.get(name), value, stage+' '+name)
    return True


def compact_unit(report):
    """Keep score paths and checkpoint metrics for collection, release large banks."""
    result = {k:deepcopy(report[k]) for k in ('unit_name', 'coordinates', 'sequence_id', 'scenario',
              'completed', 'passes', 'conditional_runs', 'checkpoints')}
    candidates = [r for r in report['rows'] if r['parameter_nodes_per_axis']==COUNTS[0]
                  and r['fv_label']=='middle']
    result['observation_times'] = candidates[0]['observation_times'] if candidates else []
    result['bank_log_evidence'] = candidates[0]['bank_log_evidence'] if candidates else []
    return result


def summarize_units(stage, records):
    """Descriptive per-record summaries; never treat hours as independent units."""
    expected = {unit_name(u) for u in design_units(stage)}
    names = [r['unit_name'] for r in records]
    if len(set(names)) != len(names) or not set(names)<=expected:
        raise ValueError('Collection contains duplicate or unregistered units.')
    complete = set(names)==expected and all(r['completed'] for r in records)
    groups = []
    for pair in core.F4_G_PAIRS:
        for scenario in SCENARIOS:
            selected = [r for r in records if tuple(r['coordinates'])==pair and r['scenario']==scenario and r['completed']]
            if not selected: continue
            for day in grid.G2A_DAYS:
                if day>RECORD_DAYS: continue
                points = [next(c for c in r['checkpoints'] if c['days']==day) for r in selected]
                parameters = {}
                for name in pair:
                    values = [p['parameters'][name] for p in points]
                    errors = np.array([p['mean']-p['truth'] for p in values])
                    hits = sum(p['interval_contains_truth'] for p in values)
                    parameters[name] = dict(bias=float(errors.mean()), rmse=float(np.sqrt(np.mean(errors**2))),
                        median_posterior_sd=float(np.median([p['sd'] for p in values])),
                        median_interval_width=float(np.median([p['interval_95'][1]-p['interval_95'][0] for p in values])),
                        interval_hits=int(hits), interval_hit_rate=hits/len(values),
                        wilson_95=list(core._wilson_interval(hits, len(values))))
                metric_names = ('predicted_state_rmse','posterior_state_rmse','predicted_demand_rmse',
                                'posterior_demand_rmse','mean_predictive_y_log_score')
                metrics = {name:float(np.mean([p[name] for p in points])) for name in metric_names}
                groups.append(dict(coordinates=list(pair), scenario=scenario, days=day, records=len(points),
                    parameters=parameters, mean_record_metrics=metrics,
                    interpretation='Descriptive fixed-truth hit rates with binomial uncertainty; no universal calibration or recovery guarantee.'))
    lookup = {(tuple(r['coordinates']),r['sequence_id'],r['scenario']):r for r in records if r['completed']}
    paired = []
    for pair in core.F4_G_PAIRS:
        for scenario in SCENARIOS[1:]:
            for day in grid.G2A_DAYS:
                if day>RECORD_DAYS: continue
                differences = []
                ids = []
                for (coordinates,sid,name), r in lookup.items():
                    if coordinates!=pair or name!=scenario or (pair,sid,'matched') not in lookup: continue
                    base = lookup[pair,sid,'matched']
                    times = np.asarray(r['observation_times']); keep = times<=day
                    common = np.searchsorted(base['observation_times'], times[keep])
                    if not np.array_equal(np.asarray(base['observation_times'])[common],times[keep]):
                        raise ValueError('Paired scores must refer to identical observation times.')
                    differences.append(float(np.mean(np.asarray(r['bank_log_evidence'])[keep] -
                                                     np.asarray(base['bank_log_evidence'])[common])))
                    ids.append(sid)
                if differences:
                    paired.append(dict(coordinates=list(pair),scenario=scenario,days=day,
                        sequence_ids=ids, records=len(ids), per_record_score_differences=differences,
                        mean_score_difference=float(np.mean(differences)),
                        standard_error=None if len(ids)<2 else float(np.std(differences,ddof=1)/np.sqrt(len(ids))),
                        score_contract='scenario minus matched, evaluated only on common Y times before assimilation'))
    return dict(stage=stage, complete=bool(complete), expected_units=len(expected), completed_units=len(records),
        missing_units=sorted(expected-set(names)), failed_units=[r['unit_name'] for r in records if not r['passes']],
        passes=bool(complete and all(r['passes'] for r in records)),
        conditional_runs=sum(r['conditional_runs'] for r in records), groups=groups, paired_scores=paired)
