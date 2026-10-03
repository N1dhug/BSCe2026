"""G2b v2: fixed rectangular robustness design, saved in execution blocks.

The two fresh records and five paired scenarios are unchanged from G2b v1.
Parameter/FV settings follow the accepted G2a-v2 pilot; every scenario still
gets its own full numerical controls. An eight-hour pause grants no acceptance.
The old square G2b/G2c commands and their prerequisites remain untouched.
"""
import numpy as np

from src.validation.bm_1 import estimation as core, joint_estimation as fv
from src.validation.bm_1 import joint_grid as grid, joint_pilot as pilot
from src.validation.bm_1 import joint_robustness as robustness


STAGE = 'G2b-rectangular'
DAYS = 7
GRIDS = pilot.GRIDS
SCENARIOS = robustness.SCENARIOS
PILOT_IDS = robustness.PILOT_IDS


def configurations(pair):
    return pilot.configurations(pair)


def units():
    return [(pair, sid, scenario, label)
            for pair, sid in zip(core.F4_G_PAIRS, PILOT_IDS)
            for scenario in SCENARIOS for label in pilot.LABELS]


def unit_name(unit):
    pair, sid, scenario, label = unit
    return f"{'_'.join(pair)}_{sid}_{scenario}_{label}"


def design_header():
    base = pilot.design_header()
    return dict(base, stage=STAGE, design_version=2, sequence_ids=list(PILOT_IDS),
        unit_names=[unit_name(u) for u in units()],
        expected_conditional_runs=sum(int(np.prod(GRIDS['__'.join(u[0])][-1])) for u in units()),
        scenarios=robustness.design_header('G2b')['scenarios'],
        expected_record_scenarios=len(core.F4_G_PAIRS)*len(SCENARIOS),
        design_origin='Fixed before fresh G2b-v2 data; rectangular numerical settings from accepted G2a-v2',
        demand_mean_rule='Each parameter node uses its own conditional state and cell Gauss demand integral',
        interpretation='Fresh-record numerical robustness pilot; no population, recovery or forecast acceptance; '
                       'does not release the old square G2c design.')


def make_group(unit, truth=None):
    if unit not in units():
        raise ValueError('Unregistered G2b-v2 unit.')
    pair, sid, scenario, label = unit
    truth_case, sequence, record = truth if truth is not None else grid._g2a_truth_record(pair, sid, DAYS)
    if not record['passes']:
        raise ValueError('G2b-v2 truth gate failed; retain the record and do not substitute a seed.')
    case, observations, indices = robustness._scenario(truth_case, sequence.observations, scenario)
    counts = GRIDS['__'.join(pair)]
    return dict(stage=STAGE, pair=pair, sequence_id=sid, label=label, case=case,
        observations=observations, hypotheses=pilot.parameter_grid(pair, counts[-1])[0],
        counts=counts, settings=dict(configurations(pair))[label], prior_bridge=False,
        backend='cached', array_output=True, keep_demand=True, scenario=scenario,
        truth_record=record, observation_indices=indices.tolist())


class RectangularReduction(pilot.RectangularReduction):
    def finish(self):
        rows = super().finish()
        for row in rows:
            row.update(scenario=self.group['scenario'], truth_record=self.group['truth_record'],
                       observation_indices=self.group['observation_indices'])
        return rows


advance = pilot.advance


def validate_rows(rows, group):
    pilot.validate_rows(rows, group)
    count = len(group['observations'].times)
    for row in rows:
        for key in ('scenario', 'truth_record', 'observation_indices'):
            grid._check_recorded(row.get(key), group[key], 'G2b-v2 '+key)
        h = len(row['hypotheses'])
        reduced = fv.parameter_bank_weights(row['conditional_log_evidence'], row['initial_log_weights'],
                                           group['observations'].observed)
        for phase, weight_field in (('predicted', 'prior_log_weights'), ('posterior', 'log_weights')):
            conditional = fv._array(row['conditional_'+phase+'_demand_mean'], (count, h), 'conditional demand')
            expected = np.sum(np.exp(reduced[weight_field])*conditional, axis=1)
            fv._equal_array(row['marginal_'+phase+'_demand_mean'], expected, 'G2b-v2 marginal demand')
    reference = rows[-1]
    finest = tuple(reference['parameter_grid_shape'])
    for row in rows[:-1]:
        shape = tuple(row['parameter_grid_shape'])
        indices = np.arange(np.prod(finest)).reshape(finest)[
            ::(finest[0]-1)//(shape[0]-1), ::(finest[1]-1)//(shape[1]-1)].ravel()
        for phase in ('predicted', 'posterior'):
            field = 'conditional_'+phase+'_demand_mean'
            fv._equal_array(row[field], np.asarray(reference[field])[:, indices], 'nested demand', 0.)
    return True


def _group_checks(rows, pair, sid, scenario):
    shapes = GRIDS['__'.join(pair)]
    lookup = {(tuple(r['parameter_grid_shape']), r['fv_label']):r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate G2b-v2 banks.')
    comparisons = []
    identity = dict(coordinates=list(pair), sequence_id=sid, scenario=scenario)
    for shape in shapes:
        if (shape, 'middle') not in lookup: continue
        middle = lookup[shape, 'middle']
        for axis in ('time', 'spatial', 'quadrature'):
            if (shape, axis+'_fine') not in lookup: continue
            fine = lookup[shape, axis+'_fine']
            comparison = fv.compare_banks(middle, fine)
            if axis == 'time':
                refined = bool(np.all(np.asarray(fine['conditional_actual_max_steps']) <
                    np.asarray(middle['conditional_actual_max_steps'])*(1-1e-12)))
                comparison.update(actual_step_refined=refined, passes=bool(comparison['passes'] and refined))
            comparisons.append(dict(identity, kind='FV_'+axis, parameter_grid_shape=list(shape),
                left=middle['label'], right=fine['label'], **comparison))
    if all((shape, 'middle') in lookup for shape in shapes):
        a, b = [lookup[shape, 'middle'] for shape in shapes]
        comparisons.append(dict(identity, kind='parameter_hat', left=a['label'], right=b['label'],
                                **pilot.compare_parameter_rows(a, b)))
    return comparisons


def _group_rows(rows, pair, sid, scenario):
    return [r for r in rows if tuple(r['coordinates'])==pair and r['sequence_id']==sid
            and r['scenario']==scenario]


def completed_group_checks(rows, unit):
    pair, sid, scenario, _ = unit
    return _group_checks(_group_rows(rows, pair, sid, scenario), pair, sid, scenario)


def summarize(rows):
    comparisons, records = [], []
    for pair, sid in zip(core.F4_G_PAIRS, PILOT_IDS):
        for scenario in SCENARIOS:
            selected = _group_rows(rows, pair, sid, scenario)
            checks = _group_checks(selected, pair, sid, scenario)
            comparisons.extend(checks)
            if len(selected) != 8 or len(checks) != 7: continue
            candidate = next(r for r in selected if tuple(r['parameter_grid_shape'])==GRIDS['__'.join(pair)][0]
                             and r['fv_label']=='middle')
            records.append(dict(unit_name=robustness.unit_name((pair, sid, scenario)),
                coordinates=list(pair), sequence_id=sid, scenario=scenario, completed=True,
                passes=bool(fv._probability_gate(selected) and all(c['passes'] for c in checks)),
                conditional_runs=4*int(np.prod(GRIDS['__'.join(pair)][-1])),
                observation_times=candidate['observation_times'], bank_log_evidence=candidate['bank_log_evidence'],
                checkpoints=robustness.checkpoints(candidate, candidate['truth_record'],
                                                   candidate['observation_indices'], DAYS)))
    statistical = robustness.summarize_units('G2b', records)
    return dict(statistical, stage=STAGE, design_version=2, completed=statistical['complete'],
        expected_record_scenarios=statistical['expected_units'],
        completed_record_scenarios=statistical['completed_units'],
        bank_reductions=len(rows), comparisons=comparisons,
        failed_comparisons=[{k:c[k] for k in ('coordinates','sequence_id','scenario','kind','left','right')}
                            for c in comparisons if not c['passes']],
        releases_existing_g2b_g2c=False)
