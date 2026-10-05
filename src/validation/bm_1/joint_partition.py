"""Whole record/scenario partitions; no split floating-point reductions."""
import numpy as np

from src.validation.bm_1 import joint_night as night, joint_population as population
from src.validation.bm_1 import joint_estimation as fv, joint_robustness as robustness


def study_for(stage):
    return {'G2b': night, 'G2c': population}[stage]


def jobs(study):
    result = {}
    for unit in study.units():
        result.setdefault(robustness.unit_name(unit[:3]), []).append(unit)
    return result


def compact_record(rows, study):
    if not rows:
        return None
    first = rows[0]
    pair, sid, scenario = tuple(first['coordinates']), first['sequence_id'], first['scenario']
    checks = night._group_checks(rows, pair, sid, scenario) if study is night else []
    expected = 8 if study is night else 1
    if len(rows) != expected:
        return None
    candidate = next(r for r in rows if tuple(r['parameter_grid_shape']) == study.GRIDS['__'.join(pair)][0]
                     and r['fv_label'] == 'middle')
    return dict(unit_name=robustness.unit_name((pair, sid, scenario)),
        coordinates=list(pair), sequence_id=sid, scenario=scenario, completed=True,
        passes=bool(fv._probability_gate(rows) and all(c['passes'] for c in checks)
                    and (study is not night or len(checks) == 7)),
        conditional_runs=len(study.configurations(pair))*int(np.prod(study.GRIDS['__'.join(pair)][-1])),
        observation_times=candidate['observation_times'], bank_log_evidence=candidate['bank_log_evidence'],
        checkpoints=robustness.checkpoints(candidate, candidate['truth_record'],
                                           candidate['observation_indices'], study.DAYS))


def summary_records(stage, records, comparisons):
    statistical = robustness.summarize_units(stage, [r for r in records if r is not None])
    failed = [c for c in comparisons if not c['passes']]
    return dict(statistical, stage=study_for(stage).STAGE, design_version=2,
        completed=statistical['complete'], passes=bool(statistical['passes'] and not failed),
        comparisons=comparisons, failed_comparisons=failed,
        expected_record_scenarios=statistical['expected_units'],
        completed_record_scenarios=statistical['completed_units'],
        releases_existing_g2b_g2c=False)


class JobStudy:
    """Restrict execution/collection to one immutable job; retain full design header."""
    counter_label = 'job'

    def __init__(self, study, name):
        self.study = study
        self.selected = jobs(study)[name]
        self.execution_runs = sum(int(np.prod(study.GRIDS['__'.join(u[0])][-1])) for u in self.selected)

    def __getattr__(self, name):
        return getattr(self.study, name)

    def units(self):
        return self.selected

    def summarize(self, rows):
        record = compact_record(rows, self.study)
        pair, sid, scenario, _ = self.selected[0]
        comparisons = night._group_checks(rows, pair, sid, scenario) if self.study is night else []
        return dict(completed=record is not None, complete=record is not None,
            passes=bool(record is not None and record['passes']),
            failed_comparisons=[c for c in comparisons if not c['passes']], comparisons=comparisons)
