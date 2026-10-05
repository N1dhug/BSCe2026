"""Registered rectangular G2c follow-up; execution requires accepted G2b raw data.

Uses the G2b candidate grids, not a new grid selected on population outcomes.
Numerics are pilot-supported: this is not a per-record convergence proof.
"""
import sys
import numpy as np

from src.validation.bm_1 import joint_grid as grid, joint_pilot as pilot
from src.validation.bm_1 import joint_night as night, joint_robustness as robustness

STAGE = 'G2c-rectangular'
DAYS = 7
GRIDS = {key: (shapes[0],) for key, shapes in pilot.GRIDS.items()}


def units():
    return [(*unit, 'middle') for unit in robustness.design_units('G2c')]


unit_name = night.unit_name
RectangularReduction = night.RectangularReduction
advance = pilot.advance
validate_rows = night.validate_rows


def configurations(pair):
    return [item for item in pilot.configurations(pair) if item[0] == 'middle']


def design_header():
    base = robustness.design_header('G2c')
    return dict(base, stage=STAGE, design_version=2, record_days=DAYS,
        parameter_grid_counts=None, parameter_grid_anchor=pilot.ANCHOR,
        parameter_grids={k: [list(s) for s in v] for k, v in GRIDS.items()},
        numerical_levels=None,
        fv_settings={k: dict(configurations(tuple(k.split('__')))) for k in GRIDS},
        population_starts=list(robustness.POPULATION_STARTS),
        unit_names=[unit_name(u) for u in units()],
        expected_conditional_runs=sum(int(np.prod(GRIDS['__'.join(u[0])][-1])) for u in units()),
        expected_record_scenarios=len(units()),
        prerequisite='Complete raw-validated passing rectangular G2b-v2, all 70 controls',
        design_origin='Registered before G2c data; candidate grids and middle FV settings from G2b-v2',
        interpretation='200 matched records per pair and 20 paired robustness records per pair; '
                       'pilot-supported numerics, no per-record convergence proof, recovery pass target or forecast acceptance.')


def make_group(unit, truth=None):
    if unit not in units():
        raise ValueError('Unregistered rectangular G2c unit.')
    pair, sid, scenario, label = unit
    case, sequence, record = truth if truth is not None else grid._g2a_truth_record(pair, sid, DAYS)
    if not record['passes']:
        raise ValueError('G2c truth gate failed; do not replace the seed.')
    case, observations, indices = robustness._scenario(case, sequence.observations, scenario)
    counts = GRIDS['__'.join(pair)]
    return dict(stage=STAGE, pair=pair, sequence_id=sid, label=label, case=case,
        observations=observations, hypotheses=pilot.parameter_grid(pair, counts[-1])[0],
        counts=counts, settings=dict(configurations(pair))[label], prior_bridge=False,
        backend='cached', array_output=True, keep_demand=True, scenario=scenario,
        truth_record=record, observation_indices=indices.tolist())


def summarize(rows):
    # Imported lazily to keep the shared compact-record helper in one place.
    from src.validation.bm_1.joint_partition import compact_record, summary_records
    records = [compact_record([row], sys.modules[__name__]) for row in rows]
    return summary_records('G2c', records, [])
