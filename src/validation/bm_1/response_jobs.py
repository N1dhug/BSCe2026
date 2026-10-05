"""Spawn-safe independent H1 numerical jobs; parent owns publication and locks."""

from src.validation.bm_1.response_cases import h1_training_records, h1_candidate_panel
from src.validation.bm_1.response_numerics import run_numerical_unit


def run_h1_job(spec):
    records, _ = h1_training_records(spec['case'])
    record = records[spec['record_index']]
    candidates = {name: (f, g) for name, f, g in h1_candidate_panel()}
    f, g = candidates[spec['candidate']]
    if spec['unit'] != f"{spec['case']}_{record.record_id}_{spec['candidate']}":
        raise ValueError('H1 task identity mismatch.')
    return run_numerical_unit(record, f, g, numerics=spec['numerics'], backend=spec['backend'])
