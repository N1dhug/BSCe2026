"""Old-record shared-curve development pilot; not H scientific acceptance."""

from time import perf_counter

import numpy as np

from src.control.bm_1 import beta_initial_cell_mass
from src.method.bm_1_response import (CurveTrainingRecord, curve_pair_logits,
    fit_shared_curve_pair, pooled_curve_log_likelihood)
from src.params.bm_1_response import BM1PiecewiseResponse
from src.validation.bm_1 import estimation as synthetic


def _initial_mass(edges):
    return beta_initial_cell_mass(edges, alpha=20., beta=20.)


def _score_record(score):
    return dict(log_likelihood=score['log_likelihood'], records=[dict(
        record_id=r['record_id'], log_likelihood=r['log_likelihood'],
        log_evidence=[float(v) if seen else None for v,seen in zip(r['log_evidence'],r['observed'])],
        observed=r['observed'].tolist(),
        x_edges=r['final_step'].x_edges.tolist(),
        final_posterior_mass=r['final_step'].posterior_mass.tolist(),
        maximum_abs_mass_error=r['maximum_abs_mass_error'],
        minimum_cell_mass=r['minimum_cell_mass']) for r in score['records']])


def run_curve_fit_pilot(*, progress=None):
    """Frozen development specification: two previously used F3 records.

    Fits are conditional on known C/k/Xi/noise, with a shared beta initial law.
    Linear, two and four pieces are compared descriptively on TRAIN. There is
    no model selection or holdout; lambda/budget/starts are fixed before replay.
    """
    begun = perf_counter()
    case = synthetic.build_f4_b_case()
    records, observed = [], []
    for sid in (1326,1327):
        sequence = synthetic.generate_synthetic_sequence(case, sequence_id=sid)
        if not sequence.truth.valid:
            raise RuntimeError('Development truth violated bounds; no record dropped/repaired.')
        model, parameters = case.build_model()
        records.append(CurveTrainingRecord(str(sid), 'TRAIN', model.B, case.horizon,
            parameters, case.sigma_x, case.sigma_y, _initial_mass,
            sequence.observations, case.breakpoints))
        observed.append(dict(record_id=str(sid), role='TRAIN',
                             times=sequence.observations.times.tolist(),
                             values=sequence.observations.values.tolist(),
                             observed=sequence.observations.observed.tolist()))
    records = tuple(records)
    linear = BM1PiecewiseResponse.linear()
    baseline = pooled_curve_log_likelihood(records,linear,linear)
    budget, regularization = 250, .1
    two_starts = np.array([[0.,0.],[.8,-.8]])
    def callback(offset):
        return None if progress is None else lambda done,total: progress(offset+done,2*budget)
    two = fit_shared_curve_pair(records,pieces=2,starts=two_starts,
        regularization=regularization,max_evaluations=budget,progress=callback(0))
    four_starts = np.array([curve_pair_logits(two['f'],two['g'],pieces=4),np.zeros(6)])
    four = fit_shared_curve_pair(records,pieces=4,starts=four_starts,
        regularization=regularization,max_evaluations=budget,progress=callback(two['distinct_evaluations']))
    fitted = {'two':two,'four':four}
    geometries = dict(linear=(linear,linear),two=(two['f'],two['g']),four=(four['f'],four['g']))
    controls = {}
    for name,(f,g) in geometries.items():
        base = baseline if name=='linear' else fitted[name]['replay']
        rows = {}
        for axis,settings in (('time',dict(max_step=1/1536)),
                              ('spatial',dict(cells=801)),('quadrature',dict(quadrature_order=5))):
            score = pooled_curve_log_likelihood(records,f,g,**settings)
            rows[axis] = dict(**_score_record(score),
                             likelihood_difference_from_base=score['log_likelihood']-base['log_likelihood'])
        controls[name] = rows
    fits = {}
    for name,result in fitted.items():
        fits[name] = dict(f=result['f'].to_record(),g=result['g'].to_record(),
            logits=result['logits'].tolist(),objective=result['objective'],
            log_likelihood=result['log_likelihood'],
            training_likelihood_gain=result['log_likelihood']-baseline['log_likelihood'],
            budget_exhausted=result['budget_exhausted'],distinct_evaluations=result['distinct_evaluations'],
            starts_attempted=result['starts_attempted'],starts_completed=result['starts_completed'],
            starts=result['starts'].tolist(),optimizer_results=result['optimizer_results'],
            elapsed_seconds=result['elapsed_seconds'],trace=result['trace'],selected_replay=_score_record(result['replay']))
    return dict(stage='H-shared-curves-development-v1', registered_design=False,
        scientific_acceptance=False, interpretation='Pooled conditional TRAIN fit and local budget on old F3 data, not model selection, nonlinear-truth validation, historical fit or holdout acceptance.',
        data=dict(records=observed,time_unit='day',observation_kind='point',price_side='right',
                  baseline=case.baseline,price_edges=case.price_edges.tolist(),prices=case.price_values.tolist(),
                  independent_initializations=True,initial_beta=[20.,20.],
                  physical_parameters=dict(C=case.C,k=case.k,Xi=case.Xi,sigma_x=case.sigma_x,sigma_y=case.sigma_y)),
        design=dict(regularization=regularization,penalty='Sum squared inner-knot deviation from 1-2z; lambda fixed before fit.',
                    optimizer='L-BFGS-B on positive-drop logits',logit_bound=4.,max_iterations=40,
                    optimizer_options=dict(ftol=1e-9,gtol=1e-5),
                    candidate_budget_per_family=budget,exact_flat_pieces_in_family=False,
                    cells=401,max_step=1/768,likelihood_quadrature_order=3,
                    holdout_records_used=0,curve_pair_shared_across_records=True),
        linear_reference=_score_record(baseline),fits=fits,numerical_diagnostics=controls,
        conditional_training_replays=2*(1+two['distinct_evaluations']+four['distinct_evaluations']+9),
        elapsed_seconds=perf_counter()-begun)
