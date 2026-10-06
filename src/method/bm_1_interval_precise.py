"""F4-H2 precise, resumable whole-TRAIN replay; experimental fit integration.

Uses the unchanged terminal-tilt kernel. Each candidate owns its log state.
Only the registered hourly-hold adapters are frozen into this portable input.
No historical numerical acceptance or optimizer convergence is implied.
"""

import hashlib
import json
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from src.method.bm_1_response import curve_pair_from_logits
from src.model.bm_1 import bm_1
from src.params.bm_1_interval import BM1IntervalObservationSeries
from src.solver.finite_volume import finite_volume_1d_fitted
from src.tools.input_generator.input_builder import input_builder
from src.validation.bm_1.interval_characteristic import (
    CharacteristicNumericalError, balanced_characteristic_prediction)
from src.validation.bm_1.interval_terminal_tilt import terminal_tilt_update


PRECISE_SETTINGS = dict(cells=1921, max_step=1/24576, frequencies=64, span=2.,
    centers=[.5, .25, .75, 0., 1.], relative_guard=1e-8, overlap_log_tolerance=1e-7)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def candidate_panel(step=.5):
    """One linear reference plus +/- each of 2 and 6 independent drop logits.

    Exactly 17 fixed candidates; a single coordinate probe, not a converged fit.
    The shared linear reference is not re-evaluated for each curve family.
    """
    if not np.isfinite(step) or not 0. < step <= 4.:
        raise ValueError('Declare a positive logit step <=4.')
    rows = [dict(id='linear', pieces=2, logits=[0., 0.])]
    for pieces in (2, 4):
        for coordinate in range(2*(pieces-1)):
            for sign in (1, -1):
                logits = [0.] * (2*(pieces-1)); logits[coordinate] = sign*float(step)
                rows.append(dict(id=f'p{pieces}_c{coordinate}_{"plus" if sign == 1 else "minus"}',
                                 pieces=pieces, logits=logits))
    return rows


def freeze_hourly_record(record, settings=PRECISE_SETTINGS):
    """Freeze a record produced by the historical or saved-development adapter."""
    if not isinstance(record.observations, BM1IntervalObservationSeries) or record.observations.time_unit != 'day':
        raise ValueError('Precise H2 requires the explicit interval-average/day contract.')
    edges = np.r_[record.horizon[0], record.breakpoints, record.horizon[-1]]
    left, right = edges[:-1], edges[1:]
    b, u = np.asarray(record.baseline(left)), np.asarray(record.parameters[0](left))
    # A sanity check on the registered hourly-hold callbacks, not a proof for
    # arbitrary time-dependent functions. Callers only use the two named adapters.
    for fraction in (.25, .75):
        if (not np.array_equal(record.baseline(left+fraction*(right-left)), b)
                or not np.array_equal(record.parameters[0](left+fraction*(right-left)), u)):
            raise ValueError('Only frozen time-held B/U inputs are supported here.')
    grid, _, _ = finite_volume_1d_fitted().grid(settings['cells'])
    initial = np.asarray(record.initial_mass_builder(grid), dtype=float)
    with np.errstate(divide='ignore'):
        logs = np.log(initial)
    obs = record.observations
    return dict(record_id=record.record_id, horizon=list(record.horizon),
        input_edges=edges.tolist(), baseline=b.tolist(), price=u.tolist(),
        physical=dict(C=record.parameters[3], k=record.parameters[4], Xi=record.parameters[5],
                      sigma_x=record.sigma_x, sigma_y=record.sigma_y),
        starts=obs.starts.tolist(), ends=obs.ends.tolist(), observed=obs.observed.tolist(),
        values=[float(y) if seen else None for y, seen in zip(obs.values, obs.observed)],
        initial_log_mass=[None if np.isneginf(x) else float(x) for x in logs],
        settings=dict(settings), time_unit='day', midnight_resets=0)


def replay_units(spec):
    """Include pre-TRAIN evolution and gaps, split only their prediction units."""
    observations = BM1IntervalObservationSeries(spec['starts'], spec['ends'],
        [np.nan if y is None else y for y in spec['values']], 'day', spec['observed'])
    if spec['time_unit'] != 'day' or spec['midnight_resets'] != 0:
        raise ValueError('Never change time units or reset daily state.')
    edges = np.asarray(spec['input_edges'], dtype=float)
    if (np.any(np.diff(edges) <= 0.) or edges[0] != spec['horizon'][0]
            or edges[-1] != spec['horizon'][-1] or len(edges)-1 != len(spec['baseline'])
            or len(spec['baseline']) != len(spec['price'])
            or observations.ends[-1] != edges[-1] or observations.starts[0] < edges[0]):
        raise ValueError('Frozen inputs must cover the whole declared TRAIN and prior lead.')
    units, current = [], float(edges[0])
    for index, (start, end, seen, value) in enumerate(zip(observations.starts,
            observations.ends, observations.observed, observations.values)):
        gap_edges = np.r_[current, edges[(edges > current) & (edges < start)], start]
        for a, z in zip(gap_edges[:-1], gap_edges[1:]):
            if z > a:
                units.append(dict(start=float(a), end=float(z), observed=False,
                                  value=None, observation_index=None))
        units.append(dict(start=float(start), end=float(end), observed=bool(seen),
                          value=float(value) if seen else None, observation_index=index))
        current = float(end)
    return units


def validate_checkpoint(saved, spec, candidate):
    """Reject changed candidates/inputs and corrupt evidence before state reuse."""
    identity = digest(dict(spec=spec, candidate=candidate))
    if saved['identity'] != identity or saved['checksum'] != digest(saved['payload']):
        raise ValueError('Checkpoint candidate/input identity or checksum changed.')
    payload, units = saved['payload'], replay_units(spec)
    rows = payload['rows']
    if len(rows) > len(units):
        raise ValueError('Checkpoint exceeds TRAIN.')
    for index, row in enumerate(rows):
        if row['unit'] != index or any(row[k] != units[index][k] for k in units[index]):
            raise ValueError('Checkpoint chronology or observation mask changed.')
        if (row['observed'] != (row['log_evidence'] is not None)
                or not np.isfinite(row['state_mean']) or row['maximum_abs_mass_error'] > 1e-10):
            raise ValueError('Invalid saved evidence or probability guard.')
    score = sum(r['log_evidence'] for r in rows if r['observed'])
    if score != payload['log_likelihood'] or not np.isfinite(score):
        raise ValueError('Checkpoint cumulative evidence changed.')
    logs = np.array([-np.inf if x is None else x for x in payload['log_mass']])
    if (logs.shape != (spec['settings']['cells'],) or np.isnan(logs).any()
            or np.isposinf(logs).any() or abs(float(logsumexp(logs))) > 1e-10):
        raise ValueError('Invalid checkpoint log state; finite tails must be retained.')
    if not rows and payload['log_mass'] != spec['initial_log_mass']:
        raise ValueError('The declared initial law changed.')
    return payload


def replay_precise_candidate(spec, candidate, *, saved=None, limit_units=None,
                             should_pause=None, checkpoint=None):
    """Whole-stream candidate; hourly checkpoints own the actual log posterior."""
    settings = dict(spec['settings']); cells = settings.pop('cells')
    f, g = curve_pair_from_logits(candidate['logits'], candidate['pieces'])
    units = replay_units(spec)
    payload = (dict(rows=[], log_mass=spec['initial_log_mass'], log_likelihood=0., status='paused')
               if saved is None else dict(validate_checkpoint(saved, spec, candidate)))
    if payload.get('status') == 'numerical_guard_failed':
        raise ValueError('Preserve the failed candidate; no automatic numerical retry.')
    rows = list(payload['rows'])
    logs = np.array([-np.inf if x is None else x for x in payload['log_mass']])
    if not rows:
        validate_checkpoint(dict(identity=digest(dict(spec=spec, candidate=candidate)),
            payload=payload, checksum=digest(payload)), spec, candidate)
    B = input_builder.price_intervals((spec['input_edges'], spec['baseline']))
    U = input_builder.price_intervals((spec['input_edges'], spec['price']))
    p = spec['physical']; parameters = (U, f, g, p['C'], p['k'], p['Xi'])
    model = bm_1(B, B, spec['horizon'], 2)
    solver = finite_volume_1d_fitted(vectorized_inputs=True)
    model.fokker_planc_reference = solver
    _, states, _ = solver.grid(cells)
    identity = digest(dict(spec=spec, candidate=candidate))
    def publish(status, **extra):
        nonlocal payload
        payload = dict(rows=rows, log_mass=[None if np.isneginf(x) else float(x) for x in logs],
            log_likelihood=float(sum(r['log_evidence'] for r in rows if r['observed'])),
            status=status, **extra)
        result = dict(identity=identity, checksum=digest(payload), payload=payload)
        if checkpoint is not None:
            checkpoint(result)
        return result
    stop = len(units) if limit_units is None else min(len(units), limit_units)
    for index in range(len(rows), stop):
        if should_pause is not None and should_pause():
            return publish('paused')
        unit = units[index]; start, end = unit['start'], unit['end']
        breaks = np.asarray(spec['input_edges'])[1:-1]
        breaks = breaks[(breaks > start) & (breaks < end)]
        drift, diffusion, options = model._fv_evolution_fields(parameters, p['sigma_x'],
            np.array([start, end]), breaks, settings['max_step'])
        reward = lambda t, x: model.eval(t, x, U, f, g, p['k'], p['Xi'])[-1]
        begun = perf_counter()
        try:
            if unit['observed']:
                result = terminal_tilt_update(solver, [start, end], np.exp(logs), drift, diffusion,
                    reward, unit['value'], p['sigma_y'], log_prior=logs, breakpoints=breaks,
                    **settings, **options)
                next_logs = result['log_posterior_mass']; evidence = float(result['log_evidence'])
                diagnostics = {k: result[k] for k in ('contours', 'maximum_relative_conditional_guard',
                    'maximum_overlap_log_difference', 'maximum_abs_mass_error')}
            else:
                result = balanced_characteristic_prediction(solver, [start, end], np.exp(logs),
                    drift, diffusion, lambda t, x: np.zeros_like(x), log_prior=logs,
                    max_step=settings['max_step'], frequencies=1, breakpoints=breaks,
                    state_scheme='positive_crank_nicolson', **options)
                next_logs = result['log_amplitude']; evidence = None
                diagnostics = dict(contours=[], maximum_relative_conditional_guard=0.,
                    maximum_overlap_log_difference=0., maximum_abs_mass_error=result['maximum_abs_mass_error'])
        except (CharacteristicNumericalError, ValueError) as exc:
            return publish('numerical_guard_failed', failed_unit=index, reason=str(exc),
                           diagnostics=getattr(exc, 'diagnostics', {}))
        logs = next_logs
        rows.append(dict(unit=index, **unit, log_evidence=evidence,
            state_mean=float(np.exp(logs) @ states), elapsed_seconds=perf_counter()-begun,
            **diagnostics))
        publish('completed' if len(rows) == len(units) else 'paused')
    return publish('completed' if len(rows) == len(units) else 'prefix_complete')
