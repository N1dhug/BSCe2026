"""Bounded conditional H2 pilot; metadata declarations are not scientific proof.

Only the complete declared TRAIN stream enters the likelihood. Baseline values
are supplied separately with their frozen provenance. This runner never fits a
baseline, chooses a population, infers units, or evaluates a holdout.
"""

import csv
from functools import partial
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy.special import betainc

from src.control.bm_1 import BM1Parameters
from src.method.bm_1_response import (
    curve_pair_from_logits, fit_shared_curve_pair, pooled_curve_log_likelihood,
)
from src.params.bm_1_history import BM1ChronologicalSplit, BM1HistoricalCurveContract
from src.params.bm_1_response import BM1PiecewiseResponse
from src.tools.bm_1_data import prepare_historical_curve_training
from src.tools.bm_1_history_io import load_historical_training_workbooks
from src.tools.bm_1_reports import _RunLock, _atomic_report, _read_likelihood_report
from src.validation import estimation as checks


EVIDENCE = ('population', 'observations', 'timezone', 'price', 'scale', 'baseline', 'exposure')


def configuration_template(*, hourly_assumptions=False):
    """Intentionally incomplete: no invented scientific metadata or parameters."""
    config = dict(schema='bm1-h2-pilot-config-v1', evidence={k: None for k in EVIDENCE},
        files=dict(input_workbook='files/input/data/input.xlsx',
                   price_workbook='files/input/data/prices.xlsx', baseline_csv=None),
        timezone=None, price_area=None, origin=None, baseline_fitted_before=None,
        model_selection_protocol=None,
        contract=dict(population_id=None, measurement_columns=[],
            split=dict(train_start=None, selection_start=None, holdout_start=None,
                holdout_end=None, exposure_intervals=[], holdout_audit_reference=None),
            power_unit=None, power_scale=None, price_low=None, price_high=None,
            price_unit=None, price_role=None, observation_kind=None,
            metadata_reference=None, baseline_reference=None, scale_fitted_before=None),
        parameters=dict(C=None, k=None, Xi=None, sigma_x=None, sigma_y=None, time_unit='day'),
        prior=dict(alpha=None, beta=None),
        numerics=dict(cells=401, max_step=1/768, quadrature_order=3, backend='fitted'),
        fit=dict(regularization=.1, max_iterations=40, logit_bound=4.,
            families=[dict(pieces=2, starts=[[0., 0.]], max_evaluations=24),
                      dict(pieces=4, starts=[[0.]*6], max_evaluations=48)]))
    if hourly_assumptions:
        config['timezone'] = 'Europe/Copenhagen'
        config['contract'].update(power_unit='kW', measurement_unit='kWh',
            observation_kind='interval_energy', timestamp_label='hour_start',
            metadata_status='working_assumptions')
        config['evidence']['observations'] = ('Johan, 5 October 2026: one-hour energy, '
            'provisionally kWh; hour 03 means [03:00,04:00). Working assumption, not verified meter documentation.')
        config['evidence']['timezone'] = ('Denmark; Europe/Copenhagen is a working assumption. '
            'Ambiguous/nonexistent naive DST labels still fail; no silent inference.')
        config['numerics'].update(accumulator_spacing=1/512, max_accumulator_nodes=4096)
    return config


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Document {name} before reading observations or fitting.')
    return value


def validate_configuration(config):
    """Validate all declarations/settings before any observation-file access."""
    if config.get('schema') != 'bm1-h2-pilot-config-v1':
        raise ValueError('Unknown H2 pilot configuration.')
    for key in EVIDENCE:
        _text(config['evidence'][key], f'evidence.{key}')
    _text(config['model_selection_protocol'], 'model_selection_protocol')
    for key in ('timezone', 'price_area'):
        _text(config[key], key)
    if config['price_area'] not in ('DK1', 'DK2'):
        raise ValueError('Explicitly select DK1 or DK2.')
    declared = dict(config['contract'])
    split = BM1ChronologicalSplit(**declared.pop('split'))
    split.require_blind_holdout()
    contract = BM1HistoricalCurveContract(split=split, **declared)
    if contract.price_role != 'spot_proxy' or contract.price_unit != 'DKK/MWh':
        raise ValueError('This workbook runner requires spot_proxy in DKK/MWh.')
    origin = pd.Timestamp(config['origin'])
    fitted = pd.Timestamp(config['baseline_fitted_before'])
    if (pd.isna(origin) or origin.tzinfo is None or origin >= split.train_start
            or pd.isna(fitted) or fitted.tzinfo is None or fitted > origin):
        raise ValueError('Declare an earlier prior origin and baseline fitted no later than origin.')
    for key in ('input_workbook', 'price_workbook', 'baseline_csv'):
        _text(config['files'][key], f'files.{key}')
    parameters = dict(config['parameters'])
    if parameters.pop('time_unit') != 'day':
        raise ValueError('Physical/noise parameters must explicitly use day units.')
    sigma_y = checks.finite_scalar(parameters.pop('sigma_y'), 'sigma_y', positive=True)
    linear = BM1PiecewiseResponse.linear()
    parameters = BM1Parameters(linear, linear, **parameters)
    prior = {k: checks.finite_scalar(config['prior'][k], k, positive=True) for k in ('alpha', 'beta')}
    numerics = dict(config['numerics'])
    numerics['cells'] = checks.positive_integer(numerics['cells'], 'cells', minimum=3)
    numerics['quadrature_order'] = checks.positive_integer(numerics['quadrature_order'], 'quadrature_order')
    numerics['max_step'] = checks.finite_scalar(numerics['max_step'], 'max_step', positive=True)
    if numerics['backend'] != 'fitted':
        raise ValueError('The H2 pilot uses the fitted FV backend; declare its refinements separately.')
    if contract.observation_kind != 'point':
        numerics['accumulator_spacing'] = checks.finite_scalar(
            numerics.get('accumulator_spacing'), 'accumulator_spacing', positive=True)
        numerics['max_accumulator_nodes'] = checks.positive_integer(
            numerics.get('max_accumulator_nodes'), 'max_accumulator_nodes', minimum=2)
    elif 'accumulator_spacing' in numerics or 'max_accumulator_nodes' in numerics:
        raise ValueError('Accumulator settings apply only to interval observations.')
    fit = dict(config['fit'])
    fit['regularization'] = checks.finite_scalar(fit['regularization'], 'regularization')
    fit['logit_bound'] = checks.finite_scalar(fit['logit_bound'], 'logit_bound', positive=True)
    fit['max_iterations'] = checks.positive_integer(fit['max_iterations'], 'max_iterations')
    if fit['regularization'] < 0. or fit['logit_bound'] > 10.:
        raise ValueError('Require nonnegative regularization and logit_bound <=10.')
    if [f['pieces'] for f in fit['families']] != [2, 4]:
        raise ValueError('Declare the two-piece and four-piece families, in that order.')
    for family in fit['families']:
        checks.positive_integer(family['max_evaluations'], 'max_evaluations')
        starts = np.asarray(family['starts'], dtype=float)
        if (starts.ndim != 2 or not len(starts) or starts.shape[1] != 2*(family['pieces']-1)
                or not np.isfinite(starts).all() or np.any(np.abs(starts) > fit['logit_bound'])):
            raise ValueError('Declare finite starts inside the curve-family bounds.')
    return contract, origin, parameters, sigma_y, prior, numerics, fit


def _digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _baseline_window(path, index):
    """Select timestamps before decoding baseline values; never interpolate."""
    rows = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['timestamp', 'baseline']:
            raise ValueError('Baseline CSV needs exactly timestamp,baseline columns.')
        for row in reader:
            stamp = pd.Timestamp(row['timestamp'])
            if pd.isna(stamp) or stamp.tzinfo is None:
                raise ValueError('Baseline timestamps must be timezone-aware.')
            if not index[0] <= stamp <= index[-1]:
                continue
            if stamp not in index or stamp in rows:
                raise ValueError('Baseline has duplicate or off-grid selected timestamps.')
            rows[stamp] = checks.finite_scalar(row['baseline'], 'baseline')
    if set(rows) != set(index):
        raise ValueError('Supply the complete baseline from prior origin through TRAIN.')
    return pd.Series([rows[t] for t in index], index=index, dtype=float)


def _beta_mass(edges, *, alpha, beta):
    return np.diff(betainc(alpha, beta, edges))


def prepare_training(config, *, root):
    contract, origin, parameters, sigma_y, prior, numerics, fit = validate_configuration(config)
    paths = {k: (Path(root)/v).resolve() for k, v in config['files'].items()}
    before = {k: _digest(p) for k, p in paths.items()}
    measurements, prices, raw = load_historical_training_workbooks(
        paths['input_workbook'], paths['price_workbook'], contract=contract,
        origin=origin, timezone=config['timezone'], price_area=config['price_area'])
    baseline = _baseline_window(paths['baseline_csv'], prices.index)
    record, prepared = prepare_historical_curve_training(measurements, baseline, prices,
        contract=contract, parameters=parameters, sigma_y=sigma_y,
        initial_mass_builder=partial(_beta_mass, **prior), origin=origin)
    if before != {k: _digest(p) for k, p in paths.items()}:
        raise ValueError('Input files changed while preparing TRAIN.')
    return record, dict(raw=raw, prepared=prepared, input_sha256=before,
        baseline_fitted_before=config['baseline_fitted_before'],
        evidence=config['evidence'], model_selection_protocol=config['model_selection_protocol']), numerics, fit, paths


def _validate_saved_trace(trace, record, family, fit):
    """Reject corrupted/relabelled checkpoints without new numerical work."""
    if not isinstance(trace, list) or len(trace) > family['max_evaluations']:
        raise ValueError('Saved candidate trace exceeds the declared budget.')
    seen = set()
    for row in trace:
        logits = checks.finite_vector(row['logits'], 'saved logits')
        key = tuple(logits)
        if key in seen or np.any(np.abs(logits) > fit['logit_bound']):
            raise ValueError('Duplicate/out-of-bounds saved curve candidate.')
        seen.add(key)
        f, g = curve_pair_from_logits(logits, family['pieces'])
        penalty = fit['regularization']*sum(float(np.sum((c.values[1:-1]-(1-2*c.knots[1:-1]))**2)) for c in (f, g))
        scores = row['record_log_likelihoods']
        if set(scores) != {record.record_id}:
            raise ValueError('Saved curve population differs from TRAIN.')
        score = checks.finite_scalar(scores[record.record_id], 'saved score')
        for actual, expected in ((row['log_likelihood'], score), (row['penalty'], penalty),
                                 (row['objective'], -score+penalty)):
            if abs(checks.finite_scalar(actual, 'saved scalar')-expected) > 1e-10:
                raise ValueError('Saved candidate objective is inconsistent.')
        error = checks.finite_scalar(row['maximum_abs_mass_error'], 'saved mass error')
        if not 0. <= error <= 1e-10:
            raise ValueError('Saved probability guard failed.')


def _validate_replay(replay, record, numerics, expected_score=None):
    rows = replay['records']
    if len(rows) != 1 or rows[0]['record_id'] != record.record_id:
        raise ValueError('Saved H2 replay population changed.')
    row = rows[0]
    observed = np.asarray(row['observed'])
    if observed.dtype.kind != 'b' or not np.array_equal(observed, record.observations.observed):
        raise ValueError('Saved H2 replay observation mask changed.')
    values = row['log_evidence']
    if len(values) != len(observed) or any(v is not None for v, seen in zip(values, observed) if not seen):
        raise ValueError('Saved H2 missing observation scores are invalid.')
    scores = checks.finite_vector([v for v, seen in zip(values, observed) if seen], 'saved evidence')
    total = float(scores.sum())
    expected = [row['log_likelihood'], replay['log_likelihood']]
    if expected_score is not None:
        expected.append(expected_score)
    if any(abs(checks.finite_scalar(v, 'saved likelihood')-total) > 1e-10 for v in expected):
        raise ValueError('Saved H2 replay likelihood differs from its evidence/candidate.')
    mass = checks.probability_mass(row['final_posterior_mass'], 'saved posterior')
    edges = checks.increasing_grid(row['x_edges'], 'saved edges')
    error = checks.finite_scalar(row['maximum_abs_mass_error'], 'saved mass error')
    minimum = checks.finite_scalar(row['minimum_cell_mass'], 'saved minimum mass')
    if (not np.array_equal(edges, np.linspace(0., 1., numerics['cells']+1))
            or len(mass) != numerics['cells'] or not 0. <= error <= 1e-10 or minimum < 0.):
        raise ValueError('Saved H2 replay probability/grid guard failed.')


def run_pilot(args):
    """Atomic per-candidate checkpoints; time stop is a resumable pause."""
    from src.exe.run_model.bm_1.curves import source_snapshot
    from src.exe.run_model.progress import runtime_text
    from src.validation.bm_1.response import _score_record
    begun = perf_counter()
    seconds = checks.finite_scalar(args.max_seconds, 'max_seconds', positive=True)
    limit = None if args.limit_candidates is None else checks.positive_integer(args.limit_candidates, 'limit_candidates')
    config_path = Path(args.config).resolve()
    config_hash = _digest(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_configuration(config)  # Fail closed before observation-file access.
    source = source_snapshot()
    if source['dirty']:
        raise ValueError('Run H2 on a clean committed source; do not change source during a run.')
    record, metadata, numerics, fit, paths = prepare_training(config, root=args.input_root)
    source_match = dict(schema='bm1-h2-pilot-manifest-v1', source=source,
        numpy_version=np.__version__, scipy_version=scipy.__version__, pandas_version=pd.__version__,
        config_sha256=config_hash, config=config, training=metadata,
        declared_distinct_candidate_budget=1+sum(f['max_evaluations'] for f in fit['families']),
        interpretation='Conditional TRAIN pilot. No model selection, numerical acceptance, H3 or forecast acceptance.')

    def unchanged():
        if (source_snapshot() != source or _digest(config_path) != config_hash
                or metadata['input_sha256'] != {k: _digest(p) for k, p in paths.items()}):
            raise ValueError('Source/config/input changed; no H2 checkpoint published.')

    directory = Path(args.run_dir)
    with _RunLock(directory):
        manifest_path = directory/'manifest.json'
        if manifest_path.exists():
            if not args.resume:
                raise FileExistsError('Existing H2 pilot requires --resume or a new run directory.')
            if _read_likelihood_report(manifest_path)[0] != source_match:
                raise ValueError('H2 resume source, configuration or input fingerprint differs.')
        else:
            if args.resume:
                raise FileNotFoundError('No H2 manifest to resume.')
            if any(directory.glob('*.json*')):
                raise ValueError('Unregistered H2 files exist without a manifest.')
            unchanged()
            _atomic_report(manifest_path, source_match)
        if args.prepare_only:
            print(f'H2 TRAIN prepared: {metadata["prepared"]["training_rows"]} rows; no likelihood evaluated.', flush=True)
            return dict(prepared=True, fitting_run=False, scientific_acceptance=False)
        new = 0
        reports = []
        paused = False
        reference_path = directory/'linear.json.gz'
        if reference_path.exists():
            reference = _read_likelihood_report(reference_path)[0]
            _validate_replay(reference, record, numerics)
        elif perf_counter()-begun < seconds and not (directory/'STOP').exists():
            linear = BM1PiecewiseResponse.linear()
            reference = _score_record(pooled_curve_log_likelihood((record,), linear, linear, **numerics))
            unchanged()
            _atomic_report(reference_path, reference)
            new += 1
        else:
            reference = None
            paused = True
        for family in fit['families']:
            pieces = family['pieces']
            trace_path = directory/f'{pieces}_trace.json.gz'
            result_path = directory/f'{pieces}_result.json.gz'
            trace = _read_likelihood_report(trace_path)[0] if trace_path.exists() else []
            _validate_saved_trace(trace, record, family, fit)
            if result_path.exists():
                saved = _read_likelihood_report(result_path)[0]
                if not trace or saved['trace'] != trace or saved['paused'] is not False:
                    raise ValueError('Saved H2 fit and candidate trace disagree.')
                best = min(trace, key=lambda row: row['objective'])
                f, g = curve_pair_from_logits(best['logits'], pieces)
                if (saved['pieces'] != pieces or saved['logits'] != best['logits']
                        or saved['f'] != f.to_record() or saved['g'] != g.to_record()
                        or saved['objective'] != best['objective'] or saved['log_likelihood'] != best['log_likelihood']
                        or saved['starts'] != family['starts'] or saved['max_evaluations'] != family['max_evaluations']
                        or saved['regularization'] != fit['regularization'] or saved['backend'] != numerics['backend']
                        or saved['distinct_evaluations'] != len(trace)
                        or reference is None or saved['linear_log_likelihood'] != reference['log_likelihood']):
                    raise ValueError('Saved H2 result differs from selected candidate.')
                _validate_replay(saved['replay'], record, numerics, best['log_likelihood'])
                reports.append(saved)
                continue
            if (paused or perf_counter()-begun >= seconds or (directory/'STOP').exists()
                    or limit is not None and new >= limit):
                paused = True
                break
            clock = perf_counter()
            resumed = len(trace)
            has_candidate = [bool(trace)]

            def checkpoint(rows):
                unchanged()
                _atomic_report(trace_path, rows, replace_existing=True)
                has_candidate[0] = True

            def progress(done, total):
                if not args.quiet:
                    print(f'H2 {pieces} pieces: {done}/{total}; '+runtime_text(clock, done, total, initial_completed=resumed), flush=True)

            result = fit_shared_curve_pair((record,), **family, **{k:v for k,v in fit.items() if k != 'families'},
                **numerics, resume_trace=trace, checkpoint=checkpoint, progress=progress,
                max_new_evaluations=None if limit is None else limit-new,
                should_pause=lambda: has_candidate[0] and (perf_counter()-begun >= seconds or (directory/'STOP').exists()))
            new += result['new_evaluations']
            saved = {k:v for k,v in result.items() if k not in ('f', 'g', 'logits', 'starts', 'replay')}
            saved.update(f=result['f'].to_record(), g=result['g'].to_record(),
                logits=result['logits'].tolist(), starts=result['starts'].tolist(),
                replay=_score_record(result['replay']), linear_log_likelihood=reference['log_likelihood'],
                numerical_acceptance=False, scientific_acceptance=False)
            reports.append(saved)
            if result['paused']:
                paused = True
                break
            unchanged()
            _atomic_report(result_path, saved)
        summary = dict(schema='bm1-h2-pilot-summary-v1', completed=len(reports)==2 and not paused,
            paused=paused, new_candidate_evaluations=new, families=reports,
            elapsed_seconds=perf_counter()-begun, scientific_acceptance=False,
            interpretation='TRAIN scores do not choose a winner. Selected curves still require numerical refinements and H3.',
            budget_scope='Distinct objective candidates, including the linear reference. Restored best-candidate replays are reported separately in each family; no extra optimizer candidates.',
            time_stop_scope='Pause between complete candidates; a running candidate and an existing fitter best-replay restoration may finish after the wall-clock deadline.')
        unchanged()
        _atomic_report(directory/'summary.json', summary, replace_existing=True)
    return summary
