"""Model-independent five-week training / sixth-week forecast orchestration.

Only calendar, data isolation, persistence and scoring are shared. Baselines,
likelihoods, state initialization, numerical gates and fit budgets belong to
the adapter. No fitting algorithm or automatic retry lives in this module.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from src.tools.bm_1_reports import _atomic_report, _RunLock

PROTOCOL = 'five-weeks-by-weekday-backtest-v1'


def _monday(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize() or stamp.weekday() != 0:
        raise ValueError('Declare a local-naive Monday at midnight.')
    return stamp


@dataclass(frozen=True)
class WeekWindow:
    """Local calendar edges; physical hours are generated in the declared zone."""
    origin: pd.Timestamp
    zone: str = 'Europe/Copenhagen'

    def __post_init__(self):
        object.__setattr__(self, 'origin', _monday(self.origin))
        # Validate the zone before any output directory or model work.
        self.origin.tz_localize(self.zone)

    @property
    def start(self):
        return self.origin - pd.DateOffset(weeks=5)

    @property
    def end(self):
        return self.origin + pd.DateOffset(weeks=1)

    def hours(self, start, end):
        return pd.date_range(pd.Timestamp(start).tz_localize(self.zone),
                             pd.Timestamp(end).tz_localize(self.zone), freq='h', inclusive='left')

    def training_days(self, weekday):
        if type(weekday) is not int or not 0 <= weekday <= 6:
            raise ValueError('Weekday must be an integer from 0 (Monday) to 6.')
        target = self.origin + pd.DateOffset(days=weekday)
        return tuple(target - pd.DateOffset(weeks=w) for w in range(5, 0, -1))

    def record(self):
        return {'origin':str(self.origin), 'training_start':str(self.start),
                'training_end_exclusive':str(self.origin), 'test_end_exclusive':str(self.end),
                'zone':self.zone, 'training_hours':len(self.hours(self.start, self.origin)),
                'test_hours':len(self.hours(self.origin, self.end)),
                'weekdays':[{ 'weekday':d, 'training_days':[str(t.date()) for t in self.training_days(d)],
                             'test_day':str((self.origin + pd.DateOffset(days=d)).date())}
                            for d in range(7)]}


def rolling_windows(first_origin, last_origin, *, zone='Europe/Copenhagen'):
    """Inclusive Monday range: weeks 1–6, then 2–7, then 3–8, etc."""
    first, last = _monday(first_origin), _monday(last_origin)
    if last < first:
        raise ValueError('Last origin must not precede first origin.')
    return tuple(WeekWindow(t, zone) for t in pd.date_range(first, last, freq='7D'))


def _frame(frame, columns):
    if (not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing
            or not frame.columns.is_unique or set(columns) - set(frame.columns)):
        raise ValueError('Require unique sorted hourly timestamps and the declared columns.')
    if np.any(frame.index != frame.index.floor('h', ambiguous='infer')):
        raise ValueError('Hour-start observations/inputs required; align end labels before calling.')


def _grid(frame, physical):
    """Preserve aware timestamps; naive autumn ambiguity is deliberately rejected."""
    if frame.index.tz is None:
        index = physical.tz_localize(None)
        if index.has_duplicates:
            raise ValueError('Autumn repeated hours need timezone-aware data; naive labels are ambiguous.')
    else:
        index = physical.tz_convert(frame.index.tz)
    selected = frame.loc[(frame.index >= index[0]) & (frame.index <= index[-1])]
    extra = selected.index.difference(index)
    if len(extra) and selected.loc[extra].notna().any().any():
        raise ValueError('Finite values at nonexistent physical hours cannot be silently discarded.')
    return frame.reindex(index).copy()


@dataclass
class WeekData:
    window: WeekWindow
    history: pd.DataFrame
    inputs: pd.DataFrame
    target_column: str
    input_columns: tuple


def split_week(history, inputs, window, *, target_column='Demand', input_columns=('Price', 'Temp'), warmup_hours=0):
    """Copy five weeks plus INPUT-ONLY warmup; never pass test targets to a fit.

    Missing Y remains NaN. Every physical hour of the declared exogenous inputs
    must exist and be finite. Population aggregation and target normalization
    must be performed separately under the adapter's declared policy.
    """
    if type(warmup_hours) is not int or warmup_hours < 0:
        raise ValueError('Warmup must be a nonnegative integer number of physical hours.')
    columns = tuple(input_columns)
    if not columns or len(set(columns)) != len(columns) or target_column in columns:
        raise ValueError('Declare distinct input columns excluding the target.')
    _frame(history, (target_column, *columns)); _frame(inputs, columns)
    start = window.start.tz_localize(window.zone) - pd.Timedelta(hours=warmup_hours)
    h = _grid(history[[target_column, *columns]], window.hours(window.start, window.origin)
              if warmup_hours == 0 else pd.date_range(start, window.origin.tz_localize(window.zone), freq='h', inclusive='left'))
    x = _grid(inputs[list(columns)], window.hours(window.origin, window.end))
    h = h.apply(pd.to_numeric, errors='raise'); x = x.apply(pd.to_numeric, errors='raise')
    if np.iscomplexobj(h.to_numpy()) or np.iscomplexobj(x.to_numpy()):
        raise ValueError('Real input and target values required.')
    boundary = window.start if h.index.tz is None else window.start.tz_localize(window.zone)
    h.loc[h.index < boundary, target_column] = np.nan
    if np.isinf(h[target_column].to_numpy(float)).any():
        raise ValueError('Historical targets may be finite or NaN, never infinite.')
    if not np.isfinite(h[list(columns)].to_numpy(float)).all() or not np.isfinite(x.to_numpy(float)).all():
        raise ValueError('Complete finite historical and forecast input hours required; no filling.')
    return WeekData(window, h, x, target_column, columns)


@dataclass
class WeekdayData:
    weekday: int
    training_days: tuple
    records: tuple
    context: pd.DataFrame
    inputs: pd.DataFrame
    target_column: str


def split_weekday(data, weekday):
    """Five separate day records, never a concatenated pseudo-continuous stream.

    Context keeps all physical input hours, with targets masked outside the
    selected weekday. F can propagate a continuous state through these gaps;
    G can use its own separate-day warmups. The framework does not reset states.
    """
    days = data.window.training_days(weekday)
    records = tuple(_grid(data.history, data.window.hours(d, d + pd.DateOffset(days=1))) for d in days)
    if any(not np.isfinite(r[data.target_column].to_numpy(float)).any() for r in records):
        raise ValueError('Each of the five matching weekdays needs measured targets; no dropping a day.')
    context = data.history.copy()
    local = context.index if context.index.tz is None else context.index.tz_convert(data.window.zone)
    context.loc[local.weekday != weekday, data.target_column] = np.nan
    day = data.window.origin + pd.DateOffset(days=weekday)
    x = _grid(data.inputs, data.window.hours(day, day + pd.DateOffset(days=1)))
    return WeekdayData(weekday, days, records, context, x, data.target_column)


class WeekdayCallbacks:
    """Usable F/G extension interface: own baseline fit, parameter fit, forecast.

    fit_baseline(day), fit_parameters(day, baseline), predict(day, baseline,
    parameters) are called independently seven times. save_fit writes the
    adapter's frozen fit artifacts. Fit functions must enforce
    their own budgets and convergence/numerical gates, raising on incomplete
    fits. predict must return a raw-unit DataFrame on day.inputs.index.
    """
    def __init__(self, *, name, metadata, prediction_columns, fit_baseline: Callable,
                 fit_parameters: Callable, predict: Callable, save_fit: Callable, warmup_hours=0,
                 input_columns=('Price', 'Temp'), target_column='Demand'):
        if not name or not all(callable(f) for f in (fit_baseline, fit_parameters, predict, save_fit)):
            raise ValueError('Declare a model name and fit, prediction and persistence callbacks.')
        required = ('fit_budget', 'initialization_policy', 'population', 'target_unit', 'source_sha256')
        if any(k not in metadata for k in required):
            raise ValueError('Declare budgets, initialization, population, units and model source hashes.')
        self.name, self.metadata = name, dict(metadata)
        self.prediction_columns = dict(prediction_columns)
        self.fit_baseline, self.fit_parameters, self.predict = fit_baseline, fit_parameters, predict
        self.save_fit = save_fit
        self.warmup_hours, self.input_columns, self.target_column = warmup_hours, tuple(input_columns), target_column

    def forecast_week(self, data, directory):
        forecasts = []
        for weekday in range(7):
            day = split_weekday(data, weekday)
            baseline = self.fit_baseline(day)
            parameters = self.fit_parameters(day, baseline)
            self.save_fit(day, baseline, parameters, directory)
            # Target-free prediction object: even historical Y is unnecessary
            # for prediction once parameters / forecast state are frozen.
            # The intervening forecast inputs allow F's own posterior to
            # propagate continuously from Monday to the selected test day.
            forecast_prefix = data.inputs.loc[data.inputs.index <= day.inputs.index[-1]]
            prediction_context = pd.concat([day.context.drop(columns=[day.target_column]), forecast_prefix])
            prediction_day = WeekdayData(day.weekday, day.training_days, (),
                                        prediction_context, day.inputs.copy(), day.target_column)
            prediction = self.predict(prediction_day, baseline, parameters)
            if not isinstance(prediction, pd.DataFrame) or not prediction.index.equals(day.inputs.index):
                raise ValueError('Each weekday forecast must match its own physical test-day calendar.')
            forecasts.append(prediction)
            _atomic_report(directory/f'weekday_{weekday}.json', {'weekday':weekday,
                'training_days':[str(d.date()) for d in day.training_days], 'status':'forecast-returned',
                'measured_samples':[int(r[day.target_column].notna().sum()) for r in day.records]})
        return pd.concat(forecasts).sort_index()


def _fingerprint(frame):
    payload = '|'.join(map(str, frame.columns)).encode() + pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def score_predictions(predictions, observations, prediction_columns):
    """One common observed mask, raw-unit errors; no filling missing test Y."""
    if (not isinstance(observations, pd.Series) or not isinstance(observations.index, pd.DatetimeIndex)
            or observations.index.hasnans or not observations.index.is_unique or not observations.index.is_monotonic_increasing
            or (observations.index.tz is None) != (predictions.index.tz is None)):
        raise ValueError('Scoring requires a unique timestamped observation Series.')
    y = pd.to_numeric(observations.reindex(predictions.index), errors='raise').to_numpy(float)
    if np.isinf(y).any():
        raise ValueError('Scoring observations may be finite or NaN, never infinite.')
    mask = np.isfinite(y); out = {}
    for name, column in prediction_columns.items():
        error = predictions[column].to_numpy(float)[mask] - y[mask]
        row = {'samples':int(mask.sum()), 'squared_error_sum':float(np.sum(error**2)),
               'absolute_error_sum':float(np.sum(abs(error))), 'error_sum':float(np.sum(error))}
        row.update(rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                   mae=float(np.mean(abs(error))) if len(error) else None,
                   bias=float(np.mean(error)) if len(error) else None)
        out[name] = row
    return out


def _verify_artifacts(directory, hashes):
    """Validate exact saved bytes, refusing paths outside this run directory."""
    for name, expected in hashes.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError('Invalid or missing frozen artifact path.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen forecast/model artifacts changed before reuse or scoring.')


def _verify_freezes(forecasts):
    for directory, _, hashes in forecasts:
        _verify_artifacts(directory, hashes)


def _atomic_csv(path, frame):
    temporary = path.with_suffix(path.suffix+'.partial')
    frame.to_csv(temporary, index_label='Datetime')
    temporary.replace(path)


def _read_predictions(path, index):
    frame = pd.read_csv(path, float_precision='round_trip')
    stamps = pd.to_datetime(frame.pop('Datetime'), utc=index.tz is not None)
    actual = pd.DatetimeIndex(stamps)
    if index.tz is not None:
        actual = actual.tz_convert(index.tz)
    if not actual.equals(index):
        raise ValueError('Saved prediction calendar changed.')
    frame.index = index.copy()
    return frame


def _check_predictions(predictions, data, adapter):
    if (not isinstance(predictions, pd.DataFrame) or not predictions.index.equals(data.inputs.index)
            or not predictions.columns.is_unique or set(adapter.prediction_columns.values()) - set(predictions.columns)
            or not np.isfinite(predictions[list(adapter.prediction_columns.values())].to_numpy(float)).all()
            or adapter.target_column in predictions.columns):
        raise ValueError('Complete input-only forecast on the exact physical test calendar required.')


def run_backtest(history, inputs, windows, adapter, output, *, load_observations=None, resume=False):
    """Freeze all forecasts before targets; explicit resume reuses verified boundaries.

    Completed weeks can be reused for any adapter. Partial weeks require an
    adapter with its own explicit resume contract. No automatic retry occurs.
    """
    windows = tuple(windows)
    if (not windows or any(not isinstance(w, WeekWindow) for w in windows)
            or any(a.origin + pd.DateOffset(weeks=1) != b.origin or a.zone != b.zone for a,b in zip(windows, windows[1:]))):
        raise ValueError('Require consecutive chronological Monday windows in one zone.')
    if not adapter.prediction_columns or len(set(adapter.prediction_columns.values())) != len(adapter.prediction_columns):
        raise ValueError('Declare distinct scored prediction columns.')
    # Only selected historical targets and exogenous future inputs enter this identity.
    selected = [split_week(history, inputs, w, target_column=adapter.target_column,
        input_columns=adapter.input_columns, warmup_hours=adapter.warmup_hours) for w in windows]
    records = [{**d.window.record(), 'history_sha256':_fingerprint(d.history),
        'inputs_sha256':_fingerprint(d.inputs), 'test_targets_passed_to_adapter':False} for d in selected]
    plan = {'protocol':PROTOCOL, 'resume_schema':'completed-boundaries-v1',
        'adapter':adapter.name, 'adapter_metadata':adapter.metadata, 'windows':records,
        'prediction_columns':adapter.prediction_columns,
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'runtime':{'numpy':np.__version__, 'pandas':pd.__version__},
        'warmup_hours':adapter.warmup_hours, 'input_columns':list(adapter.input_columns),
        'target_column':adapter.target_column, 'scoring_requested':load_observations is not None,
        'automatic_retry':False, 'scoring_targets_opened':False}
    root = Path(output)
    if resume:
        if not root.is_dir():
            raise ValueError('Resume requires an existing run directory.')
    else:
        root.mkdir(parents=True, exist_ok=False)
    with _RunLock(root):
        plan_path = root/'plan.json'
        if resume and plan_path.exists():
            saved = json.loads(plan_path.read_text(encoding='utf-8'))
            if saved != json.loads(json.dumps(plan, allow_nan=False)):
                raise ValueError('Resume source/runtime/config/model/calendar/input identity changed.')
        elif resume and any(p.name != 'run.lock' and not p.name.endswith('.partial') for p in root.iterdir()):
            raise ValueError('Missing run plan with existing work; preserve artifacts and use a new directory.')
        else:
            _atomic_report(plan_path, plan)
        return _run_windows(selected, records, adapter, root, load_observations, resume)


def _run_windows(selected, records, adapter, root, load_observations, resume):
    begun = perf_counter(); origin = None; forecasts = []; reused = 0
    try:
        for data, record in zip(selected, records):
            window = data.window; origin = str(window.origin)
            directory = root/window.origin.strftime('%Y%m%d')
            freeze_path = directory/'forecast_freeze.json'
            reuse_window = resume and freeze_path.exists()
            if reuse_window:
                frozen = json.loads(freeze_path.read_text(encoding='utf-8'))
                hashes = frozen['artifact_sha256']
                _verify_artifacts(directory, hashes)
                if json.loads((directory/'window.json').read_text(encoding='utf-8')) != record:
                    raise ValueError('Saved window identity changed.')
                predictions = _read_predictions(directory/'predictions.csv', data.inputs.index)
                reused += 1
            else:
                partial = directory.exists()
                if partial and (not resume or not getattr(adapter, 'supports_resume', False)):
                    raise ValueError('Partial week requires an adapter supporting explicit resume; artifacts retained.')
                directory.mkdir(exist_ok=partial)
                if partial and (directory/'window.json').exists():
                    if json.loads((directory/'window.json').read_text(encoding='utf-8')) != record:
                        raise ValueError('Partial window identity changed.')
                elif partial and any(not p.name.endswith('.partial') for p in directory.iterdir()):
                    raise ValueError('Missing window identity with existing artifacts.')
                else:
                    _atomic_report(directory/'window.json', record)
                predictions = (adapter.forecast_week(data, directory, resume=True) if partial
                               else adapter.forecast_week(data, directory))
                _check_predictions(predictions, data, adapter)
                _atomic_csv(directory/'predictions.csv', predictions)
                hashes = {p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(directory.rglob('*')) if p.is_file() and not p.name.endswith('.partial')}
                _atomic_report(freeze_path, {'origin':origin, 'status':'frozen-input-only',
                    'scoring_targets_opened':False, 'artifact_sha256':hashes})
            _check_predictions(predictions, data, adapter)
            hashes = dict(hashes, **{'forecast_freeze.json':hashlib.sha256(freeze_path.read_bytes()).hexdigest()})
            forecasts.append((directory, predictions.copy(), hashes))
            print(f'{adapter.name}: {window.origin.date()} frozen ({len(predictions)} hours; reused={reuse_window})', flush=True)
        _verify_freezes(forecasts)
        receipt = {'origins':[str(d.window.origin) for d in selected],
            'freeze_sha256':{d.name:hashlib.sha256((d/'forecast_freeze.json').read_bytes()).hexdigest() for d,_,_ in forecasts},
            'scoring_targets_opened':False}
        if (root/'all_forecasts_frozen.json').exists():
            if json.loads((root/'all_forecasts_frozen.json').read_text(encoding='utf-8')) != receipt:
                raise ValueError('All-forecast freeze identity changed.')
        else:
            _atomic_report(root/'all_forecasts_frozen.json', receipt)
        if resume and (root/'summary.json').exists():
            summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
            if summary['status'] != 'complete':
                raise ValueError('Unexpected saved completion status.')
            _atomic_report(root/'run_status.json', summary, replace_existing=True)
            print('complete: all frozen weeks reused; no model work or rescoring', flush=True)
            return summary
        reports = []
        if load_observations is not None:
            observations = load_observations()
            _verify_freezes(forecasts)
            for directory, predictions, _ in forecasts:
                scores = score_predictions(predictions, observations, adapter.prediction_columns)
                _atomic_report(directory/'scores.json', scores, replace_existing=resume); reports.append(scores)
        pooled = {}
        for name in adapter.prediction_columns:
            rows = [r[name] for r in reports]; n = sum(r['samples'] for r in rows)
            pooled[name] = {'samples':n, 'rmse':float(np.sqrt(sum(r['squared_error_sum'] for r in rows)/n)) if n else None,
                            'mae':sum(r['absolute_error_sum'] for r in rows)/n if n else None,
                            'bias':sum(r['error_sum'] for r in rows)/n if n else None}
        summary = {'status':'complete', 'protocol':PROTOCOL, 'adapter':adapter.name,
                   'weeks':len(selected), 'scored':load_observations is not None, 'pooled':pooled,
                   'seconds':perf_counter()-begun, 'reused_weeks':reused, 'automatic_retry':False}
        _atomic_report(root/'summary.json', summary)
        _atomic_report(root/'run_status.json', summary, replace_existing=True)
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        failure = {'status':'paused' if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else 'failed',
            'origin':origin, 'error_type':type(exc).__name__, 'error':str(exc),
            'seconds':perf_counter()-begun, 'automatic_retry':False}
        _atomic_report(root/'failure.json', failure, replace_existing=True)
        _atomic_report(root/'run_status.json', failure, replace_existing=True)
        raise
