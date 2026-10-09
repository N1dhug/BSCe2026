"""New two-piece weekday F fits using an explicitly earlier frozen F baseline.

Reuses the response optimizer and native interval likelihood. Supplied Demand
must be the declared fixed-population mean, NaN if any member is missing.
There is no baseline fit, saved-curve fallback, automatic retry or historic run.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import betainc

from src.method.bm_1_response import CurveTrainingRecord, curve_pair_from_logits, fit_shared_curve_pair
from src.method.bm_1_interval_native import (advance_frequency_block, compare_native_states,
    freeze_native_operator, replay_native_record, native_runtime)
from src.method.bm_1_interval_forecast import native_interval_moments, compare_interval_moments
from src.method.bm_1_interval_likelihood import EXTENDED_CENTERS
from src.params.bm_1 import BM1IntervalObservationSeries
from src.tools.input_generator.input_builder import input_builder
from src.tools.bm_1_reports import _atomic_report
from src.tools.weekday_backtest import (split_weekday, _fingerprint, _verify_artifacts,
    _atomic_csv, _read_predictions)


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and np.isneginf(value):
        return None  # Exact zero state mass, following the native checkpoint contract.
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FWeekdayAdapter:
    name = 'F-native-weekday'
    supports_resume = True
    warmup_hours = 0
    input_columns = ('Price',)
    target_column = 'Demand'
    prediction_columns = {'F': 'f_prediction', 'baseline': 'baseline'}

    def __init__(self, config, model):
        self.config, self.model = deepcopy(config), deepcopy(model)
        if config.get('schema') != 'bm1-f-weekday-adapter-v1' or model.get('schema') != 'bm1-f4-fixed-model-v1':
            raise ValueError('Explicit F weekday configuration and frozen F4 baseline/model required.')
        if any(not isinstance(config.get(key), str) or not config[key].strip()
               for key in ('target_unit', 'input_scenario_policy')):
            raise ValueError('Declare physical-unit assumptions and availability of supplied price scenarios.')
        if (config['population_id'] != model['population_id']
                or config['measurement_columns'] != model['measurement_columns']
                or config['aggregation'] != 'fixed-population mean; any missing member makes Demand NaN'):
            raise ValueError('Declare the frozen F population and complete-group masking contract.')
        columns = model['measurement_columns']
        if (not isinstance(columns, list) or not columns or any(not isinstance(c, str) for c in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError('Frozen F population requires a nonempty unique meter list.')
        groups = {column.rsplit('_', 1)[-1] for column in columns}
        if len(groups) != 1 or not groups <= {'MED', 'UDEN'}:
            raise ValueError('Frozen F population must contain one separate MED or UDEN group.')
        self.group = next(iter(groups))
        if (not model['population_id'].startswith(self.group+'_')
                or config.get('group', self.group) != self.group):
            raise ValueError('Requested group differs from the frozen F measurement population.')
        if model['timezone'] != 'Europe/Copenhagen':
            raise ValueError('The registered F weekday profile uses Europe/Copenhagen.')
        self.cutoff = pd.Timestamp(model['baseline_information_cutoff'])
        if pd.isna(self.cutoff) or self.cutoff.tzinfo is None:
            raise ValueError('Frozen baseline/scale information cutoff needs an explicit timezone.')
        scales = model['scales']
        self.power = float(scales['power_kW'])
        self.low, self.high = scales['price_low_DKK_MWh'], scales['price_high_DKK_MWh']
        if not np.isfinite([self.power, self.low, self.high]).all() or self.power <= 0 or self.low >= self.high:
            raise ValueError('Invalid frozen F scales.')
        self.profile = np.asarray(model['baseline_weekday_hour_kW'], float)/self.power
        if self.profile.shape != (168,) or not np.isfinite(self.profile).all() or np.any((self.profile < 0) | (self.profile > 1)):
            raise ValueError('Invalid frozen weekday/hour baseline; no clipping or filling.')
        p = model['physical']
        if (p['time_unit'] != 'day' or not np.isfinite([p[k] for k in ('C','k','Xi','sigma_x','sigma_y')]).all()
                or p['C'] <= 0 or p['k'] <= 0 or not 0 <= p['Xi'] <= 1 or p['sigma_x'] < 0 or p['sigma_y'] <= 0):
            raise ValueError('Finite nominal F physics in days required.')
        prior = model['prior']
        if not np.isfinite([prior['alpha'], prior['beta']]).all() or min(prior['alpha'], prior['beta']) <= 0:
            raise ValueError('Invalid declared Beta prior.')
        for key in ('fit_seconds', 'validation_seconds', 'forecast_seconds'):
            if isinstance(config[key], bool) or not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError('Declare positive finite per-weekday time budgets.')
        for key in ('max_evaluations', 'max_iterations'):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError('Declare positive integer optimizer budgets.')
        if (not np.isfinite(config['regularization']) or config['regularization'] < 0
                or not np.isfinite(config['logit_bound']) or not 0 < config['logit_bound'] <= 10):
            raise ValueError('Declare bounded logits and nonnegative regularization.')
        starts = np.asarray(config['starts'], float)
        if (starts.ndim != 2 or starts.shape[1] != 2 or not len(starts)
                or not np.isfinite(starts).all() or np.any(abs(starts) > config['logit_bound'])):
            raise ValueError('Supply explicit two-piece starts; saved fitted curves are not used.')
        for row in starts:
            curve_pair_from_logits(row, 2)
        n = model['numerics']
        if n['fine_cells'] <= n['base_cells'] or n['fine_max_step'] >= n['base_max_step']:
            raise ValueError('Require the saved distinct finer space/time validation grid.')

    @property
    def metadata(self):
        root = Path(__file__).resolve().parents[2]
        return dict(fit_budget={key:self.config[key] for key in ('fit_seconds','validation_seconds',
            'forecast_seconds','max_evaluations','max_iterations')},
            initialization_policy='Fresh declared Beta at five-week history start per candidate/weekday; continuous state, no daily resets.',
            group=self.group, population=self.model['population_id'], measurement_columns=self.model['measurement_columns'],
            aggregation=self.config['aggregation'], target_unit=self.config['target_unit'],
            input_scenario_policy=self.config['input_scenario_policy'],
            source_sha256={p.relative_to(root).as_posix():_sha(p) for p in sorted((root/'src').rglob('*.py'))},
            numerical_runtime=native_runtime(),
            adapter_config=self.config, baseline_information_cutoff=self.cutoff.isoformat(),
            curve_fit='NEW two-piece weekday TRAIN likelihood fit; saved F4 curves ignored',
            declared_seconds_per_week=7*sum(self.config[key] for key in ('fit_seconds','validation_seconds','forecast_seconds')),
            candidate_cap_per_week=7*self.config['max_evaluations'],
            baseline_policy='Explicit earlier frozen F weekday/hour profile and scales; no new baseline fit',
            uncertainty='Native interval initial-state/process and nominal Gaussian observation moments; no parameter/input uncertainty or coverage guarantee',
            time_budget_policy='Cooperative checks before/after each native block/hour; expired work never becomes an accepted score.')

    def _settings(self, grid):
        n = self.model['numerics']
        return dict(cells=n[grid+'_cells'], max_step=n[grid+'_max_step'], frequencies=n['frequencies'],
            span=n['span'], centers=list(EXTENDED_CENTERS), relative_guard=n['relative_guard'],
            overlap_log_tolerance=n['overlap_log_tolerance'])

    def _signals(self, frame):
        index = frame.index
        local = index.tz_localize(self.model['timezone'], ambiguous='raise', nonexistent='raise') if index.tz is None else index.tz_convert(self.model['timezone'])
        if len(local) > 1 and not np.all(np.diff(local.tz_convert('UTC').asi8) == pd.Timedelta(hours=1).value):
            raise ValueError('F needs every physical input hour, including gaps between target days.')
        price = (frame.Price.to_numpy(float)-self.low)/(self.high-self.low)
        if not np.isfinite(price).all() or np.any((price < 0) | (price > 1)):
            raise ValueError('Price outside frozen F scale or missing; no clipping.')
        baseline = self.profile[local.dayofweek*24+local.hour]
        return np.arange(len(local)+1, dtype=float)/24., baseline, price, local

    def _record(self, day):
        edges, b, u, local = self._signals(day.context)
        if self.cutoff > local[0]:
            raise ValueError('Frozen baseline/scale cutoff must be no later than the full five-week history start.')
        y = day.context[self.target_column].to_numpy(float)/self.power
        if np.isinf(y).any() or np.any(y[np.isfinite(y)] < 0):
            raise ValueError('TRAIN demand must be nonnegative or missing under the declared population mask.')
        f, g = curve_pair_from_logits([0., 0.], 2)
        p, prior = self.model['physical'], self.model['prior']
        initial = lambda x: np.diff(betainc(prior['alpha'], prior['beta'], x))
        B = input_builder.price_intervals((edges, b))
        U = input_builder.price_intervals((edges, u))
        return CurveTrainingRecord(f'weekday-{day.weekday}', 'TRAIN', B, (edges[0], edges[-1]),
            (U, f, g, p['C'], p['k'], p['Xi']), p['sigma_x'], p['sigma_y'], initial,
            BM1IntervalObservationSeries(edges[:-1], edges[1:], y, 'day', np.isfinite(y)), breakpoints=edges[1:-1])

    def _options(self, grid, deadline, path, state_callback=None):
        def mapper(jobs):
            results = []
            for job in jobs:
                if perf_counter() >= deadline:
                    raise TimeoutError('Native block budget exhausted.')
                results.append(advance_frequency_block(job))
                if perf_counter() >= deadline:
                    raise TimeoutError('Native block budget exhausted.')
            return results
        return dict(settings=self._settings(grid), should_pause=lambda:perf_counter() >= deadline,
            checkpoint=lambda value:_atomic_report(path, value, replace_existing=True),
            mapper=mapper, state_callback=state_callback)

    def _fit(self, day, directory):
        record = self._record(day)
        c = self.config
        deadline = perf_counter()+c['fit_seconds']
        options = self._options('base', deadline, directory/'candidate_checkpoint.json')
        base = options['settings']
        fit = fit_shared_curve_pair([record], pieces=2, starts=c['starts'], regularization=c['regularization'],
            max_evaluations=c['max_evaluations'], max_iterations=c['max_iterations'], logit_bound=c['logit_bound'],
            cells=base['cells'], max_step=base['max_step'], backend='fitted', native_options=options,
            should_pause=options['should_pause'],
            checkpoint=lambda trace:_atomic_report(directory/'optimizer_trace.json', _plain(trace), replace_existing=True))
        report = {key:fit[key] for key in ('logits','objective','log_likelihood','budget_exhausted','paused',
            'optimizer_results','distinct_evaluations','elapsed_seconds')}
        _atomic_report(directory/'fit_status.json', _plain(report))
        converged_best = any(row['success'] and abs(row['objective']-fit['objective']) <= 1e-10
                             and np.array_equal(np.asarray(row['logits']), fit['logits'])
                             for row in fit['optimizer_results'])
        if (fit['budget_exhausted'] or fit['paused'] or perf_counter() >= deadline or not converged_best):
            raise RuntimeError('Incomplete weekday F fit; no fixed-curve fallback or forecast acceptance.')
        # Same whole stream on saved base/fine grids. Reuse the historical .05
        # likelihood and .002/.05 state gates at every physical hour.
        deadline = perf_counter()+c['validation_seconds']
        states, comparisons = {}, []
        base_options = self._options('base', deadline, directory/'validation_base_checkpoint.json',
                                    lambda hour,logs:states.__setitem__(hour, logs))
        selected = replay_native_record(record, fit['f'], fit['g'], **base_options)
        def compare(hour, logs):
            comparisons.append(compare_native_states(states.pop(hour), logs))
        fine = replay_native_record(record, fit['f'], fit['g'],
            **self._options('fine', deadline, directory/'validation_fine_checkpoint.json', compare))
        difference = abs(selected.total_log_likelihood-fine.total_log_likelihood)
        if states or difference > .05:
            raise RuntimeError('Whole-TRAIN native base/fine likelihood or continuity gate failed.')
        _atomic_report(directory/'numerical_gate.json', dict(passed=True, train_loglik_difference=difference,
            compared_hours=len(comparisons), maximum_state_mean_difference=max(x['state_mean_difference'] for x in comparisons),
            maximum_state_L1_difference=max(x['conservative_state_L1_difference'] for x in comparisons),
            scientific_acceptance=False))
        result = dict(logits=fit['logits'].tolist(), pieces=2,
            base_log_state=selected.final_step['log_posterior_mass'],
            fine_log_state=fine.final_step['log_posterior_mass'])
        _atomic_report(directory/'frozen_fit.json', _plain(dict(result,
            origin=day.context.index[-1]+pd.Timedelta(hours=1),
            history_sha256=_fingerprint(day.context), new_curve_fit=True)))
        return result

    def _predict(self, inputs, candidate, directory):
        # inputs contains the entire origin-to-target-day prefix and no Demand.
        edges, b, u, local = self._signals(inputs)
        deadline = perf_counter()+self.config['forecast_seconds']
        states = {grid:np.asarray([float('-inf') if x is None else x
                  for x in candidate[grid+'_log_state']], dtype=float) for grid in ('base','fine')}
        specs = {grid:dict(settings=self._settings(grid), physical=self.model['physical'],
            input_edges=edges, baseline=b, price=u, horizon=[edges[0], edges[-1]]) for grid in states}
        rows = []
        for hour in range(len(inputs)):
            if perf_counter() >= deadline:
                raise TimeoutError('F forecast prefix incomplete; no accepted week.')
            output = {}
            for grid in states:
                operator = freeze_native_operator(specs[grid], candidate, dict(start=edges[hour], end=edges[hour+1]))
                output[grid] = native_interval_moments(operator, states[grid], self.model['physical']['sigma_y'])
                states[grid] = output[grid]['log_state']
            if perf_counter() >= deadline:
                raise TimeoutError('F forecast prefix exceeded its budget; no accepted week.')
            compare_native_states(states['base'], states['fine'])
            compare_interval_moments(output['base'], output['fine'])
            moment = output['base']
            rows.append(dict(f_prediction=moment['interval_mean']*self.power, baseline=b[hour]*self.power,
                f_demand_variance=moment['interval_variance']*self.power**2,
                f_observation_variance=moment['observation_variance']*self.power**2,
                interval_start_utc=local[hour].tz_convert('UTC').isoformat(),
                interval_end_utc=(local[hour]+pd.Timedelta(hours=1)).tz_convert('UTC').isoformat(),
                forecast_horizon_hours=hour+1))
            _atomic_report(directory/'forecast_checkpoint.json', _plain(dict(completed_hours=hour+1,
                base_log_state=states['base'], fine_log_state=states['fine'], rows=rows,
                status='complete-prefix' if hour+1 == len(inputs) else 'paused')), replace_existing=True)
        return pd.DataFrame(rows, index=inputs.index)

    def forecast_week(self, data, directory, *, resume=False):
        """Reuse complete, hash-bound fits; restart one interrupted attempt explicitly.

        Each unfinished attempt stays in its own directory. Optimizer internals
        and partial forecast prefixes are not resumed or silently accepted.
        """
        if data.window.zone != self.model['timezone']:
            raise ValueError('F profile and calendar timezone differ.')
        manifest = dict(metadata=self.metadata, model=self.model, history_sha256=_fingerprint(data.history),
            inputs_sha256=_fingerprint(data.inputs), test_targets_used=False)
        path = directory/'f_adapter_freeze.json'
        if resume and path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != json.loads(json.dumps(manifest)):
                raise ValueError('F resume model/source/runtime/config/input identity changed.')
        elif resume and any(p.name != 'window.json' and not p.name.endswith('.partial') for p in directory.iterdir()):
            raise ValueError('Missing F identity with existing work; no implicit migration.')
        else:
            _atomic_report(path, manifest)
        forecasts = []
        for weekday in range(7):
            day = split_weekday(data, weekday)
            dest = directory/f'weekday_{weekday}'; dest.mkdir(exist_ok=resume)
            attempts = sorted(dest.glob('attempt_[0-9][0-9][0-9][0-9]'))
            previous = attempts[-1] if attempts else None
            fit_receipt = previous/'fit_complete.json' if previous else None
            if resume and fit_receipt is not None and fit_receipt.exists():
                attempt = previous
                _verify_artifacts(attempt, json.loads(fit_receipt.read_text(encoding='utf-8'))['artifact_sha256'])
                frozen = json.loads((attempt/'frozen_fit.json').read_text(encoding='utf-8'))
                if frozen['history_sha256'] != _fingerprint(day.context) or not frozen['new_curve_fit']:
                    raise ValueError('Frozen weekday fit history or fit identity changed.')
                candidate = {key:frozen[key] for key in ('logits','pieces','base_log_state','fine_log_state')}
                print(f'F weekday {weekday}: reused verified fit', flush=True)
            else:
                attempt = dest/f'attempt_{len(attempts)+1:04d}'; attempt.mkdir()
                print(f'F weekday {weekday}: new fit attempt {attempt.name}; earlier partial work retained', flush=True)
                candidate = self._fit(day, attempt)
                hashes = {p.name:_sha(p) for p in attempt.iterdir() if p.is_file() and not p.name.endswith('.partial')}
                if 'frozen_fit.json' not in hashes:
                    raise RuntimeError('Fit returned without an atomic frozen fit artifact.')
                _atomic_report(attempt/'fit_complete.json', {'status':'complete', 'artifact_sha256':hashes})
            receipt = attempt/'forecast_complete.json'
            if resume and receipt.exists():
                _verify_artifacts(attempt, json.loads(receipt.read_text(encoding='utf-8'))['artifact_sha256'])
                predictions = _read_predictions(attempt/'predictions.csv', day.inputs.index)
            else:
                prefix = data.inputs.loc[data.inputs.index <= day.inputs.index[-1], list(self.input_columns)]
                predictions = self._predict(prefix, candidate, attempt).loc[day.inputs.index]
                _atomic_csv(attempt/'predictions.csv', predictions)
                _atomic_report(receipt, {'status':'complete', 'artifact_sha256':
                    {name:_sha(attempt/name) for name in ('predictions.csv','fit_complete.json')}})
            forecasts.append(predictions)
        if self.metadata['source_sha256'] != manifest['metadata']['source_sha256']:
            raise RuntimeError('F source changed during fit/forecast; no accepted week.')
        return pd.concat(forecasts).sort_index()


def make_adapter(config_path):
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    model_path = Path(config['model_file'])
    model_path = model_path if model_path.is_absolute() else path.parent/model_path
    payload = model_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != config['model_sha256']:
        raise ValueError('Frozen F model/baseline bytes changed.')
    model = json.loads(payload)
    return FWeekdayAdapter(config, model)
