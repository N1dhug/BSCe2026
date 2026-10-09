"""Deploy a frozen G weekday model without importing its historical fit drivers.

Time is hours. f(x)=1-2x, delta=tanh(k*(f(x)+g(u))/2), and
D=B+Xi*delta*((1-B) if delta>=0 else B). The preserved RK4 kernel solves
dx/dh=(D-B)/C and reports hourly Dbar=Bbar+C*(x_end-x_start).

Seven independent daily paths use their own frozen historical median starts.
A later forecast changes only the calendar and declared Price/Temp scenario;
it does not update parameters, baseline, states or the empirical error law.
Raw demand is an affine coordinate: demand_min+demand_scale*D. Negative raw
values are not clipped, and C*demand_scale is not an identified battery size.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import scipy

from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.params.bm_1_response import BM1PiecewiseResponse
from src.tools.bm_1_deterministic_snapshot import restore_baseline
from src.tools import weekday_backtest as calendar

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'BM-G-frozen-weekday-deployment-v1'
SOURCES = (
    'src/method/bm_1_g_forecast.py',
    'src/method/bm_1_deterministic_response.py',
    'src/method/bm_1_deterministic_fit.py',
    'src/params/bm_1_deterministic.py', 'src/params/bm_1_response.py',
    'src/validation/estimation.py', 'src/tools/bm_1_deterministic_snapshot.py',
    'src/exe/run_model/wm_4/output_model.py',
    'src/validation/model/wm_4.py', 'src/validation/model/wm_1.py',
    'src/tools/weekday_backtest.py',
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in SOURCES}


def runtime():
    return {'python':platform.python_version(), 'platform':platform.platform(),
            'numpy':np.__version__, 'pandas':pd.__version__, 'scipy':scipy.__version__}


def load_fit_binding(path, snapshot, explicit_path=None, *, legacy=False):
    """Require original fit receipts for pre-marker declarations; no silent downgrade."""
    path = Path(path)
    binding_path = Path(explicit_path) if explicit_path else path.parent/'fit_population_binding.json'
    marked = any('population_binding_sha256' in snapshot.get(key,{}) for key in ('parent','refined_view'))
    freeze_path = binding_path.parent/'forecast_freeze.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8')) if freeze_path.exists() else None
    receipts = (binding_path.parent/'population_binding.json').exists()
    if freeze is not None:
        receipts |= 'fit_population_binding.json' in freeze.get('artifact_sha256',{})
    if not binding_path.exists():
        if explicit_path or marked or receipts or not legacy:
            raise ValueError('Missing fit population binding; --legacy-unverified cannot override a declared fit.')
        return None, None, None
    raw = binding_path.read_bytes()
    binding = json.loads(raw)
    if freeze is None and not marked:
        raise ValueError('Unmarked declared fits require their original forecast_freeze.json receipt.')
    if freeze is not None:
        required = {'fit_population_binding.json','population_binding.json','weekday_snapshot.json','weekday_export_snapshot.json'}
        hashes = freeze['artifact_sha256']
        if freeze.get('status')!='frozen-input-only' or not required<=set(hashes):
            raise ValueError('Fit freeze must bind the declaration and both original snapshots.')
        calendar._verify_artifacts(binding_path.parent,hashes)
        if (hashlib.sha256(raw).hexdigest()!=hashes['fit_population_binding.json']
                or hashlib.sha256(path.read_bytes()).hexdigest()!=hashes['weekday_export_snapshot.json']):
            raise ValueError('Supplied snapshot or declaration differs from the frozen fit artifacts.')
        initial = json.loads((binding_path.parent/'population_binding.json').read_text(encoding='utf-8'))
        if any(binding.get(key)!=value for key,value in initial.items()):
            raise ValueError('Fit binding disagrees with its frozen pre-fit declaration.')
    return binding, raw, freeze

def validate_input_manifest(config):
    """A declaration binds an aggregate; it cannot prove its meter membership."""
    if not isinstance(config, dict) or config.get('schema')!='bm1-g-weekday-input-v1':
        raise ValueError('New G fits require an explicit population/input manifest.')
    if config.get('group')!='MED':
        raise ValueError('This preserved G fitting kernel supports declared MED only; UDEN needs its own reviewed fitting contract.')
    columns = config.get('measurement_columns')
    if (not isinstance(columns, list) or not columns
            or any(not isinstance(c,str) or not c.endswith('_MED') or not c[:-4] for c in columns)
            or len(set(columns))!=len(columns)):
        raise ValueError('Declare a nonempty unique MED meter list for the aggregate.')
    if (not isinstance(config.get('population_id'),str) or not config['population_id'].startswith('MED_')
            or config.get('aggregation')!='mean over available measured meters'
            or config.get('demand_timestamp_label')!='hour_start'):
        raise ValueError('Declare the MED population, available-meter aggregation and hour-start observation contract.')
    digest = config.get('history_csv_sha256')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Declare the lowercase SHA256 of the exact historical aggregate CSV.')
    if any(not isinstance(config.get(k),str) or not config[k].strip() for k in ('target_unit','input_scenario_policy')):
        raise ValueError('Declare raw-unit assumptions and scenario availability.')
    return deepcopy(config)


def population_contract(original, binding=None, *, legacy=False):
    """Check the fit receipt against both immutable snapshots, then project it."""
    view = original['refined_view']
    parent = original.get('parent', view)
    marked = 'population_binding_sha256' in parent or 'population_binding_sha256' in view
    if binding is None:
        if marked or not legacy:
            raise ValueError('G fit population binding is required; only unmarked legacy fits allow --legacy-unverified.')
        return {'label':view['population'],
            'group':'MED' if view['population']=='MED mean over available measured meters' else None,
            'verification':'legacy-unverified', 'meters':None,
            'reason':'Original fit records only an aggregate label, not meter membership or missing masks per meter.'}
    try:
        metadata = binding['metadata']
        config = validate_input_manifest(metadata['input_manifest'])
        final_fields = ('snapshot_sha256','export_snapshot_sha256','fitted_training_fingerprint','scales','baseline_sha256')
        initial = {key:value for key,value in binding.items() if key not in final_fields}
        if marked and any(s.get('population_binding_sha256')!=fingerprint(initial) for s in (parent,view)):
            raise ValueError('G declaration/history binding changed after fitting.')
        if (binding['snapshot_sha256']!=fingerprint(parent)
                or original['parent_snapshot_sha256']!=fingerprint(parent)
                or binding['export_snapshot_sha256']!=fingerprint(original)):
            raise ValueError('G population binding belongs to different snapshots.')
        for snapshot in (parent,view):
            if (snapshot['population']!='MED mean over available measured meters'
                    or snapshot['training_start']!=binding['training_start']
                    or snapshot['training_end_exclusive']!=binding['training_end_exclusive']
                    or snapshot['origin']!=binding['training_end_exclusive']
                    or snapshot['training_fingerprint']!=binding['fitted_training_fingerprint']
                    or snapshot['scales']!=binding['scales']
                    or snapshot['demand_timestamp_label']!=config['demand_timestamp_label']
                    or {str(day['weekday']):fingerprint(day['baseline']) for day in snapshot['days']}!=binding['baseline_sha256']):
                raise ValueError('G population binding disagrees with fitted history, window, scales or baselines.')
        if (binding['membership_status']!='declared-unverified'
                or metadata['membership_status']!='declared-unverified'
                or metadata['input_manifest_sha256']!=fingerprint(config)
                or type(metadata['history_file_sha256_verified']) is not bool
                or len(binding['history_sha256'])!=64):
            raise ValueError('Invalid declared G population or historical data identity.')
        for key, declared in (('group','group'),('population','population_id'),('measurement_columns','measurement_columns'),
                              ('aggregation','aggregation'),('target_unit','target_unit'),('input_scenario_policy','input_scenario_policy')):
            if metadata[key]!=config[declared]:
                raise ValueError('G population metadata differs from its input declaration.')
        return {'label':view['population'], 'group':config['group'], 'meters':config['measurement_columns'],
            'verification':'declared-unverified', 'declaration':config, 'binding_sha256':fingerprint(binding),
            'reason':'Byte-bound caller declaration; aggregate meter membership is not independently verified.'}
    except (KeyError,TypeError) as exc:
        raise ValueError('Incomplete G fit population binding.') from exc


def export_model(snapshot, binding=None, *, legacy=False, fit_directory=None):
    """One-time validated projection; no history read, replay or optimizer call.

    The complete original record stays in the model as immutable provenance.
    Its missing meter membership cannot be inferred from its aggregate label.
    """
    from src.method.g_weekday_export import validate_export
    original = validate_export(snapshot)
    fitted = original['refined_view']
    freeze = None
    marked = any('population_binding_sha256' in original.get(key,{}) for key in ('parent','refined_view'))
    if binding is not None and not marked:
        if fit_directory is None:
            raise ValueError('Unmarked declared G export requires its original frozen fit directory.')
        checked, binding_bytes, freeze = load_fit_binding(Path(fit_directory)/'weekday_export_snapshot.json',original)
        if binding!=checked:
            raise ValueError('G binding differs from the original frozen fit directory.')
    days = []
    for day in fitted['days']:
        g = day['G']
        days.append({'weekday':day['weekday'], 'baseline':deepcopy(day['baseline']),
            'parameters':list(g['parameters']), 'price_response':deepcopy(g['price_response']),
            'state':g['forecast_state'], 'fine_state':g['fine_forecast_state'],
            'errors':list(day['g_residual_samples_normalized'])})
    record = {'kind':'G-frozen-weekday-model', 'schema_version':1, 'protocol':PROTOCOL,
        'source_sha256':source_hashes(), 'runtime':runtime(),
        'fit_cutoff':fitted['origin'], 'training_start':fitted['training_start'],
        'training_fingerprint':fitted['training_fingerprint'], 'scales':deepcopy(fitted['scales']),
        'population':population_contract(original,binding,legacy=legacy),
        'demand_unit':fitted['demand_unit'], 'days':days,
        'original_snapshot_sha256':fingerprint(original), 'original_snapshot':original,
        'initialization':'seven separate weekday paths; unchanged frozen daily median states',
        'uncertainty':'frozen in-sample weekday empirical output errors; no future coverage guarantee',
        'conversion':'validated projection only; zero fitting and historical replay'}
    if binding is not None:
        record['fit_population_binding'] = deepcopy(binding)
    if freeze is not None:
        record['fit_population_freeze'] = freeze
        record['fit_population_binding_json'] = binding_bytes.decode('utf-8')
    record['model_sha256'] = fingerprint(record)
    return validate_model(record)


def validate_model(record, group=None):
    """Validate deployment identity without loading the historical fit stack."""
    try:
        if (record['kind']!='G-frozen-weekday-model' or record['schema_version']!=1
                or record['protocol']!=PROTOCOL or record['source_sha256']!=source_hashes()
                or record['runtime']!=runtime()):
            raise ValueError('Incompatible G deployment sources/runtime; explicitly export a reviewed model again.')
        body = {k:v for k,v in record.items() if k!='model_sha256'}
        if fingerprint(body)!=record['model_sha256']:
            raise ValueError('G model content changed after export.')
        original = record['original_snapshot']
        if fingerprint(original)!=record['original_snapshot_sha256']:
            raise ValueError('Original G snapshot provenance changed.')
        fitted = original['refined_view']
        for key, old_key in (('fit_cutoff','origin'), ('training_start','training_start'),
                             ('training_fingerprint','training_fingerprint'), ('scales','scales'),
                             ('demand_unit','demand_unit')):
            if record[key]!=fitted[old_key]:
                raise ValueError('Deployment changed fitted history, scales or units.')
        population = record['population']
        if population!=population_contract(original,record.get('fit_population_binding'),legacy=True):
            raise ValueError('Deployment population differs from the bound fit declaration.')
        if (record.get('fit_population_binding') is not None
                and not any('population_binding_sha256' in original.get(key,{}) for key in ('parent','refined_view'))):
            freeze = record['fit_population_freeze']
            if (freeze['status']!='frozen-input-only'
                    or not {'fit_population_binding.json','population_binding.json','weekday_snapshot.json','weekday_export_snapshot.json'}<=set(freeze['artifact_sha256'])):
                raise ValueError('Unmarked declared model must preserve its original fit freeze.')
            raw = record['fit_population_binding_json']
            if (not isinstance(raw,str)
                    or hashlib.sha256(raw.encode('utf-8')).hexdigest()!=freeze['artifact_sha256']['fit_population_binding.json']
                    or json.loads(raw.encode('utf-8'))!=record['fit_population_binding']):
                raise ValueError('G model original binding proof changed after the frozen fit.')
        if group is not None and (group!=population['group'] or population['group'] is None):
            raise ValueError('Requested group differs from the saved G population; a separate group fit is required.')
        cutoff = calendar.WeekWindow(record['fit_cutoff']).origin
        if pd.Timestamp(record['training_start'])!=cutoff-pd.DateOffset(weeks=5):
            raise ValueError('G fit requires its original five-week training period.')
        scale = np.array([record['scales'][k] for k in ('demand_min','demand_scale','price_min','price_scale')])
        if not np.isfinite(scale).all() or scale[1]<=0 or scale[3]<=0:
            raise ValueError('Invalid frozen demand/price scales.')
        if len(record['days'])!=7:
            raise ValueError('Seven own weekday models are required.')
        for number, (day, old) in enumerate(zip(record['days'], fitted['days'])):
            g = old['G']
            expected = {'weekday':number, 'baseline':old['baseline'], 'parameters':g['parameters'],
                'price_response':g['price_response'], 'state':g['forecast_state'],
                'fine_state':g['fine_forecast_state'], 'errors':old['g_residual_samples_normalized']}
            if day!=expected or old['weekday']!=number or g['steps_per_hour']!=60:
                raise ValueError('Deployment must preserve exact fitted weekday parameters, starts and error banks.')
            _baseline(day)
            curve = BM1PiecewiseResponse.from_record(day['price_response'])
            p = np.asarray(day['parameters'], float)
            if (p.shape!=(4,) or not np.isfinite(p).all() or not 1<=p[0]<=744
                    or not .05<=p[1]<=10 or not 0<=p[2]<=1 or p[3]!=.5
                    or curve.knots.tolist()!=[0., .5, 1.]
                    or not 0<=day['state']<=1 or not 0<=day['fine_state']<=1):
                raise ValueError('Invalid G parameters or daily states.')
            errors = np.asarray(day['errors'], float)
            if (errors.ndim!=1 or len(errors)<2 or not np.isfinite(errors).all()
                    or np.any(np.diff(errors)<0) or np.var(errors)<=0):
                raise ValueError('Finite sorted positive-variance empirical error bank required.')
        return record
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError('Incomplete G deployment model.') from exc


def _baseline(day):
    record = deepcopy(day['baseline'])
    if record['implementation']!='weekday-hourly-mean-statistical-output-v1':
        raise ValueError('Own fitted weekday baseline required.')
    record['implementation'] = 'statistical_output'
    return restore_baseline(record)


def prepare(record, inputs, origin, *, group=None, hours=None):
    """Reject group/cutoff/calendar errors before any propagation."""
    validate_model(record, group)
    window = calendar.WeekWindow(origin)
    if window.origin<pd.Timestamp(record['fit_cutoff']):
        raise ValueError('Forecast origin precedes the saved fit cutoff.')
    scenario = prepare_inputs(inputs, window, record['scales'])
    count = len(scenario) if hours is None else hours
    if type(count) is not int or not 1<=count<=len(scenario):
        raise ValueError('--hours must select one or more physical hours inside this forecast week.')
    return scenario, scenario.index[:count]


def prepare_inputs(inputs, window, scale):
    """One physical-hour input contract for saved-week replay and new forecasts."""
    calendar._frame(inputs, ('Price', 'Temp'))
    if inputs.index.tz is not None:
        raise ValueError('G requires unique local-naive Copenhagen hour-start labels.')
    scenario = calendar._grid(inputs[['Price', 'Temp']], window.hours(window.origin, window.end))
    scenario = scenario.apply(pd.to_numeric, errors='raise')
    if np.iscomplexobj(scenario.to_numpy()) or not np.isfinite(scenario.to_numpy(float)).all():
        raise ValueError('Complete finite Price/Temp are required for the forecast week; no filling.')
    if not scenario.Price.between(scale['price_min'], scale['price_min']+scale['price_scale']).all():
        raise ValueError('Price outside the frozen domain; no clipping or rescaling.')
    return scenario


def _day_inputs(scenario, day, baseline, scale):
    """One exact-minute baseline grid, held hourly price and declared temperature."""
    physical = pd.date_range(day.tz_localize('Europe/Copenhagen'),
        (day+pd.Timedelta(days=1)).tz_localize('Europe/Copenhagen'), freq='h', inclusive='left')
    index = physical.tz_localize(None)
    source = scenario.loc[index]
    n = len(index)
    minutes = np.arange(n*60+1)
    elapsed = minutes/60
    endpoint = (physical[0]+pd.Timedelta(hours=n)).tz_localize(None)
    values = source.Temp.to_numpy(float)
    time = np.arange(n, dtype=float)
    if endpoint in scenario.index:
        values = np.r_[values, float(scenario.loc[endpoint, 'Temp'])]
        time = np.r_[time, n]
    fine_index = (physical[0].tz_convert('UTC')+pd.to_timedelta(minutes, unit='min')).tz_convert('Europe/Copenhagen')
    b = baseline.predict(fine_index, np.interp(elapsed, time, values))
    u = (source.Price.to_numpy(float)-scale['price_min'])/scale['price_scale']
    return BM1ResponseInputs(elapsed, b, u, observation_contract='hourly-mean', steps_per_hour=60), index


def forecast(record, inputs, origin, *, group=None, hours=None):
    begun = perf_counter()
    scenario, selected = prepare(record, inputs, origin, group=group, hours=hours)
    origin = pd.Timestamp(origin); scale = record['scales']; rows = []; errors = []
    for day in record['days']:
        inp, index = _day_inputs(scenario, origin+pd.Timedelta(days=day['weekday']), _baseline(day), scale)
        p = list(day['parameters']); p[3] = day['state']
        q = p.copy(); q[3] = day['fine_state']
        curve = BM1PiecewiseResponse.from_record(day['price_response'])
        coarse = inp.solve(p, price_response=curve)
        fine = inp.refined().solve(q, price_response=curve)
        errors.append(float(np.max(abs(coarse['demand']-fine['demand']))))
        if errors[-1]>1e-5:
            raise RuntimeError('Frozen G forecast fails unchanged 60/120 hourly-mean refinement tolerance.')
        samples = np.asarray(day['errors'], float)
        low, high = np.quantile(samples, [(1-.95)/2, 1-(1-.95)/2], method='inverted_cdf')
        rows.append(pd.DataFrame({'g_prediction':coarse['demand'], 'baseline':inp.b_mean,
            'state':coarse['state'][:-1], 'state_end':coarse['state'][1:],
            'g_prediction_output_mean':coarse['demand']+np.mean(samples),
            'g_prediction_output_variance':np.var(samples),
            'g_prediction_lower':coarse['demand']+low, 'g_prediction_upper':coarse['demand']+high,
            'weekday':day['weekday'], 'price':scenario.loc[index, 'Price'],
            'price_normalized':inp.prices, 'temperature':scenario.loc[index, 'Temp']}, index=index))
    table = pd.concat(rows).loc[selected].copy()
    physical = selected.tz_localize('Europe/Copenhagen').tz_convert('UTC')
    table['interval_start_utc'] = physical.astype(str)
    table['interval_end_utc'] = (physical+pd.Timedelta(hours=1)).astype(str)
    table['forecast_horizon_hours'] = np.arange(len(table))
    table['horizon_start_hours'] = table['forecast_horizon_hours']
    table['horizon_end_hours'] = table['horizon_start_hours']+1
    raw = table.copy()
    for name in ('g_prediction', 'baseline', 'g_prediction_output_mean', 'g_prediction_lower', 'g_prediction_upper'):
        raw[name] = scale['demand_min']+scale['demand_scale']*table[name]
    raw['g_prediction_output_variance'] = scale['demand_scale']**2*table['g_prediction_output_variance']
    sources, versions = source_hashes(), runtime()
    if sources!=record['source_sha256'] or versions!=record['runtime']:
        raise ValueError('G source/runtime changed during forecast; refusing to publish mixed-version results.')
    return {'status':'complete-frozen-model-forecast', 'predictions':raw, 'normalized_predictions':table,
        'manifest':{'protocol':PROTOCOL, 'source_sha256':sources, 'runtime':versions,
            'model_sha256':record['model_sha256'], 'original_snapshot_sha256':record['original_snapshot_sha256'],
            'fit_cutoff':record['fit_cutoff'], 'forecast_origin':str(origin), 'cutoff':str(origin),
            'model_age_hours':(origin.tz_localize('Europe/Copenhagen')-pd.Timestamp(record['fit_cutoff']).tz_localize('Europe/Copenhagen')).total_seconds()/3600,
            'training_start':record['training_start'], 'training_end_exclusive':record['fit_cutoff'],
            'end_exclusive':str((physical[-1]+pd.Timedelta(hours=1)).tz_convert('Europe/Copenhagen').tz_localize(None)),
            'physical_hours':len(table), 'timezone':'Europe/Copenhagen',
            'interval_definition':'one physical hour [interval_start_utc, interval_end_utc)',
            'horizon_definition':'physical elapsed hours from forecast_origin; forecast_horizon_hours aliases horizon_start_hours; horizon_end_hours=start+1',
            'demand_timestamp_label':'hour_start', 'demand_unit':record['demand_unit'],
            'population':deepcopy(record['population']), 'scales':deepcopy(scale),
            'demand_used':False, 'model_refitted':False, 'baseline_refitted':False,
            'historical_replay_executed':False, 'state_path':record['initialization'],
            'scenario':'caller supplied Price/Temp; availability at forecast origin not independently verified',
            'scenario_fingerprint':hashlib.sha256(pd.util.hash_pandas_object(scenario,index=True).values.tobytes()).hexdigest(),
            'uncertainty':record['uncertainty'], 'wm_comparison':'not executed; population membership is not independently verified',
            'normalization':'unchanged train-only affine demand coordinates; no raw sign clipping',
            'propagation':'seven complete separate daily paths; requested prefix exported'},
        'numerics':{'passed':True, 'tolerance':1e-5, 'daily_max_demand_errors':errors},
        'scoring':{'status':'not-scored'}, 'seconds':perf_counter()-begun}
