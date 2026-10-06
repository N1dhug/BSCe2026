"""One focused OLS/fixed-group/cutoff/packet check; no FV/model runs."""
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.tools.bm_1_h2_data_packet import (MED_COLUMNS, conditional_time_settings,
    file_sha256, prepare_compact_packet, validate_policy, weekday_hour_profile)
from src.tools.bm_1_reports import _read_likelihood_report


def compact_data_control(policy):
    from openpyxl import Workbook
    history = pd.date_range(policy['history']['start'],policy['history']['information_cutoff'],freq='h',inclusive='left')
    targets = pd.date_range(policy['split']['prior_origin'],policy['split']['holdout_end'],freq='h',inclusive='left')
    keys = history.dayofweek*24+history.hour
    values = 2.+np.asarray(keys)[:,None]/200.+np.arange(13)[None,:]/100.
    measurements = pd.DataFrame(values,index=history,columns=MED_COLUMNS)
    measurements.iloc[0,0] = np.nan
    baseline,profile = weekday_hour_profile(measurements,
        cutoff=policy['history']['information_cutoff'],targets=targets)
    complete = measurements.notna().all(axis=1)
    # Independent least-squares design matrix verifies all 168 coefficients.
    design = np.eye(168)[keys[complete]]
    response = measurements.loc[complete].to_numpy().sum(axis=1)/13.
    expected = np.linalg.lstsq(design,response,rcond=None)[0]
    error = float(np.max(np.abs(profile['coefficients']-expected)))
    if error > 1e-12 or profile['missing_group_hours'] != 1 or not pd.isna(profile['observed'].iloc[0]):
        raise AssertionError('OLS or complete fixed-population rule failed.')
    rejected = []
    def must_fail(label,call):
        try:
            call()
        except (ValueError,FileExistsError):
            rejected.append(label)
        else:
            raise AssertionError(label+' was silently accepted.')
    bad = deepcopy(policy); bad['population']['measurement_columns'][0] = 'other_MED'
    must_fail('changed population',lambda:validate_policy(bad))
    bad = deepcopy(policy); bad['exposure']['blind_holdout_verified'] = True
    must_fail('unsupported blindness',lambda:validate_policy(bad))
    must_fail('future baseline sample',lambda:weekday_hour_profile(measurements,
        cutoff=history[-1],targets=targets))
    must_fail('insufficient weekday/hour history',lambda:weekday_hour_profile(measurements.iloc[:336],
        cutoff=policy['history']['information_cutoff'],targets=targets))
    old = dict(policy['numerics'],base_max_step=1/24576,fine_max_step=1/49152)
    must_fail('old CN time steps under nominal F diffusion',lambda:conditional_time_settings(policy['physical'],old))
    cn = conditional_time_settings(policy['physical'],policy['numerics'])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); input_path,price_path = root/'input.xlsx',root/'prices.xlsx'
        book = Workbook(); sheet = book.active; sheet.title = policy['header_evidence']['input_sheet']
        sheet.append([None,None]+[c[:-4] for c in MED_COLUMNS]); sheet.append(['Dato','Time']+['MED']*13)
        for i,t in enumerate(history):
            row = measurements.iloc[i].to_numpy()
            sheet.append([t.strftime('%d-%m-%Y'),t.hour]+[None if np.isnan(v) else float(v) for v in row])
        # These formulas MUST NOT be decoded, including the cutoff hour itself.
        for stamp in (policy['history']['information_cutoff'],policy['split']['train_start'],policy['split']['holdout_start']):
            t = pd.Timestamp(stamp); sheet.append([t.strftime('%d-%m-%Y'),t.hour]+['=1/0']*13)
        book.save(input_path); book.close()
        book = Workbook(); sheet = book.active; sheet.title = policy['header_evidence']['price_sheet']
        sheet.append(['HourUTC','PriceArea','SpotPriceDKK'])
        price_times = pd.date_range(history[0],policy['split']['holdout_start'],freq='h',inclusive='left')
        for i,t in enumerate(price_times):
            sheet.append([t.tz_convert('UTC').tz_localize(None).isoformat(),'DK1',100.+i%24])
        t = pd.Timestamp(policy['split']['holdout_start'])
        sheet.append([t.tz_convert('UTC').tz_localize(None).isoformat(),'DK1','=1/0'])
        book.save(price_path); book.close()
        fixture = deepcopy(policy)
        fixture['header_evidence'].update(input_workbook_sha256=file_sha256(input_path),
            price_workbook_sha256=file_sha256(price_path),report='synthetic fixture, not historical provenance')
        packet = prepare_compact_packet(fixture,input_path=input_path,price_path=price_path,output_dir=root/'packet')
        np.testing.assert_allclose(packet['frozen_scales']['power_kW'],1.1*response.max(),rtol=0,atol=1e-14)
        np.testing.assert_allclose([r['mean_kW'] for r in packet['baseline_profile']],expected,rtol=0,atol=1e-12)
        inputs,_ = _read_likelihood_report(root/'packet/inputs.npz')
        if len(inputs['baseline']) != 264 or len(inputs['input_edges']) != 265 or inputs['observation_values'] is not None:
            raise AssertionError('Packet interval geometry or no-target-Y contract failed.')
        np.testing.assert_allclose(inputs['price'][96:],
            (np.arange(168)%24+100.-97.7)/27.6,rtol=0,atol=1e-14)
        from src.params.bm_1_history import BM1ChronologicalSplit
        record = packet['contract']['split']
        split = BM1ChronologicalSplit(**{k:record[k] for k in
            ('train_start','selection_start','holdout_start','holdout_end')},
            exposure_intervals=tuple(record['exposure_intervals']),holdout_audit_reference=record['holdout_audit_reference'])
        must_fail('development packet as blind holdout',split.require_blind_holdout)
        must_fail('overwrite frozen packet',lambda:prepare_compact_packet(fixture,
            input_path=input_path,price_path=price_path,output_dir=root/'packet'))
        fixture['header_evidence']['input_workbook_sha256'] = '0'*64
        must_fail('changed workbook bytes',lambda:prepare_compact_packet(fixture,
            input_path=input_path,price_path=price_path,output_dir=root/'other'))
        if any(packet[k] for k in ('historical_fit_ready','runtime_admitted','numerical_likelihood_ready','blind_h3_acceptance','scientific_acceptance')):
            raise AssertionError('Data preparation promoted an unsupported acceptance gate.')
    return dict(passed=True,ols_max_absolute_error=error,positive_cn_bound=cn,
        missing_group_hours=1,rejected=rejected,synthetic_only=True,
        target_measurement_values_decoded=0,final_realized_prices_decoded=0)
