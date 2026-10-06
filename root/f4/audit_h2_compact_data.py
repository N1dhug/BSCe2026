"""Read-only F compact packet audit; never execute uploaded source or fit data."""
from time import perf_counter
BEGUN = perf_counter()-.25

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT = ROOT/'files/output/bm_1/runs/estimation'


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def ordered_sum(values):
    total = 0.
    for v in values:
        total += v
    return total


def members(raw):
    from src.tools.bm_1_reports import _RunLock  # stable I/O, no model execution
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for entry in archive.infolist():
            name = entry.filename
            require(not entry.is_dir() and name not in result and '\\' not in name
                and not name.startswith('/') and '..' not in Path(name).parts
                and ':' not in name and entry.file_size <= 10_000_000,'Unsafe/duplicate ZIP member.')
            result[name] = archive.read(entry)  # Check original CRC.
    return result


def audit(archive_path,price_path,git_evidence_path,known_ledger_path):
    import numpy as np
    import pandas as pd
    from src.tools.bm_1_reports import _read_array_checkpoint
    from src.tools.bm_1_history_io import read_raw_spot_window
    from src.tools.bm_1_h2_data_packet import validate_policy, file_sha256
    from src.params.bm_1_history import BM1ChronologicalSplit
    raw = archive_path.read_bytes(); files = members(raw)
    require(set(files)=={'manifest.json','report.json','packet.json','baseline.csv','inputs.npz','source.zip','ledger.json'},'Unexpected receipt layout.')
    manifest,report,packet,ledger = (json.loads(files[n]) for n in
        ('manifest.json','report.json','packet.json','ledger.json'))
    evidence = json.loads(git_evidence_path.read_text())
    require(manifest['source_commit']==evidence['source_commit']=='d95fd26be038866ddfca1214fa5fcfd957d092a7','Wrong exported source commit.')
    require(manifest['mode']=='prepare' and manifest['grant_seconds']==180.
        and manifest['stop_reserve_seconds']==60. and manifest['environment']['workers']==1
        and manifest['environment']['blas_threads']==1,'Wrong data invocation contract.')
    sources = members(files['source.zip']); hashes = manifest['source_sha256']
    require(set(sources)==set(hashes)==set(evidence['source_git_blobs']) and len(sources)==60,'Incomplete source closure.')
    matched = {}
    for p,source in sources.items():
        require(hashlib.sha256(source).hexdigest()==hashes[p],'Original source hash mismatch: '+p)
        choices = dict(original_bytes=source,exact_checkout_lf=source.replace(b'\r\n',b'\n'))
        valid = [label for label,content in choices.items() if hashlib.sha1(
            b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==evidence['source_git_blobs'][p]]
        require(bool(valid),'Source differs from audited Git blob: '+p)
        matched[p] = valid[0]
    policy = packet['policy']; validate_policy(policy)
    require(policy==json.loads(sources['root/f4/h2_compact_data_policy_20261006.json']),'Packet policy changed after source freeze.')
    unsigned = {k:v for k,v in packet.items() if k!='identity'}
    identity = hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(identity==packet['identity']==report['result']['packet_identity'],'Packet identity mismatch.')
    require(report['passed'] is True and packet['conditional_data_ready'] is True
        and report['result']['conditional_data_ready'] is True,'Preparation did not pass.')
    for name in ('historical_fit_ready','runtime_admitted','scientific_acceptance'):
        require(report[name] is False and packet[name] is False,'Unsupported acceptance: '+name)
    require(packet['numerical_likelihood_ready'] is False and packet['blind_h3_acceptance'] is False
        and packet['target_measurement_values_decoded']==0 and packet['final_realized_prices_decoded']==0,'Final-data/readiness policy changed.')
    expected = policy['header_evidence']
    require(packet['input_sha256']==dict(input=expected['input_workbook_sha256'],prices=expected['price_workbook_sha256']),'Observed/audited workbook hashes differ.')
    for name,sha in packet['outputs_sha256'].items():
        require(hashlib.sha256(files[name]).hexdigest()==sha,'Raw output hash mismatch: '+name)
    split,history = policy['split'],packet['baseline_history']
    times = pd.date_range(split['prior_origin'],split['holdout_end'],freq='h',inclusive='left')
    historical = pd.date_range(policy['history']['start'],policy['history']['information_cutoff'],freq='h',inclusive='left')
    require(len(times)==264 and len(historical)==1104 and packet['prediction_times']==[t.isoformat() for t in times],'Chronology changed.')
    require(history['start']==policy['history']['start'] and history['information_cutoff']==policy['history']['information_cutoff']
        and history['history_hours']==1104 and history['complete_group_hours']+history['missing_group_hours']==1104,'History/cutoff count mismatch.')
    coefficients,counts = [],[]
    for i,row in enumerate(packet['baseline_profile']):
        require(row['weekday']==i//24 and row['hour']==i%24 and type(row['count']) is int
            and 3 <= row['count'] <= int(np.sum(historical.dayofweek*24+historical.hour==i)),'Invalid weekday/hour profile bin.')
        coefficients.append(row['mean_kW']);counts.append(row['count'])
    require(len(coefficients)==168 and sum(counts)==history['complete_group_hours'],'Incomplete profile/count total.')
    scales = packet['frozen_scales']; power = scales['power_kW']; contract = packet['contract']
    require(math.isfinite(power) and power>0. and power==contract['power_scale']
        and contract['population_id']==policy['population']['id']
        and contract['measurement_columns']==policy['population']['measurement_columns']
        and contract['scale_fitted_before']==policy['history']['information_cutoff'],'Frozen scale/population mismatch.')
    require(all(math.isfinite(v) and 0. <= v <= power/1.1 for v in coefficients),'Profile violates pre-origin positive scale.')
    rows = list(csv.DictReader(io.StringIO(files['baseline.csv'].decode())))
    expected_B = np.array(coefficients)[times.dayofweek*24+times.hour]
    require(len(rows)==264 and [r['timestamp'] for r in rows]==packet['prediction_times'],'Baseline CSV chronology differs.')
    np.testing.assert_array_equal([float(r['baseline']) for r in rows],expected_B)
    inputs = _read_array_checkpoint(files['inputs.npz'])
    np.testing.assert_array_equal(inputs['input_edges'],np.arange(265)/24.)
    np.testing.assert_array_equal(inputs['baseline'],expected_B/power)
    require(inputs['observation_values'] is None and [inputs[k] for k in
        ('prior_prediction_hours','train_hours','selection_hours','open_hours')]==[24,48,24,168],'Target-Y or interval geometry changed.')
    # Independently re-read only the small price workbook, NOT demand history.
    # Hash-identical to the original audited file; no final realized prices.
    require(file_sha256(price_path)==expected['price_workbook_sha256'],'Independent price evidence differs.')
    prices = read_raw_spot_window(price_path,start=policy['history']['start'],end=split['holdout_start'],
        price_area='DK1',sheet=expected['price_sheet'])
    calibration = prices[prices.index<pd.Timestamp(policy['history']['information_cutoff'])]
    require(not prices.isna().any() and len(calibration)==1104,'Incomplete independent price history.')
    margin = .1*float(calibration.max()-calibration.min())
    low,high = float(calibration.min()-margin),float(calibration.max()+margin)
    require(low==scales['price_low_DKK_MWh']==contract['price_low']
        and high==scales['price_high_DKK_MWh']==contract['price_high'],'Frozen price calibration differs.')
    origin = pd.Timestamp(split['holdout_start'])
    prefinal = prices.reindex(pd.date_range(times[0],origin,freq='h',inclusive='left')).to_numpy()
    previous = prices.reindex(pd.date_range(origin-pd.Timedelta(days=7),origin,freq='h',inclusive='left')).to_numpy()
    U = (np.r_[prefinal,previous]-low)/(high-low)
    require(U.shape==(264,) and np.isfinite(U).all() and np.all((0.<=U)&(U<=1.)),'Input price scope/normalization failed.')
    np.testing.assert_array_equal(inputs['price'],U)
    chronology = contract['split']
    blind = BM1ChronologicalSplit(**{k:chronology[k] for k in
        ('train_start','selection_start','holdout_start','holdout_end')},
        exposure_intervals=tuple(chronology['exposure_intervals']),holdout_audit_reference=chronology['holdout_audit_reference'])
    try:
        blind.require_blind_holdout()
    except ValueError:
        pass
    else:
        raise ValueError('Development packet silently qualifies as blind holdout.')
    for grid in ('base','fine'):
        n=packet['numerics'][grid+'_cells'];h=packet['numerics'][grid+'_max_step']
        rate=.5**2*n*n/16.+2*n
        require(packet['positive_cn_bound'][grid]==dict(cells=n,max_step=h,
            conservative_outgoing_rate=rate,cn_explicit_load_bound=h*rate/2.) and h*rate/2.<1.,'Positive-CN bound changed.')
    known = json.loads(known_ledger_path.read_text())
    require(ledger['entries'][:-1]==known['entries'] and ledger['entries'][-1]['label']=='native-development|compact-data-preparation'
        and ledger['entries'][-1]['status']=='closed','Receipt ledger did not extend the known complete prefix once.')
    require(ledger['ceiling_seconds']==36000. and ledger['target_seconds']==21600.
        and ledger['validation_reserve_seconds']>=7200.
        and all(e['status']=='closed' and math.isfinite(e['charged_seconds']) and e['charged_seconds']>=0. for e in ledger['entries']),'Invalid PC budget ledger.')
    require(ledger['charged_seconds']==425.8919726690074+ordered_sum(e['charged_seconds'] for e in ledger['entries']),'PC Python 3.11 ordered ledger sum differs.')
    used = ordered_sum(e['charged_seconds'] for e in known['entries'] if e['label'].startswith('native-development|'))
    require(manifest['native_seconds_before']==used,'PC native allowance was reset.')
    duration = ledger['entries'][-1]['charged_seconds']
    require(0.<=report['elapsed_seconds_before_ledger_close']<=duration<120.,'Preparation exceeded compute/stop-reserve allowance.')
    return dict(schema='f4-h2-compact-data-raw-audit-v1',passed=True,
        review_zip_sha256=hashlib.sha256(raw).hexdigest(),packet_identity=identity,
        source_commit=manifest['source_commit'],source_files_verified=60,source_matches=matched,
        input_sha256=packet['input_sha256'],frozen_scales=scales,baseline_history=history,
        baseline_bins=168,baseline_bin_count_min=min(counts),baseline_bin_count_max=max(counts),
        baseline_min_kW=min(coefficients),baseline_max_kW=max(coefficients),
        independent_price_history_min=float(calibration.min()),independent_price_history_max=float(calibration.max()),
        arrays_bitwise_equal=True,conditional_data_ready=True,historical_fit_ready=False,
        numerical_likelihood_ready=False,runtime_admitted=False,blind_h3_acceptance=False,scientific_acceptance=False,
        pc_invocation_charged_seconds=duration,pc_ledger_charged_seconds=ledger['charged_seconds'],
        environment=manifest['environment'],limitations=[
            'Demand workbook not included: original household history/OLS coefficients/power maximum are not independently recomputed; provenance follows hash-verified runner and audited workbook hashes.',
            'Unknown earlier final-week exposure; development evaluation only.',
            'Nominal physical/noise/prior assumptions, not historical physical calibration.',
            'No FV likelihood, precision or full-work-graph runtime admission.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--prices',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,default=OUT/'f4_h2_remaining_budget.json')
    parser.add_argument('--git-evidence',type=Path,default=ROOT/'root/f4/h2_compact_data_git_evidence_20261007.json')
    parser.add_argument('--known-ledger',type=Path,default=OUT/'f4_h2_compact_data_ledger_after_20261006.json')
    args=parser.parse_args()
    require(not args.output.exists(),'Preserve completed audit; no automatic overwrite.')
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[name]='1'
    from src.exe.run_model.bm_1_h2_native import native_reservation
    from src.tools.bm_1_reports import _atomic_report
    with native_reservation(args.campaign,120.,'compact-data-raw-review',begun=BEGUN) as reservation:
        result=audit(args.archive,args.prices,args.git_evidence,args.known_ledger)
        from time import time
        require(time()<reservation['deadline'],'Read-only audit exceeded compute allowance.')
        result['audit_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['git_evidence_sha256']=hashlib.sha256(args.git_evidence.read_bytes()).hexdigest()
        result['audit_seconds_before_ledger_close']=perf_counter()-BEGUN
        _atomic_report(args.output,result)
    ledger=json.loads(args.campaign.read_text())
    print(json.dumps(dict(passed=True,output=str(args.output),H_charged_seconds=ledger['charged_seconds'],
        audit_charged_seconds=ledger['entries'][-1]['charged_seconds'])))


if __name__=='__main__':
    main()
