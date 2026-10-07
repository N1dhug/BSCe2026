"""Price the recovered H2 workload using existing evidence; never run a model.

Empirical averages are a planning scenario, not permission to lower the protected
validation reserve. The maximum-rate scenario retains the original factor 1.5.
No native imports, data preparation, optimizer updates or campaign writes occur.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from time import perf_counter
import zipfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plan(run, audit_path):
    begun = perf_counter()
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'passed' or audit['raw_jobs'] != 1076:
        raise ValueError('This review is scoped to the audited 1076-job recovery.')
    jobs, groups, streams = {}, defaultdict(list), defaultdict(list)
    hashes = audit['source_job_sha256']
    if {p.name for p in (run/'jobs').glob('*.npz')} != set(hashes):
        raise ValueError('Snapshot changed; obtain a new raw audit first.')
    for name, expected in sorted(hashes.items()):
        raw = (run/'jobs'/name).read_bytes()
        if sha(raw) != expected:
            raise ValueError('Audited job changed: '+name)
        with zipfile.ZipFile(run/'jobs'/name) as z:
            row = json.loads(z.read('metadata.json'))['report']
        token = row['token']
        if token['identity'] != audit['numerical_manifest_identity']:
            raise ValueError('Mixed numerical identities.')
        seconds = row['elapsed_seconds']
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid measured duration.')
        jobs[name] = row
        groups[(token['grid'], token['kind'])].append(row)
        streams[(token['candidate'], token['grid'])].append(row)
    summary = {}
    for (grid, kind), rows in groups.items():
        values = sorted(r['elapsed_seconds'] for r in rows)
        contour_rows = [r for r in rows if 'contours' in r['result']]
        summary.setdefault(grid, {})[kind] = dict(
            count=len(values), mean=statistics.mean(values), median=statistics.median(values),
            p90=values[math.ceil(.9*len(values))-1], maximum=max(values),
            contours=dict(Counter(len(r['result']['contours']) for r in contour_rows)),
            mean_outside_contours=(statistics.mean(r['elapsed_seconds']-
                sum(c['elapsed_seconds'] for c in r['result']['contours'])
                for r in contour_rows) if contour_rows else None))
    complete = []
    for (key, grid), rows in streams.items():
        train = [r for r in rows if r['token']['hour'] < 72
                 and r['token']['kind'] in ('observed', 'prediction')]
        if grid == 'base' and key != 'linear' and len(train) == 72:
            complete.append(dict(key=key, seconds=sum(r['elapsed_seconds'] for r in train)))
    if audit['completed_objective_calls'] != {'2':12, '4':0}:
        raise ValueError('Optimizer inventory changed.')
    if list(audit['pending_stream_hours'].values()) != [34]:
        raise ValueError('Pending stream changed.')
    # Keep the inherited pilot/audit maxima as well as subsequent raw maxima.
    old_rates = json.loads((run/'budgeted_execution.json').read_text())['rates']
    caps = {g:{k:max(s['maximum'], old_rates[g][k]) for k,s in row.items()}
            for g,row in summary.items()}
    # Imported linear prefix contributes 24 prior + 3 TRAIN jobs. The saved
    # four-piece PROBE is the first positive coordinate of the frozen COBYQA
    # start and also reuses its 24 prior + 3 TRAIN jobs in the full-cap scenario.
    done_train = 3+sum(r['token']['grid']=='base' and r['token']['kind']=='observed'
                       and r['token']['hour'] < 72 for r in jobs.values())
    done_prior = 24+sum(r['token']['grid']=='base' and r['token']['kind']=='prediction'
                        and r['token']['hour'] < 24 for r in jobs.values())
    rows = [
        ('search_remaining_train', 'search', 49*48-done_train, 0, 'observed'),
        ('search_remaining_prior', 'search', 49*24-done_prior, 0, 'prediction'),
        ('nonlinear_base_selection', 'validation', 48, 0, 'observed'),
        ('nonlinear_fine_train_selection', 'validation', 0, 144, 'observed'),
        ('nonlinear_fine_prior', 'validation', 0, 48, 'prediction'),
        ('final_winner_and_linear', 'validation', 336, 336, 'observed'),
        ('remaining_selection_open_final_moments', 'validation', 720, 720, 'moments'),
        ('open_distributions', 'validation', 14, 14, 'cdf'),
    ]
    scenarios = {}
    for name, statistic, factor in [('empirical_mean', 'mean', 1.),
            ('empirical_mean_with_margin', 'mean', 1.5),
            ('observed_maximum_with_margin', 'maximum', 1.5)]:
        priced = []
        for label, role, base, fine, kind in rows:
            rate = lambda g: caps[g][kind] if statistic == 'maximum' else summary[g][kind][statistic]
            priced.append(dict(job=label, role=role, base_units=base, fine_units=fine,
                               seconds=factor*(base*rate('base')+fine*rate('fine'))))
        search = sum(r['seconds'] for r in priced if r['role']=='search')+300.
        validation = sum(r['seconds'] for r in priced if r['role']=='validation')+360.
        scenarios[name] = dict(rows=priced, search_seconds=search,
            validation_seconds=validation, remaining_hours=(search+validation)/3600.,
            factor=factor, admission=False)
    return dict(schema='f4-h2-remaining-work-review-v1',
        raw_audit_sha256=sha(audit_path.read_bytes()),
        numerical_manifest_identity=audit['numerical_manifest_identity'],
        verified_jobs=len(jobs), rates=summary, protected_rates=caps,
        complete_nonlinear_candidates=complete, scenarios=scenarios,
        limits=['Same 49-pair cap; completed and pending work deducted once.',
                'Mean rates include cold jobs; no unmeasured warm-up subtraction.',
                'Four-piece full TRAIN and final-week timing remain unmeasured.',
                'Final jobs assume all 168 hours observed and a distinct nonlinear winner.',
                'Empirical rates are not a validated new reserve or convergence prediction.',
                'Controller IO/optimizer and job-boundary timeouts may add work.',
                'Native runtime restoration, PC final export and I/J review are outside model rates.'],
        native_propagations=0, additional_search_admitted=False,
        optimizer_convergence_claim=False, scientific_acceptance=False,
        read_only_seconds=perf_counter()-begun)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier planning evidence.')
    report = plan(args.run, args.audit)
    report['source_sha256'] = sha(Path(__file__).read_bytes())
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({name:row['remaining_hours'] for name,row in report['scenarios'].items()}))
