"""Read-only review of saved fine chain and contours; zero native propagation."""
from time import perf_counter,time
BEGUN=perf_counter()
import os,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from root.f4.check_h2_terminal_bank_extension import read,write,sha,ordered,PRIOR_SECONDS,CAMPAIGN,RUN
OUT=RUN.parent/'f4_h2_bank_fine_control_20261007'

def main():
    ledger=read(CAMPAIGN)
    entry=dict(label='native-development|terminal-bank-fine-raw-review',status='reserved',started_utc_seconds=time(),charged_seconds=10.)
    assert ordered(e['charged_seconds'] for e in ledger['entries'] if e['label'].startswith('native-development|'))+10 <=1080
    ledger['entries'].append(entry); ledger['charged_seconds']=PRIOR_SECONDS+ordered(e['charged_seconds'] for e in ledger['entries']);write(CAMPAIGN,ledger)
    try:
        os.environ['OPENBLAS_NUM_THREADS']='1'
        import numpy as np
        from scipy.special import logsumexp
        from src.exe.run_model import bm_1_h2_compact as c
        from src.exe.run_model.bm_1_history import _beta_mass
        from src.validation.bm_1.interval_terminal_tilt import _terminal_candidate
        packet=read(RUN/'packet.json'); p=packet['prior']
        with np.errstate(divide='ignore'):
            previous=np.log(_beta_mass(np.linspace(0.,1.,2562),alpha=p['alpha'],beta=p['beta']))
        rows=read(OUT/'prior_chain.json');assert len(rows)==24
        for hour,row in enumerate(rows):
            path=OUT/f'prior_{hour:02d}.npz';assert row['hour']==hour and row['sha256']==sha(path)
            raw=np.load(path);assert np.array_equal(previous,raw['prior_log_state']);c.state_check(raw['log_state'])
            assert float(np.sum(np.abs(np.exp(raw['log_state'])-raw['predicted_mass']))) <= 1e-10
            previous=raw['log_state'].copy()
        saved=np.load(OUT/'result.npz');assert np.array_equal(previous,saved['prior_log_state'])
        selected=np.full(2561,-np.inf);guard=np.full(2561,np.inf);neutral=None;overlap_max=0.
        value=read(RUN/'jobs/6910062eb2676f4470c7b543ec9e30c88528773ba149c4e4bb5503476534819d.failed.json')['token']['value']
        rows=read(OUT/'contours.json');assert len(rows)==9
        for index,row in enumerate(rows):
            path=OUT/f'contour_{index:02d}.npz';assert sha(path)==row['sha256']
            raw=np.load(path);prediction={k:raw[k] for k in ('cf','log_amplitude','predicted_mass','center','tilt','span','frequencies')}
            prediction.update(center=float(prediction['center']),tilt=float(prediction['tilt']),span=float(prediction['span']))
            candidate=_terminal_candidate(prediction,value,packet['physical']['sigma_y'],packet['numerics']['guards']['conditional'])
            for key in ('density','relative','eligible','active','log_joint'):
                assert np.array_equal(candidate[key],raw[key]),key
            if neutral is None:neutral=raw['predicted_mass'].copy()
            else:assert np.array_equal(neutral,raw['predicted_mass'])
            eligible=candidate['eligible'];overlap=eligible & np.isfinite(selected)
            discrepancy=float(np.max(np.abs(candidate['log_joint'][overlap]-selected[overlap]))) if overlap.any() else 0.
            overlap_max=max(overlap_max,discrepancy);assert discrepancy<=1e-7
            better=eligible & (candidate['relative']<guard)
            selected[better]=candidate['log_joint'][better];guard[better]=candidate['relative'][better]
            assert np.array_equal(raw['unresolved'],candidate['active'] & ~np.isfinite(selected))
        active=candidate['active'];assert np.isfinite(selected[active]).all()
        score=float(logsumexp(selected));posterior=selected-score
        assert np.array_equal(posterior,saved['log_posterior_mass'])
        report=read(OUT/'report.json');assert score==report['log_evidence'] and report['status']=='passed'
        assert c.source_hashes()==read(RUN/'manifest.json')['source_sha256']
        review=dict(schema='f4-h2-bank-fine-raw-review-v1',status='passed',native_propagations=0,
            own_fine_prior_hours=24,contours=9,all_checksums_passed=True,own_prior_continuity=True,
            conditional_reconstruction_exact=True,posterior_reconstruction_exact=True,log_evidence=score,
            maximum_relative_guard=float(guard[active].max()),maximum_overlap_difference=overlap_max,
            result_sha256=sha(OUT/'result.npz'),elapsed_seconds=perf_counter()-BEGUN)
        assert perf_counter()-BEGUN<10
        write(OUT/'raw_review.json',review);print(json.dumps(review))
    finally:
        entry.update(status='closed',charged_seconds=perf_counter()-BEGUN)
        ledger['charged_seconds']=PRIOR_SECONDS+ordered(e['charged_seconds'] for e in ledger['entries']);write(CAMPAIGN,ledger)

if __name__=='__main__':main()
