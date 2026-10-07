"""Read-only audit/extraction of G7 night evidence. No model forward or optimizer."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import zlib

import numpy as np


def unpack(bundle_path, destination=None):
    bundle=json.loads(Path(bundle_path).read_text())
    raw=zlib.decompress(base64.b64decode(bundle['payload'],validate=True))
    assert len(raw)==bundle['uncompressed_bytes']
    assert hashlib.sha256(raw).hexdigest()==bundle['uncompressed_sha256']
    files=json.loads(raw)
    assert list(files)==bundle['files']
    for name,entry in files.items():
        assert Path(name).name==name
        content=entry['content']
        text=(json.dumps(content,indent=2)+'\n') if name=='invocation.json' else json.dumps(content,separators=(',',':'),allow_nan=False)
        assert hashlib.sha256(text.encode()).hexdigest()==entry['sha256'],name
        if destination is not None:
            out=Path(destination);out.mkdir(parents=True,exist_ok=True)
            target=out/name
            if target.exists(): assert target.read_text()==text
            else: target.write_text(text,encoding='utf-8')
    return {name:entry['content'] for name,entry in files.items()}


def audit(bundle_path, packet_path):
    start=time.perf_counter();files=unpack(bundle_path)
    packet=json.loads(Path(packet_path).read_text());report=files['report.json']
    assert report['all_six_precision_pass'] and report['status']=='complete'
    baseline_t=np.array(packet['baseline_time_hours']);baseline=np.array(packet['baseline'])
    rows=[];forwards=0
    for name in report['units'].values():
        item=files[name]
        if 'search' in item: forwards+=item['search']['actual_forwards']
        else: forwards+=item['actual_forwards']
    for name in packet['models']:
        for mode in ('free','neutral'):
            item=files[name+'_'+mode+'_fine.json'];f=item['forward']
            dt=np.diff(f['time_hours']);D=np.array(f['demand']);A=np.array(f['target']);res=np.sqrt(dt)*(D-A)
            J=float(res@res);E=float(dt@(D-A))
            assert abs(J-f['objective_normalized'])<1e-13
            assert abs(E-item['energy_gap'])<1e-13
            assert abs(J*packet['scales']['demand_scale']**2-f['objective_raw'])<1e-12
            assert item['precision']['passed']
            B=np.interp(f['midpoints'],baseline_t,baseline)
            JB=float(dt@((B-A)**2))
            assert abs(dt@(B-A))<1e-12
            if mode=='neutral': assert abs(E)<1e-5
            rows.append(dict(model=name,mode=mode,J=J,energy_gap=E,
                prices=f['block_prices'],raw_prices=[packet['scales']['price_min']+packet['scales']['price_scale']*u for u in f['block_prices']],
                terminal_change=item['terminal_change'],baseline_J=JB,
                baseline_reduction_percent=100*(1-J/JB),precision=item['precision'],
                discrete_balance_error=item['balance_residual']))
    result=dict(schema='g7-night-read-only-audit-v1',raw_files=len(files),actual_forwards=forwards,
                rows=rows,all_six_pass=True,raw_byte_hashes_verified=True,
                report_unit_seconds=sum(e['charged_seconds'] for e in report['ledger']),
                process_seconds=files['invocation.json']['process_seconds'],
                independent_audit_seconds=time.perf_counter()-start)
    return result,files


def plot(files,packet,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#2474a6','#d17b17','#218b69'];labels=['BM linear','BM G4','WM4']
    x=np.array(packet['baseline_time_hours']);B=np.array(packet['baseline']);A=np.array(packet['target'])
    axes[0,0].plot(x,B,':',color='grey',label='Baseline B')
    axes[0,0].plot(x,A,'--',color='black',label='Fixed target A')
    for j,(name,label,col) in enumerate(zip(packet['models'],labels,colors)):
        f=files[name+'_neutral_fine.json']['forward'];t=np.array(f['midpoints']);D=np.array(f['demand'])
        axes[0,0].plot(t,D,color=col,label=label,linewidth=1.2)
        axes[0,1].plot(t,D-np.interp(t,x,B),color=col,label=label)
        u=f['block_prices'];axes[1,0].step([0,6,12,18,24],u+[u[-1]],where='post',color=col,label=label)
        free=files[name+'_free_fine.json']['forward']['objective_normalized'];neutral=f['objective_normalized']
        axes[1,1].bar(j-.18,free,.34,color=col,alpha=.45)
        axes[1,1].bar(j+.18,neutral,.34,color=col)
    axes[0,0].set(title='Preserved total demand: resulting profiles',ylabel='Normalized demand')
    axes[0,0].legend(fontsize=8,ncol=2)
    axes[0,1].set(title='Redistribution relative to baseline',ylabel='Demand minus B')
    axes[0,1].axhline(0,color='grey',linewidth=.6)
    axes[1,0].set(title='Prices with preserved total demand',ylabel='Common price u',ylim=(-.03,1.03))
    axes[1,0].legend(fontsize=8)
    axes[1,1].set(title='Tracking cost: free (pale) / conserved (solid)',ylabel='J (normalized squared demand × hours)',xticks=range(3),xticklabels=labels,ylim=(0,.33))
    for ax in (axes[0,0],axes[0,1],axes[1,0]): ax.set(xlim=(0,24),xticks=[0,6,12,18,24],xlabel='Physical hours')
    for ax in axes.flat: ax.grid(alpha=.15)
    fig.suptitle('G7 frozen January parameters — 1 February 2022, 24 h\nNumerical model-conditional comparison; no causal or physical ranking',fontsize=12)
    fig.savefig(out)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--packet',required=True)
    p.add_argument('--output',required=True);p.add_argument('--figure');p.add_argument('--extract')
    a=p.parse_args();result,files=audit(a.bundle,a.packet)
    if a.extract: unpack(a.bundle,a.extract)
    if a.figure: plot(files,json.loads(Path(a.packet).read_text()),a.figure)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
