"""Post-run raw review/export; stdlib only, no solver imports or forwards."""
from pathlib import Path
import hashlib,json,math,time,zipfile,os
root=Path(__file__).resolve().parents[4]
run=root/'g7_bounded_quad_recovery_run';old=root/'g7_bounded_price_inputs_fixed_run';native=root/'g7_native_recovered';seed=root/'g7_seed_repair_run'
start=time.perf_counter()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
digest=lambda v:hashlib.sha256(canonical(v)).hexdigest()
reg=json.loads((run/'registration.json').read_bytes());pm=json.loads((run/'partial_manifest.json').read_bytes());result=json.loads((run/'result.json').read_bytes());inv=json.loads((run/'invocation.json').read_bytes())
assert sha(old/'registration.json')==pm['source_registration_sha256'] and sha(old/'campaign/ledger.json')==pm['source_ledger_sha256']
ledger=json.loads((run/'campaign/ledger.json').read_bytes());assert len(ledger['entries'])==5 and all(e['status']=='complete' and e['attempt']==1 for e in ledger['entries'])
new=[];rows=[]
for e in ledger['entries']:
 p=run/'campaign/nodes'/(e['job_id']+'.json');r=json.loads(p.read_bytes());assert digest(r['job'])==e['job_id'] and digest(r['payload'])==r['payload_sha256'];rows.append(dict(job_id=e['job_id'],job=r['job'],receipt_sha256=sha(p),payload_sha256=r['payload_sha256']))
 if r['job']['kind']=='missing_quad_nodes':new.append(r)
assert sorted(r['job']['index'] for r in new)==[3,4] and result['quad_new_node_indices']==[3,4] and result['quad_reused_node_indices']==[0,1,2]
reused=[];sources={}
for e in pm['entries']:
 if e['job']['kind']=='quadrature_selected':
  p=old/'campaign/nodes'/(e['job_id']+'.json');assert sha(p)==e['receipt_sha256'];r=json.loads(p.read_bytes());assert r['job']==e['job'] and digest(r['payload'])==e['payload_sha256'];reused.append(r);sources['reused_partial/campaign/nodes/'+p.name]=p
nodes=sorted(reused+new,key=lambda r:r['job']['index']);assert [r['job']['index'] for r in nodes]==list(range(5));cid=json.loads((run/'quad_setup.json').read_bytes())['contract_id'];assert all(r['job']['contract']==cid and r['job']['prices']==[.5,.5] for r in nodes)
setup=json.loads((run/'quad_setup.json').read_bytes());widths=setup['contract']['widths'];origin=setup['contract']['origin'];weights=result['selected_quadrature']['weights'];J=bias=within=between=0.;max_mass=0.;max_coeff=0.;npoints=len(widths)
for j,r in enumerate(nodes):
 p=r['payload'];assert p['index']==j and p['times']==setup['contract']['times'] and p['mass'][0]==origin['conditional_mass'][j] and p['coefficients'][0]==origin['conditional_coefficients'][j]
 assert all(math.isfinite(x) and x>=-1e-12 for row in p['mass'] for x in row)
 max_mass=max(max_mass,max(abs(sum(row)-1.) for row in p['mass']));max_coeff=max(max_coeff,max(abs(sum(row)-1.) for row in p['coefficients']))
 cost=sum(h*(s-m+.25) for h,m,s in zip(widths,p['expected_demand'],p['second_moment']));assert abs(cost-p['node_total'])<1e-10
 J+=weights[j]*cost
for t in range(npoints):
 means=[r['payload']['expected_demand'][t] for r in nodes];seconds=[r['payload']['second_moment'][t] for r in nodes];mean=sum(w*m for w,m in zip(weights,means));bias+=widths[t]*(mean-.5)**2;within+=widths[t]*sum(w*(s-m*m) for w,m,s in zip(weights,means,seconds));between+=widths[t]*sum(w*(m-mean)**2 for w,m in zip(weights,means))
s=result['selected_quadrature'];assert max_mass<1e-9 and max_coeff<1e-9
for key,value in [('J',J),('bias',bias),('within_parameter_variance',within),('between_parameter_variance',between)]:assert abs(s[key]-value)<1e-10
assert abs(J-bias-within-between)<1e-10 and result['forward_gates_passed'] and result['new_projection_gate_passed'] and not result['numerical_validation_passed'] and not result['scientific_accept'] and inv['within_scope_ceiling'] and inv['additional_charge']<=110
review=dict(schema='g7-completed-quad-raw-review-v1',passed=True,complete_new_receipts=5,new_numerical_forwards=2,reused_new_forwards=13,full_quad_indices=list(range(5)),J=J,bias=bias,within=within,between=between,maximum_trajectory_mass_error=max_mass,maximum_coefficient_sum_error=max_coeff,source_partial_journal_unchanged=True,forward_pairs=result['forward_pairs'],missing_projection_audit_indices=[0,1,2],numerical_validation_passed=False,scientific_accept=False,stdlib_only=True,forward_replays=0,review_and_final_export_reserve_seconds=10)
(run/'raw_review.json').write_text(json.dumps(review,indent=2)+'\n');(run/'new_receipt_manifest.json').write_text(json.dumps(dict(schema='g7-recovery-complete-receipt-index-v1',registration_sha256=sha(run/'registration.json'),ledger_sha256=sha(run/'campaign/ledger.json'),entries=rows),indent=2)+'\n')
for name in ('registration.json','selection.json','quad_setup.json','campaign/ledger.json'):
 sources['reused_partial/'+name]=old/name
for label in ('coarse','fine'):
 sources['original_owned/'+label+'_bank.json']=native/(label+'_bank.json');sources['original_owned/inputs/'+label+'.json']=native/'inputs'/(label+'.json')
for name in ('registration.json','source_manifest.json','campaign/ledger.json'):
 sources['original_owned/'+name]=native/name
sources['reused_fv/fv_fine_bank.json']=seed/'fv_fine_bank.json';sources['reused_fv/fv_fine_setup.json']=seed/'fv_fine_setup.json'
for name in json.loads((run/'native_manifest.json').read_bytes()):sources['frozen_source/'+name]=native/'frozen_source'/name
(run/'raw_source_index.json').write_text(json.dumps(dict(files={n:dict(sha256=sha(p),bytes=p.stat().st_size) for n,p in sources.items()},prior_partial_archive=dict(sha256=pm['source_archive_sha256'],library_file_id='libfile_442960f1658c8191b40d38b2501115c6'),original_archive=dict(sha256='ad730a148e770fffeeccdfcfff13af30fc33510d0bbcba9b988c15767af1e80d',library_file_id='libfile_51a15bcce7c48191a7cabbd96b0c20f3'),unchosen_coarse_receipts='All10 are in the preserved partial archive and externally pinned partial_manifest.json; not replayed.'),indent=2)+'\n')
archive=root/'G7_bounded_price_complete_quad_review_20261007.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in sorted(run.rglob('*')):
  if p.is_file() and p.name!='writer.lock':z.write(p,'recovery/'+str(p.relative_to(run)))
 for n,p in sorted(sources.items()):z.write(p,n)
 z.write(Path(__file__),'review_quad_recovery.py')
with archive.open('rb') as stream:os.fsync(stream.fileno())
fd=os.open(archive.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
elapsed=time.perf_counter()-start;assert elapsed<=10, 'Raw review/export exceeded registered reserve.'
index=dict(schema='g7-complete-quad-archive-index-v1',path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),review_and_final_export_seconds=elapsed,raw_review_sha256=sha(run/'raw_review.json'),invocation_sha256=sha(run/'invocation.json'),registration_sha256=sha(run/'registration.json'),campaign_debit=inv['campaign_debit'],additional_charge=inv['additional_charge'],archive_review_timing_external=True)
(root/'g_price_launcher/root/g7/bounded_quad_recovery_20261007/final_archive_index.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps(dict(review=review,index=index),indent=2))
