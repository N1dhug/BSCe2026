import base64,hashlib,json,pathlib,time,zipfile
import numpy as np
from src.method.bm_1_g7 import OwnedBankPriceEvaluator
from src.method.bm_1_g7_jobs import digest
from src.method.bm_1_g8 import G8BankOrigin
from src.method.bm_1_distribution_panel import baseline,target
from src.control.bm_1 import BM1ControlNumerics
from src.control import objective as tracking
start=time.perf_counter()
out=pathlib.Path('/workspace/scratch/783a396a9c9c/g7_node_rate_verified_run')
reg=json.loads((out/'registration.json').read_text())
ledger=json.loads((out/'campaign/ledger.json').read_text())
assert len(ledger['entries'])==12 and all(e['status']=='complete' for e in ledger['entries'])
assert len({e['job_id'] for e in ledger['entries']})==12 and all(e['attempt']==1 for e in ledger['entries'])
records=[json.loads(p.read_text()) for p in (out/'campaign/nodes').glob('*.json')]
assert len(records)==12
for r in records:
 assert digest(r['payload'])==r['payload_sha256']
 assert any(e['job_id']==digest(r['job']) for e in ledger['entries'])
results={}
for cells,label in [(513,'fine'),(257,'coarse')]:
 o=G8BankOrigin.from_verified_bytes((out/'inputs'/(label+'.json')).read_bytes(),reg['input_files'][label]['sha256'])
 e=OwnedBankPriceEvaluator(o,baseline=baseline,target=target,horizon=(12.,14.),price_edges=[12.,13.,14.],
    numerics=BM1ControlNumerics(x_steps=cells,objective_max_step=1/96,solver_max_step=reg['solver_steps'][str(cells)]))
 contract=[p for p in (out/'campaign/contracts').glob('*.json')
           if json.loads(p.read_text())['per_node'][0]['x'] == e.supports[0].x.tolist()][0]
 identity=contract.stem
 assert digest(json.loads(contract.read_text()))==identity
 node_records=sorted([r for r in records if r['job'].get('contract')==identity],key=lambda r:r['job']['index'])
 assert len(node_records)==5
 max_moment_error=0.
 for j,r in enumerate(node_records):
  payload=r['payload']; s=e.supports[j];p=s.parameters_for_prices([.5,.5])
  mass=np.asarray(payload['mass'])
  np.testing.assert_array_equal(np.asarray(payload['coefficients'])[0],o.conditional_coefficients[j])
  mean,second,_=tracking.discrete_demand_moments(tracking.demand_grid(s.model,p,s.midpoints,s.x),mass[1:-1])
  max_moment_error=max(max_moment_error,float(np.max(abs(mean-payload['expected_demand']))),
                       float(np.max(abs(second-payload['second_moment']))))
 assert max_moment_error<=1e-12
 reduced=e.combine_nodes([.5,.5],[r['payload'] for r in node_records])
 stored=json.loads((out/(label+'_bank.json')).read_text())
 assert abs(reduced['total']-stored['total'])<=1e-12
 np.testing.assert_array_equal(reduced['conditional_mass'],np.asarray(stored['conditional_mass']))
 np.testing.assert_array_equal(reduced['conditional_coefficients'],np.asarray(stored['conditional_coefficients']))
 assert all(r['job']['prices']==[.5,.5] for r in node_records)
 results[label]=dict(verified_nodes=5,verified_exact_start_coefficients=True,moment_error=max_moment_error,
                     reduced_J=reduced['total'],bank_payload_identical=True,
                     node_forward_seconds=[r['payload']['measurement']['forward_seconds'] for r in node_records])
assert all(hashlib.sha256((out/'frozen_source'/p).read_bytes()).hexdigest()==h
           for p,h in json.loads((out/'source_manifest.json').read_text()).items())
# Scientific data only. Source and original input bytes already persist in Git:
# failed raw archive + execution commits. Do not duplicate repo-backed code.
selected=[p for p in out.rglob('*') if p.is_file() and not any(x in ('frozen_source','inputs') for x in p.relative_to(out).parts)
          and p.name!='writer.lock']
z=pathlib.Path('/workspace/scratch/783a396a9c9c/G7_M254_node_runtime_20261007_data.zip')
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
 for p in selected:archive.write(p,str(p.relative_to(out)))
files={str(p.relative_to(out)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in selected}
with zipfile.ZipFile(z) as archive:
 assert set(archive.namelist())==set(files)
 assert all(hashlib.sha256(archive.read(p)).hexdigest()==v['sha256'] for p,v in files.items())
report=dict(schema='g7-owned-node-runtime-raw-audit-v1',read_only=True,new_forwards=0,
    registered_inputs_verified=2,source_hashes_verified=41,ledger_entries=12,completed_nodes=10,
    banks=results,raw_data_archive=dict(path=z.name,sha256=hashlib.sha256(z.read_bytes()).hexdigest(),
        bytes=z.stat().st_size,files=files),
    original_input_source_archive='files/output/bm_1/runs/g7_node_runtime_failed_20261007.raw.zlib',
    execution_commit='872e77714d1bc5848d8dce5b1fac058417b796cf',
    seconds=time.perf_counter()-start)
pathlib.Path('root/g7/node_runtime_raw_audit_20261007.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='raw_data_archive'},indent=2))
print(json.dumps(dict(zip_bytes=z.stat().st_size,zip_path=str(z))))
