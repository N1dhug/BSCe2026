"""One cheap integrated crash/resume/ownership/budget check; stub transport only."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.control.bm_1 import BM1ControlNumerics
from src.method.bm_1_g7 import OwnedBankPriceEvaluator
from src.method.bm_1_g7_jobs import CheckpointedBankControl,NodeJobStore,NodeJobStop,source_manifest,REQUIRED_SOURCE_PATHS
from src.method.bm_1_g7_budget import full_job_plan,price_job_plan
from src.method.bm_1_g8 import G8BankOrigin,context,RHO_VALUES
from src.method.bm_1_g8_bernstein import BASIS_KIND,prior_coefficients
from src.method.bm_1_distribution_panel import baseline,target


class G7NodeJobTests(unittest.TestCase):
    def test_crash_resume_fingerprints_complete_graph_and_own_bank(self):
        root=Path(__file__).resolve().parents[1]
        hashes=source_manifest(root,REQUIRED_SOURCE_PATHS)
        _,fixed,_,edges,_,projector=context(257,basis_kind=BASIS_KIND)
        c=np.stack([np.roll(prior_coefficients(),j*8) for j in range(5)])
        origin=G8BankOrigin(12.,fixed,RHO_VALUES,np.log([.1,.15,.3,.2,.25]),edges,
            c@projector.basis*np.diff(edges),backend='mixture',basis=projector.basis,
            conditional_coefficients=c,basis_kind=BASIS_KIND)
        evaluator=OwnedBankPriceEvaluator(origin,baseline=baseline,target=target,
            horizon=(12.,12.2),price_edges=[12.,12.1,12.2],
            numerics=BM1ControlNumerics(x_steps=257,objective_max_step=.1,solver_max_step=.01))
        calls=[];interrupt=[True]
        def transport(**kw):
            starts=kw['initial_coefficients'];j=next(j for j,row in enumerate(c) if np.array_equal(row,starts))
            calls.append(j)
            if j==1 and interrupt[0]:
                interrupt[0]=False;raise KeyboardInterrupt('Simulated process loss inside node1')
            coeff=np.tile(starts,(len(kw['t_eval']),1));coeff[-1]=np.roll(starts,1)
            return dict(coefficients=coeff,density=coeff@kw['basis'],diagnostics={'stub':True})
        with TemporaryDirectory() as tmp,patch('src.method.bm_1_g7.evolve_projected_density',side_effect=transport):
            store=NodeJobStore(tmp,prior_debit=0.,ceiling=100.)
            run=CheckpointedBankControl(evaluator,store,source_root=root,source_sha256=hashes)
            with self.assertRaises(KeyboardInterrupt):run.evaluate([.3,.7],seconds_per_node=1.)
            self.assertEqual(calls,[0,1]);charged_before=store.charged()
            store2=NodeJobStore(tmp,prior_debit=0.,ceiling=100.)
            resumed=CheckpointedBankControl(evaluator,store2,source_root=root,source_sha256=hashes)
            with self.assertRaises(NodeJobStop):resumed.evaluate([.3,.7],seconds_per_node=1.)
            self.assertEqual(calls,[0,1]);self.assertEqual(charged_before,store2.charged())
            interrupted_id=store2.read()['entries'][-1]['job_id']
            result=resumed.evaluate([.3,.7],seconds_per_node=1.,retry_ids=[interrupted_id])
            self.assertEqual(calls,[0,1,1,2,3,4])
            before=store2.charged();again=resumed.evaluate([.3,.7],seconds_per_node=1.)
            self.assertEqual(calls,[0,1,1,2,3,4]);self.assertEqual(before,store2.charged())
            reference=evaluator.evaluate([.3,.7])
            np.testing.assert_array_equal(result['conditional_mass'],reference['conditional_mass'])
            np.testing.assert_array_equal(result['conditional_coefficients'],reference['conditional_coefficients'])
            self.assertEqual(result['total'],reference['total']);self.assertEqual(result['total'],again['total'])
            self.assertAlmostEqual(result['total'],origin.weights@result['node_costs'])
            np.testing.assert_allclose(result['demand_variance'],result['within_variance']+result['between_variance'],atol=1e-12)
            self.assertGreater(result['between_parameter_variance'],0.)
            np.testing.assert_array_equal(origin.conditional_coefficients,c)
            self.assertNotIn('_drift',evaluator.supports[1].model.__dict__)
            # Crash after atomic receipt, before ledger commit: recover file, no replay.
            count=len(calls)
            with patch.object(store2,'_finish',side_effect=KeyboardInterrupt('Loss after durable artifact')):
                with self.assertRaises(KeyboardInterrupt):resumed.evaluate([.4,.6],seconds_per_node=1.)
            recovered=resumed.evaluate([.4,.6],seconds_per_node=1.)
            self.assertEqual(len(calls),count+5)
            self.assertTrue(any(e['status']=='recovered-complete' for e in store2.read()['entries']))
            self.assertNotEqual(recovered['total'],result['total'])
            with self.assertRaises(ValueError):NodeJobStore(tmp,prior_debit=1.,ceiling=100.)
            with self.assertRaises(NodeJobStop):
                with store2.lock():
                    with store2.lock():pass
            # Mutated live grid cannot reuse the saved contract.
            old=evaluator.supports[0].solver_max_step;evaluator.supports[0].solver_max_step=.02
            with self.assertRaises(ValueError):resumed.evaluate([.3,.7],seconds_per_node=1.)
            evaluator.supports[0].solver_max_step=old
            bad=dict(hashes);bad['src/method/bm_1_g7.py']='0'*64
            with self.assertRaises(ValueError):CheckpointedBankControl(evaluator,store2,source_root=root,source_sha256=bad)
            # Payload corruption is a stop, not a cache miss/automatic replay.
            filename=next((Path(tmp)/'nodes').glob('*.json'));payload=json.loads(filename.read_text())
            job=payload['job'];payload['payload']['node_total']+=1.;filename.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):store2.receipt(filename,job)
        with TemporaryDirectory() as limited:
            store=NodeJobStore(limited,prior_debit=0.,ceiling=100.,max_new_seconds=.01)
            with store.lock(),self.assertRaises(NodeJobStop):
                store.node(dict(index=0,contract='stub'),1.,lambda deadline: {})
        plan=full_job_plan();rates={}
        admission=price_job_plan(plan,rates,campaign_debit=1118.580078)
        self.assertEqual(len(admission['unknown']),14);self.assertFalse(admission['budget_admitted'])
        for job in plan['jobs']:
            rates[job['work']]=dict(seconds=.01,complete=True,raw_reviewed=True,horizon_hours=24.,
                                   nodes=job['nodes'],receipt_sha256='stub',source_fingerprint='stub')
        self.assertTrue(price_job_plan(plan,rates,campaign_debit=1118.580078)['budget_admitted'])
        rates['em_coupled']['horizon_hours']=2.
        self.assertFalse(price_job_plan(plan,rates,campaign_debit=1118.580078)['budget_admitted'])
        plan['jobs']=[j for j in plan['jobs'] if j['work']!='cross_fv_fine']
        with self.assertRaises(ValueError):price_job_plan(plan,rates,campaign_debit=1118.580078)


if __name__=='__main__':unittest.main()
