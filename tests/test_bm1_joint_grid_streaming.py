"""Bounded joint-grid reduction preserves full conditional Bayes mixtures."""
import unittest
from unittest.mock import patch

import numpy as np

from src.validation.bm_1 import estimation as core
from src.validation.bm_1 import joint_estimation as fv
from src.validation.bm_1 import joint_grid as grid


def group(pair=('rho','Xi'), counts=(3,5,9), cells=11):
    case=core.build_f4_b_case(Xi=.75 if pair[0]=='rho' else 1.)
    seq=core.generate_synthetic_sequence(case,sequence_id=1356 if pair[0]=='rho' else 1357)
    return dict(pair=pair,sequence_id=seq.sequence_id,case=case,observations=seq.observations,
                hypotheses=grid._parameter_grid(pair,counts[-1],counts[-1])[0],counts=counts,
                label='middle',prior_bridge=False,
                settings=dict(x_steps=cells,max_step=1/48,likelihood_quadrature_order=3))


class StreamingGridTests(unittest.TestCase):
    def assert_rows(self, got, want):
        self.assertEqual(len(got),len(want))
        for a,b in zip(got,want):
            self.assertEqual(set(a),set(b))
            for key in a:
                if key in ('conditional_seconds','maximum_abs_mass_error'): continue
                if key in ('settings','coordinates','label','prior_kind','fv_label','marginal_parameters','fixed_parameters','map_hypotheses'):
                    self.assertEqual(a[key],b[key],key)
                else:
                    np.testing.assert_allclose(a[key],b[key],rtol=0,atol=1e-10,err_msg=key)

    def test_extreme_scores_and_distinct_states_match_batch_all_prefixes(self):
        rng=np.random.default_rng(7041)
        for pair in core.F4_G_PAIRS:
            g=group(pair);conditionals=[]
            for j in range(len(g['hypotheses'])):
                pred=rng.dirichlet(np.ones(11),size=24);post=rng.dirichlet(np.ones(11),size=24)
                # Huge common offsets plus changing dominance stress log rescaling.
                scores=rng.normal(0,8,24)+np.where(np.arange(24)%2,1000.,-1000.)
                conditionals.append(dict(scores=scores,predicted=pred,posterior=post,
                                         maximum_abs_mass_error=4e-16,minimum_cell_mass=min(pred.min(),post.min()),
                                         actual_max_step=1/49,elapsed_seconds=.1))
            reducer=grid._StreamingGridReduction(g)
            for j,c in enumerate(conditionals): reducer.add(j,c)
            self.assert_rows(reducer.finish(),grid._reduce_grids(g,conditionals))
            # No conditional trajectory list survives reduction.
            for bank in reducer.banks:
                self.assertEqual(bank['numerator'].shape,(2,24,11))

    def test_incomplete_duplicate_and_missing_observations_are_rejected(self):
        g=group();r=grid._StreamingGridReduction(g)
        with self.assertRaises(RuntimeError): r.finish()
        with self.assertRaises(RuntimeError): r.add(1,{})
        bad=dict(g,prior_bridge=True)
        with self.assertRaises(ValueError): grid._StreamingGridReduction(bad)

    def test_serial_and_spawn_match_fresh_batch_replays(self):
        groups=[group(pair,counts=(2,3,5),cells=11) for pair in core.F4_G_PAIRS]
        batch=fv._run_groups(groups,1,reducer=grid._reduce_grids)
        for workers in (1,2):
            progress=[]
            streamed=fv._run_groups_streaming(groups,workers,progress.append,grid._StreamingGridReduction)
            for got,want in zip(streamed,batch): self.assert_rows(got,want)
            self.assertEqual([p['completed'] for p in progress],list(range(1,51)))

    def test_pool_submission_is_bounded_and_failures_propagate(self):
        g=group();events=[]
        class Future:
            def __init__(self,task): self.task=task
            def result(self):
                events.append('consume')
                raise RuntimeError('deliberate replay failure')
        class Pool:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def submit(self,fn,task): events.append('submit');return Future(task)
        with patch.object(fv,'ProcessPoolExecutor',Pool):
            with self.assertRaisesRegex(RuntimeError,'deliberate replay failure'):
                fv._run_groups_streaming([g],3,None,grid._StreamingGridReduction)
        self.assertEqual(events,['submit']*6+['consume'])

    def test_lapack_conditional_matches_reference_and_bad_backend_rejected(self):
        for pair in core.F4_G_PAIRS:
            g=group(pair,counts=(2,3,5),cells=31)
            for j in (0,12,24):
                a=fv._candidate_replay((0,j,g))[2]
                b=fv._candidate_replay((0,j,dict(g,backend='lapack')))[2]
                for field in ('scores','predicted','posterior','actual_max_step'):
                    np.testing.assert_allclose(a[field],b[field],atol=1e-10,rtol=0)
        with self.assertRaisesRegex(ValueError,'backend'):
            fv._candidate_replay((0,0,dict(g,backend='invalid')))


if __name__=='__main__': unittest.main()
