"""Rectangular Bayes reduction, raw gates and crash-safe resume behavior."""
import copy
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.validation.bm_1 import estimation as core, joint_estimation as fv
from src.validation.bm_1 import joint_grid as grid, joint_pilot as pilot
from src.validation.bm_1 import joint_diagnostics as hats
from src.exe.run_model.bm_1 import estimate


@contextmanager
def tiny_design():
    grids = {k:((2,3),(3,5)) for k in pilot.GRIDS}
    cells = {k:(11,21) for k in pilot.CELLS}
    def configs(pair):
        return [('middle',dict(x_steps=11,max_step=1/24,likelihood_quadrature_order=3)),
                ('time_fine',dict(x_steps=11,max_step=1/48,likelihood_quadrature_order=3)),
                ('spatial_fine',dict(x_steps=21,max_step=1/24,likelihood_quadrature_order=3)),
                ('quadrature_fine',dict(x_steps=11,max_step=1/24,likelihood_quadrature_order=5))]
    with patch.object(pilot,'DAYS',1), patch.object(pilot,'ANCHOR',5), \
         patch.object(pilot,'GRIDS',grids), patch.object(pilot,'CELLS',cells), \
         patch.object(pilot,'configurations',side_effect=configs), \
         patch.object(grid,'G2A_IDS',(90030,90031)):
        yield


def flat_replay(task):
    i,j,g = task
    k,cells = len(g['observations'].times),g['settings']['x_steps']
    mass = np.full((k,cells),1/cells)
    return i,j,dict(scores=np.zeros(k),predicted=mass,posterior=mass.copy(),
        maximum_abs_mass_error=3e-16,minimum_cell_mass=1/cells,
        actual_max_step=g['settings']['max_step'],elapsed_seconds=.01)


class RectangularMathTests(unittest.TestCase):
    def test_fixed_design_counts_and_nested_physical_prior(self):
        self.assertEqual(pilot.design_header()['expected_conditional_runs'],42312)
        self.assertEqual(len(pilot.units()),8)
        for pair in core.F4_G_PAIRS:
            coarse,fine = pilot.GRIDS['__'.join(pair)]
            a,ua,ea,pa = pilot.parameter_grid(pair,coarse)
            b,ub,eb,pb = pilot.parameter_grid(pair,fine)
            selected = b.reshape(*fine,2)[::2,::2].reshape(-1,2)
            np.testing.assert_array_equal(a,selected)
            self.assertAlmostEqual(np.exp(pa).sum(),1.)
            for nodes,edges in zip(ua,ea):
                q = np.diff(edges)
                self.assertAlmostEqual(q[0],1/(2*(len(nodes)-1)))
                np.testing.assert_allclose(q[1:-1],1/(len(nodes)-1),atol=1e-16)
        with self.assertRaises(ValueError):pilot.parameter_grid(core.F4_G_PAIRS[0],(8,65))

    def test_rectangular_continuous_uniform_moments_and_square_equivalence(self):
        law = hats.tensor_hat_law(np.zeros((3,15)),(3,5))
        np.testing.assert_allclose(law['density'],1,atol=1e-15)
        np.testing.assert_allclose(law['mean'],.5,atol=1e-15)
        np.testing.assert_allclose(law['sd'],1/np.sqrt(12),atol=1e-15)
        np.testing.assert_allclose(law['covariance'],0,atol=1e-15)
        rng = np.random.default_rng(8003);scores=rng.normal(size=(4,25))
        a=hats.rectangular_hat_law(scores,(3,5))
        b=hats.tensor_hat_law(scores.reshape(4,5,5)[:,::2,:].reshape(4,15),(3,5))
        for key in ('density','mean','sd','evidence','covariance'):
            np.testing.assert_array_equal(a[key],b[key])

    def test_partial_restore_and_final_sum_match_independent_batch(self):
        with tiny_design():
            g=pilot.make_group(pilot.units()[0])
            rng=np.random.default_rng(12945);results=[]
            for j in range(15):
                pred=rng.dirichlet(np.ones(11),size=24);post=rng.dirichlet(np.ones(11),size=24)
                results.append(dict(scores=rng.normal(0,8,24)+np.where(np.arange(24)%2,1000.,-1000.),
                    predicted=pred,posterior=post,minimum_cell_mass=min(pred.min(),post.min()),
                    maximum_abs_mass_error=1e-15,actual_max_step=1/25,elapsed_seconds=.1))
            r=pilot.RectangularReduction(g)
            for j in range(7):r.add(j,results[j])
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'checkpoint.npz'
                estimate._atomic_report(path,r.checkpoint())
                saved,_=estimate._read_likelihood_report(path)
                restored=pilot.RectangularReduction(g).restore(saved)
            for j in range(7,15):restored.add(j,results[j]);r.add(j,results[j])
            for row,bank in zip(restored.finish(),restored.banks):
                selected=[results[j] for j in bank['indices']]
                batch=fv._reduce(dict(g,hypotheses=bank['h'],initial_log_weights=bank['prior']),selected)
                for key in ('conditional_log_evidence','log_weights','cumulative_log_evidence',
                            'marginal_predicted_mass','marginal_posterior_mass'):
                    np.testing.assert_allclose(row[key],batch[key],atol=1e-10,rtol=0)
            for a,b in zip(r.finish(),restored.finish()):
                for key in ('log_weights','marginal_predicted_mass','marginal_posterior_mass'):
                    np.testing.assert_array_equal(a[key],b[key])

    def test_partial_corruption_order_and_incomplete_banks_are_rejected(self):
        with tiny_design(),patch.object(fv,'_candidate_replay',side_effect=flat_replay):
            g=pilot.make_group(pilot.units()[0]);r=pilot.RectangularReduction(g)
            with self.assertRaises(RuntimeError):r.finish()
            with self.assertRaises(RuntimeError):r.add(1,{})
            pilot.advance(r,7)
            original=r.checkpoint()
            changes=[lambda p:p.update(completed_hypotheses=16),
                lambda p:p['banks'][0]['scores'].__setitem__((0,0),1.),
                lambda p:p['banks'][1]['numerator'].__setitem__((0,0,0),-1.),
                lambda p:p['banks'][1]['denominator'].__setitem__((0,0),99.),
                lambda p:p['banks'][1].update(mass_error=1e-8),
                lambda p:p['banks'][1]['steps'].__setitem__(0,0.)]
            for change in changes:
                bad=copy.deepcopy(original);change(bad)
                with self.assertRaises(ValueError):pilot.RectangularReduction(g).restore(bad)

    def test_real_serial_spawn_and_split_resume_agree(self):
        with tiny_design():
            g=pilot.make_group(pilot.units()[0])
            serial=pilot.RectangularReduction(g);pilot.advance(serial,15)
            split=pilot.RectangularReduction(g);pilot.advance(split,7)
            split=pilot.RectangularReduction(g).restore(split.checkpoint())
            with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
                pilot.advance(split,15,pool,workers=2)
            self.assertTrue(pilot.validate_rows(split.finish(),g))
            for a,b in zip(serial.finish(),split.finish()):
                for key in ('conditional_log_evidence','log_weights','marginal_predicted_mass','marginal_posterior_mass'):
                    np.testing.assert_allclose(a[key],b[key],atol=1e-13,rtol=0)

    def test_worker_failure_keeps_the_published_predecessor_and_queue_bounded(self):
        with tiny_design():
            reducer=pilot.RectangularReduction(pilot.make_group(pilot.units()[0]))
            events=[]
            class Future:
                def result(self):
                    events.append('consume');raise RuntimeError('deliberate worker failure')
            class Pool:
                def submit(self,*args):events.append('submit');return Future()
            with self.assertRaisesRegex(RuntimeError,'deliberate worker failure'):
                pilot.advance(reducer,15,Pool(),workers=3)
            self.assertEqual(events,['submit']*6+['consume'])
            self.assertEqual(reducer.seen,0)


class RectangularCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tiny_design():
            cls.truth={p:grid._g2a_truth_record(p,sid,1) for p,sid in zip(core.F4_G_PAIRS,grid.G2A_IDS)}

    @contextmanager
    def fixture(self):
        previous=dict(source_commit='b'*40,dependencies={'g1c0':'chain'})
        with tiny_design(),patch.object(grid,'_g2a_truth_record',side_effect=lambda p,s,d:self.truth[tuple(p)]), \
             patch.object(estimate,'_read_bank_dependency',return_value=(previous,'c'*64)), \
             patch.object(estimate,'_source_snapshot',return_value=dict(source_commit='a'*40,working_tree_dirty=False)), \
             patch.object(fv,'_candidate_replay',side_effect=flat_replay) as replay,patch('builtins.print'):
            yield replay

    def arguments(self,directory):
        return estimate.build_parser().parse_args(['joint-bank-rectangular-pilot',
            '--g2a-report',str(Path(directory)/'previous.json.gz'),'--run-dir',str(Path(directory)/'run'),
            '--workers','1','--chunk-size','4','--limit-chunks','1'])

    def test_resume_collect_and_complete_design_never_replay_saved_nodes(self):
        with tempfile.TemporaryDirectory() as directory,self.fixture() as replay:
            args=self.arguments(directory)
            a=estimate.run_joint_rectangular_pilot(args)
            self.assertFalse(a['passes']);self.assertEqual(a['conditional_runs'],4)
            self.assertEqual(a['completed_units'],0)
            b=estimate.run_joint_rectangular_pilot(args)
            self.assertEqual(b['conditional_runs'],8);self.assertEqual(replay.call_count,8)
            args.collect_only=True
            c=estimate.run_joint_rectangular_pilot(args)
            self.assertEqual(c['partial_unit'],b['partial_unit']);self.assertEqual(replay.call_count,8)
            args.collect_only=False;args.limit_chunks=None
            done=estimate.run_joint_rectangular_pilot(args)
            self.assertTrue(done['passes']);self.assertTrue(done['completed'])
            self.assertEqual(done['completed_units'],8);self.assertEqual(len(done['comparisons']),14)
            self.assertFalse(done['releases_existing_g2b_g2c'])
            self.assertEqual(replay.call_count,120)
            self.assertFalse((args.run_dir/'checkpoint.npz').exists())
            args.collect_only=True
            collected=estimate.run_joint_rectangular_pilot(args)
            self.assertTrue(collected['passes']);self.assertEqual(replay.call_count,120)
            path=next((args.run_dir/'units').glob('*.npz'));bad,_=estimate._read_likelihood_report(path)
            bad['rows'][0]['bank_log_evidence'][0]+=1.
            estimate._atomic_report(path,bad,replace_existing=True)
            with self.assertRaises(ValueError):estimate.run_joint_rectangular_pilot(args)
            self.assertEqual(replay.call_count,120)

    def test_changed_source_dependency_and_partial_damage_block_resume(self):
        with tempfile.TemporaryDirectory() as directory,self.fixture() as replay:
            args=self.arguments(directory);estimate.run_joint_rectangular_pilot(args)
            with patch.object(estimate,'_source_snapshot',return_value=dict(source_commit='d'*40,working_tree_dirty=False)):
                with self.assertRaises(ValueError):estimate.run_joint_rectangular_pilot(args)
            with patch.object(estimate,'_read_bank_dependency',return_value=(dict(source_commit='b'*40,dependencies={}), 'e'*64)):
                with self.assertRaises(ValueError):estimate.run_joint_rectangular_pilot(args)
            path=args.run_dir/'checkpoint.npz';bad,_=estimate._read_likelihood_report(path)
            bad['reduction']['banks'][1]['scores'][0,0]+=1.
            estimate._atomic_report(path,bad,replace_existing=True)
            with self.assertRaises(ValueError):estimate.run_joint_rectangular_pilot(args)
            self.assertEqual(replay.call_count,4)

    def test_crash_before_final_unit_publish_resumes_full_checkpoint_without_rerun(self):
        with tempfile.TemporaryDirectory() as directory,self.fixture() as replay:
            args=self.arguments(directory);args.chunk_size=15
            atomic=estimate._atomic_report
            def fail_final(path,payload,**kwargs):
                if path.parent.name=='units':raise OSError('deliberate final publish failure')
                return atomic(path,payload,**kwargs)
            with patch.object(estimate,'_atomic_report',side_effect=fail_final):
                with self.assertRaises(OSError):estimate.run_joint_rectangular_pilot(args)
            self.assertEqual(replay.call_count,15)
            done=estimate.run_joint_rectangular_pilot(args)
            # The already completed first unit is reconstructed from its full
            # checkpoint; limit 1 then runs exactly one new following unit.
            self.assertEqual(done['completed_units'],2);self.assertEqual(replay.call_count,30)

    def test_dirty_source_rejected_before_loading_expensive_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.arguments(directory)
            with patch.object(estimate,'_source_snapshot',return_value=dict(source_commit='a'*40,working_tree_dirty=True)), \
                 patch.object(estimate,'_read_bank_dependency') as read:
                with self.assertRaises(ValueError):estimate.run_joint_rectangular_pilot(args)
                read.assert_not_called()

    def test_probability_failure_cannot_be_accepted_as_a_complete_pilot(self):
        with self.fixture():
            rows=[]
            for unit in pilot.units():
                g=pilot.make_group(unit,self.truth[unit[0]]);r=pilot.RectangularReduction(g)
                pilot.advance(r,15);rows.extend(r.finish())
            self.assertTrue(pilot.summarize(rows)['passes'])
            rows[0]['maximum_abs_mass_error']=2e-10
            self.assertFalse(pilot.summarize(rows)['passes'])
            with self.assertRaises(ValueError):
                from src.validation.bm_1 import joint_robustness
                joint_robustness.require_g2a(pilot.summarize(rows))


if __name__=='__main__':unittest.main()
