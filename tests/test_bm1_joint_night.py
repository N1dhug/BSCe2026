"""Scientific scope, independent reductions and bounded crash-safe night runs."""
import copy
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.exe.run_model.bm_1 import estimate
from src.validation.bm_1 import estimation as core, joint_estimation as fv
from src.validation.bm_1 import joint_grid as grid, joint_pilot as pilot
from src.validation.bm_1 import joint_night as night, joint_robustness as robustness
from tests.test_bm1_joint_pilot import tiny_design, flat_replay


@contextmanager
def small_night():
    with tiny_design(), patch.object(night, 'DAYS', 1), patch.object(night, 'PILOT_IDS', grid.G2A_IDS), \
         patch.object(night, 'GRIDS', pilot.GRIDS), patch.object(robustness, 'PILOT_IDS', grid.G2A_IDS), \
         patch.object(robustness, 'RECORD_DAYS', 1):
        yield


def demand_replay(task):
    i, j, group = task
    _, _, result = flat_replay(task)
    count = len(group['observations'].times)
    result.update(predicted_demand_mean=np.full(count, .3+.01*j),
                  posterior_demand_mean=np.full(count, .4+.02*j))
    return i, j, result


class NightMathTests(unittest.TestCase):
    def test_design_is_forty_units_with_seventy_controls_and_no_scope_cut(self):
        design = night.design_header()
        self.assertEqual(len(night.units()), 40)
        self.assertEqual(design['expected_conditional_runs'], 211560)
        self.assertEqual(design['expected_record_scenarios'], 10)
        self.assertEqual(design['sequence_ids'], [1360, 1361])
        self.assertEqual(design['parameter_grids'], pilot.design_header()['parameter_grids'])
        self.assertEqual(design['fv_settings'], pilot.design_header()['fv_settings'])

    def test_scenarios_share_truth_noise_and_use_exact_observation_subsets(self):
        with small_night():
            pair, sid = night.units()[0][:2]
            truth = grid._g2a_truth_record(pair, sid, 1)
            groups = {name:night.make_group((pair,sid,name,'middle'), truth) for name in night.SCENARIOS}
            base = groups['matched']
            for name, group in groups.items():
                indices = group['observation_indices']
                np.testing.assert_array_equal(group['observations'].values, base['observations'].values[indices])
                np.testing.assert_array_equal(group['observations'].times, base['observations'].times[indices])
                self.assertIs(group['truth_record'], base['truth_record'])
            self.assertEqual(groups['sampling_2h']['observation_indices'], list(range(1,24,2)))
            self.assertEqual(groups['sampling_4h']['observation_indices'], list(range(3,24,4)))
            self.assertEqual(groups['baseline_055']['case'].baseline, .55)
            self.assertEqual((groups['initial_beta_16_24']['case'].initial_alpha,
                              groups['initial_beta_16_24']['case'].initial_beta), (16.,24.))

    def test_demand_checkpoint_and_weights_match_independent_node_sum(self):
        def weighted_replay(task):
            i,j,result=demand_replay(task)
            result['scores']=.03*(j-7)*np.arange(1,len(result['scores'])+1)
            return i,j,result
        with small_night(), patch.object(fv,'_candidate_replay',side_effect=weighted_replay):
            group = night.make_group(night.units()[0])
            reducer = night.RectangularReduction(group)
            night.advance(reducer, 7)
            saved = reducer.checkpoint()
            restored = night.RectangularReduction(group).restore(saved)
            night.advance(restored, 15)
            rows = restored.finish()
            self.assertTrue(night.validate_rows(rows, group))
            for row, bank in zip(rows, restored.banks):
                indices = np.array(list(bank['indices']))
                for phase, intercept, slope in (('predicted', .3, .01), ('posterior', .4, .02)):
                    count=len(group['observations'].times)
                    cumulative=np.cumsum(np.arange(1,count+1))
                    if phase=='predicted':cumulative=np.r_[0,cumulative[:-1]]
                    raw=bank['prior']+.03*(indices[None,:]-7)*cumulative[:,None]
                    weights=np.exp(raw-raw.max(axis=1)[:,None])
                    weights/=weights.sum(axis=1)[:,None]
                    expected=np.sum(weights*(intercept+slope*indices),axis=1)
                    np.testing.assert_allclose(row['marginal_'+phase+'_demand_mean'], expected, atol=1e-14)
            bad = copy.deepcopy(saved)
            bad['banks'][0]['demand_means'][0,0,0] += 1.
            with self.assertRaises(ValueError): night.RectangularReduction(group).restore(bad)
            rows[0]['marginal_posterior_demand_mean'][0] += .01
            with self.assertRaises(ValueError): night.validate_rows(rows, group)

    def test_real_spawn_and_resumed_serial_preserve_conditional_demand(self):
        with small_night():
            unit = next(u for u in night.units() if u[2]=='sampling_4h')
            group = night.make_group(unit)
            serial = night.RectangularReduction(group); night.advance(serial,15)
            split = night.RectangularReduction(group); night.advance(split,7)
            split = night.RectangularReduction(group).restore(split.checkpoint())
            with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
                night.advance(split,15,pool,workers=2)
            self.assertTrue(night.validate_rows(split.finish(),group))
            for a,b in zip(serial.finish(),split.finish()):
                for key in ('conditional_log_evidence','marginal_predicted_mass',
                            'conditional_predicted_demand_mean','conditional_posterior_demand_mean',
                            'marginal_predicted_demand_mean','marginal_posterior_demand_mean'):
                    np.testing.assert_allclose(a[key],b[key],atol=1e-13,rtol=0)


class NightExecutionTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with small_night(), patch.object(estimate,'_source_snapshot',return_value=self.source), \
             patch.object(fv,'_candidate_replay',side_effect=demand_replay) as replay, patch('builtins.print'):
            truth = {pair:grid._g2a_truth_record(pair,sid,1)
                     for pair,sid in zip(core.F4_G_PAIRS, night.PILOT_IDS)}
            yield truth, replay

    source = dict(source_commit='a'*40,working_tree_dirty=False)

    def arguments(self, directory):
        return estimate.build_parser().parse_args(['joint-bank-rectangular-robustness',
            '--g2a-report',str(Path(directory)/'old.json.gz'),
            '--g2a-run-dir',str(Path(directory)/'g2a'), '--run-dir',str(Path(directory)/'g2b'),
            '--workers','1','--chunk-size','4','--max-hours','8'])

    def run_blocks(self,args,truth,**kwargs):
        return estimate._run_rectangular_blocks(args,self.source,{'g2a_v2':'fixed'},truth,study=night,**kwargs)

    def test_budget_pauses_after_saved_block_and_resume_skips_saved_replays(self):
        with tempfile.TemporaryDirectory() as directory,self.fixture() as (truth,replay):
            args=self.arguments(directory);args.max_hours=1/3600
            clock=[0.]
            advance=night.advance
            def timed(*a,**k):
                advance(*a,**k);clock[0]=2.
            with patch.object(estimate,'perf_counter',side_effect=lambda:clock[0]), \
                 patch.object(night,'advance',side_effect=timed):
                first=self.run_blocks(args,truth,invocation_started=0.)
            self.assertEqual(first['stop_reason'],'time_budget')
            self.assertEqual(first['conditional_runs'],4)
            self.assertFalse(first['completed']);self.assertFalse(first['passes'])
            args.max_hours=8;args.limit_chunks=1
            second=self.run_blocks(args,truth)
            self.assertEqual(second['conditional_runs'],8);self.assertEqual(replay.call_count,8)
            args.collect_only=True
            collected=self.run_blocks(args,truth)
            self.assertEqual(collected['partial_unit'],second['partial_unit'])
            self.assertEqual(replay.call_count,8)

    def test_expired_setup_budget_does_not_start_new_work(self):
        with tempfile.TemporaryDirectory() as directory,self.fixture() as (truth,replay):
            args=self.arguments(directory);args.max_hours=1/3600
            with patch.object(estimate,'perf_counter',return_value=2.):
                result=self.run_blocks(args,truth,invocation_started=0.)
            self.assertEqual(result['stop_reason'],'time_budget')
            self.assertEqual(result['conditional_runs'],0);replay.assert_not_called()

    def test_complete_run_retains_all_scenarios_controls_and_common_time_scores(self):
        with tempfile.TemporaryDirectory() as directory,self.fixture() as (truth,replay):
            args=self.arguments(directory);args.chunk_size=15
            result=self.run_blocks(args,truth)
            self.assertTrue(result['completed']);self.assertTrue(result['passes'])
            self.assertEqual(result['completed_units'],40)
            self.assertEqual(result['completed_record_scenarios'],10)
            self.assertEqual(len(result['comparisons']),70)
            self.assertEqual(result['conditional_runs'],600);self.assertEqual(replay.call_count,600)
            self.assertEqual(len(result['paired_scores']),8)
            self.assertFalse(result['releases_existing_g2b_g2c'])
            self.assertFalse((args.run_dir/'checkpoint.npz').exists())
            args.collect_only=True;self.run_blocks(args,truth)
            self.assertEqual(replay.call_count,600)

    def test_failed_group_stops_future_work_and_cannot_be_resumed_as_passed(self):
        def negative(task):
            i,j,result=demand_replay(task)
            if task[2]['label']=='time_fine' and j==0:result['scores']=np.ones(len(result['scores']))
            return i,j,result
        with tempfile.TemporaryDirectory() as directory,self.fixture() as (truth,replay):
            args=self.arguments(directory);args.chunk_size=15;replay.side_effect=negative
            with self.assertRaises(RuntimeError):self.run_blocks(args,truth)
            self.assertEqual(len(list((args.run_dir/'units').glob('*.npz'))),4)
            saved,_=estimate._read_likelihood_report(args.run_dir/'summary.json')
            self.assertEqual(saved['stop_reason'],'numerical_failure')
            self.assertTrue(saved['failed_comparisons'])
            self.assertEqual(replay.call_count,60)
            with self.assertRaises(RuntimeError):self.run_blocks(args,truth)
            self.assertEqual(replay.call_count,60)

    def test_full_checkpoint_survives_final_publish_failure_without_rerun(self):
        with tempfile.TemporaryDirectory() as directory,self.fixture() as (truth,replay):
            args=self.arguments(directory);args.chunk_size=15;args.limit_chunks=1
            atomic=estimate._atomic_report
            def failing(path,payload,**kwargs):
                if path.parent.name=='units':raise OSError('deliberate final publish failure')
                return atomic(path,payload,**kwargs)
            with patch.object(estimate,'_atomic_report',side_effect=failing):
                with self.assertRaises(OSError):self.run_blocks(args,truth)
            self.assertEqual(replay.call_count,15)
            result=self.run_blocks(args,truth)
            self.assertEqual(result['completed_units'],2);self.assertEqual(replay.call_count,30)

    def test_locked_completed_checkpoint_is_replaced_and_collected_without_reruns(self):
        from tests.test_checkpoint_io import sharing_error
        with tempfile.TemporaryDirectory() as directory,self.fixture() as (truth,replay):
            args=self.arguments(directory);args.chunk_size=15
            unlink=Path.unlink
            def locked(path,*a,**k):
                if path.name=='checkpoint.npz': raise sharing_error()
                return unlink(path,*a,**k)
            with patch.object(Path,'unlink',new=locked), patch('src.tools.checkpoint_io.sleep'):
                result=self.run_blocks(args,truth)
                self.assertTrue(result['completed']);self.assertTrue(result['passes'])
                self.assertEqual(result['conditional_runs'],600);self.assertEqual(replay.call_count,600)
                self.assertTrue((args.run_dir/'checkpoint.npz').exists())
                args.collect_only=True
                collected=self.run_blocks(args,truth)
                self.assertIsNone(collected['partial_unit'])
                self.assertEqual(collected['conditional_runs'],600)
                args.collect_only=False
                self.run_blocks(args,truth)
                self.assertEqual(replay.call_count,600)

    def test_interrupted_block_keeps_its_predecessor_for_replay(self):
        with tempfile.TemporaryDirectory() as directory,self.fixture() as (truth,replay):
            args=self.arguments(directory);args.limit_chunks=1
            self.run_blocks(args,truth)
            path=args.run_dir/'checkpoint.npz';before=path.read_bytes()
            advance=night.advance
            def interrupted(reducer,stop,*a,**k):
                advance(reducer,reducer.seen+2,*a,**k)
                raise KeyboardInterrupt()
            with patch.object(night,'advance',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):self.run_blocks(args,truth)
            self.assertEqual(path.read_bytes(),before)
            result=self.run_blocks(args,truth)
            self.assertEqual(result['conditional_runs'],8)
            self.assertEqual(replay.call_count,10)  # Only the two unsaved nodes were repeated.

    def test_invalid_budget_and_dirty_source_stop_before_dependency_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.arguments(directory)
            with patch.object(estimate,'_read_bank_dependency') as read:
                for hours in (0.,-1.,float('nan'),float('inf')):
                    args.max_hours=hours
                    with self.assertRaises(ValueError):estimate.run_joint_night_pilot(args)
                args.max_hours=8
                with patch.object(estimate,'_source_snapshot',return_value=dict(self.source,working_tree_dirty=True)):
                    with self.assertRaises(ValueError):estimate.run_joint_night_pilot(args)
                read.assert_not_called()

    def test_dependency_requires_complete_raw_g2a_and_performs_no_old_fv_replays(self):
        previous=dict(source_commit='b'*40,dependencies={'old':'chain'})
        with tempfile.TemporaryDirectory() as directory,self.fixture() as (truth,replay), \
             patch.object(estimate,'_read_bank_dependency',return_value=(previous,'c'*64)):
            args=self.arguments(directory)
            old=estimate.build_parser().parse_args(['joint-bank-rectangular-pilot',
                '--g2a-report',str(args.g2a_report),'--run-dir',str(args.g2a_run_dir),
                '--workers','1','--chunk-size','4','--limit-chunks','1'])
            estimate.run_joint_rectangular_pilot(old)
            with self.assertRaisesRegex(ValueError,'all eight'):estimate.run_joint_night_pilot(args)
            self.assertEqual(replay.call_count,4)
            old.limit_chunks=None;old.chunk_size=15
            estimate.run_joint_rectangular_pilot(old)
            self.assertEqual(replay.call_count,120)
            args.limit_chunks=1
            result=estimate.run_joint_night_pilot(args)
            self.assertEqual(result['conditional_runs'],4);self.assertEqual(replay.call_count,124)
            path=next((args.g2a_run_dir/'units').glob('*.npz'))
            bad,_=estimate._read_likelihood_report(path);bad['rows'][0]['bank_log_evidence'][0]+=1.
            estimate._atomic_report(path,bad,replace_existing=True)
            with self.assertRaises(ValueError):estimate.run_joint_night_pilot(args)
            self.assertEqual(replay.call_count,124)


if __name__=='__main__':unittest.main()
