"""Paired G1b1 keeps the failed artifact and certifies every new grid node."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.validation.bm_1 import estimation as core
from src.validation.bm_1 import joint_estimation as fv
from src.validation.bm_1 import joint_grid as grid
from tests.test_bm1_joint_grid import constant_replay


def peaked_replay(task):
    """The same likelihood on shared physical nodes, independent of FV level."""
    i,j,result = constant_replay(task)
    g = task[2];value = g['hypotheses'][j,1]
    unit = np.log(value/.0075)/np.log(16.) if g['pair'][1]=='sigma_y' else value
    index = round(8*unit)
    if abs(8*unit-index)<1e-10 and index%2:
        result['scores'][0] = 2.
    return i,j,result


class JointGridRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(fv,'_candidate_replay',side_effect=peaked_replay):
            cls.previous = grid.run_joint_parameter_grid(workers=1)
        cls.previous.update(source_commit='b'*40,working_tree_dirty=False,
            dependencies=dict(g1a=dict(sha256='a'*64,source_commit='c'*40,
                                      g0=dict(sha256='d'*64,source_commit='e'*40))))
        cls.events = []
        with patch.object(fv,'_candidate_replay',side_effect=peaked_replay) as replay:
            cls.current = grid.run_joint_parameter_grid_refinement(workers=1,progress=cls.events.append)
            cls.calls = replay.call_count

    def test_new_registered_design_keeps_records_support_prior_and_all_fv_controls(self):
        result = self.current
        self.assertEqual((result['stage'],result['design_version']),('G1b1',2))
        self.assertTrue(result['registered_design'])
        self.assertEqual(result['sequence_ids'],self.previous['sequence_ids'])
        self.assertEqual(result['parameter_grid_counts'],[5,9,17])
        self.assertEqual((self.calls,result['conditional_runs']), (2312,2312))
        self.assertEqual((result['bank_reductions'],len(result['comparisons'])),(24,16))
        self.assertEqual(sum(c['selected'] for c in result['comparisons']),14)
        self.assertEqual([e['completed'] for e in self.events],list(range(1,2313)))
        self.assertTrue(all(r['prior_kind']=='quadrature' for r in result['rows']))
        self.assertEqual(result['parameter_criteria'],self.previous['parameter_criteria'])
        self.assertEqual(result['fv_controls'],self.previous['fv_controls'])
        self.assertEqual(result['prior_measure'],self.previous['prior_measure'])
        active = [c for c in result['comparisons'] if c['selected']]
        self.assertEqual(sum(c['kind'].startswith('FV_') for c in active),12)
        self.assertTrue(all(c['passes'] for c in active if c['kind'].startswith('FV_')))
        self.assertTrue(all(c['left'].startswith('9/') and c['right'].startswith('17/')
                            for c in active if c['kind']=='parameter_grid'))
        for pair in core.F4_G_PAIRS:
            h = grid._parameter_grid(pair,17,17)[0].reshape(17,17,2)
            np.testing.assert_array_equal(h[::4,::4].reshape(25,2),grid._parameter_grid(pair,5,9)[0])
            np.testing.assert_allclose(h[::2,::2].reshape(81,2),grid._parameter_grid(pair,9,9)[0],atol=1e-14,rtol=0)

    def test_complete_rejected_dependency_is_required_and_raw_forgery_is_rejected(self):
        self.assertFalse(self.previous['passes'])
        self.assertTrue(grid.validated_g1b0(self.previous,require_source=True))
        for mutation in ('dirty','accepted','counts','criteria','dependency','missing','duplicate',
                         'prior','physical','dual','observations','moment','state','prefix','selected','steps','nested'):
            bad = dict(self.previous,rows=list(self.previous['rows']),comparisons=list(self.previous['comparisons']))
            bad['rows'][0] = copy.deepcopy(bad['rows'][0]);r = bad['rows'][0]
            bad['comparisons'][0] = copy.deepcopy(bad['comparisons'][0]);c = bad['comparisons'][0]
            if mutation=='dirty': bad['working_tree_dirty']=True
            elif mutation=='accepted': bad['passes']=True
            elif mutation=='counts': bad['parameter_grid_counts']=[5,9,17]
            elif mutation=='criteria': bad['parameter_criteria']=dict(bad['parameter_criteria'],maximum_parameter_tv=1.)
            elif mutation=='dependency': bad['dependencies']={}
            elif mutation=='missing': bad['comparisons'].pop()
            elif mutation=='duplicate': bad['rows'][-1]=bad['rows'][0]
            elif mutation=='prior': r['initial_log_weights'][0]+=.1
            elif mutation=='physical': r['hypotheses'][0][0]+=.1
            elif mutation=='dual': r['parameter_dual_edges'][1]+=.1
            elif mutation=='observations': r['observations'][0]+=.1
            elif mutation=='moment': r['parameter_sd'][0][0]+=.1
            elif mutation=='state': r['marginal_posterior_mass'][0][0]=-.1
            elif mutation=='prefix': c['prefix_errors']['weight_tv'][0]+=.1
            elif mutation=='selected': c['selected']=False
            elif mutation=='steps': r['conditional_actual_max_steps'][0]=1.
            else:
                # A self-consistent standalone bank cannot break exact nested reuse.
                r['conditional_actual_max_steps'][0]/=2
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                grid.validated_g1b0(bad,require_source=True)

    def test_shared_replay_gate_checks_all_banks_scores_states_and_steps(self):
        result = grid.compare_grid_overlap(self.previous,self.current)
        self.assertTrue(result['passes']);self.assertEqual(result['bank_comparisons'],16)
        self.assertLess(max(result['maximum_errors'].values()),1e-10)
        for field in ('conditional_log_evidence','marginal_predicted_mass','conditional_actual_max_steps'):
            bad = dict(self.current,rows=list(self.current['rows']))
            index = next(i for i,r in enumerate(bad['rows']) if r['parameter_nodes_per_axis']==9)
            bad['rows'][index]=copy.deepcopy(bad['rows'][index]);r = bad['rows'][index]
            if field=='conditional_actual_max_steps': r[field][0]+=.1
            else: r[field][0][0]+=.1
            with self.subTest(field=field):
                self.assertFalse(grid.compare_grid_overlap(self.previous,bad)['passes'])
        missing = dict(self.current,rows=self.current['rows'][1:])
        with self.assertRaises(ValueError):grid.compare_grid_overlap(self.previous,missing)
        changed = dict(self.current,rows=list(self.current['rows']));changed['rows'][0]=copy.deepcopy(changed['rows'][0])
        changed['rows'][0]['observations'][0]+=.1
        with self.assertRaises(ValueError):grid.compare_grid_overlap(self.previous,changed)

    def test_api_overrides_remain_unregistered_and_failed_replay_is_not_dropped(self):
        with patch.object(fv,'_candidate_replay',side_effect=constant_replay):
            result = grid.run_joint_parameter_grid_refinement(workers=1,sequence_ids=(0,1),counts=(2,3,5),
                numerics={k:dict(time_steps=(1/12,1/24,1/48),spatial_cells=(7,11,15),quadrature_orders=(1,3,5))
                          for k in fv.G1A_NUMERICS})
        self.assertFalse(result['registered_design']);self.assertEqual(result['conditional_runs'],200)
        for options in (dict(workers=True),dict(counts=(3,4,17)),dict(sequence_ids=(0,0))):
            with self.subTest(options=options),patch.object(fv,'_candidate_replay') as replay,self.assertRaises((ValueError,TypeError)):
                grid.run_joint_parameter_grid_refinement(**options)
            replay.assert_not_called()
        with patch.object(fv,'_candidate_replay',side_effect=RuntimeError('fresh replay failed')),self.assertRaisesRegex(RuntimeError,'fresh replay failed'):
            grid.run_joint_parameter_grid_refinement(workers=1)

    def test_cli_preserves_failed_dependency_and_output_before_grid_or_replay_rejection(self):
        from src.exe.run_model.bm_1 import estimate
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'):
            dependency=Path(directory)/'g1b0.json.gz';output=Path(directory)/'g1b1.json.gz'
            with gzip.open(dependency,'wt') as stream:json.dump(self.previous,stream)
            args=estimate.build_parser().parse_args(['joint-bank-grid-refine','--g1b0-report',str(dependency),
                                                     '--output',str(output),'--blas-threads','1'])
            self.assertEqual(args.workers,8)
            original=dependency.read_bytes()
            for grid_pass,replay_pass in ((False,True),(True,False)):
                result=dict(stage='G1b1',registered_design=True,comparisons=[],passes=grid_pass)
                replay_result=dict(bank_comparisons=16,maximum_errors=dict(scores=0. if replay_pass else .1),passes=replay_pass)
                with patch.object(grid,'run_joint_parameter_grid_refinement',return_value=result) as run,\
                     patch.object(grid,'compare_grid_overlap',return_value=replay_result),\
                     self.assertRaisesRegex(RuntimeError,'report retained'):
                    estimate.run_joint_parameter_grid_refinement(args)
                self.assertEqual(run.call_args.kwargs['workers'],8)
                with gzip.open(output,'rt') as stream:saved=json.load(stream)
                self.assertFalse(saved['passes']);self.assertEqual(saved['blas_threads_requested'],1)
                self.assertEqual(saved['dependencies']['g1b0']['sha256'],hashlib.sha256(original).hexdigest())
                self.assertEqual(saved['dependencies']['g1b0']['g1a'],self.previous['dependencies']['g1a'])
                self.assertEqual(saved['replay_reproduction'],replay_result)
                with patch.object(grid,'run_joint_parameter_grid_refinement') as run,self.assertRaises(FileExistsError):
                    estimate.run_joint_parameter_grid_refinement(args)
                run.assert_not_called();output.unlink()
            self.assertEqual(original,dependency.read_bytes())
            bad=dict(self.previous,working_tree_dirty=True)
            with gzip.open(dependency,'wt') as stream:json.dump(bad,stream)
            with patch.object(grid,'run_joint_parameter_grid_refinement') as run,self.assertRaises(ValueError):
                estimate.run_joint_parameter_grid_refinement(args)
            run.assert_not_called();self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
