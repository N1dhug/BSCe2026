"""Continuation, new-record design and raw failure paths for G2a."""
import copy
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.method.bm_1_estimation import run_fv_filter
from src.params.bm_1 import BM1ObservationSeries
from src.validation.bm_1 import estimation as core
from src.validation.bm_1 import joint_estimation as fv
from src.validation.bm_1 import joint_grid as grid


TINY_LEVELS = {key:dict(time_steps=(1/96,1/192,1/384),spatial_cells=(7,11,21),
                      quadrature_orders=(1,3,5)) for key in fv.G1A_NUMERICS}


def flat_replay(task):
    i,j,g = task
    k=len(g['observations'].times);c=g['settings']['x_steps']
    states=np.full((k,c),1/c)
    return i,j,dict(scores=np.full(k,.1),predicted=states,posterior=states,
        maximum_abs_mass_error=float(np.max(abs(states.sum(axis=1)-1))),minimum_cell_mass=1/c,
        actual_max_step=g['settings']['max_step']*.9,elapsed_seconds=.001)


class ContinuedRecordTests(unittest.TestCase):
    def test_repeated_inputs_and_coupled_prefixes_are_one_continuous_record(self):
        pair=('rho','Xi');sid=grid.G2A_IDS[0]
        seven=core.generate_coupled_refinement(grid.g2a_case(pair,7),sequence_id=sid)
        for d in (1,3):
            case=grid.g2a_case(pair,d)
            np.testing.assert_array_equal(case.price_values,np.tile([.65,.35,.65],d))
            shorter=core.generate_coupled_refinement(case,sequence_id=sid)
            for phase in ('coarse','fine'):
                a,b=getattr(shorter,phase),getattr(seven,phase)
                np.testing.assert_array_equal(a.measurement_noise,b.measurement_noise[:24*d])
                np.testing.assert_allclose(a.truth.state,b.truth.state[:24*d+1],atol=1e-13,rtol=0)
                np.testing.assert_allclose(a.observations.values,b.observations.values[:24*d],atol=1e-13,rtol=0)
                self.assertEqual(a.truth.initial_state,b.truth.initial_state)
        self.assertEqual(len(seven.fine.observations.times),168)
        self.assertTrue(seven.coarse.truth.valid and seven.fine.truth.valid)
        for days in (True,1.5,2,0):
            with self.assertRaises((ValueError,TypeError)):grid.g2a_case(pair,days)
        with self.assertRaises(ValueError):grid.g2a_case(('k','Xi'),7)

    def test_seven_day_conditional_equals_manual_continuation_not_a_daily_reset(self):
        case=grid.g2a_case(('rho','Xi'),7)
        seq=core.generate_coupled_refinement(case,sequence_id=9900).fine
        model,parameters=case.build_model()
        initial=core.beta_initial_mass(case,model,21)
        settings=dict(breakpoints=case.breakpoints,max_step=1/96,likelihood_quadrature_order=3)
        def observations(lo,hi):
            return BM1ObservationSeries(seq.observations.times[lo:hi],seq.observations.values[lo:hi],
                'day',seq.observations.observed[lo:hi],'point','right')
        def run(mass,start,obs):
            return run_fv_filter(model,parameters,case.sigma_x,case.sigma_y,mass,start,obs,**settings)
        full=run(initial,0.,seq.observations)
        first=run(initial,0.,observations(0,24))
        continued=run(first.posterior_mass[-1],1.,observations(24,168))
        np.testing.assert_allclose(full.log_evidence,np.r_[first.log_evidence,continued.log_evidence],atol=1e-10,rtol=0)
        for field in ('predicted_mass','posterior_mass'):
            np.testing.assert_allclose(getattr(full,field),np.vstack([getattr(first,field),getattr(continued,field)]),
                                       atol=1e-10,rtol=0)
        reset=run(initial,1.,observations(24,168))
        self.assertGreater(np.abs(reset.predicted_mass[0]-continued.predicted_mass[0]).sum(),1e-3)

    def test_truth_failure_stops_before_any_replay_and_preserves_both_records(self):
        original=grid._g2a_truth_record
        def failed(*args):
            case,seq,r=original(*args);r['passes']=False
            return case,seq,r
        with patch.object(grid,'_g2a_truth_record',side_effect=failed),patch.object(fv,'_candidate_replay') as replay:
            result=grid.run_joint_continued_pilot(record_days=1,workers=1)
        replay.assert_not_called()
        self.assertFalse(result['completed']);self.assertFalse(result['passes'])
        self.assertEqual(result['conditional_runs'],0);self.assertEqual(len(result['truth_records']),2)


class ContinuedRawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Raw-tamper cases exercise the validator against fixed expected seed
        # data. Generate those expensive independent paths once; the positive
        # test below still checks real regeneration, and continuation tests
        # separately compare the actual 1/3/7-day coupled prefixes.
        cls.truth = {(pair,sid,7):grid._g2a_truth_record(pair,sid,7)
                     for pair,sid in zip(core.F4_G_PAIRS,grid.G2A_IDS)}
        with patch.object(fv,'G1A_NUMERICS',TINY_LEVELS),patch.object(grid,'G2A_COUNTS',(3,5)),\
             patch.object(fv,'_candidate_replay',side_effect=flat_replay),\
             patch.object(grid,'_g2a_truth_record',side_effect=lambda *key:cls.truth[key]):
            cls.report=grid.run_joint_continued_pilot(record_days=7,workers=1,counts=(3,5))
        dependency={}
        for stage in ('g0','g1a','g1b0','g1b3','g1c0'):
            dependency={stage:dict(sha256='b'*64,source_commit='c'*40,**dependency)}
        cls.report.update(source_commit='a'*40,working_tree_dirty=False,dependencies=dependency)

    def validate(self,report, *, regenerate_truth=False):
        truth = (nullcontext() if regenerate_truth else
                 patch.object(grid,'_g2a_truth_record',side_effect=lambda *key:self.truth[key]))
        with patch.object(fv,'G1A_NUMERICS',TINY_LEVELS),patch.object(grid,'G2A_COUNTS',(3,5)),truth:
            return grid.validated_g2a(report,require_source=True)

    def test_all_candidate_and_reference_controls_and_prefix_checkpoints_are_present(self):
        r=self.report
        self.assertTrue(self.validate(r,regenerate_truth=True));self.assertTrue(r['passes'])
        self.assertEqual(r['bank_reductions'],16);self.assertEqual(r['conditional_runs'],200)
        controls=[c for c in r['comparisons'] if c['kind'].startswith('FV_')]
        self.assertEqual(len(controls),12)
        self.assertEqual({c['parameter_nodes_per_axis'] for c in controls},{3,5})
        self.assertEqual(len(r['comparisons']),14);self.assertEqual(len(r['original_dual_cell_comparisons']),2)
        self.assertTrue(all(not c['selected'] for c in r['original_dual_cell_comparisons']))
        self.assertFalse(r['certifies_original_dual_cell_law'])
        self.assertEqual(len(r['checkpoints']),6)
        for c in r['checkpoints']:
            self.assertEqual(c['observation_count'],24*c['days'])
            self.assertAlmostEqual(c['cumulative_log_evidence'],.1*24*c['days'],places=12)
            np.testing.assert_allclose(c['hat_unit_mean'],.5,atol=1e-13)
            np.testing.assert_allclose(c['hat_unit_sd'],np.sqrt(1/12),atol=1e-13)

    def test_raw_validation_rejects_tampering_not_just_a_false_pass_flag(self):
        changes=[lambda r:r.update(working_tree_dirty=True),
            lambda r:r.update(conditional_runs=199),lambda r:r.update(record_days=1),
            lambda r:r['rows'].pop(),lambda r:r['comparisons'].pop(),
            lambda r:r['comparisons'][0].update(selected=False),
            lambda r:r['rows'][0]['observations'].__setitem__(0,.99),
            lambda r:r['rows'][0]['conditional_log_evidence'][10].__setitem__(0,.2),
            lambda r:r['checkpoints'][0].update(observation_count=25),
            lambda r:r['truth_records'][0]['fine']['state'].__setitem__(40,.9),
            lambda r:r.update(certifies_original_dual_cell_law=True),
            lambda r:r.update(dependencies={}),lambda r:r.update(raw_validation={'passes':False})]
        for change in changes:
            r=copy.deepcopy(self.report);change(r)
            with self.assertRaises((ValueError,KeyError,TypeError)):self.validate(r)

    def test_consistent_negative_probability_report_remains_reviewable(self):
        r=copy.deepcopy(self.report)
        r['rows'][0]['maximum_abs_mass_error']=2e-10;r['passes']=False
        self.assertTrue(self.validate(r))
        r['passes']=True
        with self.assertRaises(ValueError):self.validate(r)

    def test_diagnostic_overrides_cannot_claim_registered_acceptance(self):
        with patch.object(fv,'_candidate_replay',side_effect=flat_replay):
            r=grid.run_joint_continued_pilot(record_days=1,workers=1,counts=(3,5),numerics=TINY_LEVELS)
        self.assertFalse(r['registered_design'])
        with self.assertRaises(ValueError):grid.validated_g2a(r)


class ContinuedCLITests(unittest.TestCase):
    def test_mismatched_dependency_stops_and_failed_completed_report_is_not_lost(self):
        from src.exe.run_model.bm_1 import estimate
        dense=dict(source_commit='a'*40);original=dict(sha256='b'*64);replay=dict(passes=True)
        expected=dict(sha256='c'*64,source_commit='a'*40,g1b0=original)
        hat=dict(source_commit='d'*40,dependencies=dict(g1b3=expected),replay_reproduction=replay)
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'):
            output=Path(directory)/'g2a.json.gz'
            args=estimate.build_parser().parse_args(['joint-bank-continued-pilot','--grid-report','dense.gz',
                '--hat-report','hat.gz','--g1b0-report','original.gz','--days','7','--output',str(output)])
            with patch.object(estimate,'_read_joint_grid_inputs',return_value=(dense,'c'*64,original,replay)),\
                 patch.object(estimate,'_read_likelihood_report',return_value=(dict(hat,dependencies={}), 'e'*64)),\
                 patch.object(grid,'run_joint_continued_pilot') as run:
                with self.assertRaisesRegex(ValueError,'exact G1b3'):estimate.run_joint_continued_pilot(args)
                run.assert_not_called();self.assertFalse(output.exists())
            result=dict(truth_records=[],comparisons=[],completed=True,truth_passes=True,passes=False)
            with patch.object(estimate,'_read_joint_grid_inputs',return_value=(dense,'c'*64,original,replay)),\
                 patch.object(estimate,'_read_likelihood_report',return_value=(hat,'e'*64)),\
                 patch.object(grid,'validated_g1c0',return_value=True),\
                 patch.object(grid,'validated_g2a',return_value=True),\
                 patch.object(grid,'run_joint_continued_pilot',return_value=result),\
                 self.assertRaisesRegex(RuntimeError,'retained'):
                estimate.run_joint_continued_pilot(args)
            with gzip.open(output,'rt') as f:saved=json.load(f)
            self.assertFalse(saved['passes']);self.assertEqual(saved['dependencies']['g1c0']['sha256'],'e'*64)
            with self.assertRaises(FileExistsError):estimate.run_joint_continued_pilot(args)
            output.unlink();result['passes']=True
            with patch.object(estimate,'_read_joint_grid_inputs',return_value=(dense,'c'*64,original,replay)),\
                 patch.object(estimate,'_read_likelihood_report',return_value=(hat,'e'*64)),\
                 patch.object(grid,'validated_g1c0',return_value=True),\
                 patch.object(grid,'validated_g2a',side_effect=ValueError('final raw mismatch')),\
                 patch.object(grid,'run_joint_continued_pilot',return_value=result),\
                 self.assertRaisesRegex(RuntimeError,'retained'):
                estimate.run_joint_continued_pilot(args)
            with gzip.open(output,'rt') as f:saved=json.load(f)
            self.assertFalse(saved['passes']);self.assertEqual(saved['raw_validation']['error'],'final raw mismatch')

    def test_hat_dependency_is_recomputed_and_rejects_altered_continuous_law(self):
        from tests.test_bm1_joint_hat import JointHatDependencyTests
        # This is a dependency/reconstruction test with mocked constant
        # conditionals; production-size state grids add no numerical coverage.
        with patch.object(fv,'G1A_NUMERICS',TINY_LEVELS):
            JointHatDependencyTests.setUpClass();dense=JointHatDependencyTests.report
        with patch.object(fv,'G1A_NUMERICS',TINY_LEVELS),patch.object(grid,'G1B3_COUNTS',JointHatDependencyTests.counts):
            hat=grid.run_joint_parameter_reconstruction(dense)
            hat.update(source_commit='a'*40,working_tree_dirty=False)
            self.assertTrue(grid.validated_g1c0(hat,dense,require_source=True))
            bad=copy.deepcopy(hat);bad['rows'][0]['nodal_density'][0][0][0]*=2
            with self.assertRaises(ValueError):grid.validated_g1c0(bad,dense,require_source=True)
            bad=copy.deepcopy(hat);bad['certifies_original_dual_cell_law']=True
            with self.assertRaises(ValueError):grid.validated_g1c0(bad,dense,require_source=True)


if __name__=='__main__':unittest.main()
