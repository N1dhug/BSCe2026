"""Robustness design, continuous-law statistics and recoverable job boundaries."""
import copy
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.validation.bm_1 import estimation as core, joint_estimation as fv, joint_grid as grid
from src.validation.bm_1 import joint_robustness as robust
from tests.test_bm1_joint_continued import TINY_LEVELS, flat_replay


def replay_with_demand(task):
    i, j, result = flat_replay(task)
    result.update(predicted_demand_mean=np.full(len(result['scores']), .5),
                  posterior_demand_mean=np.full(len(result['scores']), .5))
    return i, j, result


@contextmanager
def tiny_design():
    with patch.object(robust,'RECORD_DAYS',1), patch.object(robust,'COUNTS',(3,5)), \
         patch.object(fv,'G1A_NUMERICS',TINY_LEVELS), patch.object(robust,'POPULATION_SIZE',2), \
         patch.object(robust,'ROBUSTNESS_SIZE',1), \
         patch.object(robust,'PILOT_IDS',(9902,9903)), \
         patch.object(robust,'POPULATION_STARTS',(9910,9920)):
        yield


class RobustnessMathTests(unittest.TestCase):
    def test_physical_log_uniform_moments_and_exact_hat_quantiles(self):
        nodes = np.linspace(0,1,9)
        result = robust._physical_hat_moments(nodes,np.ones(9),'sigma_x')
        a,b = np.asarray(core.F4_E_PARAMETER_GRIDS['sigma_x'])[[0,-1]]
        mean = (b-a)/np.log(b/a)
        variance = (b*b-a*a)/(2*np.log(b/a))-mean*mean
        self.assertAlmostEqual(result['mean'],mean,places=13)
        self.assertAlmostEqual(result['sd']**2,variance,places=13)
        np.testing.assert_allclose(result['interval_95'],a*(b/a)**np.array([.025,.975]),atol=1e-13)
        for p in (0.,.025,.5,.975,1.):
            self.assertAlmostEqual(robust._linear_quantile(nodes,2*nodes,p),np.sqrt(p),places=13)

    def test_registered_workload_and_unique_population_ids(self):
        pilot = robust.design_units('G2b'); population=robust.design_units('G2c')
        self.assertEqual(len(pilot),10); self.assertEqual(len(population),560)
        self.assertEqual(8*65**2*5,169000)
        self.assertEqual(len(population)*33**2,609840)
        self.assertEqual(len({robust.unit_name(u) for u in population}),560)
        for pair,start in zip(core.F4_G_PAIRS,robust.POPULATION_STARTS):
            ids={sid for p,sid,_ in population if p==pair}
            self.assertEqual(ids,set(range(start,start+200)))
            self.assertEqual(sum(p==pair and s!='matched' for p,_,s in population),80)

    def test_g2a_gate_requires_complete_passing_seven_days_even_after_raw_validation(self):
        with patch.object(grid,'validated_g2a',return_value=True):
            for override in ({'record_days':1},{'completed':False},{'passes':False}):
                report=dict(record_days=7,completed=True,passes=True);report.update(override)
                with self.assertRaises(ValueError):robust.require_g2a(report)
            robust.require_g2a(dict(record_days=7,completed=True,passes=True))

    def test_sampling_reuses_exact_observations_and_mismatch_changes_one_factor(self):
        case, seq, _ = grid._g2a_truth_record(('rho','Xi'),1358,1)
        for scenario in robust.SCENARIOS:
            changed, obs, indices=robust._scenario(case,seq.observations,scenario)
            np.testing.assert_array_equal(obs.values,seq.observations.values[indices])
            np.testing.assert_array_equal(obs.times,seq.observations.times[indices])
            self.assertEqual(changed.Xi,case.Xi)
            self.assertEqual(changed.baseline,.55 if scenario=='baseline_055' else .5)
            self.assertEqual((changed.initial_alpha,changed.initial_beta),
                             (16.,24.) if scenario=='initial_beta_16_24' else (20.,20.))
            if scenario=='sampling_4h':np.testing.assert_array_equal(indices,np.arange(3,24,4))

    def test_real_cached_population_unit_matches_single_node_demand_and_spawn(self):
        # Real SDE observations and FV filters, not the flat fixture used to
        # isolate file/validator failure paths below. The grids are diagnostic.
        with tiny_design():
            unit=(core.F4_G_PAIRS[1],robust.POPULATION_STARTS[1],'sampling_4h')
            serial=robust.run_unit('G2c',unit,workers=1)
            parallel=robust.run_unit('G2c',unit,workers=2)
            self.assertTrue(robust.validate_unit(serial,'G2c',unit))
            for field in ('conditional_log_evidence','log_weights','marginal_predicted_mass','marginal_posterior_mass',
                          'conditional_predicted_demand_mean','marginal_predicted_demand_mean','marginal_posterior_demand_mean'):
                np.testing.assert_allclose(serial['rows'][0][field],parallel['rows'][0][field],atol=1e-13,rtol=0)
            row=serial['rows'][0]
            scores=np.array(row['conditional_log_evidence']);prior=np.array(row['initial_log_weights'])
            reduced=fv.parameter_bank_weights(scores,prior,np.ones(6,dtype=bool))
            for phase,weights in [('predicted',np.exp(reduced['prior_log_weights'])),('posterior',np.exp(reduced['log_weights']))]:
                expected=np.sum(weights*np.asarray(row['conditional_'+phase+'_demand_mean']),axis=1)
                np.testing.assert_allclose(row['marginal_'+phase+'_demand_mean'],expected,atol=1e-13,rtol=0)
            self.assertGreater(np.max(abs(np.sum(np.exp(reduced['log_weights'])*np.asarray(row['conditional_predicted_demand_mean']),axis=1)
                                          -row['marginal_predicted_demand_mean'])),1e-6)


class RobustnessRawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Diagnostic seed namespace keeps CI away from registered study records.
        with tiny_design():
            cls.truth = {pair:grid._g2a_truth_record(pair,sid,1)
                         for pair,sid in zip(core.F4_G_PAIRS,robust.PILOT_IDS)}
        with tiny_design(), patch.object(fv,'_candidate_replay',side_effect=replay_with_demand), \
             patch.object(grid,'_g2a_truth_record',side_effect=lambda pair,sid,days:cls.truth[tuple(pair)]):
            cls.reports = {robust.unit_name(unit):robust.run_unit('G2b',unit,workers=1)
                           for unit in robust.design_units('G2b')}
        for r in cls.reports.values():r.update(source_commit='a'*40,working_tree_dirty=False,raw_validation={'passes':True})

    @contextmanager
    def fixture(self):
        with tiny_design(), patch.object(grid,'_g2a_truth_record',side_effect=lambda pair,sid,days:self.truth[tuple(pair)]):
            yield

    def test_all_scenarios_raw_gates_and_population_summary(self):
        with self.fixture():
            records=[]
            for unit in robust.design_units('G2b'):
                r=self.reports[robust.unit_name(unit)]
                self.assertTrue(robust.validate_unit(r,'G2b',unit,require_source=True))
                self.assertTrue(r['passes']);self.assertEqual(len(r['comparisons']),7)
                self.assertEqual(len(r['rows']),8)
                records.append(robust.compact_unit(r))
            summary=robust.summarize_units('G2b',records)
            self.assertTrue(summary['passes']);self.assertEqual(summary['conditional_runs'],1000)
            self.assertFalse(robust.summarize_units('G2b',records[:-1])['passes'])
            with self.assertRaises(ValueError):robust.summarize_units('G2b',records+[records[0]])

    def test_raw_changes_and_missing_controls_are_detected(self):
        with self.fixture():
            unit=robust.design_units('G2b')[0];original=self.reports[robust.unit_name(unit)]
            changes=[lambda r:r['rows'].pop(),lambda r:r['comparisons'].pop(),
                lambda r:r['truth_record']['fine']['observations'].__setitem__(0,.123),
                lambda r:r['rows'][0]['conditional_log_evidence'][0].__setitem__(0,1.),
                lambda r:r['rows'][0]['marginal_predicted_demand_mean'].__setitem__(0,.99),
                lambda r:r['checkpoints'][0]['parameters']['rho'].update(mean=99.),
                lambda r:r.update(working_tree_dirty=True),lambda r:r.update(conditional_runs=99)]
            for change in changes:
                bad=copy.deepcopy(original);change(bad)
                with self.assertRaises((ValueError,KeyError)):robust.validate_unit(bad,'G2b',unit,require_source=True)
            bad=copy.deepcopy(original);bad['rows'][0]['maximum_abs_mass_error']=2e-10;bad['passes']=False
            self.assertTrue(robust.validate_unit(bad,'G2b',unit))

    def test_paired_y_scores_use_common_times(self):
        with self.fixture():
            pair=core.F4_G_PAIRS[0];sid=robust.PILOT_IDS[0]
            a=robust.compact_unit(self.reports[robust.unit_name((pair,sid,'matched'))])
            b=robust.compact_unit(self.reports[robust.unit_name((pair,sid,'sampling_2h'))])
            a['bank_log_evidence']=np.arange(24).tolist()
            b['bank_log_evidence']=(np.arange(1,24,2)+.1).tolist()
            result=robust.summarize_units('G2b',[a,b])
            self.assertAlmostEqual(result['paired_scores'][0]['mean_score_difference'],.1,places=12)

    def test_checkpoint_resume_revalidates_saved_units_without_replays(self):
        from src.exe.run_model.bm_1 import estimate
        source=dict(source_commit='a'*40,working_tree_dirty=False)
        g2a=dict(source_commit='b'*40,dependencies={'retained':'chain'})
        with self.fixture(),tempfile.TemporaryDirectory() as directory,patch('builtins.print'):
            run_dir=Path(directory)/'run';input_path=Path(directory)/'g2a.json'
            input_path.write_text(json.dumps(g2a))
            args=estimate.build_parser().parse_args(['joint-bank-robustness-pilot','--g2a-report',str(input_path),
                '--run-dir',str(run_dir),'--limit-units','1'])
            def run(stage,unit,**kwargs):return copy.deepcopy(self.reports[robust.unit_name(unit)])
            with patch.object(robust,'require_g2a',return_value=None),patch.object(estimate,'_source_snapshot',return_value=source), \
                 patch.object(robust,'run_unit',side_effect=run) as execute:
                first=estimate.run_joint_robustness(args)
                self.assertEqual(first['completed_units'],1);self.assertFalse(first['passes'])
                second=estimate.run_joint_robustness(args)
                self.assertEqual(second['completed_units'],2);self.assertEqual(execute.call_count,2)
                self.assertNotEqual(execute.call_args_list[0].args[1],execute.call_args_list[1].args[1])
                args.collect_only=True
                collected=estimate.run_joint_robustness(args)
                self.assertEqual(collected,second);self.assertEqual(execute.call_count,2)
                path=next((run_dir/'units').glob('*.npz'))
                bad,_=estimate._read_likelihood_report(path)
                bad['rows'][0]['bank_log_evidence'][0]+=1
                estimate._atomic_report(path,bad,replace_existing=True)
                with self.assertRaises(ValueError):estimate.run_joint_robustness(args)
                self.assertEqual(execute.call_count,2)

    def test_atomic_output_and_exclusive_lock_preserve_complete_evidence(self):
        from src.exe.run_model.bm_1 import estimate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'test.json.gz'
            with estimate._RunLock(root):
                with self.assertRaises(RuntimeError):
                    with estimate._RunLock(root):pass
                with self.assertRaises(ValueError):estimate._atomic_report(path,{'bad':float('nan')})
                self.assertFalse(path.exists())
                estimate._atomic_report(path,{'good':1})
                before=path.read_bytes()
                with self.assertRaises(FileExistsError):estimate._atomic_report(path,{'good':2})
                self.assertEqual(path.read_bytes(),before)
            with estimate._RunLock(root):pass

    def test_binary_arrays_roundtrip_without_precision_loss_or_pickle(self):
        from src.exe.run_model.bm_1 import estimate
        rng=np.random.default_rng(381)
        array=np.r_[rng.normal(size=1000),np.nextafter(1.,2.),1e-300].reshape(2,501)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unit.npz'
            report={'array':array,'metadata':{'name':'unchanged','items':[1,2,3]}}
            estimate._atomic_report(path,report)
            loaded,_=estimate._read_likelihood_report(path)
            np.testing.assert_array_equal(loaded['array'].view(np.uint64),array.view(np.uint64))
            self.assertEqual(loaded['metadata'],report['metadata'])
            with self.assertRaises(TypeError):
                estimate._atomic_report(Path(directory)/'object.npz',{'bad':np.array([{}],dtype=object)})


if __name__=='__main__':unittest.main()
