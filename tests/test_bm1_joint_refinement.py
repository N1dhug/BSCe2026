"""Joint-bank numerical gates, provenance checks and real spawn replays."""

import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from src.method.bm_1_estimation import run_joint_parameter_bank
from src.validation.bm_1 import estimation as core
from src.validation.bm_1 import joint_estimation as joint


class JointRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def replay(task):
            phase,pair,indices,h,case,obs,settings = task
            initial = core.beta_initial_mass(case,case.build_model()[0],settings['x_steps'])
            mass = np.broadcast_to(initial,(24,len(indices),len(initial)))
            return dict(phase=phase,pair=pair,indices=indices,scores=np.zeros((24,len(indices))),
                        predicted_mass=mass,posterior_mass=mass,maximum_abs_mass_error=abs(initial.sum()-1),
                        minimum_cell_mass=float(initial.min()),actual_max_step=settings['max_step'])
        # Assemble the complete real G0 report/gates using cheap conditional laws.
        with patch.object(core,'_f4_g_identity_task',side_effect=replay):
            cls.g0 = core.run_f4_g_joint_identity(workers=1)
        cls.g0.update(source_commit='c'*40,working_tree_dirty=False)

    @staticmethod
    def numerics():
        return {key:dict(time_steps=(1/12,1/24,1/48),spatial_cells=(11,21,41),quadrature_orders=(1,3,5))
                for key in joint.G1A_NUMERICS}

    @staticmethod
    def row(scores, *, cells=3, mass=None):
        scores = np.asarray(scores,dtype=float)
        count,h = scores.shape
        prior = np.full(h,-np.log(h))
        raw = prior+np.cumsum(scores,axis=0)
        evidence = logsumexp(raw,axis=1)
        return dict(coordinates=['rho','Xi'],sequence_id=0,hypotheses=[[.5,0.],[1.,.5],[1.5,1.]][:h],
                    truth=[1.,.75],fixed_parameters=dict(k=2.,sigma_x=.5,sigma_y=.03),
                    initial_log_weights=prior.tolist(),observation_times=(np.arange(1,count+1)/24).tolist(),
                    observations=[.5]*count,observed=[True]*count,conditional_log_evidence=scores.tolist(),
                    log_weights=(raw-evidence[:,None]).tolist(),cumulative_log_evidence=evidence.tolist(),
                    x_edges=np.linspace(0,1,cells+1).tolist(),
                    marginal_predicted_mass=(np.full((count,cells),1/cells) if mass is None else mass).tolist(),
                    marginal_posterior_mass=(np.full((count,cells),1/cells) if mass is None else mass).tolist())

    def test_g0_dependency_recomputes_raw_values_instead_of_trusting_flags(self):
        self.assertTrue(joint.validated_g0(self.g0,require_source=True))
        for mutation in ('dirty','missing','hypotheses','scores','weights','moments','negative_mass','beta','prefix','settings'):
            bad = copy.deepcopy(self.g0)
            r = bad['rows'][0]
            if mutation=='dirty': bad['working_tree_dirty']=True
            elif mutation=='missing': bad['rows'].pop()
            elif mutation=='hypotheses': r['hypotheses'][0][0]+=.001
            elif mutation=='scores': r['offline_conditional_log_evidence'][0][0]+=.2
            elif mutation=='weights': r['online_log_weights'][0][0]+=.2
            elif mutation=='moments': r['parameter_mean'][0][0]+=.2
            elif mutation=='negative_mass': r['marginal_posterior_mass'][0][0]=-.01
            elif mutation=='beta': r['initial_mass']=r['initial_mass'][:1]
            elif mutation=='prefix': r['prefix_errors']['conditional_state_mass'][0]=1.
            else: r['settings']['max_step']/=2
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                joint.validated_g0(bad,require_source=True)

    def test_comparison_checks_absolute_evidence_and_whole_bank_tv(self):
        a = self.row([[0.,0.,0.]])
        shifted = self.row([[.11,.11,.11]])
        result = joint.compare_banks(a,shifted)
        self.assertAlmostEqual(result['maximum_log_odds_error'],0.)
        self.assertAlmostEqual(result['maximum_weight_tv'],0.)
        self.assertFalse(result['passes'])
        moved = joint.compare_banks(a,self.row([[.08,0.,0.]]))
        self.assertLess(moved['maximum_log_odds_error'],.1)
        self.assertGreater(moved['maximum_weight_tv'],.01)
        self.assertFalse(moved['passes'])

    def test_likelihood_window_retains_full_weights_and_all_node_diagnostic(self):
        a,b = self.row([[-100.,0.,0.]]),self.row([[-90.,0.,0.]])
        result = joint.compare_banks(a,b)
        self.assertEqual(result['maximum_log_odds_error'],0.)
        self.assertEqual(result['maximum_all_node_log_odds_error'],10.)
        self.assertTrue(result['passes'])
        self.assertTrue(np.isfinite(b['log_weights']).all())
        self.assertEqual(len(b['log_weights'][0]),3)

    def test_every_prefix_is_gated_even_when_final_evidence_agrees(self):
        a = self.row([[0.,0.,0.],[0.,0.,0.]])
        b = self.row([[.2,.2,.2],[-.2,-.2,-.2]])
        result = joint.compare_banks(a,b)
        self.assertAlmostEqual(result['prefix_errors']['cumulative_evidence_error'][-1],0.)
        self.assertAlmostEqual(result['maximum_cumulative_evidence_error'],.2)
        self.assertFalse(result['passes'])

    def test_state_comparison_conserves_mass_and_rejects_incomparable_data(self):
        a = self.row([[0.,0.,0.]])
        fine = self.row([[0.,0.,0.]],cells=6)
        result = joint.compare_banks(a,fine)
        self.assertTrue(result['passes'])
        self.assertLess(result['maximum_posterior_l1'],1e-15)
        concentrated = self.row([[0.,0.,0.]],mass=np.array([[1.,0.,0.]]))
        self.assertFalse(joint.compare_banks(a,concentrated)['passes'])
        for mutation in ('observations','prior','shape','negative','nan'):
            bad = copy.deepcopy(fine)
            if mutation=='observations': bad['observations'][0]+=.1
            elif mutation=='prior': bad['initial_log_weights'][0]+=.1
            elif mutation=='shape': bad['conditional_log_evidence']=[[0.]]
            elif mutation=='negative': bad['marginal_predicted_mass'][0][0]=-.1
            else: bad['log_weights'][0][0]=float('nan')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                joint.compare_banks(a,bad)

    def test_real_serial_spawn_and_recursive_joint_bank_agree_for_both_pairs(self):
        options = dict(sequence_ids=((0,),(1,)),grid_stride=40,numerics=self.numerics())
        events = []
        serial = joint.run_joint_bank_refinement(**options,workers=1,progress=lambda r:events.append(r))
        parallel = joint.run_joint_bank_refinement(**options,workers=2)
        self.assertFalse(serial['registered_design'])
        self.assertEqual(serial['conditional_runs'],56)
        self.assertEqual([r['completed'] for r in events],list(range(1,57)))
        self.assertEqual(len(serial['rows']),14)
        self.assertEqual(sum(c['selected'] for c in serial['comparisons']),6)
        for left,right in zip(serial['rows'],parallel['rows']):
            left,right = dict(left),dict(right)
            left.pop('conditional_seconds');right.pop('conditional_seconds')
            self.assertEqual(left,right)
        self.assertEqual(serial['comparisons'],parallel['comparisons'])
        for r in [v for v in serial['rows'] if v['label']=='middle']:
            case = core.build_f4_b_case(Xi=.75 if r['coordinates'][0]=='rho' else 1.)
            model,base = case.build_model()
            obs = core.generate_synthetic_sequence(case,sequence_id=r['sequence_id']).observations
            bank = run_joint_parameter_bank(model,base,case.sigma_x,case.sigma_y,r['initial_mass'],0.,obs,
                       coordinates=r['coordinates'],hypotheses=r['hypotheses'],initial_log_weights=r['initial_log_weights'],
                       breakpoints=case.breakpoints,max_step=r['settings']['max_step'],likelihood_quadrature_order=3)
            np.testing.assert_allclose(r['conditional_log_evidence'],bank['conditional_log_evidence'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(r['log_weights'],bank['log_weights'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(r['marginal_predicted_mass'],np.array([s['predicted_mass'] for s in bank['steps']]),atol=1e-12,rtol=0)
            np.testing.assert_allclose(r['marginal_posterior_mass'],np.array([s['posterior_mass'] for s in bank['steps']]),atol=1e-12,rtol=0)
        serial['numerical_levels']['rho__Xi']['spatial_cells']=(1,2,3)
        self.assertEqual(joint.G1A_NUMERICS['rho__Xi']['spatial_cells'],(201,401,801))
        json.dumps(parallel,allow_nan=False)

    def test_invalid_design_and_failed_replay_never_drop_work(self):
        bad_levels = self.numerics();bad_levels['rho__Xi']['time_steps']=(1.,1.,.5)
        for options in (dict(workers=True),dict(sequence_ids=((0,),(0,))),dict(grid_stride=3),dict(numerics=bad_levels)):
            with self.subTest(options=options),patch.object(joint,'_candidate_replay') as replay,self.assertRaises((ValueError,TypeError)):
                joint.run_joint_bank_refinement(**options)
            replay.assert_not_called()
        with patch.object(joint,'_candidate_replay',side_effect=RuntimeError('candidate failed')):
            with self.assertRaisesRegex(RuntimeError,'candidate failed'):
                joint.run_joint_bank_refinement(workers=1,sequence_ids=((0,),(1,)),grid_stride=40,numerics=self.numerics())

    def test_time_acceptance_requires_actual_refinement_for_every_hypothesis(self):
        original = joint._candidate_replay
        def stalled(task):
            i,j,c = original(task)
            if task[2]['label']=='time_fine' and j==0:
                c['actual_max_step']=task[2]['settings']['max_step']*2
            return i,j,c
        with patch.object(joint,'_candidate_replay',side_effect=stalled):
            report = joint.run_joint_bank_refinement(workers=1,sequence_ids=((0,),(1,)),grid_stride=40,numerics=self.numerics())
        selected = [c for c in report['comparisons'] if c['selected'] and c['axis']=='time']
        self.assertTrue(all(not c['actual_step_refined'] and not c['passes'] for c in selected))
        self.assertFalse(report['passes'])

    def test_cli_validates_dependency_before_work_and_retains_failed_gzip(self):
        from src.exe.run_model.bm_1 import estimate
        result = dict(stage='G1a',registered_design=True,rows=[],comparisons=[],passes=False)
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'):
            g0_path = Path(directory)/'g0.json.gz'
            with gzip.open(g0_path,'wt') as stream: json.dump(self.g0,stream)
            path = Path(directory)/'g1.json.gz'
            args = estimate.build_parser().parse_args(['joint-bank-refine','--g0-report',str(g0_path),
                    '--output',str(path),'--blas-threads','1'])
            self.assertEqual(args.workers,8)
            with patch.object(joint,'run_joint_bank_refinement',return_value=result) as run:
                with self.assertRaisesRegex(RuntimeError,'report retained'): estimate.run_joint_bank_refinement(args)
            self.assertEqual(run.call_args.kwargs['workers'],8)
            with gzip.open(path,'rt') as stream: saved=json.load(stream)
            self.assertFalse(saved['passes']);self.assertEqual(saved['blas_threads_requested'],1)
            import hashlib
            self.assertEqual(saved['dependencies']['g0']['sha256'],hashlib.sha256(g0_path.read_bytes()).hexdigest())
            raw = path.read_bytes()
            with patch.object(joint,'run_joint_bank_refinement') as run,self.assertRaises(FileExistsError):
                estimate.run_joint_bank_refinement(args)
            run.assert_not_called();self.assertEqual(path.read_bytes(),raw)
            path.unlink()
            bad = copy.deepcopy(self.g0);bad['rows'][0]['observations'][0]+=.1
            with gzip.open(g0_path,'wt') as stream: json.dump(bad,stream)
            with patch.object(joint,'run_joint_bank_refinement') as run,self.assertRaises(ValueError):
                estimate.run_joint_bank_refinement(args)
            run.assert_not_called();self.assertFalse(path.exists())


if __name__=='__main__':
    unittest.main()
