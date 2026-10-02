"""Grid convergence compares a fixed prior measure, with independent FV controls."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.method.bm_1_estimation import run_joint_parameter_bank
from src.validation.bm_1 import estimation as core
from src.validation.bm_1 import joint_estimation as fv
from src.validation.bm_1 import joint_grid as grid


def constant_replay(task):
    i,j,g = task
    cells = g['settings']['x_steps'];case = g['case']
    mass = core.beta_initial_mass(case,case.build_model()[0],cells)
    return i,j,dict(scores=np.zeros(24),predicted=np.broadcast_to(mass,(24,cells)),
                   posterior=np.broadcast_to(mass,(24,cells)),maximum_abs_mass_error=abs(mass.sum()-1),
                   minimum_cell_mass=float(mass.min()),actual_max_step=g['settings']['max_step'],elapsed_seconds=0.)


class JointGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(fv,'_candidate_replay',side_effect=constant_replay):
            cls.g1a = fv.run_joint_bank_refinement(workers=1)
        cls.g1a.update(source_commit='b'*40,working_tree_dirty=False,
                       dependencies=dict(g0=dict(sha256='a'*64,source_commit='c'*40)))

    @staticmethod
    def numerics():
        return {key:dict(time_steps=(1/12,1/24,1/48),spatial_cells=(7,11,15),quadrature_orders=(1,3,5))
                for key in fv.G1A_NUMERICS}

    @staticmethod
    def row(n, *, kind='quadrature', scores=None):
        case = core.build_f4_b_case(Xi=.75);obs = core.generate_synthetic_sequence(case,sequence_id=0).observations
        h,unit,edges,prior = grid._parameter_grid(('rho','Xi'),n,n)
        g = dict(pair=('rho','Xi'),sequence_id=0,case=case,observations=obs,hypotheses=h,
                 initial_log_weights=prior if kind=='quadrature' else np.full(n*n,-np.log(n*n)),
                 settings=dict(x_steps=11,max_step=1/24,likelihood_quadrature_order=3),label='fixture')
        conditionals = [constant_replay((0,j,g))[2] for j in range(n*n)]
        if scores is not None:
            for j,c in enumerate(conditionals): c['scores']=np.array(scores)[:,j]
        r = fv._reduce(g,conditionals)
        r.update(parameter_nodes_per_axis=n,parameter_unit_nodes=unit.tolist(),
                 parameter_dual_edges=edges.tolist(),prior_kind=kind)
        return r

    def test_fixed_prior_and_common_dual_cell_comparison_preserve_constant_likelihood(self):
        for pair in core.F4_G_PAIRS:
            h,u,e,log = grid._parameter_grid(pair,5,9)
            old = np.array([(a,b) for a in core.F4_E_PARAMETER_GRIDS[pair[0]][::10]
                            for b in core.F4_E_PARAMETER_GRIDS[pair[1]][::10]])
            np.testing.assert_array_equal(h,old)
            np.testing.assert_allclose(np.exp(log).reshape(5,5).sum(axis=0),np.diff(e),atol=1e-15)
            self.assertAlmostEqual(np.exp(log).sum(),1.)
            self.assertAlmostEqual(np.exp(log)[12]/np.exp(log)[0],4.)
        r = grid.compare_parameter_banks(self.row(5),self.row(9))
        self.assertTrue(r['passes'])
        self.assertLess(r['maximum_parameter_tv'],1e-14)
        self.assertLess(r['maximum_unit_mean_error'],1e-14)
        self.assertLess(r['maximum_cumulative_evidence_error'],1e-14)

    def test_prior_bridge_is_explicit_and_does_not_count_as_grid_convergence(self):
        a,b = self.row(5,kind='discrete_uniform'),self.row(5)
        with self.assertRaises(ValueError): grid.compare_parameter_banks(a,b)
        r = grid.compare_parameter_banks(a,b,allow_prior_change=True)
        self.assertGreater(r['maximum_parameter_tv'],.02)
        self.assertFalse(r['passes'])
        bad = copy.deepcopy(b);bad['hypotheses'][0][0]+=.1
        with self.assertRaises(ValueError): grid.compare_parameter_banks(a,bad,allow_prior_change=True)

    def test_joint_tv_detects_dependence_even_when_both_marginals_agree(self):
        a = self.row(3)
        prior = np.exp(a['initial_log_weights']).reshape(3,3)
        moved = prior.copy();moved[:2,:2] += np.array([[.03,-.03],[-.03,.03]])
        scores = np.zeros((24,9));scores[0] = np.log(moved.ravel()/prior.ravel())
        result = grid.compare_parameter_banks(a,self.row(3,scores=scores))
        self.assertAlmostEqual(result['maximum_parameter_tv'],.06)
        self.assertLess(result['maximum_unit_mean_error'],1e-14)
        self.assertLess(result['maximum_unit_sd_error'],1e-14)
        self.assertFalse(result['passes'])

    def test_absolute_evidence_intermediate_prefix_and_state_are_active_gates(self):
        a = self.row(5);score = np.zeros((24,81));score[0]=.11;score[-1]=-.11
        r = grid.compare_parameter_banks(a,self.row(9,scores=score))
        self.assertAlmostEqual(r['prefix_errors']['cumulative_evidence_error'][-1],0.)
        self.assertGreater(r['maximum_cumulative_evidence_error'],.1)
        self.assertFalse(r['passes'])
        b = self.row(9);b['marginal_predicted_mass'][0]=[1.]+[0.]*10
        self.assertFalse(grid.compare_parameter_banks(a,b)['passes'])
        b = self.row(9);b['log_weights'][0][0]+=.1
        with self.assertRaises(ValueError): grid.compare_parameter_banks(a,b)

    def test_complete_g1a_dependency_rejects_raw_forgery_and_missing_controls(self):
        self.assertTrue(grid.validated_g1a(self.g1a,require_source=True))
        for mutation in ('dirty','prior','observations','moment','missing','duplicate','prefix','selection','steps'):
            bad = dict(self.g1a,rows=list(self.g1a['rows']),comparisons=list(self.g1a['comparisons']))
            bad['rows'][0]=copy.deepcopy(bad['rows'][0])
            bad['comparisons'][0]=copy.deepcopy(bad['comparisons'][0])
            r = bad['rows'][0]
            if mutation=='dirty': bad['working_tree_dirty']=True
            elif mutation=='prior': r['initial_log_weights'][0]+=.1
            elif mutation=='observations': r['observations'][0]+=.1
            elif mutation=='moment': r['parameter_mean'][0][0]+=.1
            elif mutation=='missing': bad['comparisons'].pop()
            elif mutation=='duplicate': bad['rows'][-1]=bad['rows'][0]
            elif mutation=='prefix': bad['comparisons'][0]['prefix_errors']['weight_tv'][0]=.5
            elif mutation=='selection': bad['comparisons'][0]['selected']=True
            else: r['conditional_actual_max_steps'][0]=1.
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                grid.validated_g1a(bad,require_source=True)

    def test_replay_reuse_complete_gates_and_actual_time_refinement(self):
        events = []
        with patch.object(fv,'_candidate_replay',side_effect=constant_replay) as replay:
            result = grid.run_joint_parameter_grid(workers=1,progress=events.append)
        self.assertTrue(result['registered_design'])
        self.assertTrue(result['passes'])
        self.assertEqual(replay.call_count,648)
        self.assertEqual(result['conditional_runs'],648)
        self.assertEqual(len(result['rows']),26)
        self.assertEqual(sum(c['selected'] for c in result['comparisons']),14)
        self.assertEqual([r['completed'] for r in events],list(range(1,649)))
        self.assertEqual(sum(c['kind']=='prior_bridge' for c in result['comparisons']),2)
        self.assertTrue(all(not c['selected'] for c in result['comparisons'] if c['kind']=='prior_bridge'))
        def stalled(task):
            i,j,c = constant_replay(task)
            if task[2]['label']=='time_fine' and j==0: c['actual_max_step']*=2
            return i,j,c
        with patch.object(fv,'_candidate_replay',side_effect=stalled):
            failed = grid.run_joint_parameter_grid(workers=1)
        self.assertFalse(failed['passes'])
        self.assertTrue(all(not c['actual_step_refined'] for c in failed['comparisons'] if c['kind']=='FV_time'))

    def test_real_serial_spawn_and_recursive_bank_preserve_both_priors_and_own_states(self):
        options = dict(sequence_ids=(0,1),counts=(2,3,5),numerics=self.numerics())
        serial = grid.run_joint_parameter_grid(workers=1,**options)
        parallel = grid.run_joint_parameter_grid(workers=2,**options)
        self.assertFalse(serial['registered_design'])
        self.assertEqual(serial['conditional_runs'],200)
        for a,b in zip(serial['rows'],parallel['rows']):
            a,b = dict(a),dict(b);a.pop('conditional_seconds');b.pop('conditional_seconds')
            self.assertEqual(a,b)
        self.assertEqual(serial['comparisons'],parallel['comparisons'])
        for r in [r for r in serial['rows'] if r['parameter_nodes_per_axis']==3 and r['fv_label']=='middle']:
            case = core.build_f4_b_case(Xi=.75 if r['coordinates'][0]=='rho' else 1.)
            model,base = case.build_model();obs = core.generate_synthetic_sequence(case,sequence_id=r['sequence_id']).observations
            bank = run_joint_parameter_bank(model,base,case.sigma_x,case.sigma_y,r['initial_mass'],0.,obs,
                       coordinates=r['coordinates'],hypotheses=r['hypotheses'],initial_log_weights=r['initial_log_weights'],
                       breakpoints=case.breakpoints,max_step=r['settings']['max_step'],likelihood_quadrature_order=3)
            np.testing.assert_allclose(r['conditional_log_evidence'],bank['conditional_log_evidence'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(r['log_weights'],bank['log_weights'],atol=1e-10,rtol=0)
            for phase in ('predicted','posterior'):
                np.testing.assert_allclose(r['marginal_'+phase+'_mass'],np.array([s[phase+'_mass'] for s in bank['steps']]),atol=1e-12,rtol=0)
        json.dumps(parallel,allow_nan=False)

    def test_invalid_design_and_replay_failure_do_not_drop_work(self):
        for options in (dict(workers=True),dict(sequence_ids=(0,0)),dict(counts=(3,4,9)),dict(counts=(3,3,9))):
            with self.subTest(options=options),patch.object(fv,'_candidate_replay') as replay,self.assertRaises((ValueError,TypeError)):
                grid.run_joint_parameter_grid(**options)
            replay.assert_not_called()
        with patch.object(fv,'_candidate_replay',side_effect=RuntimeError('replay failed')),self.assertRaisesRegex(RuntimeError,'replay failed'):
            grid.run_joint_parameter_grid(workers=1)

    def test_cli_dependency_gzip_failure_retention_and_no_overwrite(self):
        from src.exe.run_model.bm_1 import estimate
        result = dict(stage='G1b0',registered_design=True,comparisons=[],passes=False)
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'):
            dependency = Path(directory)/'g1a.json.gz';output = Path(directory)/'g1b.json.gz'
            with gzip.open(dependency,'wt') as stream: json.dump(self.g1a,stream)
            args = estimate.build_parser().parse_args(['joint-bank-grid','--g1a-report',str(dependency),'--output',str(output),'--blas-threads','1'])
            self.assertEqual(args.workers,8)
            with patch.object(grid,'run_joint_parameter_grid',return_value=result) as run,self.assertRaisesRegex(RuntimeError,'report retained'):
                estimate.run_joint_parameter_grid(args)
            self.assertEqual(run.call_args.kwargs['workers'],8)
            with gzip.open(output,'rt') as stream: saved=json.load(stream)
            self.assertFalse(saved['passes']);self.assertEqual(saved['blas_threads_requested'],1)
            self.assertEqual(saved['dependencies']['g1a']['sha256'],hashlib.sha256(dependency.read_bytes()).hexdigest())
            self.assertEqual(saved['dependencies']['g1a']['g0'],self.g1a['dependencies']['g0'])
            raw = output.read_bytes()
            with patch.object(grid,'run_joint_parameter_grid') as run,self.assertRaises(FileExistsError):
                estimate.run_joint_parameter_grid(args)
            run.assert_not_called();self.assertEqual(raw,output.read_bytes())
            output.unlink()
            bad = dict(self.g1a,working_tree_dirty=True)
            with gzip.open(dependency,'wt') as stream: json.dump(bad,stream)
            with patch.object(grid,'run_joint_parameter_grid') as run,self.assertRaises(ValueError):
                estimate.run_joint_parameter_grid(args)
            run.assert_not_called();self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
