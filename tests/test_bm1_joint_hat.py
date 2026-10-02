"""Independent identities and failure paths for the separate G1c0 law."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from src.validation.bm_1 import joint_grid as grid
from src.validation.bm_1 import joint_estimation as fv
from tests.test_bm1_joint_grid import constant_replay


def law_row(n, density):
    unit=np.linspace(0,1,n);q=np.full(n,1/(n-1));q[[0,-1]]*=.5
    u,v=np.meshgrid(unit,unit,indexing='ij');values=density(u,v)
    prior=np.outer(q,q).ravel();weights=prior*values.ravel();weights/=weights.sum()
    return dict(parameter_nodes_per_axis=n,parameter_unit_nodes=unit.tolist(),prior_kind='quadrature',
        initial_log_weights=np.log(prior).tolist(),log_weights=np.log(weights)[None,:].tolist(),
        coordinates=['rho','Xi'],sequence_id=1356,observation_times=[1.],observations=[.4],
        x_edges=[0.,.5,1.],marginal_predicted_mass=[[.4,.6]],marginal_posterior_mass=[[.3,.7]],
        cumulative_log_evidence=[0.])


class JointHatMathTests(unittest.TestCase):
    def test_uniform_prior_exact_on_every_grid_including_endpoint_hats(self):
        for n in (2,5,9,17,33,65):
            law=grid._hat_law(law_row(n,lambda u,v:np.ones_like(u)))
            np.testing.assert_allclose(law['density'],1,atol=2e-14)
            np.testing.assert_allclose(law['mass'],1,atol=2e-14)
            np.testing.assert_allclose(law['mean'],.5,atol=2e-14)
            np.testing.assert_allclose(law['sd']**2,1/12,atol=2e-14)
            np.testing.assert_allclose(law['covariance'],0,atol=2e-14)
        c=grid.compare_hat_laws(law_row(5,lambda u,v:1+0*u),law_row(65,lambda u,v:1+0*u))
        self.assertTrue(c['passes']);self.assertLess(c['maximum_parameter_tv'],1e-14)

    def test_bilinear_density_exact_across_grids_and_joint_dependence_detected(self):
        density=lambda u,v:1+.8*(2*u-1)*(2*v-1)
        a,b=law_row(5,density),law_row(17,density)
        result=grid.compare_hat_laws(a,b)
        self.assertTrue(result['passes']);self.assertLess(result['maximum_parameter_tv'],1e-14)
        law=grid._hat_law(a)
        np.testing.assert_allclose(law['mean'],.5,atol=1e-14)
        np.testing.assert_allclose(law['sd']**2,1/12,atol=1e-14)
        np.testing.assert_allclose(law['covariance'],.8/36,atol=1e-14)
        # Same two marginals, different joint law: marginal-only TV is inadequate.
        comparison=grid.compare_hat_laws(law_row(9,lambda u,v:1+0*u),b)
        self.assertAlmostEqual(comparison['maximum_parameter_tv'],.8/8,places=13)
        self.assertFalse(comparison['passes'])

    def test_affine_density_has_analytic_moments_and_tv(self):
        law=grid._hat_law(law_row(9,lambda u,v:.2+1.6*u))
        expected_mean=.2/2+1.6/3
        np.testing.assert_allclose(law['mean'],[[expected_mean,.5]],atol=1e-14)
        self.assertAlmostEqual(law['sd'][0,0]**2,.2/3+1.6/4-expected_mean**2,places=13)
        c=grid.compare_hat_laws(law_row(5,lambda u,v:1+0*u),law_row(17,lambda u,v:.2+1.6*u))
        self.assertAlmostEqual(c['maximum_parameter_tv'],.2,places=13)

    def test_root_split_integral_against_independent_adaptive_quadrature(self):
        rng=np.random.default_rng(623)
        cases=list(rng.normal(size=(30,4)))+[[0,0,0,0],[1,-1,-1,1],[0,0,-2,3]]
        for a0,a1,b0,b1 in cases:
            def inner(x):
                a=a0+(a1-a0)*x;b=b0+(b1-b0)*x
                if a*b>=0:return (abs(a)+abs(b))/2
                root=-a/(b-a)
                return (abs(a)*root+abs(b)*(1-root))/2
            points=[-start/(end-start) for start,end in ((a0,a1),(b0,b1))
                    if start!=end and 0<-start/(end-start)<1]
            expected=quad(inner,0,1,points=points,epsabs=1e-12,epsrel=1e-12)[0]
            got=grid._bilinear_abs_integral(tuple(np.array([x],dtype=float) for x in (a0,a1,b0,b1)),32)[0]
            self.assertAlmostEqual(got,expected,places=10)

    def test_integrated_joint_state_equals_existing_bayes_mixture(self):
        row=law_row(9,lambda u,v:1+.8*(2*u-1)*(2*v-1));law=grid._hat_law(row)
        weights=np.exp(row['log_weights'])[0].reshape(9,9)
        rng=np.random.default_rng(12);states=rng.uniform(size=(9,9,7));states/=states.sum(axis=2,keepdims=True)
        q=np.r_[.5,np.ones(7),.5]/8
        integrated=np.einsum('ij,ijc,i,j->c',law['density'][0],states,q,q)
        expected=np.einsum('ij,ijc->c',weights,states)
        np.testing.assert_allclose(integrated,expected,atol=1e-14)
        self.assertAlmostEqual(integrated.sum(),1,places=14)

    def test_invalid_prior_and_mass_are_rejected(self):
        row=law_row(5,lambda u,v:1+0*u)
        for field,value in [('prior_kind','discrete_uniform'),('log_weights',[[0.]*25]),
                            ('parameter_unit_nodes',[0.,.2,.5,.75,1.])]:
            bad=copy.deepcopy(row);bad[field]=value
            with self.assertRaises(ValueError):grid._hat_law(bad)


class JointHatDependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts=(5,9,17)
        with patch.object(fv,'_candidate_replay',side_effect=constant_replay):
            cls.report=grid._run_joint_grid(workers=1,progress=None,sequence_ids=grid.G1B0_IDS,
                counts=cls.counts,numerics=None,stage='G1b3',design_version=4,
                registered_counts=cls.counts,prior_bridge=False,streaming=True)
        cls.report.update(source_commit='a'*40,working_tree_dirty=False,
            conditional_backend='lapack-tridiagonal; unchanged FV scheme',
            reduction='ordered streaming log-scaled sums; at most 2*workers in flight',
            dependencies=dict(g1b0=dict(sha256='b'*64,source_commit='c'*40,
                g1a=dict(sha256='d'*64,source_commit='e'*40,g0=dict(sha256='f'*64,source_commit='1'*40)))))
        cls.report['replay_reproduction']=grid.compare_grid_overlap(cls.report,cls.report)

    def test_raw_validation_accepts_complete_fixture_and_rejects_tampering(self):
        # Smaller declared counts only reduce test cost; production uses 5/9/17/33/65.
        with patch.object(grid,'G1B3_COUNTS',self.counts):
            self.assertTrue(grid.validated_g1b3(self.report,require_source=True))
            for field,value in [('passes',False),('working_tree_dirty',True),('bank_reductions',23)]:
                bad=dict(self.report,**{field:value})
                with self.assertRaises(ValueError):grid.validated_g1b3(bad,require_source=True)
            bad=dict(self.report,rows=self.report['rows'][:-1])
            with self.assertRaises(ValueError):grid.validated_g1b3(bad)
            bad=dict(self.report,replay_reproduction=dict(self.report['replay_reproduction'],passes=False))
            with self.assertRaises(ValueError):grid.validated_g1b3(bad)
            bad=copy.deepcopy(self.report);bad['rows'][0]['conditional_log_evidence'][0][0]+=.01
            with self.assertRaises(ValueError):grid.validated_g1b3(bad)

    def test_cli_preserves_failed_pilot_and_original_dependency_without_replay(self):
        from src.exe.run_model.bm_1 import estimate
        original=dict(source_commit='c'*40,dependencies=dict(g1a=self.report['dependencies']['g1b0']['g1a']))
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'):
            output=Path(directory)/'g1c.json.gz'
            args=estimate.build_parser().parse_args(['joint-bank-reconstruct','--grid-report','dense.gz',
                '--g1b0-report','original.gz','--output',str(output),'--blas-threads','1'])
            result=dict(stage='G1c0',passes=False,original_dual_cell_passes=False,comparisons=[])
            with patch.object(estimate,'_read_bank_dependency',return_value=(original,'b'*64)),\
                 patch.object(estimate,'_read_likelihood_report',return_value=(self.report,'2'*64)),\
                 patch.object(grid,'run_joint_parameter_reconstruction',return_value=result),\
                 patch.object(grid,'compare_grid_overlap',return_value=self.report['replay_reproduction']),\
                 patch.object(fv,'_candidate_replay') as replay,\
                 self.assertRaisesRegex(RuntimeError,'report retained'):
                estimate.run_joint_parameter_reconstruction(args)
            replay.assert_not_called()
            with gzip.open(output,'rt') as f:saved=json.load(f)
            self.assertFalse(saved['passes']);self.assertFalse(saved['original_dual_cell_passes'])
            self.assertEqual(saved['dependencies']['g1b3']['sha256'],'2'*64)
            with self.assertRaises(FileExistsError):estimate.run_joint_parameter_reconstruction(args)
            output.unlink()
            with patch.object(estimate,'_read_bank_dependency',return_value=(original,'3'*64)),\
                 patch.object(estimate,'_read_likelihood_report',return_value=(self.report,'2'*64)),\
                 self.assertRaisesRegex(ValueError,'exact G1b0'):
                estimate.run_joint_parameter_reconstruction(args)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
