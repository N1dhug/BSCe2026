"""Short physical, probability and information-set checks for BM2."""
import unittest
import numpy as np
import pandas as pd
from src.model.bm_2 import bm_2
from src.params.bm_1_families import FamilyFit,curves
from src.exe.run_model.bm_2.predict import predict_origin,posterior_mass
from src.tools.baseline import statistical_output
from tests.test_bm1_thomas_pipeline import frame_fixture


def parameters():
    f,g=curves('linear',[])
    return {'f':f,'g':g,'scalar_f':f.scalar_evaluator(),'C':2.,'Xi':.5,'k':1.,'tau':1.7}


class BM2Tests(unittest.TestCase):
    def test_inertia_matches_analytic_constant_forcing(self):
        class Constant:
            max_abs_slope=0.
            def scalar_evaluator(self):return lambda x:0.
            def __call__(self,x):return np.zeros_like(x)
        p=parameters();p['f']=Constant();p['scalar_f']=p['f'].scalar_evaluator()
        x0,r0,h,q=.4,-.1,.4,0.
        x,r=bm_2.advance(x0,r0,h,.5,.5,p,4)
        expected_r=q+(r0-q)*np.exp(-h/p['tau'])
        expected_x=x0+p['Xi']/p['C']*(q*h+(r0-q)*p['tau']*(1-np.exp(-h/p['tau'])))
        self.assertAlmostEqual(r,expected_r,delta=1e-8);self.assertAlmostEqual(x,expected_x,delta=1e-8)
        x,r=bm_2.advance(.999,1.,2.,.5,0.,p,32)
        self.assertTrue(0<=x<=1 and -1<=r<=1)

    def test_small_inertia_approaches_BM1_interior(self):
        p=parameters();p['tau']=1e-4;x0=.4;b=.5;u=.3
        r0=bm_2.response_rate(x0,b,u,p)
        x,r=bm_2.advance(x0,r0,1.,b,u,p,64)
        reference=FamilyFit._advance(x0,1.,b,u,p,256)
        self.assertLess(abs(x-reference),2e-5)

    def test_joint_FV_preserves_probability_and_has_inward_R(self):
        p=parameters();x,r=bm_2.grid(31,21)
        Q=bm_2.generator(x,r,.4,.6,p,.3)
        np.testing.assert_allclose(np.asarray(Q.sum(axis=0)),0.,atol=1e-12)
        self.assertTrue(np.all((Q-Q.multiply(np.eye(Q.shape[0]))).data>=0))
        mass=np.zeros((31,21));mass[15,10]=1
        data={'Time':np.arange(4.),'B':np.full(4,.4),'U':np.full(4,.6)}
        result=bm_2.evolve(data,p,mass,.3,max_step=.1)
        np.testing.assert_allclose(result['cell_mass'].sum(axis=(1,2)),1.,atol=1e-12)
        self.assertGreaterEqual(result['cell_mass'].min(),0.)

    def test_prior_keeps_pairs_and_observation_updates_velocity(self):
        p=parameters();x,r=bm_2.grid(61,41)
        states=np.array([[.2,-.6],[.25,-.5],[.5,0.],[.75,.5],[.8,.6]])
        observed=bm_2.demand(.2,.5,.5,p)
        posterior=posterior_mass(states,x,r,p,.5,.5,observed,.005)
        self.assertLess(float(r@posterior.sum(axis=0)),-.3)
        self.assertAlmostEqual(posterior.sum(),1.)

    def test_MPC_does_not_read_future_demand(self):
        frame=frame_fixture();origin=pd.Timestamp('2022-02-20')
        baseline=statistical_output().fit(frame,origin)
        options=dict(family='linear',max_iterations=1,max_nfev=3,state_maxiter=6,x_steps=31,r_steps=15,horizon=2)
        rows,diag=predict_origin(frame,origin,baseline,**options)
        changed=frame.copy();changed.loc[origin+pd.Timedelta(hours=1):origin+pd.Timedelta(hours=2),'I']=.95
        other,other_diag=predict_origin(changed,origin,baseline,**options)
        np.testing.assert_array_equal([row['I_BM2'] for row in rows],[row['I_BM2'] for row in other])
        np.testing.assert_array_equal(diag['fit']['initial_states'],other_diag['fit']['initial_states'])
        self.assertEqual(diag['fit']['initial_states'].shape,(5,2))
        self.assertEqual([row['lead_hours'] for row in rows],[1,2])
        self.assertTrue(all(0<=row['I_BM2']<=1 for row in rows))


if __name__=='__main__':unittest.main()
