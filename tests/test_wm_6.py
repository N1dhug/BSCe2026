"""WM6 regression tests: baseline forcing, convex joint fit, noise variance."""
import unittest
import numpy as np

from src.model.wm_6 import wm_6
from src.params.wm_6 import joint_qp_params, fit_variances


class test_wm_6(unittest.TestCase):
    def setUp(self):
        self.steps=145
        self.model=wm_6((0.,1.),self.steps,lambda t: .5)
        self.r=np.array([-.7,-.12,-1.1,.14])

    def test_baseline_drift_and_sensitivity(self):
        self.model.coefficients(self.r)
        base=np.linspace(.3,.7,self.steps)
        u=np.linspace(-.4,.4,self.steps)
        t=self.model.t
        init=np.array([.1,-.04])
        state=self.model._solve_deterministic(init,u,u,t,base,self.r)
        expected=self.r[0]*state["F"]+self.r[1]*u+self.r[2]*state["R"]+self.r[3]*(2*base-1)
        np.testing.assert_allclose(state["R_dot"],expected,atol=1e-12)
        delta=np.array([.002,-.001])
        other=self.model._solve_deterministic(init+delta,u,u,t,base,self.r)
        S=self.model.sensitivity(self.r)["input"]
        np.testing.assert_allclose(other["I"]-state["I"],S@delta,atol=1e-10)

    def test_joint_fit_and_variance(self):
        data=[]
        for j in range(6):
            time=self.model.t.copy()
            U=.35*np.sin(2*np.pi*(time+j/9))
            B=.48+.12*np.cos(2*np.pi*(time+j/7))
            initial=[.04*(j-2),.025*np.cos(j)]
            state=self.model._solve_deterministic(initial,U,U,time,B,self.r)
            data.append({"I":state["I"],"U":U,"PE":U,"Time":time,"O":B})
        result=joint_qp_params(self.model).fit_many(data)
        np.testing.assert_allclose(result["r"],self.r,atol=3e-3)
        self.assertEqual(len(result["deterministic_states"]),6)
        kernel=np.linspace(0.,2.,100)
        tau2,sigma2,identified=fit_variances(np.sqrt(.01+.04*kernel),kernel)
        self.assertTrue(identified)
        self.assertAlmostEqual(tau2,.01,places=6)
        self.assertAlmostEqual(sigma2,.04,places=6)

    def test_stochastic_mean_and_variance(self):
        t=self.model.t
        params={"r":self.r,"U":lambda z: .2,"Time":lambda z:z}
        run=self.model.stochastic_solver([0.,0.],params,sigma=.2,tau=.1,seed=7)
        self.assertEqual(len(run["sample_demand"]),self.steps)
        self.assertAlmostEqual(run["variance"][0],.01,places=12)
        self.assertTrue(np.all(np.diff(run["variance"])>=-1e-12))


if __name__=="__main__": unittest.main()
