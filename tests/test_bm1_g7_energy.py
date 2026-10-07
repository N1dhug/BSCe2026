"""Independent exact QP and Euler balance identities; no historical replay."""
import unittest
import numpy as np
from src.method.bm_1_g7_energy import box_quadratic, energy_gap
from src.method.bm_1_g7_matched import MatchedDeterministicControl

class EnergyControlTests(unittest.TestCase):
    def test_qp_bounds_equality_and_discrete_energy(self):
        q=box_quadratic(np.eye(2),[-2.,-.2],(np.ones(2),1.))
        np.testing.assert_allclose(q['prices'],[1.,0.],atol=1e-12)
        self.assertLess(q['kkt_residual'],1e-12)
        q=box_quadratic(np.eye(2),[-.8,-.6],(np.ones(2),1.))
        np.testing.assert_allclose(q['prices'],[.6,.4],atol=1e-12)
        packet=dict(schema='g7-matched-fixed-target-v1',baseline_time_hours=[0.,1.,2.],
                    baseline=[.5]*3,target=[.5]*3,price_edges_hours=[0.,1.,2.],
                    scales=dict(demand_min=0.,demand_scale=1.,price_min=0.,price_scale=1.),
                    models={'wm':dict(family='WM4',r=[-1.,-.5,-3.],state_origin=[.2,.1])})
        f=MatchedDeterministicControl(packet).evaluate('wm',[.7,.2],12)
        delta=f['state'][-1]-f['state'][0]
        # Euler's left-R state balance versus midpoint-R demand quadrature.
        self.assertAlmostEqual(energy_gap(f),24*delta[0]+delta[1]/24,places=13)
        self.assertGreater(abs(energy_gap(f)-24*delta[0]),1e-5)
