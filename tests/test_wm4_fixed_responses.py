import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_fixed_responses import fixed_responses, ordered_response_pairs
from src.model.wm_5 import wm_5
from src.params.wm_5 import joint_qp_params
from src.solver.joint_drift_qp import joint_drift_qp
from src.solver.active_set import active_set
from src.validation.params.wm_5 import wm_5_params_validation
from src.exe.run_model.wm_4.sweep_fixed_responses import (
    simulate_batch, oracle_horizons, aggregate, rank,
)


class test_wm4_fixed_responses(unittest.TestCase):
    def test_five_normalized_odd_monotone_functions_and_twenty_pairs(self):
        functions = fixed_responses()
        self.assertEqual(len(functions), 5)
        pairs = ordered_response_pairs()
        self.assertEqual(len(pairs), 20)
        self.assertEqual(len(set(pairs)), 20)
        self.assertTrue(all(g != h for g, h in pairs))
        x = np.linspace(-1.0, 1.0, 1001)
        for name, f in functions.items():
            values = np.asarray(f.value(x), dtype=float)
            slopes = np.asarray(f.derivative(x), dtype=float)
            np.testing.assert_allclose(values[[0, 500, 1000]], [-1, 0, 1], atol=1e-11)
            np.testing.assert_allclose(values+f.value(-x), 0.0, atol=1e-10)
            self.assertTrue(np.all(np.diff(values) >= 0.0), name)
            self.assertTrue(np.all(slopes >= f.derivative_bounds[0]-1e-10), name)
            self.assertTrue(np.all(slopes <= f.derivative_bounds[1]+1e-10), name)
            self.assertTrue(np.all(values >= -1.0-1e-12), name)
            self.assertTrue(np.all(values <= 1.0+1e-12), name)
        cubics = [functions[name].derivative(0.0) for name in
                  ("polynomial_steep", "polynomial_very_steep",
                   "polynomial_extremely_steep")]
        self.assertEqual(sorted(cubics), cubics)
        self.assertLess(max(cubics), 1.5)

    @staticmethod
    def generated(g, h, n=97, sph=4):
        dt = 1.0/(24*sph)
        k = np.arange(n)
        U = np.sin(np.floor(k/sph)*0.4)*0.25
        O = np.full(n, .5)
        theta = np.array([-1.2, -1.8, -.3])
        F, R = np.zeros(n), np.zeros(n)
        F[0], R[0] = .05, .02
        for i in range(n-1):
            F[i+1] = F[i]+dt*R[i]
            R[i+1] = R[i]+dt*(theta[0]*F[i]+theta[1]*g.value(R[i])+
                                theta[2]*h.value(U[i]))
        return F, R, U, O, theta, dt

    def test_batched_simulator_is_same_as_wm5_model(self):
        catalog = fixed_responses()
        g, h = catalog["polynomial_extremely_steep"], catalog["logarithmic"]
        F, R, U, O, theta, dt = self.generated(g, h)
        model = wm_5((0.0,1.0),97,lambda t:.5,(g,),(h,))
        state = model._solve_deterministic([.05,.02],U,np.arange(97)/4,O,theta)
        prediction = simulate_batch([.05],.02,theta,g.value,h.value,U,O,4)
        np.testing.assert_allclose(state["I"][::4][:24], prediction[0], atol=1e-12)
        np.testing.assert_allclose(state["R"],R,atol=1e-12)

    def test_shared_qp_correctly_recovers_fixed_response_dynamics(self):
        functions = fixed_responses()
        g,h=functions["exponential"],functions["logarithmic"]
        theta=np.array([-1.2,-1.8,-.3])
        rates,prices,masks=[],[],[]
        for i in range(3):
            n,sph=97,4
            dt=1.0/(24*sph)
            U=0.25*np.sin((np.arange(n)/sph)*(.3+i*.15))
            F,R=np.zeros(n),np.zeros(n)
            F[0],R[0]=-.08+i*.06,.01+i*.01
            for k in range(n-1):
                F[k+1]=F[k]+dt*R[k]
                R[k+1]=R[k]+dt*(theta[0]*F[k]+theta[1]*g.value(R[k])+
                                    theta[2]*h.value(U[k]))
            rates.append(R)
            prices.append(U)
            masks.append(np.ones(n-1,dtype=bool))
        constraints=wm_5_params_validation.constraints((g,),(h,),dt)
        G=[np.asarray(g.value(R[:-1]),dtype=float)[:,None] for R in rates]
        H=[np.asarray(h.value(U[:-1]),dtype=float)[:,None] for U in prices]
        result=joint_drift_qp.solve_many(
            rates,prices,dt,constraints,transition_masks=masks,
            rate_features=G,price_features=H)
        self.assertEqual(result["design_columns"],6)
        np.testing.assert_allclose(result["coefficients"],theta,atol=1e-3)
        self.assertLess(result["qp_loss"],1e-8)
        self.assertTrue(np.isfinite(result["F0"]).all())
        # The executable must use the canonical WM5 params API rather than
        # construct a parallel joint QP directly inside the sweep.
        model=wm_5((0.0,1.0),n,lambda time:.5,(g,),(h,))
        datasets=[]
        for R,U in zip(rates,prices):
            obs=np.zeros(n,dtype=bool); obs[::sph]=True
            datasets.append({"I":R+.5,"U":U,"Time":np.arange(n,dtype=float)/sph,
                             "O":np.full(n,.5),"observation_mask":obs,
                             "transition_mask":np.ones(n-1,dtype=bool)})
        with patch("src.solver.joint_drift_qp.active_set.solve", wraps=active_set.solve) as solve:
            fitted=joint_qp_params(model).fit_many(datasets)
            self.assertEqual(solve.call_count,1)  # Existing active-set optimizer.
        self.assertEqual(fitted["fit_method"],"wm5_joint_qp_many")
        self.assertEqual(fitted["fit_trajectories"],3)
        self.assertEqual(fitted["init_cond"].shape,(3,2))
        self.assertEqual(len(fitted["deterministic_states"]),3)
        np.testing.assert_allclose(fitted["coefficients"],result["coefficients"],atol=1e-5)
        self.assertLess(fitted["deterministic_loss"],1e-8)

    def test_horizon_specific_oracle_never_worse_than_coarse_grid(self):
        functions=fixed_responses()
        g,h=functions["logarithmic"],functions["exponential"]
        F,R,U,O,theta,dt=self.generated(g,h)
        observed=simulate_batch([.27],.02,theta,g.value,h.value,U,O,4)[0]
        prepared={"future":{"U":U,"O":O},"R0":.02,"steps_per_hour":4,
                  "base_hourly":np.full(24,.5)}
        rows=oracle_horizons(observed,prepared,theta,g,h,6,24,grid_points=17)
        self.assertEqual(len(rows),19)
        for row in rows:
            self.assertGreaterEqual(row["oracle_F0"],-1.0)
            self.assertLessEqual(row["oracle_F0"],1.0)
            self.assertLessEqual(row["mse"]*row["horizon_hours"],
                                 row["coarse_grid_sse"]+1e-9)
            self.assertLess(row["rmse"],1e-4)

    def test_rank_uses_lowest_absolute_error_and_best_split_per_pair(self):
        pairs=ordered_response_pairs()
        days=pd.to_datetime(["2022-03-12","2022-03-13"])
        rows=[]
        for day in days:
            for i,(g,h) in enumerate(pairs):
                for n in (2,3):
                    for horizon in (6,7):
                        e=(.05 if i==1 else .1+.01*i)+.002*(n-2)+.003*(horizon-6)
                        rows.append({"day":day,"g_name":g,"h_name":h,
                                     "training_weekdays":n,"horizon_hours":horizon,
                                     "rmse":e,"mse":e*e,"baseline_mse":.25**2})
        df=pd.DataFrame(rows)
        all_splits, winners, common, basis=rank(df,pairs,(2,3),(6,7))
        self.assertEqual(len(all_splits),20*2*2)
        self.assertEqual(len(winners),20)
        self.assertEqual(len(common),2)
        self.assertEqual(basis,"strictly_matched")
        self.assertEqual(winners.iloc[0]["training_weekdays"],2)
        self.assertEqual(winners.iloc[0]["horizon_hours"],6)
        self.assertEqual((winners.iloc[0]["g_name"], winners.iloc[0]["h_name"]),pairs[1])
        self.assertTrue(np.all(np.diff(winners["aggregate_rmse"])>=0.0))


if __name__=="__main__": unittest.main()
