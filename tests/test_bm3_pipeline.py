"""Scientific and information-set checks for the experimental BM3 model."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.model.bm_3 import bm_3
from src.params.bm_3 import lsq_params, historical_diffusion_params
from src.params.bm_1 import lsq_params as BM1Fit
from src.exe.run_model.bm_3.predict_daily import predict_day, save_result
from src.exe.run_model.bm_3.predict_mpc import predict_origin
from src.tools.baseline import statistical_output
from tests.test_bm1_thomas_pipeline import frame_fixture


def parameters():
    p = dict(alpha=2., beta=2., k=1., C=2., Xi=.5, restoring_strength=10.)
    p.update(f=lsq_params.response(p['alpha']), g=lsq_params.response(p['beta']))
    return p


def model(strength=10., **kwargs):
    return bm_3(lambda t: np.zeros_like(t)+.5, lambda t: .5, (0., 2.), 3,
                restoring_strength=strength, **kwargs)


class BM3Tests(unittest.TestCase):
    def test_restoring_sign_and_quadratic_growth(self):
        np.testing.assert_allclose(bm_3.restoring(np.array([-.2,-.1,0,.5,1,1.1,1.2]),10),
                                   [.4,.1,0,0,0,-.1,-.4], atol=1e-14)
        np.testing.assert_array_equal(model()._diffusion(np.array([-1.,0.,1.,2.]),.2), [.2]*4)

    def test_deterministic_interior_matches_BM1_and_outside_is_allowed(self):
        p = parameters()
        data = {'Time': np.arange(9.), 'B': np.full(9,.5), 'U': np.full(9,.3)}
        a = lsq_params._predict(data,.4,p,64)
        b = BM1Fit._predict(data,.4,p,64)
        np.testing.assert_allclose(a,b,atol=1e-12)
        for initial in (-.2,1.2):
            states, demand = lsq_params._predict(data,initial,p,32)
            self.assertEqual(states[0], initial)
            self.assertLess(abs(states[1]-.5), abs(initial-.5))
            self.assertTrue(np.all((demand>=0)&(demand<=1)))
            args = [lambda t: np.zeros_like(t)+.3,p['f'],p['g'],p['C'],p['k'],p['Xi']]
            solved = model().deterministic_solver([initial],args)
            np.testing.assert_allclose(solved[1], states[:3], atol=2e-7)

    def test_FV_crosses_physical_limits_and_conserves_probability(self):
        p = parameters(); m = model(); mass = m.initial_mass([.98]*5,151)
        args = [lambda t: np.zeros_like(t)+.5,p['f'],p['g'],p['C'],p['k'],p['Xi']]
        output = m.fokker_planc_reference_solver(mass,args,.25,max_step=.05)
        x, weights, diag = output[1],output[3],output[-1]
        np.testing.assert_allclose(weights.sum(axis=1),1,atol=1e-12)
        self.assertGreaterEqual(weights.min(),0)
        self.assertGreater(weights[-1,x>1].sum(),.01)
        self.assertTrue(diag['tail_check_passed'])

    def test_numerical_truncation_is_not_silently_accepted(self):
        p = parameters(); m = model(state_bounds=(-.01,1.01),tail_tolerance=1e-7)
        mass = m.initial_mass([.5]*5,101)
        args = [lambda t: np.zeros_like(t)+.5,p['f'],p['g'],p['C'],p['k'],p['Xi']]
        with self.assertRaisesRegex(RuntimeError,'widen'):
            m.fokker_planc_reference_solver(mass,args,.5)

    def test_stronger_restoring_force_reduces_excursion_size(self):
        p = parameters(); excursions = []
        for strength in (1.,100.):
            m = model(strength)
            args = [lambda t: np.zeros_like(t)+.5,p['f'],p['g'],p['C'],p['k'],p['Xi']]
            result = m.fokker_planc_reference_solver(m.initial_mass([.98]*5,201),args,.25,max_step=.05)
            x,mass = result[1],result[3][-1]
            excursions.append(float(mass @ (np.maximum(-x,0)+np.maximum(x-1,0))))
        self.assertLess(excursions[1],excursions[0])

    def test_additive_diffusion_calibration_responds_to_historical_residuals(self):
        p = parameters(); p['initial_states'] = np.array([.4,.45,.5,.55,.6])
        data = {'Time':np.arange(25.),'B':np.full(25,.5),'U':np.full(25,.4)}
        training = []
        for state in p['initial_states']:
            _, D = lsq_params._predict(data,state,p,4)
            errors = .03*np.sin(np.arange(24)*1.3); errors[0] = 0
            training.append({**data,'I':D[:24]+errors})
        result = historical_diffusion_params().fit_many(training,p,substeps=4)
        self.assertGreater(result['q'],0)
        self.assertEqual(result['samples'],115)
        self.assertAlmostEqual(result['sigma_x']**2,result['q'])

    def test_shared_joint_fit_retains_BM3_dynamics_and_nonworse_objective(self):
        p = parameters()
        data = {'Time':np.arange(25.),'B':np.full(25,.5),'U':.5+.2*np.sin(np.arange(25.)/3)}
        training = [{**data,'I':lsq_params._predict(data,float(x),p,4)[1][:24]}
                    for x in (.2,.35,.5,.65,.8)]
        fitter = lsq_params(model(7.))
        reference = fitter.fit_many(training,C=2.,max_iterations=1,max_nfev=3,state_maxiter=10)
        refined = fitter.refine_joint(training,reference,C=2.,max_nfev=2,max_seconds=5.)
        self.assertEqual(refined['restoring_strength'],7.)
        self.assertLessEqual(refined['sse'],reference['sse'])
        self.assertEqual(refined['initial_states'].shape,(5,))
        self.assertEqual(float(refined['f'](1.2)),-1.)

    def test_day_forecast_is_causal_and_saves_parallel_outputs(self):
        frame = frame_fixture(); day = pd.Timestamp('2022-02-20')
        options = dict(frame=frame,max_iterations=1,max_nfev=3,state_maxiter=10,C=2.,x_steps=101,sigma_x=.1)
        result = predict_day(day,score=True,**options)
        changed = frame.copy(); changed.loc[day:day+pd.Timedelta(hours=23),'I'] = .95
        other = predict_day(day,**{**options,'frame':changed})
        np.testing.assert_array_equal(result['hourly'].I_predicted,other['hourly'].I_predicted)
        self.assertEqual(len(result['initial_states']),5)
        with tempfile.TemporaryDirectory() as directory:
            folder = save_result(result,Path(directory),plot=True)
            for name in ('hourly.csv','training_days.csv','iterations.csv','fit.json','forecast.png','state_boundary.png'):
                self.assertTrue((folder/name).is_file(),name)
            metadata = json.loads((folder/'fit.json').read_text())
            self.assertEqual(metadata['model'],'BM3')
            self.assertFalse(metadata['model_configuration']['restoring_strength_fitted'])

    def test_mpc_is_causal_and_updates_from_origin_measurement(self):
        frame = frame_fixture(); origin = pd.Timestamp('2022-02-20')
        baseline = statistical_output().fit(frame,origin)
        options = dict(max_iterations=1,max_nfev=3,state_maxiter=10,x_steps=101,horizon=2,C=2.)
        rows, diag = predict_origin(frame,origin,baseline,**options)
        changed = frame.copy(); changed.loc[origin+pd.Timedelta(hours=1):origin+pd.Timedelta(hours=3),'I'] = .95
        other, _ = predict_origin(changed,origin,baseline,**options)
        np.testing.assert_array_equal([r['I_BM3'] for r in rows],[r['I_BM3'] for r in other])
        changed.loc[origin,'I'] = .7
        _, updated = predict_origin(changed,origin,baseline,fitted=diag['fit'],**options)
        self.assertGreater(np.max(abs(diag['posterior']-updated['posterior'])),.001)
        self.assertEqual([r['lead_hours'] for r in rows],[1,2])
        self.assertTrue(all(0<=r['I_BM3']<=1 for r in rows))


if __name__=='__main__': unittest.main()
