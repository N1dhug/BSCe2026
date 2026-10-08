"""Exact WM interval law, variable projection and common-information contract."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from src.method import g_wm_hourly as model
from src.model.wm_4 import wm_4
from tests import test_bm1_s3_interval as fixture
from src.method import bm_1_s3_residual as residual


class MatchedHourlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.S3IntervalTests.setUpClass();cls.parent=fixture.S3IntervalTests.snapshot;cls.raw=fixture.S3IntervalTests.raw
        def optimizer(fun,seed,**kwargs):
            fun(np.asarray(seed));return SimpleNamespace(x=np.asarray(seed),success=True,message='synthetic fixture',
                active_mask=np.zeros(3,dtype=int),jac=np.eye(3))
        with patch('src.method.g_wm_hourly.least_squares',side_effect=optimizer):
            cls.snapshot=model.fit_wm(cls.parent,cls.raw)['snapshot']
        cls.errors=residual.fit_residual(cls.parent,cls.raw)['snapshot']

    def test_exact_WM_hour_integral_balance_and_native_Euler_limit(self):
        r=[-2.,-1.,-4.];u=.3;initial=[.2,-.1]
        out=model.solve([u],[.5],r,initial)
        self.assertAlmostEqual(out['prediction'][0]-.5,24*(out['states'][1,0]-initial[0]),places=13)
        stationary=model.solve([u]*24,[.5]*24,r,[-u/2,0])
        np.testing.assert_allclose(stationary['prediction'],.5,atol=1e-14)
        native=wm_4((0.,1/24),4097,lambda _: .5)
        state=native.deterministic_solver(initial,{'U':lambda _:u,'Time':lambda t:24*t,'r':r})
        reference=np.trapz(state['I'],dx=1/4096)
        self.assertLess(abs(reference-out['prediction'][0]),2e-6)
        # Split the hour: temporal integral must be additive.
        A,b,H,j=model.hourly_operator(r,fraction=.5)
        middle=A@initial+b*u
        split=.5+.5*(H@initial+j*u+H@middle+j*u)
        self.assertAlmostEqual(split,out['prediction'][0],places=13)

    def test_bounded_state_projection_matches_independent_solver(self):
        H=np.array([[1.,0.],[1.,1.],[1.,2.],[1.,3.]])
        for y in (H@np.array([3.,.2]),H@np.array([-.2,.3])):
            actual=model.project_initial(H,y)
            expected=lsq_linear(H,y,bounds=([-1.,-np.inf],[1.,np.inf]),tol=1e-12).x
            np.testing.assert_allclose(actual,expected,atol=1e-10)
        singular=np.column_stack((np.zeros(4),np.ones(4)))
        np.testing.assert_allclose(singular@model.project_initial(singular,np.ones(4)),1.)

    def test_parent_binding_training_cutoff_and_input_only_forecast(self):
        changed=self.raw.copy();changed.loc[changed.index>=pd.Timestamp(self.parent['origin']),:]=1e20
        a=model.history(self.parent,self.raw);b=model.history(self.parent,changed)
        for left,right in zip(a,b):np.testing.assert_array_equal(left['observed'],right['observed'])
        parent=deepcopy(self.parent);parent['population']='changed'
        with self.assertRaises(ValueError):model.validate_wm(self.snapshot,parent)
        out=model.predict_comparison(self.parent,self.snapshot,self.raw,g_residual=self.errors)
        changed=self.raw.copy();changed['Demand']=1e20
        other=model.predict_comparison(self.parent,self.snapshot,changed,g_residual=self.errors)
        pd.testing.assert_frame_equal(out['predictions'],other['predictions'])
        actual=fixture.interval.predict_interval(self.parent,self.raw)['predictions']
        np.testing.assert_array_equal(out['predictions'].g_prediction,actual.prediction)
        np.testing.assert_array_equal(out['predictions'].baseline,actual.baseline)
        standalone=residual.predict_residual(self.parent,self.errors,self.raw)['predictions']
        for new,old in (('g_prediction_lower','observed_lower'),('g_prediction_upper','observed_upper'),
                        ('g_prediction_output_mean','observed_mean')):
            np.testing.assert_allclose(out['predictions'][new],standalone[old],atol=1e-15,rtol=0)

    def test_common_mask_and_empirical_bands_use_raw_scaling(self):
        out=model.predict_comparison(self.parent,self.snapshot,self.raw,g_residual=self.errors)
        scale=self.parent['scales']['demand_scale']
        for name in ('g_prediction','wm_prediction'):
            np.testing.assert_allclose(out['predictions'][name+'_output_variance'],
                out['normalized_predictions'][name+'_output_variance']*scale**2)
        observed=out['predictions'].wm_prediction.copy();observed.iloc[0]=np.nan
        scored=model.score_comparison(out,observed)
        self.assertEqual(scored['scoring']['scored_samples'],167)
        for stats in scored['scoring']['metrics'].values():self.assertEqual(stats['samples'],167)
        self.assertEqual(scored['scoring']['metrics']['wm_prediction']['rmse'],0.)

    def test_synthetic_interval_fit_recovers_free_run_law(self):
        true=[-4.,-2.,-7.];items=[]
        for d in range(7):
            for row in range(2):
                price=.6*np.sin(np.arange(24)*(.25+.1*row)+d)
                baseline=np.full(24,.5);initial=[.2*np.sin(d),.03*(row+1)]
                y=model.solve(price,baseline,true,initial)['prediction']
                items.append({'weekday':d,'day':f'{d}-{row}','index':pd.date_range('2022-01-01',periods=24,freq='h'),
                    'price':price,'baseline':baseline,'observed':y,'mask':np.ones(24,bool)})
        with patch('src.method.g_wm_hourly.history',return_value=items):
            out=model.fit_wm(self.parent,self.raw,max_calls=400,seconds=60.)
        self.assertEqual(out['status'],'complete')
        self.assertLess(out['snapshot']['fit']['mse'],1e-11)
        np.testing.assert_allclose(out['snapshot']['r_per_day'],true,atol=.01,rtol=.01)


if __name__=='__main__':unittest.main()
