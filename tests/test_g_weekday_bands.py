"""Calendar/sample isolation, frozen prediction, rank calibration and scoring."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.method import g_weekday_rolling as rolling
from src.method import g_weekday_export as numerical
from src.method import g_prequential_bands as bands
from src.exe.run_model.bm_1.g_weekday_continue import check_saved_converged
from src.exe.run_model.bm_1.g_wm_solver_extension import fit_extension


def packet(origin,error=1.):
    start=pd.Timestamp(origin);end=start+pd.Timedelta(days=7)
    index=pd.date_range(start.tz_localize(rolling.interval.ZONE),end.tz_localize(rolling.interval.ZONE),freq='h',inclusive='left').tz_localize(None)
    raw=pd.DataFrame({'baseline':10.,'g_prediction':10.,'wm_prediction':10.},index=index)
    for name in bands.MODELS:
        raw[name+'_output_mean']=10.;raw[name+'_output_variance']=4.
        raw[name+'_lower']=9.;raw[name+'_upper']=11.
    normalized=raw.copy()
    for c in raw:
        normalized[c]=raw[c]/4 if c.endswith('_output_variance') else (raw[c]-6)/2
    result={'predictions':raw,'normalized_predictions':normalized,'manifest':{
        'comparison_protocol':numerical.PROTOCOL,'source_sha256':numerical.source_hashes(),
        'cutoff':str(start),'end_exclusive':str(end),'demand_used':False,'g_residual_attached':True,
        'demand_timestamp_label':'hour_start','scales':{'demand_min':6.,'demand_scale':2.}},
        'scoring':{'status':'not-scored'},'status':'complete-development-comparison'}
    return {'forecast':result,'observations':pd.Series(10.+error,index=index)}


class RollingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.origin=pd.Timestamp('2022-03-21')
        index=pd.date_range('2022-02-13','2022-03-28',freq='h',inclusive='left')
        index=index[index!=pd.Timestamp('2022-03-27 02:00')]
        n=np.arange(len(index));cls.raw=pd.DataFrame({'Demand':10+np.sin(n/4)+.3*np.cos(n/17),
            'Price':100+80*np.sin(n/8),'Temp':5+3*np.sin(n/90)},index=index)
        def gfit(inputs,*a,**kw):
            assert len(inputs.trajectories)==5
            return {'status':'converged','best_converged':{'parameters':[12.,.2,.5,.5],'g_mid':.25},
                'counts':{'residual_calls':0},'observed_samples':int(inputs.observation_mask.sum())}
        def optimizer(fun,seed,**kw):
            fun(np.asarray(seed));return SimpleNamespace(x=np.asarray(seed),success=True,message='synthetic fixture',
                active_mask=np.zeros(3,dtype=int),jac=np.eye(3))
        cls.gfit=staticmethod(gfit);cls.optimizer=staticmethod(optimizer)
        with patch.object(rolling.interval,'fit_strategy',side_effect=gfit),patch.object(rolling,'least_squares',side_effect=optimizer):
            cls.result=rolling.fit_rolling(cls.raw,cls.origin)

    def test_seven_five_day_fits_and_sixth_week_spring_calendar(self):
        s=self.result['snapshot'];self.assertEqual(self.result['status'],'complete')
        for d,item in enumerate(s['days']):
            self.assertEqual(len(item['training_days']),5)
            self.assertTrue(all(pd.Timestamp(day).weekday()==d for day in item['training_days']))
            self.assertEqual(item['G']['fit']['observed_samples'],120)
            self.assertEqual(item['baseline']['metadata']['samples'],120)
            self.assertEqual(item['baseline']['metadata']['training_days'],sorted(item['training_days']))
        forecast=rolling.predict_comparison(s,self.raw[['Price','Temp']])
        self.assertEqual(len(forecast['predictions']),167)
        self.assertNotIn(pd.Timestamp('2022-03-27 02:00'),forecast['predictions'].index)
        self.assertTrue(forecast['numerics']['passed'])
        scale=s['scales']['demand_scale']
        for model in bands.MODELS:
            np.testing.assert_allclose(forecast['predictions'][model+'_output_variance'],
                forecast['normalized_predictions'][model+'_output_variance']*scale**2)

    def test_no_older_or_future_demand_in_training_and_input_only_prediction(self):
        changed=self.raw.copy();start=self.origin-pd.Timedelta(weeks=5)
        changed.loc[(changed.index<start)|(changed.index>=self.origin),'Demand']=1e20
        left=rolling.training_data(self.raw,self.origin);right=rolling.training_data(changed,self.origin)
        pd.testing.assert_frame_equal(left[0],right[0]);self.assertEqual(left[2],right[2])
        changed['Demand']=1e20
        with patch.object(rolling,'fit_baseline',side_effect=AssertionError('refit')),patch.object(rolling,'fit_g',side_effect=AssertionError('refit')):
            a=rolling.predict_comparison(self.result['snapshot'],self.raw)
            b=rolling.predict_comparison(self.result['snapshot'],changed)
        pd.testing.assert_frame_equal(a['predictions'],b['predictions'])

    def test_baseline_fits_only_selected_weekday_and_snapshot_fails_closed(self):
        _,frame,_,start,origin=rolling.training_data(self.raw,self.origin)
        a=rolling.baseline_record(rolling.fit_baseline(frame,start,origin,0))
        changed=frame.copy();changed.loc[changed.index.weekday!=0,'I']=1e20
        b=rolling.baseline_record(rolling.fit_baseline(changed,start,origin,0));self.assertEqual(a,b)
        for mutation in ('days','source','parameters'):
            s=deepcopy(self.result['snapshot'])
            if mutation=='days':s['days'][0]['training_days'][0]='2022-03-21'
            elif mutation=='source':s['source_sha256']={}
            else:s['days'][0]['G']['parameters']=[700.,.2,.5,.5]
            with self.assertRaises(ValueError):rolling.validate_snapshot(s)

    def test_own_five_day_WM_fit_recovers_synthetic_hourly_law(self):
        true=[-4.,-2.,-7.];items=[]
        for d in range(5):
            price=.6*np.sin(np.arange(24)*(.25+.1*d)+d);baseline=np.full(24,.5)
            y=rolling.wm.solve(price,baseline,true,[.1*d,.03*(d+1)])['prediction']
            items.append({'price':price,'baseline':baseline,'observed':y,'mask':np.ones(24,dtype=bool)})
        fitted=rolling.fit_wm(items)
        self.assertEqual(fitted['status'],'complete')
        self.assertLess(fitted['fit']['mse'],1e-10)
        np.testing.assert_allclose(fitted['r_per_day'],true,atol=.02)

    def test_minute_export_preserves_parameters_and_uses_no_optimizer(self):
        parent=self.result['snapshot']
        with patch.object(rolling,'fit_g',side_effect=AssertionError('refit')),patch.object(rolling,'fit_wm',side_effect=AssertionError('refit')):
            exported=numerical.refine_for_export(parent,self.raw)['snapshot']
            prediction=numerical.predict_comparison(exported,self.raw[['Price','Temp']])
        self.assertEqual(exported['optimizer_calls'],0)
        self.assertEqual(prediction['manifest']['parent_snapshot_sha256'],rolling.residual.snapshot_fingerprint(parent))
        for old,new in zip(parent['days'],exported['refined_view']['days']):
            self.assertEqual(old['G']['parameters'],new['G']['parameters'])
            self.assertEqual(old['WM'],new['WM']);self.assertEqual(old['baseline'],new['baseline'])
        changed=deepcopy(exported);changed['refined_view']['days'][0]['G']['parameters']=[700.,.2,.5,.5]
        with self.assertRaises(ValueError):numerical.validate_export(changed)

    def test_exhausted_grid_refit_reuses_only_saved_converged_parameters(self):
        _,frame,_,start,origin=rolling.training_data(self.raw,self.origin)
        day=self.result['snapshot']['days'][0];g=deepcopy(day['G']);g['status']='unfinished';g['calls']=750;g['seconds']=1.
        with patch.object(rolling.interval,'fit_strategy',side_effect=AssertionError('new optimizer')):
            checked=check_saved_converged(g,frame,rolling.load_baseline(day['baseline']),day['training_days'])
        self.assertEqual(checked['parameters'],g['parameters']);self.assertEqual(checked['calls'],750)
        self.assertEqual(checked['numerical_check_optimizer_calls'],0);self.assertLess(checked['seconds'],30.)
        g['attempts']=[]
        with self.assertRaises(RuntimeError):check_saved_converged(g,frame,rolling.load_baseline(day['baseline']),day['training_days'])

    def test_explicit_WM_extension_preserves_failed_calls_and_original_time_cap(self):
        true=[-4.,-2.,-7.];items=[]
        for d in range(5):
            price=.6*np.sin(np.arange(24)*(.25+.1*d)+d);baseline=np.full(24,.5)
            y=rolling.wm.solve(price,baseline,true,[.1*d,.03*(d+1)])['prediction']
            items.append({'price':price,'baseline':baseline,'observed':y,'mask':np.ones(24,dtype=bool)})
        previous={'status':'unfinished','calls':400,'seconds':.5,'attempts':[{'converged':False}]}
        fit=fit_extension(items,previous)
        self.assertEqual(fit['status'],'complete');self.assertEqual(fit['previous_failed_fit'],previous)
        self.assertEqual(fit['calls'],400+fit['new_calls']);self.assertLessEqual(fit['calls'],1600);self.assertLess(fit['seconds'],10.)
        np.testing.assert_allclose(fit['r_per_day'],true,atol=.02)
        previous['seconds']=10.
        with self.assertRaises(RuntimeError):fit_extension(items,previous)


class BandTests(unittest.TestCase):
    def test_exact_finite_rank_boundaries_no_finite_small_sample_substitution(self):
        self.assertFalse(bands.finite_rank(np.ones(18))['deployable'])
        self.assertEqual(bands.finite_rank(np.arange(19))['rank'],19)
        self.assertEqual(bands.finite_rank(np.arange(399))['rank'],380)
        self.assertEqual(bands.finite_rank(np.zeros(504))['q'],0.)
        for scores in ([],[-1.],[np.nan]):
            with self.assertRaises(ValueError):bands.finite_rank(scores)

    def test_calibration_excludes_later_observations_and_wrong_calendar(self):
        packets=[packet(day,error=i+1.) for i,day in enumerate(bands.CALIBRATION_ORIGINS)]
        a=bands.calibrate(packets)['snapshot'];self.assertEqual(a['models']['g_prediction']['samples'],504)
        self.assertEqual(a['models']['g_prediction']['rank'],480)
        for p in packets:p['observations']=pd.concat([p['observations'],pd.Series(1e20,index=pd.date_range(bands.FREEZE_CUTOFF,periods=168,freq='h'))])
        self.assertEqual(a,bands.calibrate(packets)['snapshot'])
        with self.assertRaises(ValueError):bands.calibrate(packets[::-1])
        with self.assertRaises(ValueError):bands.calibrate([packet(bands.EVALUATION_ORIGINS[0]),*packets[1:]])
        bad=deepcopy(a);bad['models']['g_prediction']['q']+=1
        with self.assertRaises(ValueError):bands.validate_calibration(bad)
        bad=packet('2022-03-07')['forecast'];bad['predictions']['g_prediction_output_variance']=0
        with self.assertRaises(ValueError):bands.validate_forecast(bad)

    def test_apply_preserves_mean_variance_and_old_bands_with_affine_units(self):
        record=bands.calibrate([packet(d,3.) for d in bands.CALIBRATION_ORIGINS])['snapshot']
        original=packet('2022-03-21')['forecast'];out=bands.apply_calibration(original,record)
        pd.testing.assert_frame_equal(original['predictions'],out['predictions'][original['predictions'].columns])
        self.assertEqual(len(out['predictions']),167)
        for m in bands.MODELS:
            np.testing.assert_allclose(out['predictions'][m+'_calibrated_lower'],7.)
            np.testing.assert_allclose(out['predictions'][m+'_calibrated_lower'],6+2*out['normalized_predictions'][m+'_calibrated_lower'])
        with self.assertRaises(ValueError):bands.apply_calibration(packet('2022-02-28')['forecast'],record)

    def test_common_missing_mask_interval_score_and_pooled_rmse(self):
        record=bands.calibrate([packet(d,2.) for d in bands.CALIBRATION_ORIGINS])['snapshot'];scored=[]
        for i,day in enumerate(bands.EVALUATION_ORIGINS):
            p=packet(day,error=[1.,2.,4.][i]);p['observations'].iloc[0]=np.nan
            result=bands.score_calibrated(bands.apply_calibration(p['forecast'],record),p['observations']);scored.append(result)
            self.assertEqual(result['scoring']['calibrated_bands']['g_prediction']['samples'],len(p['observations'])-1)
        report=bands.pooled_report(scored);counts=np.array([166,167,167]);errors=np.array([1.,2.,4.])
        self.assertAlmostEqual(report['metrics']['g_prediction']['rmse'],np.sqrt(np.sum(counts*errors**2)/counts.sum()))
        self.assertAlmostEqual(scored[-1]['scoring']['calibrated_bands']['g_prediction']['mean_interval_score'],84.)
        self.assertEqual(report['scored_samples'],500)

    def test_training_calendar_skips_nonexistent_hour_without_price_filling(self):
        index=pd.date_range('2022-02-20','2022-04-04',freq='h',inclusive='left')
        index=index[index!=pd.Timestamp('2022-03-27 02:00')];n=np.arange(len(index))
        raw=pd.DataFrame({'Demand':10+np.sin(n/3),'Price':100.,'Temp':5.},index=index)
        train,frame,scales,start,origin=rolling.training_data(raw,'2022-04-04')
        self.assertEqual(len(train.loc[train.index>=start]),839)
        self.assertNotIn(pd.Timestamp('2022-03-27 02:00'),frame.index)
        self.assertEqual(frame.loc[frame.index>=start].I.count(),839)


if __name__=='__main__':unittest.main()
