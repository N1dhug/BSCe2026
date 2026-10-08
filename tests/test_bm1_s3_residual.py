"""Output-error law, exact empirical moments, cutoff and missing-hour checks."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tests import test_bm1_s3_interval as fixture
from src.method import bm_1_s3_residual as residual


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.S3IntervalTests.setUpClass();cls.parent=fixture.S3IntervalTests.snapshot;cls.raw=fixture.S3IntervalTests.raw
        with patch('src.method.bm_1_s3_interval.fit_strategy',side_effect=AssertionError('refit')):
            cls.result=residual.fit_residual(cls.parent,cls.raw)
        cls.record=cls.result['snapshot']

    def test_replay_no_future_or_refit_and_own_parent_binding(self):
        changed=self.raw.copy();changed.loc[changed.index>=pd.Timestamp(self.parent['origin']),:]=1e20
        other=residual.fit_residual(self.parent,changed)
        self.assertEqual(self.record,other['snapshot'])
        self.assertEqual(self.result['optimizer_calls'],0)
        parent=deepcopy(self.parent);parent['population']='changed population'
        with self.assertRaises(ValueError):residual.validate_residual(self.record,parent)
        bad=deepcopy(self.record);bad['residual_samples_normalized'][0]=np.nan
        with self.assertRaises(ValueError):residual.validate_residual(bad,self.parent)

    def test_exact_empirical_law_quantiles_and_raw_scaling(self):
        r=deepcopy(self.record);r['residual_samples_normalized']=[-.2,0.,.1,.5];r['sample_count']=4
        with patch('src.exe.run_model.wm_4.output_model.statistical_output.fit',side_effect=AssertionError('refit')):
            a=residual.predict_residual(self.parent,r,self.raw)
        t=a['normalized_predictions'];scale=self.parent['scales']['demand_scale']
        np.testing.assert_allclose(t.observed_mean-t.prediction,.1)
        np.testing.assert_allclose(t.observed_variance,np.var([-.2,0.,.1,.5]))
        np.testing.assert_allclose(t.observed_lower-t.prediction,-.2)
        np.testing.assert_allclose(t.observed_upper-t.prediction,.5)
        np.testing.assert_allclose(a['predictions'].observed_variance,t.observed_variance*scale**2)
        self.assertEqual(a['manifest']['residual_model']['process_sigma'],0.)

    def test_scoring_mask_interval_score_and_future_demand_independence(self):
        a=residual.predict_residual(self.parent,self.record,self.raw)
        changed=self.raw.copy();changed['Demand']=1e20
        b=residual.predict_residual(self.parent,self.record,changed)
        pd.testing.assert_frame_equal(a['predictions'],b['predictions'])
        y=a['predictions'].observed_mean.copy();y.iloc[0]=np.nan
        scored=residual.score_residual(a,y);stats=scored['scoring']['residual_bands']
        self.assertEqual(stats['scored_samples'],167)
        self.assertEqual(scored['scoring']['metrics']['observed_mean']['rmse'],0.)
        self.assertAlmostEqual(stats['coverage'],1.)
        self.assertAlmostEqual(stats['mean_interval_score'],stats['mean_width'])

    def test_missing_hours_do_not_create_lag_pairs_and_support_lower_bound(self):
        t=pd.DataFrame({'observed':[0.,np.nan,2.,3.],'prediction':[1.,1.,1.,1.],
            'baseline':[1.,1.,1.,1.],'lower':[.5,.5,.5,.5],'upper':[1.5,1.5,1.5,1.5],
            'day':['d']*4,'interval_start_utc':pd.date_range('2022-01-01',periods=4,freq='h',tz='UTC').astype(str)})
        audit=residual.summarize_replay(t)
        self.assertEqual(audit['within_day_adjacent_pairs'],1)
        self.assertEqual(audit['outside_integrated_support'],3)
        self.assertAlmostEqual(audit['support_distance_rmse_lower_bound'],np.sqrt((.25+.25+2.25)/3))


if __name__=='__main__':unittest.main()
