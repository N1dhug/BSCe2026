"""Selected-candidate checks detect corrupt scores and recompute own laws."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import numpy as np
from src.method.bm_1_response import CurveTrainingRecord,fit_shared_curve_pair
from src.params.bm_1 import BM1ObservationSeries
from src.params.bm_1_response import BM1PiecewiseResponse
from src.validation.bm_1.response import _score_record
from src.validation.bm_1 import response_fit_checks as checks
from src.validation.bm_1.response_numerics import run_numerical_unit

class SelectedCurveTests(unittest.TestCase):
    def fixture(self):
        c=lambda t:np.full(np.asarray(t).shape,.5);f=BM1PiecewiseResponse.linear()
        record=CurveTrainingRecord('a','TRAIN',c,(0.,1.),(c,f,f,1.,2.,0.),0.,.1,np.diff,
            BM1ObservationSeries([.25,.75],[.5,.6],'day'))
        fit=fit_shared_curve_pair([record],pieces=2,starts=[[0.,0.],[.8,-.8]],regularization=.1,
            max_evaluations=250,cells=11,max_step=.25,quadrature_order=1,backend='fitted')
        payload={k:v for k,v in fit.items() if k not in ('f','g','replay','logits','starts')}
        payload.update(case='no_response',pieces=2,f=fit['f'].to_record(),g=fit['g'].to_record(),
                       logits=fit['logits'].tolist(),replay=_score_record(fit['replay']))
        return record,fit,payload

    def test_raw_fit_algebra_and_selected_replay_are_checked(self):
        _,fit,payload=self.fixture()
        with patch.dict(checks.H1_V2_NUMERICS,{'base':dict(cells=11,max_step=.25,quadrature_order=1)},clear=True):
            checks.validate_fit_result(payload,fit['trace'],case='no_response',pieces=2,record_ids=['a'])
            for field in ('log_likelihood','objective'):
                bad=deepcopy(payload);bad[field]+=1
                with self.assertRaises(ValueError):checks.validate_fit_result(bad,fit['trace'],case='no_response',pieces=2,record_ids=['a'])
            bad=deepcopy(payload);bad['replay']['records'][0]['log_evidence'][0]+=1
            with self.assertRaises(ValueError):checks.validate_fit_result(bad,fit['trace'],case='no_response',pieces=2,record_ids=['a'])

    def test_selected_curve_has_own_full_history_and_support(self):
        record,fit,payload=self.fixture()
        settings={k:dict(cells=11,max_step=.25,quadrature_order=1) for k in ('base','time','spatial','quadrature')}
        with patch.dict(checks.H1_V2_NUMERICS,settings,clear=True):
            f=BM1PiecewiseResponse.linear();linear=run_numerical_unit(record,f,f,numerics=settings,backend='fitted')
            result=checks.selected_curve_unit(record,fit['f'],fit['g'],linear,payload['replay']['records'][0])
            self.assertTrue(result['numerical_passes'])
            np.testing.assert_allclose(result['posterior_quarter_mass'].sum(axis=1),1.,rtol=0,atol=1e-12)
            bad=deepcopy(payload['replay']['records'][0]);bad['log_evidence'][0]+=1
            with self.assertRaises(ValueError):checks.selected_curve_unit(record,fit['f'],fit['g'],linear,bad)

if __name__=='__main__':unittest.main()
