"""Shared single-day API, exact historical selection and executable smoke tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.method.single_day import selection, prepare
from src.exe.run_model.bm_1 import fit_single_day as bm_script
from src.exe.run_model.wm_4 import fit_single_day as wm_script


class FakeEstimator:
    NAMES=("alpha","beta","k","Xi","C","sigma_x","sigma_y")
    def __init__(self):
        self.solver=self
        self.fitted_histories=None

    def initial_guess(self, historical):
        n=len(historical)
        return {"parameters":dict(zip(self.NAMES,[2.,2.,1.,.5,2.,.15,.075])),
                "initial_distributions":np.full((n,3),1./3)}

    def fit(self, historical, **kwargs):
        self.fitted_histories=historical
        n=len(historical)
        fitted=dict(zip(self.NAMES,[2.,2.,1.,.5,2.,.15,.075]))
        return {**fitted,"initial_distributions":np.full((n,3),1./3),
                "nll":float(n*len(historical[0]["I"])*.5),
                "converged":True,"message":"success","evaluations":1}

    def simulate(self, data, fitted, initial, observations=False):
        observed=np.asarray(data["I"],dtype=float)
        return {"mean":observed.copy(),"loglik":np.full(len(observed),-.5),
                "nll":float(len(observed)*.5)}

    @staticmethod
    def intervals(forecast, sigma_y):
        return forecast["mean"]-.1,forecast["mean"]+.1

    def oracle(self, target, fitted, maxiter=35):
        simulation=self.simulate(target,fitted,np.full(3,1./3))
        lo,hi=self.intervals(simulation,fitted["sigma_y"])
        return {"prediction":simulation,"lower_95":lo,"upper_95":hi,
                "initial_distribution":np.full(3,1./3),"oracle_converged":True,
                "oracle_message":"success","oracle_evaluations":1}


class SingleDayTests(unittest.TestCase):
    @staticmethod
    def frame():
        index=pd.date_range("2022-01-01",periods=24*170,freq="h")
        h=index.hour.to_numpy(dtype=float)
        t=np.arange(len(index),dtype=float)
        return pd.DataFrame({"I":.5+.09*np.sin(2*np.pi*h/24)+.02*np.cos(2*np.pi*t/168),
                             "U":.3*np.cos(2*np.pi*t/24),
                             "Temp":8+4*np.sin(2*np.pi*t/(24*19))},index=index)

    def test_origin_cross_midnight_and_argument_validation(self):
        origin,history=selection("2022-04-11",20,12,3,"weekly_temperature")
        self.assertEqual(origin,pd.Timestamp("2022-04-11 20:00"))
        self.assertEqual(history,[origin-pd.Timedelta(weeks=w) for w in (3,2,1)])
        for kwargs in ({"origin_point":24},{"k":1},{"k":25},{"N":0},
                       {"baseline":"not_registered"}):
            opts={"day":"2022-04-11","origin_point":20,"k":12,
                  "N":3,"baseline":"weekly_temperature"}
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                selection(**{**opts,**kwargs})

    def test_exact_cross_midnight_baseline_and_bm_price_domain(self):
        frame=self.frame()
        with patch("src.method.single_day.group_frame",return_value=frame):
            case=prepare("2022-04-11",20,12,3,"weekly_temperature",bm_price=True)
        self.assertEqual(case["baseline"].metadata["samples"],36)
        self.assertEqual(len(case["history"]),3)
        self.assertEqual(case["history"][-1],pd.Timestamp("2022-04-04 20:00"))
        self.assertAlmostEqual(case["raw"].loc["2022-04-11 20:00","U"],
                               .5*(1+frame.loc["2022-04-11 20:00","U"]))
        target_idx=pd.date_range("2022-04-11 20:00",periods=12,freq="h")
        changed=frame.copy()
        changed.loc[target_idx,"I"]=.95
        with patch("src.method.single_day.group_frame",return_value=changed):
            alternative=prepare("2022-04-11",20,12,3,"weekly_temperature",bm_price=True)
        probe=target_idx
        temp=frame.reindex(probe)["Temp"].to_numpy()
        np.testing.assert_allclose(case["baseline"].predict(probe,temp),
                                   alternative["baseline"].predict(probe,temp))
        self.assertTrue(case["training"].index.max()<case["origin"])

    def test_target_observation_is_not_imputed(self):
        frame=self.frame()
        frame.loc[pd.Timestamp("2022-04-12 03:00"),"I"]=np.nan
        with patch("src.method.single_day.group_frame",return_value=frame):
            with self.assertRaisesRegex(ValueError,"complete observed"):
                prepare("2022-04-11",20,12,3,"weekly_temperature",bm_price=True)

    def test_bm_executable_uses_historical_mle_and_target_oracle(self):
        frame=self.frame()
        estimator=FakeEstimator()
        class FakeModel:
            def fixed_beta_estimator(self,degree,quadrature,steps_per_hour):
                return estimator
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.method.single_day.group_frame",return_value=frame),\
                 patch.object(bm_script,"bm_1",return_value=FakeModel()):
                result=bm_script.run("2022-04-11",20,5,2,"weekly_average",
                                     output=directory,quiet=True)
            summary=result["summary"]
            self.assertEqual(summary["k"],5)
            self.assertEqual(summary["N"],2)
            self.assertEqual(summary["origin_point"],20)
            self.assertEqual(summary["historical"]["observations"],10)
            self.assertEqual(summary["target"]["observations"],5)
            self.assertEqual(len(estimator.fitted_histories),2)
            self.assertTrue(all(len(d["I"])==5 for d in estimator.fitted_histories))
            self.assertTrue(all(np.all((d["U"]>=0)&(d["U"]<=1)) for d in estimator.fitted_histories))
            for name in ("historical.png","target.png","historical_hourly.csv",
                         "target_hourly.csv","summary.json"):
                self.assertTrue((Path(directory)/name).is_file(),name)
            self.assertEqual(json.loads((Path(directory)/"summary.json").read_text())["k"],5)

    def test_wm_executable_maps_shared_keywords_to_report_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(wm_script,"fit_and_plot",return_value={"fit":{"r":[-.8,-.2,-1.4]}}) as fit:
                result=wm_script.run(day="2022-04-11",origin_point=20,k=8,N=3,
                                     baseline="weekly_temperature",output=directory)
            self.assertEqual(result["fit"]["r"],[-.8,-.2,-1.4])
            fit.assert_called_once()
            opts=fit.call_args.kwargs
            self.assertEqual((opts["day"],opts["hour"],opts["k"],opts["N"],opts["B"]),
                             ("2022-04-11",20,8,3,"weekly_temperature"))
            self.assertEqual(opts["output"],Path(directory)/"historical.png")


if __name__=="__main__": unittest.main()
