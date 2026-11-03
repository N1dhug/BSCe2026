import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.plot_corresponding_interval_fit import (
    BASELINES,_clock_axis,_context,_fit_sparse_baseline,_intervals,_origin,_window,run,
)


class test_wm4_corresponding_interval_fit(unittest.TestCase):
    @staticmethod
    def frame():
        index=pd.date_range("2022-01-01",periods=24*250,freq="h")
        hour=index.hour.to_numpy(dtype=float)
        days=(index-index[0]).total_seconds().to_numpy()/86400
        demand=.50+.06*np.cos(2*np.pi*hour/24)+.015*np.cos(2*np.pi*days/7)
        temperature=11+6*np.cos(2*np.pi*hour/24)+2*np.sin(2*np.pi*days/30)
        price=.3*np.sin(2*np.pi*hour/24)
        return pd.DataFrame({"I":demand,"U":price,"Temp":temperature},index=index)

    def test_corresponding_origins_and_window_sizes(self):
        origin=_origin("2022-06-20",8)
        self.assertEqual(_intervals(origin,12,4),[
            origin-pd.Timedelta(weeks=i) for i in (4,3,2,1)])
        with self.assertRaises(ValueError): _origin("2022-06-20",24)
        with self.assertRaises(ValueError): _intervals(origin,1,4)
        with self.assertRaises(ValueError): _intervals(origin,12,0)
        self.assertEqual(len(_intervals(origin,7,5)),5)

    def test_exact_window_baselines(self):
        frame=self.frame()
        starts=_intervals(_origin("2022-06-20",8),7,4)
        samples=pd.concat([frame.reindex(pd.date_range(start,periods=7,freq="h"))
                           [["I","Temp"]] for start in starts])
        self.assertEqual(len(samples),28)
        for name in BASELINES:
            with self.subTest(name=name):
                fitted=_fit_sparse_baseline(samples,name)
                self.assertEqual(fitted.metadata["samples"],28)
                predicted=fitted.predict(samples.index,samples["Temp"].to_numpy())
                self.assertEqual(predicted.shape,(28,))
                self.assertTrue(np.isfinite(predicted).all())
                self.assertTrue(((predicted>=0)&(predicted<=1)).all())

    def test_model_grid_and_exact_24_to_25_hour_context(self):
        frame=self.frame()
        origin=_origin("2022-06-20",8)
        for k,edge,size in [(12,6,24),(7,9,25),(24,0,24),(23,1,25)]:
            with self.subTest(k=k):
                x,values=_context(frame,origin,k,edge)
                self.assertEqual(len(x),size+1)
                self.assertEqual(len(values),size+1)
                self.assertEqual(x[0],-edge)
                self.assertEqual(x[-1],k+edge)
                starts=_intervals(origin,k,2)
                data=pd.concat([frame.reindex(pd.date_range(t,periods=k,freq="h"))[["I","Temp"]]
                                for t in starts])
                baseline=_fit_sparse_baseline(data,"weekly_average")
                window=_window(frame,starts[0],k,baseline,4)
                self.assertEqual(len(window["I"]),k*4+1)
                self.assertEqual(window["observation_mask"].sum(),k)
                self.assertEqual(window["transition_mask"].sum(),(k-1)*4)
                self.assertEqual(len(window["O"]),k*4+1)

    def test_clock_axis_uses_actual_hour_and_wraps_at_midnight(self):
        for hour,k,edge,expected_left,expected_right in [
            (6,12,6,0,24),(8,12,6,2,26),(6,7,9,-3,22),(23,7,9,14,39)]:
            with self.subTest(hour=hour,k=k):
                start=pd.Timestamp("2022-06-20")+pd.Timedelta(hours=hour)
                actual,left,right,ticks,labels=_clock_axis(start,k,edge)
                self.assertEqual(actual,hour)
                self.assertEqual(left,expected_left)
                self.assertEqual(right,expected_right)
                self.assertIn(hour,ticks)
                self.assertIn(hour+k,ticks)
                self.assertEqual(labels[list(ticks).index(hour)],f"{hour:02d}")
                self.assertEqual(labels[list(ticks).index(hour+k)],
                                 f"{(hour+k)%24:02d}")
                self.assertTrue(all(0<=int(label)<=23 for label in labels))
        _,_,_,ticks,labels=_clock_axis(pd.Timestamp("2022-06-20 08:00"),12,6)
        self.assertEqual(labels[list(ticks).index(24)],"00")

    def test_run_produces_single_grid_with_n_independent_states(self):
        frame=self.frame()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"example.png"
            def example_result(inputs):
                n=len(inputs)
                return {
                    "r":np.array([-1.0,-.1,-2.0]),
                    "qp_loss":.005,
                    "init_cond":np.array([[.05+i*.01,
                        float(data["I"][0]-data["O"][0])] for i,data in enumerate(inputs)]),
                    "deterministic_states":[{"I":data["I"]+.001*i}
                                            for i,data in enumerate(inputs)],
                }
            with patch("src.exe.run_model.wm_4.plot_corresponding_interval_fit.group_frame",
                       return_value=frame),patch(
                       "src.exe.run_model.wm_4.plot_corresponding_interval_fit.joint_qp_params"
                       ) as joint,patch(
                       "src.exe.run_model.wm_4.plot_corresponding_interval_fit.optimized_qp_params"
                       ) as oracle_qp,patch(
                       "src.exe.run_model.wm_4.plot_corresponding_interval_fit.plt.close"
                       ) as close_plot:
                joint.return_value.fit_many.side_effect=example_result
                def predict_oracle(data,theta):
                    np.testing.assert_array_equal(theta,np.array([-1.0,-.1,-2.0]))
                    self.assertEqual(len(data["I"]),7*4+1)
                    self.assertEqual(data["observation_mask"].sum(),7)
                    self.assertEqual(float(data["I"][0]),float(frame.at[pd.Timestamp("2022-06-20 08:00"),"I"]))
                    return {"init_cond":np.array([.12,-.03]),
                            "deterministic_state":{"I":data["I"]+.002},
                            "loss":.000004}
                oracle_qp.return_value.optimize_prediction_initial_state.side_effect=predict_oracle
                outcome=run(day="2022-06-20",hour=8,k=7,N=4,
                            B="weekly_temperature",group="MED",output=output)
                joint.return_value.fit_many.assert_called_once()
                oracle_qp.return_value.optimize_prediction_initial_state.assert_called_once()
                self.assertEqual(close_plot.call_count,2)
                grid=close_plot.call_args_list[0].args[0]
                prediction=close_plot.call_args_list[1].args[0]
                for ax in grid.axes:
                    fit_line=next(line for line in ax.get_lines()
                                  if line.get_label()=="WM4 fit (fitting interval)")
                    baseline_line=next(line for line in ax.get_lines()
                                       if line.get_label()=="Baseline (fitting interval)")
                    demand_line=next(line for line in ax.get_lines()
                                     if line.get_label()=="Observed demand (full context)")
                    self.assertAlmostEqual(float(fit_line.get_xdata()[0]),8)
                    self.assertAlmostEqual(float(fit_line.get_xdata()[-1]),15)
                    self.assertAlmostEqual(float(baseline_line.get_xdata()[0]),8)
                    self.assertAlmostEqual(float(demand_line.get_xdata()[0]),-1)
                    self.assertEqual(ax.get_xticklabels()[list(ax.get_xticks()).index(8)].get_text(),"08")
                self.assertEqual(len(prediction.axes),1)
                ax=prediction.axes[0]
                oracle_line=next(line for line in ax.get_lines()
                                 if line.get_label()=="WM4 oracle prediction (prediction interval)")
                self.assertEqual(oracle_line.get_color(),"red")
                self.assertAlmostEqual(float(oracle_line.get_xdata()[0]),8)
                self.assertAlmostEqual(float(oracle_line.get_xdata()[-1]),15)
                self.assertEqual(len(oracle_line.get_xdata()),7*4+1)
                self.assertAlmostEqual(outcome["oracle"]["oracle_F0"],.12)
                self.assertAlmostEqual(outcome["oracle"]["oracle_R0"],-.03)
                self.assertAlmostEqual(outcome["oracle"]["oracle_rmse"],.002)
                self.assertEqual(ax.get_xticklabels()[list(ax.get_xticks()).index(8)].get_text(),"08")
                import matplotlib.pyplot as plt
                plt.close(grid)
                plt.close(prediction)
            self.assertTrue(output.is_file())
            self.assertEqual(len(list(Path(tmp).glob("*.png"))),2)
            self.assertTrue(outcome["oracle_plot"].is_file())
            self.assertEqual(outcome["oracle_plot"].name,"example_oracle_prediction.png")
            self.assertTrue(outcome["oracle_plot"].with_suffix(".csv").is_file())
            self.assertTrue(output.with_suffix(".csv").is_file())
            self.assertTrue(output.with_suffix(".json").is_file())
            self.assertEqual(len(outcome["summary"]),4)
            self.assertEqual(len(outcome["oracle"]["oracle_hourly"]),7)
            self.assertEqual(outcome["baseline"].metadata["samples"],4*7)
            self.assertEqual(len(outcome["historical_intervals"]),4)
            self.assertTrue((outcome["summary"]["wm4_rmse"]>=0).all())


if __name__=="__main__":unittest.main()
