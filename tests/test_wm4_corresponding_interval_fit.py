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
                                  if line.get_label()=="ARM fit (fitting interval)")
                    baseline_line=next(line for line in ax.get_lines()
                                       if line.get_label()=="Baseline (fitting interval)")
                    demand_line=next(line for line in ax.get_lines()
                                     if line.get_label()=="Observed demand (full context)")
                    self.assertAlmostEqual(float(fit_line.get_xdata()[0]),8)
                    self.assertAlmostEqual(float(fit_line.get_xdata()[-1]),14)
                    self.assertEqual(len(fit_line.get_xdata()),7)
                    self.assertEqual(fit_line.get_drawstyle(),"default")
                    self.assertEqual(baseline_line.get_drawstyle(),"default")
                    np.testing.assert_array_equal(np.diff(fit_line.get_xdata()),np.ones(6))
                    np.testing.assert_array_equal(np.diff(baseline_line.get_xdata()),np.ones(6))
                    self.assertAlmostEqual(float(baseline_line.get_xdata()[0]),8)
                    self.assertEqual(len(baseline_line.get_xdata()),7)
                    self.assertAlmostEqual(float(demand_line.get_xdata()[0]),-1)
                    self.assertEqual(demand_line.get_marker(),"o")
                    self.assertEqual(demand_line.get_drawstyle(),"default")
                    self.assertEqual(ax.get_xticklabels()[list(ax.get_xticks()).index(8)].get_text(),"08")
                self.assertEqual(len(prediction.axes),1)
                self.assertTrue(any(collection.get_label()=="95% demand interval (historical fit)"
                                    for ax in grid.axes for collection in ax.collections))
                self.assertTrue(any(collection.get_label()=="95% oracle-conditioned demand interval"
                                    for collection in prediction.axes[0].collections))
                ax=prediction.axes[0]
                oracle_line=next(line for line in ax.get_lines()
                                 if line.get_label()=="ARM oracle prediction (prediction interval)")
                self.assertEqual(oracle_line.get_color(),"red")
                self.assertAlmostEqual(float(oracle_line.get_xdata()[0]),8)
                self.assertAlmostEqual(float(oracle_line.get_xdata()[-1]),14)
                self.assertEqual(len(oracle_line.get_xdata()),7)
                self.assertEqual(oracle_line.get_drawstyle(),"default")
                np.testing.assert_array_equal(np.diff(oracle_line.get_xdata()),np.ones(6))
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
            self.assertIn("demand_std",outcome["oracle"]["oracle_hourly"])
            self.assertIn("lower_95",outcome["oracle"]["oracle_hourly"])
            self.assertIn("process_sigma",outcome["summary"])
            self.assertIn("observation_tau",outcome["summary"])
            self.assertEqual(len(outcome["uncertainty"]["sd_fine"]),7*4+1)
            self.assertEqual(outcome["baseline"].metadata["samples"],4*7)
            self.assertEqual(len(outcome["historical_intervals"]),4)
            self.assertTrue((outcome["summary"]["wm4_rmse"]>=0).all())


    def test_full_day_historical_and_oracle_plots_connect_hourly_observations(self):
        """At k=24, lines and intervals use the same hourly grid as BM1."""
        from src.exe.run_model.wm_4.plot_corresponding_interval_fit import _plot, _plot_oracle
        frame=self.frame()
        origin=_origin("2022-06-20",0)
        history=_intervals(origin,24,2)
        baseline_frame=pd.concat([frame.reindex(pd.date_range(t,periods=24,freq="h"))[["I","Temp"]]
                                  for t in history])
        baseline=_fit_sparse_baseline(baseline_frame,"weekly_average")
        sph=4
        inputs=[_window(frame,start,24,baseline,sph) for start in history]
        target=_window(frame,origin,24,baseline,sph)
        fit={"r":np.array([-.8,-.1,-1.4]),"qp_loss":0.,
             "init_cond":np.zeros((2,2)),
             "deterministic_states":[{"I":item["I"]} for item in inputs]}
        uncertainty={"sigma":.1,"tau":.01,"historical_coverage_95":1.,
                     "sd_fine":np.full(24*sph+1,.025),
                     "sd_hourly":np.full(24,.025)}
        oracle={"init_cond":np.zeros(2),"deterministic_state":{"I":target["I"]}}
        with tempfile.TemporaryDirectory() as tmp:
            historical_file=Path(tmp)/"history.png"
            oracle_file=Path(tmp)/"oracle.png"
            with patch("src.exe.run_model.wm_4.plot_corresponding_interval_fit.plt.close") as close_fig, \
                 patch("matplotlib.axes.Axes.step",side_effect=AssertionError("No step plots")):
                _plot(history,frame,24,2,"weekly_average",sph,fit,None,inputs,
                      historical_file,uncertainty)
                _plot_oracle(origin,frame,24,2,"weekly_average",sph,target,oracle,
                             oracle_file,uncertainty)
                hist_figure,oracle_figure=[call.args[0] for call in close_fig.call_args_list]
                for index,ax in enumerate(hist_figure.axes):
                    lines={line.get_label():line for line in ax.lines}
                    for label,expected in (
                        ("ARM fit (fitting interval)",inputs[index]["I"][np.arange(24)*sph]),
                        ("Baseline (fitting interval)",inputs[index]["O"][np.arange(24)*sph])
                    ):
                        line=lines[label]
                        np.testing.assert_array_equal(line.get_xdata(),np.arange(24))
                        np.testing.assert_allclose(line.get_ydata(),expected)
                        self.assertEqual(line.get_drawstyle(),"default")
                    observed=lines["Observed demand (full context)"]
                    self.assertEqual(observed.get_marker(),"o")
                    self.assertEqual(len(observed.get_xdata()),24)
                lines={line.get_label():line for line in oracle_figure.axes[0].lines}
                predicted=lines["ARM oracle prediction (prediction interval)"]
                np.testing.assert_array_equal(predicted.get_xdata(),np.arange(24))
                np.testing.assert_allclose(predicted.get_ydata(),target["I"][np.arange(24)*sph])
                self.assertEqual(predicted.get_drawstyle(),"default")
                self.assertEqual(lines["Observed demand (full context)"].get_marker(),"o")
                import matplotlib.pyplot as plt
                plt.close(hist_figure)
                plt.close(oracle_figure)
            self.assertTrue(historical_file.is_file())
            self.assertTrue(oracle_file.is_file())


if __name__=="__main__":unittest.main()
