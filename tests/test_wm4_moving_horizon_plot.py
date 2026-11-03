import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.plot_moving_horizon_predictions import _origins, run


class test_wm4_moving_horizon_plot(unittest.TestCase):
    @staticmethod
    def frame():
        index=pd.date_range("2022-01-01",periods=24*250,freq="h")
        hour=index.hour.to_numpy(dtype=float)
        day=(index-index[0]).total_seconds().to_numpy()/86400
        return pd.DataFrame({
            "I":.50+.08*np.cos(2*np.pi*hour/24)+.02*np.sin(2*np.pi*day/7),
            "U":.3+.07*np.sin(2*np.pi*hour/24),
            "Temp":11+4*np.cos(2*np.pi*hour/24)+2*np.sin(2*np.pi*day/30),
        },index=index)

    def test_origins_are_hourly_and_cross_midnight(self):
        origins=_origins("2022-06-20",22,4)
        self.assertEqual(origins,[
            pd.Timestamp("2022-06-20 22:00"),
            pd.Timestamp("2022-06-20 23:00"),
            pd.Timestamp("2022-06-21 00:00"),
            pd.Timestamp("2022-06-21 01:00"),
        ])
        for invalid in (0,-1,1.5,True):
            with self.subTest(M=invalid),self.assertRaises(ValueError):
                _origins("2022-06-20",22,invalid)

    def test_run_refits_every_origin_and_saves_only_prediction_grid(self):
        frame=self.frame()
        origins=_origins("2022-06-20",22,4)
        joint_calls,oracle_calls,baseline_starts=[],[],[]
        def historical_fit(inputs):
            j=len(joint_calls)
            self.assertEqual(len(inputs),3)
            self.assertTrue(all(np.sum(item["observation_mask"])==7 for item in inputs))
            joint_calls.append(j)
            return {
                "r":np.array([-1.0-.01*j,-.1,-2.0]),
                "qp_loss":.01+j*.001,
                "init_cond":np.zeros((3,2)),
                "deterministic_states":[{"I":np.asarray(item["I"])}
                                        for item in inputs],
            }

        def prediction_fit(data,theta):
            j=len(oracle_calls)
            oracle_calls.append(j)
            np.testing.assert_allclose(theta,[-1.0-.01*j,-.1,-2.0])
            self.assertAlmostEqual(
                float(data["I"][0]),float(frame.at[origins[j],"I"]))
            self.assertEqual(np.sum(data["observation_mask"]),7)
            return {
                "init_cond":np.array([.1+j*.01,-.03]),
                "deterministic_state":{"I":np.asarray(data["I"])+.004},
                "loss":.004**2,
            }

        from src.exe.run_model.wm_4 import plot_moving_horizon_predictions as mod
        actual_baseline=mod._fit_sparse_baseline

        def baseline_fit(samples,B):
            j=len(baseline_starts)
            self.assertEqual(B,"weekly_temperature")
            self.assertEqual(len(samples),3*7)
            self.assertLess(samples.index.max(),origins[j])
            self.assertTrue(all(t in samples.index for t in [
                origins[j]-pd.Timedelta(weeks=n) for n in (1,2,3)]))
            baseline_starts.append(origins[j])
            return actual_baseline(samples,B)

        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"grid.png"
            with patch.object(mod,"group_frame",return_value=frame),patch.object(
                 mod,"_fit_sparse_baseline",side_effect=baseline_fit),patch.object(
                 mod,"joint_qp_params") as joint,patch.object(
                 mod,"optimized_qp_params") as oracle,patch.object(
                 mod.plt,"close") as close:
                joint.return_value.fit_many.side_effect=historical_fit
                oracle.return_value.optimize_prediction_initial_state.side_effect=prediction_fit
                result=run(day="2022-06-20",hour=22,k=7,N=3,M=4,
                           B="weekly_temperature",output=output)
                self.assertEqual(joint.return_value.fit_many.call_count,4)
                self.assertEqual(oracle.return_value.optimize_prediction_initial_state.call_count,4)
                self.assertEqual(close.call_count,1)
                figure=close.call_args.args[0]
                self.assertEqual(len(figure.axes),4)
                for j,ax in enumerate(figure.axes):
                    origin=origins[j]
                    line=next(line for line in ax.get_lines()
                              if line.get_label()=="WM4 oracle prediction")
                    self.assertEqual(line.get_color(),"red")
                    self.assertAlmostEqual(float(line.get_xdata()[0]),origin.hour)
                    self.assertAlmostEqual(float(line.get_xdata()[-1]),origin.hour+7)
                    self.assertEqual(len(line.get_xdata()),7*4+1)
                    tick_map=dict(zip(ax.get_xticks(),[label.get_text() for label in ax.get_xticklabels()]))
                    self.assertEqual(tick_map[origin.hour],f"{origin.hour:02d}")
                plt.close(figure)

            self.assertEqual(joint_calls,list(range(4)))
            self.assertEqual(oracle_calls,list(range(4)))
            self.assertEqual(baseline_starts,origins)
            self.assertTrue(output.exists())
            self.assertEqual(len(list(Path(tmp).glob("*.png"))),1)
            self.assertTrue(output.with_suffix(".csv").exists())
            self.assertTrue(output.with_name("grid_hourly.csv").exists())
            self.assertTrue(output.with_suffix(".json").exists())
            self.assertEqual(len(result["summary"]),4)
            self.assertEqual(len(result["hourly"]),4*7)
            self.assertEqual(result["summary"]["historical_samples"].tolist(),[3*7]*4)
            self.assertEqual(result["summary"]["origin"].tolist(),origins)
            self.assertEqual(result["summary"]["oracle_F0"].round(2).tolist(),[.10,.11,.12,.13])
            self.assertAlmostEqual(result["aggregate"]["rmse"],.004,places=10)
            self.assertEqual(result["aggregate"]["origins"],4)
            self.assertEqual(result["aggregate"]["total_forecast_samples"],28)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            run(day="2022-06-20",hour=6,k=1,N=4,M=2,B="weekly_average")
        with self.assertRaises(ValueError):
            run(day="2022-06-20",hour=6,k=7,N=0,M=2,B="weekly_average")
        with self.assertRaises(ValueError):
            run(day="2022-06-20",hour=6,k=7,N=4,M=0,B="weekly_average")
        with self.assertRaises(ValueError):
            run(day="2022-06-20",hour=6,k=7,N=4,M=2,B="not_a_baseline")
        with self.assertRaises(ValueError):
            run(day="2022-06-20",hour=6,k=7,N=4,M=2,B="weekly_average",
                steps_per_hour=1)


if __name__=="__main__": unittest.main()
