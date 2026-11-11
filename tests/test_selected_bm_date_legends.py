"""Validate per-sample parameter legends for the WM4/WM6 August 14-day plots."""
import unittest

import numpy as np

from src.exe.run_model.wm_4.selected_bm_dates_arm import sample_parameter_legend as wm4_legend
from src.exe.run_model.wm_6.selected_bm_dates_arm import sample_parameter_legend as wm6_legend


class test_selected_bm_date_legends(unittest.TestCase):
    def test_wm4_legend_includes_day_specific_fit(self):
        row={"theta":np.array([-.123456,-.045678,-1.23456]),"tau2":.025**2,"sigma2":.125**2}
        label=wm4_legend(row)
        for part in (r"\theta_F=-0.1235",r"\theta_U=-0.04568",
                     r"\theta_R=-1.235",r"\tau=0.025",r"\sigma=0.125"):
            self.assertIn(part,label)
        self.assertNotIn(r"\theta_B",label)

    def test_wm6_legend_includes_baseline_coefficient(self):
        row={"theta":np.array([-.8,-.1,-1.4,.15]),"tau2":.03**2,"sigma2":.04**2}
        label=wm6_legend(row)
        for part in (r"\theta_F=-0.8",r"\theta_U=-0.1",
                     r"\theta_R=-1.4",r"\theta_B=+0.15",
                     r"\tau=0.03",r"\sigma=0.04"):
            self.assertIn(part,label)

    def test_incorrect_number_of_coefficients_is_rejected(self):
        row={"theta":np.array([-.8,-.1,-1.4]),"tau2":0.,"sigma2":0.}
        with self.assertRaises(ValueError): wm6_legend(row)
        with self.assertRaises(ValueError): wm4_legend({**row,"theta":np.r_[row["theta"],.2]})


if __name__=="__main__": unittest.main()
