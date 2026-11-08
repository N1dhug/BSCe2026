import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from src.exe.run_model.wm_4.plot_stochastic_double_price_step import run, simulate


def configuration(**overrides):
    values = dict(theta_f=-.25, theta_r=-.65, theta_u=-.1875,
                  sigma=.03, tau=.012, baseline=.5,
                  price_start=0., price_down=-1., price_up=1.,
                  first_step=2., second_step=24., duration=52.,
                  steady_tolerance=.005, steps_per_hour=10,
                  paths=3, seed=2026, output=None)
    values.update(overrides)
    return Namespace(**values)


class test_wm4_stochastic_double_price_step(unittest.TestCase):
    def test_double_step_under_damped_and_settled_before_second_step(self):
        result = simulate(configuration())
        state, t = result["state"], result["t"]
        self.assertLess(result["damping_ratio"], 1.)
        self.assertGreater(result["damping_ratio"], 0.)
        self.assertAlmostEqual(result["equilibrium_down"], .75)
        self.assertAlmostEqual(result["equilibrium_up"], -.75)
        before = np.searchsorted(t, 24.)-1
        self.assertLess(abs(state["F"][before]-.75), .005)
        self.assertLess(abs(state["R"][before]), .005)
        self.assertAlmostEqual(state["F"][-1], -.75, delta=.005)
        self.assertEqual(result["samples"].shape, (3, len(t), 2))

    def test_uncertainty_matches_euler_covariance(self):
        args = configuration()
        result = simulate(args)
        dt = result["dt"]
        theta_f, theta_r = args.theta_f, args.theta_r
        transition = np.array([[1., dt], [theta_f*dt, 1.+theta_r*dt]])
        self.assertAlmostEqual(result["sd_d"][0], args.tau)
        self.assertAlmostEqual(result["sd_f"][0], 0.)
        self.assertAlmostEqual(result["sd_r"][0], 0.)
        first_cov = result["covariance"][1]
        expected = args.sigma**2*dt
        self.assertAlmostEqual(first_cov[1, 1], expected)
        self.assertAlmostEqual(result["covariance"][2, 1, 1],
                               expected+(transition[1, 1]**2)*expected)
        np.testing.assert_allclose(result["sd_d"]**2,
                                   result["covariance"][:, 1, 1]+args.tau**2)
        self.assertGreater(result["sd_d"][-1], args.tau)

    def test_zero_process_diffusion_only_observation_uncertainty(self):
        args = configuration(sigma=0., tau=.02, paths=0)
        result = simulate(args)
        np.testing.assert_allclose(result["sd_d"], .02)
        np.testing.assert_allclose(result["sd_f"], 0.)
        np.testing.assert_allclose(result["sd_r"], 0.)
        np.testing.assert_allclose(result["sd_drift"], 0.)
        self.assertEqual(result["samples"].shape[0], 0)

    def test_reject_nonsettled_second_step_and_overdamped_parameters(self):
        with self.assertRaisesRegex(ValueError, "not settled"):
            simulate(configuration(second_step=6.))
        with self.assertRaisesRegex(ValueError, "Underdamped"):
            simulate(configuration(theta_r=-1.25))

    def test_plot_and_csv_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"double_step.png"
            result = run(configuration(output=str(path), paths=0))
            self.assertTrue(path.is_file())
            csv = path.with_suffix(".csv")
            self.assertTrue(csv.is_file())
            data = np.genfromtxt(csv, delimiter=",", names=True)
            self.assertEqual(len(data), len(result["t"]))
            self.assertAlmostEqual(data["sd_demand"][0], .012)
            self.assertGreater(data["sd_demand"][-1], data["sd_demand"][0])


if __name__ == "__main__": unittest.main()
