"""Joint and fixed-X0 fits checked against an independent analytic BM1 trajectory."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scipy.optimize import least_squares

import numpy as np

from src.method.bm_1_deterministic_fit import exact_linear_interval
from src.exe.run_model.bm_1.bm_1_known_input_fit import fit_known_inputs, main


def fixture():
    prices = np.random.default_rng(86).uniform(.05, .95, 72)
    parameters = (7.3, 2.6, .67)
    x0 = .23
    x = x0
    demand = []
    for u in prices:
        x, d = exact_linear_interval(x, .43, u, 1., *parameters)
        demand.append(d)
    return np.arange(73.), np.full(73, .43), prices, np.array(demand), parameters, x0, x


class KnownInputFitTests(unittest.TestCase):
    def test_recovers_parameters_from_analytic_demand_with_fixed_x0(self):
        time, baseline, price, demand, truth, x0, final_x = fixture()
        result = fit_known_inputs(time, baseline, price, demand, x0, prior_weight=0, initial=(12., 2., .5))
        self.assertEqual(result['status'], 'converged')
        np.testing.assert_allclose(list(result['parameters'].values()), truth, rtol=2e-5)
        self.assertEqual(result['state'][0], x0)
        self.assertFalse(result['x0_estimated'])
        self.assertAlmostEqual(result['state'][-1], final_x, places=6)
        self.assertLess(result['rmse'], 1e-7)
        self.assertLess(result['rmse'], result['initial_rmse'])
        self.assertEqual(len(result['state']), len(demand)+1)

    def test_recovers_coefficients_and_unknown_x0_together(self):
        time, baseline, price, demand, truth, x0, final_x = fixture()
        result = fit_known_inputs(time, baseline, price, demand, initial_x0=.8, prior_weight=0, initial=(12., 2., .5))
        fixed = fit_known_inputs(time, baseline, price, demand, x0=.5, prior_weight=0, initial=(12., 2., .5))
        self.assertEqual(result['status'], 'converged')
        self.assertTrue(result['x0_estimated'])
        np.testing.assert_allclose(list(result['parameters'].values()), truth, rtol=2e-5)
        self.assertAlmostEqual(result['x0'], x0, places=6)
        self.assertEqual(result['state'][0], result['x0'])
        self.assertAlmostEqual(result['state'][-1], final_x, places=6)
        self.assertLess(result['rmse'], 1e-7)
        self.assertLess(result['rmse'], fixed['rmse']/100)
        self.assertEqual(result['fit_settings']['parameter_names'], ['C_hours', 'k', 'Xi', 'x0'])

    def test_log_penalty_symmetry_and_measured_mse_objective(self):
        time = np.arange(9.)
        baseline = np.full(9, .5)
        price = np.tile([.2, .8], 4)
        demand = np.full(8, .4)
        demand[3] = np.nan
        mask = np.isfinite(demand)
        penalties = []
        for k in (20., .2):
            residual_vectors = []
            def record(fun, x0, **kwargs):
                residual_vectors.append(fun(x0))
                return least_squares(fun, x0, **kwargs)
            with patch('src.exe.run_model.bm_1.bm_1_known_input_fit.least_squares', side_effect=record):
                result = fit_known_inputs(time, baseline, price, demand,
                    initial=(1., k, .5), bounds=((1., .05, 0.), (744., 30., 1.)),
                    steps_per_hour=32, max_nfev=1, observation_mask=mask)
            prior = result['regularization']
            penalties.append(prior['penalty'])
            self.assertEqual(len(residual_vectors[0]), mask.sum()+3)
            self.assertAlmostEqual(float(residual_vectors[0] @ residual_vectors[0]),
                                   result['rmse']**2 + prior['penalty'], places=10)
            self.assertAlmostEqual(prior['data_mse_normalized'], result['rmse']**2)
        self.assertAlmostEqual(penalties[0], .01*np.log(10)**2, places=10)
        self.assertAlmostEqual(penalties[0], penalties[1], places=10)

    def test_fixed_capacity_is_not_penalized_and_prior_pulls_free_parameters(self):
        time, baseline, price, demand, truth, x0, _ = fixture()
        result = fit_known_inputs(time, baseline, price, demand, x0=x0, c=truth[0])
        self.assertEqual(result['status'], 'converged')
        self.assertEqual(result['parameters']['C_hours'], truth[0])
        self.assertFalse(result['c_estimated'])
        self.assertEqual(result['regularization']['penalized_parameters'], ['k', 'Xi'])
        fitted = np.array([result['parameters']['k'], result['parameters']['Xi']])
        center = np.array([2., .5])
        self.assertLess(np.sum(np.log(fitted/center)**2), np.sum(np.log(np.array(truth[1:])/center)**2))
        self.assertAlmostEqual(result['regularization']['penalty'], .01*np.sum(np.log(fitted/center)**2))

    def test_joint_five_piece_fit_disables_all_penalties_and_handles_fixed_parameters(self):
        time, baseline, price, demand, _, x0, _ = fixture()
        for c, fixed_x0 in ((None,None),(5.,x0)):
            result = fit_known_inputs(time,baseline,price,demand,c=c,x0=fixed_x0,
                                      fit_response=True,prior_weight=100,max_nfev=1)
            self.assertTrue(result['response_fitted'])
            self.assertEqual(result['regularization']['weight'],0)
            self.assertEqual(result['regularization']['penalty'],0)
            self.assertEqual(result['regularization']['penalized_parameters'],[])
            self.assertEqual(len(result['fit_settings']['parameter_names']),12-(c is not None)-(fixed_x0 is not None))
            for curve in result['response'].values():
                self.assertEqual(len(curve['knots']),6)
                self.assertTrue(np.all(np.diff(curve['values']) <= 0))
            if c is not None:
                self.assertEqual(result['parameters']['C_hours'],c)
                self.assertEqual(result['x0'],fixed_x0)

    def test_csv_scaling_and_unfinished_fit_are_reported(self):
        time, baseline, price, demand, _, x0, _ = fixture()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder/'inputs.csv'
            with source.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['time_hours', 'baseline', 'price', 'demand'])
                for t, b, u, d in zip(time[:-1], baseline[:-1], price, demand):
                    writer.writerow([t+100, 10+200*b, -50+500*u, 10+200*d])
                writer.writerow([time[-1]+100, 10+200*baseline[-1], '', ''])
            output = folder/'result'
            exit_code = main(['--input', str(source), '--initial-x0', '.8', '--output', str(output),
                '--demand-offset', '10', '--demand-scale', '200', '--price-offset', '-50',
                '--price-scale', '500', '--max-nfev', '1'])
            summary = json.loads((output/'fit.json').read_text())
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary['status'], 'not-converged')
            self.assertTrue(summary['x0_estimated'])
            self.assertEqual(summary['regularization']['center'], {'C_hours': 1., 'k': 2., 'Xi': .5})
            self.assertEqual(summary['regularization']['weight'], .01)
            self.assertAlmostEqual(summary['regularization']['data_mse_normalized'], summary['rmse_normalized']**2)
            self.assertAlmostEqual(summary['rmse'], 200*summary['rmse_normalized'])
            with (output/'trajectory.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]['X']), summary['x0'])
            self.assertAlmostEqual(summary['fit_settings']['initial_x0'], .8)
            self.assertAlmostEqual(float(rows[0]['observed_demand']), 10+200*demand[0])
            self.assertEqual(float(rows[-1]['time_hours']), 172.)
            self.assertTrue(rows[-1]['X'])
            self.assertFalse(rows[-1]['fitted_demand'])


if __name__ == '__main__':
    unittest.main()
