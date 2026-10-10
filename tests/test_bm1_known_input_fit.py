"""Known-X0 fit checked against an independent analytic BM1 trajectory."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.method.bm_1_deterministic_fit import exact_linear_interval
from src.method.bm_1_known_input_fit import fit_known_inputs, main


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
        result = fit_known_inputs(time, baseline, price, demand, x0)
        self.assertEqual(result['status'], 'converged')
        np.testing.assert_allclose(list(result['parameters'].values()), truth, rtol=2e-5)
        self.assertEqual(result['state'][0], x0)
        self.assertAlmostEqual(result['state'][-1], final_x, places=6)
        self.assertLess(result['rmse'], 1e-7)
        self.assertLess(result['rmse'], result['initial_rmse'])
        self.assertEqual(len(result['state']), len(demand)+1)

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
            exit_code = main(['--input', str(source), '--x0', str(x0), '--output', str(output),
                '--demand-offset', '10', '--demand-scale', '200', '--price-offset', '-50',
                '--price-scale', '500', '--max-nfev', '1'])
            summary = json.loads((output/'fit.json').read_text())
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary['status'], 'not-converged')
            self.assertAlmostEqual(summary['rmse'], 200*summary['rmse_normalized'])
            with (output/'trajectory.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]['X']), x0)
            self.assertAlmostEqual(float(rows[0]['observed_demand']), 10+200*demand[0])
            self.assertEqual(float(rows[-1]['time_hours']), 172.)
            self.assertTrue(rows[-1]['X'])
            self.assertFalse(rows[-1]['fitted_demand'])


if __name__ == '__main__':
    unittest.main()
