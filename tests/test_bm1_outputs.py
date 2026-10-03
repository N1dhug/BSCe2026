"""Output contracts: exact joint moments, no data repair/replay and raw redraw."""

from argparse import Namespace
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.tools.bm_1_outputs import export_distribution_outputs, summarize_distribution
from src.exe.run_model.bm_1.outputs import main, run_i_report


class BMOutputTests(unittest.TestCase):
    def inputs(self):
        # Nonuniform cells and parameter/state dependence. Hypothesis 0 has
        # weight .25 at X=.1, hypothesis 1 has weight .75 at X=.6.
        return dict(times=np.array([0., .5, 1.]), x_edges=np.array([0., .2, 1.]),
            joint_mass=np.array([[[.25, 0.]] * 3, [[0., .75]] * 3]),
            demand_values=np.array([[[0., 1.]] * 3, [[2., 3.]] * 3]),
            baseline=np.array([.5] * 3), price=np.array([.25, .75, .75]))

    def test_joint_moments_preserve_parameter_state_dependence(self):
        values = self.inputs()
        result = summarize_distribution(**values, observations=[2., np.nan, 3.])
        np.testing.assert_allclose(result['X']['mean'], .475)
        np.testing.assert_allclose(result['D']['mean'], 2.25)
        np.testing.assert_allclose(result['D']['std'] ** 2, 1.6875)
        np.testing.assert_array_equal(result['D']['p05'], [0.] * 3)
        np.testing.assert_array_equal(result['D']['p95'], [3.] * 3)
        self.assertEqual(result['metrics']['BM']['n'], 2)
        self.assertAlmostEqual(result['metrics']['BM']['bias'], -.25)
        self.assertAlmostEqual(result['metrics']['BM']['MSE'], .3125)
        # Input weights must not be normalized, updated or mutated by export.
        np.testing.assert_array_equal(result['joint_mass'], values['joint_mass'])

    def test_invalid_laws_and_alignment_are_rejected_without_repair(self):
        for field, bad in [('joint_mass', np.ones((2, 3, 2))),
                           ('joint_mass', np.array([[[-.1, 1.1]] * 3])),
                           ('demand_values', np.ones((3, 2))),
                           ('times', [0., .5, .5]), ('price', [.2, .2, 1.1])]:
            with self.subTest(field=field, bad=np.shape(bad)):
                values = self.inputs()
                values[field] = bad
                with self.assertRaises(ValueError):
                    summarize_distribution(**values)
        with self.assertRaises(ValueError):
            summarize_distribution(**self.inputs(), observations=[0., 1.])

    def test_nonuniform_density_export_and_redraw_need_no_solver(self):
        import matplotlib.pyplot as plt
        captured = []
        from matplotlib.axes import Axes
        original = Axes.pcolormesh
        def capture(axis, *args, **kwargs):
            if len(args) == 3:
                captured.append(np.array(args[2]))
            return original(axis, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'first'
            with patch.object(Axes, 'pcolormesh', capture):
                export_distribution_outputs(folder, **self.inputs(), mode='fixed_origin_forecast',
                    metadata=dict(interpretation='test joint forecast', demand_unit='Normalized demand'),
                    observations=[2., np.nan, 3.])
            np.testing.assert_allclose(captured[0], [[1.25] * 3, [.9375] * 3])
            summary = json.loads((folder / 'summary.json').read_text())
            for name in summary['files']:
                self.assertGreater((folder / name).stat().st_size, 0)
            self.assertIn('observation noise', summary['uncertainty'])
            with (folder / 'moments.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[1]['elapsed_hours']), 12.)
            with patch('src.model.bm_1.bm_1.fokker_planc_reference_solver', side_effect=AssertionError('FV replay')), \
                 patch('src.model.bm_1.bm_1.deterministic_solver', side_effect=AssertionError('ODE replay')):
                main(['redraw', '--input', str(folder), '--output-dir', str(Path(temp) / 'redraw')])
            with np.load(folder / 'distribution.npz', allow_pickle=False) as a, \
                 np.load(Path(temp) / 'redraw/distribution.npz', allow_pickle=False) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            with self.assertRaises(FileExistsError):
                export_distribution_outputs(folder, **self.inputs(), mode='reference_simulation',
                    metadata=dict(interpretation='test', demand_unit='Normalized demand'))
            self.assertEqual(plt.get_fignums(), [])

    def test_counterfactual_is_not_scored_against_observed_tariff_data(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'bad'
            with self.assertRaisesRegex(ValueError, 'counterfactual'):
                export_distribution_outputs(folder, **self.inputs(), mode='counterfactual_control',
                    metadata=dict(interpretation='test', demand_unit='Normalized demand'),
                    observations=[1., 1., 1.])
            self.assertFalse(folder.exists())

    def test_sparse_slices_preserve_full_declared_price_plan_and_origin(self):
        from matplotlib.axes import Axes
        original = Axes.step
        steps = []
        def capture(axis, x, y, *args, **kwargs):
            steps.append((np.asarray(x), np.asarray(y)))
            return original(axis, x, y, *args, **kwargs)
        values = self.inputs()
        values['times'] = np.array([1., 2., 4.])
        values['price'] = np.array([.25, .75, .25])
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(Axes, 'step', capture):
                export_distribution_outputs(Path(temp) / 'sparse', **values, time_origin=0., time_unit='hour',
                    price_edges=[0., 1.5, 3., 5.], price_levels=[.25, .75, .25],
                    mode='fixed_origin_forecast', metadata=dict(interpretation='sparse test', demand_unit='Normalized demand'))
            np.testing.assert_array_equal(steps[0][0], [0., 1.5, 3., 5.])
            np.testing.assert_array_equal(steps[0][1], [.25, .75, .25, .25])
            summary = json.loads((Path(temp) / 'sparse/summary.json').read_text())
            self.assertEqual(summary['time_origin'], 0.)
            with self.assertRaisesRegex(ValueError, 'disagree'):
                export_distribution_outputs(Path(temp) / 'wrong', **values,
                    price_edges=[0., 5.], price_levels=[.25],
                    mode='fixed_origin_forecast', metadata=dict(interpretation='test', demand_unit='Normalized demand'))

    def test_i_report_keeps_exact_stored_forecast_times_without_replay(self):
        inputs = self.inputs()
        report = dict(stage='I-baseline-development-v1', interpretation='integration only',
            forecast=dict(rho_map=dict(times=[1.25, 2., 8.], x_edges=[0., .2, 1.],
                cell_mass=inputs['joint_mass'].sum(axis=0).tolist(), time_unit='day')),
            origins=dict(rho_map=dict(time=1., parameters=dict(state_response='1-2*x',
                price_response='1-2*u', k=2., Xi=1.))),
            future_inputs=dict(baseline=.5, price_edges=[1., 8.], prices=[.5]))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'report.json'
            path.write_text(json.dumps(report))
            with patch('src.model.bm_1.bm_1.fokker_planc_reference_solver', side_effect=AssertionError('FV replay')):
                values = run_i_report(Namespace(input=path, belief='rho_map'))
            np.testing.assert_array_equal(values['times'], [1.25, 2., 8.])
            self.assertFalse(values['metadata']['forecast_scientific_acceptance'])


if __name__ == '__main__':
    unittest.main()
