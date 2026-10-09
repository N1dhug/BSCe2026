"""Output contracts: exact joint moments, no data repair/replay and raw redraw."""

from argparse import Namespace
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.tools.bm_1_outputs import _calendar_days, export_distribution_outputs, summarize_distribution
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

    def test_weekly_redraw_adds_seven_observed_daily_views_without_replay(self):
        from matplotlib.figure import Figure
        values = self.inputs()
        values['times'] = np.arange(168, dtype=float) / 24
        for field in ('joint_mass', 'demand_values'):
            values[field] = np.repeat(values[field][:, :1, :], 168, axis=1)
        values['baseline'] = np.full(168, .5)
        values['price'] = np.tile(np.r_[np.full(12, .25), np.full(12, .75)], 7)
        values['observations'] = 2 + .2 * np.sin(np.arange(168) / 4)
        values['observations'][76] = np.nan
        values['timestamps'] = np.array([(datetime(2022, 2, 7) + timedelta(hours=i)).isoformat()
                                          for i in range(168)])
        views = {}
        original = Figure.savefig
        def capture(fig, path, *args, **kwargs):
            axis = fig.axes[-1] if Path(path).name == 'bm_response.png' else fig.axes[0]
            lines = {line.get_label(): line.get_ydata().copy() for line in axis.lines}
            views[str(path)] = (lines, axis.get_ylim())
            return original(fig, path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / 'first', Path(temp) / 'redraw'
            export_distribution_outputs(source, **values, mode='fixed_origin_forecast',
                metadata=dict(interpretation='one frozen forecast', demand_unit='Normalized demand'))
            with patch('src.model.bm_1.bm_1.fokker_planc_reference_solver', side_effect=AssertionError('FV replay')), \
                 patch('src.model.bm_1.bm_1.deterministic_solver', side_effect=AssertionError('ODE replay')), \
                 patch.object(Figure, 'savefig', capture):
                main(['redraw', '--input', str(source), '--output-dir', str(target)])
            summary = json.loads((target / 'summary.json').read_text())
            self.assertEqual(len(summary['daily_figures']), 7)
            limits = []
            for day, name in enumerate(summary['daily_figures']):
                self.assertTrue((target / name).is_file())
                self.assertIn(f'2022-02-{7+day:02d}', name)
                lines, scale = views[str(target / name)]
                np.testing.assert_array_equal(lines['Observed demand I'], values['observations'][day*24:(day+1)*24])
                np.testing.assert_array_equal(lines['Predicted demand E[D]'], np.full(24, 2.25))
                limits.append(scale)
            self.assertEqual(len(set(limits)), 1)
            np.testing.assert_array_equal(views[str(target / 'bm_response.png')][0]['Observed demand I'],
                                          values['observations'])
            with np.load(source / 'distribution.npz', allow_pickle=False) as a, \
                 np.load(target / 'distribution.npz', allow_pickle=False) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])

    def test_daily_dates_must_match_times_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            for stamps in (['bad'] * 3, ['2022-02-07T00:00', '2022-02-07T01:00', '2022-02-08T00:00']):
                folder = Path(temp) / 'bad'
                with self.assertRaisesRegex(ValueError, 'timestamps'):
                    export_distribution_outputs(folder, **self.inputs(), timestamps=stamps,
                        mode='fixed_origin_forecast', metadata=dict(interpretation='test', demand_unit='Normalized demand'))
                self.assertFalse(folder.exists())

    def test_daily_grouping_preserves_repeated_offset_aware_clock_hour(self):
        stamps = ['2022-10-30T01:00:00+02:00', '2022-10-30T02:00:00+02:00',
                  '2022-10-30T02:00:00+01:00', '2022-10-30T03:00:00+01:00']
        days = _calendar_days(stamps, np.arange(4, dtype=float))
        self.assertEqual(len(days), 1)
        self.assertEqual(days[0][0], '2022-10-30')
        np.testing.assert_array_equal(days[0][1], np.arange(4))
        np.testing.assert_array_equal(days[0][2], [1., 2., 3., 4.])

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
