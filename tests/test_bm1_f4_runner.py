"""Short temporal-boundary and checkpoint checks for the portable F4 entry."""

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.exe.run_model import bm_1_f4 as runner


def fixture():
    index = pd.date_range('2022-03-01', periods=3, freq='h', tz='Europe/Copenhagen')
    candidate = dict(pieces=2, logits=[0., 0.])
    model = dict(timezone='Europe/Copenhagen', measurement_columns=['house_MED'],
        input_sheet='measurements', scales=dict(power_kW=1.), prior=dict(alpha=20., beta=20.),
        physical=dict(C=1., k=2., Xi=1., sigma_x=.5, sigma_y=.16, time_unit='day'),
        candidates=dict(fitted=candidate))
    settings = dict(cells=51, max_step=1/768, frequencies=64, span=2.,
        centers=list(runner.EXTENDED_CENTERS), relative_guard=1e-8, overlap_log_tolerance=1e-7)
    spec = dict(settings=settings, physical=model['physical'],
        input_edges=(np.arange(4)/24.).tolist(), baseline=[.5]*3, price=[.5]*3,
        horizon=[0.,3/24.], time_unit='day', midnight_resets=0)
    manifest = dict(identity='a'*64, origin=index[1].isoformat(),
        end=(index[-1]+pd.Timedelta(hours=1)).isoformat(), missing_filter_hours=0)
    return dict(config=dict(models=['fitted'], evaluate_forecast=True), manifest=manifest,
        model=model, paths=dict(input_workbook=Path('synthetic.xlsx')),
        spec=spec, index=index, filter_hours=1, observed=np.array([True]),
        values=np.array([.5]), baseline=np.full(3,.5), prices=np.full(3,1000.), quantiles=[])


class PortableF4Tests(unittest.TestCase):
    def test_cli_overrides_are_validated_before_any_model_work(self):
        start = pd.Timestamp('2022-02-22T00:00:00+01:00')
        cutoff = start+pd.Timedelta(hours=2)
        measured = pd.DataFrame(np.ones((2,13)), index=pd.date_range(start, cutoff, freq='h', inclusive='left'))
        past = pd.date_range(cutoff-pd.Timedelta(days=7), cutoff, freq='h', inclusive='left')
        prices = pd.Series(1000., index=past)
        output = io.StringIO()
        with patch.object(runner, 'read_raw_power_window', return_value=measured), \
                patch.object(runner, 'read_raw_spot_window', return_value=prices), \
                patch.object(runner, 'run') as run, redirect_stdout(output), redirect_stderr(io.StringIO()) as progress:
            result = runner.main(['--check', '--filter-start', start.isoformat(),
                '--forecast-origin', cutoff.isoformat(), '--forecast-hours', '6',
                '--model', 'linear', '--grid', 'fine'])
        self.assertEqual(result, 0)
        run.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual((plan['filter_hours'], plan['forecast_hours']), (2,6))
        self.assertEqual(plan['models'], ['linear'])
        self.assertEqual(plan['grid'], 'fine')
        self.assertEqual(pd.Timestamp(plan['forecast_origin']), cutoff)
        self.assertEqual(pd.Timestamp(plan['forecast_end_exclusive']), cutoff+pd.Timedelta(hours=6))
        self.assertIn('Working assumption', plan['observation_contract']['unit_status'])
        self.assertIn('validating input/workbook metadata', progress.getvalue())
        self.assertIn('input control complete in', progress.getvalue())
        phases = plan['preparation_phases']
        self.assertEqual([p['phase'] for p in phases],
            ['input_hash_before', 'demand_input', 'price_input', 'input_hash_after', 'source_model_hash'])
        self.assertTrue(all(p['seconds'] >= 0 and p['status'] == 'complete' for p in phases))
        self.assertTrue(all(pd.Timestamp(p['started_at_utc']).utcoffset().total_seconds() == 0 for p in phases))
        self.assertIn('demand_input started', progress.getvalue())
        self.assertIn('source_model_hash complete', progress.getvalue())
        for args, message in [(['--workers', '0'], 'positive integer'),
                              (['--max-seconds', 'nan'], 'finite and at least 15'),
                              (['--forecast-hours', '0'], 'positive integer'),
                              (['--forecast-origin', 'not-a-date'], 'F4 input check failed'),
                              (['--forecast-origin', '2022-10-30T02:00:00'], 'ambiguous local timestamp'),
                              (['--forecast-origin', '2022-02-20T12:00:00+01:00',
                                '--filter-start', '2022-02-20T00:00:00+01:00'], 'curve fit/selection cutoff')]:
            with self.subTest(args=args), patch.object(runner, 'run') as run, \
                    patch.object(runner, 'read_raw_power_window') as read, \
                    redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as stopped:
                    runner.main(['--check', *args])
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn(message, errors.getvalue())
                run.assert_not_called()
                read.assert_not_called()

    def test_cli_interrupted_input_control_does_not_create_a_run(self):
        with patch.object(runner, 'prepare', side_effect=KeyboardInterrupt), \
                patch.object(runner, 'run') as run, redirect_stderr(io.StringIO()) as progress:
            self.assertEqual(runner.main(['--check']), 2)
        self.assertIn('input control stopped after', progress.getvalue())
        self.assertIn('no run created', progress.getvalue())
        run.assert_not_called()

    def test_workbook_preparation_masks_incomplete_group_and_uses_only_past_prices(self):
        config = json.loads((runner.ROOT/'root/f4_quick.json').read_text())
        config['forecast_origin'] = '2022-02-21T02:00:00+01:00'
        start = pd.Timestamp(config['filter_start']); cutoff = pd.Timestamp(config['forecast_origin'])
        past = pd.date_range(cutoff-pd.Timedelta(days=7), cutoff, freq='h', inclusive='left')
        prices = pd.Series(np.arange(len(past))+500., index=past)
        measured = pd.DataFrame(np.ones((2,13)), index=pd.date_range(start, cutoff, freq='h', inclusive='left'))
        measured.iloc[1,3] = np.nan
        with patch.object(runner, 'read_raw_power_window', return_value=measured) as demand_reader, \
                patch.object(runner, 'read_raw_spot_window', return_value=prices) as price_reader:
            data = runner.prepare(config)
        np.testing.assert_array_equal(data['observed'], [True, False])
        self.assertTrue(np.isnan(data['values'][1]))
        self.assertEqual(data['values'][0], 1./data['model']['scales']['power_kW'])
        self.assertEqual(demand_reader.call_args.kwargs['end'], cutoff)
        self.assertEqual(price_reader.call_args.kwargs['end'], cutoff)
        expected = prices.reindex(data['index'][2:]-pd.Timedelta(days=7)).to_numpy()
        np.testing.assert_array_equal(data['prices'][2:], expected)
        config['forecast_origin'] = '2022-02-20T12:00:00+01:00'
        config['filter_start'] = '2022-02-20T00:00:00+01:00'
        with self.assertRaisesRegex(ValueError, 'curve fit/selection cutoff'):
            runner.prepare(config)

    def run_case(self, data, directory, evaluation, *, resume=False):
        # Reuse the actual native probability/likelihood kernels, with a small
        # synthetic grid. Only process scheduling and workbook boundary are mocked.
        def direct_mapper(pool, jobs):
            pool.check()
            return list(map(runner.advance_frequency_block, jobs))

        def read_targets(*args, **kwargs):
            saved = runner._read_array_checkpoint((directory/'checkpoint.npz').read_bytes())
            self.assertTrue(all(len(s['rows']) == 3 for s in saved['streams'].values()))
            return pd.DataFrame({'house_MED': evaluation}, index=data['index'][1:])

        with patch.object(runner.FrequencyPool, 'mapper', direct_mapper), \
                patch.object(runner, 'read_raw_power_window', side_effect=read_targets):
            return runner.run(deepcopy(data), directory=directory, resume=resume,
                              max_seconds=60., workers=1)

    def test_future_targets_do_not_change_forecasts_or_states(self):
        data = fixture()
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as temporary:
            a, b = Path(temporary)/'a', Path(temporary)/'b'
            self.assertEqual(self.run_case(data, a, [.1,.2]), 'complete')
            self.assertEqual(self.run_case(data, b, [.8,.9]), 'complete')
            sa = runner._read_array_checkpoint((a/'checkpoint.npz').read_bytes())['streams']['fitted']
            sb = runner._read_array_checkpoint((b/'checkpoint.npz').read_bytes())['streams']['fitted']
            np.testing.assert_array_equal(sa['log_state'], sb['log_state'])
            np.testing.assert_array_equal(sa['origin_log_state'], sb['origin_log_state'])
            for ra, rb in zip(sa['rows'][1:], sb['rows'][1:]):
                self.assertFalse(ra['assimilated'])
                self.assertIsNone(ra['log_evidence'])
                self.assertEqual(ra['prediction_kW'], rb['prediction_kW'])
                self.assertEqual(ra['state_mean'], rb['state_mean'])
            self.assertNotEqual(sa['rows'][1]['observed_kWh'], sb['rows'][1]['observed_kWh'])
            row = sa['rows'][1]
            self.assertEqual(pd.Timestamp(row['interval_end'])-pd.Timestamp(row['timestamp']), pd.Timedelta(hours=1))
            self.assertEqual(pd.Timestamp(row['timestamp_utc']), pd.Timestamp(row['timestamp']))
            self.assertEqual(pd.Timestamp(row['interval_end_utc']), pd.Timestamp(row['interval_end']))
            summary = json.loads((a/'summary.json').read_text())
            phases = summary['timing']['last_invocation_phases']
            self.assertEqual([p['phase'] for p in phases],
                ['filtering', 'forecast', 'forecast', 'evaluation_input', 'worker_shutdown', 'export'])
            self.assertTrue(all(p['seconds'] >= 0 and p['status'] == 'complete' for p in phases))
            receipts = runner._read_array_checkpoint((a/'checkpoint.npz').read_bytes())['invocation_timings']
            self.assertEqual(receipts[-1]['phases'], phases)
            forecast = sa['rows'][1:]
            error = np.array([r['prediction_kW']-r['observed_kWh'] for r in forecast])
            baseline_error = np.array([r['baseline_kW']-r['observed_kWh'] for r in forecast])
            metrics = summary['metrics']['fitted']['forecast']
            self.assertAlmostEqual(metrics['mse_kW2'], np.mean(error**2))
            self.assertAlmostEqual(metrics['bias_kW'], np.mean(error))
            self.assertAlmostEqual(metrics['baseline_mse_kW2'], np.mean(baseline_error**2))
            self.assertAlmostEqual(metrics['baseline_mae_kW'], np.mean(abs(baseline_error)))
            self.assertAlmostEqual(metrics['baseline_bias_kW'], np.mean(baseline_error))

    def test_pause_resume_retains_exact_origin_and_matches_uninterrupted_run(self):
        data = fixture()
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as temporary:
            paused, full = Path(temporary)/'paused', Path(temporary)/'full'
            original = runner._checkpoint

            def stop_after_filter(directory, saved):
                original(directory, saved)
                if len(saved['streams']['fitted']['rows']) == 1:
                    (directory/'STOP').touch()

            with patch.object(runner, '_checkpoint', side_effect=stop_after_filter):
                self.assertEqual(self.run_case(data, paused, [.6,.7]), 'paused')
            (paused/'STOP').unlink()
            self.assertEqual(self.run_case(data, paused, [.6,.7], resume=True), 'complete')
            self.assertEqual(self.run_case(data, full, [.6,.7]), 'complete')
            a = runner._read_array_checkpoint((paused/'checkpoint.npz').read_bytes())['streams']['fitted']
            b = runner._read_array_checkpoint((full/'checkpoint.npz').read_bytes())['streams']['fitted']
            self.assertEqual(a['rows'], b['rows'])
            np.testing.assert_array_equal(a['log_state'], b['log_state'])
            timing = json.loads((paused/'summary.json').read_text())['timing']
            receipt = runner._read_array_checkpoint((paused/'checkpoint.npz').read_bytes())['invocation_timings']
            self.assertEqual(timing['recorded_invocations'], 2)
            self.assertAlmostEqual(timing['cumulative_recorded_seconds'], sum(t['seconds'] for t in receipt))
            self.assertGreater(timing['cumulative_recorded_seconds'], timing['last_invocation_seconds'])
            self.assertFalse(timing['includes_prepare_all_invocations'])
            origin = runner.load_origin(paused)
            np.testing.assert_array_equal(origin.posterior_mass, np.exp(a['origin_log_state']))
            self.assertEqual(origin.time, 1/24.)
            changed = deepcopy(data); changed['manifest']['identity'] = 'b'*64
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.run_case(changed, paused, [.6,.7], resume=True)
            preserved = {path.name:(path.read_bytes(), path.stat().st_mtime_ns)
                         for path in paused.iterdir() if path.is_file()}
            with patch.object(runner, '_export') as export, patch.object(runner, '_checkpoint') as checkpoint:
                self.assertEqual(self.run_case(data, paused, [.6,.7], resume=True), 'complete')
                export.assert_not_called()
                checkpoint.assert_not_called()
            self.assertEqual(preserved, {path.name:(path.read_bytes(), path.stat().st_mtime_ns)
                                       for path in paused.iterdir() if path.is_file()})
            broken_summary = json.loads((paused/'summary.json').read_text())
            broken_summary['completed_hours']['fitted'] = 2
            (paused/'summary.json').write_text(json.dumps(broken_summary))
            with self.assertRaisesRegex(ValueError, 'Completed summary/checkpoint disagree'):
                self.run_case(data, paused, [.6,.7], resume=True)

    def test_missing_measurement_advances_state_without_assimilation(self):
        data = fixture(); data['observed'][0] = False; data['values'][0] = np.nan
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as temporary:
            directory = Path(temporary)/'missing'
            self.assertEqual(self.run_case(data, directory, [.6,.7]), 'complete')
            saved = runner._read_array_checkpoint((directory/'checkpoint.npz').read_bytes())
            first = saved['streams']['fitted']['rows'][0]
            self.assertFalse(first['assimilated'])
            self.assertIsNone(first['log_evidence'])
            self.assertIsNone(first['observed_kWh'])
            self.assertIsNotNone(saved['streams']['fitted']['origin_log_state'])


if __name__ == '__main__':
    unittest.main()
