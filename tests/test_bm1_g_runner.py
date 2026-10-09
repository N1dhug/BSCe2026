"""Saved G runner: validate dates/inputs, never fit, freeze before score."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.exe.run_model import bm_1_g as runner
from src.method import g_weekday_export as legacy
from tests.test_g_weekday_bands import packet


class SavedGRunnerTests(unittest.TestCase):
    def fixture(self, origin='2022-03-21'):
        index = pd.date_range(pd.Timestamp(origin).tz_localize('Europe/Copenhagen'),
            (pd.Timestamp(origin)+pd.Timedelta(days=7)).tz_localize('Europe/Copenhagen'), freq='h', inclusive='left').tz_localize(None)
        inputs = pd.DataFrame({'Price':100., 'Temp':5., 'Demand':999.}, index=index)
        snapshot = {'parent':{'origin':origin, 'scales':{'price_min':-500., 'price_scale':6500.}}}
        return snapshot, inputs

    def test_spring_physical_selection_and_no_target_inputs(self):
        snapshot, inputs = self.fixture()
        with patch.object(legacy, 'validate_export', side_effect=lambda s:s):
            _, scenario, selected = runner.prepare(snapshot, inputs, '2022-03-27', 23)
        self.assertEqual(len(scenario), 167); self.assertEqual(len(selected), 23)
        self.assertNotIn('Demand', scenario)
        self.assertNotIn(pd.Timestamp('2022-03-27 02:00'), selected)

    def test_invalid_dates_prices_gaps_and_nonexistent_hours_rejected(self):
        snapshot, inputs = self.fixture()
        with patch.object(legacy, 'validate_export', side_effect=lambda s:s):
            for start, hours in [('2022-03-28', None), ('2022-03-27 02:00', None), (None, 168), (None, 0)]:
                with self.subTest(start=start, hours=hours), self.assertRaises(ValueError):
                    runner.prepare(snapshot, inputs, start, hours)
            invalid = inputs.copy(); invalid.loc[pd.Timestamp('2022-03-27 02:00')] = 1
            with self.assertRaises(ValueError): runner.prepare(snapshot, invalid.sort_index())
            with self.assertRaises(ValueError): runner.prepare(snapshot, inputs.iloc[1:])
            invalid = inputs.copy(); invalid.iloc[0, 0] = 6001
            with self.assertRaises(ValueError): runner.prepare(snapshot, invalid)

    def test_output_horizon_uses_utc_and_preserves_model_state_names(self):
        _, inputs = self.fixture(); result = packet('2022-03-21')['forecast']
        selected = inputs.index[-23:]
        result['predictions']['state'] = .5
        out = runner.select_output(result, inputs, selected)
        self.assertEqual(out['predictions'].forecast_horizon_hours.iloc[-1], 166)
        self.assertEqual(out['manifest']['cutoff'], result['manifest']['cutoff'])
        self.assertIn('state', out['predictions']); self.assertEqual(len(result['predictions']), 167)

    def test_check_never_propagates_and_completed_directory_is_preserved(self):
        snapshot, inputs = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'snapshot.json'; path.write_text(json.dumps(snapshot))
            csv = root/'inputs.csv'; inputs.to_csv(csv, index_label='Datetime')
            out = root/'finished'; out.mkdir(); sentinel = out/'keep'; sentinel.write_text('original')
            args = ['--snapshot', str(path), '--inputs-csv', str(csv), '--output', str(out)]
            with patch.object(legacy, 'validate_export', side_effect=lambda s:s), patch.object(legacy, 'predict_comparison') as predict:
                self.assertEqual(runner.main(args+['--check', '--score', str(root/'missing.csv')]), 0)
                self.assertEqual(runner.main(args), 2); predict.assert_not_called()
            self.assertEqual(sentinel.read_text(), 'original'); self.assertFalse((out/'failure.json').exists())

    def test_prediction_frozen_before_observation_file_opens(self):
        snapshot, inputs = self.fixture(); forecast = packet('2022-03-21')['forecast']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'snapshot.json'; path.write_text(json.dumps(snapshot))
            csv = root/'inputs.csv'; inputs.to_csv(csv, index_label='Datetime')
            target = root/'observations.csv'; inputs[['Demand']].to_csv(target, index_label='Datetime')
            out = root/'run'; read_csv = pd.read_csv
            def read(path, *args, **kwargs):
                if Path(path) == target:
                    freeze = json.loads((out/'forecast_freeze.json').read_text())
                    self.assertFalse(freeze['target_observation_file_opened'])
                    for name, expected in freeze['artifact_sha256'].items():
                        self.assertEqual(runner.digest(out/name), expected)
                return read_csv(path, *args, **kwargs)
            with patch.object(legacy, 'validate_export', side_effect=lambda s:s), \
                 patch.object(legacy, 'predict_comparison', return_value=forecast) as predict, \
                 patch.object(runner, 'plot'), patch.object(runner.pd, 'read_csv', side_effect=read):
                forecast['seconds'] = .001
                self.assertEqual(runner.main(['--snapshot', str(path), '--inputs-csv', str(csv), '--score', str(target), '--output', str(out)]), 0)
                self.assertEqual(predict.call_count, 1)
                self.assertNotIn('Demand', predict.call_args.args[1])
            self.assertTrue((out/'scored.csv').exists())


if __name__ == '__main__': unittest.main()
