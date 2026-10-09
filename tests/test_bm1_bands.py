"""Generic band dates retain old math, identities and observation isolation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.method import bm_1_bands as bands
from src.exe.run_model.bm_1_g import bands_main, write_forecast
from tests.test_g_weekday_bands import packet as old_packet


def packet(origin, error=2.):
    result = old_packet(origin, error)
    result['forecast']['manifest'].update(population='synthetic', demand_unit='synthetic units',
        timezone='Europe/Copenhagen', scenario='declared synthetic inputs', baseline_policy='weekday', fit_policy='synthetic')
    return result


class DeclaredDateBandsTests(unittest.TestCase):
    def calibration(self):
        return bands.calibrate([packet('2022-05-02'), packet('2022-05-09')], ['2022-05-16', '2022-05-23'])['snapshot']

    def test_new_dates_exact_rank_and_unchanged_means_variances(self):
        record = self.calibration(); item = record['models'][bands.MODELS[0]]
        self.assertEqual(item['samples'], 336); self.assertEqual(item['rank'], 321)
        before = packet('2022-05-16')['forecast']; after = bands.apply(before, record)
        for model in bands.MODELS:
            for suffix in ('_output_mean', '_output_variance', '_lower', '_upper'):
                np.testing.assert_array_equal(before['predictions'][model+suffix], after['predictions'][model+suffix])
            np.testing.assert_allclose(after['predictions'][model+'_calibrated_lower'], 8.)

    def test_chronology_counts_contract_and_source_tampering_fail_closed(self):
        for packets, evaluation in [([packet('2022-05-09'), packet('2022-05-02')], ['2022-05-16']),
                                    ([packet('2022-05-02')], ['2022-05-16']),
                                    ([packet('2022-05-02'), packet('2022-05-09')], ['2022-05-09'])]:
            with self.assertRaises(ValueError): bands.calibrate(packets, evaluation)
        record = self.calibration()
        for key in ('source_sha256', 'freeze_cutoff'):
            changed = deepcopy(record); changed[key] = {} if key == 'source_sha256' else '2022-05-09'
            with self.assertRaises(ValueError): bands.validate(changed)
        changed = packet('2022-05-16')['forecast']; changed['manifest']['population'] = 'different'
        with self.assertRaises(ValueError): bands.apply(changed, record)
        with self.assertRaises(ValueError): bands.apply(packet('2022-05-30')['forecast'], record)

    def write_packet(self, directory, origin, shared=False):
        directory.mkdir(); forecast = packet(origin)['forecast']; write_forecast(directory, forecast)
        receipt = {'status':'frozen-input-only', 'origin':origin+' 00:00:00' if shared else origin,
            'scoring_targets_opened' if shared else 'target_observation_file_opened':False,
            'artifact_sha256':{name:bands.digest(directory/name) for name in ('forecast.json', 'forecast.csv', 'forecast_normalized.csv')}}
        (directory/'forecast_freeze.json').write_text(json.dumps(receipt))
        return forecast

    def write_calibration(self, directory):
        directory.mkdir()
        result = bands.calibrate([packet('2022-05-02'), packet('2022-05-09')], ['2022-05-16', '2022-05-23'])
        path = directory/'calibration_snapshot.json'; path.write_text(json.dumps(result['snapshot']))
        result['scores'].to_csv(directory/'calibration_scores.csv')
        receipt = {'evaluation_observations_used':False,
            'artifact_sha256':{name:bands.digest(directory/name) for name in ('calibration_snapshot.json', 'calibration_scores.csv')}}
        (directory/'calibration_freeze.json').write_text(json.dumps(receipt))
        return path

    def test_old_and_shared_receipts_verified_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for shared in (False, True):
                directory = Path(tmp)/str(shared); self.write_packet(directory, '2022-05-16', shared)
                self.assertEqual(len(bands.load_forecast(directory)['predictions']), 168)
                with (directory/'forecast.csv').open('a') as file: file.write('\n')
                with self.assertRaises(ValueError): bands.load_forecast(directory)

    def test_observation_calendar_rejects_finite_nonexistent_hour_duplicates_and_infinity(self):
        source = packet('2022-03-21'); y = source['observations']; forecast = source['forecast']
        self.assertEqual(len(bands.observations_for_forecast(y, forecast)), 167)
        invalid = y.copy(); invalid.loc[pd.Timestamp('2022-03-27 02:00')] = 3.
        with self.assertRaises(ValueError): bands.observations_for_forecast(invalid.sort_index(), forecast)
        with self.assertRaises(ValueError): bands.observations_for_forecast(pd.concat([y, y.iloc[-1:]]), forecast)
        invalid = y.copy(); invalid.iloc[0] = np.inf
        with self.assertRaises(ValueError): bands.observations_for_forecast(invalid, forecast)
        missing = y.copy(); missing.iloc[0] = np.nan
        self.assertTrue(pd.isna(bands.observations_for_forecast(missing, forecast).iloc[0]))

    def test_all_evaluation_forecasts_frozen_before_any_target_is_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); calibration = self.write_calibration(root/'calibration')
            forecasts = []; observations = []
            for origin in ('2022-05-16', '2022-05-23'):
                directory = root/origin; self.write_packet(directory, origin); forecasts.append(str(directory))
                path = root/(origin+'.csv'); packet(origin)['observations'].rename('Demand').to_csv(path, index_label='Datetime'); observations.append(str(path))
            output = root/'evaluation'; reader = pd.read_csv
            def read(path, *args, **kwargs):
                if str(path) in observations:
                    receipt = json.loads((output/'evaluation_forecasts_freeze.json').read_text())
                    self.assertFalse(receipt['target_observation_files_opened'])
                    self.assertEqual(len(receipt['forecasts']), 2)
                return reader(path, *args, **kwargs)
            with patch.object(bands.pd, 'read_csv', side_effect=read):
                self.assertEqual(bands_main(['evaluate', '--calibration', str(calibration), '--forecasts', *forecasts,
                    '--observations', *observations, '--output', str(output)]), 0)
            report = json.loads((output/'evaluation_report.json').read_text())
            self.assertEqual(report['scored_samples'], 336)

    def test_invalid_calibration_chronology_opens_no_observation_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); forecasts = []; observations = []
            for origin in ('2022-05-02', '2022-05-09'):
                directory = root/origin; self.write_packet(directory, origin); forecasts.append(str(directory))
                path = root/(origin+'.csv'); packet(origin)['observations'].rename('Demand').to_csv(path, index_label='Datetime'); observations.append(str(path))
            output = root/'invalid'; reader = pd.read_csv; target_reads = []
            def read(path, *args, **kwargs):
                if str(path) in observations:
                    target_reads.append(str(path))
                    self.fail('Targets were opened before chronology preflight rejected the run.')
                return reader(path, *args, **kwargs)
            with patch.object(bands.pd, 'read_csv', side_effect=read):
                result = bands_main(['calibrate', '--forecasts', *forecasts, '--observations', *observations,
                    '--evaluation-origins', '2022-05-09', '--output', str(output)])
            self.assertEqual(result, 2); self.assertEqual(target_reads, [])
            self.assertFalse(output.exists())

    def test_calibration_receipt_rejects_self_consistent_edits_and_modified_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_calibration(Path(tmp)/'calibration')
            original = path.read_text(); record = bands.load_calibration(path)
            edited = deepcopy(record)
            for model in bands.MODELS:
                edited['models'][model]['scores'] = [2*score for score in edited['models'][model]['scores']]
                edited['models'][model]['q'] *= 2
            bands.validate(edited)  # Mathematically valid does not mean the frozen artifact is unchanged.
            path.write_text(json.dumps(edited))
            with self.assertRaises(ValueError): bands.load_calibration(path)
            path.write_text(original)
            with (path.parent/'calibration_scores.csv').open('a') as handle: handle.write('\n')
            with self.assertRaises(ValueError): bands.load_calibration(path)


if __name__ == '__main__': unittest.main()
