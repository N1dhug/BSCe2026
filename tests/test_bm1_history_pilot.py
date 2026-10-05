"""Real small FV H2 pilots on synthetic XLSX, including crash/resume parity."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_bm1_history_io as fixtures
from src.exe.run_model.bm_1 import curves
from src.exe.run_model import bm_1_history as history
from src.tools.bm_1_reports import _atomic_report, _read_likelihood_report


class HistoryPilotTests(unittest.TestCase):
    setUp = fixtures.HistoryWorkbookTests.setUp
    write_power = fixtures.HistoryWorkbookTests.write_power
    write_prices = fixtures.HistoryWorkbookTests.write_prices
    contract = fixtures.HistoryWorkbookTests.contract

    def config(self):
        config = history.configuration_template()
        config.update(evidence={k: 'Synthetic fixture only' for k in history.EVIDENCE},
            timezone='Europe/Copenhagen', price_area='DK1', origin=self.origin.isoformat(),
            baseline_fitted_before=self.origin.isoformat(),
            model_selection_protocol='Frozen candidates will be compared on MODEL_SELECTION; fixture only.')
        config['contract'] = self.contract().to_record()
        for key in ('schema', 'aggregation', 'interpretation'):
            config['contract'].pop(key)
        config['contract']['split'].pop('schema')
        config['contract']['split']['holdout_audit_reference'] = 'Synthetic fixture audit'
        self.baseline = self.root/'baseline.csv'
        self.baseline.write_text('timestamp,baseline\n'+''.join(
            f'{t.isoformat()},5\n' for t in [self.origin, self.start, self.start+(self.end-self.start)/2]), encoding='utf-8')
        config['files'] = dict(input_workbook=str(self.input), price_workbook=str(self.prices), baseline_csv=str(self.baseline))
        config['parameters'] = dict(C=1., k=2., Xi=.75, sigma_x=.2, sigma_y=.03, time_unit='day')
        config['prior'] = dict(alpha=2., beta=2.)
        config['numerics'].update(cells=17, max_step=1/96)
        for family in config['fit']['families']:
            family['max_evaluations'] = 4
        return config

    def run_config(self, config, directory, **kwargs):
        path = self.root/'config.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        args = SimpleNamespace(config=path, input_root=self.root, run_dir=directory,
            max_seconds=60., limit_candidates=None, prepare_only=False, resume=False, quiet=True)
        for key, value in kwargs.items():
            setattr(args, key, value)
        with patch.object(curves, 'source_snapshot', return_value=dict(commit='synthetic-fixture', dirty=False, sha256={})):
            return history.run_pilot(args)

    def test_incomplete_config_rejected_before_any_observation_access(self):
        with patch.object(history, 'prepare_training', side_effect=AssertionError('read observations')):
            with self.assertRaisesRegex(ValueError, 'evidence.population'):
                self.run_config(history.configuration_template(), self.root/'blocked')
        self.assertFalse((self.root/'blocked').exists())

    def test_prior_baseline_dates_and_interval_semantics_are_explicit(self):
        config = self.config()
        invalid = deepcopy(config)
        invalid['baseline_fitted_before'] = self.start.isoformat()
        with self.assertRaisesRegex(ValueError, 'baseline fitted'):
            history.validate_configuration(invalid)
        invalid = deepcopy(config)
        invalid['contract']['observation_kind'] = 'interval'
        with self.assertRaises(NotImplementedError):
            history.validate_configuration(invalid)

    def test_prepare_only_keeps_complete_train_and_reads_no_holdout_measurements(self):
        config = self.config()
        with patch.object(history, 'pooled_curve_log_likelihood', side_effect=AssertionError('fitting')):
            result = self.run_config(config, self.root/'prepared', prepare_only=True)
        self.assertFalse(result['fitting_run'])
        manifest = json.loads((self.root/'prepared'/'manifest.json').read_text())
        self.assertEqual(manifest['training']['prepared']['training_rows'], 2)
        self.assertEqual(manifest['training']['prepared']['scored_observations'], 1)
        self.assertFalse(manifest['training']['prepared']['initial_observation_excluded'])

    def test_real_candidate_budget_resume_matches_uninterrupted_fit(self):
        config = self.config()
        directory = self.root/'split'
        paused = self.run_config(config, directory, limit_candidates=3)
        self.assertTrue(paused['paused'])
        self.assertEqual(paused['new_candidate_evaluations'], 3)
        resumed = self.run_config(config, directory, resume=True)
        whole = self.run_config(config, self.root/'whole')
        self.assertTrue(resumed['completed'])
        self.assertEqual(len(resumed['families']), 2)
        for left, right in zip(resumed['families'], whole['families']):
            self.assertEqual(left['trace'], right['trace'])
            self.assertEqual(left['f'], right['f'])
            self.assertEqual(left['g'], right['g'])
            self.assertEqual(left['replay'], right['replay'])
            self.assertTrue(left['budget_exhausted'])
            self.assertFalse(left['scientific_acceptance'])
        with (patch.object(history, 'pooled_curve_log_likelihood', side_effect=AssertionError('reference replay')),
              patch.object(history, 'fit_shared_curve_pair', side_effect=AssertionError('new fit'))):
            cached = self.run_config(config, directory, resume=True)
        self.assertEqual(cached['new_candidate_evaluations'], 0)

    def test_resume_rejects_modified_input_and_corrupted_trace(self):
        config = self.config()
        directory = self.root/'split'
        self.run_config(config, directory, limit_candidates=3)
        self.baseline.write_text(self.baseline.read_text()+'2022-12-01T00:00:00+01:00,bad\n')
        with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
            self.run_config(config, directory, resume=True)
        config = self.config()
        trace_path = directory/'2_trace.json.gz'
        trace = _read_likelihood_report(trace_path)[0]
        trace[0]['objective'] += 1.
        _atomic_report(trace_path, trace, replace_existing=True)
        with self.assertRaisesRegex(ValueError, 'objective is inconsistent'):
            self.run_config(config, directory, resume=True)

    def test_baseline_values_outside_train_are_not_decoded(self):
        config = self.config()
        self.baseline.write_text(self.baseline.read_text()+f'{self.end.isoformat()},poisoned excluded baseline\n')
        record, metadata, _, _, _ = history.prepare_training(config, root=self.root)
        np.testing.assert_array_equal(record.observations.observed, [True, False])
        self.assertEqual(metadata['prepared']['midnight_resets'], 0)

    def test_completed_resume_rejects_corrupted_saved_replay(self):
        config = self.config()
        directory = self.root/'finished'
        self.run_config(config, directory)
        path = directory/'2_result.json.gz'
        saved = _read_likelihood_report(path)[0]
        saved['replay']['records'][0]['log_evidence'][0] += 1.
        _atomic_report(path, saved, replace_existing=True)
        with self.assertRaisesRegex(ValueError, 'likelihood differs'):
            self.run_config(config, directory, resume=True)

    def test_stop_file_pauses_without_likelihood_work_and_config_cli_has_no_observation_io(self):
        output = self.root/'template.json'
        with patch.object(history, 'prepare_training', side_effect=AssertionError('observations')):
            self.assertEqual(curves.main(['history-config', '--output', str(output)]), 0)
        self.assertIsNone(json.loads(output.read_text())['contract']['observation_kind'])
        config = self.config()
        directory = self.root/'stopped'
        self.run_config(config, directory, prepare_only=True)
        (directory/'STOP').touch()
        with patch.object(history, 'pooled_curve_log_likelihood', side_effect=AssertionError('likelihood')):
            result = self.run_config(config, directory, resume=True)
        self.assertTrue(result['paused'])
        self.assertEqual(result['new_candidate_evaluations'], 0)
