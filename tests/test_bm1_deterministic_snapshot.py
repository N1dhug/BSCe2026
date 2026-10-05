"""Small G3 round-trip, causal scoring and durable-export checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.deterministic import _jsonable, main, write_result
from src.method.bm_1_deterministic_benchmark import (run_benchmark, forecast_from_snapshot,
                                                   score_forecast)
from src.params.bm_1_deterministic import BM1DeterministicConfig
from src.tools.bm_1_deterministic_snapshot import (export_fit_snapshot, load_fit_snapshot,
                                                 migrate_legacy_fit, validate_snapshot,
                                                 source_hashes, LEGACY_SOURCES, ROOT)


class FrozenForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = pd.date_range('2022-01-01', periods=360, freq='h')
        t = np.arange(len(index))
        cls.raw = pd.DataFrame({'Demand':2+.3*np.sin(t/5)+.01*np.cos(t/2),
                                'Price':100+30*np.sin(t/4),'Temp':5+np.cos(t/8)}, index=index)
        config = BM1DeterministicConfig(starts=((12.,2.,0.,.5),), fixed_x0=.5,
                                        max_calls=150, steps_per_hour=12)
        # A declared nonzero-response fixture exercises origin continuation;
        # the existing fit suite separately tests the actual LS optimizer.
        fitted = {'status':'converged', 'fixed_x0':.5,
                  'best_converged':{'parameters':[4.,2.,.6,.5]}}
        with patch('src.method.bm_1_deterministic_benchmark.fit_deterministic',return_value=fitted):
            cls.result = run_benchmark(cls.raw, index[0], index[192], forecast_hours=24,
                                       config=config, wm_steps_per_hour=12, price_bounds=(60.,140.))
        if cls.result['status'] != 'complete':
            raise AssertionError(cls.result.get('wm_failure', cls.result['status']))
        cls.snapshot = _jsonable(export_fit_snapshot(cls.result))

    def write_snapshot(self, directory):
        path = Path(directory)/'fit_snapshot.json'
        path.write_text(json.dumps(self.snapshot), encoding='utf-8')
        return path

    def write_legacy(self, directory, *, crlf=False):
        result = deepcopy(self.result)
        manifest = result['manifest']
        manifest['baseline'].pop('settings')
        manifest.pop('wm_steps_per_hour')
        manifest['source_sha256'] = source_hashes()
        if crlf:
            manifest['source_sha256'] = {
                name: hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()
                for name in LEGACY_SOURCES}
        result['wm_fit'].pop('state_forecast_origin')
        for name, data in (('manifest',manifest), ('fit',result['fit'])):
            (directory/(name+'.json')).write_text(json.dumps(_jsonable(data)), encoding='utf-8')
        plain = {name: value for name, value in result.items()
                 if not isinstance(value, pd.DataFrame) and name not in ('performance','fit_performance')}
        (directory/'result.json').write_text(json.dumps(_jsonable(plain)), encoding='utf-8')
        result['fit_signals'].to_csv(directory/'fit_signals_normalized.csv', index_label='Datetime')

    def test_round_trip_equals_existing_forecast_without_any_fit_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_snapshot(temporary)
            with patch('src.method.bm_1_deterministic_benchmark.fit_deterministic', side_effect=AssertionError('refit')), \
                 patch('src.exe.run_model.wm_4.output_model.statistical_output.fit', side_effect=AssertionError('baseline refit')), \
                 patch('src.params.wm_4.qp_params.fit', side_effect=AssertionError('WM refit')):
                snapshot = load_fit_snapshot(path)
                actual = forecast_from_snapshot(snapshot, self.raw.drop(columns='Demand'), forecast_hours=24)
        columns = ['O','bm1','wm4','U','x','F','R']
        np.testing.assert_allclose(actual['normalized_predictions'][columns],
                                   self.result['normalized_predictions'][columns], rtol=0, atol=1e-13)
        first = actual['normalized_predictions'].iloc[0]
        self.assertEqual(first.x, snapshot['bm']['x_origin'])
        np.testing.assert_array_equal(first[['F','R']], snapshot['wm']['state_origin'])
        self.assertEqual(snapshot['fit']['best_converged']['parameters'][3], .5)
        self.assertGreater(abs(snapshot['bm']['x_origin']-.5), .005)

    def test_future_observations_only_change_scoring_and_use_a_shared_mask(self):
        original = json.dumps(self.snapshot, sort_keys=True)
        changed = self.raw.copy()
        changed['Demand'] = np.nan
        first = forecast_from_snapshot(self.snapshot, self.raw, forecast_hours=24)
        second = forecast_from_snapshot(self.snapshot, changed, forecast_hours=24)
        pd.testing.assert_frame_equal(first['predictions'], second['predictions'])
        self.assertEqual(first['manifest']['forecast_input_audit'], second['manifest']['forecast_input_audit'])
        missing = score_forecast(first, None)
        self.assertEqual(missing['scoring']['status'], 'not-scored')
        self.assertEqual(missing['metrics'], {})
        self.assertEqual(missing['performance'], {})
        self.assertTrue(missing['daily_metrics']['samples'].eq(0).all())
        measured = self.raw.Demand.copy()
        measured.loc[first['predictions'].index[:3]] = np.nan
        scored = score_forecast(first, measured)
        self.assertEqual(scored['scoring']['scored_samples'],21)
        self.assertEqual(scored['scoring']['status'],'partial')
        self.assertEqual(scored['scoring']['models'],['O','bm1','wm4'])
        for name in ('O','bm1','wm4'):
            mask = scored['predictions'].scored
            residual = scored['predictions'].loc[mask,name]-measured.reindex(first['predictions'].index)[mask]
            self.assertAlmostEqual(scored['metrics'][name]['mse'],float(np.mean(residual**2)))
        self.assertTrue(first['predictions'].observed.isna().all())
        self.assertEqual(json.dumps(self.snapshot,sort_keys=True), original)

    def test_corrupt_or_incompatible_snapshots_are_rejected(self):
        mutations = [lambda s:s.update(schema_version=2),
                     lambda s:s['manifest']['scales'].update(price_scale=0.),
                     lambda s:s['bm'].update(x_origin=1.1),
                     lambda s:s['numerics'].update(wm_steps_per_hour=1),
                     lambda s:s['manifest']['baseline']['feature_scale'].__setitem__(0,0.),
                     lambda s:s['numerics'].update(training_refinement_error=1.),
                     lambda s:s['numerics'].update(tolerance=1e-3),
                     lambda s:s['forward_source_sha256'].update({'src/model/wm_4.py':'wrong'}),
                     lambda s:s['bm'].update(parameter_order=['k','C_hours','Xi','x0'])]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                snapshot = deepcopy(self.snapshot); mutate(snapshot)
                with self.assertRaises(ValueError): validate_snapshot(snapshot)

    def test_frozen_price_domain_failure_never_refits(self):
        raw = self.raw.copy()
        raw.loc[raw.index[192], 'Price'] = 200.
        with patch('src.method.bm_1_deterministic_benchmark.fit_deterministic', side_effect=AssertionError('refit')):
            with self.assertRaisesRegex(ValueError,'frozen training domain'):
                forecast_from_snapshot(self.snapshot,raw,forecast_hours=24)

    def test_verified_legacy_import_preserves_originals_and_exact_endpoint(self):
        for crlf in (False, True):
            with self.subTest(crlf=crlf), tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary); self.write_legacy(directory,crlf=crlf)
                # Use the actual old WM4 digest from Johan's 07fad9f source.
                digest = '8e92c6bca6d1de2dcbf971bf2d752b4992f236171fa3209e6da1b408684a2f71' if crlf else '38853b1b0992c76039c6c17e941aa2d1b35ee030fc63e5edeb05abd3d6abd6d6'
                manifest=json.loads((directory/'manifest.json').read_text())
                result=json.loads((directory/'result.json').read_text())
                manifest['source_sha256']['src/model/wm_4.py']=digest
                result['manifest']=manifest
                (directory/'manifest.json').write_text(json.dumps(manifest))
                (directory/'result.json').write_text(json.dumps(result))
                before={p.name:p.read_bytes() for p in directory.iterdir()}
                actual=migrate_legacy_fit(directory)
                self.assertEqual(actual['provenance']['mode'],'verified-legacy-import')
                self.assertEqual(actual['bm']['x_origin'],self.snapshot['bm']['x_origin'])
                np.testing.assert_allclose(actual['wm']['state_origin'],self.snapshot['wm']['state_origin'],atol=1e-14)
                self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir()})
                fine=pd.read_csv(directory/'fit_signals_normalized.csv')
                fine.iloc[:-1].to_csv(directory/'fit_signals_normalized.csv',index=False)
                with self.assertRaisesRegex(ValueError,'exact training start and cutoff'):
                    migrate_legacy_fit(directory)

    def test_legacy_without_proven_numerical_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);self.write_legacy(directory)
            manifest=json.loads((directory/'manifest.json').read_text())
            result=json.loads((directory/'result.json').read_text())
            manifest.pop('source_sha256');result['manifest']=manifest
            (directory/'manifest.json').write_text(json.dumps(manifest))
            (directory/'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'Cannot verify legacy'):
                migrate_legacy_fit(directory)

    def test_cli_saves_forecast_before_failed_workbook_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=self.write_snapshot(root);out=root/'output'
            snapshot=deepcopy(self.snapshot)
            snapshot['manifest']['data_audit']={'tag':'MED'}
            path.write_text(json.dumps(snapshot))
            def missing_workbook(*args):
                self.assertTrue((out/'forecast_predictions.csv').exists())
                self.assertTrue((out/'forecast_status.json').exists())
                raise FileNotFoundError('missing observation workbook')
            with patch('src.tools.bm_1_deterministic_data.load_forecast_inputs',return_value=self.raw[['Price','Temp']]), \
                 patch('src.tools.bm_1_deterministic_data.load_forecast_observations',side_effect=missing_workbook),redirect_stdout(io.StringIO()):
                code=main(['--from-fit',str(path),'--output',str(out)])
            self.assertEqual(code,2)
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['forecast_status'],'complete')
            self.assertEqual(result['scoring']['status'],'failed')
            prediction=pd.read_csv(out/'forecast_predictions.csv')
            self.assertEqual(len(prediction),168)
            self.assertNotIn('observed',prediction)
            self.assertTrue((out/'week.png').exists())

    def test_cli_accepts_price_temperature_csv_without_observations_or_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=self.write_snapshot(root);out=root/'output'
            csv=root/'inputs.csv';self.raw[['Price','Temp']].to_csv(csv,index_label='Datetime')
            with patch('src.exe.run_model.wm_4.output_model.statistical_output.fit',side_effect=AssertionError('refit')), \
                 redirect_stdout(io.StringIO()):
                code=main(['--from-fit',str(path),'--input-csv',str(csv),'--output',str(out)])
            self.assertEqual(code,0)
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['scoring']['status'],'not-scored')
            self.assertTrue((out/'fit_snapshot.json').exists())
            self.assertTrue((out/'week.png').exists())

    def test_new_fit_report_exports_a_loadable_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            write_result(directory,self.result)
            restored=load_fit_snapshot(directory)
            self.assertEqual(restored['bm'],self.snapshot['bm'])
            np.testing.assert_array_equal(restored['wm']['state_origin'],self.snapshot['wm']['state_origin'])
            self.assertTrue((directory/'fit_signals_normalized.csv').exists())

    def test_cli_rejects_fit_options_in_frozen_mode(self):
        with self.assertRaises(SystemExit) as exit_code,redirect_stderr(io.StringIO()):
            main(['--from-fit','unused','--initial-state','fit'])
        self.assertEqual(exit_code.exception.code,2)


if __name__=='__main__': unittest.main()
