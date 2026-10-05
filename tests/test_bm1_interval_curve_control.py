"""Frozen synthetic control registration, failure persistence and resume."""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.exe.run_model.bm_1_interval_curve_control import main,_digest,_load_rows
from src.tools.bm_1_reports import _atomic_report
from src.validation.bm_1.interval_characteristic import CharacteristicNumericalError
from src.validation.bm_1.interval_curve_control import declared_control,run_curve_control


def fixtures():
    edges = np.linspace(0.,1.,25).tolist()
    design = dict(rows=24,observations=dict(starts=edges[:-1],ends=edges[1:],
        values=[.5]*24,observed=[True]*24),baseline_values=[.5]*24,price_values=[.5]*24,
        parameters=dict(C=.15,k=3.,Xi=.8,sigma_x=.4,sigma_y=.08))
    report = dict(schema='bm1-hourly-interval-development-v1',declared_design=design,
        historical_observation_values_read=0,
        development_tolerances=dict(total_log_likelihood=.05,final_state_mean=.002,final_mass_l1=.05))
    row = dict(status='completed',cells=1281,max_step=1/12288,frequencies=64,span=4.,
        state_scheme='positive_crank_nicolson',x_edges=[0.,.5,1.],final_posterior_mass=[.3,.7],
        log_likelihood=-1.,posterior_state_means=[.6]*24)
    reference = dict(schema='bm1-interval-characteristic-cn-control-v1',fixed_development_design=design,
        historical_observation_values_read=0,development_numerics_pass=True,replays=[row])
    return report,reference,row


class CurveControlTests(unittest.TestCase):
    def test_registration_keeps_one_shared_pair_and_all_saved_observations(self):
        report,reference,_ = fixtures()
        for case,sigma in (('two-narrow',.04),('four-wide',.16)):
            record,design = declared_control(report,reference,case)
            self.assertEqual(record.sigma_y,sigma)
            self.assertEqual(len(record.observations.times),24)
            np.testing.assert_array_equal(record.observations.values,report['declared_design']['observations']['values'])
            self.assertEqual(len(design['settings']),5)
            self.assertEqual(record.parameters[1].to_record(),design['f'])
            self.assertEqual(record.parameters[2].to_record(),design['g'])
        bad = dict(reference,historical_observation_values_read=1)
        with self.assertRaises(ValueError): declared_control(report,bad,'two-narrow')

    def test_numerical_failures_are_checkpointed_without_retries_or_acceptance(self):
        report,reference,_ = fixtures()
        saved=[]
        error = CharacteristicNumericalError('declared guard',completed_intervals=2)
        with patch('src.validation.bm_1.interval_curve_control.replay_characteristic',side_effect=error) as replay:
            result = run_curve_control(report,reference,'two-narrow',checkpoint=lambda rows:saved.append(len(rows)))
        self.assertEqual(replay.call_count,5)
        self.assertEqual(saved,[1,2,3,4,5])
        self.assertFalse(result['development_numerics_pass'])
        self.assertFalse(result['scientific_acceptance'])
        self.assertTrue(all(row['status']=='numerical_guard_failed' for row in result['replays']))
        self.assertTrue(all(not row['available'] for row in result['comparisons']))

    def test_paused_cli_resumes_without_replaying_completed_unit_and_rejects_corruption(self):
        report,reference,row = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data,ref,out = (root/name for name in ('data.json','ref.json','out.json'))
            data.write_text(json.dumps(report));ref.write_text(json.dumps(reference))
            args = ['--development-report',str(data),'--cn-reference',str(ref),'--case','two-narrow','--output',str(out)]
            def fake_replay(record,**settings):
                return dict(row,**settings)
            def pause_after_write(path,value,**kwargs):
                _atomic_report(path,value,**kwargs)
                raise RuntimeError('pause after committed unit')
            source = dict(commit='frozen',dirty=False,files={})
            with patch('src.exe.run_model.bm_1.curves.source_snapshot',return_value=source), \
                 patch('src.validation.bm_1.interval_curve_control.replay_characteristic',side_effect=fake_replay) as replay, \
                 patch('src.tools.bm_1_reports._atomic_report',side_effect=pause_after_write),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'pause'): main(args)
                self.assertEqual(replay.call_count,1)
            checkpoint = out.with_suffix('.checkpoint.json')
            original = json.loads(checkpoint.read_text())
            with patch('src.exe.run_model.bm_1.curves.source_snapshot',return_value=source), \
                 patch('src.validation.bm_1.interval_curve_control.replay_characteristic',side_effect=fake_replay) as replay, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(main(args+['--resume']),0)
                self.assertEqual(replay.call_count,4)
            final = json.loads(out.read_text())
            self.assertEqual(final['resumed_units'],1)
            self.assertEqual(final['replays'][0],original['rows'][0])
            original['rows'][0]['log_likelihood']=7.
            checkpoint.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,'corrupted rows'):
                _load_rows(checkpoint,original['manifest'])
            original['rows_sha256']=_digest(original['rows'])
            checkpoint.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError,'mismatch'):
                _load_rows(checkpoint,dict(original['manifest'],source='changed'))


if __name__ == '__main__': unittest.main()
