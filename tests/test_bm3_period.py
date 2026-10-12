"""Calendar, causal scaling and saved-reference guards for the full-period runner."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.exe.run_model.bm_3.run_all_days import eligible_tasks,prepare_day,reuse_manifest
from tests.test_bm1_thomas_pipeline import frame_fixture


class PeriodTests(unittest.TestCase):
    def test_all_day_tasks_have_history_and_correct_horizons(self):
        frame = frame_fixture()
        tasks,skipped = eligible_tasks(frame,'2022-01-01','2022-02-28',hours=[0,23])
        self.assertTrue(skipped)
        self.assertEqual(len({t for t in tasks}),len(tasks))
        self.assertFalse(any(t[0]<'2022-02-06' for t in tasks))
        self.assertFalse(any(t[2]=='daily' and t[0]<'2022-02-12' for t in tasks))
        self.assertIn(('2022-02-20','logistic','mpc',23),tasks)
        self.assertIn(('2022-02-20','logistic','daily',-1),tasks)

    def test_missing_future_hour_excludes_only_affected_origins(self):
        frame = frame_fixture().drop(pd.Timestamp('2022-02-21 04:00'))
        tasks,skipped = eligible_tasks(frame,'2022-02-20','2022-02-20',hours=[0,23])
        self.assertIn(('2022-02-20','logistic','daily',-1),tasks)
        self.assertIn(('2022-02-20','logistic','mpc',0),tasks)
        self.assertNotIn(('2022-02-20','logistic','mpc',23),tasks)
        self.assertEqual(skipped[0]['reason'],'incomplete target/forecast window')

    def test_missing_history_is_not_imputed(self):
        frame = frame_fixture().drop(pd.Timestamp('2022-02-13 04:00'))
        tasks,skipped = eligible_tasks(frame,'2022-02-20','2022-02-20',hours=[0])
        self.assertFalse(tasks)
        self.assertTrue(all(row['reason']=='incomplete historical 24h window' for row in skipped))

    def test_preparation_does_not_use_future_demand_for_baseline(self):
        frame = frame_fixture();day = pd.Timestamp('2022-02-20');raw = 2+3*frame.I
        a = prepare_day(frame,raw,day,with_daily=True)
        changed = frame.copy();changed.loc[changed.index>=day,'I'] = .99
        b = prepare_day(changed,2+3*changed.I,day,with_daily=True)
        np.testing.assert_array_equal(a['target']['B'],b['target']['B'])
        for x,y in zip(a['training'],b['training']):
            np.testing.assert_array_equal(x['I'],y['I'])
        self.assertEqual(a['raw_offset'],b['raw_offset'])
        self.assertEqual(a['raw_scale'],b['raw_scale'])
        # Oldest-training-day baselines are not required by the existing MPC recipe.
        mpc = prepare_day(frame,raw,'2022-02-06',with_daily=False)
        self.assertIsNone(mpc['target'])
        self.assertIsNone(mpc['training'])

    def test_saved_reference_identity_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'settings.json').write_text(json.dumps({'inputs_sha256':{'input.xlsx':'old'},'fingerprint':'original'}))
            with self.assertRaisesRegex(ValueError,'hashes'):
                reuse_manifest(root,[('2022-05-02','logistic','daily',-1)],{'input.xlsx':'new'})
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(reuse_manifest(Path(directory),[],{}),({},{}))


if __name__=='__main__':unittest.main()
