"""The pruned G entry point retains interval predictions and safe outputs."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.exe.run_model import bm_1_g
from src.exe.run_model.bm_1 import s3
from src.method import bm_1_s3_interval as interval
from tests import test_bm1_s3_interval as fixture


class GBasicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.S3IntervalTests.setUpClass()
        cls.snapshot=fixture.S3IntervalTests.snapshot
        cls.raw=fixture.S3IntervalTests.raw

    def test_main_alias_and_removed_options(self):
        self.assertIs(bm_1_g.main,s3.main)
        for argv in (['residual'],['point-process'],['diagnose'],
                     ['predict','--sigma-x','.2'],['bands']):
            with self.subTest(argv=argv),redirect_stderr(StringIO()),self.assertRaises(SystemExit) as error:
                s3.parser().parse_args(argv)
            self.assertEqual(error.exception.code,2)

    def test_real_prediction_preserved_and_saved_before_target_failure(self):
        expected=interval.predict_interval(self.snapshot,self.raw)['predictions']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snapshot=root/'snapshot.json';inputs=root/'inputs.csv';out=root/'run'
            interval.save_snapshot(snapshot,self.snapshot)
            self.raw.to_csv(inputs,index_label='Datetime')
            with patch.object(s3,'plot'),redirect_stderr(StringIO()):
                code=bm_1_g.main(['predict','--snapshot',str(snapshot),'--inputs-csv',str(inputs),
                    '--observations-csv',str(root/'missing.csv'),'--output',str(out)])
            self.assertEqual(code,2)
            self.assertTrue((out/'forecast.json').is_file())
            actual=pd.read_csv(out/'forecast.csv',index_col='Datetime',parse_dates=['Datetime'])
            self.assertEqual(len(actual),len(expected))
            for name in ('prediction','baseline','state','state_end'):
                self.assertLess(abs(actual[name]-expected[name]).max(),1e-12)
            self.assertTrue((out/'failure.json').is_file())

    def test_completed_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'finished';out.mkdir();sentinel=out/'keep.txt';sentinel.write_text('original')
            with redirect_stderr(StringIO()),patch.object(interval,'load_snapshot') as load:
                result=bm_1_g.main(['predict','--snapshot','unused','--inputs-csv','unused','--output',str(out)])
            self.assertEqual(result,2);load.assert_not_called()
            self.assertEqual(sentinel.read_text(),'original')
            self.assertFalse((out/'failure.json').exists())


if __name__=='__main__':unittest.main()
