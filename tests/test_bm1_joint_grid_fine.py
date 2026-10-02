"""G1b2 keeps the original rejection, gates and full provenance chain."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.validation.bm_1 import joint_grid as grid
from src.validation.bm_1 import joint_estimation as fv
from tests.test_bm1_joint_grid import constant_replay
from tests.test_bm1_joint_grid_refinement import peaked_replay


class JointFineGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(fv,'_candidate_replay',side_effect=peaked_replay):
            cls.previous=grid.run_joint_parameter_grid(workers=1)
        cls.previous.update(source_commit='a'*40,working_tree_dirty=False,
            dependencies=dict(g1a=dict(sha256='d'*64,source_commit='e'*40,
                                      g0=dict(sha256='f'*64,source_commit='1'*40))))

    def test_full_registered_design_and_nested_overlap(self):
        events=[]
        with patch.object(fv,'_candidate_replay',side_effect=constant_replay) as replay:
            result=grid.run_joint_parameter_grid_fine(workers=1,progress=events.append)
        self.assertEqual((result['stage'],result['design_version']),('G1b2',3))
        self.assertEqual(result['parameter_grid_counts'],[5,9,17,33])
        self.assertTrue(result['registered_design']);self.assertTrue(result['passes'])
        self.assertEqual(replay.call_count,8712)
        self.assertEqual((len(result['rows']),len(result['comparisons'])),(32,24))
        self.assertEqual(sum(c['selected'] for c in result['comparisons']),20)
        self.assertTrue(all(c['left'].startswith('17/') and c['right'].startswith('33/')
                            for c in result['comparisons'] if c['selected'] and c['kind']=='parameter_grid'))
        self.assertEqual([e['completed'] for e in events],list(range(1,8713)))
        self.assertEqual(grid.compare_grid_overlap(result,result,counts=(9,17))['bank_comparisons'],16)

    def test_cli_preserves_rejection_and_records_complete_chain(self):
        from src.exe.run_model.bm_1 import estimate
        with tempfile.TemporaryDirectory() as directory,patch('builtins.print'):
            source=Path(directory)/'g1b0.json.gz';output=Path(directory)/'g1b2.json.gz'
            with gzip.open(source,'wt') as f:json.dump(self.previous,f)
            args=estimate.build_parser().parse_args(['joint-bank-grid-fine','--g1b0-report',str(source),
                                                     '--output',str(output),'--blas-threads','1'])
            self.assertEqual(args.workers,8)
            original=source.read_bytes()
            for science,reproduction in ((False,True),(True,False)):
                result=dict(stage='G1b2',registered_design=True,passes=science,comparisons=[])
                overlap=dict(bank_comparisons=16,maximum_errors=dict(scores=0.),passes=reproduction)
                with patch.object(grid,'run_joint_parameter_grid_fine',return_value=result) as run,\
                     patch.object(grid,'compare_grid_overlap',return_value=overlap) as compare,\
                     self.assertRaisesRegex(RuntimeError,'report retained'):
                    estimate.run_joint_parameter_grid_fine(args)
                self.assertEqual(run.call_args.kwargs['workers'],8)
                self.assertEqual(compare.call_args.kwargs,{})
                with gzip.open(output,'rt') as f:saved=json.load(f)
                self.assertFalse(saved['passes'])
                self.assertEqual(saved['dependencies']['g1b0']['sha256'],hashlib.sha256(original).hexdigest())
                self.assertEqual(saved['dependencies']['g1b0']['g1a'],self.previous['dependencies']['g1a'])
                with self.assertRaises(FileExistsError):estimate.run_joint_parameter_grid_fine(args)
                output.unlink()
            self.assertEqual(source.read_bytes(),original)

    def test_diagnostic_overrides_are_not_registered(self):
        numerics={k:dict(time_steps=(1/12,1/24,1/48),spatial_cells=(7,11,15),quadrature_orders=(1,3,5))
                  for k in fv.G1A_NUMERICS}
        with patch.object(fv,'_candidate_replay',side_effect=constant_replay):
            r=grid.run_joint_parameter_grid_fine(workers=1,counts=(2,3,5,9),numerics=numerics,sequence_ids=(0,1))
        self.assertFalse(r['registered_design']);self.assertEqual(r['conditional_runs'],648)


if __name__=='__main__':unittest.main()
