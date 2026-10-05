"""Partitioned execution must preserve raw science, saved work and acceptance gates."""
from contextlib import contextmanager
import io
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.exe.run_model.bm_1 import estimate
from src.tools import bm_1_distributed as dist
from src.tools.bm_1_reports import _atomic_report, _read_likelihood_report
from src.validation.bm_1 import joint_night as night, joint_population as population
from src.validation.bm_1 import joint_robustness as robustness, joint_estimation as fv
from src.validation.bm_1.joint_partition import JobStudy, jobs
from tests.test_bm1_joint_night import small_night, demand_replay


class DistributedTests(unittest.TestCase):
    source = dict(source_commit='a'*40, working_tree_dirty=False)
    dependencies = dict(g2a_v2={'fixed': 'raw fixture'})

    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as directory, small_night(), \
             patch.object(estimate, '_source_snapshot', return_value=self.source), \
             patch.object(dist, '_dependencies', return_value=self.dependencies), \
             patch.object(dist, 'scientific_fingerprint', return_value={'solver': 'unchanged'}), \
             patch.object(fv, '_candidate_replay', side_effect=demand_replay) as replay, \
             patch('sys.stdout', new_callable=io.StringIO):
            root = Path(directory)
            _atomic_report(root/'g2a/manifest.json', {})
            _atomic_report(root/'v1.json.gz', {})
            yield root, replay

    def prepare(self, root, resume=None):
        args = SimpleNamespace(stage='G2b', run_dir=root/'distributed', shards=2,
            g2a_report=root/'v1.json.gz', g2a_run_dir=root/'g2a', g2b_run_dir=None, resume_from=resume)
        return dist.prepare(args)

    def run_args(self, root, shard):
        return SimpleNamespace(run_dir=root/'distributed', shard=shard, workers=1, chunk_size=4,
                               max_hours=7.5, limit_chunks=None, quiet=True, collect_only=False)

    def test_registered_scopes_and_disjoint_assignments(self):
        self.assertEqual(len(jobs(night)), 10)
        self.assertEqual(night.design_header()['expected_conditional_runs'], 211560)
        self.assertEqual(len(jobs(population)), 560)
        self.assertEqual(population.design_header()['expected_conditional_runs'], 764400)
        for study in (night, population):
            assignment = dist.assignments(study, 2)
            self.assertEqual(set(assignment), set(jobs(study)))
            self.assertEqual(set(assignment.values()), {0, 1})
            self.assertEqual(assignment, dist.assignments(study, 2))
        design = population.design_header()
        self.assertEqual(design['population_size_per_pair'], 200)
        self.assertEqual(design['paired_robustness_size_per_pair'], 20)
        self.assertEqual(design['population_starts'], [1362, 1562])
        self.assertLess(max(u[1] for u in population.units()), 1762)
        root = Path('C:/Users/jvils/OneDrive/Skrivebord/Til VS/fag/7 semester/BSCe2026/'
                    'files/output/bm_1/runs/estimation/g2b_distributed_cloud_return_001')
        plan = dict(assignment=dist.assignments(night, 2))
        paths = [dist._job_dir(root, plan, name)/'units'/(night.unit_name(u)+'.npz.abcdefgh.partial')
                 for name, units in jobs(night).items() for u in units]
        self.assertLess(max(len(str(path)) for path in paths), 250)

    def test_scientific_fingerprint_follows_imports_and_ignores_execution_only_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(['git', '-c', 'user.name=Test', '-c',
                    'user.email=test@example.invalid', *args], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
            git('init')
            entry = root/'src/validation/bm_1/joint_night.py'
            entry.parent.mkdir(parents=True)
            entry.write_text('from src import solver\n')
            solver = root/'src/solver.py'
            solver.write_text('value = 1\n')
            git('add', '.')
            git('commit', '-m', 'first')
            first = git('rev-parse', 'HEAD')
            (root/'launcher.py').write_text('workers = 2\n')
            git('add', '.')
            git('commit', '-m', 'execution')
            with patch.object(dist, '_cli', return_value=SimpleNamespace(ROOT=root)):
                original = dist.scientific_fingerprint(first)
                self.assertIn('src/solver.py', original)
                self.assertEqual(original, dist.scientific_fingerprint('HEAD'))
                solver.write_text('value = 2\n')
                git('add', '.')
                git('commit', '-m', 'scientific change')
                self.assertNotEqual(original, dist.scientific_fingerprint('HEAD'))

    def test_two_shards_resume_collect_all_seventy_controls_without_duplicate_replays(self):
        with self.fixture() as (root, replay):
            self.prepare(root)
            args = self.run_args(root, 0)
            args.limit_chunks = 1
            self.assertEqual(dist.run(args)['new_blocks'], 1)
            self.assertEqual(replay.call_count, 4)
            partial = dist.collect(SimpleNamespace(run_dir=args.run_dir, from_run=[], output=root/'partial'))
            self.assertFalse(partial['completed'])
            self.assertFalse(partial['passes'])
            args.limit_chunks = None
            dist.run(args)
            args.shard = 1
            dist.run(args)
            self.assertEqual(replay.call_count, 600)
            complete = dist.collect(SimpleNamespace(run_dir=args.run_dir, from_run=[], output=root/'complete'))
            self.assertTrue(complete['completed'])
            self.assertTrue(complete['passes'])
            self.assertEqual(len(complete['comparisons']), 70)
            self.assertEqual(complete['completed_units'], 40)
            dist.run(args)
            self.assertEqual(replay.call_count, 600)

    def test_migration_preserves_complete_and_partial_work_and_original_bytes(self):
        with self.fixture() as (root, replay):
            old = root/'old'
            args = self.run_args(root, 0)
            args.run_dir, args.limit_chunks, args.chunk_size = old, 5, 4
            truths = {}
            for unit in night.units():
                if unit[0] not in truths:
                    truths.update(dist._truth(unit, night))
            estimate._run_rectangular_blocks(args, self.source, self.dependencies, truths, study=night)
            self.assertEqual(replay.call_count, 19)  # One 15-node final unit plus four saved nodes.
            before = {p.relative_to(old): dist.digest(p) for p in old.rglob('*') if p.is_file()}
            plan = self.prepare(root, old)
            self.assertEqual(sum(plan['imported_hypotheses'].values()), 19)
            for relative, sha in before.items():
                self.assertEqual(dist.digest(old/relative), sha)
            for shard in range(2):
                dist.run(self.run_args(root, shard))
            self.assertEqual(replay.call_count, 600)
            complete = dist.collect(SimpleNamespace(run_dir=root/'distributed', from_run=[], output=root/'all'))
            self.assertTrue(complete['passes'])

    def test_source_and_assignment_changes_fail_before_replay(self):
        with self.fixture() as (root, replay):
            plan = self.prepare(root)
            args = self.run_args(root, 0)
            with patch.object(estimate, '_source_snapshot', return_value=dict(self.source, source_commit='b'*40)):
                with self.assertRaisesRegex(ValueError, 'pinned'):
                    dist.run(args)
            plan['assignment'][next(iter(plan['assignment']))] = 3
            _atomic_report(args.run_dir/'plan.json', plan, replace_existing=True)
            with self.assertRaises(ValueError):
                dist.run(args)
            replay.assert_not_called()

    def test_time_budget_applies_to_whole_invocation_including_setup(self):
        with self.fixture() as (root, replay):
            self.prepare(root)
            args = self.run_args(root, 0)
            args.max_hours = 1/3600
            clock = iter([0.] + [2.]*1000)
            with patch.object(dist, 'perf_counter', side_effect=lambda: next(clock)):
                dist.run(args)
            replay.assert_not_called()

    def test_stop_marker_pauses_at_a_saved_block_and_can_resume(self):
        with self.fixture() as (root, replay):
            self.prepare(root)
            args = self.run_args(root, 0)
            stop = args.run_dir/'shards/0/STOP'
            advance = night.advance
            def stop_after_block(*a, **kwargs):
                advance(*a, **kwargs)
                stop.touch()
            with patch.object(night, 'advance', side_effect=stop_after_block):
                self.assertEqual(dist.run(args)['new_blocks'], 1)
            self.assertEqual(replay.call_count, 4)
            self.assertEqual(dist.run(args)['new_blocks'], 0)
            stop.unlink()
            args.limit_chunks = 1
            self.assertEqual(dist.run(args)['new_blocks'], 1)
            self.assertEqual(replay.call_count, 8)

    def test_bundle_cloud_snapshot_and_conflicting_completed_copies(self):
        with self.fixture() as (root, replay):
            self.prepare(root)
            index = root/'transfer/cloud.json'
            dist.export(SimpleNamespace(run_dir=root/'distributed', shard=1, output=index, part_mib=1))
            cloud = root/'cloud'
            dist.unpack(SimpleNamespace(bundle=index, run_dir=cloud))
            args = self.run_args(root, 1)
            args.run_dir, args.limit_chunks = cloud, 4
            dist.run(args)
            files = list((cloud/'shards').glob('*/jobs/*/units/*.npz'))
            self.assertEqual(len(files), 1)
            source = files[0]
            relative = source.relative_to(cloud)
            dist._copy(source, root/'distributed'/relative)
            dist.collect(SimpleNamespace(run_dir=root/'distributed', from_run=[cloud], output=root/'same'))
            report = _read_likelihood_report(source)[0]
            report['duplicate_marker'] = 'different bytes'
            _atomic_report(source, report, replace_existing=True)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                dist.collect(SimpleNamespace(run_dir=root/'distributed', from_run=[cloud], output=root/'conflict'))

    def test_population_refuses_incomplete_pilot_before_any_fv_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _atomic_report(root/'inputs/g2b/manifest.json', dict(dependencies=self.dependencies))
            with patch.object(estimate, '_night_dependencies', return_value=self.dependencies), \
                 patch.object(dist, '_validate_flat', return_value={'completed': False, 'passes': False}), \
                 patch.object(fv, '_candidate_replay') as replay:
                with self.assertRaisesRegex(ValueError, 'G2c requires'):
                    dist._dependencies(root, 'G2c')
                replay.assert_not_called()

    def test_population_single_grid_uses_registered_paired_observations(self):
        with small_night(), patch.object(population, 'DAYS', 1), \
             patch.object(population, 'GRIDS', {k:(v[0],) for k,v in night.GRIDS.items()}), \
             patch.object(robustness, 'POPULATION_STARTS', night.PILOT_IDS), \
             patch.object(robustness, 'POPULATION_SIZE', 1), \
             patch.object(robustness, 'ROBUSTNESS_SIZE', 1), \
             patch.object(fv, '_candidate_replay', side_effect=demand_replay):
            unit = population.units()[0]
            group = population.make_group(unit)
            reducer = population.RectangularReduction(group)
            population.advance(reducer, len(group['hypotheses']))
            rows = reducer.finish()
            self.assertEqual(len(rows), 1)
            self.assertTrue(population.validate_rows(rows, group))
            self.assertTrue(JobStudy(population, next(iter(jobs(population)))).summarize(rows)['passes'])
            self.assertFalse(population.summarize(rows)['completed'])

    def test_real_solver_checkpoint_transfer_and_spawn_match_serial(self):
        with self.fixture() as (root, _):
            # Override the fixture's fake replay with the original real solver implementation.
            from tests.test_bm1_joint_pilot import fv as actual_fv
            # mock.patch keeps the original callable; captured below before applying fixture.
            with patch.object(actual_fv, '_candidate_replay', REAL_REPLAY):
                plan = self.prepare(root)
                name = next(iter(jobs(night)))
                unit = jobs(night)[name][0]
                owner = plan['assignment'][name]
                args = self.run_args(root, owner)
                args.limit_chunks = 1
                dist.run(args)
                index = root/'cloud.json'
                dist.export(SimpleNamespace(run_dir=args.run_dir, shard=owner, output=index, part_mib=1))
                cloud = root/'cloud'
                dist.unpack(SimpleNamespace(bundle=index, run_dir=cloud))
                args.run_dir, args.limit_chunks, args.workers = cloud, 3, 2
                dist.run(args)
                directory = dist._job_dir(cloud, plan, name)
                actual = _read_likelihood_report(directory/'units'/(night.unit_name(unit)+'.npz'))[0]['rows']
                group = night.make_group(unit)
                serial = night.RectangularReduction(group)
                night.advance(serial, len(group['hypotheses']))
                for a, b in zip(actual, serial.finish()):
                    for key in ('conditional_log_evidence', 'marginal_posterior_mass',
                                'conditional_predicted_demand_mean', 'marginal_posterior_demand_mean'):
                        np.testing.assert_allclose(a[key], b[key], atol=1e-13, rtol=0)


REAL_REPLAY = fv._candidate_replay


if __name__ == '__main__':
    unittest.main()
