"""Portable, immutable G2 job assignments and raw-validated result collection.

One owner per shard. A shard is a collection of complete record/scenario jobs;
the existing ordered reducer and atomic block checkpoints remain authoritative.
This module never accepts a saved summary as numerical evidence.
"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from types import SimpleNamespace
import uuid

import numpy as np

from src.tools.bm_1_reports import _RunLock, _atomic_report, _read_likelihood_report
from src.validation.bm_1 import joint_grid as grid, joint_night as night
from src.validation.bm_1.joint_partition import (
    JobStudy, compact_record, jobs, study_for, summary_records,
)
from src.exe.run_model.progress import fv_work, runtime_text


def _cli():
    from src.exe.run_model.bm_1 import estimate
    return estimate


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    before = digest(source)
    shutil.copyfile(source, target)
    if digest(target) != before or digest(source) != before:
        raise ValueError('File changed during snapshot: '+str(source))


def _snapshot_run(source, target, *, checkpoint=False):
    _copy(source/'manifest.json', target/'manifest.json')
    for path in sorted((source/'units').glob('*.npz')):
        _copy(path, target/'units'/path.name)
    if checkpoint and (source/'checkpoint.npz').exists():
        _copy(source/'checkpoint.npz', target/'checkpoint.npz')


def _truth(unit, study):
    pair, sid = unit[:2]
    return {pair: grid._g2a_truth_record(pair, sid, study.DAYS)}


class _LazyTruth(dict):
    def __init__(self, unit, study):
        super().__init__()
        self.unit, self.study = unit, study

    def __missing__(self, pair):
        self.update(_truth(self.unit, self.study))
        return self[pair]


def _cost(unit, study):
    hours = 2 if unit[2] == 'sampling_2h' else 4 if unit[2] == 'sampling_4h' else 1
    return fv_work(dict(study.configurations(unit[0]))[unit[-1]], study.DAYS, study.DAYS*24//hours)


def _count(unit, study):
    return int(np.prod(study.GRIDS['__'.join(unit[0])][-1]))


def assignments(study, shards, saved=None):
    if isinstance(shards, bool) or not isinstance(shards, int) or not 1 <= shards <= len(jobs(study)):
        raise ValueError('Shard count must be between one and the number of jobs.')
    saved = saved or {}
    costs = {name: sum((_count(u, study)-saved.get(study.unit_name(u), 0))*_cost(u, study)
                       for u in units) for name, units in jobs(study).items()}
    loads, result = [0.]*shards, {}
    # Largest first, stable tie breaking. This changes placement, never scientific order within a job.
    for name in sorted(costs, key=lambda n: (-costs[n], n)):
        owner = min(range(shards), key=lambda i: (loads[i], i))
        result[name] = owner
        loads[owner] += costs[name]
    return result


def scientific_fingerprint(revision):
    """Conservative G2b numerical closure. I/O/CLI/H-only response code is excluded.

    A migration can change scheduling/storage only. An unavailable old revision
    or any numerical source change fails closed, even if a summary claims success.
    """
    import ast
    repository = _cli().ROOT
    tracked = set(subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', revision], cwd=repository, text=True).splitlines())
    pending, result = ['src/validation/bm_1/joint_night.py'], {}
    while pending:
        path = pending.pop()
        if path in result:
            continue
        raw = subprocess.check_output(['git', 'show', revision+':'+path], cwd=repository)
        result[path] = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        for parent in Path(path).parents:
            initializer = (parent/'__init__.py').as_posix()
            if initializer in tracked:
                pending.append(initializer)
        for node in ast.walk(ast.parse(raw, filename=path)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level:
                    package = path[:-3].split('.') if '/' not in path else path[:-3].split('/')
                    module = '.'.join(package[:-node.level] + ([module] if module else []))
                modules = [module] + [module+'.'+alias.name for alias in node.names]
            for module in modules:
                if module == 'src' or module.startswith('src.'):
                    stem = module.replace('.', '/')
                    for candidate in (stem+'.py', stem+'/__init__.py'):
                        if candidate in tracked and candidate not in result:
                            pending.append(candidate)
    return result



def _validate_flat(directory, study, dependencies):
    """Stream one record/scenario at a time; G2c raw arrays need not fit in RAM."""
    cli = _cli()
    manifest, manifest_digest = _read_likelihood_report(directory/'manifest.json')
    grid._check_recorded(manifest['stage'], study.STAGE, 'stage')
    grid._check_recorded(manifest['design'], study.design_header(), 'design')
    grid._check_recorded(manifest['dependencies'], dependencies, 'dependencies')
    _clean_source(manifest['source'])
    expected = {study.unit_name(u)+'.npz' for u in study.units()}
    if {p.name for p in (directory/'units').glob('*.npz')} - expected:
        raise ValueError('Unregistered raw unit.')
    records, comparisons, hashes = [], [], {}
    for name in jobs(study):
        adapter = JobStudy(study, name)
        # _collect_rectangular rejects other jobs, so validate selected files explicitly here.
        rows = []
        truth = None
        for unit in adapter.units():
            path = directory/'units'/(study.unit_name(unit)+'.npz')
            if not path.exists():
                continue
            if truth is None:
                truth = _truth(unit, study)
            report, sha = _read_likelihood_report(path)
            for key, value in cli._rectangular_identity(manifest, unit, study).items():
                grid._check_recorded(report.get(key), value, key)
            _clean_source(manifest['source'])
            study.validate_rows(report['rows'], study.make_group(unit, truth[unit[0]]))
            if report.get('raw_validation') != {'passes': True}:
                raise ValueError('Raw validation did not pass.')
            rows.extend(report['rows'])
            hashes[study.unit_name(unit)] = sha
        records.append(compact_record(rows, study))
        comparisons.extend(adapter.summarize(rows)['comparisons'])
    summary = summary_records('G2b' if study is night else 'G2c', records, comparisons)
    summary.update(unit_sha256=hashes, manifest_sha256=manifest_digest, source=manifest['source'],
                   completed_units=len(hashes), expected_units=len(expected))
    return summary


def _clean_source(source):
    import re
    if source.get('working_tree_dirty') is not False or not re.fullmatch('[0-9a-f]{40}', str(source.get('source_commit'))):
        raise ValueError('A clean recorded source commit is required.')


def _dependencies(root, stage):
    base = _cli()._night_dependencies(root/'inputs/g2a_v1.json.gz', root/'inputs/g2a')
    if stage == 'G2c':
        directory = root/'inputs/g2b'
        manifest = _read_likelihood_report(directory/'manifest.json')[0]
        if manifest['dependencies'].get('g2a_v2') != base['g2a_v2']:
            raise ValueError('G2b/G2a dependency mismatch.')
        accepted = _validate_flat(directory, night, manifest['dependencies'])
        if not accepted['completed'] or not accepted['passes'] or len(accepted['comparisons']) != 70:
            raise ValueError('G2c requires complete raw-validated passing G2b and all 70 controls.')
        base['g2b_v2'] = dict(manifest_sha256=accepted['manifest_sha256'],
            source_commit=manifest['source']['source_commit'], unit_sha256=accepted['unit_sha256'],
            dependencies=manifest['dependencies'])
    return base


def _import_state(root, dependencies, source):
    """Validate old evidence in place before creating derived, explicitly linked envelopes."""
    directory = root/'inputs/resume'
    if not directory.exists():
        return {}, None, None
    manifest = _read_likelihood_report(directory/'manifest.json')[0]
    _clean_source(manifest['source'])
    old_fingerprint = scientific_fingerprint(manifest['source']['source_commit'])
    if old_fingerprint != scientific_fingerprint(source['source_commit']):
        raise ValueError('Scientific source differs; old work cannot be imported by an execution-only migration.')
    summary = _validate_flat(directory, night, dependencies)
    if summary['failed_comparisons']:
        raise ValueError('Imported numerical comparisons failed.')
    saved = {name: _count(u, night) for u in night.units()
             if (name := night.unit_name(u)) in summary['unit_sha256']}
    checkpoint_path = directory/'checkpoint.npz'
    checkpoint = _read_likelihood_report(checkpoint_path)[0] if checkpoint_path.exists() else None
    if checkpoint is not None:
        unit = next((u for u in night.units() if night.unit_name(u) == checkpoint.get('unit_name')), None)
        if unit is None:
            raise ValueError('Unregistered imported checkpoint.')
        for key, value in _cli()._rectangular_identity(manifest, unit, night).items():
            grid._check_recorded(checkpoint.get(key), value, 'import '+key)
        reducer = night.RectangularReduction(night.make_group(unit)).restore(checkpoint['reduction'])
        pending = [u for u in night.units() if night.unit_name(u) not in saved]
        if night.unit_name(unit) in saved:
            if reducer.seen != _count(unit, night):
                raise ValueError('Imported final unit conflicts with its partial checkpoint.')
            checkpoint = None
        elif not pending or unit != pending[0]:
            raise ValueError('Imported checkpoint is not the first pending unit.')
        else:
            saved[night.unit_name(unit)] = reducer.seen
    return saved, checkpoint, old_fingerprint


def _job_dir(root, plan, name):
    return root/'shards'/str(plan['assignment'][name])/'jobs'/name


def _runtime_dependencies(plan):
    return dict(plan['dependencies'], distributed_plan=plan_hash(plan))


def prepare(args):
    cli, root = _cli(), args.run_dir
    source = cli._source_snapshot()
    _clean_source(source)
    study = study_for(args.stage)
    assignments(study, args.shards)  # Validate options before copying any evidence.
    if args.stage == 'G2c' and (args.g2b_run_dir is None or args.resume_from is not None):
        raise ValueError('G2c requires --g2b-run-dir; legacy resume import is G2b-only.')
    if root.exists():
        raise FileExistsError('Use a new run directory; existing runs are never overwritten.')
    with ExitStack() as locks:
        for directory in (args.g2a_run_dir, args.g2b_run_dir, args.resume_from):
            if directory is not None:
                locks.enter_context(_RunLock(directory))
        root.mkdir(parents=True)
        _copy(args.g2a_report, root/'inputs/g2a_v1.json.gz')
        _snapshot_run(args.g2a_run_dir, root/'inputs/g2a')
        if args.g2b_run_dir is not None:
            _snapshot_run(args.g2b_run_dir, root/'inputs/g2b')
        if args.resume_from is not None:
            _snapshot_run(args.resume_from, root/'inputs/resume', checkpoint=True)
    dependencies = _dependencies(root, args.stage)
    saved, checkpoint, fingerprint = _import_state(root, dependencies, source)
    inventory = {p.relative_to(root).as_posix(): digest(p) for p in (root/'inputs').rglob('*')
                 if p.is_file() and p.name != 'run.lock'}
    plan = dict(format='bm1-distributed-v1', id=str(uuid.uuid4()), stage=args.stage,
        source=source, design=study.design_header(), dependencies=dependencies,
        shards=args.shards, assignment=assignments(study, args.shards, saved),
        imported_hypotheses=saved, imported_scientific_blobs=fingerprint, inputs=inventory)
    manifest = dict(stage=study.STAGE, design=study.design_header(), source=source,
                    dependencies=_runtime_dependencies(plan))
    # Original files remain byte-for-byte in inputs/resume. Only derived envelopes get new source.
    for name, units in jobs(study).items():
        directory = _job_dir(root, plan, name)
        _atomic_report(directory/'manifest.json', manifest)
        for unit in units:
            old_path = root/'inputs/resume/units'/(study.unit_name(unit)+'.npz')
            if old_path.exists():
                report, sha = _read_likelihood_report(old_path)
                report.update(cli._rectangular_identity(manifest, unit, study),
                              imported_from=dict(sha256=sha, source=report['source']))
                _atomic_report(directory/'units'/old_path.name, report)
            if checkpoint is not None and checkpoint['unit_name'] == study.unit_name(unit):
                old_source = checkpoint['source']
                checkpoint.update(cli._rectangular_identity(manifest, unit, study), imported_from=dict(
                    sha256=inventory['inputs/resume/checkpoint.npz'], source=old_source))
                _atomic_report(directory/'checkpoint.npz', checkpoint)
    if cli._source_snapshot() != source:
        raise ValueError('Source changed during preparation; plan not published.')
    _atomic_report(root/'plan.json', plan)  # Published last: an incomplete preparation cannot run.
    print(f'Prepared {args.stage}: {len(jobs(study))} jobs, {args.shards} shards; '
          f'{sum(saved.values())} imported hypotheses; plan={plan_hash(plan)}', flush=True)
    return plan


def read_plan(root):
    plan = _read_likelihood_report(root/'plan.json')[0]
    if plan.get('format') != 'bm1-distributed-v1':
        raise ValueError('Unknown distributed plan.')
    study = study_for(plan['stage'])
    _clean_source(plan['source'])
    grid._check_recorded(plan['design'], study.design_header(), 'distributed design')
    grid._check_recorded(plan['assignment'], assignments(study, plan['shards'], plan['imported_hypotheses']),
                        'immutable assignment')
    for relative, sha in plan['inputs'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or '\\' in relative or path.parts[0] != 'inputs':
            raise ValueError('Unsafe input path.')
        if digest(root/path) != sha:
            raise ValueError('Changed immutable input: '+relative)
    grid._check_recorded(_dependencies(root, plan['stage']), plan['dependencies'], 'raw prerequisites')
    if plan['imported_scientific_blobs'] is not None:
        saved, _, fingerprint = _import_state(root, plan['dependencies'], plan['source'])
        grid._check_recorded(saved, plan['imported_hypotheses'], 'imported work')
        grid._check_recorded(fingerprint, plan['imported_scientific_blobs'], 'imported source')
    _validate_layout(root, plan)
    return plan


def _validate_layout(root, plan):
    """Reject misplaced evidence rather than silently ignoring another shard's files."""
    study = study_for(plan['stage'])
    expected = {(_job_dir(root, plan, name)/'units'/(study.unit_name(u)+'.npz')).relative_to(root)
                for name, units in jobs(study).items() for u in units}
    for path in (root/'shards').glob('*/jobs/*/units/*.npz'):
        if path.relative_to(root) not in expected:
            raise ValueError('Raw unit is outside its assigned job: '+str(path))
    manifest = dict(stage=study.STAGE, design=study.design_header(), source=plan['source'],
                    dependencies=_runtime_dependencies(plan))
    for name, units in jobs(study).items():
        directory = _job_dir(root, plan, name)
        if (directory/'manifest.json').exists():
            grid._check_recorded(_read_likelihood_report(directory/'manifest.json')[0], manifest,
                                'assigned job manifest')
        for unit in units:
            filename = study.unit_name(unit)+'.npz'
            original = root/'inputs/resume/units'/filename
            derived = directory/'units'/filename
            if original.exists() and derived.exists():
                old, sha = _read_likelihood_report(original)
                new = _read_likelihood_report(derived)[0]
                if new.get('imported_from') != dict(sha256=sha, source=old['source']):
                    raise ValueError('Imported unit lost its original provenance.')
                if not _exact(old['rows'], new['rows']):
                    raise ValueError('Imported completed arrays were changed.')


def _exact(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return left == right


def run(args):
    started = perf_counter()
    cli = _cli()
    cli._rectangular_options(args)
    plan = read_plan(args.run_dir)
    if not 0 <= args.shard < plan['shards']:
        raise ValueError('Shard is outside this plan.')
    if cli._source_snapshot() != plan['source']:
        raise ValueError('Run requires the pinned clean plan source. Do not pull during a run.')
    study, blocks, new_work = study_for(plan['stage']), 0, 0.
    selected = [n for n in jobs(study) if plan['assignment'][n] == args.shard]
    stop_file = args.run_dir/'shards'/str(args.shard)/'STOP'
    with _RunLock(args.run_dir/'shards'/str(args.shard)):
        pending, remaining = [], 0.
        # Preflight all assigned saved raw units/checkpoints, including numerical failures.
        for name in selected:
            adapter = JobStudy(study, name)
            truth = _LazyTruth(adapter.units()[0], study)
            directory = _job_dir(args.run_dir, plan, name)
            check_args = SimpleNamespace(**vars(args))
            check_args.run_dir, check_args.collect_only = directory, True
            summary = cli._run_rectangular_blocks(check_args, plan['source'], _runtime_dependencies(plan),
                                                  truth, study=adapter, invocation_started=started)
            if summary['failed_comparisons'] or (summary['completed'] and not summary['passes']):
                raise ValueError('Saved job failed numerical controls; inspect before continuing.')
            for unit in adapter.units():
                if study.unit_name(unit) in summary['unit_sha256']:
                    continue
                seen = 0
                if summary['partial_unit'] and summary['partial_unit']['unit_name'] == study.unit_name(unit):
                    seen = summary['partial_unit']['completed_hypotheses']
                remaining += (_count(unit, study)-seen)*_cost(unit, study)
            if not summary['completed']:
                pending.append(name)
        def saved(unit, count):
            nonlocal new_work
            new_work += count*_cost(unit, study)
            print(f'  shard {args.shard} total: '+runtime_text(started, new_work, remaining)+
                  '; excludes work assigned to other machines', flush=True)
        for name in pending:
            if stop_file.exists():
                break
            if args.max_hours is not None and perf_counter()-started >= args.max_hours*3600:
                break
            if args.limit_chunks is not None and blocks >= args.limit_chunks:
                break
            child = SimpleNamespace(**vars(args))
            child.run_dir, child.collect_only = _job_dir(args.run_dir, plan, name), False
            child.limit_chunks = None if args.limit_chunks is None else args.limit_chunks-blocks
            adapter = JobStudy(study, name)
            result = cli._run_rectangular_blocks(child, plan['source'], _runtime_dependencies(plan),
                _truth(adapter.units()[0], study), study=adapter, invocation_started=started, on_saved=saved,
                stop_requested=stop_file.exists)
            blocks += result['new_blocks']
        print(f'Shard {args.shard} paused/finished; new blocks={blocks}; '
              f'elapsed={(perf_counter()-started)/3600:.3f}h. Collect all shards for study acceptance.', flush=True)
    return dict(new_blocks=blocks)


def collect(args):
    roots = [args.run_dir] + list(args.from_run or [])
    plan = read_plan(roots[0])
    for root in roots[1:]:
        if read_plan(root) != plan:
            raise ValueError('Cannot combine different plans.')
    study = study_for(plan['stage'])
    if args.output.exists():
        raise FileExistsError('Collection output must be a new directory.')
    with ExitStack() as locks:
        for root in sorted(set(p.resolve() for p in roots)):
            for shard in range(plan['shards']):
                locks.enter_context(_RunLock(root/'shards'/str(shard)))
        selected = {}
        for name, units in jobs(study).items():
            for unit in units:
                filename = study.unit_name(unit)+'.npz'
                for root in roots:
                    path = _job_dir(root, plan, name)/'units'/filename
                    if path.exists():
                        if filename in selected and digest(path) != digest(selected[filename]):
                            raise ValueError('Conflicting copies of completed unit: '+filename)
                        selected[filename] = path
        args.output.mkdir(parents=True)
        manifest = dict(stage=study.STAGE, design=study.design_header(), source=plan['source'],
                        dependencies=_runtime_dependencies(plan))
        _atomic_report(args.output/'manifest.json', manifest)
        for filename, path in selected.items():
            _copy(path, args.output/'units'/filename)
        summary = _validate_flat(args.output, study, manifest['dependencies'])
        summary.update(plan_sha256=plan_hash(plan), dependencies=manifest['dependencies'])
        _atomic_report(args.output/'summary.json', summary)
    print(f'{plan["stage"]}: {summary["completed_units"]}/{summary["expected_units"]} units; '
          f'complete={summary["completed"]}; passes={summary["passes"]}', flush=True)
    if summary['failed_comparisons'] or (summary['completed'] and not summary['passes']):
        raise ValueError('Collected numerical controls failed; evidence retained.')
    return summary


def export(args):
    from src.tools.run_bundle import pack
    plan = read_plan(args.run_dir)
    if not 0 <= args.shard < plan['shards']:
        raise ValueError('Shard is outside this plan.')
    if args.output.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError('Write transfer bundles outside the run directory.')
    with _RunLock(args.run_dir/'shards'/str(args.shard)):
        files = ['plan.json'] + list(plan['inputs'])
        subtree = args.run_dir/'shards'/str(args.shard)
        files += [p.relative_to(args.run_dir).as_posix() for p in subtree.rglob('*')
                  if p.is_file() and p.name in ('manifest.json', 'checkpoint.npz', 'summary.json')]
        files += [p.relative_to(args.run_dir).as_posix() for p in subtree.glob('jobs/*/units/*.npz')]
        result = pack(args.run_dir, files, args.output, part_mib=args.part_mib)
    print(f'Bundle: {args.output}; upload the index and all {len(result["parts"])} parts. '
          'One owner per shard; this is a snapshot, not a second work allocation.', flush=True)
    return result


def unpack(args):
    from src.tools.run_bundle import unpack as extract
    extract(args.bundle, args.run_dir)
    read_plan(args.run_dir)
    print(f'Extracted and raw-checked: {args.run_dir}', flush=True)


def add_commands(parsers):
    prepare_parser = parsers.add_parser('joint-distributed-prepare', help='Freeze whole-job assignments and snapshot prerequisites.')
    prepare_parser.add_argument('--stage', choices=('G2b', 'G2c'), required=True)
    prepare_parser.add_argument('--run-dir', type=Path, required=True)
    prepare_parser.add_argument('--shards', type=int, default=2)
    prepare_parser.add_argument('--g2a-report', type=Path, required=True)
    prepare_parser.add_argument('--g2a-run-dir', type=Path, required=True)
    prepare_parser.add_argument('--g2b-run-dir', type=Path)
    prepare_parser.add_argument('--resume-from', type=Path)
    prepare_parser.set_defaults(handler=prepare)
    runner = parsers.add_parser('joint-distributed-run', help='Resume only one assigned shard, with saved block/time pauses.')
    runner.add_argument('--run-dir', type=Path, required=True)
    runner.add_argument('--shard', type=int, required=True)
    runner.add_argument('--workers', type=int, default=1)
    runner.add_argument('--chunk-size', type=int, default=64)
    runner.add_argument('--max-hours', type=float, default=7.5)
    runner.add_argument('--limit-chunks', type=int)
    runner.add_argument('--quiet', action='store_true')
    runner.set_defaults(handler=run, collect_only=False)
    collector = parsers.add_parser('joint-distributed-collect', help='Rebuild acceptance from complete raw units across identical plans.')
    collector.add_argument('--run-dir', type=Path, required=True)
    collector.add_argument('--from-run', type=Path, action='append')
    collector.add_argument('--output', type=Path, required=True)
    collector.set_defaults(handler=collect)
    exporter = parsers.add_parser('joint-distributed-export', help='Snapshot one shard into checksum-verified upload-sized parts.')
    exporter.add_argument('--run-dir', type=Path, required=True)
    exporter.add_argument('--shard', type=int, required=True)
    exporter.add_argument('--output', type=Path, required=True, help='New bundle index JSON, outside the run directory.')
    exporter.add_argument('--part-mib', type=int, default=24)
    exporter.set_defaults(handler=export)
    importer = parsers.add_parser('joint-distributed-unpack', help='Verify all bundle parts and extract into a new directory.')
    importer.add_argument('--bundle', type=Path, required=True)
    importer.add_argument('--run-dir', type=Path, required=True)
    importer.set_defaults(handler=unpack)
