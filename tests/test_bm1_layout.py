"""BM1 regression tests grouped by responsibility."""

import ast
import numpy as np
import subprocess
import sys
import unittest
from pathlib import Path
from src.control.bm_1 import (DynamicMixturePriceEvaluator, FVStochasticPriceEvaluator, beta_initial_cell_mass,
    multistart_powell, sample_beta_initial_states)
from src.method.beta_mixture import evolve_projected_density
from src.validation.bm_1.cases import build_reference_case
from unittest.mock import patch

class BMReviewCompatibilityTests(unittest.TestCase):

    def test_executable_imports_do_not_run_simulations(self):
        code = "\nfrom unittest.mock import patch\nfrom importlib import import_module\nimport matplotlib.pyplot  # Finish pyplot setup before patching Figure.savefig.\nwith patch('src.model.bm_1.bm_1.euler_maruyama_solver', side_effect=AssertionError('EM at import')), \\\n     patch('src.model.bm_1.bm_1.fokker_planc_reference_solver', side_effect=AssertionError('FV at import')), \\\n     patch('matplotlib.figure.Figure.savefig', side_effect=AssertionError('figure at import')):\n    for name in ('verify', 'optimize', 'refine', 'real_data', 'outputs', 'curves'):\n        import_module('src.exe.run_model.bm_1.' + name)\n"
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')


    def test_sampling_rejects_fractional_path_count(self):
        with self.assertRaises(ValueError):
            sample_beta_initial_states(2.5, seed=0)
        with self.assertRaises(TypeError):
            sample_beta_initial_states(True, seed=0)

    def test_beta_grid_rejects_nonunit_domain(self):
        with self.assertRaises(ValueError):
            beta_initial_cell_mass(np.array([-1e-12, 0.5, 1.0]))

class BMReviewEvaluatorTests(unittest.TestCase):

    def setUp(self):
        self.model, self.parameters = build_reference_case()
        self.edges = np.array([0.0, 0.5, 1.0])
        xedges, _, _ = self.model.fokker_planc_reference.grid(31)
        self.initial = beta_initial_cell_mass(xedges)

    def evaluator(self, **kwargs):
        return FVStochasticPriceEvaluator(self.model, self.parameters, self.edges, 0.5, self.initial,
            objective_max_step=0.1, solver_max_step=0.02, **kwargs)

    def test_evaluator_owns_input_arrays(self):
        evaluator = self.evaluator()
        saved_mass = evaluator.initial_mass.copy()
        self.edges[1] = 0.7
        self.initial[:] = 0.0
        np.testing.assert_array_equal(evaluator.interval_edges, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(evaluator.initial_mass, saved_mass)

    def test_deterministic_midpoint_cost_handles_a_known_jump(self):
        from src.model.bm_1 import bm_1
        from src.control.objective import deterministic_midpoint_objective
        jump = 0.37
        baseline = lambda t: np.where(np.asarray(t) < jump, 0.3, 0.8)
        target = lambda t: np.full_like(np.asarray(t, dtype=float), 0.1)
        model = bm_1(baseline, target, (0.0, 1.0), 11)
        parameters = list(self.parameters)
        parameters[5] = 0.0
        result = deterministic_midpoint_objective(model, [0.5], parameters, objective_max_step=0.1, breakpoints=[jump],
            solver_max_step=0.1)
        exact = jump * (0.3 - 0.1) ** 2 + (1.0 - jump) * (0.8 - 0.1) ** 2
        self.assertAlmostEqual(result.total, exact, places=14)
        self.assertEqual(result.variance, 0.0)

    def test_extra_target_and_baseline_breakpoints_are_respected(self):
        evaluator = self.evaluator(breakpoints=[0.37])
        np.testing.assert_array_equal(evaluator.breakpoints, [0.37, 0.5])
        self.assertIn(0.37, evaluator.objective_edges)
        self.assertFalse(np.any((evaluator.objective_edges[:-1] < 0.37) & (evaluator.objective_edges[1:] > 0.37)))
        with self.assertRaises(ValueError):
            self.evaluator(breakpoints=[np.nan])

    def test_projection_setup_is_reused_between_price_evaluations(self):
        evaluator = DynamicMixturePriceEvaluator(self.evaluator(), degree=10)
        with patch('src.method.beta_mixture._projection_matrices',
            side_effect=AssertionError('repeated fixed Hessian setup')):
            left = evaluator.evaluate([0.7, 0.3])
            right = evaluator.evaluate([0.7, 0.3])
        self.assertEqual(left.objective.total, right.objective.total)
        np.testing.assert_array_equal(left.cell_mass, right.cell_mass)

    def test_mixture_rejects_noninteger_degree(self):
        with self.assertRaises(ValueError):
            DynamicMixturePriceEvaluator(self.evaluator(), degree=10.5)

    def test_exact_price_cache_does_not_change_powell_solution(self):
        calls = [0]

        def objective(prices):
            calls[0] += 1
            return float(np.sum((prices - np.array([0.2, 0.8])) ** 2))
        starts = [[0.1, 0.1], [0.9, 0.9]]
        messages = []
        cached = multistart_powell(objective, starts, progress=messages.append)
        cached_calls = calls[0]
        calls[0] = 0
        uncached = multistart_powell(objective, starts, cache=False)
        np.testing.assert_array_equal(cached.best.x, uncached.best.x)
        self.assertEqual(cached.best.fun, uncached.best.fun)
        self.assertLess(cached_calls, calls[0])
        self.assertEqual(cached.objective_evaluations, cached_calls)
        self.assertGreater(cached.cache_hits, 0)
        self.assertTrue(messages)

    def test_dynamic_core_rejects_invalid_probability(self):
        _, x, dx = self.model.fokker_planc_reference.grid(31)
        with self.assertRaises(ValueError):
            evolve_projected_density(method=self.model.fokker_planc, t_eval=np.array([0.0, 1.0]), breakpoints=None,
                x=x, dx=dx, basis=np.ones((1, x.size)), initial_coefficients=np.array([0.9]), drift=lambda t,
                x: np.zeros_like(x), diffusion=lambda t, x: np.zeros_like(x), max_step=0.1, cfl=0.4)

    def test_raw_mass_loss_is_rejected_before_projection(self):
        _, x, dx = self.model.fokker_planc_reference.grid(31)
        with patch.object(self.model.fokker_planc, '_flux_rhs', return_value=-np.ones_like(x)):
            with self.assertRaisesRegex(RuntimeError, 'probability conservation'):
                evolve_projected_density(method=self.model.fokker_planc, t_eval=np.array([0.0, 1.0]), breakpoints=None,
                    x=x, dx=dx, basis=np.ones((1, x.size)), initial_coefficients=np.array([1.0]), drift=lambda t,
                    x: np.zeros_like(x), diffusion=lambda t, x: np.zeros_like(x), max_step=0.1, cfl=0.4)



class ConsolidatedLayoutTests(unittest.TestCase):
    def test_expected_bm_entrypoints(self):
        root = Path(__file__).resolve().parents[1]
        actual = {p.name for p in (root / 'src/exe/run_model/bm_1').glob('*.py')}
        self.assertEqual(actual, {'verify.py', 'optimize.py', 'refine.py', 'real_data.py', 'estimate.py', 'outputs.py', 'curves.py', 'deterministic.py'})
        for retired in (
            'src/method/bm_1_control.py', 'src/method/bm_1_objective.py',
            'src/method/bm_1_mixture.py', 'src/validation/method/bm_1.py',
        ):
            self.assertFalse((root / retired).exists(), retired)

    def test_reference_studies_have_no_executable_or_plot_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'src/validation/bm_1').glob('*.py'):
            for n in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(n, ast.ImportFrom):
                    self.assertFalse((n.module or '').startswith(('src.exe', 'matplotlib')), str(path))
                elif isinstance(n, ast.Import):
                    self.assertFalse(any(x.name.startswith(('src.exe', 'matplotlib')) for x in n.names), str(path))

    def test_core_control_has_no_reference_scenario_dependency(self):
        root = Path(__file__).resolve().parents[1]
        paths = [root / name for name in (
            'src/control/bm_1.py', 'src/control/bm_1_handoff.py',
            'src/control/objective.py', 'src/method/beta_mixture.py',
            'src/method/bm_1_response.py',
        )]
        self.assertTrue(all(path.is_file() for path in paths))
        for path in paths:
            forbidden = ('src.exe', 'src.validation.bm_1', 'matplotlib')
            if path.name == 'beta_mixture.py':
                forbidden += ('src.control',)
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or '']
                elif isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                else:
                    continue
                self.assertFalse(any(name.startswith(forbidden) for name in modules), str(path))

    def test_reference_cli_help(self):
        root = Path(__file__).resolve().parents[1]
        for module in ('verify', 'optimize', 'refine', 'real_data', 'estimate'):
            result = subprocess.run([sys.executable, '-B', '-m', 'src.exe.run_model.bm_1.' + module, '--help'],
                                    cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.strip())

    def test_data_paths_are_repository_relative_after_move(self):
        from src.tools.bm_1_data import DATA_DIR
        self.assertEqual(DATA_DIR, Path(__file__).resolve().parents[1] / 'files/input/data')

    def test_custom_initial_sampler_is_forwarded_explicitly(self):
        from src.control.bm_1 import evaluate_mc_candidates_common_random_numbers
        from src.validation.bm_1.control import build_f3_a_problem
        from dataclasses import replace
        problem, numerics = build_f3_a_problem()
        numerics = replace(numerics, x_steps=21, objective_max_step=1/12, solver_max_step=1/48)
        calls = []
        def sampler(n, seed):
            calls.append((n, seed))
            return np.full(n, .5)
        evaluator = problem.build_evaluator(numerics)
        result = evaluate_mc_candidates_common_random_numbers(evaluator, {'test': [.72, .28]}, n_paths=8,
                                                              initial_seed=91, brownian_seed=92, initial_sampler=sampler)
        self.assertEqual(calls, [(8, 91)])
        self.assertEqual(result['test'].objective.path_objectives.shape, (8,))

    def test_all_source_modules_parse(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'src').rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_water_model_imports_still_work(self):
        for module in ('wm_1', 'wm_2', 'wm_3', 'wm_4'):
            self.assertIsNotNone(__import__('importlib').import_module('src.model.' + module))


class ReportingTests(unittest.TestCase):
    """Reporting flags must not change the model or create unexpected output files."""

    def test_objective_report_is_short_by_default_and_plotting_is_opt_in(self):
        import contextlib
        import io
        from src.exe.run_model.bm_1 import verify
        from src.validation.bm_1.mixture import run_objective_reference

        result = run_objective_reference(n_paths=32, degrees=(4,), x_steps=21,
                                         objective_max_step=0.25, solver_max_step=0.125)
        outputs = []
        with patch.object(verify.mixture, 'run_objective_reference', return_value=result), \
             patch.object(verify, '_save_lines', side_effect=AssertionError('implicit plot')):
            for options in ([], ['--verbose']):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    verify.main(['objective', *options])
                outputs.append(stream.getvalue())
        self.assertIn('MC SE=', outputs[0])
        self.assertNotIn('pointwise_decomposition', outputs[0])
        self.assertIn('pointwise_decomposition', outputs[1])
        self.assertLess(len(outputs[0].splitlines()), 12)

    def test_default_output_paths_cannot_overwrite_retained_evidence(self):
        from src.exe.run_model.bm_1 import verify, optimize, refine, real_data, estimate
        root = Path(__file__).resolve().parents[1]
        runs = root / 'files/output/bm_1/runs'
        for module in (verify, optimize, refine, real_data, estimate):
            with self.subTest(module=module.__name__):
                self.assertTrue(module.OUTPUT_DIR.is_relative_to(runs))
        self.assertIsNone(real_data._plot_location(None))
        path = root / 'files/output/bm_1/real_data/example.png'
        self.assertEqual(real_data._plot_location(path), 'files/output/bm_1/real_data/example.png')

    def test_refinement_save_is_explicit_and_uses_selected_directory(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from src.exe.run_model.bm_1 import refine

        evaluation = SimpleNamespace(prices=np.array([0.7, 0.3]),
            objective=SimpleNamespace(total=0.2, bias=0.1, variance=0.1),
            diagnostics={'n_steps': 1, 'maximum_abs_mass_error': 0.0, 'minimum_cell_mass': 0.1})
        row = {'candidate': 'reference', 'level': 2.0, 'total': 0.2}
        result = {'candidates': {'reference': np.array([0.7, 0.3])},
                  'reference_fv': {'reference': evaluation}, 'summary': {}, 'mc': None, 'mc_time': None,
                  'spatial': [row], 'fv_time': [row], 'quadrature': [row]}
        with tempfile.TemporaryDirectory() as folder:
            path = refine.save_json(result, folder)
            self.assertEqual(path.parent, Path(folder))
            self.assertEqual(json.loads(path.read_text())['reference_fv']['reference']['total'], 0.2)
            figure = refine.save_plot(result, folder)
            self.assertGreater(figure.stat().st_size, 0)

    def test_reference_evaluator_validates_each_candidate_once(self):
        from src.control import bm_1 as control
        from src.validation import control as checks
        model, parameters = build_reference_case()
        edges, _, _ = model.fokker_planc_reference.grid(21)
        mass = beta_initial_cell_mass(edges)
        for cls in (control.FVStochasticPriceEvaluator, control.FastFVStochasticPriceEvaluator):
            evaluator = cls(model, parameters, [0, 0.5, 1], 0.5, mass,
                            objective_max_step=0.25, solver_max_step=0.125)
            with self.subTest(backend=cls.__name__), patch.object(checks, 'validate_prices',
                                                                 wraps=checks.validate_prices) as validate:
                evaluator.evaluate([0.7, 0.3])
                self.assertEqual(validate.call_count, 1)


    def test_estimate_synthetic_writes_only_with_explicit_save(self):
        import contextlib
        import io
        import tempfile
        from src.exe.run_model.bm_1 import estimate
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'estimation'
            with patch.object(estimate, 'OUTPUT_DIR', output):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    result = estimate.main(['synthetic', '--sequence-id', '0'])
                self.assertFalse(output.exists())
                self.assertIn('raw boundary violations=0', stream.getvalue())
                self.assertTrue(result.truth.valid)
                with contextlib.redirect_stdout(io.StringIO()):
                    estimate.main(['synthetic', '--sequence-id', '0', '--save'])
            saved = list(output.glob('*.json'))
            self.assertEqual(len(saved), 1)
            self.assertGreater(saved[0].stat().st_size, 0)


class F4DocumentationBuildTests(unittest.TestCase):
    """Build-tool and documentation regressions; no installed LaTeX required."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / 'root/f4/build.py'
        spec = importlib.util.spec_from_file_location('f4_note_builder', path)
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def test_import_and_help_have_no_build_side_effect(self):
        with patch.object(self.builder.subprocess, 'run', side_effect=AssertionError('compile')):
            self.assertTrue(callable(self.builder.main))
            with self.assertRaises(SystemExit) as result:
                with patch('sys.stdout'):
                    self.builder.main(['--help'])
            self.assertEqual(result.exception.code, 0)

    def test_missing_engine_gives_one_actionable_error(self):
        with patch.object(self.builder.shutil, 'which', return_value=None), \
             patch.dict(self.builder.os.environ, {}, clear=True):
            with self.assertRaisesRegex(self.builder.BuildError, 'GitHub PR'):
                self.builder.find_engine()

    def test_miktex_can_be_found_without_changing_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder) / 'Programs/MiKTeX/miktex/bin/x64/pdflatex.exe'
            exe.parent.mkdir(parents=True)
            exe.touch()
            with patch.object(self.builder.shutil, 'which', return_value=None), \
                 patch.dict(self.builder.os.environ, {'LOCALAPPDATA': folder, 'PATH': 'unchanged'}, clear=True):
                self.assertEqual(self.builder.find_engine(), str(exe))
                self.assertEqual(self.builder.os.environ['PATH'], 'unchanged')

    def _fake_build(self, folder, *, failure=False, unresolved=False, duplicate=False):
        from types import SimpleNamespace
        directory = Path(folder)
        for name in ('main.tex', 'notes.tex'):
            (directory / name).write_text('test', encoding='utf-8')
        calls = []
        def run(command, **kwargs):
            if command[-1] == '--version':
                return SimpleNamespace(returncode=0, stdout='pdfTeX', stderr='')
            self.assertEqual(kwargs['cwd'], directory.resolve())
            calls.append(command)
            build = directory / 'build'
            (build / 'main.pdf').write_bytes(b'%PDF-stub')
            (build / 'main.aux').write_text('stable', encoding='utf-8')
            message = 'There were undefined references.' if unresolved else ''
            if duplicate:
                message = 'destination with the same identifier (name{page.1})'
            return SimpleNamespace(returncode=int(failure), stdout=message, stderr='')
        return run, calls

    def test_two_passes_stabilize_without_changing_working_directory(self):
        import tempfile
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            run, calls = self._fake_build(folder)
            with patch.object(self.builder.subprocess, 'run', side_effect=run), patch('sys.stdout'):
                pdf = self.builder.build_notes(Path(folder), 'pdflatex')
            self.assertTrue(pdf.is_file())
            self.assertEqual(len(calls), 2)
            self.assertIn('-no-shell-escape', calls[0])
        self.assertEqual(Path.cwd(), original)

    def test_failed_pass_stops_and_removes_partial_pdf(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            run, calls = self._fake_build(folder, failure=True)
            with patch.object(self.builder.subprocess, 'run', side_effect=run):
                with self.assertRaisesRegex(self.builder.BuildError, 'pass 1'):
                    self.builder.build_notes(Path(folder), 'pdflatex')
            self.assertEqual(len(calls), 1)
            self.assertFalse((Path(folder) / 'build/main.pdf').exists())

    def test_undefined_references_fail_after_bounded_retries(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            run, calls = self._fake_build(folder, unresolved=True)
            with patch.object(self.builder.subprocess, 'run', side_effect=run):
                with self.assertRaisesRegex(self.builder.BuildError, 'did not stabilize'):
                    self.builder.build_notes(Path(folder), 'pdflatex', max_runs=3)
            self.assertEqual(len(calls), 3)
            self.assertFalse((Path(folder) / 'build/main.pdf').exists())

    def test_duplicate_anchors_fail(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            run, _ = self._fake_build(folder, duplicate=True)
            with patch.object(self.builder.subprocess, 'run', side_effect=run):
                with self.assertRaisesRegex(self.builder.BuildError, 'Duplicate'):
                    self.builder.build_notes(Path(folder), 'pdflatex')

    def test_active_document_links_and_tex_inputs_resolve(self):
        import re
        root = Path(__file__).resolve().parents[1] / 'root'
        for name in ('BM_GUIDE.md', 'BM_REFERENCE.md', 'F4.md'):
            path = root / name
            for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)', path.read_text(encoding='utf-8')):
                if not link.startswith(('http:', 'https:')):
                    self.assertTrue((path.parent / link).is_file(), link)
        for path in (root / 'f4').glob('*.tex'):
            for target in re.findall(r'\\input\{([^}]+)\}', path.read_text(encoding='utf-8')):
                self.assertTrue((path.parent / target).is_file(), target)
        self.assertFalse((root / 'f4/README.md').exists())
        self.assertNotIn('Afventer F4-A', (root / 'F4.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
