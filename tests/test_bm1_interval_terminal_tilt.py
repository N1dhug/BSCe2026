"""Small independent references for the F4-H2 terminal-contour prototype."""

import unittest
import gzip
import json
import hashlib
import os
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, roots_legendre

from src.method.bm_1_observation import gaussian_log_likelihood
from src.solver.finite_volume import finite_volume_1d_fitted
from src.validation.bm_1.interval_characteristic import CharacteristicNumericalError
from src.validation.bm_1.interval_terminal_tilt import terminal_tilt_update


class TerminalTiltTests(unittest.TestCase):
    def test_cross_reuse_rejects_changed_evidence_inputs_or_numerical_source(self):
        from src.exe.run_model.bm_1_interval_terminal_tilt import _frozen_terminal_reference
        raw = Path('files/output/bm_1/runs/estimation/f4_h2_terminal_tilt_day_20261005.json.gz').read_bytes()
        saved = json.loads(gzip.decompress(raw))
        args = (saved['inputs_sha256'], saved['source'], saved['design'])
        # Current relocated code must remain ineligible for historical byte-exact
        # reuse. Positive coverage runs against an explicitly restored original.
        with self.assertRaisesRegex(ValueError, 'update kernel changed'):
            _frozen_terminal_reference(raw, *args)
        root = Path.cwd()
        relative = 'src/validation/bm_1/interval_terminal_tilt.py'
        archive = root/'archive/bm/originals/source-7249bf9.zip'
        manifest = json.loads((archive.with_suffix('.json')).read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), manifest['archive_sha256'])
        with zipfile.ZipFile(archive) as saved_source:
            original = saved_source.read(relative)
        self.assertEqual(hashlib.sha256(original).hexdigest(), manifest['files'][relative]['sha256'])
        # The preserved Windows bytes have CRLF; PR #81's exact guard binds the
        # LF Git representation. Verify both identities, without changing it.
        original = original.replace(b'\r\n', b'\n')
        self.assertEqual(hashlib.sha256(original.split(b'\ndef terminal_reference_control():')[0]).hexdigest(),
            '351b08fa915b125b9df8b08bab031d8c1c452aa231a94d67831989bf125e8292')
        with tempfile.TemporaryDirectory() as temporary:
            restored = Path(temporary)/relative
            restored.parent.mkdir(parents=True)
            restored.write_bytes(original)
            try:
                os.chdir(temporary)
                reused, proof = _frozen_terminal_reference(raw, *args)
                self.assertEqual(reused['comparison'], saved['comparison'])
                self.assertIn('src/solver/finite_volume.py', proof['unchanged_dependency_paths'])
                windows_inputs = {p.replace('/', '\\'): sha for p, sha in args[0].items()}
                _frozen_terminal_reference(raw, windows_inputs, *args[1:])
                with self.assertRaisesRegex(ValueError, 'archived'):
                    _frozen_terminal_reference(gzip.decompress(raw)+b' ', *args)
                with self.assertRaisesRegex(ValueError, 'inputs changed'):
                    _frozen_terminal_reference(raw, {}, *args[1:])
                source = deepcopy(saved['source'])
                source['sha256']['src/solver/finite_volume.py'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'dependencies changed'):
                    _frozen_terminal_reference(raw, args[0], source, args[2])
                source = deepcopy(saved['source'])
                source['sha256']['src/model/wm_4.py'] = 'unrelated WM change'
                source['sha256']['src/exe/run_model/bm_1/deterministic.py'] = 'unrelated G change'
                _frozen_terminal_reference(raw, args[0], source, args[2])
            finally:
                os.chdir(root)
        with self.assertRaisesRegex(ValueError, 'update kernel changed'):
            _frozen_terminal_reference(raw, *args)

    def test_exact_gaussian_tails_and_recursive_revival_below_float_range(self):
        solver = finite_volume_1d_fitted()
        zero = lambda t, x: np.zeros(x.shape)
        rewards = np.array([.05, .5, .95])
        for initial, values in ((np.log([.2, .5, .3]), [.6]),
                                (np.array([0., -900., -950.]), [1.5, 1.5]),
                                (np.array([0., -np.inf, -np.inf]), [.6])):
            logs, exact = initial.copy(), initial.copy()
            for value in values:
                result = terminal_tilt_update(solver, [0., 1.], np.exp(logs), zero, zero,
                    lambda t, x: rewards, value, .04, log_prior=logs, max_step=1/8)
                exact += gaussian_log_likelihood(value, rewards, .04)
                expected_score = logsumexp(exact)
                exact -= expected_score
                self.assertAlmostEqual(result['log_evidence'], expected_score, delta=2e-8)
                np.testing.assert_allclose(result['log_posterior_mass'], exact, rtol=0., atol=2e-8)
                logs = result['log_posterior_mass']
                self.assertLessEqual(result['maximum_relative_conditional_guard'], 1e-8)
            if len(values) == 2:
                self.assertGreater(result['posterior_mass'][2], .99)

    def test_absorbing_jump_matches_positive_path_integral_and_time_refines(self):
        # A single jump time has positive density rate*exp(-rate*t).
        # This quadrature has no Fourier inversion and no FV time stepping.
        solver, rate, value, sigma = finite_volume_1d_fitted(), .8, .55, .04
        rewards = np.array([.1, .9, .9])  # Third state is disconnected.
        references = []
        for order in (96, 160):
            nodes, weights = roots_legendre(order)
            times = (nodes + 1.) / 2.
            average = rewards[1] + (rewards[0] - rewards[1]) * times
            jumped = logsumexp(np.log(weights / 2. * rate) - rate * times
                              + gaussian_log_likelihood(value, average, sigma))
            references.append(np.array([-rate + gaussian_log_likelihood(value, rewards[:1], sigma)[0], jumped]))
        np.testing.assert_allclose(references[0], references[1], rtol=0., atol=1e-12)
        errors = []
        for resolution in (128, 256):
            result = terminal_tilt_update(solver, [0., 1.], [1., 0., 0.],
                lambda t, x: np.where(x < .5, rate / 3., 0.), lambda t, x: np.zeros(x.shape),
                lambda t, x: rewards, value, sigma, max_step=1/resolution)
            errors.append(float(np.max(np.abs(result['log_joint_mass'][:2] - references[-1]))))
            self.assertTrue(np.isneginf(result['log_joint_mass'][2]))
        self.assertLess(errors[-1], 2e-4)
        self.assertGreater(errors[0] / errors[1], 3.5)

    def test_insufficient_bank_and_frequency_budget_fail_without_prior_mutation(self):
        solver = finite_volume_1d_fitted()
        prior = np.array([.2, .5, .3])
        original = prior.copy()
        zero = lambda t, x: np.zeros(x.shape)
        for kwargs in (dict(centers=(.5,)), dict(frequencies=1)):
            with self.assertRaisesRegex(CharacteristicNumericalError, 'bank exhausted'):
                terminal_tilt_update(solver, [0., 1.], prior, zero, zero,
                    lambda t, x: np.array([.05, .5, .95]), .6, .04, max_step=1/8, **kwargs)
        np.testing.assert_array_equal(prior, original)


if __name__ == '__main__':
    unittest.main()
