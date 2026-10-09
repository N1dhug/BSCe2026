"""F4 estimation primitives, synthetic generator and known-parameter FV filter."""

import ast
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, logsumexp
from scipy.stats import norm

from src.params.bm_1 import BM1ObservationSeries
from src.method.bm_1_observation import bayes_update, conservative_mass_transfer, gaussian_log_likelihood, mass_to_log
from src.method.bm_1_estimation import (
    BM1FVFilterStep, BM1FVRhoBankStep, fv_rho_bank_step, run_fv_rho_bank,
    bm1_observation_update, fv_filter_step, mixture_filter_step,
    run_fv_filter, run_mixture_filter,
)
from src.method.beta_mixture import (
    PreparedBetaProjector, beta_cell_average_basis,
    _project_with_warm_start, _projection_matrices, evolve_projected_density,
)
from src.validation import estimation as checks
from src.validation.bm_1 import estimation as synthetic
from src.validation.bm_1 import mixture_estimation as mixture_f4


class ObservationContractTests(unittest.TestCase):
    def test_gaussian_observations_can_be_outside_physical_range(self):
        series = BM1ObservationSeries([0, 1], [-0.2, 1.3], 'day')
        np.testing.assert_array_equal(series.values, [-0.2, 1.3])

    def test_inputs_are_copied_and_read_only(self):
        t, y, mask = np.array([0., 1.]), np.array([.2, .3]), np.array([True, False])
        series = BM1ObservationSeries(t, y, 'hour', mask)
        t[0] = 99
        y[0] = 99
        mask[0] = False
        self.assertEqual(series.times[0], 0)
        self.assertEqual(series.values[0], .2)
        self.assertTrue(series.observed[0])
        for array in (series.times, series.values, series.observed):
            with self.assertRaises(ValueError):
                array[0] = 0

    def test_explicit_missing_observations(self):
        series = BM1ObservationSeries([0, 1], [.3, np.nan], 'hour', [True, False])
        self.assertTrue(np.isnan(series.values[1]))
        with self.assertRaises(ValueError):
            BM1ObservationSeries([0, 1], [.3, np.nan], 'hour')
        with self.assertRaises(ValueError):
            BM1ObservationSeries([0, 1], [.3, np.inf], 'hour', [True, False])

    def test_integer_masks_are_not_silently_coerced(self):
        with self.assertRaises(ValueError):
            BM1ObservationSeries([0, 1], [.3, .4], 'hour', [1, 0])

    def test_invalid_times_or_shapes(self):
        for times in ([], [0, 0], [1, 0], [0, np.inf], [[0, 1]]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                BM1ObservationSeries(times, [.3, .4], 'hour')
        with self.assertRaises(ValueError):
            BM1ObservationSeries([0, 1], [[.3, .4]], 'hour')

    def test_interval_data_rejected_until_model_is_defined(self):
        for kind in ('interval_energy', 'interval_average'):
            with self.assertRaises(NotImplementedError):
                BM1ObservationSeries([0, 1], [.3, .4], 'hour', observation_kind=kind)

    def test_units_and_price_side_are_explicit(self):
        for kwargs in ({'time_unit': 'week'}, {'time_unit': 'day', 'price_side': 'automatic'}):
            with self.assertRaises(ValueError):
                BM1ObservationSeries([0, 1], [.3, .4], **kwargs)
        series = BM1ObservationSeries([0, 1], [.3, .4], 'day', price_side='left')
        self.assertEqual(series.price_side, 'left')

    def test_prefix_excludes_future_measurements(self):
        left = BM1ObservationSeries([0, 1, 2], [.3, .4, .5], 'day').prefix(1)
        right = BM1ObservationSeries([0, 1, 2], [.3, .4, 99], 'day').prefix(1)
        np.testing.assert_array_equal(left.values, right.values)
        np.testing.assert_array_equal(left.times, [0, 1])
        with self.assertRaises(ValueError):
            left.prefix(-1)


class BayesPrimitiveTests(unittest.TestCase):
    def test_discrete_update_matches_direct_bayes(self):
        p, likelihood = np.array([.2, .3, .5]), np.array([4., 2., 1.])
        result = bayes_update(mass_to_log(p), np.log(likelihood))
        np.testing.assert_allclose(result.mass, p * likelihood / (p @ likelihood), atol=1e-15)
        self.assertAlmostEqual(result.log_evidence, np.log(p @ likelihood), places=14)

    def test_constant_likelihood_changes_evidence_not_state(self):
        p = np.array([.2, .3, .5])
        result = bayes_update(mass_to_log(p), np.full(3, np.log(3.7)))
        np.testing.assert_allclose(result.mass, p, atol=1e-15)
        self.assertAlmostEqual(result.log_evidence, np.log(3.7), places=14)

    def test_zero_support_stays_zero(self):
        result = bayes_update(mass_to_log([0, .3, .7]), [100., 0., 0.])
        self.assertEqual(result.mass[0], 0)
        self.assertTrue(np.isneginf(result.log_mass[0]))

    def test_ordinary_likelihood_underflow_does_not_break_update(self):
        result = bayes_update(mass_to_log([.5, .5]), [-10000., -10001.])
        np.testing.assert_allclose(result.mass, [1/(1+np.exp(-1)), 1/(1+np.exp(1))], atol=2e-13)
        self.assertTrue(np.isfinite(result.log_evidence))

    def test_tiny_log_weight_can_recover_on_later_update(self):
        first = bayes_update(mass_to_log([.5, .5]), [-1000., 0.])
        self.assertEqual(first.mass[0], 0.)  # exp(-1000) underflows in float64
        self.assertTrue(np.isfinite(first.log_mass[0]))
        second = bayes_update(first.log_mass, [1000., 0.])
        np.testing.assert_allclose(second.mass, [.5, .5], atol=1e-14)

    def test_common_log_offset_preserves_posterior(self):
        prior = mass_to_log([.1, .3, .6])
        a = bayes_update(prior, [-3., -2., -1.])
        b = bayes_update(prior, np.array([-3., -2., -1.]) - 10000.)
        np.testing.assert_allclose(a.mass, b.mass, atol=5e-13)
        self.assertAlmostEqual(a.log_evidence - 10000., b.log_evidence, places=10)

    def test_no_overlap_is_not_repaired(self):
        with self.assertRaises(FloatingPointError):
            bayes_update(mass_to_log([1., 0.]), [-np.inf, 0.])

    def test_invalid_priors_are_rejected(self):
        for mass in ([-1e-16, 1.], [1., 1.], [0., 0.], [np.nan, 1.], [[.5, .5]], np.array([.5+1j, .5])):
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                mass_to_log(mass)
        for logs in ([-np.inf, -np.inf], [0., 0.], [np.nan, 0.], [np.inf, -np.inf]):
            with self.subTest(logs=logs), self.assertRaises(ValueError):
                bayes_update(logs, [0., 0.])

    def test_invalid_likelihoods_are_rejected(self):
        for ll in ([np.nan, 0.], [np.inf, 0.], [0.], [[0., 0.]], np.array([1j, 0.])):
            with self.subTest(ll=ll), self.assertRaises(ValueError):
                bayes_update(mass_to_log([.5, .5]), ll)

    def test_gaussian_matches_independent_scipy_density(self):
        means = np.array([.1, .5, .9])
        for y in (-.2, .5, 1.3):
            np.testing.assert_allclose(gaussian_log_likelihood(y, means, .08),
                                       norm.logpdf(y, loc=means, scale=.08), atol=1e-13)

    def test_measurement_density_can_exceed_one(self):
        result = bayes_update([0.], gaussian_log_likelihood(.5, [.5], .01))
        self.assertGreater(result.log_evidence, 0.)
        self.assertEqual(result.mass[0], 1.)

    def test_invalid_noise_and_observation_are_rejected(self):
        for sigma in (0., -1., np.inf, np.nan, True, [.1]):
            with self.subTest(sigma=sigma), self.assertRaises((ValueError, TypeError)):
                gaussian_log_likelihood(.5, [.3, .7], sigma)
        with self.assertRaises(ValueError):
            gaussian_log_likelihood(np.nan, [.3, .7], .1)

    def test_recursive_algebra_matches_accumulated_likelihood(self):
        prior = mass_to_log([.1, .2, .7])
        likelihoods = np.array([[-20., -10., -30.], [-11., -18., -8.], [-2., -1., -3.]])
        current, log_z = prior, 0.
        for ll in likelihoods:
            update = bayes_update(current, ll)
            current = update.log_mass
            log_z += update.log_evidence
        expected = prior + likelihoods.sum(axis=0)
        np.testing.assert_allclose(current, expected - logsumexp(expected), atol=1e-13)
        self.assertAlmostEqual(log_z, logsumexp(expected), places=13)

    def test_no_extra_cell_width_in_evidence(self):
        for cells in (21, 101):
            prior = mass_to_log(np.full(cells, 1/cells))
            update = bayes_update(prior, np.full(cells, np.log(2.)))
            self.assertAlmostEqual(update.log_evidence, np.log(2.), places=13)


class ConservativeTransferTests(unittest.TestCase):
    def test_identity_returns_an_independent_copy(self):
        mass = np.array([.2, .8])
        result = conservative_mass_transfer([0, .3, 1], mass, [0, .3, 1])
        np.testing.assert_array_equal(result, mass)
        result[0] = 0
        self.assertEqual(mass[0], .2)

    def test_uniform_density_on_nonuniform_target(self):
        target = np.array([0, .03, .4, .9, 1])
        result = conservative_mass_transfer([0, .2, .5, 1], [.2, .3, .5], target)
        np.testing.assert_allclose(result, np.diff(target), atol=1e-15)

    def test_exact_known_overlap(self):
        result = conservative_mass_transfer([0, .5, 1], [.2, .8], [0, .25, .75, 1])
        np.testing.assert_allclose(result, [.1, .5, .4], atol=1e-15)

    def test_nested_refine_then_coarsen(self):
        source, mass = [0, .5, 1], [.2, .8]
        fine = np.linspace(0, 1, 101)
        refined = conservative_mass_transfer(source, mass, fine)
        restored = conservative_mass_transfer(fine, refined, source)
        np.testing.assert_allclose(restored, mass, atol=2e-15)

    def test_random_nonuniform_grids_conserve_mass(self):
        rng = np.random.default_rng(4030)
        for _ in range(30):
            a = np.r_[0., np.sort(rng.uniform(size=15)), 1.]
            b = np.r_[0., np.sort(rng.uniform(size=23)), 1.]
            mass = rng.dirichlet(np.ones(16))
            result = conservative_mass_transfer(a, mass, b)
            self.assertGreaterEqual(np.min(result), 0.)
            self.assertAlmostEqual(float(np.sum(result)), 1., places=13)

    def test_support_truncation_or_extension_is_rejected(self):
        for target in ([.1, .5, 1], [0, .5, 1.1]):
            with self.assertRaises(ValueError):
                conservative_mass_transfer([0, .5, 1], [.2, .8], target)

    def test_invalid_grids_and_masses(self):
        for source in ([0, 0, 1], [0, np.nan, 1], [1, .5, 0]):
            with self.assertRaises(ValueError):
                conservative_mass_transfer(source, [.2, .8], [0, .5, 1])
        with self.assertRaises(ValueError):
            conservative_mass_transfer([0, .5, 1], [1.], [0, .5, 1])


class IntegratedBasisTests(unittest.TestCase):
    def test_uniform_beta_density_is_one(self):
        np.testing.assert_allclose(beta_cell_average_basis(np.linspace(0, 1, 42), 1., 1.), 1., atol=1e-14)

    def test_polynomial_beta_mass_matches_analytic_integral(self):
        edges = np.linspace(0, 1, 18)
        basis = beta_cell_average_basis(edges, [2, 1], [1, 2])
        np.testing.assert_allclose(basis[0]/17, np.diff(edges**2), atol=1e-15)
        np.testing.assert_allclose(basis[1]/17, -np.diff((1-edges)**2), atol=1e-15)

    def test_narrow_and_boundary_components_preserve_mass(self):
        basis = beta_cell_average_basis(np.linspace(0, 1, 802), [.3, 3., 1200.], [3., .3, 1200.])
        self.assertTrue(np.all(np.isfinite(basis)))
        self.assertGreaterEqual(basis.min(), 0.)
        np.testing.assert_allclose(basis.sum(axis=1)/801, 1., atol=5e-14)

    def test_complementary_cdf_preserves_resolvable_right_tail(self):
        basis = beta_cell_average_basis(np.linspace(0, 1, 101), .5, 8.)
        self.assertGreater(basis[0, -1], 0.)

    def test_continuous_component_variance_separated_from_cell_moments(self):
        for cells in (201, 801):
            edges = np.linspace(0, 1, cells+1)
            x = (edges[1:]+edges[:-1])/2
            mass = beta_cell_average_basis(edges, 20., 20.)[0]/cells
            center_var = float(mass @ (x-.5)**2)
            self.assertLess(abs(center_var-1/164), 2/cells**2)

    def test_invalid_shapes_and_grid_are_rejected(self):
        for a, b in ((0., 1.), (-1., 2.), ([1, 2], [1]), ([np.inf], [1]), (np.array([1+2j]), [1])):
            with self.assertRaises(ValueError):
                beta_cell_average_basis(np.linspace(0, 1, 11), a, b)
        for edges in ([0, .1, 1], [.1, .5, 1], [0, 1, .5], [0, 1]):
            with self.assertRaises(ValueError):
                beta_cell_average_basis(edges, 1., 1.)


class PreparedProjectionTests(unittest.TestCase):
    def setUp(self):
        self.dx = 1/51
        self.edges = np.linspace(0, 1, 52)
        self.basis = beta_cell_average_basis(self.edges, [1, 5, 10], [1, 10, 5])
        self.weights = np.array([.2, .35, .45])
        self.density = self.weights @ self.basis

    def test_public_projection_matches_existing_primitive(self):
        projector = PreparedBetaProjector(self.basis, self.dx)
        actual = projector.project(self.density)
        matrix, hessian, _ = _projection_matrices(self.basis, self.dx)
        _, expected = _project_with_warm_start(self.density, self.basis, matrix, hessian,
                                              self.dx, np.full(3, 1/3))
        np.testing.assert_allclose(actual['density'], expected, atol=1e-14)
        self.assertLess(actual['diagnostics']['density_l2_error'], 1e-8)

    def test_geometry_cached_without_hidden_target_warm_start(self):
        projector = PreparedBetaProjector(self.basis, self.dx)
        first = projector.project(self.density)
        projector.project(self.basis[0])
        with patch('src.method.beta_mixture._projection_matrices', side_effect=AssertionError('reprepare')):
            last = projector.project(self.density)
        np.testing.assert_array_equal(first['density'], last['density'])
        self.assertGreater(projector.projection_solver.cache_hits, 0)

    def test_custom_basis_boundary_diagnostic_is_cell_mass(self):
        p = PreparedBetaProjector(self.basis[::-1], self.dx)
        result = p.project(self.density)
        self.assertAlmostEqual(result['diagnostics']['boundary_cell_mass'],
                               self.dx*(result['density'][0]+result['density'][-1]), places=15)

    def test_invalid_density_or_start_is_rejected(self):
        projector = PreparedBetaProjector(self.basis, self.dx)
        for density in (self.density*2, np.full(51, np.nan), np.r_[-1e-15, self.density[1:]]):
            with self.assertRaises(ValueError):
                projector.project(density)
        with self.assertRaises(ValueError):
            projector.project(self.density, [.5, .5])
        with self.assertRaises(ValueError):
            PreparedBetaProjector(self.basis*2, self.dx)

    def test_owned_basis(self):
        basis = self.basis.copy()
        projector = PreparedBetaProjector(basis, self.dx)
        basis[:] = 0
        np.testing.assert_array_equal(projector.basis, self.basis)
        self.assertFalse(projector.basis.flags.writeable)

    def test_explicit_fallback_preserves_same_problem(self):
        projector = PreparedBetaProjector(self.basis, self.dx)
        with patch.object(projector.projection_solver, 'solve', side_effect=RuntimeError('forced')):
            result = projector.project(self.density)
        self.assertTrue(result['diagnostics']['used_fallback'])
        self.assertLess(result['diagnostics']['density_l2_error'], 1e-7)
        self.assertFalse(result['diagnostics']['used_clipping'])


class PreparationIntegrationTests(unittest.TestCase):
    def test_one_fv_prediction_then_update(self):
        from src.solver.finite_volume import finite_volume_1d
        from src.model.bm_1 import bm_1
        edges, x, dx = finite_volume_1d.grid(101)
        model = bm_1(lambda t: np.asarray(t)*0+.5, lambda t: np.asarray(t)*0, (0, .1), 2)
        params = [lambda t: np.asarray(t)*0+.4, lambda x: 1-2*x, lambda u: 1-2*u, 1., 2., .8]
        prior = beta_cell_average_basis(edges, 20, 20)[0]*dx
        predicted = model.fokker_planc_reference_solver(prior, params, .3,
                     t_eval=[0, .1], max_step=.002)[3][-1]
        demand = model.eval(.1, x, params[0], params[1], params[2], 2., .8)[-1]
        ll = gaussian_log_likelihood(.55, demand, .03)
        update = bayes_update(mass_to_log(predicted), ll)
        direct = predicted*np.exp(ll)
        np.testing.assert_allclose(update.mass, direct/direct.sum(), atol=1e-14)
        self.assertAlmostEqual(update.mass.sum(), 1., places=14)

    def test_prepared_geometry_can_feed_existing_dynamic_propagator(self):
        from src.method.fokker_planc import bm_1_fokker_planc
        edges = np.linspace(0, 1, 52)
        x, dx = (edges[1:]+edges[:-1])/2, 1/51
        basis = beta_cell_average_basis(edges, [1, 20, 80], [1, 20, 80])
        projector = PreparedBetaProjector(basis, dx)
        weights = [.2, .4, .4]
        result = evolve_projected_density(method=bm_1_fokker_planc(), t_eval=[0, .1], breakpoints=None,
            x=x, dx=dx, basis=projector.basis, initial_coefficients=weights,
            drift=lambda t,x: np.zeros_like(x), diffusion=lambda t,x: np.zeros_like(x),
            max_step=.01, cfl=.4, projection_data=projector.projection_data,
            projection_solver=projector.projection_solver)
        np.testing.assert_allclose(result['density'][-1], np.asarray(weights)@basis, atol=1e-12)

    def test_informative_bayes_posterior_below_old_basis_floor(self):
        from src.diagnostics.bm_1 import beta_mixture_variance_floor
        def kernel(x):
            if not 0 < x < 1:
                return 0.
            log_prior = 19*np.log(x)+19*np.log1p(-x)-betaln(20,20)
            demand = .5+.5*np.tanh(2*(.5-x))
            return np.exp(log_prior-.5*((.5-demand)/.01)**2)
        z = quad(kernel, 0, 1, epsabs=1e-12, points=[.5])[0]
        var = quad(lambda x: (x-.5)**2*kernel(x), 0, 1, epsabs=1e-13, points=[.5])[0]/z
        self.assertAlmostEqual(var, .0000986543, delta=1e-10)
        self.assertLess(var, beta_mixture_variance_floor(.5, 38)/50)

    def test_remapped_posterior_can_initialize_f3_contract(self):
        from dataclasses import replace
        from src.validation.bm_1.control import build_f3_a_problem
        from src.solver.finite_volume import finite_volume_1d
        problem, numerics = build_f3_a_problem()
        source = np.linspace(0, 1, 22)
        mass = beta_cell_average_basis(source, 20, 20)[0]/21
        new = replace(problem, initial_mass_builder=lambda e: conservative_mass_transfer(source, mass, e))
        for n in (31, 101):
            actual = new.initial_mass(new.build_model(numerics), n)
            target = finite_volume_1d.grid(n)[0]
            expected = conservative_mass_transfer(source, mass, target)
            np.testing.assert_array_equal(actual, expected)

    def test_new_core_modules_have_no_reverse_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        for path in (
            root/'src/params/bm_1.py', root/'src/method/bm_1_estimation.py',
            root/'src/validation/estimation.py',
        ):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                         else [x.name for x in node.names] if isinstance(node, ast.Import) else [])
                self.assertFalse(any(x.startswith(('src.control', 'src.exe', 'src.validation.bm_1', 'matplotlib'))
                                     for x in names))


class SyntheticGeneratorTests(unittest.TestCase):
    def test_locked_case_matches_f4_a_contract(self):
        case = synthetic.build_f4_b_case()
        np.testing.assert_allclose(case.price_edges, [0., 1/3, 2/3, 1.], atol=0., rtol=0.)
        np.testing.assert_array_equal(case.price_values, [.65, .35, .65])
        np.testing.assert_allclose(case.observation_times, np.arange(1, 25)/24, atol=0., rtol=0.)
        self.assertEqual(case.horizon, (0., 1.))
        self.assertEqual(case.baseline, .5)
        self.assertEqual((case.C, case.k, case.Xi, case.sigma_x, case.sigma_y), (1., 2., 1., .5, .03))
        self.assertEqual((case.initial_seed_root, case.brownian_seed_root, case.measurement_seed_root),
                         (4101, 4102, 4103))

    def test_sequence_is_reproducible_owned_and_truth_is_separate(self):
        left = synthetic.generate_synthetic_sequence(sequence_id=0)
        right = synthetic.generate_synthetic_sequence(sequence_id=0)
        np.testing.assert_array_equal(left.observations.times, right.observations.times)
        np.testing.assert_array_equal(left.observations.values, right.observations.values)
        np.testing.assert_array_equal(left.truth.state, right.truth.state)
        np.testing.assert_array_equal(left.measurement_noise, right.measurement_noise)
        self.assertTrue(left.truth.valid)
        self.assertEqual(left.truth.diagnostics['boundary'], 'none')
        self.assertEqual(left.truth.diagnostics['boundary_corrections'], 0)
        self.assertEqual(left.truth.diagnostics['total_violations'], 0)
        self.assertFalse(left.truth.state.flags.writeable)
        self.assertFalse(left.measurement_noise.flags.writeable)
        self.assertFalse(hasattr(left.observations, 'truth'))

    def test_observations_are_exactly_demand_plus_measurement_noise(self):
        result = synthetic.generate_synthetic_sequence(sequence_id=0)
        np.testing.assert_allclose(
            result.observations.values,
            result.truth.demand[1:] + result.measurement_noise,
            atol=0.0, rtol=0.0,
        )
        self.assertEqual(result.observations.time_unit, 'day')
        self.assertEqual(result.observations.price_side, 'right')
        np.testing.assert_array_equal(result.observations.observed, np.ones(24, dtype=bool))

    def test_price_edges_use_new_right_continuous_price(self):
        result = synthetic.generate_synthetic_sequence(sequence_id=0)
        np.testing.assert_allclose(result.truth.times, np.arange(25)/24, atol=2e-16, rtol=0.)
        self.assertAlmostEqual(result.truth.price[8], .35, places=15)
        self.assertAlmostEqual(result.truth.price[16], .65, places=15)
        self.assertTrue(np.isfinite(result.truth.state[8]))
        self.assertTrue(np.isfinite(result.truth.state[16]))

    def test_rng_streams_are_separate_and_sequence_ids_are_stable(self):
        base = synthetic.build_f4_b_case()
        measurement_changed = synthetic.build_f4_b_case(measurement_seed_root=5103)
        brownian_changed = synthetic.build_f4_b_case(brownian_seed_root=5102)
        a = synthetic.generate_synthetic_sequence(base, sequence_id=3)
        b = synthetic.generate_synthetic_sequence(measurement_changed, sequence_id=3)
        c = synthetic.generate_synthetic_sequence(brownian_changed, sequence_id=3)
        np.testing.assert_array_equal(a.truth.state, b.truth.state)
        self.assertFalse(np.array_equal(a.measurement_noise, b.measurement_noise))
        np.testing.assert_array_equal(a.measurement_noise, c.measurement_noise)
        self.assertFalse(np.array_equal(a.truth.state, c.truth.state))
        lone = synthetic.generate_synthetic_sequence(sequence_id=7)
        after_other_ids = [synthetic.generate_synthetic_sequence(sequence_id=i) for i in range(8)][7]
        np.testing.assert_array_equal(lone.truth.state, after_other_ids.truth.state)
        np.testing.assert_array_equal(lone.observations.values, after_other_ids.observations.values)

    def test_zero_process_noise_removes_brownian_stream_dependence(self):
        a = synthetic.generate_synthetic_sequence(
            synthetic.build_f4_b_case(sigma_x=0., brownian_seed_root=4102), sequence_id=2)
        b = synthetic.generate_synthetic_sequence(
            synthetic.build_f4_b_case(sigma_x=0., brownian_seed_root=9992), sequence_id=2)
        np.testing.assert_array_equal(a.truth.state, b.truth.state)
        np.testing.assert_array_equal(a.observations.values, b.observations.values)

    def test_coupled_refinement_reuses_x0_noise_and_nested_brownian_path(self):
        base = synthetic.generate_synthetic_sequence(sequence_id=0)
        result = synthetic.generate_coupled_refinement(sequence_id=0)
        np.testing.assert_array_equal(base.truth.state, result.coarse.truth.state)
        np.testing.assert_array_equal(base.observations.values, result.coarse.observations.values)
        self.assertEqual(result.coarse.truth.initial_state, result.fine.truth.initial_state)
        np.testing.assert_array_equal(result.coarse.measurement_noise, result.fine.measurement_noise)
        self.assertEqual(result.coarse.truth.diagnostics['n_steps'], 1152)
        self.assertEqual(result.fine.truth.diagnostics['n_steps'], 2304)
        self.assertEqual(result.coarse.truth.diagnostics['total_violations'], 0)
        self.assertEqual(result.fine.truth.diagnostics['total_violations'], 0)
        self.assertLess(result.maximum_abs_state_difference, 0.02)
        self.assertLess(result.maximum_abs_demand_difference, 0.02)

    def test_alternative_baseline_and_flexibility_case_exercises_piecewise_demand(self):
        result = synthetic.generate_synthetic_sequence(
            synthetic.build_f4_b_case(baseline=.3, Xi=.7), sequence_id=0)
        self.assertTrue(np.all(result.truth.demand >= 0.0))
        self.assertTrue(np.all(result.truth.demand <= 1.0))
        self.assertTrue(np.any(result.truth.demand > .3))
        self.assertTrue(np.any(result.truth.demand < .3))

    def test_invalid_case_and_sequence_inputs_fail_explicitly(self):
        for kwargs in (
            {'sigma_y': 0.0}, {'sigma_x': -1.0}, {'Xi': 1.1},
            {'price_values': [.2, 1.2, .4]}, {'observation_times': [0., 1.]},
            {'truth_fine_max_step': 1/1000},
            {'initial_seed_root': 4101, 'brownian_seed_root': 4101},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                synthetic.build_f4_b_case(**kwargs)
        for value in (-1, 1.5, True):
            with self.subTest(sequence_id=value), self.assertRaises((ValueError, TypeError)):
                synthetic.generate_synthetic_sequence(sequence_id=value)


class FVStateFilterTests(unittest.TestCase):
    def setUp(self):
        self.case = synthetic.build_f4_b_case()
        self.sequence = synthetic.generate_synthetic_sequence(self.case, sequence_id=0)
        self.model, self.parameters = self.case.build_model()
        self.initial = synthetic.beta_initial_mass(self.case, self.model, 61)

    def test_beta_initial_mass_preserves_resolvable_tails(self):
        self.assertTrue(np.all(self.initial > 0.0))
        self.assertAlmostEqual(float(np.sum(self.initial)), 1.0, places=14)

    def test_one_step_matches_direct_prediction_and_center_bayes(self):
        time = self.sequence.observations.times[0]
        y = self.sequence.observations.values[0]
        step = fv_filter_step(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, time, y, observed=True, price_side='right',
            breakpoints=self.case.breakpoints, max_step=1/96, likelihood_quadrature_order=1,
        )
        direct = self.model.fokker_planc_reference_solver(
            self.initial, self.parameters, self.case.sigma_x,
            t_eval=[0.0, time], max_step=1/96,
        )
        predicted = direct[3][-1]
        demand = self.model.eval(
            time, step.x_grid, self.parameters[0], self.parameters[1],
            self.parameters[2], self.parameters[4], self.parameters[5],
        )[-1]
        expected = bayes_update(
            mass_to_log(predicted),
            gaussian_log_likelihood(y, demand, self.case.sigma_y),
        )
        np.testing.assert_allclose(step.predicted_mass, predicted, atol=1e-15, rtol=0.)
        np.testing.assert_allclose(step.posterior_mass, expected.mass, atol=1e-14, rtol=0.)
        self.assertAlmostEqual(step.log_evidence, expected.log_evidence, places=13)

    def test_missing_observation_is_prediction_only(self):
        step = fv_filter_step(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, 1/24, np.nan, observed=False, price_side='right',
            breakpoints=self.case.breakpoints, max_step=1/96,
        )
        np.testing.assert_array_equal(step.posterior_mass, step.predicted_mass)
        self.assertTrue(np.isnan(step.log_evidence))

    def test_replay_matches_manual_streaming_kernel(self):
        observations = self.sequence.observations.prefix(4/24)
        replay = run_fv_filter(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, observations,
            breakpoints=self.case.breakpoints, max_step=1/96,
        )
        mass = self.initial
        time = 0.0
        predicted, posterior, evidence = [], [], []
        grids = []
        for t, y in zip(observations.times, observations.values):
            step = fv_filter_step(
                self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
                mass, time, t, y, observed=True, price_side='right',
                breakpoints=self.case.breakpoints, max_step=1/96,
            )
            predicted.append(step.predicted_mass)
            posterior.append(step.posterior_mass)
            evidence.append(step.log_evidence)
            grid = step.diagnostics['integration_t']
            grids.append(grid if not grids else grid[1:])
            mass, time = step.posterior_mass, step.time
        np.testing.assert_array_equal(replay.predicted_mass, np.stack(predicted))
        np.testing.assert_array_equal(replay.posterior_mass, np.stack(posterior))
        np.testing.assert_array_equal(replay.log_evidence, evidence)
        np.testing.assert_array_equal(replay.integration_t, np.concatenate(grids))

    def test_no_assimilation_matches_direct_fp_on_identical_actual_grid(self):
        observations = synthetic.missing_copy(self.sequence.observations.prefix(6/24))
        replay = run_fv_filter(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, observations,
            breakpoints=self.case.breakpoints, max_step=1/96,
        )
        direct_t = np.r_[0.0, observations.times]
        direct = self.model.fokker_planc_reference_solver(
            self.initial, self.parameters, self.case.sigma_x,
            t_eval=direct_t, max_step=1/96,
        )
        np.testing.assert_array_equal(replay.integration_t, direct[-1]['integration_t'])
        np.testing.assert_allclose(replay.posterior_mass, direct[3][1:], atol=2e-14, rtol=0.)

    def test_right_edge_update_uses_new_price(self):
        previous_time, time = 7/24, 8/24
        y = 0.65
        step = fv_filter_step(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, previous_time, time, y, observed=True, price_side='right',
            breakpoints=self.case.breakpoints, max_step=1/192,
        )
        prediction = self.model.fokker_planc_reference_solver(
            self.initial, self.parameters, self.case.sigma_x,
            t_eval=[previous_time, time], max_step=1/192,
        )[3][-1]
        self.assertAlmostEqual(float(self.parameters[0](time)), .35, places=15)
        demand_new = self.model.eval(
            time, step.x_grid, self.parameters[0], self.parameters[1],
            self.parameters[2], self.parameters[4], self.parameters[5],
        )[-1]
        expected = bayes_update(
            mass_to_log(prediction),
            gaussian_log_likelihood(y, demand_new, self.case.sigma_y),
        )
        np.testing.assert_allclose(step.posterior_mass, expected.mass, atol=1e-14, rtol=0.)

    def test_future_observations_cannot_change_earlier_results(self):
        base = self.sequence.observations.prefix(4/24)
        changed_values = base.values.copy()
        changed_values[-2:] += 5.0
        changed = BM1ObservationSeries(base.times, changed_values, 'day', price_side='right')
        left = run_fv_filter(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, base, breakpoints=self.case.breakpoints, max_step=1/96,
        )
        right = run_fv_filter(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, changed, breakpoints=self.case.breakpoints, max_step=1/96,
        )
        np.testing.assert_array_equal(left.posterior_mass[:2], right.posterior_mass[:2])
        np.testing.assert_array_equal(left.predicted_mass[:2], right.predicted_mass[:2])

    def test_control_target_is_not_filter_input(self):
        from src.model.bm_1 import bm_1
        alternative = bm_1(self.model.B, lambda t: np.asarray(t)*0.99, self.model.T, self.model.t_steps)
        observations = self.sequence.observations.prefix(3/24)
        left = run_fv_filter(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, observations, breakpoints=self.case.breakpoints, max_step=1/96,
        )
        right = run_fv_filter(
            alternative, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, observations, breakpoints=self.case.breakpoints, max_step=1/96,
        )
        np.testing.assert_array_equal(left.predicted_mass, right.predicted_mass)
        np.testing.assert_array_equal(left.posterior_mass, right.posterior_mass)

    def test_measurement_quadrature_changes_update_not_fp_prediction(self):
        observations = self.sequence.observations.prefix(2/24)
        center = run_fv_filter(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, observations, breakpoints=self.case.breakpoints,
            max_step=1/96, likelihood_quadrature_order=1,
        )
        gauss = run_fv_filter(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, observations, breakpoints=self.case.breakpoints,
            max_step=1/96, likelihood_quadrature_order=5,
        )
        np.testing.assert_array_equal(center.predicted_mass[0], gauss.predicted_mass[0])
        self.assertFalse(np.array_equal(center.posterior_mass[0], gauss.posterior_mass[0]))
        self.assertGreater(np.max(np.abs(center.posterior_mass[0] - gauss.posterior_mass[0])), 0.0)

    def test_full_reference_preserves_mass_and_nonnegativity(self):
        reference = synthetic.run_f4_c_reference(sequence_id=0, x_steps=101, max_step=1/192)
        filtered = reference.filtered
        self.assertLessEqual(filtered.maximum_abs_mass_error, 1e-10)
        self.assertGreaterEqual(filtered.minimum_cell_mass, 0.0)
        np.testing.assert_allclose(filtered.predicted_mass.sum(axis=1), 1.0, atol=1e-10, rtol=0.)
        np.testing.assert_allclose(filtered.posterior_mass.sum(axis=1), 1.0, atol=1e-10, rtol=0.)

    def test_observation_update_helper_reuses_filter_evidence(self):
        time = self.sequence.observations.times[0]
        value = self.sequence.observations.values[0]
        step = fv_filter_step(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.initial, 0.0, time, value, observed=True, price_side='right',
            breakpoints=self.case.breakpoints, max_step=1/96,
        )
        update = bm1_observation_update(
            self.model, self.parameters, self.case.sigma_y, step.predicted_mass,
            0.0, time, value, price_side='right', likelihood_quadrature_order=1,
        )
        np.testing.assert_array_equal(update.mass, step.posterior_mass)
        self.assertEqual(update.log_evidence, step.log_evidence)

    def test_piecewise_constant_quantile_interpolates_inside_cells(self):
        edges = np.array([0.0, 0.5, 1.0])
        mass = np.array([0.5, 0.5])
        self.assertAlmostEqual(synthetic._piecewise_constant_quantile(edges, mass, .05), .05)
        self.assertAlmostEqual(synthetic._piecewise_constant_quantile(edges, mass, .50), .50)
        self.assertAlmostEqual(synthetic._piecewise_constant_quantile(edges, mass, .95), .95)


    def test_piecewise_constant_variance_includes_within_cell_spread(self):
        edges = np.array([0.0, 0.5, 1.0])
        mass = np.array([0.5, 0.5])
        self.assertAlmostEqual(synthetic._piecewise_constant_variance(edges, mass), 1/12)

    def test_filter_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            fv_filter_step(
                self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
                self.initial, 1/24, 1/24, .5, max_step=1/96,
            )
        with self.assertRaises(ValueError):
            fv_filter_step(
                self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
                self.initial, 0., 1/24, .5, max_step=-1.,
            )
        with self.assertRaises((ValueError, TypeError)):
            fv_filter_step(
                self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
                self.initial, 0., 1/24, .5, max_step=1/96, likelihood_quadrature_order=1.5,
            )


class F4DRepresentationTests(unittest.TestCase):
    def test_d1_target_suite_is_fixed_normalized_and_distinct(self):
        from src.solver.finite_volume import finite_volume_1d
        edges, _, dx = finite_volume_1d.grid(101)
        targets = mixture_f4._target_suite(edges)
        self.assertEqual(
            tuple(targets),
            (
                'broad_center', 'narrow_center', 'shifted_narrow',
                'bimodal', 'left_boundary', 'right_boundary',
            ),
        )
        for target in targets.values():
            self.assertAlmostEqual(float(np.sum(target['cell_mass'])), 1.0, places=12)
            self.assertAlmostEqual(float(np.sum(target['density']) * dx), 1.0, places=12)
            self.assertGreaterEqual(target['variance'], 0.0)
        self.assertLess(targets['left_boundary']['mean'], 0.03)
        self.assertGreater(targets['right_boundary']['mean'], 0.97)
        self.assertLess(targets['narrow_center']['variance'], targets['broad_center']['variance'])

    def test_d1_candidate_dictionaries_are_nested_and_legacy_is_comparison_only(self):
        from src.solver.finite_volume import finite_volume_1d
        edges, _, dx = finite_volume_1d.grid(101)
        dictionaries = mixture_f4.build_d1_dictionaries(edges)
        self.assertEqual(
            tuple(dictionaries),
            (
                'legacy_degree_30', 'legacy_degree_38',
                'multiscale_2', 'multiscale_3', 'multiscale_4',
                'multiscale_4_boundary',
            ),
        )
        self.assertEqual(
            [dictionaries[name]['components'] for name in dictionaries],
            [31, 39, 38, 57, 76, 92],
        )
        self.assertFalse(dictionaries['legacy_degree_30']['selectable'])
        self.assertFalse(dictionaries['legacy_degree_38']['selectable'])
        for name in ('multiscale_2', 'multiscale_3', 'multiscale_4', 'multiscale_4_boundary'):
            dictionary = dictionaries[name]
            self.assertTrue(dictionary['selectable'])
            np.testing.assert_allclose(
                np.sum(dictionary['basis'], axis=1) * dx,
                1.0,
                atol=1e-9,
                rtol=0.0,
            )
        self.assertTrue(
            set(np.round(dictionaries['multiscale_2']['concentrations'], 12))
            < set(np.round(dictionaries['multiscale_3']['concentrations'], 12))
        )
        self.assertTrue(
            set(np.round(dictionaries['multiscale_3']['concentrations'], 12))
            < set(np.round(dictionaries['multiscale_4']['concentrations'], 12))
        )


    def test_d1b_adaptive_dictionaries_are_nested_and_preregistered_sizes(self):
        from src.solver.finite_volume import finite_volume_1d
        edges, _, dx = finite_volume_1d.grid(101)
        dictionaries = mixture_f4.build_d1b_dictionaries(edges)
        self.assertEqual(tuple(dictionaries), ('adaptive_A', 'adaptive_B'))
        self.assertEqual(dictionaries['adaptive_A']['components'], 116)
        self.assertEqual(dictionaries['adaptive_B']['components'], 180)
        for dictionary in dictionaries.values():
            np.testing.assert_allclose(
                np.sum(dictionary['basis'], axis=1) * dx,
                1.0,
                atol=1e-9,
                rtol=0.0,
            )
        pairs_a = set(zip(
            np.round(dictionaries['adaptive_A']['means'], 12),
            np.round(dictionaries['adaptive_A']['concentrations'], 12),
        ))
        pairs_b = set(zip(
            np.round(dictionaries['adaptive_B']['means'], 12),
            np.round(dictionaries['adaptive_B']['concentrations'], 12),
        ))
        self.assertLess(pairs_a, pairs_b)


    def test_d1c_shape_aware_dictionaries_are_nested_and_preregistered_sizes(self):
        from src.solver.finite_volume import finite_volume_1d
        edges, _, dx = finite_volume_1d.grid(101)
        dictionaries = mixture_f4.build_d1c_dictionaries(edges)
        self.assertEqual(tuple(dictionaries), ('shape_C', 'shape_D'))
        self.assertEqual(dictionaries['shape_C']['components'], 120)
        self.assertEqual(dictionaries['shape_D']['components'], 184)
        for dictionary in dictionaries.values():
            np.testing.assert_allclose(
                np.sum(dictionary['basis'], axis=1) * dx,
                1.0,
                atol=1e-9,
                rtol=0.0,
            )
        pairs_c = set(zip(
            np.round(dictionaries['shape_C']['means'], 12),
            np.round(dictionaries['shape_C']['concentrations'], 12),
        ))
        pairs_d = set(zip(
            np.round(dictionaries['shape_D']['means'], 12),
            np.round(dictionaries['shape_D']['concentrations'], 12),
        ))
        self.assertLess(pairs_c, pairs_d)
        self.assertIn((0.02, 50.0), pairs_c)
        self.assertIn((0.98, 50.0), pairs_c)
        self.assertIn((0.5, 800.0), pairs_c)


    def test_d2_holdout_constants_are_frozen_beyond_c_population_ids(self):
        self.assertEqual(mixture_f4.D2_SEQUENCE_START, 1001)
        self.assertEqual(mixture_f4.D2_SEQUENCE_COUNT, 10)
        self.assertEqual(mixture_f4.D2_X_STEPS, 801)
        self.assertEqual(mixture_f4.D2_MAX_STEP, 1/192)
        self.assertEqual(mixture_f4.D2_QUADRATURE_ORDER, 1)
        self.assertEqual(mixture_f4.D2_DICTIONARY, 'shape_D')

    def test_d2_snapshot_summary_requires_both_representation_and_numerics(self):
        good = {
            'status': 'ok',
            'density_l2': .01,
            'cell_mass_l1': .02,
            'cdf_sup': .01,
            'wasserstein_1': .002,
            'mean_error': 1e-4,
            'variance_error': 2e-5,
            'optimality_gap': 1e-12,
            'mass_error': 1e-12,
            'used_fallback': True,
            'active_components': 3,
            'maximum_coefficient': .7,
        }
        summary = mixture_f4._aggregate_snapshot_fits([good, dict(good)])
        self.assertTrue(summary['passes_representation_gate'])
        self.assertEqual(summary['fallback_count'], 2)
        bad = dict(good)
        bad['cell_mass_l1'] = .051
        self.assertFalse(
            mixture_f4._aggregate_snapshot_fits([bad])['passes_representation_gate']
        )
        failed = {
            'status': 'failed',
            'failure_type': 'RuntimeError',
            'failure_message': 'forced',
        }
        failed_summary = mixture_f4._aggregate_snapshot_fits([good, failed])
        self.assertFalse(failed_summary['passes_representation_gate'])
        self.assertEqual(failed_summary['failed_snapshot_indices'], (1,))


    def test_recursive_mixture_replay_matches_manual_streaming(self):
        case = synthetic.build_f4_b_case()
        sequence = synthetic.generate_synthetic_sequence(case, sequence_id=0)
        observations = sequence.observations.prefix(2/24)
        model, parameters = case.build_model()
        edges, _, dx = model.fokker_planc_reference.grid(31, bounds=(0.0, 1.0))
        dictionary = mixture_f4.build_d1c_dictionaries(edges)['shape_D']
        projector = PreparedBetaProjector(dictionary['basis'], dx)
        initial = synthetic.beta_initial_mass(case, model, 31)
        replay = run_mixture_filter(
            model, parameters, case.sigma_x, case.sigma_y,
            initial, 0.0, observations, projector=projector,
            breakpoints=case.breakpoints, max_step=1/24, cfl=.4,
        )
        initial_projection = projector.project(initial / dx)
        coefficients = initial_projection['coefficients']
        previous = 0.0
        predicted = []
        posterior = []
        evidence = []
        for time, value in zip(observations.times, observations.values):
            step = mixture_filter_step(
                model, parameters, case.sigma_x, case.sigma_y,
                coefficients, previous, float(time), float(value),
                projector=projector, breakpoints=case.breakpoints,
                max_step=1/24, cfl=.4,
            )
            predicted.append(step['predicted_mass'])
            posterior.append(step['posterior_mass'])
            evidence.append(step['log_evidence'])
            coefficients = step['posterior_coefficients']
            previous = step['time']
        np.testing.assert_allclose(replay.predicted_mass, np.stack(predicted), atol=2e-10, rtol=0.)
        np.testing.assert_allclose(replay.posterior_mass, np.stack(posterior), atol=2e-10, rtol=0.)
        np.testing.assert_allclose(replay.log_evidence, evidence, atol=2e-10, rtol=0.)

    def test_recursive_mixture_missing_observation_is_prediction_only(self):
        case = synthetic.build_f4_b_case()
        sequence = synthetic.generate_synthetic_sequence(case, sequence_id=0)
        model, parameters = case.build_model()
        edges, _, dx = model.fokker_planc_reference.grid(31, bounds=(0.0, 1.0))
        dictionary = mixture_f4.build_d1c_dictionaries(edges)['shape_D']
        projector = PreparedBetaProjector(dictionary['basis'], dx)
        initial = synthetic.beta_initial_mass(case, model, 31)
        coefficients = projector.project(initial / dx)['coefficients']
        step = mixture_filter_step(
            model, parameters, case.sigma_x, case.sigma_y,
            coefficients, 0.0, 1/24, np.nan,
            projector=projector, observed=False,
            breakpoints=case.breakpoints, max_step=1/24, cfl=.4,
        )
        np.testing.assert_array_equal(step['posterior_mass'], step['predicted_mass'])
        np.testing.assert_array_equal(
            step['posterior_coefficients'], step['predicted_coefficients']
        )
        self.assertTrue(np.isnan(step['log_evidence']))

    def test_small_d3a_sanity_reports_static_dynamic_and_total_gaps(self):
        result = mixture_f4.run_f4_d_recursive_sanity(
            sequence_id=0, x_steps=31, max_step=1/24, cfl=.4,
        )
        self.assertEqual(result['stage'], 'D3a')
        self.assertEqual(result['dictionary'], 'shape_D')
        self.assertEqual(result['sequence_id'], 0)
        for family in ('static_best_fit', 'dynamic_gap', 'total_fv_gap'):
            self.assertEqual(set(result[family]), {'prediction', 'posterior'})
            for item in result[family].values():
                self.assertTrue(np.isfinite(item['maximum_cell_mass_l1']))
                self.assertTrue(np.isfinite(item['maximum_wasserstein_1']))
        self.assertTrue(np.isfinite(
            result['log_evidence_error']['mean_absolute_per_observation']
        ))
        self.assertGreater(result['numerical']['total_internal_steps'], 0)
        self.assertGreaterEqual(
            result['numerical']['prediction_projection_fallback_count'], 0
        )


    def test_d3c_first_order_summary_checks_new_fine_level(self):
        rows = [
            {'requested_max_step': 1/8, 'actual_max_step': 1/8, 'metric': .04},
            {'requested_max_step': 1/16, 'actual_max_step': 1/16, 'metric': .02},
            {'requested_max_step': 1/32, 'actual_max_step': 1/32, 'metric': .01},
        ]
        summary = mixture_f4._d3c_first_order_summary(rows, 'metric')
        self.assertAlmostEqual(summary['extrapolated_limit'], 0.0, places=14)
        self.assertAlmostEqual(summary['predicted_fine'], .01, places=14)
        self.assertAlmostEqual(summary['relative_prediction_error'], 0.0, places=14)
        self.assertTrue(summary['passes'])

    def test_d3c_spatial_summary_requires_successive_stability_and_bound_step(self):
        stable = [
            {'requested_max_step': 1/32, 'actual_max_step': 1/32, 'metric': .0100},
            {'requested_max_step': 1/32, 'actual_max_step': 1/32, 'metric': .0105},
            {'requested_max_step': 1/32, 'actual_max_step': 1/32, 'metric': .0107},
        ]
        summary = mixture_f4._d3c_spatial_summary(stable, 'metric')
        self.assertTrue(summary['passes'])
        unstable = [dict(row) for row in stable]
        unstable[-1]['metric'] = .02
        self.assertFalse(mixture_f4._d3c_spatial_summary(unstable, 'metric')['passes'])
        unbound = [dict(row) for row in stable]
        unbound[-1]['actual_max_step'] = 1/64
        self.assertFalse(mixture_f4._d3c_spatial_summary(unbound, 'metric')['passes'])


    def test_d3d_population_constants_lock_unseen_sequences_and_numerics(self):
        self.assertEqual(mixture_f4.D3D_SEQUENCE_START, 1011)
        self.assertEqual(mixture_f4.D3D_SEQUENCE_COUNT, 200)
        self.assertEqual(mixture_f4.D3C_POPULATION_X_STEPS, 151)
        self.assertEqual(mixture_f4.D3C_POPULATION_MAX_STEP, 1/3072)
        self.assertEqual(mixture_f4.D3D_BOOTSTRAP_REPLICATES, 5000)
        self.assertEqual(mixture_f4.D3D_BOOTSTRAP_SEED, 4301)

    def test_d3d_equivalence_and_gap_summaries_apply_declared_gates(self):
        mixture_mse = np.array([1.00, 1.02, 0.98]) * 0.04
        fv_mse = np.full(3, 0.04)
        rmse = mixture_f4._d3d_rmse_summary(
            mixture_mse, fv_mse, replicates=100, seed=7,
        )
        self.assertTrue(rmse['passes_equivalence'])
        good_l1 = mixture_f4._d3d_gap_summary(
            np.array([0.005, 0.01, 0.015]), l1=True,
        )
        self.assertTrue(good_l1['passes'])
        bad_l1 = mixture_f4._d3d_gap_summary(
            np.array([0.005, 0.01, 0.06]), l1=True,
        )
        self.assertFalse(bad_l1['passes'])
        self.assertTrue(
            mixture_f4._d3d_gap_summary(
                np.array([1e-4, 2e-4, 3e-4]), l1=False,
            )['passes']
        )


class F4EFixedLikelihoodTests(unittest.TestCase):
    def test_candidate_contract_uses_rho_and_rejects_invalid_values(self):
        candidate = synthetic.F4ELikelihoodCandidate(1.25, .8, 2., .4, .03)
        self.assertAlmostEqual(candidate.C, .8)
        self.assertEqual(candidate.as_tuple(), (1.25, .8, 2., .4, .03))
        bad = (
            (0., .8, 2., .4, .03),
            (1., -0.1, 2., .4, .03),
            (1., 1.1, 2., .4, .03),
            (1., .8, 0., .4, .03),
            (1., .8, 2., -0.1, .03),
            (1., .8, 2., .4, 0.),
        )
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                synthetic.F4ELikelihoodCandidate(*values)

    def test_truth_candidate_matches_direct_known_parameter_filter(self):
        case = synthetic.build_f4_b_case()
        sequence = synthetic.generate_synthetic_sequence(case, sequence_id=0)
        observations = sequence.observations.prefix(6/24)
        candidate = synthetic.f4_e_truth_candidate(case)
        result = synthetic.evaluate_f4_e_candidate(
            case, observations, candidate, x_steps=31, max_step=1/24,
        )

        model, parameters = case.build_model()
        initial = synthetic.beta_initial_mass(case, model, 31)
        direct = run_fv_filter(
            model, parameters, case.sigma_x, case.sigma_y, initial,
            case.horizon[0], observations, breakpoints=case.breakpoints,
            max_step=1/24, likelihood_quadrature_order=1,
        )
        np.testing.assert_allclose(
            result['log_evidence'], direct.log_evidence[direct.observed],
            atol=1e-12, rtol=0.,
        )
        self.assertAlmostEqual(
            result['total_log_likelihood'], direct.total_log_likelihood, places=12,
        )

    def test_candidate_order_and_duplicates_do_not_change_likelihood(self):
        case = synthetic.build_f4_b_case()
        observations = synthetic.generate_synthetic_sequence(
            case, sequence_id=1,
        ).observations.prefix(4/24)
        truth = synthetic.f4_e_truth_candidate(case)
        alternative = synthetic.F4ELikelihoodCandidate(
            .8, case.Xi, case.k, case.sigma_x, case.sigma_y,
        )
        first = synthetic.evaluate_f4_e_candidates(
            case, observations, (truth, alternative, truth),
            x_steps=31, max_step=1/24,
        )
        second = synthetic.evaluate_f4_e_candidates(
            case, observations, (alternative, truth),
            x_steps=31, max_step=1/24,
        )
        np.testing.assert_array_equal(first[0]['log_evidence'], first[2]['log_evidence'])
        np.testing.assert_array_equal(first[0]['log_evidence'], second[1]['log_evidence'])
        np.testing.assert_array_equal(first[1]['log_evidence'], second[0]['log_evidence'])
        self.assertEqual(first[0]['total_log_likelihood'], first[2]['total_log_likelihood'])

    def test_xi_zero_makes_rho_k_and_sigma_x_likelihood_flat(self):
        case = synthetic.build_f4_b_case(Xi=0.0)
        observations = synthetic.generate_synthetic_sequence(
            case, sequence_id=2,
        ).observations.prefix(6/24)
        candidates = (
            synthetic.F4ELikelihoodCandidate(.6, 0., 2., .5, case.sigma_y),
            synthetic.F4ELikelihoodCandidate(1.4, 0., 2., .5, case.sigma_y),
            synthetic.F4ELikelihoodCandidate(1., 0., 1., .5, case.sigma_y),
            synthetic.F4ELikelihoodCandidate(1., 0., 3., .5, case.sigma_y),
            synthetic.F4ELikelihoodCandidate(1., 0., 2., .2, case.sigma_y),
            synthetic.F4ELikelihoodCandidate(1., 0., 2., .8, case.sigma_y),
        )
        results = synthetic.evaluate_f4_e_candidates(
            case, observations, candidates, x_steps=31, max_step=1/24,
        )
        totals = np.array([item['total_log_likelihood'] for item in results])
        self.assertLessEqual(float(np.ptp(totals)), 1e-10)
        reference = results[0]['log_evidence']
        for item in results[1:]:
            np.testing.assert_allclose(item['log_evidence'], reference, atol=1e-10, rtol=0.)

    def test_centered_likelihood_has_zero_maximum_and_preserves_order(self):
        evaluations = (
            {'total_log_likelihood': 2.},
            {'total_log_likelihood': -1.},
            {'total_log_likelihood': 1.5},
        )
        centered = synthetic.f4_e_centered_log_likelihood(evaluations)
        np.testing.assert_array_equal(centered, [0., -3., -.5])
        self.assertFalse(centered.flags.writeable)


class F4FRhoBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = synthetic.build_f4_b_case()
        cls.model, cls.parameters = cls.case.build_model()
        cls.initial = synthetic.beta_initial_mass(cls.case, cls.model, 31)
        cls.observations = synthetic.generate_synthetic_sequence(
            cls.case, sequence_id=0).observations.prefix(4/24)
        cls.rho = np.array([.6, 1., 1.4])
        cls.logs = np.log([.2, .5, .3])
        cls.bank = cls.replay(cls.rho, cls.logs)

    @classmethod
    def replay(cls, rho, logs, observations=None):
        return run_fv_rho_bank(
            cls.model, cls.parameters, cls.case.sigma_x, cls.case.sigma_y,
            cls.initial, 0., cls.observations if observations is None else observations,
            rho_values=rho, initial_log_weights=logs, breakpoints=cls.case.breakpoints,
            max_step=1/96, likelihood_quadrature_order=3,
        )

    def test_all_prefixes_match_independent_candidates_and_correct_marginal_weights(self):
        offline = []
        for rho in self.rho:
            parameters = list(self.parameters)
            parameters[3] = 1/rho
            offline.append(run_fv_filter(
                self.model, parameters, self.case.sigma_x, self.case.sigma_y,
                self.initial, 0., self.observations, max_step=1/96,
                likelihood_quadrature_order=3,
            ))
        scores = np.stack([item.log_evidence for item in offline], axis=1)
        cumulative = np.cumsum(scores, axis=0)
        evidence = logsumexp(self.logs + cumulative, axis=1)
        np.testing.assert_allclose(self.bank.log_weights, self.logs + cumulative - evidence[:, None], atol=1e-13)
        np.testing.assert_allclose(np.cumsum(self.bank.log_evidence), evidence, atol=1e-13)
        for k, step in enumerate(self.bank.steps):
            predictions = np.stack([item.predicted_mass[k] for item in offline])
            posteriors = np.stack([item.posterior_mass[k] for item in offline])
            np.testing.assert_allclose(step.conditional_predicted_mass, predictions, atol=1e-14)
            np.testing.assert_allclose(step.conditional_posterior_mass, posteriors, atol=1e-14)
            np.testing.assert_allclose(step.predicted_mass, np.exp(step.prior_log_weights) @ predictions, atol=1e-14)
            np.testing.assert_allclose(step.posterior_mass, step.weights @ posteriors, atol=1e-14)
        self.assertGreater(np.max(np.ptp(self.bank.steps[-1].conditional_posterior_mass, axis=0)), 1e-5)

    def test_permutation_and_repeated_calls_do_not_mix_candidate_states(self):
        order = [2, 0, 1]
        permuted = self.replay(self.rho[order], self.logs[order])
        np.testing.assert_allclose(permuted.log_weights, self.bank.log_weights[:, order], atol=1e-14)
        for left, right in zip(permuted.steps, self.bank.steps):
            np.testing.assert_allclose(left.conditional_posterior_mass, right.conditional_posterior_mass[order], atol=1e-14)
            np.testing.assert_allclose(left.posterior_mass, right.posterior_mass, atol=1e-14)
        np.testing.assert_array_equal(self.replay(self.rho, self.logs).log_weights, self.bank.log_weights)

    def test_identical_hypotheses_preserve_nonuniform_prior(self):
        bank = self.replay([1., 1.], np.log([.6, .4]))
        np.testing.assert_allclose(bank.log_weights, np.tile(np.log([.6, .4]), (4, 1)), atol=1e-14)
        for step in bank.steps:
            np.testing.assert_array_equal(step.conditional_posterior_mass[0], step.conditional_posterior_mass[1])

    def test_missing_measurement_only_propagates_and_does_not_update_weights(self):
        mask = np.array([True, False, True, False])
        values = self.observations.values.copy()
        values[~mask] = np.nan
        observations = BM1ObservationSeries(self.observations.times, values, 'day', mask)
        bank = self.replay(self.rho, self.logs, observations)
        for k in [1, 3]:
            step = bank.steps[k]
            self.assertTrue(np.isnan(step.log_evidence))
            np.testing.assert_array_equal(step.log_weights, bank.steps[k-1].log_weights)
            np.testing.assert_array_equal(step.conditional_predicted_mass, step.conditional_posterior_mass)
            self.assertGreater(np.max(np.abs(step.conditional_posterior_mass - bank.steps[k-1].conditional_posterior_mass)), 1e-5)
        scores = np.cumsum(np.where(mask[:, None], bank.conditional_log_evidence, 0.), axis=0)
        expected = self.logs + scores
        np.testing.assert_allclose(bank.log_weights, expected - logsumexp(expected, axis=1)[:, None], atol=1e-13)
        self.assertAlmostEqual(bank.total_log_likelihood, logsumexp(expected[-1]), places=13)
        missing = self.replay(self.rho, self.logs, synthetic.missing_copy(self.observations))
        np.testing.assert_array_equal(missing.log_weights, np.tile(self.logs, (4, 1)))
        self.assertEqual(missing.total_log_likelihood, 0.)

    def test_future_data_cannot_change_earlier_weights_or_states(self):
        values = self.observations.values.copy()
        values[-1] += 2.
        changed = self.replay(self.rho, self.logs, BM1ObservationSeries(self.observations.times, values, 'day'))
        np.testing.assert_array_equal(changed.log_weights[:-1], self.bank.log_weights[:-1])
        for left, right in zip(changed.steps[:-1], self.bank.steps[:-1]):
            np.testing.assert_array_equal(left.conditional_posterior_mass, right.conditional_posterior_mass)

    def test_step_api_continues_from_conditional_states_and_log_weights(self):
        prior = self.bank.steps[-2]
        step = fv_rho_bank_step(
            self.model, self.parameters, self.case.sigma_x, self.case.sigma_y,
            self.rho, prior.log_weights, prior.conditional_posterior_mass,
            prior.time, self.observations.times[-1], self.observations.values[-1],
            breakpoints=self.case.breakpoints, max_step=1/96, likelihood_quadrature_order=3,
        )
        np.testing.assert_array_equal(step.log_weights, self.bank.steps[-1].log_weights)
        np.testing.assert_array_equal(step.conditional_posterior_mass, self.bank.steps[-1].conditional_posterior_mass)

    @staticmethod
    def given_steps(scores, time):
        return tuple(BM1FVFilterStep(time, [0., .5, 1.], [.25, .75], [.5, .5],
                                   [.5, .5], True, score, {}) for score in scores)

    def test_underflowed_ordinary_weight_recovers_but_prior_zero_stays_zero(self):
        first = BM1FVRhoBankStep([.5, 1.5], np.log([.5, .5]), self.given_steps([-1000., 0.], 1.))
        self.assertEqual(first.weights[0], 0.)
        self.assertTrue(np.isfinite(first.log_weights[0]))
        second = BM1FVRhoBankStep([.5, 1.5], first.log_weights, self.given_steps([1001., 0.], 2.))
        self.assertAlmostEqual(second.weights[0], 1/(1+np.exp(-1.)), places=13)
        zero = BM1FVRhoBankStep([.5, 1.5], [-np.inf, 0.], self.given_steps([1001., 0.], 1.))
        self.assertEqual(zero.weights[0], 0.)
        self.assertTrue(np.isneginf(zero.log_weights[0]))

    def test_invalid_bank_inputs_are_rejected_and_weights_are_owned(self):
        for rho, logs in [([0., 1.], np.log([.5, .5])), ([1., np.nan], np.log([.5, .5])),
                          ([1., 2.], [0., 0.]), ([1., 2.], [0.])]:
            with self.assertRaises(ValueError):
                BM1FVRhoBankStep(rho, logs, self.given_steps([0., 0.], 1.))
        with self.assertRaises(ValueError):
            BM1FVRhoBankStep([1., 2.], np.log([.5, .5]),
                            self.given_steps([0.], 1.) + self.given_steps([0.], 2.))
        rho, logs = np.array([1., 2.]), np.log([.5, .5])
        step = BM1FVRhoBankStep(rho, logs, self.given_steps([0., 0.], 1.))
        rho[0], logs[0] = 99., 99.
        self.assertEqual(step.rho_values[0], 1.)
        for array in (step.rho_values, step.prior_log_weights, step.log_weights):
            self.assertFalse(array.flags.writeable)


class OneParameterBankTests(unittest.TestCase):
    def setUp(self):
        from src.method.bm_1_estimation import run_parameter_bank, parameter_bank_weights
        self.run_bank, self.reduce = run_parameter_bank, parameter_bank_weights
        self.case = synthetic.build_f4_b_case(observation_times=np.arange(1, 4)/24.)
        self.model, self.parameters = self.case.build_model()
        self.observations = synthetic.generate_synthetic_sequence(self.case, sequence_id=80).observations
        self.initial = synthetic.beta_initial_mass(self.case, self.model, 21)
        self.values = np.array([.75, 1.25])
        self.prior = np.log([.3, .7])

    def bank(self, **overrides):
        options = dict(coordinate='rho', values=self.values, initial_log_weights=self.prior,
                       max_step=1/96, likelihood_quadrature_order=3)
        options.update(overrides)
        return self.run_bank(self.model, self.parameters, .5, .03, self.initial, 0,
                             self.observations, **options)

    def projector(self):
        edges, _, dx = self.model.fokker_planc_reference.grid(21, bounds=(0., 1.))
        return PreparedBetaProjector(beta_cell_average_basis(edges, [4., 8., 12.], [12., 8., 4.]), dx)

    def test_general_fv_bank_matches_original_rho_bank(self):
        generic = self.bank()
        old = run_fv_rho_bank(self.model, self.parameters, .5, .03, self.initial, 0,
                             self.observations, rho_values=self.values,
                             initial_log_weights=self.prior, max_step=1/96,
                             likelihood_quadrature_order=3)
        np.testing.assert_array_equal(generic['log_weights'], old.log_weights)
        np.testing.assert_array_equal(generic['posterior_mass'], np.stack([s.posterior_mass for s in old.steps]))

    def test_sigma_bank_and_mixture_match_independent_conditional_prefixes(self):
        for coordinate in ('rho', 'sigma_x'):
            for backend in ('fv', 'mixture'):
                projector = self.projector() if backend == 'mixture' else None
                bank = self.bank(coordinate=coordinate, backend=backend, projector=projector)
                results = []
                for value in self.values:
                    parameters, noise = synthetic.parameter_bank_candidate(self.parameters, .5, coordinate, value)
                    options = dict(max_step=1/96, likelihood_quadrature_order=3)
                    runner = run_fv_filter if backend == 'fv' else run_mixture_filter
                    if projector is not None:
                        options['projector'] = projector
                    results.append(runner(self.model, parameters, noise, .03, self.initial, 0,
                                          self.observations, **options))
                scores = np.stack([r.log_evidence for r in results], axis=1)
                prefix = self.prior + np.cumsum(scores, axis=0)
                np.testing.assert_allclose(bank['log_weights'], prefix-logsumexp(prefix, axis=1)[:, None], atol=1e-12)
                np.testing.assert_allclose(bank['final_conditional_mass'], np.stack([r.posterior_mass[-1] for r in results]), atol=1e-12)
                prediction = np.stack([r.predicted_mass for r in results], axis=1)
                previous = np.vstack((self.prior, bank['log_weights'][:-1]))
                np.testing.assert_allclose(bank['predicted_mass'], np.einsum('kh,khx->kx', np.exp(previous), prediction), atol=1e-12)

    def test_mixture_continuation_keeps_conditional_coefficients_and_weights(self):
        projector = self.projector()
        full = self.bank(backend='mixture', projector=projector)
        prefix_obs = BM1ObservationSeries(self.observations.times[:2], self.observations.values[:2], 'day')
        prefix = self.run_bank(self.model, self.parameters, .5, .03, self.initial, 0, prefix_obs,
                              coordinate='rho', values=self.values, initial_log_weights=self.prior,
                              backend='mixture', projector=projector, max_step=1/96, likelihood_quadrature_order=3)
        tail_obs = BM1ObservationSeries(self.observations.times[2:], self.observations.values[2:], 'day')
        tail = self.run_bank(self.model, self.parameters, .5, .03, self.initial, self.observations.times[1], tail_obs,
                            coordinate='rho', values=self.values, initial_log_weights=prefix['log_weights'][-1],
                            initial_conditional_mass=prefix['final_conditional_mass'],
                            initial_conditional_coefficients=prefix['final_conditional_coefficients'],
                            backend='mixture', projector=projector, max_step=1/96, likelihood_quadrature_order=3)
        np.testing.assert_array_equal(tail['log_weights'][-1], full['log_weights'][-1])
        np.testing.assert_array_equal(tail['final_conditional_coefficients'], full['final_conditional_coefficients'])

    def test_mixture_permutation_duplicate_prior_and_future_causality(self):
        projector = self.projector()
        full = self.bank(backend='mixture', projector=projector)
        permuted = self.bank(backend='mixture', projector=projector, values=self.values[::-1], initial_log_weights=self.prior[::-1])
        np.testing.assert_allclose(full['log_weights'], permuted['log_weights'][:, ::-1], atol=1e-12)
        duplicate = self.bank(values=[1., 1.], backend='mixture', projector=projector)
        np.testing.assert_allclose(duplicate['log_weights'], np.tile(self.prior, (3, 1)), atol=1e-12)
        altered = BM1ObservationSeries(self.observations.times, [*self.observations.values[:2], 99.], 'day')
        changed = self.run_bank(self.model, self.parameters, .5, .03, self.initial, 0, altered,
                                coordinate='rho', values=self.values, initial_log_weights=self.prior,
                                backend='mixture', projector=projector, max_step=1/96, likelihood_quadrature_order=3)
        np.testing.assert_array_equal(full['log_weights'][:2], changed['log_weights'][:2])

    def test_missing_scores_underflow_recovery_and_true_prior_zero(self):
        scores = [[-1000., 0.], [np.nan, np.nan], [2000., 0.]]
        reduced = self.reduce(scores, np.log([.5, .5]), np.array([True, False, True]))
        self.assertEqual(np.exp(reduced['log_weights'][0, 0]), 0.)
        self.assertEqual(np.exp(reduced['log_weights'][-1, 0]), 1.)
        np.testing.assert_array_equal(reduced['log_weights'][0], reduced['log_weights'][1])
        zero = self.reduce(scores, [-np.inf, 0.], np.array([True, False, True]))
        self.assertTrue(np.all(np.isneginf(zero['log_weights'][:, 0])))
        with self.assertRaises(ValueError):
            self.reduce([[0., np.nan]], self.prior, np.array([True]))

    def test_prior_bin_masses_preserve_measure_under_refinement(self):
        nodes = np.array([.5, 1., 1.5])
        uniform = synthetic.f4_f_parameter_prior(nodes, 'uniform_parameter', 'rho')
        capacity = synthetic.f4_f_parameter_prior(nodes, 'uniform_capacity', 'rho')
        np.testing.assert_allclose(uniform, [.25, .5, .25])
        edges = np.array([.5, .75, 1.25, 1.5])
        np.testing.assert_allclose(capacity, -np.diff(1/edges)/(2-2/3))
        self.assertGreater(capacity[0], capacity[-1])
        self.assertTrue(np.all(synthetic.f4_f_parameter_prior(nodes, 'wrong_concentrated', 'rho') > 0))
        with self.assertRaises(ValueError):
            synthetic.f4_f_parameter_prior(nodes, 'uniform_log', 'rho')

    def test_real_spawn_matches_serial_replay_reduction(self):
        group = synthetic._f4_f_group('sigma_x', 1263, .5, [.3, .7],
                                      synthetic._f4_f_numerics(21, 1/24, 1), 'diagnostic')
        serial = synthetic._f4_f_execute_groups([group], 1, keep_mass=True)
        parallel = synthetic._f4_f_execute_groups([group], 2, keep_mass=True)
        for rows in (serial, parallel):
            rows[0].pop('conditional_seconds')
        self.assertEqual(serial, parallel)

    @staticmethod
    def refinement_fixture():
        rows = []
        for coordinate in ('rho', 'sigma_x'):
            for sid in synthetic.F4_F_REFINEMENT_IDS:
                obs = synthetic.generate_synthetic_sequence(sequence_id=sid).observations
                for i, settings in enumerate(synthetic._f4_f_refinement_configs(coordinate)):
                    rows.append(dict(coordinate=coordinate, sequence_id=sid, label=str(i), settings=settings,
                        truth=1. if coordinate == 'rho' else .5, initial_log_weights=[-np.log(41)]*41,
                        values=list(synthetic.F4_E_PARAMETER_GRIDS[coordinate]),
                        observation_times=obs.times.tolist(), observations=obs.values.tolist(),
                        conditional_log_evidence=np.zeros((24,41)).tolist(),
                        log_weights=np.full((24,41), -np.log(41)).tolist(), cumulative_log_evidence=[0.]*24,
                        parameter_mean=[float(np.mean(synthetic.F4_E_PARAMETER_GRIDS[coordinate]))]*24,
                        maximum_abs_mass_error=0., minimum_cell_mass=0., fallbacks=0))
        return dict(**synthetic._f4_f_report_header('F1', 7), rows=rows,
                    source_commit='abc', working_tree_dirty=False)


    def test_rejects_unknown_coordinate_and_bad_mixture_continuation(self):
        with self.assertRaises(ValueError):
            self.bank(coordinate='k')
        with self.assertRaises(ValueError):
            self.bank(values=[0., 1.])
        with self.assertRaises(ValueError):
            self.bank(backend='mixture', projector=self.projector(),
                      initial_conditional_coefficients=np.full((2, 3), 1/3))


    @staticmethod
    def refined_mixture_fixture():
        # A coarse representation can fail against the reference while the
        # explicitly refined candidate and its controls agree exactly.
        configs = [('fv_reference', 401, 1/768, 'fv'),
                   ('fv_matched', 151, 1/3072, 'fv'),
                   ('mixture_coarse', 151, 1/768, 'mixture'),
                   ('mixture_middle', 151, 1/1536, 'mixture'),
                   ('mixture_fine', 151, 1/3072, 'mixture'),
                   ('fv401_fine', 401, 1/32768, 'fv'),
                   ('mixture401_middle', 401, 1/16384, 'mixture'),
                   ('mixture401_fine', 401, 1/32768, 'mixture'),
                   ('mixture801_fine', 801, 1/32768, 'mixture')]
        rows = []
        for sid in (1326, 1327):
            obs = synthetic.generate_synthetic_sequence(sequence_id=sid).observations
            for label, cells, step, backend in configs:
                mass = np.full((24, cells), 1/cells)
                if cells == 151:
                    mass[:] = 0.
                    mass[:, cells//2] = 1.
                rows.append(dict(coordinate='rho', truth=1., sequence_id=sid, label=label,
                    settings=synthetic._f4_f_numerics(cells, step, backend=backend),
                    actual_max_step=step,
                    values=list(synthetic.F4_E_RHO_GRID),
                    observation_times=obs.times.tolist(), observations=obs.values.tolist(),
                    conditional_log_evidence=np.zeros((24,41)).tolist(),
                    initial_log_weights=[-np.log(41)]*41, log_weights=np.full((24,41),-np.log(41)).tolist(),
                    cumulative_log_evidence=[0.]*24, parameter_mean=[1.]*24,
                    maximum_abs_mass_error=0., minimum_cell_mass=0., fallbacks=0,
                    x_edges=np.linspace(0,1,cells+1).tolist(),
                    marginal_predicted_mass=mass.tolist(), marginal_posterior_mass=mass.tolist()))
        return dict(**synthetic._f4_f_report_header('F3', 7), rows=rows,
                    source_commit='abc', working_tree_dirty=False, design_version=2,
                    selected_mixture_numerics=synthetic._f4_f_numerics(401, 1/32768, backend='mixture'),
                    mixture_basis='shape_D', mixture_cfl=.4, passes=False)


    def test_likelihood_window_does_not_remove_hypotheses_from_bayes_update(self):
        a = dict(values=[.5,1.,1.5], observations=[.2],
                 conditional_log_evidence=[[-100.,0.,0.]], log_weights=[[-101.,-np.log(2),-np.log(2)]],
                 cumulative_log_evidence=[0.], parameter_mean=[1.25])
        b = copy.deepcopy(a)
        b['conditional_log_evidence'][0][0] = -90.
        result = synthetic._f4_f_weight_comparison(a,b)
        self.assertEqual(result['maximum_log_odds_error'], 0.)
        self.assertEqual(result['maximum_all_node_log_odds_error'], 10.)
        self.assertEqual(len(a['values']),3)
        updated = self.reduce(b['conditional_log_evidence'], np.log([1/3]*3), np.array([True]))
        self.assertTrue(np.isfinite(updated['log_weights'][0,0]))

    def test_grid_reuse_requires_exact_candidate_case_observations_and_numerics(self):
        settings = synthetic._f4_f_numerics(21,1/24,1)
        a = synthetic._f4_f_group('rho',1263,1.,[.75,1.],settings,'first')
        b = synthetic._f4_f_group('rho',1263,1.,[1.,1.25],settings,'second')
        original = synthetic._f4_f_candidate_replay
        with patch.object(synthetic,'_f4_f_candidate_replay',wraps=original) as run:
            rows = synthetic._f4_f_execute_groups([a,b],1)
            self.assertEqual(run.call_count,3)
        standalone = synthetic._f4_f_execute_groups([b],1)[0]
        np.testing.assert_array_equal(rows[1]['conditional_log_evidence'],standalone['conditional_log_evidence'])
        for change in ('observations','numerics','node'):
            changed = copy.deepcopy(b)
            if change == 'observations':
                obs=changed['observations']
                changed['observations']=BM1ObservationSeries(obs.times,obs.values+.001,'day')
            elif change == 'numerics':
                changed['settings']['likelihood_quadrature_order']=3
            else:
                changed['values']=(np.nextafter(1.,2.),1.25)
            with self.subTest(change=change), patch.object(synthetic,'_f4_f_candidate_replay',wraps=original) as run:
                synthetic._f4_f_execute_groups([a,changed],1)
                self.assertEqual(run.call_count,4)


class JointParameterBankTests(unittest.TestCase):
    """Behavioral tests for both selected pairs and the joint state recursion."""

    @staticmethod
    def prepared():
        case = synthetic.build_f4_b_case(Xi=.75)
        model, parameters = case.build_model()
        initial = synthetic.beta_initial_mass(case, model, 21)
        observations = synthetic.generate_synthetic_sequence(case, sequence_id=0).observations.prefix(3/24)
        return case, model, parameters, initial, observations

    def bank(self, pair, hypotheses, *, prior=None, observations=None, **options):
        from src.method.bm_1_estimation import run_joint_parameter_bank
        case, model, parameters, initial, default_obs = self.prepared()
        return run_joint_parameter_bank(
            model, parameters, case.sigma_x, case.sigma_y, initial, 0,
            default_obs if observations is None else observations, coordinates=pair,
            hypotheses=hypotheses, initial_log_weights=mass_to_log(
                np.full(len(hypotheses), 1/len(hypotheses)) if prior is None else prior),
            breakpoints=case.breakpoints, max_step=1/48, likelihood_quadrature_order=3, **options)

    def test_both_pairs_match_independent_replays_and_arbitrary_joint_prior(self):
        for pair, nodes in ((('rho', 'Xi'), [[.6, .5], [.6, 1], [1.4, .5], [1.4, 1]]),
                            (('sigma_x', 'sigma_y'), [[.2, .02], [.2, .08], [.8, .02], [.8, .08]])):
            prior = np.array([.45, .05, .1, .4])  # deliberately non-product
            bank = self.bank(pair, nodes, prior=prior, keep_conditionals=True)
            case, model, base, initial, obs = self.prepared()
            offline = []
            for values in nodes:
                candidate = replace(synthetic.f4_e_truth_candidate(case), **dict(zip(pair, values)))
                offline.append(run_fv_filter(model, synthetic._f4_e_parameters(base, candidate),
                    candidate.sigma_x, candidate.sigma_y, initial, 0, obs,
                    max_step=1/48, breakpoints=case.breakpoints, likelihood_quadrature_order=3))
            scores = np.column_stack([r.log_evidence for r in offline])
            raw = np.log(prior)+np.cumsum(scores, axis=0)
            evidence = logsumexp(raw, axis=1)
            expected = raw-evidence[:, None]
            with self.subTest(pair=pair):
                np.testing.assert_allclose(bank['conditional_log_evidence'], scores, atol=1e-12, rtol=0)
                np.testing.assert_allclose(bank['log_weights'], expected, atol=1e-12, rtol=0)
                np.testing.assert_allclose(np.cumsum(bank['log_evidence']), evidence, atol=1e-12, rtol=0)
                for k, step in enumerate(bank['steps']):
                    old = prior if k == 0 else np.exp(expected[k-1])
                    np.testing.assert_allclose(step['predicted_mass'], old @ np.array(
                        [r.predicted_mass[k] for r in offline]), atol=1e-12, rtol=0)
                    np.testing.assert_allclose(step['posterior_mass'], np.exp(expected[k]) @ np.array(
                        [r.posterior_mass[k] for r in offline]), atol=1e-12, rtol=0)
                    for j, conditional in enumerate(step['conditional_steps']):
                        np.testing.assert_array_equal(conditional.posterior_mass, offline[j].posterior_mass[k])

    def test_pair_and_hypothesis_order_duplicates_and_repeat_are_invariant(self):
        nodes = np.array([[.5, .5], [1.5, .75], [.5, .5]])
        first = self.bank(('rho', 'Xi'), nodes)
        repeated = self.bank(('rho', 'Xi'), nodes)
        order = [2, 0, 1]
        changed = self.bank(('Xi', 'rho'), nodes[order, ::-1])
        for name in ('conditional_log_evidence', 'log_weights', 'final_conditional_mass'):
            np.testing.assert_array_equal(first[name], repeated[name])
            np.testing.assert_allclose(changed[name], first[name][:, order] if name != 'final_conditional_mass'
                                       else first[name][order], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(first['conditional_log_evidence'][:, 0], first['conditional_log_evidence'][:, 2])
        np.testing.assert_allclose(changed['predicted_mass'], first['predicted_mass'], atol=1e-12, rtol=0)


    def test_zero_prior_remains_zero_and_xi_zero_is_negative_control(self):
        bank = self.bank(('rho', 'Xi'), [[.5, 0], [1.5, 0]], prior=[1, 0])
        self.assertTrue(np.isneginf(bank['log_weights'][:, 1]).all())
        np.testing.assert_allclose(bank['conditional_log_evidence'][:, 0],
                                   bank['conditional_log_evidence'][:, 1], atol=1e-12, rtol=0)

    def test_joint_mixture_matches_one_parameter_bank_when_other_coordinate_is_fixed(self):
        from src.method.bm_1_estimation import run_parameter_bank
        case, model, base, initial, obs = self.prepared()
        projector = synthetic._f4_f_projector(21, model)
        options = dict(backend='mixture', projector=projector)
        joint = self.bank(('rho', 'Xi'), [[.5, .75], [1.5, .75]], **options)
        single = run_parameter_bank(model, base, case.sigma_x, case.sigma_y, initial, 0, obs,
            coordinate='rho', values=[.5, 1.5], initial_log_weights=np.log([.5, .5]),
            max_step=1/48, likelihood_quadrature_order=3, breakpoints=case.breakpoints, **options)
        for name in ('log_weights', 'conditional_log_evidence', 'final_conditional_mass',
                     'final_conditional_coefficients', 'predicted_mass', 'posterior_mass'):
            np.testing.assert_array_equal(joint[name], single[name])
        with self.assertRaises(ValueError):
            self.bank(('rho', 'Xi'), [[.5, .75], [1.5, .75]],
                      initial_conditional_coefficients=np.full((2, 3), 1/3))

    def test_invalid_pairs_nodes_and_shapes_are_rejected_before_propagation(self):
        from src.method.bm_1_estimation import parameter_bank_pair_candidate
        _, _, base, _, _ = self.prepared()
        # Both physically valid zero endpoints are accepted explicitly.
        self.assertEqual(parameter_bank_pair_candidate(base, .5, .03, ('rho', 'Xi'), [1, 0])[0][5], 0)
        self.assertEqual(parameter_bank_pair_candidate(base, .5, .03, ('sigma_y', 'sigma_x'), [.03, 0])[1], 0)
        for pair, values in ((('rho', 'rho'), [[1, 1]]), (('rho', 'k'), [[1, 2]]),
                             (('rho', 'Xi'), [[0, .5]]), (('rho', 'Xi'), [[1, 1.1]]),
                             (('sigma_x', 'sigma_y'), [[-.1, .03]]),
                             (('sigma_x', 'sigma_y'), [[.5, 0]]),
                             (('rho', 'Xi'), [[np.nan, .5]]), (('rho', 'Xi'), [[1+1j, .5]]),
                             (('rho', 'Xi'), [1, .5]), (('rho', 'Xi'), [[[1, .5]]])):
            with self.subTest(pair=pair, values=values), patch(
                    'src.method.bm_1_estimation.fv_filter_step') as propagate:
                with self.assertRaises((ValueError, TypeError)):
                    self.bank(pair, values)
                propagate.assert_not_called()
        with self.assertRaises(ValueError):
            self.bank(('rho', 'Xi'), [[1, .75]], initial_conditional_mass=np.full((2, 21), 1/21))


if __name__ == '__main__':
    unittest.main()
