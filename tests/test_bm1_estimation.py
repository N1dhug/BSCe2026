"""F4-0 primitives only: not evidence that a recursive F4 filter is complete."""

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, logsumexp
from scipy.stats import norm

from src.params.bm_1 import (
    BM1ObservationSeries, bayes_update, conservative_mass_transfer,
    gaussian_log_likelihood, mass_to_log,
)
from src.method.beta_mixture import (
    PreparedBetaProjector, beta_cell_average_basis,
    _project_with_warm_start, _projection_matrices, evolve_projected_density,
)
from src.validation import estimation as checks
from src.validation.bm_1 import estimation as synthetic


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
        for path in (root/'src/params/bm_1.py', root/'src/validation/estimation.py'):
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
        self.assertAlmostEqual(result.truth.state[8], result.truth.state[8], places=15)

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
        result = synthetic.generate_coupled_refinement(sequence_id=0)
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


if __name__ == '__main__':
    unittest.main()
