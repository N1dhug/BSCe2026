"""F4 estimation primitives, synthetic generator and known-parameter FV filter."""

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
from src.method.bm_1_estimation import (
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

    def test_small_population_study_is_paired_and_reproducible(self):
        kwargs = dict(
            n_sequences=3, sequence_start=1, x_steps=31, max_step=1/24,
            likelihood_quadrature_order=1, bootstrap_replicates=100, bootstrap_seed=4321,
        )
        left = synthetic.run_f4_c_population(**kwargs)
        right = synthetic.run_f4_c_population(**kwargs)
        np.testing.assert_array_equal(left['sequence_ids'], [1, 2, 3])
        self.assertEqual(left['rmse'], right['rmse'])
        self.assertEqual(left['predictive_log_score'], right['predictive_log_score'])
        self.assertEqual(left['coverage90_posterior'], right['coverage90_posterior'])
        self.assertLess(left['predictive_log_score']['first_observation_max_abs_difference'], 1e-14)
        self.assertEqual(left['diagnostics']['invalid_truth_sequences'], ())
        for hour in (6, 12, 24):
            item = left['coverage90_posterior'][hour]
            self.assertEqual(item['total'], 3)
            self.assertLessEqual(item['successes'], 3)
            self.assertGreaterEqual(item['mean_width'], 0.0)

    def test_refinement_changes_one_numerical_dimension_at_a_time(self):
        result = synthetic.run_f4_c_refinement(
            sequence_id=0,
            spatial_cells=(31, 61),
            time_steps=(1/24, 1/48),
            quadrature_orders=(1, 3),
        )
        self.assertEqual([row['x_steps'] for row in result['spatial']], [31, 61])
        self.assertTrue(all(row['requested_max_step'] == 1/192 for row in result['spatial']))
        self.assertTrue(all(row['quadrature_order'] == 1 for row in result['spatial']))
        self.assertTrue(all(row['x_steps'] == 201 for row in result['time']))
        self.assertEqual([row['requested_max_step'] for row in result['time']], [1/24, 1/48])
        self.assertTrue(all(row['quadrature_order'] == 1 for row in result['time']))
        self.assertTrue(all(row['x_steps'] == 201 for row in result['quadrature']))
        self.assertTrue(all(row['requested_max_step'] == 1/192 for row in result['quadrature']))
        self.assertEqual([row['quadrature_order'] for row in result['quadrature']], [1, 3])
        for axis in ('spatial', 'time', 'quadrature'):
            self.assertTrue(np.isfinite(result['summary'][axis]['maximum_posterior_rmse_effect']))
            self.assertTrue(np.isfinite(result['summary'][axis]['maximum_log_likelihood_effect']))

    def test_refinement_requires_ordered_unique_levels(self):
        with self.assertRaises(ValueError):
            synthetic.run_f4_c_refinement(spatial_cells=(61, 31))
        with self.assertRaises(ValueError):
            synthetic.run_f4_c_refinement(time_steps=(1/48, 1/24))
        with self.assertRaises(ValueError):
            synthetic.run_f4_c_refinement(quadrature_orders=(1, 1))

    def test_calibration_sensitivity_reuses_fixed_sequences(self):
        result = synthetic.run_f4_c_calibration_sensitivity(
            n_sequences=3,
            sequence_start=1,
            variants=(
                ('base', 31, 1/24, 1),
                ('space', 61, 1/24, 1),
                ('time', 31, 1/48, 1),
                ('quad', 31, 1/24, 3),
            ),
        )
        np.testing.assert_array_equal(result['sequence_ids'], [1, 2, 3])
        self.assertEqual(result['invalid_truth_sequences'], ())
        self.assertEqual(set(result['variants']), {'base', 'space', 'time', 'quad'})
        for variant in result['variants'].values():
            self.assertEqual(set(variant['horizons']), {6, 12, 24})
            for item in variant['horizons'].values():
                self.assertEqual(item['total'], 3)
                self.assertGreaterEqual(item['coverage'], 0.0)
                self.assertLessEqual(item['coverage'], 1.0)
                self.assertGreater(item['mean_width'], 0.0)
                self.assertGreaterEqual(item['posterior_mean_rmse'], 0.0)
                self.assertGreater(item['root_mean_posterior_variance'], 0.0)
                self.assertTrue(np.isfinite(item['rmse_over_posterior_sd']))

    def test_piecewise_constant_variance_includes_within_cell_spread(self):
        edges = np.array([0.0, 0.5, 1.0])
        mass = np.array([0.5, 0.5])
        self.assertAlmostEqual(synthetic._piecewise_constant_variance(edges, mass), 1/12)

    def test_truth_refinement_uses_coupled_sequence_pairs(self):
        result = synthetic.run_f4_c_truth_refinement(
            n_sequences=2, sequence_start=1, x_steps=31, max_step=1/24,
            likelihood_quadrature_order=1,
        )
        np.testing.assert_array_equal(result['sequence_ids'], [1, 2])
        self.assertEqual(result['invalid_truth_pairs'], ())
        self.assertEqual(set(result['coarse']['horizons']), {6, 12, 24})
        self.assertEqual(set(result['fine']['horizons']), {6, 12, 24})
        self.assertGreaterEqual(result['truth_difference']['maximum_abs_state'], 0.0)
        self.assertGreaterEqual(result['truth_difference']['rms_state'], 0.0)
        self.assertGreaterEqual(result['truth_difference']['maximum_abs_demand'], 0.0)
        self.assertAlmostEqual(
            result['truth_difference']['maximum_abs_observation'],
            result['truth_difference']['maximum_abs_demand'],
            places=14,
        )
        for label in ('coarse', 'fine'):
            self.assertLessEqual(result[label]['maximum_abs_mass_error'], 1e-10)
            for hour in (6, 12, 24):
                self.assertEqual(result[label]['horizons'][hour]['total'], 2)
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

    def test_small_d1_study_reports_all_declared_metrics_and_is_deterministic(self):
        left = mixture_f4.run_f4_d_representation_study(x_steps=81)
        right = mixture_f4.run_f4_d_representation_study(x_steps=81)
        self.assertEqual(left['selected_dictionary'], right['selected_dictionary'])
        self.assertEqual(left['dictionary_order'], right['dictionary_order'])
        self.assertEqual(left['gates'], right['gates'])
        for name in left['dictionary_order']:
            litem = left['results'][name]
            ritem = right['results'][name]
            self.assertEqual(litem['summary'], ritem['summary'])
            self.assertEqual(set(litem['fits']), set(left['targets']))
            for target, fit in litem['fits'].items():
                self.assertEqual(fit, ritem['fits'][target])
                if fit['status'] == 'failed':
                    self.assertIn('failure_type', fit)
                    self.assertIn('failure_message', fit)
                    self.assertFalse(litem['summary']['eligible'])
                    continue
                self.assertEqual(fit['status'], 'ok')
                for key in (
                    'density_l2', 'cell_mass_l1', 'cdf_sup', 'wasserstein_1',
                    'mean_error', 'variance_error', 'optimality_gap', 'mass_error',
                    'maximum_coefficient',
                ):
                    self.assertTrue(np.isfinite(fit[key]))
                self.assertGreaterEqual(fit['active_components'], 1)
        if left['selected_dictionary'] is not None:
            chosen = left['results'][left['selected_dictionary']]['summary']
            self.assertTrue(chosen['selectable'])
            self.assertTrue(chosen['numerical_ok'])
            self.assertTrue(chosen['representation_ok'])
            self.assertTrue(chosen['eligible'])

    def test_d1_failed_projection_is_recorded_not_repaired(self):
        with patch.object(
            mixture_f4.PreparedBetaProjector,
            'project',
            side_effect=FloatingPointError('forced signed projection'),
        ):
            result = mixture_f4.run_f4_d_representation_study(x_steps=61)
        self.assertIsNone(result['selected_dictionary'])
        for name in result['dictionary_order']:
            summary = result['results'][name]['summary']
            self.assertEqual(set(summary['failed_targets']), set(result['targets']))
            self.assertFalse(summary['numerical_ok'])
            self.assertFalse(summary['representation_ok'])
            self.assertFalse(summary['eligible'])
            for fit in result['results'][name]['fits'].values():
                self.assertEqual(fit['status'], 'failed')
                self.assertEqual(fit['failure_type'], 'FloatingPointError')
                self.assertIn('forced signed projection', fit['failure_message'])

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

    def test_small_d1b_study_keeps_same_gates_and_allows_validated_fallback(self):
        result = mixture_f4.run_f4_d_adaptive_representation_study(x_steps=61)
        self.assertEqual(result['stage'], 'D1b')
        self.assertFalse(result['fallback_disqualifies'])
        self.assertEqual(result['gates']['cell_mass_l1'], mixture_f4.D1_L1_GATE)
        self.assertEqual(result['gates']['wasserstein_1'], mixture_f4.D1_W1_GATE)
        self.assertEqual(result['dictionary_order'], ('adaptive_A', 'adaptive_B'))
        for name in result['dictionary_order']:
            summary = result['results'][name]['summary']
            self.assertEqual(summary['components'], 116 if name == 'adaptive_A' else 180)
            if summary['eligible']:
                self.assertTrue(summary['numerical_ok'])
                self.assertTrue(summary['representation_ok'])

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

    def test_small_d1c_study_keeps_same_gates_and_reports_conditioning(self):
        result = mixture_f4.run_f4_d_shape_aware_representation_study(x_steps=61)
        self.assertEqual(result['stage'], 'D1c')
        self.assertFalse(result['fallback_disqualifies'])
        self.assertEqual(result['dictionary_order'], ('shape_C', 'shape_D'))
        self.assertEqual(result['gates']['cell_mass_l1'], mixture_f4.D1_L1_GATE)
        self.assertEqual(result['gates']['wasserstein_1'], mixture_f4.D1_W1_GATE)
        for name in result['dictionary_order']:
            summary = result['results'][name]['summary']
            self.assertTrue(np.isfinite(summary['projection_matrix_condition_number']))
            self.assertEqual(summary['components'], 120 if name == 'shape_C' else 184)
            if summary['eligible']:
                self.assertTrue(summary['numerical_ok'])
                self.assertTrue(summary['representation_ok'])

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

    def test_small_d2_holdout_uses_frozen_shape_d_and_reports_evidence(self):
        result = mixture_f4.run_f4_d_snapshot_holdout(
            sequence_start=0,
            n_sequences=1,
            x_steps=201,
            max_step=1/24,
            likelihood_quadrature_order=1,
        )
        self.assertEqual(result['stage'], 'D2')
        self.assertEqual(result['dictionary'], 'shape_D')
        self.assertEqual(result['sequence_ids'], (0,))
        self.assertEqual(result['invalid_truth_sequences'], ())
        self.assertEqual(result['x_steps'], 201)
        self.assertEqual(
            result['predictive_log_evidence_error']['expected_count'], 24
        )
        self.assertGreaterEqual(
            result['predictive_log_evidence_error']['count'], 0
        )
        self.assertTrue(np.isfinite(result['projection_matrix_condition_number']))
        for label in ('prediction', 'posterior'):
            self.assertIn('maximum_cell_mass_l1', result[label])
            self.assertIn('passes_representation_gate', result[label])

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

    def test_d3b_refinement_reuses_shared_configuration_and_reports_gate(self):
        def fake_result(*, sequence_id, x_steps, max_step, cfl, likelihood_quadrature_order):
            scale = 1.0 + 0.01 * x_steps + max_step
            metric = {
                'maximum_cell_mass_l1': 0.02 * scale,
                'maximum_cdf_sup': 0.005 * scale,
                'maximum_wasserstein_1': 0.001 * scale,
                'maximum_density_l2': 0.03 * scale,
                'maximum_abs_mean_error': 1e-4 * scale,
                'maximum_abs_variance_error': 1e-5 * scale,
            }
            static = {
                'maximum_cell_mass_l1': 1e-5,
                'maximum_wasserstein_1': 1e-6,
            }
            return {
                'x_steps': x_steps,
                'max_step': max_step,
                'total_fv_gap': {'prediction': dict(metric), 'posterior': dict(metric)},
                'dynamic_gap': {'prediction': dict(metric), 'posterior': dict(metric)},
                'static_best_fit': {'prediction': dict(static), 'posterior': dict(static)},
                'log_evidence_error': {
                    'mean_absolute_per_observation': 0.005 * scale,
                    'maximum_absolute_per_observation': 0.01 * scale,
                    'absolute_total_log_likelihood_difference': 0.02 * scale,
                },
                'state_rmse': {
                    'fv_predictive': 0.03,
                    'fv_posterior': 0.02,
                    'mixture_predictive': 0.031,
                    'mixture_posterior': 0.021,
                },
                'numerical': {
                    'maximum_internal_step': max_step,
                    'minimum_internal_step': max_step / 2,
                    'total_internal_steps': 10,
                    'prediction_projection_fallback_count': 0,
                    'posterior_projection_fallback_count': 0,
                    'maximum_abs_raw_prediction_mass_error': 1e-15,
                    'maximum_abs_projected_prediction_mass_error': 1e-15,
                    'maximum_posterior_projection_mass_error': 1e-15,
                    'maximum_bayes_to_projected_posterior_l1': 1e-6,
                    'projection_matrix_condition_number': 10.0,
                    'hessian_condition_number': 100.0,
                },
            }

        with patch.object(
            mixture_f4, 'run_f4_d_recursive_sanity', side_effect=fake_result,
        ) as mocked:
            result = mixture_f4.run_f4_d_recursive_refinement(
                sequence_id=0,
                time_steps=(1/24, 1/48),
                spatial_cells=(31, 101),
                spatial_max_step=1/48,
            )
        self.assertEqual(result['stage'], 'D3b')
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual(len(result['unique_configurations']), 3)
        self.assertEqual([row['level'] for row in result['time']], [1/24, 1/48])
        self.assertEqual([row['level'] for row in result['spatial']], [31, 101])
        for axis in ('time', 'spatial'):
            self.assertIn('posterior_l1', result['summary'][axis])
            self.assertIn('evidence_mean_abs', result['summary'][axis])
            self.assertIsInstance(result['summary'][axis]['passes_stability_gate'], bool)

    def test_d3b_refinement_requires_ordered_unique_levels(self):
        with self.assertRaises(ValueError):
            mixture_f4.run_f4_d_recursive_refinement(time_steps=(1/48, 1/24))
        with self.assertRaises(ValueError):
            mixture_f4.run_f4_d_recursive_refinement(spatial_cells=(101, 51))
        with self.assertRaises(ValueError):
            mixture_f4.run_f4_d_recursive_refinement(time_steps=(1/384, 1/384))

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

    def test_d3c_convergence_wrapper_caches_shared_configuration(self):
        def fake_result(*, sequence_id, x_steps, max_step, cfl, likelihood_quadrature_order):
            hscale = max_step / (1/768)
            spatial = 1.0 + 0.0005 * (x_steps - 101)
            post = 0.01 * hscale * spatial
            evidence = 0.002 * hscale * spatial
            metric = {
                'maximum_cell_mass_l1': post,
                'maximum_cdf_sup': post / 4,
                'maximum_wasserstein_1': post / 10,
                'maximum_density_l2': post * 2,
                'maximum_abs_mean_error': 1e-4,
                'maximum_abs_variance_error': 1e-5,
            }
            static = {
                'maximum_cell_mass_l1': 1e-6,
                'maximum_wasserstein_1': 1e-7,
            }
            return {
                'x_steps': x_steps, 'max_step': max_step,
                'total_fv_gap': {'prediction': dict(metric), 'posterior': dict(metric)},
                'dynamic_gap': {'prediction': dict(metric), 'posterior': dict(metric)},
                'static_best_fit': {'prediction': dict(static), 'posterior': dict(static)},
                'log_evidence_error': {
                    'mean_absolute_per_observation': evidence,
                    'maximum_absolute_per_observation': evidence * 2,
                    'absolute_total_log_likelihood_difference': evidence * 24,
                },
                'state_rmse': {
                    'fv_predictive': .03, 'fv_posterior': .02,
                    'mixture_predictive': .031, 'mixture_posterior': .021,
                },
                'numerical': {
                    'maximum_internal_step': max_step,
                    'minimum_internal_step': max_step / 2,
                    'total_internal_steps': 10,
                    'prediction_projection_fallback_count': 0,
                    'posterior_projection_fallback_count': 0,
                    'maximum_abs_raw_prediction_mass_error': 1e-15,
                    'maximum_abs_projected_prediction_mass_error': 1e-15,
                    'maximum_posterior_projection_mass_error': 1e-15,
                    'maximum_bayes_to_projected_posterior_l1': 1e-6,
                    'projection_matrix_condition_number': 10.0,
                    'hessian_condition_number': 100.0,
                },
            }

        with patch.object(
            mixture_f4, 'run_f4_d_recursive_sanity', side_effect=fake_result,
        ) as mocked:
            result = mixture_f4.run_f4_d_recursive_convergence(
                sequence_id=0,
                time_steps=(1/768, 1/1536, 1/3072),
                spatial_cells=(101, 151, 201),
                spatial_max_step=1/3072,
            )
        self.assertEqual(result['stage'], 'D3c')
        self.assertEqual(mocked.call_count, 5)
        self.assertEqual(len(result['unique_configurations']), 5)
        self.assertTrue(result['summary']['time']['passes'])
        self.assertTrue(result['summary']['spatial']['passes'])
        self.assertTrue(result['summary']['passes_convergence'])
        self.assertEqual(result['summary']['population_setting']['x_steps'], 151)

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

    def test_small_d3d_population_runs_paired_without_dropping_sequences(self):
        result = mixture_f4.run_f4_d_recursive_population(
            n_sequences=2,
            sequence_start=0,
            x_steps=31,
            max_step=1/24,
            cfl=.4,
            likelihood_quadrature_order=1,
            bootstrap_replicates=100,
            bootstrap_seed=11,
            workers=1,
        )
        self.assertEqual(result['stage'], 'D3d')
        self.assertEqual(result['sequence_ids'], (0, 1))
        self.assertEqual(result['successful_sequences'], 2)
        self.assertEqual(result['invalid_truth_sequences'], ())
        self.assertEqual(result['failed_sequences'], ())
        self.assertEqual(result['workers'], 1)
        self.assertIn('predictive', result['rmse'])
        self.assertIn('posterior', result['rmse'])
        self.assertIn('prediction_l1', result['distribution_gap'])
        self.assertIn('posterior_l1', result['distribution_gap'])
        self.assertTrue(np.isfinite(
            result['log_evidence_difference']['mean_absolute_per_observation']
        ))
        self.assertLessEqual(result['numerical']['maximum_mass_error'], 1e-10)

    def test_d1_selection_never_selects_legacy_comparison(self):
        result = mixture_f4.run_f4_d_representation_study(x_steps=61)
        self.assertNotIn(result['selected_dictionary'], ('legacy_degree_30', 'legacy_degree_38'))

    def test_d1_invalid_grid_size_fails_explicitly(self):
        for value in (20, 20.5, True):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                mixture_f4.run_f4_d_representation_study(x_steps=value)


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



if __name__ == '__main__':
    unittest.main()
