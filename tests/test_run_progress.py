"""Runtime extrapolation for heterogeneous and resumed long-running jobs."""
import unittest

from src.exe.run_model.progress import estimate_seconds, fv_work, runtime_text


class RuntimeEstimateTests(unittest.TestCase):
    def test_resume_uses_only_new_work_and_current_elapsed(self):
        self.assertEqual(estimate_seconds(60, 160, 1000, initial_completed=100), (840., 900.))
        self.assertEqual(estimate_seconds(60, 100, 1000, initial_completed=100), (None, None))

    def test_grid_weighting_accounts_for_future_cost(self):
        base = dict(x_steps=100, max_step=1/100, likelihood_quadrature_order=3)
        fine = dict(base, x_steps=200)
        a, b = fv_work(base, 7, 168), fv_work(fine, 7, 168)
        self.assertEqual(b, 2*a)
        self.assertEqual(estimate_seconds(10, a, a+b), (20., 30.))
        self.assertGreater(fv_work(dict(base, max_step=1/200), 7, 168), a)

    def test_startup_empty_work_and_completion(self):
        self.assertEqual(estimate_seconds(2, 0, 10), (None, None))
        self.assertEqual(estimate_seconds(0, 1, 10), (None, None))
        self.assertEqual(estimate_seconds(7, 10, 10, initial_completed=4), (0., 7.))
        self.assertEqual(estimate_seconds(7, 0, 0), (0., 7.))

    def test_readable_duration_and_positive_remaining_rounds_up(self):
        self.assertEqual(runtime_text(100, 1, 2, now=3761),
                         'elapsed=1h 01m 01s; remaining~1h 01m 01s; total~2h 02m 02s')
        self.assertIn('remaining~1s', runtime_text(0, 99, 100, now=1))
        self.assertIn('total=estimating', runtime_text(0, 50, 100, initial_completed=50, now=1))

    def test_invalid_rates_do_not_produce_misleading_estimates(self):
        for args in ((-1, 1, 2), (1, -1, 2), (1, 3, 2), (1, 1, float('inf'))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                estimate_seconds(*args)
        with self.assertRaises(ValueError):
            estimate_seconds(1, 1, 2, initial_completed=2)
        self.assertEqual(estimate_seconds(1e308, 1, 3), (None, None))


if __name__ == '__main__':
    unittest.main()
