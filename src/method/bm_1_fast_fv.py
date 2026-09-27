"""Opt-in compiled/blocked evaluation of the existing stochastic BM1 FV cost.

Same face drift, generator, time grid, implicit Euler and midpoint objective
as FVStochasticPriceEvaluator. No coarse grid, changed tolerance or density
repair. SciPy's LAPACK tridiagonal solve replaces scalar Python elimination.

Callbacks must be pure functions. With vectorized=True they must support
NumPy broadcasting across time and state axes; vectorized=False retains the
scalar-time callback convention. No model method is monkey-patched.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg.lapack import dgtsv

from src.method.bm_1_stochastic_objective import (
    stochastic_tracking_objective, StochasticTrackingResult,
)
from src.method.bm_1_stochastic_optimization import (
    FVStochasticPriceEvaluator, PriceObjectiveEvaluation,
)
from src.validation.model.bm_1 import bm_1_validation
from src.validation.solver.finite_volume import finite_volume_1d_validation


@dataclass(frozen=True)
class FVCostEvaluation:
    """Cost-only evaluation: no full density history is retained."""
    objective: StochasticTrackingResult
    diagnostics: dict


class FastFVStochasticPriceEvaluator(FVStochasticPriceEvaluator):
    """Drop-in opt-in evaluator; construct a new instance if grids change.

    The mathematical model/input callbacks and the discretization are fixed
    for the life of an evaluator, just as its target quadrature values are.
    Candidate prices vary between calls. Coefficients are *not* cached across
    calls and are evaluated at every original left time (including varying B).
    """
    def __init__(self, *args, batch_size=128, vectorized=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = finite_volume_1d_validation.integer(
            batch_size, name="batch_size", minimum=1,
        )
        if not isinstance(vectorized, (bool, np.bool_)):
            raise TypeError("vectorized must be a boolean.")
        self.vectorized = bool(vectorized)
        self.integration_t, self.output_indices = (
            self.model.fokker_planc_reference.build_time_grid(
                self.solver_t_eval, self.breakpoints, self.solver_max_step,
            )
        )
        self._steps = np.diff(self.integration_t)
        self._output_at = np.full(self.integration_t.size, -1, dtype=int)
        self._output_at[self.output_indices] = np.arange(self.output_indices.size)

    def _drift_block(self, times, parameters):
        faces = self.edges[1:-1]
        if self.vectorized:
            values = self.model._drift(times[:, None], faces[None, :], parameters)
            return finite_volume_1d_validation.field(
                values, (times.size, faces.size), "drift",
            )
        return np.asarray([
            finite_volume_1d_validation.field(
                self.model._drift(float(t), faces, parameters), faces.shape, "drift",
            ) for t in times
        ])

    def _demand_block(self, times, parameters):
        u, f, g, _, k, xi = parameters
        if self.vectorized:
            values = self.model.eval(
                times[:, None], self.x[None, :], u, f, g, k, xi,
            )[-1]
            return finite_volume_1d_validation.field(
                values, (times.size, self.x.size), "demand",
            )
        return np.asarray([
            finite_volume_1d_validation.field(
                self.model.eval(float(t), self.x, u, f, g, k, xi)[-1],
                self.x.shape, "demand",
            ) for t in times
        ])

    def _run(self, prices, keep_history):
        prices = self.validate_prices(prices)
        _, parameters, sigma = bm_1_validation.stochastic_inputs(
            [0.5], self.parameters_for_prices(prices), self.sigma_x,
        )
        mass = finite_volume_1d_validation.initial_mass(self.initial_mass)
        diffusion = finite_volume_1d_validation.field(
            self.model.diffusion_coefficient(self.x, sigma), self.x.shape, "diffusion",
        )
        diffusion_term = 0.5 * np.square(diffusion)
        diffusion_rates = diffusion_term / (self.dx * self.dx)
        if not np.all(np.isfinite(diffusion_rates)):
            raise ValueError("Diffusion rates must be finite.")

        n_output = self.solver_t_eval.size
        history = np.empty((n_output, mass.size)) if keep_history else None
        if keep_history:
            history[0] = mass
        expected = np.full(self.midpoints.size, np.nan)
        second = np.full_like(expected, np.nan)
        min_mass = float(np.min(mass))
        max_mass_error = abs(float(np.sum(mass)) - 1.0)
        max_column_sum = 0.0
        min_offdiag = np.inf

        for start in range(0, self._steps.size, self.batch_size):
            stop = min(start + self.batch_size, self._steps.size)
            face_drift = self._drift_block(self.integration_t[start:stop], parameters)
            # Same arithmetic and boundary rates as generator_tridiagonal.
            right = np.zeros((stop - start, mass.size))
            left = np.zeros_like(right)
            right[:, :-1] = np.maximum(face_drift, 0.0) / self.dx + diffusion_rates[:-1]
            left[:, 1:] = np.maximum(-face_drift, 0.0) / self.dx + diffusion_rates[1:]
            diagonal = -(left + right)
            lower = right[:, :-1]
            upper = left[:, 1:]
            column_sum = diagonal.copy()
            column_sum[:, :-1] += lower
            column_sum[:, 1:] += upper
            max_column_sum = max(max_column_sum, float(np.max(np.abs(column_sum))))
            min_offdiag = min(min_offdiag, float(np.min(lower)), float(np.min(upper)))
            if not np.all(np.isfinite(diagonal)):
                raise ValueError("Generator rates must be finite.")

            output_ids = self._output_at[start + 1:stop + 1]
            moment_ids = output_ids[(output_ids > 0) & (output_ids < n_output - 1)] - 1
            demand = self._demand_block(self.midpoints[moment_ids], parameters) if moment_ids.size else None
            demand_row = 0
            for local, step in enumerate(range(start, stop)):
                h = self._steps[step]
                # New diagonals are disposable; RHS is our private state array.
                # The solver may pivot. Never modify caller-owned initial masses.
                dl = -h * lower[local]
                d = 1.0 - h * diagonal[local]
                du = -h * upper[local]
                if not (np.all(np.isfinite(dl)) and np.all(np.isfinite(d)) and np.all(np.isfinite(du))):
                    raise ValueError("Implicit step matrix must be finite.")
                _, _, _, solved, info = dgtsv(
                    dl, d, du, mass[:, None],
                    overwrite_dl=1, overwrite_d=1, overwrite_du=1, overwrite_b=1,
                )
                if info != 0:
                    raise RuntimeError(f"LAPACK tridiagonal solve failed (info={info}).")
                mass = finite_volume_1d_validation.evolved_mass(solved[:, 0])
                min_mass = min(min_mass, float(np.min(mass)))
                max_mass_error = max(max_mass_error, abs(float(np.sum(mass)) - 1.0))
                if max_mass_error > 1.0e-9:
                    raise RuntimeError("FV probability mass drift exceeds tolerance; no repair applied.")
                index = output_ids[local]
                if index >= 0:
                    if keep_history:
                        history[index] = mass
                    if 0 < index < n_output - 1:
                        values = demand[demand_row]
                        expected[index - 1] = np.sum(values * mass)
                        second[index - 1] = np.sum(np.square(values) * mass)
                        demand_row += 1

        objective = stochastic_tracking_objective(self.widths, expected, second, self.target)
        diagnostics = {
            "integration_t": self.integration_t.copy(),
            "output_indices": self.output_indices.copy(),
            "n_steps": self._steps.size,
            "maximum_abs_generator_column_sum": max_column_sum,
            "minimum_off_diagonal_rate": min_offdiag,
            "minimum_cell_mass": min_mass,
            "maximum_abs_mass_error": max_mass_error,
            "used_clipping": False, "used_renormalization": False,
            "backend": "lapack-blocked-fv", "batch_size": self.batch_size,
            "saved_density_history": bool(keep_history),
        }
        if keep_history:
            return PriceObjectiveEvaluation(prices.copy(), objective, history, diagnostics)
        return FVCostEvaluation(objective, diagnostics)

    def evaluate(self, prices):
        """Return the standard objective, requested mass history and diagnostics."""
        return self._run(prices, keep_history=True)

    def evaluate_cost(self, prices):
        """Return objective and diagnostics without saving a density history."""
        return self._run(prices, keep_history=False)

    def __call__(self, prices):
        return self.evaluate_cost(prices).objective.total
