"""Analytical reference and diagnostics for deterministic BM1 verification.

The exact solution in this module is intentionally restricted to the linear
responses ``f(x) = 1 - 2x`` and ``g(u) = 1 - 2u``, a constant baseline, and
piecewise-constant prices. It is verification code, not a general BM1 solver.
The executable layer remains responsible for constructing inputs and calling
the model solvers.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BM1DeterministicBenchmark:
    """Mathematics and checks for the deterministic F1 benchmark."""

    time_interval: tuple[float, float]
    price_step_time: float
    price_before: float
    price_after: float
    initial_state: float
    baseline: float
    capacity: float
    k: float
    flexible_share: float
    energy_tol: float = 1.0e-12
    state_tol: float = 1.0e-12

    @staticmethod
    def state_response(x):
        """Return the benchmark state response f(x) = 1 - 2x."""
        return 1.0 - 2.0 * np.asarray(x, dtype=float)

    @staticmethod
    def price_response(u):
        """Return the benchmark price response g(u) = 1 - 2u."""
        return 1.0 - 2.0 * np.asarray(u, dtype=float)

    def exact_constant_price_state(
        self, t, initial_state, initial_time, price
    ):
        """Return the exact state while price and baseline are constant."""
        times = np.asarray(t, dtype=float)
        equilibrium = 1.0 - price

        if initial_state < equilibrium:
            demand_scale = 1.0 - self.baseline
        else:
            demand_scale = self.baseline

        decay_rate = self.k * self.flexible_share * demand_scale / self.capacity
        initial_offset = self.k * (initial_state - equilibrium)
        elapsed = times - initial_time

        return equilibrium + np.arcsinh(
            np.sinh(initial_offset) * np.exp(-decay_rate * elapsed)
        ) / self.k

    def exact_state(self, t):
        """Return the exact state for the benchmark's two price intervals."""
        times = np.asarray(t, dtype=float)
        state = np.empty_like(times)

        before_step = times <= self.price_step_time
        state[before_step] = self.exact_constant_price_state(
            times[before_step],
            self.initial_state,
            self.time_interval[0],
            self.price_before,
        )

        state_at_step = float(
            self.exact_constant_price_state(
                np.array([self.price_step_time]),
                self.initial_state,
                self.time_interval[0],
                self.price_before,
            )[0]
        )
        state[~before_step] = self.exact_constant_price_state(
            times[~before_step],
            state_at_step,
            self.price_step_time,
            self.price_after,
        )
        return state

    def energy_residual(self, t, state, baseline, demand):
        """Return the cumulative Euler energy-balance residual."""
        increments = np.diff(t) * (demand[:-1] - baseline[:-1])
        accumulated_energy = np.concatenate(([0.0], np.cumsum(increments)))
        return self.capacity * (state - state[0]) - accumulated_energy

    @staticmethod
    def observed_orders(euler_results):
        """Return observed orders between consecutive halved time steps."""
        orders = [np.nan]
        for coarse, fine in zip(euler_results[:-1], euler_results[1:]):
            orders.append(
                np.log(coarse["error"] / fine["error"]) / np.log(2.0)
            )
        return orders

    def sanity_checks(
        self,
        euler_results,
        equilibrium,
        no_flexibility,
        low_price,
        high_price,
        lower_boundary_drift,
        upper_boundary_drift,
    ):
        """Evaluate sanity-check data produced by the executable layer."""
        equilibrium_state = equilibrium[1]
        no_flexibility_state = no_flexibility[1]
        low_price_state = low_price[1]
        high_price_state = high_price[1]
        all_states = np.concatenate(
            [result["state"] for result in euler_results]
        )
        max_energy_error = max(
            result["energy_error"] for result in euler_results
        )

        return {
            "equilibrium remains fixed": (
                np.max(np.abs(equilibrium_state - self.initial_state))
                <= self.state_tol
            ),
            "Xi = 0 gives D = B and fixed X": (
                np.max(
                    np.abs(no_flexibility_state - self.initial_state)
                )
                <= self.state_tol
                and np.max(
                    np.abs(no_flexibility[7] - no_flexibility[3])
                )
                <= self.energy_tol
            ),
            "low price increases X toward 0.75": (
                low_price_state[-1] > self.initial_state
                and np.all(np.diff(low_price_state) >= -self.state_tol)
                and low_price_state[-1] < 1.0 - self.price_after
            ),
            "high price decreases X toward 0.25": (
                high_price_state[-1] < self.initial_state
                and np.all(np.diff(high_price_state) <= self.state_tol)
                and high_price_state[-1] > 1.0 - self.price_before
            ),
            "continuous drift is inward at both boundaries": (
                np.min(lower_boundary_drift) >= -self.state_tol
                and np.max(upper_boundary_drift) <= self.state_tol
            ),
            "Euler states stay in [0, 1] without correction": (
                np.min(all_states) >= -self.state_tol
                and np.max(all_states) <= 1.0 + self.state_tol
            ),
            "Euler energy identity": max_energy_error <= self.energy_tol,
        }
