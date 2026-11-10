"""Beta-mixture approximation for the one-dimensional BM1 Fokker-Planck equation.

The component count is always fixed. With fixed_steps_per_hour provided,
moments use exactly that many RK4 steps per observation interval, with one
beta-mixture reconstruction at the end of each hour and no adaptive step
halving. Without that option, the original adaptive integrator is retained
for reference predictions and numerical validation.
"""
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq


class bm_1_beta:
    def __init__(self, components=8, quadrature=8, reconstruction_quadrature=48, max_step=0.25,
                 fixed_steps_per_hour=None):
        self.components = int(components)
        self.quadrature = int(quadrature)
        self.reconstruction_quadrature = int(reconstruction_quadrature)
        self.max_step = float(max_step)
        if fixed_steps_per_hour is not None and (
                isinstance(fixed_steps_per_hour, (bool, np.bool_))
                or not isinstance(fixed_steps_per_hour, (int, np.integer))
                or fixed_steps_per_hour < 1):
            raise ValueError("fixed_steps_per_hour must be a positive integer or None.")
        self.fixed_steps_per_hour = None if fixed_steps_per_hour is None else int(fixed_steps_per_hour)
        if min(self.components, self.quadrature, self.reconstruction_quadrature) < 1 or not 0.0 < self.max_step <= 1.0:
            raise ValueError("Invalid beta-mixture resolution.")

    @staticmethod
    def _demand(x, b, u, p):
        fx = np.tanh(0.5 * p["alpha"] * (0.5 - x)) / np.tanh(p["alpha"] / 4.0)
        gu = np.tanh(0.5 * p["beta"] * (0.5 - u)) / np.tanh(p["beta"] / 4.0)
        delta = np.tanh(0.5 * p["k"] * (fx + gu))
        return b + p["Xi"] * delta * np.where(delta >= 0.0, 1.0 - b, b)

    @staticmethod
    def _check(mix):
        weight, mean, var = (np.asarray(a, dtype=float) for a in mix)
        if not (np.isfinite(weight).all() and np.isfinite(mean).all() and np.isfinite(var).all()):
            raise FloatingPointError("Non-finite beta-mixture moments.")
        if np.any(weight < -1e-12) or np.any(mean < -1e-9) or np.any(mean > 1.0 + 1e-9):
            raise FloatingPointError("Inadmissible beta-mixture weights or means.")
        mean = np.clip(mean, 0.0, 1.0)
        upper = mean * (1.0 - mean)
        if np.any(var < -1e-11) or np.any(var > upper + 1e-11):
            raise FloatingPointError("Inadmissible beta-mixture variance.")
        var = np.maximum(0.0, np.minimum(var, np.maximum(0.0, upper * (1.0 - 1e-12))))
        total = float(np.sum(weight))
        if total <= 0.0: raise FloatingPointError("Zero beta-mixture mass.")
        return weight / total, mean, var

    @staticmethod
    def _component_nodes(mean, var, n):
        """Gauss-Jacobi rule with normalized weights from a stable Jacobi matrix.

        scipy.special.roots_jacobi calculates an unnormalized weight integral
        whose exponential normalization overflows for concentrated beta laws,
        even when all component moments are admissible. Golub-Welsch avoids
        that exponential: eigenvalues give the nodes and the squared first
        eigenvector row gives weights summing directly to one.
        """
        if var <= 1e-18 or var <= 1e-8 * mean * (1.0 - mean) or mean <= 0.0 or mean >= 1.0:
            return np.array([float(mean)]), np.array([1.0])
        strength = mean * (1.0 - mean) / var - 1.0
        a, b = mean * strength, (1.0 - mean) * strength
        if not np.isfinite(a + b) or a <= 0.0 or b <= 0.0:
            raise FloatingPointError("Invalid beta component shape.")
        total = a + b
        diagonal = np.empty(n, dtype=float)
        diagonal[0] = mean
        for j in range(1, n):
            s = 2.0 * j + total
            diagonal[j] = 0.5 * (1.0 + (a - b) * (total - 2.0) / ((s - 2.0) * s))
        off_diagonal = np.empty(n - 1, dtype=float)
        if n > 1:
            off_diagonal[0] = np.sqrt((a / total) * (b / total) / (total + 1.0))
        for j in range(2, n):
            s = 2.0 * j + total
            # Factor the rational formula to avoid needless large products.
            numerator = j * (j + a - 1.0) * (j + b - 1.0) * (j + total - 2.0)
            denominator = (s - 2.0) ** 2 * (s - 3.0) * (s - 1.0)
            off_diagonal[j - 1] = np.sqrt(numerator / denominator)
        if not np.all(np.isfinite(diagonal)) or not np.all(np.isfinite(off_diagonal)):
            raise FloatingPointError("Non-finite beta Jacobi recurrence coefficients.")
        nodes, eigenvectors = eigh_tridiagonal(diagonal, off_diagonal)
        weights = np.square(eigenvectors[0, :])
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(weights)) or weights.sum() <= 0.0:
            raise FloatingPointError("Invalid normalized beta quadrature.")
        return np.clip(nodes, 0.0, 1.0), weights / weights.sum()

    def support(self, mix, quadrature=None):
        w, mu, var = self._check(mix)
        n = self.quadrature if quadrature is None else int(quadrature)
        points, masses = [], []
        for wi, mi, vi in zip(w, mu, var):
            if wi <= 0.0: continue
            x, q = self._component_nodes(mi, vi, n)
            points.extend(x)
            masses.extend(wi * q)
        x = np.asarray(points, dtype=float)
        mass = np.asarray(masses, dtype=float)
        return x, mass / np.sum(mass)

    def _rhs(self, mu, var, b, u, p):
        derivative_mu, derivative_var = np.empty_like(mu), np.empty_like(var)
        for i, (m, v) in enumerate(zip(mu, var)):
            x, q = self._component_nodes(float(m), float(v), self.quadrature)
            a = (self._demand(x, b, u, p) - b) / p["C"]
            diffusion2 = (p["sigma_x"] * x * (1.0 - x)) ** 2
            derivative_mu[i] = q @ a
            derivative_var[i] = 2.0 * (q @ ((x - m) * a)) + q @ diffusion2
        return derivative_mu, derivative_var

    def _rk4(self, mix, h, b_left, b_right, u, elapsed, p):
        w, mu, var = mix
        def rhs(m, v, z):
            _, m, v = self._check((w, m, v))
            b = (1.0 - z) * b_left + z * b_right
            return self._rhs(m, v, b, u, p)
        a1, v1 = rhs(mu, var, elapsed)
        a2, v2 = rhs(mu + 0.5 * h * a1, var + 0.5 * h * v1, elapsed + 0.5 * h)
        a3, v3 = rhs(mu + 0.5 * h * a2, var + 0.5 * h * v2, elapsed + 0.5 * h)
        a4, v4 = rhs(mu + h * a3, var + h * v3, elapsed + h)
        return self._check((w, mu + h * (a1 + 2*a2 + 2*a3 + a4) / 6.0,
                            var + h * (v1 + 2*v2 + 2*v3 + v4) / 6.0))

    def _reconstruct(self, mix):
        x, mass = self.support(mix, self.reconstruction_quadrature)
        order = np.argsort(x)
        x, mass = x[order], mass[order]
        L = self.components
        means, second = np.zeros(L), np.zeros(L)
        group, filled = 0, 0.0
        for point, original_mass in zip(x, mass):
            remaining = float(original_mass)
            while remaining > 1e-15 and group < L:
                available = (1.0 / L) - filled
                take = min(remaining, available)
                means[group] += take * point
                second[group] += take * point * point
                filled += take
                remaining -= take
                if filled >= 1.0 / L - 1e-13:
                    group += 1
                    filled = 0.0
        if group != L:
            # Roundoff in the final group is below probability resolution.
            if group == L-1 and filled > 1.0 / L - 1e-10: group = L
            else: raise FloatingPointError("Beta reconstruction lost probability mass.")
        means *= L
        variance = np.maximum(0.0, second * L - means * means)
        return self._check((np.full(L, 1.0 / L), means, variance))

    def _propagate_hour(self, mix, b0, b1, u, p, first=False):
        if self.fixed_steps_per_hour is not None:
            # One hourly observation interval, exactly n equally sized RK4
            # steps, then exactly one fixed-M-component reconstruction.
            # No stiffness-controlled subdivision and no rescue halving:
            # inadmissible candidate parameters are rejected by the MLE.
            n = self.fixed_steps_per_hour
            h = 1.0 / n
            for j in range(n):
                try:
                    mix = self._rk4(mix, h, b0, b1, u, j*h, p)
                except (FloatingPointError, ValueError) as exc:
                    raise FloatingPointError(
                        f"Fixed-step beta RK4 failed on step {j+1}/{n}: {exc}") from exc
            return self._reconstruct(mix)
        # Adaptive mode for reference predictions / verification.
        # Conservative drift derivative bound limits stiff RK4 steps.
        slope = p["Xi"] * p["k"] * p["alpha"] / (4.0 * p["C"] * np.tanh(p["alpha"] / 4.0))
        step = min(self.max_step, 0.5 / max(1.0, slope))
        if step < 1.0 / 4096.0: raise FloatingPointError("BM1 moment dynamics too stiff for the selected resolution.")
        boundaries = (0.0, 0.25, 1.0) if first else (0.0, 1.0)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            t = left
            while t < right - 1e-12:
                h = min(step, right - t)
                last_error = None
                for _ in range(18):
                    try:
                        candidate = self._rk4(mix, h, b0, b1, u, t, p)
                        break
                    except (FloatingPointError, ValueError) as exc:
                        last_error = exc
                        h *= 0.5
                else:
                    raise FloatingPointError(
                        f"Adaptive beta RK4 failed at hour_fraction={t:.6g}, "
                        f"last_step={h:.3g}: {last_error}") from last_error
                if h < 1e-9: raise FloatingPointError("Beta RK4 substep underflow.")
                mix, t = candidate, t + h
            if first and right == 0.25: mix = self._reconstruct(mix)
        return self._reconstruct(mix)

    def _observe(self, mix, observation, b, u, p):
        w, mu, var = mix
        sigma = p["sigma_y"]
        log_masses, moments, second = [], [], []
        for wi, mi, vi in zip(w, mu, var):
            x, q = self._component_nodes(float(mi), float(vi), self.quadrature)
            error = (observation - self._demand(x, b, u, p)) / sigma
            log_q = np.log(q) - 0.5 * error * error
            evidence = logsumexp(log_q)
            conditional = np.exp(log_q - evidence)
            log_masses.append(np.log(max(float(wi), 1e-300)) + evidence)
            moments.append(float(conditional @ x))
            second.append(float(conditional @ (x * x)))
        log_masses = np.asarray(log_masses)
        log_total = logsumexp(log_masses)
        updated_weights = np.exp(log_masses - log_total)
        moments = np.asarray(moments)
        return (self._check((updated_weights, moments, np.maximum(0.0, np.asarray(second)-moments**2))),
                float(-log_total + np.log(sigma) + 0.5 * np.log(2.0 * np.pi)))

    def simulate(self, data, p, x0, observations=None, nll_only=False):
        """Sample hourly; nll_only skips costly unused forecast distributions during MLE."""
        b, u = np.asarray(data["B"], dtype=float), np.asarray(data["U"], dtype=float)
        observed = None if observations is None else np.asarray(observations, dtype=float)
        if b.ndim != 1 or b.size < 1 or u.shape != b.shape or not np.isfinite(b).all() or not np.isfinite(u).all():
            raise ValueError("Beta propagation needs finite, equally sized hourly B and U.")
        if observed is not None and (observed.shape != b.shape or not np.isfinite(observed).all()):
            raise ValueError("Historical observations must match the hourly inputs.")
        if nll_only and observed is None:
            raise ValueError("nll_only requires historical observations.")
        if not (0.0 <= x0 <= 1.0 and 0.0 < p["sigma_x"] <= 5.0 and 0.0 < p["sigma_y"] <= 0.5):
            raise ValueError("Initial state or stochastic parameters outside admissible bounds.")
        mix = (np.array([1.0]), np.array([float(x0)]), np.array([0.0]))
        means, supports, weights, states = [], [], [], []
        nll = 0.0
        for j in range(len(b)):
            if not nll_only:
                x, mass = self.support(mix)
                demand = self._demand(x, b[j], u[j], p)
                means.append(float(mass @ demand))
                supports.append(demand)
                weights.append(mass)
                states.append((mix[0].copy(), mix[1].copy(), mix[2].copy()))
            if observed is not None:
                mix, loss = self._observe(mix, float(observed[j]), float(b[j]), float(u[j]), p)
                nll += loss
            if j + 1 < len(b):
                try:
                    mix = self._propagate_hour(mix, float(b[j]), float(b[j+1]), float(u[j]), p, first=(j == 0))
                except (FloatingPointError, ValueError) as exc:
                    raise FloatingPointError(f"BM1 beta propagation failed between hours {j} and {j+1}: {exc}") from exc
        if nll_only: return {"nll":float(nll)}
        return {"mean": np.asarray(means), "supports": supports, "weights": weights,
                "states": states, "nll": float(nll)}

    @staticmethod
    def quantile(support, weights, observation_sd, q):
        support, weights = np.asarray(support), np.asarray(weights)
        if not 0.0 < q < 1.0: raise ValueError("Quantile probability must be in (0,1).")
        lo, hi = float(support.min() - 8.0*observation_sd), float(support.max() + 8.0*observation_sd)
        return float(brentq(lambda y: np.dot(weights, ndtr((y - support) / observation_sd)) - q, lo, hi, xtol=1e-10))

    def intervals(self, result, observation_sd):
        output = {}
        for key, q in (("lower_68", 0.16), ("upper_68", 0.84), ("lower_95", 0.025), ("upper_95", 0.975)):
            output[key] = np.array([self.quantile(x, w, observation_sd, q)
                                    for x, w in zip(result["supports"], result["weights"])])
        return output
