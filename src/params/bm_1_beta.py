"""Independent historical BM1 fits and beta-mixture stochastic maximum likelihood.

Parameter estimation uses N corresponding hourly windows of arbitrary length.
The shared logistic parameters are fitted by alternating bounded least squares
and independent historical X0 searches. Stochastic likelihood then optimizes
all physical parameters, N initial states, sigma_x and sigma_y jointly.
"""
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar

from src.method.bm_1_beta import bm_1_beta
from src.params.bm_1 import lsq_params


class bm_1_beta_params:
    NAMES = ("alpha", "beta", "k", "Xi", "C")
    BOUNDS = {"alpha": (0.05, 20.0), "beta": (0.05, 20.0), "k": (0.05, 12.0),
              "Xi": (1e-4, 1.0), "C": (0.05, 100.0),
              "sigma_x": (1e-4, 5.0), "sigma_y": (0.001, 0.5)}

    @classmethod
    def _prepare(cls, datasets):
        if not isinstance(datasets, (tuple, list)) or not datasets: raise ValueError("At least one historical interval is required.")
        count = len(np.asarray(datasets[0]["B"]))
        if count < 2: raise ValueError("The fitting horizon must contain at least two hourly observations.")
        for data in datasets:
            for name in ("B", "U", "I"):
                a = np.asarray(data[name], dtype=float)
                if a.shape != (count,) or not np.isfinite(a).all() or np.any((a < 0.0) | (a > 1.0)):
                    raise ValueError(f"Historical {name} must contain {count} finite normalized observations.")
            if np.asarray(data.get("Time", np.arange(count))).shape != (count,):
                raise ValueError("Historical time grid must match the fitting horizon.")
        return list(datasets)

    @staticmethod
    def _deterministic(data, p, x0, substeps):
        B, U = np.asarray(data["B"]), np.asarray(data["U"])
        x = float(x0)
        prediction = np.empty(len(B))
        for j in range(len(B)):
            prediction[j] = lsq_params._demand(x, float(B[j]), float(U[j]), p)
            if j == len(B) - 1: break
            steps = max(int(substeps), int(np.ceil(p["Xi"] / (0.2 * p["C"]))))
            h = 1.0 / steps
            b0, b1, u = float(B[j]), float(B[j+1]), float(U[j])
            def rate(value, phase):
                b = b0 + (b1-b0)*phase
                return (lsq_params._demand(value, b, u, p)-b) / p["C"]
            for i in range(steps):
                phase = i * h
                k1 = rate(x, phase)
                k2 = rate(x + h*k1/2.0, phase+h/2.0)
                k3 = rate(x + h*k2/2.0, phase+h/2.0)
                k4 = rate(x + h*k3, phase+h)
                x += h * (k1 + 2*k2 + 2*k3 + k4) / 6.0
                if x < 0.0 or x > 1.0: raise FloatingPointError("BM1 RK4 exited [0,1]; trajectory is not clipped.")
        return prediction

    def fit_deterministic(self, datasets, *, initial_state=0.5, cycles=4, max_nfev=120, state_maxiter=60, substeps=2):
        datasets = self._prepare(datasets)
        n = len(datasets)
        if not 0.0 <= initial_state <= 1.0: raise ValueError("initial_state must be in [0,1].")
        if min(cycles, max_nfev, state_maxiter, substeps) < 1: raise ValueError("Deterministic fitting controls must be positive.")
        lower, upper = [np.array([self.BOUNDS[key][i] for key in self.NAMES]) for i in (0,1)]
        theta = np.array([2.0, 2.0, 1.0, 0.5, 1.0])
        states = np.full(n, initial_state, dtype=float)
        best, iterations = None, 0
        def residual(candidate, x0):
            p = dict(zip(self.NAMES, candidate))
            return np.concatenate([self._deterministic(data, p, state, substeps) - data["I"]
                                   for data, state in zip(datasets, x0)])
        started = perf_counter()
        for cycle in range(int(cycles)):
            fit = least_squares(lambda candidate: residual(candidate, states), theta,
                                bounds=(lower, upper), max_nfev=max_nfev,
                                method="trf", loss="linear")
            theta_new = np.asarray(fit.x)
            p = dict(zip(self.NAMES, theta_new))
            states_new = []
            for data, original in zip(datasets, states):
                def error(value):
                    r = self._deterministic(data, p, float(value), substeps) - data["I"]
                    return float(r @ r)
                solution = minimize_scalar(error, method="bounded", bounds=(0.0, 1.0),
                                           options={"xatol": 1e-6, "maxiter": state_maxiter})
                candidates = (float(original), 0.0, 1.0, float(solution.x))
                states_new.append(min(candidates, key=error))
            states_new = np.asarray(states_new)
            errors = residual(theta_new, states_new)
            score = float(errors @ errors)
            iterations += 1
            old_best = np.inf if best is None else best["sse"]
            if score < old_best:
                best = {"theta": theta_new.copy(), "states": states_new.copy(), "sse": score,
                        "lsq_converged": bool(fit.success), "lsq_message": str(fit.message)}
            if cycle > 0 and old_best - score < 1e-11: break
            theta, states = theta_new, states_new
        return {**dict(zip(self.NAMES, best["theta"])), "initial_states": best["states"],
                "mse": best["sse"] / sum(len(d["I"]) for d in datasets), "sse": best["sse"],
                "iterations": iterations, "lsq_converged": best["lsq_converged"],
                "lsq_message": best["lsq_message"], "seconds": perf_counter()-started}

    def fit_mle(self, datasets, deterministic, *, components=8, quadrature=8,
                reconstruction_quadrature=48, max_step=0.25, maxiter=45, maxfun=700,
                sigma_x0=0.15, sigma_y0=None, progress=False):
        datasets = self._prepare(datasets)
        n = len(datasets)
        solver = bm_1_beta(components, quadrature, reconstruction_quadrature, max_step)
        names = (*self.NAMES, "sigma_x", "sigma_y")
        raw_seed = np.array([float(deterministic[key]) for key in self.NAMES], dtype=float)
        start_states = np.asarray(deterministic["initial_states"], dtype=float)
        if start_states.shape != (n,): raise ValueError("Deterministic fit returned an incorrect number of initial states.")
        if sigma_y0 is None:
            diffs = np.concatenate([self._deterministic(data, dict(zip(self.NAMES, raw_seed)), float(state), 2) - data["I"]
                                    for data, state in zip(datasets, start_states)])
            sigma_y0 = float(np.clip(np.sqrt(np.mean(diffs*diffs)), 0.005, 0.3))
        values = np.r_[raw_seed, sigma_x0, sigma_y0]
        limits = [self.BOUNDS[key] for key in names]
        values = np.array([np.clip(a, low+1e-10, high-1e-10)
                           for a, (low, high) in zip(values, limits)])
        seed = np.r_[np.log(values[:5]), start_states, np.log(values[5:])]
        bounds = [(np.log(self.BOUNDS[key][0]), np.log(self.BOUNDS[key][1])) for key in self.NAMES]
        bounds += [(0.0, 1.0)] * n
        bounds += [(np.log(self.BOUNDS[key][0]), np.log(self.BOUNDS[key][1])) for key in ("sigma_x", "sigma_y")]
        evaluations = 0
        best = [np.inf, None]
        started = perf_counter()
        def decode(q):
            p = {key: float(np.exp(q[j])) for j, key in enumerate(self.NAMES)}
            p["sigma_x"], p["sigma_y"] = float(np.exp(q[-2])), float(np.exp(q[-1]))
            return p, q[5:-2]
        def objective(q):
            nonlocal evaluations
            evaluations += 1
            p, x0 = decode(q)
            try:
                value = sum(solver.simulate(d, p, float(x), observations=d["I"])["nll"]
                            for d, x in zip(datasets, x0))
                if not np.isfinite(value): raise FloatingPointError("nonfinite likelihood")
            except (FloatingPointError, ValueError, OverflowError):
                # Numerically inadmissible iterates are rejected, rather than
                # altering the model or silently projecting its state.
                value = 1e15
            if value < best[0]: best[0], best[1] = float(value), q.copy()
            if progress and evaluations % 10 == 0:
                print(f"[beta MLE] calls={evaluations} best_NLL={best[0]:.7f} "
                      f"elapsed={perf_counter()-started:.1f}s", flush=True)
            return value
        result = minimize(objective, seed, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": int(maxiter), "maxfun": int(maxfun), "ftol": 1e-9, "maxls": 20})
        if best[1] is None or not np.isfinite(best[0]) or best[0] >= 1e14:
            raise RuntimeError("BM1 beta-mixture likelihood returned no admissible parameter candidate.")
        p, states = decode(best[1])
        return {**p, "initial_states": np.asarray(states), "nll": float(best[0]),
                "converged": bool(result.success), "message": str(result.message),
                "optimizer_status": int(result.status), "iterations": int(result.nit),
                "evaluations": evaluations, "seconds": perf_counter()-started,
                "resolution": {"components":components, "quadrature":quadrature,
                               "reconstruction_quadrature":reconstruction_quadrature, "max_step":max_step}}

    def oracle(self, target, fitted, *, components=24, quadrature=12,
               reconstruction_quadrature=48, max_step=0.125, maxiter=70):
        solver = bm_1_beta(components, quadrature, reconstruction_quadrature, max_step)
        observed = np.asarray(target["I"], dtype=float)
        started = perf_counter()
        evaluations = 0
        def loss(x):
            nonlocal evaluations
            evaluations += 1
            try:
                pred = solver.simulate(target, fitted, float(x))["mean"]
                return float(np.mean(np.square(pred-observed)))
            except (FloatingPointError, ValueError): return 1e12
        grid = np.linspace(0.0, 1.0, 5)
        candidates = [(loss(float(x)), float(x)) for x in grid]
        for left, right in zip(grid[:-1], grid[1:]):
            result = minimize_scalar(loss, method="bounded", bounds=(float(left), float(right)),
                                     options={"maxiter": int(maxiter), "xatol":1e-5})
            candidates.append((float(result.fun), float(result.x)))
        mse, x0 = min(candidates)
        if mse >= 1e11: raise RuntimeError("BM1 oracle could not find a numerically valid initial state.")
        prediction = solver.simulate(target, fitted, x0)
        prediction.update(solver.intervals(prediction, fitted["sigma_y"]))
        return {"initial_state": x0, "mse": mse, "seconds":perf_counter()-started,
                "evaluations":evaluations, "prediction":prediction,
                "resolution": {"components":components, "quadrature":quadrature,
                               "reconstruction_quadrature":reconstruction_quadrature, "max_step":max_step}}
