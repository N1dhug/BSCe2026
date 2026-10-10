"""BM1 demand-only nonlinear least-squares fitting with latent X."""

import numpy as np
from scipy.optimize import least_squares

from src.validation.params.bm_1 import bm_1_params_validation


class lsq_params:
    """Estimate BM1 response shapes and drift from observed demand.

    The state X is propagated from an X0 guess (fitted by default). The deterministic
    demand residual does not identify sigma_x; optional diffusion estimation
    inverts the demand map under a noise-free observation assumption.
    """

    NAMES = ("alpha", "beta", "k", "Xi", "C", "X0")
    BOUNDS = {"alpha": (0.05, 20.0), "beta": (0.05, 20.0), "k": (0.05, 12.0), "Xi": (1e-4, 1.0), "C": (1e-3, 100.0), "X0": (0.0, 1.0)}

    def __init__(self, model): self.model = model

    @staticmethod
    def logistic(x, slope):
        """Endpoint-normalized decreasing logistic curve: h(0)=1, h(1)=-1."""
        x = np.asarray(x, dtype=float)
        y = np.tanh(0.5 * slope * (0.5 - x)) / np.tanh(0.25 * slope)
        return float(y) if x.ndim == 0 else y

    @classmethod
    def response(cls, slope):
        slope = bm_1_params_validation.response_slope(slope)
        return lambda x: cls.logistic(x, slope)

    @classmethod
    def _demand(cls, x, b, u, p):
        delta = np.tanh(0.5 * p["k"] * (cls.logistic(x, p["alpha"]) + cls.logistic(u, p["beta"])))
        return b + p["Xi"] * delta * (1.0 - b if delta >= 0.0 else b)

    @classmethod
    def _rate(cls, x, b, u, p): return (cls._demand(x, b, u, p) - b) / p["C"]

    @classmethod
    def _advance(cls, x, dt, b, u, p, substeps):
        steps = max(substeps, int(np.ceil(dt * p["Xi"] / (0.2 * p["C"]))))
        h = dt / steps
        for _ in range(steps):
            k1 = cls._rate(x, b, u, p)
            k2 = cls._rate(x + 0.5 * h * k1, b, u, p)
            k3 = cls._rate(x + 0.5 * h * k2, b, u, p)
            k4 = cls._rate(x + h * k3, b, u, p)
            x += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return x

    @classmethod
    def _predict(cls, data, X0, p, substeps):
        t, B, U = data["Time"], data["B"], data["U"]
        X = np.empty(t.size, dtype=float)
        D = np.empty(t.size, dtype=float)
        X[0] = X0
        for i in range(t.size):
            D[i] = cls._demand(X[i], B[i], U[i], p)
            if i < t.size - 1: X[i + 1] = cls._advance(X[i], t[i + 1] - t[i], B[i], U[i], p, substeps)
        return X, D

    @staticmethod
    def _price(input_data, data):
        if callable(input_data["U"]): return input_data["U"]
        time, price = data["Time"], data["U"].copy()

        def U(t):
            query = np.asarray(t, dtype=float)
            index = np.clip(np.searchsorted(time, query, side="right") - 1, 0, len(price) - 1)
            value = price[index]
            return float(value) if query.ndim == 0 else value

        return U

    @classmethod
    def _diffusion(cls, data, p, sigma0, substeps, min_shape):
        """Infer X from D assuming noiseless consumption measurements.

        This is a conditional Euler innovation estimate, NOT identified by the
        deterministic demand LSQ and not valid with unknown observation noise.
        """
        I, B, U = data["I"], data["B"][:data["I"].size], data["U"][:data["I"].size]
        change = I - B
        denom = p["Xi"] * np.where(change >= 0.0, 1.0 - B, B)
        valid = denom > 1e-12
        delta = np.full(I.shape, np.nan)
        delta[valid] = change[valid] / denom[valid]
        valid &= np.isfinite(delta) & (np.abs(delta) < 1.0 - 1e-10)
        f_target = np.full(I.shape, np.nan)
        f_target[valid] = 2.0 * np.arctanh(delta[valid]) / p["k"] - cls.logistic(U[valid], p["beta"])
        valid &= np.isfinite(f_target) & (np.abs(f_target) <= 1.0 + 1e-8)
        X = np.full(I.shape, np.nan)
        x_arg = np.clip(f_target[valid] * np.tanh(p["alpha"] / 4.0), -1.0 + 1e-12, 1.0 - 1e-12)
        X[valid] = 0.5 - 2.0 * np.arctanh(x_arg) / p["alpha"]
        valid &= (X >= -1e-8) & (X <= 1.0 + 1e-8)
        indices, shape = bm_1_params_validation.diffusion_indices(valid, X, min_shape)
        innovations = np.array([X[j + 1] - cls._advance(X[j], data["dt"][j], B[j], U[j], p, substeps) for j in indices])
        scaled = innovations / (shape[indices] * np.sqrt(data["dt"][indices]))
        fit = least_squares(lambda v: scaled**2 - v[0], [max(sigma0**2, 1e-8)], bounds=(0.0, np.inf))
        bm_1_params_validation.optimizer_result(fit, "diffusion")
        return {"sigma_x": float(np.sqrt(max(0.0, fit.x[0]))), "diffusion_fit": fit, "diffusion_samples": int(indices.size), "inferred_X": X}

    def fit(self, input_data, X0, *, C=None, Xi=None, alpha0=2.0, beta0=2.0, k0=1.0, C0=1.0, Xi0=0.5, fit_X0=True, bounds=None, loss="linear", substeps=2, max_nfev=500, estimate_diffusion=False, sigma0=0.1, min_shape=1e-3):
        """Minimize sum((D_model(t)-I_observed(t))**2) from an X0 guess.

        Fit alpha,beta,k,Xi,C,X0; C, Xi or X0 may be fixed. The price
        response and state response use endpoint-normalized logistic functions.
        The optional sigma estimator assumes observed demand has no noise.
        """
        data = bm_1_params_validation.prepare(self.model, input_data)
        X0, free, fixed, seeds, lb, ub = bm_1_params_validation.fit_inputs(
            X0, C=C, Xi=Xi, fit_X0=fit_X0, bounds=bounds, names=self.NAMES, default_bounds=self.BOUNDS,
            substeps=substeps, max_nfev=max_nfev,
            initial={"alpha": alpha0, "beta": beta0, "k": k0, "Xi": Xi0, "C": C0, "X0": X0},
        )

        def unpack(theta): return {**fixed, **dict(zip(free, theta))}
        def residual(theta):
            _, predicted = self._predict(data, unpack(theta)["X0"], unpack(theta), substeps)
            return predicted[:data["I"].size] - data["I"]

        result = least_squares(residual, seeds, bounds=(lb, ub), loss=loss, max_nfev=max_nfev)
        bm_1_params_validation.optimizer_result(result, "demand")
        p = unpack(result.x)
        X, D = self._predict(data, p["X0"], p, substeps)
        f, g = self.response(p["alpha"]), self.response(p["beta"])
        extra = {"sigma_x": None, "diffusion_fit": None, "diffusion_samples": 0, "inferred_X": None}
        if estimate_diffusion:
            sigma0, min_shape = bm_1_params_validation.diffusion_controls(sigma0, min_shape)
            extra = self._diffusion(data, p, sigma0, substeps, min_shape)
        error = D[:len(data["I"])] - data["I"]
        return {
            **p, "gain": float(p["Xi"] / p["C"]), "X0": float(p["X0"]), "X0_seed": X0, "f": f, "g": g,
            "parameters": [self._price(input_data, data), f, g, float(p["C"]), float(p["k"]), float(p["Xi"])],
            "Time": data["Time"].copy(), "X": X, "D": D[:len(data["I"])], "I_data": data["I"].copy(),
            "mse": float(np.mean(error**2)), "rmse": float(np.sqrt(np.mean(error**2))),
            "drift_fit": result, "fit_method": "demand_lsq", "free_parameters": tuple(free),
            **extra,
        }

    def fit_many(self, datasets, X0=0.5, *, C=None, Xi=None, alpha0=2.0, beta0=2.0, k0=1.0, C0=1.0, Xi0=0.5, max_iterations=6, min_improvement=1e-6, max_nfev=150, state_maxiter=100, substeps=2, loss="linear", fit_bounds=None):
        """Alternating pooled-theta / separate-X0 demand LSQ for matching days.

        First fit one common theta from five days initialized at the same X0.
        Then minimize each day's demand MSE over only that day's X0, with
        theta fixed. Subsequent cycles refit theta jointly holding the five
        states fixed, and repeat per-day state optimization. Keep the best
        complete cycle and stop when its SSE no longer improves.
        """
        from scipy.optimize import minimize_scalar

        prepared, X0, free, fixed, seed, lower, upper = bm_1_params_validation.fit_many_inputs(
            self.model, datasets, X0, C=C, Xi=Xi, alpha0=alpha0, beta0=beta0,
            k0=k0, C0=C0, Xi0=Xi0, max_iterations=max_iterations,
            min_improvement=min_improvement, max_nfev=max_nfev,
            state_maxiter=state_maxiter, substeps=substeps,
            names=self.NAMES, bounds=self.BOUNDS, fit_bounds=fit_bounds,
        )
        def unpack(theta): return {**fixed, **dict(zip(free, theta))}
        def residual(theta, states):
            p = unpack(theta)
            return np.concatenate([self._predict(data, float(x), p, substeps)[1][:len(data["I"])] - data["I"] for data, x in zip(prepared, states)])
        def refine_states(p, initial):
            states, state_fits, state_reports = [], [], []
            for data, x in zip(prepared, initial):
                def error(candidate):
                    difference = self._predict(data, float(candidate), p, substeps)[1][:len(data["I"])] - data["I"]
                    return float(difference @ difference)
                fit = minimize_scalar(error, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6, "maxiter": state_maxiter})
                report = bm_1_params_validation.capped_optimizer_result(fit, "initial-state", limit_status=1)
                # Even a capped scalar search must not worsen the candidate state.
                states.append(float(fit.x) if error(fit.x) <= error(x) else float(x))
                state_fits.append(fit); state_reports.append(report)
            return np.asarray(states), state_fits, state_reports

        states = np.full(len(prepared), X0, dtype=float)
        history, best = [], None
        theta = seed.copy()
        stop_reason = "max_iterations"
        for iteration in range(1, max_iterations + 1):
            theta_fit = least_squares(lambda values: residual(values, states), theta, bounds=(lower, upper), loss=loss, max_nfev=max_nfev)
            theta_report = bm_1_params_validation.capped_optimizer_result(theta_fit, "pooled demand", limit_status=0)
            candidate_theta = theta_fit.x.copy()
            p = unpack(candidate_theta)
            candidate_states, state_fits, state_reports = refine_states(p, states)
            differences = residual(candidate_theta, candidate_states)
            sse = float(differences @ differences)
            previous = np.inf if best is None else best["sse"]
            threshold = min_improvement * max(previous, 1e-10)
            improved = sse < previous
            state_limit = any(report["limit_reached"] for report in state_reports)
            history.append({
                "iteration": iteration, "sse": sse, "mse": float(np.mean(differences ** 2)),
                "improvement": float(previous - sse) if best is not None else None, "accepted": improved,
                "theta_converged": theta_report["converged"], "theta_nfev": theta_report["nfev"],
                "theta_status": theta_report["status"], "theta_message": theta_report["message"],
                "state_converged": not state_limit, "state_maxiter_reached": state_limit,
            })
            if not improved:
                stop_reason = "max_nfev" if theta_report["limit_reached"] else ("state_maxiter" if state_limit else "no_improvement")
                break
            best = {
                "theta": candidate_theta, "params": p, "states": candidate_states,
                "state_fits": state_fits, "state_reports": state_reports, "theta_fit": theta_fit,
                "theta_report": theta_report, "sse": sse, "mse": float(np.mean(differences ** 2)),
            }
            theta, states = candidate_theta, candidate_states
            if theta_report["limit_reached"]:
                stop_reason = "max_nfev"
                break
            if state_limit:
                stop_reason = "state_maxiter"
                break
            if iteration > 1 and previous - sse <= threshold:
                stop_reason = "min_improvement"
                break

        converged = stop_reason in ("min_improvement", "no_improvement")
        messages = {
            "max_nfev": f"BM1 pooled LSQ reached max_nfev={max_nfev} without convergence; best finite fit retained.",
            "state_maxiter": f"BM1 initial-state optimizer reached state_maxiter={state_maxiter}; best finite fit retained.",
            "max_iterations": f"BM1 alternating fit reached max_iterations={max_iterations} without meeting its improvement stopping criterion; best fit retained.",
            "min_improvement": "BM1 alternating fit stopped after improvement fell below tolerance.",
            "no_improvement": "BM1 alternating fit stopped because the objective did not improve.",
        }
        parameters = dict(best["params"])
        f, g = self.response(parameters["alpha"]), self.response(parameters["beta"])
        return {
            **parameters, "gain": float(parameters["Xi"] / parameters["C"]),
            "f": f, "g": g, "initial_states": best["states"].copy(), "X0_seed": float(X0),
            "initial_weights": np.full(len(prepared), 1.0 / len(prepared)),
            "mse": best["mse"], "sse": best["sse"], "history": history,
            "theta_fit": best["theta_fit"], "state_fits": best["state_fits"],
            "iterations": len(history), "free_parameters": tuple(free),
            "converged": converged, "stop_reason": stop_reason, "message": messages[stop_reason],
            "warning": None if converged else messages[stop_reason],
            "theta_converged": best["theta_report"]["converged"],
            "theta_nfev": best["theta_report"]["nfev"],
            "state_converged": all(report["converged"] for report in best["state_reports"]),
            "fit_method": "pooled_demand_alternating_lsq",
        }


class historical_diffusion_params:
    """Estimate BM1 sigma_x from five fitted 24-hour demand trajectories.

    Method: linear-noise moment approximation about each deterministic state:
    V[k+1] = (dX[k+1]/dX[k])² V[k] + sigma_x² * X[k]²(1-X[k])² dt.
    Implied demand variance is (dD/dX)² V. Regress squared historical
    demand residuals minus an observational variance floor on the resulting
    unit-sigma² kernels, subject to sigma_x² >= 0.

    This is a process-variance proxy, not an observation-noise-identified
    SDE MLE. Structural model error can inflate the inferred sigma_x.
    No target-window observations participate in calibration.
    """

    def fit_many(self, datasets, fit, *, substeps=2, derivative_step=1e-4, minimum_observation_sd=0.005):
        states = bm_1_params_validation.historical_diffusion_data(datasets, fit["initial_states"], fit)
        kernels, squares, initial_errors, all_errors = [], [], [], []
        for data, x0 in zip(datasets, states):
            X, D = lsq_params._predict(data, float(x0), fit, substeps)
            I = np.asarray(data["I"], dtype=float)
            error = I - D[:len(I)]
            initial_errors.append(float(error[0]))
            all_errors.extend(error.tolist())
            variance = 0.0
            for k in range(len(I) - 1):
                x, b, u, dt = float(X[k]), float(data["B"][k]), float(data["U"][k]), float(data["Time"][k + 1] - data["Time"][k])
                plus = lsq_params._advance(x + derivative_step, dt, b, u, fit, substeps)
                minus = lsq_params._advance(x - derivative_step, dt, b, u, fit, substeps)
                slope = (plus - minus) / (2.0 * derivative_step)
                variance = slope * slope * variance + (x * (1.0 - x)) ** 2 * dt
                next_x = float(X[k + 1])
                j = k + 1
                demand_plus = lsq_params._demand(next_x + derivative_step, float(data["B"][j]), float(data["U"][j]), fit)
                demand_minus = lsq_params._demand(next_x - derivative_step, float(data["B"][j]), float(data["U"][j]), fit)
                sensitivity = (demand_plus - demand_minus) / (2.0 * derivative_step)
                kernels.append(float(sensitivity * sensitivity * variance))
                squares.append(float(error[j] ** 2))

        observation_sd = max(float(minimum_observation_sd), float(np.sqrt(np.mean(np.square(initial_errors)))))
        observed_variance = observation_sd ** 2
        kernel, residual_square = bm_1_params_validation.diffusion_variance_fit(kernels, squares, observed_variance)
        target_variance = np.maximum(residual_square - observed_variance, 0.0)
        q = float(max(0.0, np.dot(kernel, target_variance) / np.dot(kernel, kernel)))
        return {
            "sigma_x": float(np.sqrt(q)), "q": q,
            "observation_sd": observation_sd,
            "historical_rmse": float(np.sqrt(np.mean(np.square(all_errors)))),
            "variance_rmse": float(np.sqrt(np.mean(np.square(kernel * q - target_variance)))),
            "samples": int(kernel.size),
            "method": "historical linear-noise nonnegative variance regression",
        }
