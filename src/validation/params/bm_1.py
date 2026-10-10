"""Checks for BM1 demand-only parameter fitting, separate from numerical estimation."""

import numpy as np


class bm_1_params_validation:
    """Validate observations, parameter bounds and optimizer results."""

    @staticmethod
    def array(values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)): raise ValueError(f"{name} must be a nonempty finite one-dimensional array.")
        return values

    @staticmethod
    def scalar(value, name):
        value = np.asarray(value, dtype=float)
        if value.size != 1 or not np.all(np.isfinite(value)): raise ValueError(f"{name} must be a finite scalar.")
        return float(value.reshape(-1)[0])

    @staticmethod
    def sample(signal, time, name):
        if callable(signal):
            try: values = np.asarray(signal(time), dtype=float)
            except (TypeError, ValueError): values = np.array([signal(float(t)) for t in time], dtype=float)
            if values.ndim == 0: values = np.array([signal(float(t)) for t in time], dtype=float)
        else: values = np.asarray(signal, dtype=float)
        if values.shape == (len(time) - 1,): values = np.append(values, values[-1])
        if values.shape != time.shape or not np.all(np.isfinite(values)): raise ValueError(f"{name} must supply {len(time)} or {len(time)-1} finite samples.")
        return values

    @classmethod
    def prepare(cls, model, input_data):
        if not isinstance(input_data, dict): raise TypeError("input_data must contain I and U, optionally Time and B.")
        t = cls.array(input_data.get("Time", input_data.get("t", model.t)), "Time")
        if t.size < 4 or np.any(np.diff(t) <= 0.0): raise ValueError("Time must contain at least four strictly increasing points.")
        if "I" not in input_data or "U" not in input_data: raise ValueError("input_data requires observed demand I and known prices U.")
        I = cls.array(input_data["I"], "I")
        if I.size not in (t.size, t.size - 1): raise ValueError("I must contain one sample per time or per time interval.")
        U = cls.sample(input_data["U"], t, "U")
        B = cls.sample(input_data.get("B", model.B), t, "B")
        for name, values in (("I", I), ("U", U), ("B", B)):
            if np.any((values < 0.0) | (values > 1.0)): raise ValueError(f"{name} must be normalized to [0, 1].")
        return {"Time": t, "I": I, "U": U, "B": B, "dt": np.diff(t)}

    @classmethod
    def response_slope(cls, slope):
        slope = cls.scalar(slope, "response slope")
        if slope <= 0.0: raise ValueError("Logistic response slope must be positive.")
        return slope

    @classmethod
    def fit_inputs(cls, X0, *, C, Xi, fit_X0, initial, bounds, names, default_bounds, substeps, max_nfev):
        X0 = cls.scalar(X0, "X0")
        if not 0.0 <= X0 <= 1.0: raise ValueError("X0 must lie in [0, 1].")
        if not isinstance(fit_X0, (bool, np.bool_)): raise TypeError("fit_X0 must be a boolean.")
        if not isinstance(substeps, int) or isinstance(substeps, bool) or substeps < 1: raise ValueError("substeps must be a positive integer.")
        if not isinstance(max_nfev, int) or isinstance(max_nfev, bool) or max_nfev < 1: raise ValueError("max_nfev must be a positive integer.")
        bds = dict(default_bounds)
        if bounds is not None:
            if not isinstance(bounds, dict) or set(bounds) - set(names): raise ValueError("bounds must map BM1 parameter names to intervals.")
            bds.update(bounds)
        for key, value in list(bds.items()):
            pair = np.asarray(value, dtype=float)
            if pair.shape != (2,) or not np.all(np.isfinite(pair)): raise ValueError(f"Invalid bounds for {key}.")
            lower, upper = map(float, pair)
            if lower < (0.0 if key == "X0" else 1e-15) or upper <= lower: raise ValueError(f"Invalid bounds for {key}.")
            bds[key] = (lower, upper)
        if bds["Xi"][1] > 1.0: raise ValueError("Xi cannot exceed 1.")
        if bds["X0"][1] > 1.0: raise ValueError("X0 bounds must remain inside [0, 1].")
        fixed = {key: value for key, value in (("C", C), ("Xi", Xi), ("X0", X0 if not fit_X0 else None)) if value is not None}
        for key, value in fixed.items():
            value = cls.scalar(value, key)
            if not bds[key][0] <= value <= bds[key][1]: raise ValueError(f"{key} lies outside fit bounds.")
            fixed[key] = value
        free = [key for key in names if key not in fixed]
        if not free: raise ValueError("At least one fitting parameter must remain free.")
        seeds = np.array([np.clip(cls.scalar(initial[key], key + "0"), bds[key][0] + 1e-9, bds[key][1] - 1e-9) for key in free])
        lb = np.array([bds[key][0] for key in free])
        ub = np.array([bds[key][1] for key in free])
        return X0, free, fixed, seeds, lb, ub

    @classmethod
    def fit_many_inputs(cls, model, datasets, X0, *, C, Xi, alpha0, beta0, k0, C0, Xi0, max_iterations, min_improvement, max_nfev, state_maxiter, substeps, names, bounds):
        """Validate pooled training data and optimizer configuration."""
        if not isinstance(datasets, (list, tuple)) or len(datasets) != 5: raise ValueError("BM1 pooled weekday fit requires exactly five complete training days.")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1: raise ValueError("max_iterations must be a positive integer.")
        if isinstance(state_maxiter, bool) or not isinstance(state_maxiter, int) or state_maxiter < 1: raise ValueError("state_maxiter must be a positive integer.")
        min_improvement = cls.scalar(min_improvement, "min_improvement")
        if min_improvement < 0.0: raise ValueError("min_improvement must be nonnegative.")
        prepared = [cls.prepare(model, data) for data in datasets]
        if any(len(data["I"]) != 24 or len(data["Time"]) != 25 for data in prepared): raise ValueError("Each fitting day must contain 24 hourly demand samples and a 25-point hourly time grid.")
        if any(not np.allclose(data["dt"], 1.0, atol=1e-10, rtol=0.0) for data in prepared): raise ValueError("Each fitting day must use a one-hour time step.")
        x0, free, fixed, seeds, lb, ub = cls.fit_inputs(
            X0, C=C, Xi=Xi, fit_X0=False, initial={"alpha": alpha0, "beta": beta0, "k": k0, "Xi": Xi0, "C": C0, "X0": X0},
            bounds=None, names=names, default_bounds=bounds, substeps=substeps, max_nfev=max_nfev,
        )
        fixed.pop("X0")
        return prepared, x0, free, fixed, seeds, lb, ub

    @classmethod
    def diffusion_controls(cls, sigma0, min_shape):
        sigma0, min_shape = cls.scalar(sigma0, "sigma0"), cls.scalar(min_shape, "min_shape")
        if sigma0 < 0.0 or min_shape <= 0.0: raise ValueError("sigma0 must be nonnegative and min_shape positive.")
        return sigma0, min_shape

    @staticmethod
    def diffusion_indices(valid, X, min_shape):
        shape = X[:-1] * (1.0 - X[:-1])
        transitions = valid[:-1] & valid[1:] & (shape >= min_shape)
        indices = np.flatnonzero(transitions)
        if indices.size < 5: raise ValueError("Insufficient invertible interior demand observations to estimate diffusion.")
        return indices, shape

    @staticmethod
    def optimizer_result(result, name):
        if not result.success: raise RuntimeError(f"BM1 {name} least squares failed: {result.message}")
        return result

    @staticmethod
    def capped_optimizer_result(result, name, *, limit_status):
        """Accept usable bounded optimizer iterates at an iteration/evaluation cap.

        Only the documented optimizer limit status is nonfatal. Invalid
        parameters/objective values and all other failures remain errors.
        """
        point = np.asarray(result.x, dtype=float)
        objective = np.asarray(result.fun, dtype=float)
        if not point.size or not np.all(np.isfinite(point)) or not objective.size or not np.all(np.isfinite(objective)):
            raise RuntimeError(f"BM1 {name} produced nonfinite optimizer values: {result.message}")
        status, success = int(result.status), bool(result.success)
        capped = not success and status == limit_status
        if not (success or capped): raise RuntimeError(f"BM1 {name} optimizer failed: {result.message}")
        return {
            "converged": success, "limit_reached": capped, "status": status,
            "nfev": int(getattr(result, "nfev", 0)), "message": str(result.message),
        }

    @staticmethod
    def historical_diffusion_data(datasets, states, fit):
        if not isinstance(datasets, (tuple, list)) or len(datasets) != 5:
            raise ValueError("Historical BM1 diffusion requires five fitting windows.")
        states = np.asarray(states, dtype=float)
        if states.shape != (5,) or not np.all(np.isfinite(states)) or np.any((states < 0.0) | (states > 1.0)):
            raise ValueError("Diffusion estimator requires five fitted initial states in [0,1].")
        for key in ("alpha", "beta", "k", "Xi", "C"):
            if key not in fit or not np.isfinite(fit[key]) or fit[key] <= 0.0:
                raise ValueError(f"Invalid fitted parameter {key}.")
        return states

    @staticmethod
    def diffusion_variance_fit(kernel, squared_residual, noise_floor):
        kernel, squared_residual = np.asarray(kernel, dtype=float), np.asarray(squared_residual, dtype=float)
        if kernel.shape != squared_residual.shape or kernel.size < 20 or not np.all(np.isfinite(kernel)) or not np.all(np.isfinite(squared_residual)):
            raise ValueError("Historical BM1 variance calibration requires >=20 finite kernel/residual pairs.")
        if np.max(kernel) < 1e-12 or not np.isfinite(noise_floor) or noise_floor < 0:
            raise ValueError("BM1 diffusion coefficient cannot be identified from this historical dataset.")
        return kernel, squared_residual
