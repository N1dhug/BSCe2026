"""BM1 demand-only nonlinear least-squares fitting with latent X."""

import numpy as np
from scipy.optimize import least_squares


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
    def _array(values, name):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)): raise ValueError(f"{name} must be a nonempty finite one-dimensional array.")
        return values

    @staticmethod
    def _scalar(value, name):
        value = np.asarray(value, dtype=float)
        if value.size != 1 or not np.all(np.isfinite(value)): raise ValueError(f"{name} must be a finite scalar.")
        return float(value.reshape(-1)[0])

    @staticmethod
    def _sample(signal, time, name):
        if callable(signal):
            try: values = np.asarray(signal(time), dtype=float)
            except (TypeError, ValueError): values = np.array([signal(float(t)) for t in time], dtype=float)
            if values.ndim == 0: values = np.array([signal(float(t)) for t in time], dtype=float)
        else: values = np.asarray(signal, dtype=float)
        if values.shape == (len(time) - 1,): values = np.append(values, values[-1])
        if values.shape != time.shape or not np.all(np.isfinite(values)): raise ValueError(f"{name} must supply {len(time)} or {len(time)-1} finite samples.")
        return values

    @staticmethod
    def logistic(x, slope):
        """Endpoint-normalized decreasing logistic curve: h(0)=1, h(1)=-1."""
        x = np.asarray(x, dtype=float)
        y = np.tanh(0.5 * slope * (0.5 - x)) / np.tanh(0.25 * slope)
        return float(y) if x.ndim == 0 else y

    @classmethod
    def response(cls, slope):
        slope = cls._scalar(slope, "response slope")
        if slope <= 0.0: raise ValueError("Logistic response slope must be positive.")
        return lambda x: cls.logistic(x, slope)

    def _prepare(self, input_data):
        if not isinstance(input_data, dict): raise TypeError("input_data must contain I and U, optionally Time and B.")
        t = self._array(input_data.get("Time", input_data.get("t", self.model.t)), "Time")
        if t.size < 4 or np.any(np.diff(t) <= 0.0): raise ValueError("Time must contain at least four strictly increasing points.")
        if "I" not in input_data or "U" not in input_data: raise ValueError("input_data requires observed demand I and known prices U.")
        I = self._array(input_data["I"], "I")
        if I.size not in (t.size, t.size - 1): raise ValueError("I must contain one sample per time or per time interval.")
        U = self._sample(input_data["U"], t, "U")
        B = self._sample(input_data.get("B", self.model.B), t, "B")
        for name, values in (("I", I), ("U", U), ("B", B)):
            if np.any((values < 0.0) | (values > 1.0)): raise ValueError(f"{name} must be normalized to [0, 1].")
        return {"Time": t, "I": I, "U": U, "B": B, "dt": np.diff(t)}

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
        shape = X[:-1] * (1.0 - X[:-1])
        transitions = valid[:-1] & valid[1:] & (shape >= min_shape)
        indices = np.flatnonzero(transitions)
        if indices.size < 5: raise ValueError("Insufficient invertible interior demand observations to estimate diffusion.")
        innovations = np.array([X[j + 1] - cls._advance(X[j], data["dt"][j], B[j], U[j], p, substeps) for j in indices])
        scaled = innovations / (shape[indices] * np.sqrt(data["dt"][indices]))
        fit = least_squares(lambda v: scaled**2 - v[0], [max(sigma0**2, 1e-8)], bounds=(0.0, np.inf))
        if not fit.success: raise RuntimeError(f"BM1 diffusion least squares failed: {fit.message}")
        return {"sigma_x": float(np.sqrt(max(0.0, fit.x[0]))), "diffusion_fit": fit, "diffusion_samples": int(indices.size), "inferred_X": X}

    def fit(self, input_data, X0, *, C=None, Xi=None, alpha0=2.0, beta0=2.0, k0=1.0, C0=1.0, Xi0=0.5, fit_X0=True, bounds=None, loss="linear", substeps=2, max_nfev=500, estimate_diffusion=False, sigma0=0.1, min_shape=1e-3):
        """Minimize sum((D_model(t)-I_observed(t))**2) from an X0 guess.

        Fit alpha,beta,k,Xi,C,X0; C, Xi or X0 may be fixed. The price
        response and state response use endpoint-normalized logistic functions.
        The optional sigma estimator assumes observed demand has no noise.
        """
        data = self._prepare(input_data)
        X0 = self._scalar(X0, "X0")
        if not 0.0 <= X0 <= 1.0: raise ValueError("X0 must lie in [0, 1].")
        if not isinstance(substeps, int) or isinstance(substeps, bool) or substeps < 1: raise ValueError("substeps must be a positive integer.")
        if not isinstance(max_nfev, int) or max_nfev < 1: raise ValueError("max_nfev must be a positive integer.")
        bds = dict(self.BOUNDS)
        if bounds is not None:
            if not isinstance(bounds, dict) or set(bounds) - set(self.NAMES): raise ValueError("bounds must map BM1 parameter names to intervals.")
            bds.update(bounds)
        bds = {key: tuple(np.asarray(value, dtype=float).reshape(-1)) for key, value in bds.items()}
        for key, pair in bds.items():
            if len(pair) != 2 or not np.all(np.isfinite(pair)) or pair[0] < (0.0 if key == "X0" else 1e-15) or pair[1] <= pair[0]: raise ValueError(f"Invalid bounds for {key}.")
        if bds["Xi"][1] > 1.0: raise ValueError("Xi cannot exceed 1.")
        initial = {"alpha": alpha0, "beta": beta0, "k": k0, "Xi": Xi0, "C": C0, "X0": X0}
        fixed = {key: value for key, value in (("C", C), ("Xi", Xi), ("X0", X0 if not fit_X0 else None)) if value is not None}
        for key, value in fixed.items():
            value = self._scalar(value, key)
            if not bds[key][0] <= value <= bds[key][1]: raise ValueError(f"{key} lies outside fit bounds.")
            fixed[key] = value
        free = [key for key in self.NAMES if key not in fixed]
        if not free: raise ValueError("At least one fitting parameter must remain free.")
        seeds = np.array([np.clip(self._scalar(initial[key], key+"0"), bds[key][0]+1e-9, bds[key][1]-1e-9) for key in free])
        lb = np.array([bds[key][0] for key in free])
        ub = np.array([bds[key][1] for key in free])

        def unpack(theta): return {**fixed, **dict(zip(free, theta))}
        def residual(theta):
            _, predicted = self._predict(data, unpack(theta)["X0"], unpack(theta), substeps)
            return predicted[:data["I"].size] - data["I"]

        result = least_squares(residual, seeds, bounds=(lb, ub), loss=loss, max_nfev=max_nfev)
        if not result.success: raise RuntimeError(f"BM1 demand least squares failed: {result.message}")
        p = unpack(result.x)
        X, D = self._predict(data, p["X0"], p, substeps)
        f, g = self.response(p["alpha"]), self.response(p["beta"])
        extra = {"sigma_x": None, "diffusion_fit": None, "diffusion_samples": 0, "inferred_X": None}
        if estimate_diffusion:
            sigma0, min_shape = self._scalar(sigma0, "sigma0"), self._scalar(min_shape, "min_shape")
            if sigma0 < 0.0 or min_shape <= 0.0: raise ValueError("sigma0 must be nonnegative and min_shape positive.")
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
