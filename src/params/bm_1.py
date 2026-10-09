"""Least-squares estimation of BM1 drift and Ito diffusion from observed X paths."""

import numpy as np
from scipy.optimize import least_squares

from src.validation.model.bm import bm_validation


class lsq_params:
    """Fit BM1's gain=Xi/C, eagerness k and diffusion sigma_x.

    Requires observations of the latent state X, known baseline B, prices U,
    and fixed response functions f, g. Without a known C or Xi, only their
    ratio Xi/C is identifiable from drift. Diffusion assumes negligible
    observation noise and short Euler observation intervals.
    """

    def __init__(self, model): self.model = model

    @staticmethod
    def _array(values, name, n=None):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)): raise ValueError(f"{name} must be a finite one-dimensional array.")
        if n is not None and values.size != n: raise ValueError(f"{name} must contain {n} samples.")
        return values

    @staticmethod
    def _samples(signal, points, name):
        if callable(signal):
            try: values = np.asarray(signal(points), dtype=float)
            except (ValueError, TypeError): values = np.array([float(signal(float(x))) for x in points])
            if values.ndim == 0: values = np.array([float(signal(float(x))) for x in points])
        else: values = np.asarray(signal, dtype=float)
        if values.ndim != 1 or values.size not in (points.size, points.size + 1) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must evaluate to {points.size} (or {points.size + 1}) finite samples.")
        return values[:points.size]

    @staticmethod
    def _scalar(value, name, *, positive=False, unit=False):
        value = np.asarray(value, dtype=float)
        if value.size != 1 or not np.all(np.isfinite(value)): raise ValueError(f"{name} must be a finite scalar.")
        value = float(value.reshape(-1)[0])
        if positive and value <= 0.0: raise ValueError(f"{name} must be positive.")
        if unit and not 0.0 <= value <= 1.0: raise ValueError(f"{name} must lie in [0, 1].")
        return value

    def _prepare(self, input_data, f, g):
        if not isinstance(input_data, dict): raise TypeError("input_data must be a dictionary with Time, X and U.")
        t = self._array(input_data.get("Time", input_data.get("t", self.model.t)), "Time")
        if t.size < 4 or np.any(np.diff(t) <= 0.0): raise ValueError("Time must contain at least four strictly increasing observations.")
        if "X" not in input_data: raise ValueError("Observed BM1 state X is required; input demand alone cannot identify the state diffusion.")
        X = self._array(input_data["X"], "X", t.size)
        if np.any((X < 0.0) | (X > 1.0)): raise ValueError("Observed states X must lie in [0, 1].")
        if "U" not in input_data: raise ValueError("input_data must provide U prices or a price callable.")
        bm_validation.response_endpoints(f, g)
        dt, points = np.diff(t), t[:-1]
        U = self._samples(input_data["U"], points, "U")
        B = self._samples(input_data.get("B", self.model.B), points, "B")
        if np.any((U < 0.0) | (U > 1.0)): raise ValueError("U prices must lie in [0, 1].")
        if np.any((B < 0.0) | (B > 1.0)): raise ValueError("B baseline values must lie in [0, 1].")
        fX, gU = self._samples(f, X[:-1], "f(X)"), self._samples(g, U, "g(U)")
        if np.any((fX < -1.0) | (fX > 1.0)) or np.any((gU < -1.0) | (gU > 1.0)): raise ValueError("Response values must lie in [-1, 1].")
        return {"Time": t, "X": X, "U": U, "B": B, "dt": dt, "dx": np.diff(X), "response": fX + gU}

    @staticmethod
    def _drift(gain, k, data):
        delta = np.tanh(0.5 * k * data["response"])
        weight = np.where(delta >= 0.0, 1.0 - data["B"], data["B"])
        return gain * delta * weight

    @staticmethod
    def _price_callable(input_data, data):
        if callable(input_data["U"]): return input_data["U"]
        time, values = data["Time"][:-1], data["U"].copy()

        def U(t):
            query = np.asarray(t, dtype=float)
            indices = np.clip(np.searchsorted(time, query, side="right") - 1, 0, values.size - 1)
            result = values[indices]
            return float(result) if query.ndim == 0 else result

        return U

    def fit(self, input_data, f, g, *, C=None, Xi=None, k0=1.0, gain0=0.3, sigma0=0.1, k_bounds=(1e-3, 20.0), min_shape=1e-3, loss="linear", max_nfev=1000):
        """Fit Euler conditional mean, then conditional variance with SciPy LSQ.

        input_data requires X and U; Time (or t) defaults to model.t. Optional
        B overrides model.B. U/B may be sampled arrays or callable signals.
        Returned 'parameters' is [U,f,g,C,k,Xi] if C and Xi are identifiable.
        """
        data = self._prepare(input_data, f, g)
        if C is not None: C = self._scalar(C, "C", positive=True)
        if Xi is not None: Xi = self._scalar(Xi, "Xi", unit=True)
        if Xi == 0.0: raise ValueError("Xi=0 removes the drift response, so k cannot be fitted.")
        k0, gain0 = self._scalar(k0, "k0", positive=True), self._scalar(gain0, "gain0", positive=True)
        sigma0 = self._scalar(sigma0, "sigma0")
        if sigma0 < 0.0: raise ValueError("sigma0 must be nonnegative.")
        k_bounds = self._array(k_bounds, "k_bounds", 2)
        if k_bounds[0] <= 0.0 or k_bounds[1] <= k_bounds[0]: raise ValueError("k_bounds must be positive and increasing.")
        min_shape = self._scalar(min_shape, "min_shape", positive=True)
        fixed_gain = C is not None and Xi is not None
        gain_max = 1.0 / C if C is not None else np.inf
        gain_min = 1e-12 if Xi is not None and C is None else 0.0

        if fixed_gain:
            gain = Xi / C
            x0 = np.array([np.clip(k0, k_bounds[0] + 1e-9, k_bounds[1] - 1e-9)])
            lb, ub = k_bounds[:1], k_bounds[1:]
            def drift_residual(v): return (data["dx"] - self._drift(gain, v[0], data) * data["dt"]) / np.sqrt(data["dt"])
        else:
            seed = min(gain0, gain_max * 0.5) if np.isfinite(gain_max) else gain0
            x0 = np.array([max(seed, gain_min + 1e-10), np.clip(k0, k_bounds[0] + 1e-9, k_bounds[1] - 1e-9)])
            lb, ub = np.array([gain_min, k_bounds[0]]), np.array([gain_max, k_bounds[1]])
            def drift_residual(v): return (data["dx"] - self._drift(v[0], v[1], data) * data["dt"]) / np.sqrt(data["dt"])

        drift_fit = least_squares(drift_residual, x0, bounds=(lb, ub), loss=loss, max_nfev=max_nfev)
        if not drift_fit.success: raise RuntimeError(f"BM1 drift LSQ failed: {drift_fit.message}")
        if fixed_gain: k = float(drift_fit.x[0])
        else: gain, k = map(float, drift_fit.x)
        if C is not None and Xi is None: Xi = float(C * gain)
        elif Xi is not None and C is None: C = float(Xi / gain)

        drift = self._drift(gain, k, data)
        increments = data["dx"] - drift * data["dt"]
        shape = data["X"][:-1] * (1.0 - data["X"][:-1])
        valid = shape >= min_shape
        if np.count_nonzero(valid) < 3: raise ValueError("Too few interior X observations to estimate diffusion; X(1-X) vanishes at boundaries.")
        scaled = increments[valid] / (shape[valid] * np.sqrt(data["dt"][valid]))
        variance_fit = least_squares(lambda v: scaled**2 - v[0], [max(sigma0**2, 1e-10)], bounds=([0.0], [np.inf]), max_nfev=max_nfev)
        if not variance_fit.success: raise RuntimeError(f"BM1 diffusion LSQ failed: {variance_fit.message}")
        sigma_x = float(np.sqrt(max(0.0, variance_fit.x[0])))
        return {
            "gain": float(gain), "k": float(k), "sigma_x": sigma_x, "C": C, "Xi": Xi,
            "parameters": [self._price_callable(input_data, data), f, g, C, float(k), Xi] if C is not None and Xi is not None else None,
            "drift_fit": drift_fit, "diffusion_fit": variance_fit,
            "drift_increment_mse": float(np.mean(increments**2)), "drift_scaled_mse": float(np.mean(increments**2 / data["dt"])),
            "diffusion_samples": int(np.count_nonzero(valid)), "observations": int(data["X"].size),
            "identifiable": {"C": C is not None, "Xi": Xi is not None},
        }
