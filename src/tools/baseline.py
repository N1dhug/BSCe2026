import numpy as np
import pandas as pd

from src.tools.impute_hourly import impute_hourly_days


class statistical_output:
    """Robust exogenous WM4 baseline O(time, weekday, temperature)."""

    def __init__(self, base_temperature=18.0, hour_harmonics=4, season_harmonics=2, ridge=1e-3, huber_delta=1.5, max_iter=50, min_samples=168):
        self.base_temperature = float(base_temperature)
        self.hour_harmonics = int(hour_harmonics)
        self.season_harmonics = int(season_harmonics)
        self.ridge = float(ridge)
        self.huber_delta = float(huber_delta)
        self.max_iter = int(max_iter)
        self.min_samples = int(min_samples)
        if self.min_samples < 2: raise ValueError("min_samples must be at least 2.")
        if self.hour_harmonics < 1 or self.season_harmonics < 0: raise ValueError("Invalid statistical_output harmonic count.")
        if self.ridge < 0.0 or self.huber_delta <= 0.0 or self.max_iter < 1: raise ValueError("Invalid statistical_output fit settings.")
        self.coef = None
        self.feature_mean = None
        self.feature_scale = None
        self.metadata = None
        self.imputation = None

    @staticmethod
    def _index(index):
        index = pd.DatetimeIndex(index)
        if index.empty: raise ValueError("statistical_output requires a non-empty datetime index.")
        return index

    @staticmethod
    def _temperature(temperature, n):
        values = np.asarray(temperature, dtype=float).reshape(-1)
        if values.size != n or not np.all(np.isfinite(values)): raise ValueError("Temperature must contain one finite value per timestamp.")
        return values

    def _features(self, index, temperature):
        index = self._index(index)
        temperature = self._temperature(temperature, len(index))
        hour = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60.0 + index.second.to_numpy(dtype=float) / 3600.0
        hour_phase = 2.0 * np.pi * hour / 24.0
        day = index.dayofyear.to_numpy(dtype=float)
        year_phase = 2.0 * np.pi * (day - 1.0) / 365.25
        chill = np.maximum(0.0, self.base_temperature - temperature)

        columns = [np.ones(len(index), dtype=float)]
        hour_terms = []
        for harmonic in range(1, self.hour_harmonics + 1):
            sin_h = np.sin(harmonic * hour_phase)
            cos_h = np.cos(harmonic * hour_phase)
            columns.extend((sin_h, cos_h))
            hour_terms.extend((sin_h, cos_h))

        weekday = index.dayofweek.to_numpy()
        weekday_terms = [(weekday == value).astype(float) for value in range(1, 7)]
        columns.extend(weekday_terms)
        for indicator in weekday_terms:
            for term in hour_terms[:4]: columns.append(indicator * term)

        for harmonic in range(1, self.season_harmonics + 1):
            columns.extend((np.sin(harmonic * year_phase), np.cos(harmonic * year_phase)))

        columns.extend((chill, chill * chill))
        for term in hour_terms[:4]: columns.append(chill * term)
        return np.column_stack(columns)

    def _standardize_fit(self, X):
        mean = np.zeros(X.shape[1], dtype=float)
        scale = np.ones(X.shape[1], dtype=float)
        if X.shape[1] > 1:
            mean[1:] = np.mean(X[:, 1:], axis=0)
            scale[1:] = np.std(X[:, 1:], axis=0)
            scale[1:][scale[1:] < 1e-10] = 1.0
        self.feature_mean, self.feature_scale = mean, scale
        return (X - mean) / scale

    def _standardize(self, X):
        if self.feature_mean is None or self.feature_scale is None: raise RuntimeError("statistical_output must be fitted before prediction.")
        return (X - self.feature_mean) / self.feature_scale

    def fit(self, frame, end):
        if not isinstance(frame, pd.DataFrame): raise TypeError("frame must be a pandas DataFrame.")
        missing = [name for name in ("I", "Temp") if name not in frame.columns]
        if missing: raise ValueError("frame is missing required columns: " + ", ".join(missing) + ".")
        end = pd.Timestamp(end)
        source = frame.loc[frame.index < end, ["I", "Temp"]]
        train, self.imputation = impute_hourly_days(source, columns=("I", "Temp"))
        if len(train) < self.min_samples: raise ValueError(f"At least {self.min_samples} historical I/temperature observations are required to fit O.")

        X_raw = self._features(train.index, train["Temp"].to_numpy(dtype=float))
        X = self._standardize_fit(X_raw)
        y = train["I"].to_numpy(dtype=float)
        penalty = self.ridge * np.eye(X.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        weights = np.ones(len(y), dtype=float)
        beta = np.zeros(X.shape[1], dtype=float)
        iterations = 0

        for iterations in range(1, self.max_iter + 1):
            root = np.sqrt(weights)
            Xw, yw = X * root[:, None], y * root
            next_beta = np.linalg.solve(Xw.T @ Xw + penalty, Xw.T @ yw)
            residual = y - X @ next_beta
            center = np.median(residual)
            scale = 1.4826 * np.median(np.abs(residual - center))
            beta = next_beta
            if not np.isfinite(scale) or scale < 1e-10: break
            limit = self.huber_delta * scale
            absolute = np.abs(residual)
            next_weights = np.ones_like(weights)
            outside = absolute > limit
            next_weights[outside] = limit / absolute[outside]
            if np.max(np.abs(next_weights - weights)) < 1e-6:
                weights = next_weights
                break
            weights = next_weights

        self.coef = beta
        fitted = np.clip(X @ beta, 0.0, 1.0)
        residual = y - fitted
        self.metadata = {
            "fit_start": train.index.min(), "fit_end": train.index.max(), "samples": int(len(train)),
            "features": int(X.shape[1]), "iterations": int(iterations), "ridge": self.ridge,
            "huber_delta": self.huber_delta, "base_temperature": self.base_temperature,
            "temperature_min": float(train["Temp"].min()), "temperature_max": float(train["Temp"].max()),
            "mse": float(np.mean(residual * residual)), "mae": float(np.mean(np.abs(residual))),
            "imputation": dict(self.imputation),
        }
        return self

    def predict(self, index, temperature):
        if self.coef is None: raise RuntimeError("statistical_output must be fitted before prediction.")
        X = self._standardize(self._features(index, temperature))
        return np.clip(X @ self.coef, 0.0, 1.0)



class weekly_average:
    """Historical weekday/hour means O(t)=L(weekday, hour)."""

    def __init__(self, min_samples=168):
        self.coef = None
        self.metadata = None
        self.levels = None
        self.imputation = None
        self.min_samples = int(min_samples)
        if self.min_samples < 2: raise ValueError("min_samples must be at least 2.")
        self.base_temperature = 18.0
        self.beta_C = 0.0
        self.chill_center = 0.0

    @staticmethod
    def _index(index):
        index = pd.DatetimeIndex(index)
        if index.empty or index.hasnans: raise ValueError("A non-empty datetime index without NaT is required.")
        return index

    @staticmethod
    def _temperature(temperature, n):
        values = np.asarray(temperature, dtype=float).reshape(-1)
        if values.size != n or not np.all(np.isfinite(values)): raise ValueError("Temperature must contain one finite value per timestamp.")
        return values

    @staticmethod
    def _groups(index):
        return index.dayofweek.to_numpy(dtype=int) * 24 + index.hour.to_numpy(dtype=int)

    @staticmethod
    def _profile(groups, values):
        counts = np.bincount(groups, minlength=168)
        sums = np.bincount(groups, weights=values, minlength=168)
        hours = groups % 24
        hour_counts = np.bincount(hours, minlength=24)
        hour_sums = np.bincount(hours, weights=values, minlength=24)
        hour_means = np.divide(hour_sums, hour_counts, out=np.full(24, np.mean(values)), where=hour_counts > 0)
        fallback = hour_means[np.arange(168) % 24]
        return np.divide(sums, counts, out=fallback.copy(), where=counts > 0)

    def _training(self, frame, end):
        if not isinstance(frame, pd.DataFrame): raise TypeError("frame must be a pandas DataFrame.")
        missing = [name for name in ("I", "Temp") if name not in frame.columns]
        if missing: raise ValueError("frame is missing required columns: " + ", ".join(missing) + ".")
        self._index(frame.index)
        end = pd.Timestamp(end)
        if pd.isna(end): raise ValueError("end must be a valid timestamp.")
        source = frame.loc[frame.index < end, ["I", "Temp"]]
        train, self.imputation = impute_hourly_days(source, columns=("I", "Temp"))
        if len(train) < self.min_samples: raise ValueError(f"At least {self.min_samples} historical I/temperature observations are required to fit O.")
        if not np.all(np.isfinite(train[["I", "Temp"]].to_numpy(dtype=float))): raise ValueError("Training data must be finite.")
        return train

    def _finish(self, train, features):
        temp = train["Temp"].to_numpy(dtype=float)
        y = train["I"].to_numpy(dtype=float)
        residual = y - self.predict(train.index, temp)
        self.metadata = {
            "fit_start": train.index.min(), "fit_end": train.index.max(), "samples": int(len(train)),
            "features": int(features), "iterations": 1, "ridge": 0.0, "huber_delta": None,
            "base_temperature": self.base_temperature,
            "temperature_min": float(np.min(temp)), "temperature_max": float(np.max(temp)),
            "mse": float(np.mean(residual * residual)), "mae": float(np.mean(np.abs(residual))),
            "imputation": dict(self.imputation),
        }
        return self

    def fit(self, frame, end):
        train = self._training(frame, end)
        groups = self._groups(train.index)
        self.levels = self._profile(groups, train["I"].to_numpy(dtype=float))
        self.beta_C, self.chill_center = 0.0, 0.0
        self.coef = self.levels.copy()
        return self._finish(train, 168)

    def predict(self, index, temperature):
        if self.coef is None: raise RuntimeError(f"{type(self).__name__} must be fitted before prediction.")
        index = self._index(index)
        temp = self._temperature(temperature, len(index))
        chill = np.maximum(0.0, self.base_temperature - temp)
        return np.clip(self.levels[self._groups(index)] + self.beta_C * (chill - self.chill_center), 0.0, 1.0)


class weekly_temperature(weekly_average):
    """Weekday/hour fixed effects and one global linear heating-degree response."""

    def __init__(self, base_temperature=18.0, min_samples=168):
        super().__init__(min_samples=min_samples)
        self.base_temperature = float(base_temperature)
        if not np.isfinite(self.base_temperature): raise ValueError("base_temperature must be finite.")

    def fit(self, frame, end):
        train = self._training(frame, end)
        groups = self._groups(train.index)
        y = train["I"].to_numpy(dtype=float)
        chill = np.maximum(0.0, self.base_temperature - train["Temp"].to_numpy(dtype=float))
        mean_y = self._profile(groups, y)
        mean_C = self._profile(groups, chill)
        centered_y, centered_C = y - mean_y[groups], chill - mean_C[groups]
        denominator = float(np.dot(centered_C, centered_C))
        self.beta_C = float(np.dot(centered_C, centered_y) / denominator) if denominator > 1e-12 else 0.0
        self.chill_center = float(np.mean(chill))
        self.levels = mean_y - self.beta_C * (mean_C - self.chill_center)
        self.coef = np.r_[self.levels, self.beta_C]
        self._finish(train, 169)
        self.metadata["beta_C"] = self.beta_C
        self.metadata["chill_center"] = self.chill_center
        return self


def output_callable(index, values):
    index = pd.DatetimeIndex(index)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(index) != values.size or len(index) < 2: raise ValueError("output_callable requires matching time/value vectors with at least two points.")
    elapsed = np.asarray((index - index[0]).total_seconds(), dtype=float) / 3600.0

    def O(time_hours):
        return float(np.clip(np.interp(float(time_hours), elapsed, values), 0.0, 1.0))

    return O
