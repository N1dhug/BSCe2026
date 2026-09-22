import numpy as np


class Simple:
    """Generate hourly synthetic user data from daily sine/cosine harmonics and Gaussian noise."""

    def __init__(self, days=1, mean=0.5, sin_coeff=(0.2,), cos_coeff=(0.1,), std=0.0, seed=None):
        self.days = int(days)
        self.mean = float(mean)
        self.sin_coeff = np.asarray(sin_coeff, dtype=float).reshape(-1)
        self.cos_coeff = np.asarray(cos_coeff, dtype=float).reshape(-1)
        self.std = float(std)
        self.rng = np.random.default_rng(seed)

        if self.days < 1: raise ValueError("days must be at least 1.")
        if self.std < 0.0: raise ValueError("std must be non-negative.")
        if self.sin_coeff.size != self.cos_coeff.size: raise ValueError("sin_coeff and cos_coeff must have the same length.")
        if not np.isfinite(self.mean) or not np.all(np.isfinite(self.sin_coeff)) or not np.all(np.isfinite(self.cos_coeff)) or not np.isfinite(self.std): raise ValueError("Simulation parameters must be finite.")

    @property
    def hours(self):
        return 24 * self.days

    def sample(self):
        """Return hourly time indices and simulated user data."""
        t = np.arange(self.hours, dtype=float)
        y = np.full(self.hours, self.mean, dtype=float)

        for k, (a, b) in enumerate(zip(self.sin_coeff, self.cos_coeff), start=1):
            phase = 2.0 * np.pi * k * t / 24.0
            y += a * np.sin(phase) + b * np.cos(phase)

        if self.std > 0.0: y += self.rng.normal(0.0, self.std, self.hours)
        return t, y

    def data(self):
        """Return only the hourly simulated user-data vector."""
        return self.sample()[1]
