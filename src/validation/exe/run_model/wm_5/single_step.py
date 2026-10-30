import numpy as np


class single_step_validation:
    @staticmethod
    def settings(duration, dt, step_at, before, after):
        values = np.asarray([duration, dt, step_at, before, after], dtype=float)
        if not np.isfinite(values).all(): raise ValueError("Step settings must be finite.")
        if duration <= 0.0 or dt <= 0.0 or not 0.0 < step_at < duration:
            raise ValueError("Require duration>0, dt>0, and 0<step_at<duration.")
        if duration/dt > 100000: raise ValueError("Step grid is too large.")
        if not -1.0 <= before <= 1.0 or not -1.0 <= after <= 1.0:
            raise ValueError("Step prices must lie in [-1,1].")
        return map(float, values)

    @staticmethod
    def price_response(theta_U, h):
        theta_U = float(theta_U)
        if not np.isfinite(theta_U) or theta_U > 0.0:
            raise ValueError("The price coefficient theta_U must be nonpositive.")
        if not callable(h): raise TypeError("h must be a callable normalized price response.")
        x = np.linspace(-1.0, 1.0, 201)
        values = np.array([h(float(u)) for u in x], dtype=float)
        if not np.isfinite(values).all() or np.any(np.diff(values) < -1e-12):
            raise ValueError("h(U) must be nondecreasing over the normalized price domain.")
        if abs(values[0]+1.0) > 1e-10 or abs(values[-1]-1.0) > 1e-10:
            raise ValueError("Require normalized price response h(-1)=-1 and h(1)=1.")
