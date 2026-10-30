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
