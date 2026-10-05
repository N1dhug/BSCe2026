"""Small configuration for the separate G-plan; no F4 parameter API changes."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BM1DeterministicConfig:
    # C is in physical hours. These are development bounds, not capacity estimates.
    lower: tuple = (1.0, 0.05, 0.0, 0.0)
    upper: tuple = (744.0, 10.0, 1.0, 1.0)
    starts: tuple = ((12., 2., .5, .5), (48., 1., .2, .25),
                     (168., 4., .75, .75), (6., .5, .1, .5))
    steps_per_hour: int = 60
    calls_per_start: int = 250
    max_calls: int = 1000
    seconds: float = 300.
    numerical_tolerance: float = 1e-5

    def __post_init__(self):
        if len(self.lower) != 4 or len(self.upper) != 4:
            raise ValueError('Bounds must describe C_hours, k, Xi, x0.')
        if any(not math.isfinite(v) for v in (*self.lower, *self.upper)):
            raise ValueError('Bounds must be finite.')
        if any(a >= b for a, b in zip(self.lower, self.upper)):
            raise ValueError('Each lower bound must be below its upper bound.')
        if self.lower[0] <= 0 or self.lower[1] <= 0:
            raise ValueError('C and k must be positive.')
        if self.lower[2] < 0 or self.lower[3] < 0 or self.upper[2] > 1 or self.upper[3] > 1:
            raise ValueError('Xi/x0 bounds must lie in [0,1].')
        if not 1 <= len(self.starts) <= 4:
            raise ValueError('Declare one to four starts.')
        for start in self.starts:
            if len(start) != 4 or any(not a <= v <= b for v, a, b in zip(start, self.lower, self.upper)):
                raise ValueError('Starts must be finite and inside bounds.')
        for value in (self.steps_per_hour, self.calls_per_start, self.max_calls):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError('Step/call counts must be positive integers.')
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('Time budget must be positive and finite.')
        if not math.isfinite(self.numerical_tolerance) or self.numerical_tolerance <= 0:
            raise ValueError('Numerical tolerance must be positive.')
