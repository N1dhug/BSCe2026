import numpy as np

from src.validation.model.wm_4 import wm_4_validation


class wm_5_validation(wm_4_validation):
    """Certified monotone bases and WM4-equivalent signed-domain admissibility."""

    @classmethod
    def basis_functions(cls, functions, name):
        if not isinstance(functions, (tuple, list)) or not functions:
            raise ValueError(f"{name} must be a nonempty list or tuple of response_function objects.")
        names = []
        for basis in functions:
            if not all(hasattr(basis, key) for key in ("name", "value", "derivative", "derivative_bounds")):
                raise TypeError(f"{name} entries require name, value, derivative and certified derivative_bounds.")
            if not callable(basis.value) or not callable(basis.derivative):
                raise TypeError(f"{name}: the value and derivative must be callable.")
            lo, hi = map(float, basis.derivative_bounds)
            if not np.isfinite([lo, hi]).all() or lo < 0.0 or hi < lo:
                raise ValueError(f"{name}/{basis.name}: require certified 0 <= min derivative <= max derivative.")
            nodes = np.linspace(-1.0, 1.0, 65)
            values = np.array([basis.value(float(x)) for x in nodes], dtype=float)
            derivatives = np.array([basis.derivative(float(x)) for x in nodes], dtype=float)
            if not np.isfinite(values).all() or not np.isfinite(derivatives).all():
                raise ValueError(f"{name}/{basis.name}: nonfinite function or derivative.")
            if abs(float(basis.value(0.0))) > 1e-10:
                raise ValueError(f"{name}/{basis.name}: all bases must be centered at zero.")
            if np.any(derivatives < lo - 1e-8) or np.any(derivatives > hi + 1e-8):
                raise ValueError(f"{name}/{basis.name}: derivative falls outside its supplied certificate.")
            names.append(str(basis.name))
        if len(set(names)) != len(names): raise ValueError(f"{name}: response function names must be unique.")
        return tuple(functions)

    @classmethod
    def coefficients(cls, parameters, r_functions, u_functions, dt, stabilize=True, tol=2e-7):
        from src.validation.params.wm_5 import wm_5_params_validation
        p = cls.vector(parameters, 1 + len(r_functions) + len(u_functions), "coefficients")
        A, b, lb, ub = wm_5_params_validation.constraints(r_functions, u_functions, dt, stabilize=stabilize)
        if np.any(p < lb - tol) or np.any(p > ub + tol) or np.any(A @ p - b > tol):
            raise ValueError("WM5 coefficients violate restoring, monotone, equilibrium or stability restrictions.")
        return p

    @classmethod
    def initial_condition(cls, initial_condition, allow_rate=True):
        values = super().initial_condition(initial_condition, allow_rate=allow_rate)
        if values[1] < -1.0 or values[1] > 1.0:
            raise ValueError("WM5 initial rate must lie in [-1,1].")
        return values
