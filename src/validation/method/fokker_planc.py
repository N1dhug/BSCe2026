import numpy as np

from src.validation.solver.finite_volume import finite_volume_2d_validation

class bm_1_fokker_planc_validation:
    """Input and numerical-state validation for the Fokker-Planck method."""

    @staticmethod
    def _integer(value, name, minimum):
        # Verify inputs
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer.")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be an integer.") from exc
        if float(integer) != float(value):
            raise ValueError(f"{name} must be an integer.")
        if integer < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
        return integer

    @staticmethod
    def positive_scalar(value, name):
        # Verify inputs
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError(f"{name} must be a scalar.")
        scalar = float(array.reshape(-1)[0])
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive.")
        return scalar

    @classmethod
    def degree(cls, degree):
        # Verify inputs
        return cls._integer(degree, "degree", 0)

    @staticmethod
    def cfl(value):
        # Verify inputs
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError("cfl must be a scalar.")
        cfl = float(array.reshape(-1)[0])
        if not np.isfinite(cfl) or not 0.0 < cfl < 1.0:
            raise ValueError("cfl must lie strictly between zero and one.")
        return cfl

    @classmethod
    def beta_basis_inputs(cls, x, degree):
        # Verify inputs
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size == 0:
            raise ValueError("x must be a non-empty one-dimensional grid.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain finite values.")
        if np.any(x <= 0.0) or np.any(x >= 1.0):
            raise ValueError("Beta-basis evaluation points must lie strictly inside (0, 1).")
        return x, cls.degree(degree)

    @classmethod
    def initial_coefficients_inputs(cls, init_cond, degree):
        # Verify inputs
        array = np.asarray(init_cond, dtype=float).reshape(-1)
        if array.size != 1 or not np.all(np.isfinite(array)):
            raise ValueError("init_cond must contain exactly one finite value.")
        x0 = float(array[0])
        if not 0.0 <= x0 <= 1.0:
            raise ValueError("Initial state must lie in [0, 1].")
        return x0, cls.degree(degree)

    @classmethod
    def solve_inputs(cls, t, init_cond, drift, diffusion, degree, x_steps, cfl):
        # Verify inputs
        t = np.asarray(t, dtype=float)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("t must be a one-dimensional grid with at least two points.")
        if not np.all(np.isfinite(t)):
            raise ValueError("t must contain finite values.")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("t must be strictly increasing.")

        x0, degree = cls.initial_coefficients_inputs(init_cond, degree)

        if not callable(drift):
            raise TypeError("drift must be callable.")
        if not callable(diffusion):
            raise TypeError("diffusion must be callable.")

        x_steps = cls._integer(x_steps, "x_steps", 3)
        cfl = cls.cfl(cfl)

        return t, x0, drift, diffusion, degree, x_steps, cfl

    @classmethod
    def basis_normalisation_inputs(cls, basis, dx):
        # Verify inputs
        basis = np.asarray(basis, dtype=float)
        if basis.ndim != 2 or basis.size == 0:
            raise ValueError("basis must be a non-empty two-dimensional array.")
        if not np.all(np.isfinite(basis)):
            raise ValueError("basis must contain finite values.")
        dx = cls.positive_scalar(dx, "dx")
        masses = np.sum(basis, axis=1) * dx
        if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("Every basis function must have positive finite discrete mass.")
        return basis, dx, masses

    @staticmethod
    def mixture_inputs(coefficients, basis):
        # Verify inputs
        coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
        basis = np.asarray(basis, dtype=float)
        if basis.ndim != 2:
            raise ValueError("basis must be two-dimensional.")
        if coefficients.size != basis.shape[0]:
            raise ValueError("Coefficient count must match the number of basis functions.")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(basis)):
            raise ValueError("coefficients and basis must contain finite values.")
        if np.any(coefficients < -1e-12):
            raise ValueError("Mixture coefficients must be non-negative.")
        if not np.isclose(np.sum(coefficients), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("Mixture coefficients must sum to one.")
        return coefficients, basis

    @classmethod
    def projection_inputs(cls, density, basis, dx):
        # Verify inputs
        density = np.asarray(density, dtype=float).reshape(-1)
        basis = np.asarray(basis, dtype=float)
        if basis.ndim != 2:
            raise ValueError("basis must be two-dimensional.")
        if density.size != basis.shape[1]:
            raise ValueError("density length must match the basis grid dimension.")
        if not np.all(np.isfinite(density)) or not np.all(np.isfinite(basis)):
            raise ValueError("density and basis must contain finite values.")
        return density, basis, cls.positive_scalar(dx, "dx")

    @classmethod
    def density_inputs(cls, density, dx):
        # Verify inputs
        density = np.asarray(density, dtype=float)
        if density.ndim != 1 or density.size == 0:
            raise ValueError("density must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(density)):
            raise RuntimeError("Fokker-Planck density contains non-finite values.")
        return density, cls.positive_scalar(dx, "dx")

    @staticmethod
    def positive_mass(mass):
        # Verify probability mass
        if not np.isfinite(mass) or mass <= 0.0:
            raise RuntimeError("Fokker-Planck density lost positive finite mass.")
        return float(mass)

    @staticmethod
    def field_values(values, shape, name):
        # Verify inputs
        try:
            values = np.broadcast_to(np.asarray(values, dtype=float), shape)
        except ValueError as exc:
            raise ValueError(f"{name} output is not broadcastable to the spatial grid.") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} returned non-finite values.")
        return values

    @classmethod
    def flux_inputs(cls, x, density, drift_values, diffusion_values):
        # Verify inputs
        x = np.asarray(x, dtype=float)
        density = np.asarray(density, dtype=float)
        if x.ndim != 1 or density.ndim != 1 or x.size != density.size or x.size < 2:
            raise ValueError("x and density must be one-dimensional arrays of equal length >= 2.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(density)):
            raise ValueError("x and density must contain finite values.")
        spacing = np.diff(x)
        if np.any(spacing <= 0.0) or not np.allclose(spacing, spacing[0]):
            raise ValueError("x must be a strictly increasing uniform grid.")
        drift_values = cls.field_values(drift_values, density.shape, "drift")
        diffusion_values = cls.field_values(diffusion_values, density.shape, "diffusion")
        return x, density, drift_values, diffusion_values

    @staticmethod
    def positive_substep(h):
        # Verify inputs
        if not np.isfinite(h) or h <= 0.0:
            raise RuntimeError("Fokker-Planck method produced a non-positive substep.")
        return float(h)

class wm_fokker_planc_validation:
    @classmethod
    def solve_inputs(cls, t, initial_density, drift_v, drift_r, diffusion_r, v_bounds, r_bounds, v_steps, r_steps, cfl):
        # Verify inputs
        t = finite_volume_2d_validation.time_grid(t)
        for function, name in ((drift_v, "drift_v"), (drift_r, "drift_r"), (diffusion_r, "diffusion_r")):
            if not callable(function): raise TypeError(f"{name} must be callable.")
        if initial_density is None: raise ValueError("wm Fokker-Planck requires initial_density.")
        if not callable(initial_density):
            initial_density = np.asarray(initial_density, dtype=float)
            if initial_density.ndim != 2 or initial_density.size == 0: raise ValueError("initial_density must be callable or a non-empty 2D array.")
            if not np.all(np.isfinite(initial_density)) or np.any(initial_density < 0.0): raise ValueError("initial_density must contain finite non-negative values.")

        v_bounds = finite_volume_2d_validation.bounds(v_bounds, "v_bounds")
        r_bounds = finite_volume_2d_validation.bounds(r_bounds, "r_bounds")
        v_steps = finite_volume_2d_validation.integer(v_steps, "v_steps")
        r_steps = finite_volume_2d_validation.integer(r_steps, "r_steps")
        cfl = finite_volume_2d_validation.cfl(cfl)
        return t, initial_density, drift_v, drift_r, diffusion_r, v_bounds, r_bounds, v_steps, r_steps, cfl
