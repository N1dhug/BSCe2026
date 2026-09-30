"""Conservative FP prediction followed by simplex-constrained Beta projection.

This is a numerical method, not a price optimizer. Drift and diffusion are
supplied as callbacks. Invalid probabilities are rejected, never repaired.
"""

from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize

_NEGATIVE_PREDICTION_TOL = 1e-10
_MASS_TOL = 1e-09
_ACTIVE_SET_PROJECTION_MAX_ITER = 500
_PROJECTION_GAP_TOL = 1e-11
_FALLBACK_GAP_TOL = 1e-9
_FALLBACK_PROJECTION_MAX_ITER = 500


def _validate_dynamic_controls(t_eval, breakpoints, max_step, cfl):
    t_eval = np.asarray(t_eval, dtype=float)
    if (
        t_eval.ndim != 1
        or t_eval.size < 2
        or not np.all(np.isfinite(t_eval))
        or np.any(np.diff(t_eval) <= 0.0)
    ):
        raise ValueError('t_eval must be a finite strictly increasing one-dimensional grid.')
    breakpoints = np.asarray([] if breakpoints is None else breakpoints, dtype=float).reshape(-1)
    if (
        not np.all(np.isfinite(breakpoints))
        or np.any(np.diff(breakpoints) <= 0.0)
        or np.any(breakpoints <= t_eval[0])
        or np.any(breakpoints >= t_eval[-1])
    ):
        raise ValueError('breakpoints must be strictly increasing and inside the time interval.')
    max_step = float(max_step)
    if np.isnan(max_step) or max_step <= 0.0:
        raise ValueError('max_step must be strictly positive.')
    cfl = float(cfl)
    if not np.isfinite(cfl) or not 0.0 < cfl < 1.0:
        raise ValueError('cfl must lie strictly between zero and one.')
    return (t_eval, breakpoints, max_step, cfl)


def _projection_matrices(basis, dx):
    """Return weighted least-squares matrix, Hessian and its condition number."""
    basis = np.asarray(basis, dtype=float)
    dx = float(dx)
    scale = np.sqrt(dx)
    matrix = basis.T * scale
    hessian = matrix.T @ matrix
    condition = float(np.linalg.cond(hessian))
    return (matrix, hessian, condition)


class _SimplexProjector:
    """Prepared active-set least squares on the probability simplex.

    Only the free coefficients are solved for.  On each active face, an
    orthonormal zero-sum basis Z eliminates the equality constraint, and an
    SVD of A_free @ Z solves the least-squares problem directly.  A.T @ A is
    not used for the solve: forming normal equations would square conditioning.

    A bounded LRU cache stores *geometry*, not target-dependent solutions.
    Every solve starts from the supplied weights, independently of cache state.
    No ridge term, probability clipping or coefficient normalization is used.
    """

    def __init__(self, matrix, *, cache_size=512):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.all(np.isfinite(matrix)):
            raise ValueError('projection matrix must be a finite non-empty 2D array.')
        if isinstance(cache_size, (bool, np.bool_)) or int(cache_size) != cache_size or cache_size < 0:
            raise ValueError('cache_size must be a non-negative integer.')
        self.matrix = matrix.copy()
        self.matrix.setflags(write=False)
        self.cache_size = int(cache_size)
        self._faces = OrderedDict()
        singular_values = np.linalg.svd(self.matrix, compute_uv=False)
        self.matrix_condition_number = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > 0.0 else float('inf')
        )
        self.factorizations = 0
        self.cache_hits = 0

    def _face_factorization(self, indices):
        key = tuple(indices)
        if key in self._faces:
            self._faces.move_to_end(key)
            self.cache_hits += 1
            return self._faces[key]
        count = len(indices)
        tangent = np.zeros((count, count - 1), dtype=float)
        for j in range(count - 1):
            scale = 1.0 / np.sqrt((j + 1) * (j + 2))
            tangent[:j + 1, j] = scale
            tangent[j + 1, j] = -(j + 1) * scale
        reduced = self.matrix[:, indices] @ tangent
        left, singular_values, right = np.linalg.svd(reduced, full_matrices=False)
        cutoff = max(reduced.shape) * np.finfo(float).eps * singular_values[0]
        resolved = singular_values > cutoff
        factors = (tangent, left[:, resolved], singular_values[resolved], right[resolved])
        self.factorizations += 1
        if self.cache_size:
            self._faces[key] = factors
            if len(self._faces) > self.cache_size:
                self._faces.popitem(last=False)
        return factors

    def optimality_gap(self, target, coefficients):
        """Return c.T g - min(g); for feasible c this bounds objective suboptimality.

        Here g = A.T @ (A c - b). Convexity gives
        f(c)-f(c*) <= g.T(c-c*) <= g.T c-min(g) on the simplex.
        This check is independent of the SVD/SLSQP stopping flag.
        """
        residual = self.matrix @ coefficients - target
        gradient = self.matrix.T @ residual
        return float(gradient @ coefficients - np.min(gradient)), gradient

    def solve(self, target, x0):
        """Solve the unchanged simplex projection, with a feasible warm start."""
        target = np.asarray(target, dtype=float)
        coefficients = np.asarray(x0, dtype=float).copy()
        if (target.shape != (self.matrix.shape[0],)
                or coefficients.shape != (self.matrix.shape[1],)
                or not np.all(np.isfinite(target))
                or not np.all(np.isfinite(coefficients))):
            raise ValueError('projection target/start must be finite vectors matching the matrix.')
        if np.min(coefficients) < -1e-10 or abs(float(np.sum(coefficients)) - 1.0) > _MASS_TOL:
            raise ValueError('projection warm start must satisfy the probability simplex.')
        free = coefficients > 1e-10
        if not np.any(free):
            free[int(np.argmax(coefficients))] = True
        for _ in range(_ACTIVE_SET_PROJECTION_MAX_ITER):
            residual = self.matrix @ coefficients - target
            gradient = self.matrix.T @ residual
            scale = max(1.0, float(np.max(np.abs(gradient))))
            gap = float(gradient @ coefficients - np.min(gradient))
            if gap <= _PROJECTION_GAP_TOL * scale:
                return coefficients
            indices = np.flatnonzero(free)
            if indices.size > 1:
                tangent, left, singular_values, right = self._face_factorization(indices)
                direction = -tangent @ (right.T @ ((left.T @ residual) / singular_values))
            else:
                direction = np.zeros(1, dtype=float)
            if float(np.max(np.abs(direction))) > 1e-12:
                decreasing = np.flatnonzero(direction < -1e-12)
                alpha = 1.0
                blocking = np.empty(0, dtype=int)
                if decreasing.size:
                    ratios = -coefficients[indices[decreasing]] / direction[decreasing]
                    first = int(np.argmin(ratios))
                    minimum_ratio = float(ratios[first])
                    if minimum_ratio < alpha:
                        alpha = max(0.0, minimum_ratio)
                        # Several bounds can become active in the same line-search
                        # step. Enforce every numerically simultaneous blocker
                        # exactly; otherwise roundoff can leave a tied coefficient
                        # at a tiny negative value even though the constrained step
                        # itself is feasible. This is active-set geometry, not
                        # post-hoc probability clipping.
                        ratio_tolerance = (
                            64.0 * np.finfo(float).eps * max(1.0, abs(minimum_ratio))
                        )
                        tied = decreasing[ratios <= minimum_ratio + ratio_tolerance]
                        blocking = indices[tied]
                coefficients[indices] += alpha * direction
                if blocking.size:
                    coefficients[blocking] = 0.0
                    free[blocking] = False
                    continue
                gradient = self.matrix.T @ (self.matrix @ coefficients - target)
            inactive = np.flatnonzero(~free)
            if inactive.size:
                worst = int(inactive[np.argmin(gradient[inactive])])
                if gradient[worst] < np.mean(gradient[indices]) - 1e-12 * scale:
                    free[worst] = True
                    continue
            gap = float(gradient @ coefficients - np.min(gradient))
            if gap <= _PROJECTION_GAP_TOL * scale:
                return coefficients
        raise RuntimeError('Simplex active-set least-squares solver did not converge.')


def _slsqp_simplex_projection(matrix, target, x0):
    """Solve the same simplex least-squares projection if active-set stalls.

    This is a numerical fallback only. It does not clip/renormalize the target
    or change the projection problem.
    """
    x0 = np.asarray(x0, dtype=float)
    n_coefficients = x0.size

    def objective(coefficients):
        residual = matrix @ coefficients - target
        return 0.5 * float(residual @ residual)

    def gradient(coefficients):
        return matrix.T @ (matrix @ coefficients - target)

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='SLSQP',
        bounds=[(0.0, None)] * n_coefficients,
        constraints=[
            {
                'type': 'eq',
                'fun': lambda coefficients: float(np.sum(coefficients) - 1.0),
                'jac': lambda coefficients: np.ones_like(coefficients),
            },
        ],
        options={'ftol': 1e-16, 'maxiter': _FALLBACK_PROJECTION_MAX_ITER, 'disp': False},
    )
    coefficients = np.asarray(result.x, dtype=float)
    # SLSQP's status flag is advisory here.  Some ill-conditioned but convex
    # simplex projections terminate with e.g. "positive directional derivative"
    # even though the returned point is feasible and satisfies our independent
    # first-order optimality check below.  Reject infeasible/non-finite output
    # here; _project_with_warm_start remains the authoritative optimality gate.
    if (
        not np.all(np.isfinite(coefficients))
        or np.min(coefficients) < -_MASS_TOL
        or abs(float(np.sum(coefficients)) - 1.0) > _MASS_TOL
    ):
        raise RuntimeError(
            'SLSQP fallback returned an infeasible simplex point: '
            + str(result.message),
        )
    return coefficients


def _project_with_warm_start(
    density, basis, matrix, hessian, dx, x0, *, return_fallback=False, projector=None,
):
    """Project with prepared square-root active set; retry the same problem on failure.

    The Hessian argument is retained for callers/conditioning diagnostics; the
    actual least-squares solve uses the unsquared matrix. Probability and
    first-order optimality are checked after either numerical backend.
    """
    del hessian
    density = np.asarray(density, dtype=float)
    target = density * np.sqrt(dx)
    if projector is None:
        projector = _SimplexProjector(matrix)
    used_fallback = False
    try:
        coefficients = projector.solve(target, np.asarray(x0, dtype=float))
        # A primary active-set result with even a tiny signed coefficient is
        # not a valid probability vector.  Retry the *same* constrained problem
        # with the independent backend instead of clipping/renormalizing it.
        if np.any(coefficients < 0.0):
            raise RuntimeError('Primary active-set projection returned a signed coefficient.')
    except (RuntimeError, np.linalg.LinAlgError):
        coefficients = _slsqp_simplex_projection(matrix, target, x0)
        used_fallback = True
    if (not np.all(np.isfinite(coefficients))
            or np.any(coefficients < 0.0)
            or abs(float(np.sum(coefficients)) - 1.0) > _MASS_TOL):
        raise RuntimeError('Beta projection failed probability constraints.')
    gap, gradient = projector.optimality_gap(target, coefficients)
    tolerance = _FALLBACK_GAP_TOL if used_fallback else _PROJECTION_GAP_TOL
    if not np.isfinite(gap) or gap > tolerance * max(1.0, float(np.max(np.abs(gradient)))):
        raise RuntimeError('Beta projection failed the independent simplex optimality check.')
    projected = coefficients @ basis
    if return_fallback:
        return (coefficients, projected, used_fallback)
    return (coefficients, projected)


def _positive_explicit_step_limit(dx, drift_values, diffusion_values, cfl):
    """Return a positivity-oriented CFL limit for the conservative grid update.

    The outgoing rate from an interior cell is the sum of its upwind advective
    rates and the two half-diffusion rates.  Requiring h * outgoing_rate <= cfl
    is stricter than taking the minimum of separate advection/diffusion limits
    and avoids relying on post-step clipping.
    """
    drift_values = np.asarray(drift_values, dtype=float)
    diffusion_values = np.asarray(diffusion_values, dtype=float)
    face_drift = 0.5 * (drift_values[:-1] + drift_values[1:])
    diffusion_term = 0.5 * np.square(diffusion_values)
    outgoing = np.zeros_like(drift_values)
    outgoing[:-1] += np.maximum(face_drift, 0.0) / dx
    outgoing[1:] += np.maximum(-face_drift, 0.0) / dx
    outgoing[:-1] += diffusion_term[:-1] / (dx * dx)
    outgoing[1:] += diffusion_term[1:] / (dx * dx)
    maximum = float(np.max(outgoing))
    return np.inf if maximum == 0.0 else float(cfl / maximum)


def evolve_projected_density(
    *,
    method,
    t_eval,
    breakpoints,
    x,
    dx,
    basis,
    initial_coefficients,
    drift,
    diffusion,
    max_step,
    cfl,
    projection_data=None,
    projection_solver=None,
):
    """Propagate one fixed Beta basis through explicit Euler + QP projection."""
    t_eval, breakpoints, max_step, cfl = _validate_dynamic_controls(t_eval, breakpoints, max_step, cfl)
    x = np.asarray(x, dtype=float)
    basis = np.asarray(basis, dtype=float)
    coefficients = np.asarray(initial_coefficients, dtype=float).copy()
    if x.ndim != 1 or x.size < 2 or (not np.all(np.isfinite(x))):
        raise ValueError('x must be a finite one-dimensional grid.')
    dx = float(dx)
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError('dx must be finite and positive.')
    if not np.allclose(np.diff(x), dx, rtol=1e-10, atol=1e-14):
        raise ValueError('x must be uniformly spaced with spacing dx.')
    if (
        basis.ndim != 2
        or basis.shape[1] != x.size
        or basis.shape[0] != coefficients.size
        or not np.all(np.isfinite(basis))
    ):
        raise ValueError('basis and coefficients must match the state grid.')
    if (
        coefficients.ndim != 1
        or not np.all(np.isfinite(coefficients))
        or np.min(coefficients) < -_MASS_TOL
        or abs(float(np.sum(coefficients)) - 1.0) > _MASS_TOL
    ):
        raise ValueError('initial_coefficients must satisfy the probability simplex.')
    if np.min(basis) < 0.0 or not np.allclose(np.sum(basis, axis=1) * dx, 1.0, rtol=0.0, atol=_MASS_TOL):
        raise ValueError('each basis component must have non-negative unit mass.')
    density = coefficients @ basis
    if projection_data is None:
        projection_data = _projection_matrices(basis, dx)
    matrix, hessian, condition = projection_data
    if matrix.shape != basis.T.shape or hessian.shape != (basis.shape[0],) * 2:
        raise ValueError('projection_data does not match the supplied basis.')
    if projection_solver is None:
        projection_solver = _SimplexProjector(matrix)
    elif not np.array_equal(projection_solver.matrix, matrix):
        raise ValueError('projection_solver was prepared for a different basis.')
    if not np.array_equal(matrix, basis.T * np.sqrt(dx)):
        raise ValueError('projection_data was prepared for a different basis or spacing.')
    initial_factorizations = projection_solver.factorizations
    initial_cache_hits = projection_solver.cache_hits
    output_density = np.full((len(t_eval), len(x)), np.nan, dtype=float)
    output_coefficients = np.full((len(t_eval), basis.shape[0]), np.nan, dtype=float)
    output_density[0] = density
    output_coefficients[0] = coefficients
    knots = np.unique(np.concatenate((t_eval, breakpoints)))
    output_lookup = {float(value): index for index, value in enumerate(t_eval)}
    n_steps = 0
    minimum_raw_prediction = float(np.min(density))
    maximum_raw_prediction_mass_error = abs(float(np.sum(density) * dx) - 1.0)
    maximum_projection_mass_error = maximum_raw_prediction_mass_error
    maximum_coefficient_sum_error = abs(float(np.sum(coefficients)) - 1.0)
    minimum_coefficient = float(np.min(coefficients))
    maximum_boundary_coefficient = float(max(coefficients[0], coefficients[-1]))
    maximum_internal_step = 0.0
    minimum_internal_step = np.inf
    projection_fallback_count = 0
    for left, right in zip(knots[:-1], knots[1:]):
        current = float(left)
        target = float(right)
        while current < target - 1e-15:
            drift_values = np.asarray(drift(current, x), dtype=float)
            diffusion_values = np.asarray(diffusion(current, x), dtype=float)
            if drift_values.shape != x.shape or diffusion_values.shape != x.shape:
                raise ValueError('drift and diffusion must return arrays matching x.')
            if not np.all(np.isfinite(drift_values)) or not np.all(np.isfinite(diffusion_values)):
                raise ValueError('drift and diffusion must return finite values.')
            stable_h = _positive_explicit_step_limit(dx, drift_values, diffusion_values, cfl)
            remaining = target - current
            h = min(remaining, max_step, stable_h)
            if not np.isfinite(h) or h <= 0.0:
                raise RuntimeError('Dynamic mixture produced a non-positive internal step.')
            rhs = method._flux_rhs(x, density, drift_values, diffusion_values)
            prediction = density + h * rhs
            raw_minimum = float(np.min(prediction))
            raw_mass_error = abs(float(np.sum(prediction) * dx) - 1.0)
            minimum_raw_prediction = min(minimum_raw_prediction, raw_minimum)
            maximum_raw_prediction_mass_error = max(maximum_raw_prediction_mass_error, raw_mass_error)
            if raw_minimum < -_NEGATIVE_PREDICTION_TOL:
                raise RuntimeError(
                    'Raw explicit-Euler mixture prediction became materially negative; reduce cfl/max_step rather than clipping it.',
                )
            if raw_mass_error > _MASS_TOL:
                raise RuntimeError('Raw mixture prediction failed probability conservation.')
            coefficients, density, used_projection_fallback = _project_with_warm_start(
                prediction,
                basis,
                matrix,
                hessian,
                dx,
                coefficients,
                return_fallback=True,
                projector=projection_solver,
            )
            projection_fallback_count += int(used_projection_fallback)
            projected_mass_error = abs(float(np.sum(density) * dx) - 1.0)
            coefficient_sum_error = abs(float(np.sum(coefficients)) - 1.0)
            maximum_projection_mass_error = max(maximum_projection_mass_error, projected_mass_error)
            maximum_coefficient_sum_error = max(maximum_coefficient_sum_error, coefficient_sum_error)
            minimum_coefficient = min(minimum_coefficient, float(np.min(coefficients)))
            maximum_boundary_coefficient = max(
                maximum_boundary_coefficient,
                float(coefficients[0]),
                float(coefficients[-1]),
            )
            if (
                projected_mass_error > _MASS_TOL
                or coefficient_sum_error > _MASS_TOL
                or minimum_coefficient < -_MASS_TOL
            ):
                raise RuntimeError('Projected mixture failed probability constraints.')
            n_steps += 1
            maximum_internal_step = max(maximum_internal_step, h)
            minimum_internal_step = min(minimum_internal_step, h)
            current += h
            if target - current <= 1e-14 * max(1.0, abs(target)):
                current = target
        if target in output_lookup:
            index = output_lookup[target]
            output_density[index] = density
            output_coefficients[index] = coefficients
    if not np.all(np.isfinite(output_density)) or not np.all(np.isfinite(output_coefficients)):
        raise RuntimeError('Dynamic mixture failed to populate finite output histories.')
    return {
        'density': output_density,
        'coefficients': output_coefficients,
        'diagnostics': {
            'n_steps': n_steps,
            'hessian_condition_number': condition,
            'minimum_raw_prediction': minimum_raw_prediction,
            'maximum_abs_raw_prediction_mass_error': maximum_raw_prediction_mass_error,
            'maximum_abs_projected_mass_error': maximum_projection_mass_error,
            'maximum_abs_coefficient_sum_error': maximum_coefficient_sum_error,
            'minimum_coefficient': minimum_coefficient,
            'maximum_boundary_coefficient': maximum_boundary_coefficient,
            'maximum_internal_step': maximum_internal_step,
            'minimum_internal_step': minimum_internal_step,
            'projection_fallback_count': projection_fallback_count,
            'projection_matrix_condition_number': projection_solver.matrix_condition_number,
            'projection_factorizations': projection_solver.factorizations - initial_factorizations,
            'projection_factorization_cache_hits': projection_solver.cache_hits - initial_cache_hits,
            'projection_backend': 'simplex-svd',
            'used_clipping': False,
            'used_renormalization': False,
        },
    }


# F4 preparation: public opt-in geometry/projection and cell-integrated basis.
# Existing F2/F3 callers above retain their basis, defaults and algorithms.
class PreparedBetaProjector:
    """Reusable simplex projection for a fixed cell-average density dictionary.

    The supplied basis has shape (components, uniform cells). Preparation
    caches geometry, never target-dependent coefficients. Each project call
    starts from explicit coefficients, or a deterministic uniform start.
    """

    def __init__(self, basis, dx, *, cache_size=512):
        if np.iscomplexobj(basis):
            raise ValueError('basis must be real-valued.')
        basis = np.asarray(basis, dtype=float)
        dx = float(dx)
        if (basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] < 2
                or not np.all(np.isfinite(basis)) or np.any(basis < 0.0)
                or not np.isfinite(dx) or dx <= 0.0):
            raise ValueError('basis must be a finite non-negative (components, cells) matrix; dx > 0.')
        if not np.allclose(np.sum(basis, axis=1) * dx, 1.0, atol=_MASS_TOL, rtol=0.0):
            raise ValueError('Each basis component must have unit discrete mass; no repair is applied.')
        self.basis = basis.copy()
        self.basis.setflags(write=False)
        self.dx = dx
        self.projection_data = _projection_matrices(self.basis, self.dx)
        for item in self.projection_data[:2]:
            item.setflags(write=False)
        self.projection_solver = _SimplexProjector(self.projection_data[0], cache_size=cache_size)

    def project(self, density, initial_coefficients=None):
        """Return coefficients, density, and numerical/representation diagnostics.

        A small optimality gap verifies this QP, not the quality of the basis.
        Legacy boundary-coefficient diagnostics are not interpreted as boundary
        mass for an arbitrary dictionary; actual end-cell mass is reported here.
        """
        if np.iscomplexobj(density):
            raise ValueError('density must be real-valued.')
        density = np.asarray(density, dtype=float)
        if (density.shape != (self.basis.shape[1],) or not np.all(np.isfinite(density))
                or np.any(density < 0.0)
                or abs(float(np.sum(density) * self.dx) - 1.0) > _MASS_TOL):
            raise ValueError('density must be non-negative, finite and have unit discrete mass.')
        if initial_coefficients is None:
            initial_coefficients = np.full(self.basis.shape[0], 1.0 / self.basis.shape[0])
        if np.iscomplexobj(initial_coefficients):
            raise ValueError('initial_coefficients must be real-valued.')
        start = np.asarray(initial_coefficients, dtype=float)
        if (start.shape != (self.basis.shape[0],) or not np.all(np.isfinite(start))
                or np.any(start < 0.0) or abs(float(np.sum(start)) - 1.0) > _MASS_TOL):
            raise ValueError('initial_coefficients must be a feasible probability vector.')
        matrix, hessian, condition = self.projection_data
        coefficients, projected, used_fallback = _project_with_warm_start(
            density, self.basis, matrix, hessian, self.dx, start,
            return_fallback=True, projector=self.projection_solver,
        )
        # A signed probability cannot be used as log input to Bayes. Do not
        # silently clip a negative roundoff value accepted by the legacy QP.
        if np.any(coefficients < 0.0) or np.any(projected < 0.0):
            raise FloatingPointError(
                'Projection returned signed probabilities; '
                f'min coefficient={float(np.min(coefficients)):.3e}, '
                f'min density={float(np.min(projected)):.3e}. '
                'Inspect the numerical solution; no clipping is applied.'
            )
        gap, _ = self.projection_solver.optimality_gap(density * np.sqrt(self.dx), coefficients)
        error = projected - density
        return {
            'coefficients': coefficients,
            'density': projected,
            'diagnostics': {
                'optimality_gap': gap,
                'density_l2_error': float(np.sqrt(self.dx * (error @ error))),
                'cell_mass_l1_error': float(self.dx * np.sum(np.abs(error))),
                'boundary_cell_mass': float(self.dx * (projected[0] + projected[-1])),
                'mass_error': abs(float(self.dx * np.sum(projected)) - 1.0),
                'used_fallback': used_fallback,
                'hessian_condition_number': condition,
                'projection_matrix_condition_number': self.projection_solver.matrix_condition_number,
                'used_clipping': False,
                'used_renormalization': False,
            },
        }


def beta_cell_average_basis(edges, alpha, beta):
    """Integrate supplied Beta(a,b) components on a uniform grid covering [0,1].

    Returns density averages (components, cells), not center-sampled PDFs.
    Tail differences use the complementary CDF to reduce cancellation. Shapes
    are explicit: no dictionary/default concentration is selected by this API.
    Changing basis construction changes the approximation and must be validated.
    """
    from scipy.special import betainc, betaincc

    if any(np.iscomplexobj(values) for values in (edges, alpha, beta)):
        raise ValueError('grid and beta shape parameters must be real-valued.')
    edges = np.asarray(edges, dtype=float)
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    if (edges.ndim != 1 or edges.size < 3 or not np.all(np.isfinite(edges))
            or edges[0] != 0.0 or edges[-1] != 1.0 or np.any(np.diff(edges) <= 0.0)):
        raise ValueError('edges must be a strictly increasing grid spanning exactly [0,1].')
    widths = np.diff(edges)
    dx = 1.0 / widths.size
    if not np.allclose(widths, dx, rtol=1e-10, atol=1e-14):
        raise ValueError('The prepared mixture dictionary requires a uniform cell grid.')
    if (alpha.ndim != 1 or alpha.size == 0 or beta.shape != alpha.shape
            or not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(beta))
            or np.any(alpha <= 0.0) or np.any(beta <= 0.0)):
        raise ValueError('alpha and beta must be matching positive finite shape vectors.')
    cdf = betainc(alpha[:, None], beta[:, None], edges[None, :])
    survival = betaincc(alpha[:, None], beta[:, None], edges[None, :])
    mass = np.where(cdf[:, :-1] >= 0.5, -np.diff(survival, axis=1), np.diff(cdf, axis=1))
    if (not np.all(np.isfinite(mass)) or np.any(mass < 0.0)
            or not np.allclose(np.sum(mass, axis=1), 1.0, atol=_MASS_TOL, rtol=0.0)):
        raise FloatingPointError('Beta cell integration failed; inspect shapes/grid rather than normalizing.')
    return mass / dx
