import numpy as np
from scipy.optimize import linprog

from src.validation.solver.ActiveSet import ActiveSetValidation

class ActiveSet:
    """Active-set solver for convex quadratic programs with linear constraints.

    The generic problem is

        min_x  0.5 * x.T @ H @ x + f.T @ x

    subject to

        A_ub @ x <= b_ub,
        A_eq @ x == b_eq,
        lb <= x <= ub.

    A feasible starting point may be supplied through x0. If it is omitted,
    a Phase-I linear feasibility problem is solved before the active-set
    iterations begin.
    """

    @staticmethod
    def _canonical_inequalities(A_ub, b_ub, lb, ub):
        # Build inequality constraints
        n_variables = lb.size
        matrices = []
        rhs = []

        if A_ub.shape[0]:
            matrices.append(A_ub)
            rhs.append(b_ub)

        finite_ub = np.isfinite(ub)
        if np.any(finite_ub):
            matrices.append(np.eye(n_variables, dtype=float)[finite_ub])
            rhs.append(ub[finite_ub])

        finite_lb = np.isfinite(lb)
        if np.any(finite_lb):
            matrices.append(-np.eye(n_variables, dtype=float)[finite_lb])
            rhs.append(-lb[finite_lb])

        if not matrices:
            return (np.empty((0, n_variables), dtype=float), np.empty(0, dtype=float))
        return np.vstack(matrices), np.concatenate(rhs)

    @staticmethod
    def _is_feasible(x, G, h, A_eq, b_eq, tol):
        # Check feasibility
        scale = max(1.0, float(np.max(np.abs(x))) if x.size else 1.0)
        feasibility_tol = tol * scale

        if G.shape[0] and np.any(G @ x - h > feasibility_tol):
            return False
        if A_eq.shape[0] and np.any(np.abs(A_eq @ x - b_eq) > feasibility_tol):
            return False
        return True

    @staticmethod
    def _find_feasible_point(n_variables, G, h, A_eq, b_eq, tol):
        # Find feasible point
        result = linprog(np.zeros(n_variables, dtype=float), A_ub=G if G.shape[0] else None, b_ub=h if G.shape[0] else None, A_eq=A_eq if A_eq.shape[0] else None, b_eq=b_eq if A_eq.shape[0] else None, bounds=[(None, None)] * n_variables, method="highs",)
        if not result.success:
            raise ValueError("The supplied linear equality, inequality, and bound constraints are infeasible.")

        x = np.asarray(result.x, dtype=float)

        if not ActiveSet._is_feasible(x, G, h, A_eq, b_eq, 10.0 * tol):
            raise RuntimeError("Phase-I feasibility solve returned an infeasible point.")
        return x

    @staticmethod
    def _independent_working_set(G, indices, A_eq):
        # Keep only inequality rows that add rank beyond the equality/current active set.
        selected = []
        constraints = A_eq.copy()
        rank = np.linalg.matrix_rank(constraints) if constraints.shape[0] else 0

        for index in indices:
            row = G[int(index)][None, :]
            candidate = row if constraints.shape[0] == 0 else np.vstack((constraints, row))
            candidate_rank = np.linalg.matrix_rank(candidate)
            if candidate_rank > rank:
                selected.append(int(index))
                constraints, rank = candidate, candidate_rank
            if rank >= G.shape[1]: break
        return set(selected)

    @staticmethod
    def _solve_equality_qp(hessian, gradient, constraints):
        # Solve equality-constrained QP
        n_variables = hessian.shape[0]
        if constraints.shape[0] == 0:
            direction, *_ = np.linalg.lstsq(hessian, -gradient, rcond=None)
            return direction, np.empty(0, dtype=float)

        kkt = np.block([[hessian, constraints.T], [constraints,np.zeros((constraints.shape[0], constraints.shape[0]), dtype=float)]])
        rhs = np.concatenate((-gradient, np.zeros(constraints.shape[0], dtype=float)))
        solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
        return solution[:n_variables], solution[n_variables:]

    @staticmethod
    def solve(hessian,linear,A_ub=None,b_ub=None,A_eq=None, b_eq=None,lb=None,ub=None,x0=None,tol=1e-10,max_iter=None):
        """Solve a convex QP with affine equality/inequality and box constraints."""
        # Solve quadratic program
        # Validate and normalize the QP inputs.
        hessian, linear, A_ub, b_ub, A_eq, b_eq, lb, ub, x0, tol, max_iter = (ActiveSetValidation.quadratic_program(hessian,linear,A_ub,b_ub,A_eq, b_eq,lb,ub,x0,tol, max_iter))

        n_variables = hessian.shape[0]
        G, h = ActiveSet._canonical_inequalities(A_ub, b_ub, lb, ub)

        if x0 is None:
            if G.shape[0] or A_eq.shape[0]:
                x = ActiveSet._find_feasible_point(n_variables, G, h, A_eq, b_eq, tol)
            else:
                x = np.zeros(n_variables, dtype=float)
        else:
            x = x0.copy()
            if not ActiveSet._is_feasible(x, G, h, A_eq, b_eq, tol):
                raise ValueError("x0 must satisfy all supplied constraints.")

        if G.shape[0]:
            slack = h - G @ x
            working = ActiveSet._independent_working_set(G, np.flatnonzero(slack <= tol), A_eq)
        else:
            working = set()

        for _ in range(max_iter):
            active_indices = np.array(sorted(working), dtype=int)
            if active_indices.size:
                active_inequalities = G[active_indices]
                constraints = (np.vstack((A_eq, active_inequalities)) if A_eq.shape[0] else active_inequalities.copy())
            else:
                constraints = A_eq.copy()

            gradient = hessian @ x + linear
            direction, multipliers = ActiveSet._solve_equality_qp(hessian, gradient, constraints)

            direction_scale = max(1.0, float(np.linalg.norm(x, ord=np.inf)))
            if float(np.linalg.norm(direction, ord=np.inf)) <= tol * direction_scale:
                stationarity = gradient.copy()
                if constraints.shape[0]:
                    stationarity = stationarity + constraints.T @ multipliers

                stationarity_scale = max(1.0, float(np.linalg.norm(gradient, ord=np.inf)))
                if (float(np.linalg.norm(stationarity, ord=np.inf)) > 100.0 * tol * stationarity_scale):
                    raise RuntimeError("Active-set KKT solve failed to obtain a stationary direction; ""the convex QP may be unbounded or numerically singular.")
                
                if active_indices.size == 0:
                    return x

                inequality_multipliers = multipliers[A_eq.shape[0] :]
                multiplier_scale = max(1.0, float(np.max(np.abs(inequality_multipliers))), float(np.linalg.norm(gradient, ord=np.inf)),)
                worst = int(np.argmin(inequality_multipliers))
                if inequality_multipliers[worst] >= -tol * multiplier_scale:
                    return x

                working.remove(int(active_indices[worst]))
                continue

            alpha = 1.0
            blocking = None
            if G.shape[0]:
                G_direction = G @ direction
                slack = h - G @ x
                for index in range(G.shape[0]):
                    if index in working or G_direction[index] <= tol:
                        continue
                    
                    candidate = slack[index] / G_direction[index]
                    if candidate < alpha:
                        alpha = max(0.0, float(candidate))
                        blocking = index

            x = x + alpha * direction

            if blocking is not None and alpha < 1.0 - tol:
                working = ActiveSet._independent_working_set(G, sorted(working | {int(blocking)}), A_eq)

            if not ActiveSet._is_feasible(x, G, h, A_eq, b_eq, 100.0 * tol):
                raise RuntimeError("Active-set iteration lost feasibility beyond numerical tolerance.")

        raise RuntimeError("Active-set convex-QP solver did not converge.")
