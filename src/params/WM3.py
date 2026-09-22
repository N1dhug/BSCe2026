from time import perf_counter

import numpy as np

from src.validation.model.WM3 import WM3Validation


class BruteParams:
    """Grid-refined WM3 fit over underdamped and critically damped responses.

    The outer search fixes c_V, d_V, alpha_O and the damping ratio. For each
    candidate these values determine beta_V, alpha_V and alpha_U exactly.
    The remaining beta coefficients [beta_U, beta_L, beta_H] and latent
    trajectories [V, R] are then obtained from one convex least-squares QP.
    """

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _transition_stable(dt, d_v, kappa_r, tol=1e-10):
        A = np.array([[1.0, dt], [kappa_r * dt * d_v, 1.0 - kappa_r * dt]], dtype=float)
        return float(np.max(np.abs(np.linalg.eigvals(A)))) < 1.0 + tol

    def _prepare_data(self, input_data):
        data = WM3Validation.fit_data(input_data, self.model.t_steps)
        L = np.empty(self.model.t_steps)
        H = np.empty(self.model.t_steps)

        for k in range(self.model.t_steps):
            L[k] = WM3Validation.feature_value(self.model.L(data["Time"][k]), "L(Time)")
            H[k] = WM3Validation.feature_value(self.model.H(data["Chill_factor"][k]), "H(Chill_factor)")

        prepared = dict(data)
        prepared.update({"L": L, "H": H})
        return prepared

    def _solve_inner(self, data, init_cond, c_v, d_v, alpha_o, kappa_r):
        n, dt = self.model.t_steps, self.model.dt
        beta_v = c_v - d_v
        alpha_v = alpha_o * beta_v - c_v
        alpha_u = 1.0 - alpha_v - alpha_o
        tol = 1e-9

        if beta_v < -tol or beta_v > 1.0 + tol: raise ValueError("beta_V is outside [0, 1].")
        if alpha_v < -tol or alpha_u < -tol or alpha_o < -tol or alpha_v > 1.0 + tol or alpha_u > 1.0 + tol or alpha_o > 1.0 + tol: raise ValueError("alpha is outside the simplex.")

        beta_v = float(np.clip(beta_v, 0.0, 1.0))
        alpha = np.array([alpha_v, alpha_u, alpha_o], dtype=float)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha /= np.sum(alpha)

        i_b, i_v, i_r = slice(0, 3), slice(3, 3 + n), slice(3 + n, 3 + 2 * n)
        n_var = 3 + 2 * n
        F = np.column_stack((1.0 - data["U"], data["L"], data["H"]))
        fixed = alpha[0] + alpha[1] - alpha[1] * data["U"]

        X = np.zeros((n, n_var), dtype=float)
        X[:, i_b] = alpha[2] * F
        X[np.arange(n), 3 + np.arange(n)] = c_v
        y = data["I"] - fixed

        eq_rows, eq_rhs = [], []
        row = np.zeros(n_var); row[3] = 1.0; eq_rows.append(row); eq_rhs.append(float(init_cond[0]))
        row = np.zeros(n_var); row[3 + n] = 1.0; eq_rows.append(row); eq_rhs.append(float(init_cond[1]))

        for k in range(n - 1):
            row = np.zeros(n_var)
            row[3 + k + 1], row[3 + k], row[3 + n + k] = 1.0, -1.0, -dt
            eq_rows.append(row); eq_rhs.append(0.0)

            row = np.zeros(n_var)
            row[3 + n + k + 1] = 1.0
            row[3 + n + k] = -(1.0 - kappa_r * dt)
            row[3 + k] = -kappa_r * dt * d_v
            row[i_b] = -kappa_r * dt * (alpha[2] - 1.0) * F[k]
            eq_rows.append(row); eq_rhs.append(kappa_r * dt * fixed[k])

        row = np.zeros(n_var); row[i_b] = 1.0
        eq_rows.append(row); eq_rhs.append(1.0 - beta_v)
        A_eq, b_eq = np.vstack(eq_rows), np.asarray(eq_rhs, dtype=float)

        ub_rows, ub_rhs = [X, -X], [1.0 - fixed, fixed]
        for L_value in (0.0, 1.0):
            for H_value in (0.0, 1.0):
                for U_value in (0.0, 1.0):
                    features = np.array([1.0 - U_value, L_value, H_value], dtype=float)
                    constant = alpha[0] + alpha[1] - alpha[1] * U_value
                    q = (alpha[2] - 1.0) * features
                    row = np.zeros(n_var); row[i_b] = -q
                    ub_rows.append(row[None, :]); ub_rhs.append(np.array([constant]))
                    row = np.zeros(n_var); row[i_b] = q
                    ub_rows.append(row[None, :]); ub_rhs.append(np.array([-d_v - constant]))

        A_ub, b_ub = np.vstack(ub_rows), np.concatenate(ub_rhs)
        lb, ub = np.full(n_var, -np.inf), np.full(n_var, np.inf)
        lb[i_b], ub[i_b] = 0.0, 1.0

        qp_start = perf_counter()
        solution = self.model._least_squares_qp(X, y, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, ridge=self.model._FIT_RIDGE)
        qp_solve_time = perf_counter() - qp_start

        beta = np.array([beta_v, solution[0], solution[1], solution[2]], dtype=float)
        alpha, beta = WM3Validation.recovered_structural_coefficients(alpha, beta)
        input_coeff, drift_coeff = self.model.reduced_coefficients(alpha, beta)

        if not np.isclose(input_coeff[3], c_v, atol=1e-7, rtol=0.0): raise ValueError("Recovered WM3 c_V does not match the brute-force candidate.")
        if not np.isclose(drift_coeff[3], d_v, atol=1e-7, rtol=0.0): raise ValueError("Recovered WM3 d_V does not match the brute-force candidate.")

        state = self.model._solve_deterministic(init_cond, data["U"], data["PE"], data["Time"], data["Chill_factor"], data["L"], data["H"], input_coeff, drift_coeff, kappa_r)
        loss = float(np.mean((data["I"] - state["I"]) ** 2))

        return {"init_cond": np.asarray(init_cond, dtype=float).copy(), "U": data["U"].copy(), "PE": data["PE"].copy(), "Time": data["Time"].copy(), "Chill_factor": data["Chill_factor"].copy(), "L": data["L"].copy(), "H": data["H"].copy(), "alpha": alpha, "beta": beta, "I_coeff": input_coeff, "d_coeff": drift_coeff, "kappa_R": float(kappa_r), "latent_V": solution[i_v].copy(), "latent_R": solution[i_r].copy(), "deterministic_state": state, "recovered_model_loss": loss, "structural_viable": True, "brute_c_V": float(c_v), "brute_d_V": float(d_v), "brute_alpha_O": float(alpha_o), "brute_kappa_R": float(kappa_r), "qp_solve_time": qp_solve_time}

    @staticmethod
    def _refined_interval(values, best, global_lo, global_hi):
        values = np.unique(np.asarray(values, dtype=float))
        width = 0.1 * max(1.0, global_hi - global_lo) if values.size <= 1 else float(np.max(np.diff(values)))
        lo, hi = max(global_lo, best - width), min(global_hi, best + width)
        if hi <= lo: hi = min(global_hi, lo + max(width, 1e-8))
        return lo, hi

    def _search_branch(self, branch, data, init_cond, grid_points, alpha_points, refinements, c_v_range, d_v_range, alpha_o_range, ratio_eps, verbose, progress_every):
        c_bounds, d_bounds, a_bounds = tuple(map(float, c_v_range)), tuple(map(float, d_v_range)), tuple(map(float, alpha_o_range))
        rho_global = {"under": (ratio_eps, 1.0 - ratio_eps), "critical": (1.0, 1.0)}[branch]
        c_local, d_local, a_local, rho_local = c_bounds, d_bounds, a_bounds, rho_global
        best, history = None, []
        branch_start = perf_counter()

        for level in range(refinements):
            c_grid = np.linspace(*c_local, grid_points)
            d_grid = np.linspace(*d_local, grid_points)
            a_grid = np.linspace(*a_local, alpha_points)
            rho_grid = np.array([1.0]) if branch == "critical" else np.linspace(*rho_local, grid_points)
            level_best, tested, pruned, qp_times = None, 0, 0, []
            level_start = perf_counter()
            total_candidates = len(c_grid) * len(d_grid) * len(a_grid) * len(rho_grid)
            if verbose: print(f"[brute] branch={branch} refinement={level + 1}/{refinements} candidates={total_candidates}")

            for d_v in d_grid:
                if d_v >= -self.model._FIT_EPS: continue
                for c_v in c_grid:
                    beta_v = c_v - d_v
                    if beta_v < -1e-10 or beta_v > 1.0 + 1e-10:
                        pruned += len(a_grid) * len(rho_grid)
                        continue

                    for alpha_o in a_grid:
                        alpha_v = alpha_o * beta_v - c_v
                        alpha_u = 1.0 - alpha_v - alpha_o
                        if min(alpha_v, alpha_u, alpha_o) < -1e-10 or max(alpha_v, alpha_u, alpha_o) > 1.0 + 1e-10:
                            pruned += len(rho_grid)
                            continue

                        for rho in rho_grid:
                            kappa_r = 4.0 * float(rho) * abs(float(d_v))
                            if kappa_r <= 0.0 or not self._transition_stable(self.model.dt, d_v, kappa_r):
                                pruned += 1
                                continue

                            tested += 1
                            candidate_start = perf_counter()
                            try:
                                candidate = self._solve_inner(data, init_cond, c_v, d_v, alpha_o, kappa_r)
                            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                                candidate_time = perf_counter() - candidate_start
                                if verbose and tested % progress_every == 0:
                                    elapsed = perf_counter() - level_start
                                    print(f"[brute] branch={branch} refinement={level + 1} tested={tested} pruned={pruned} elapsed={elapsed:.2f}s last_candidate={candidate_time:.4f}s status=failed")
                                continue

                            qp_times.append(float(candidate["qp_solve_time"]))
                            if verbose and (tested == 1 or tested % progress_every == 0):
                                elapsed = perf_counter() - level_start
                                print(f"[brute] branch={branch} refinement={level + 1} tested={tested} pruned={pruned} elapsed={elapsed:.2f}s last_qp={qp_times[-1]:.4f}s mean_qp={np.mean(qp_times):.4f}s")

                            score = float(candidate["recovered_model_loss"])
                            if level_best is None or score < level_best["score"]: level_best = {"score": score, "c_v": float(c_v), "d_v": float(d_v), "alpha_o": float(alpha_o), "rho": float(rho), "kappa_R": float(kappa_r), "alpha": candidate["alpha"].copy(), "beta": candidate["beta"].copy(), "result": candidate}

            level_elapsed = perf_counter() - level_start
            mean_qp = float(np.mean(qp_times)) if qp_times else np.nan
            history.append({"level": level, "tested": tested, "pruned": pruned, "best_loss": np.inf if level_best is None else level_best["score"], "elapsed": level_elapsed, "mean_qp_solve_time": mean_qp, "qp_solves": len(qp_times)})
            if verbose: print(f"[brute] branch={branch} refinement={level + 1} complete tested={tested} pruned={pruned} qp_solves={len(qp_times)} elapsed={level_elapsed:.2f}s mean_qp={mean_qp:.4f}s")
            if level_best is None: break

            best = level_best
            c_local = self._refined_interval(c_grid, best["c_v"], *c_bounds)
            d_local = self._refined_interval(d_grid, best["d_v"], *d_bounds)
            a_local = self._refined_interval(a_grid, best["alpha_o"], *a_bounds)
            if branch != "critical": rho_local = self._refined_interval(rho_grid, best["rho"], *rho_global)

        if best is not None: best.update({"history": history, "branch": branch, "elapsed": perf_counter() - branch_start})
        return best

    def _search_all(self, data, init_cond, grid_points, alpha_points, refinements, c_v_range, d_v_range, alpha_o_range, ratio_eps, verbose, progress_every):
        branch_results = {branch: self._search_branch(branch, data, init_cond, grid_points, alpha_points, refinements, c_v_range, d_v_range, alpha_o_range, ratio_eps, verbose, progress_every) for branch in ("under", "critical")}
        valid = [item for item in branch_results.values() if item is not None]
        if not valid: raise RuntimeError("WM3 brute-force parameter search found no structurally admissible underdamped or critically damped grid candidate.")
        return min(valid, key=lambda item: item["score"]), branch_results

    def fit(self, input_data, initial_conditions, *, grid_points=10, alpha_points=5, refinements=3, c_v_range=(-1.0, 1.0), d_v_range=(-1.0, -1e-4), alpha_o_range=(0.0, 1.0), ratio_eps=1e-3, verbose=True, progress_every=25):
        data = self._prepare_data(input_data)
        init_cond = WM3Validation.initial_condition(initial_conditions)
        grid_points, alpha_points, refinements, progress_every = int(grid_points), int(alpha_points), int(refinements), int(progress_every)

        if grid_points < 2 or alpha_points < 2 or refinements < 1: raise ValueError("grid_points >= 2, alpha_points >= 2, and refinements >= 1 are required.")
        if progress_every < 1: raise ValueError("progress_every must be at least 1.")
        if not c_v_range[0] < c_v_range[1]: raise ValueError("c_v_range must be increasing.")
        if not d_v_range[0] < d_v_range[1] < 0.0: raise ValueError("d_v_range must be increasing and strictly negative.")
        if not 0.0 <= alpha_o_range[0] < alpha_o_range[1] <= 1.0: raise ValueError("alpha_o_range must be an increasing subset of [0, 1].")
        if not 0.0 < ratio_eps < 0.5: raise ValueError("ratio_eps must lie in (0, 0.5).")

        fit_start = perf_counter()
        best, branches = self._search_all(data, init_cond, grid_points, alpha_points, refinements, c_v_range, d_v_range, alpha_o_range, ratio_eps, verbose, progress_every)
        result = dict(best["result"])
        result.update({"fit_method": "brute", "brute_branch": best["branch"], "brute_score": float(best["score"]), "brute_grid_points": grid_points, "brute_alpha_points": alpha_points, "brute_refinements": refinements, "brute_search_history": branches, "brute_fit_time": perf_counter() - fit_start, "deterministic_loss": float(result["recovered_model_loss"])})
        return result
