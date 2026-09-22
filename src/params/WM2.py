import itertools
import numpy as np

from src.validation.model.WM2 import WM2Validation


class BaseParams:
    """Shared WM2 parameter-fit functionality."""

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _local_std(residual):
        residual = np.asarray(residual, dtype=float).reshape(-1)
        n = residual.size
        window = max(5, n // 20)
        if window % 2 == 0: window += 1
        window = min(window, n if n % 2 else n - 1)
        half = window // 2
        result = np.empty(n)

        for k in range(n):
            lo, hi = max(0, k - half), min(n, k + half + 1)
            segment = residual[lo:hi]
            result[k] = np.std(segment, ddof=1) if segment.size > 1 else 0.0
        return result

    def _fit_uncertainty_params(self, current_conditions, input_data):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        alpha, beta = WM2Validation.reduction_inputs(current_conditions["alpha"], current_conditions["beta"])
        current = WM2Validation.reduced_fit_conditions(current_conditions, self.model.t_steps)
        input_coeff, drift_coeff = self.model.reduced_coefficients(alpha, beta)
        state = self.model._solve_deterministic(current["init_cond"], current["U"], current["B_O"], current["B_I"], current["PE"], input_coeff, drift_coeff, current["kappa_R"])

        residual = data["I"] - state["I"]
        local_std = self._local_std(residual)
        X = np.column_stack((np.ones(self.model.t_steps), state["O"]))
        sigma = self.model._least_squares_qp(X, local_std, A_ub=-X, b_ub=np.zeros(self.model.t_steps), lb=np.array([self.model._FIT_EPS, self.model._FIT_EPS]), ub=np.array([np.inf, np.inf]), ridge=self.model._FIT_RIDGE)

        updated = dict(current_conditions)
        updated.update({"sigma_0": float(sigma[0]), "sigma_O": float(sigma[1]), "residual": residual, "local_std": local_std, "deterministic_state": state})
        return updated

    def _optimize_B_params(self, current_conditions, baseline_feature_vec, input_data):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        X_BI, X_BO = WM2Validation.baseline_features(baseline_feature_vec, self.model.t_steps)
        current = WM2Validation.reduced_fit_conditions(current_conditions, self.model.t_steps)
        state = self.model._solve_deterministic(current["init_cond"], current["U"], current["B_O"], current["B_I"], current["PE"], current["I_coeff"], current["d_coeff"], current["kappa_R"])

        c = current["I_coeff"]
        n_I, n_O = X_BI.shape[1], X_BO.shape[1]
        fixed = c[0] + c[3] * state["V"] + c[4] * data["U"]
        X = np.column_stack((c[1] * X_BI, c[2] * X_BO))
        y = data["I"] - fixed

        A_eq = np.zeros((3, n_I + n_O))
        A_eq[0, :n_I], A_eq[1, n_I:] = 1.0, 1.0
        A_eq[2, :n_I], A_eq[2, n_I:] = np.sum(X_BI, axis=0), -np.sum(X_BO, axis=0)
        weights = self.model._least_squares_qp(X, y, A_eq=A_eq, b_eq=np.array([1.0, 1.0, 0.0]), lb=np.zeros(n_I + n_O), ub=np.ones(n_I + n_O), ridge=self.model._FIT_RIDGE)

        updated = dict(current_conditions)
        updated.update({"B_I_weights": weights[:n_I], "B_O_weights": weights[n_I:], "B_I": X_BI @ weights[:n_I], "B_O": X_BO @ weights[n_I:]})
        validated = WM2Validation.reduced_fit_conditions(updated, self.model.t_steps)
        updated["deterministic_state"] = self.model._solve_deterministic(validated["init_cond"], validated["U"], validated["B_O"], validated["B_I"], validated["PE"], validated["I_coeff"], validated["d_coeff"], validated["kappa_R"])
        return updated

    def _recover_alpha_beta(self, current_conditions, input_data):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        current = WM2Validation.reduced_fit_conditions(current_conditions, self.model.t_steps)
        input_coeff, drift_coeff, beta, tol = WM2Validation.structural_recovery_inputs(current["I_coeff"], current["d_coeff"])

        a_B = input_coeff[1]
        a_O = input_coeff[2] / beta[0] if beta[0] >= beta[1] and beta[0] > tol else (input_coeff[0] + input_coeff[3] + input_coeff[4]) / beta[1]
        a_V = a_O * beta[1] - input_coeff[3]
        a_U = -input_coeff[4] - a_O * beta[2]
        alpha, beta = WM2Validation.recovered_structural_coefficients(np.array([a_B, a_V, a_O, a_U], dtype=float), beta, tol=tol)

        WM2Validation.expected_equilibrium_parameters(alpha, beta)
        recovered_input, recovered_drift = self.model.reduced_coefficients(alpha, beta)
        recovered_input, recovered_drift = WM2Validation.exact_reduced_match(recovered_input, recovered_drift, input_coeff, drift_coeff)
        recovered_state = self.model._solve_deterministic(current["init_cond"], current["U"], current["B_O"], current["B_I"], current["PE"], recovered_input, recovered_drift, current["kappa_R"])

        updated = dict(current_conditions)
        updated.update({"alpha": alpha, "beta": beta, "reduced_fit_input_coeff": input_coeff.copy(), "reduced_fit_d_coeff": drift_coeff.copy(), "I_coeff": recovered_input, "d_coeff": recovered_drift, "recovered_input_coeff": recovered_input, "recovered_d_coeff": recovered_drift, "deterministic_state": recovered_state, "recovered_model_loss": float(np.mean((data["I"] - recovered_state["I"]) ** 2))})
        return updated


class ProxyParams(BaseParams):
    """Legacy proxy-state WM2 reduced-parameter estimator."""

    def _optimize_WM2_params(self, current_conditions, input_data):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        current = WM2Validation.reduced_fit_conditions(current_conditions, self.model.t_steps)
        state = self.model._solve_deterministic(current["init_cond"], current["U"], current["B_O"], current["B_I"], current["PE"], current["I_coeff"], current["d_coeff"], current["kappa_R"])

        B_I, B_O, U = current["B_I"], current["B_O"], data["U"]
        X_I = np.column_stack((np.ones(self.model.t_steps), B_I, B_O, state["V"], U))
        A_ub = np.vstack((X_I, -X_I))
        b_ub = np.concatenate((np.ones(self.model.t_steps), np.zeros(self.model.t_steps)))
        input_coeff = self.model._least_squares_qp(X_I, data["I"], A_ub=A_ub, b_ub=b_ub, ridge=self.model._FIT_RIDGE)

        no_V = input_coeff[0] + input_coeff[1] * B_I + input_coeff[2] * B_O + input_coeff[4] * U
        if abs(input_coeff[3]) > 1e-5:
            V_proxy = (data["I"] - no_V) / input_coeff[3]
            V_proxy += current["init_cond"][0] - V_proxy[0]
            V_proxy = 0.9 * V_proxy + 0.1 * state["V"]
        else:
            V_proxy = state["V"].copy()

        R_proxy = np.gradient(V_proxy, self.model.dt)
        R_proxy[0] = current["init_cond"][1]
        R_dot_proxy = np.gradient(R_proxy, self.model.dt)

        X_D = np.column_stack((np.ones(self.model.t_steps), B_I, B_O, V_proxy, U, -R_proxy))
        A_D, b_D = WM2Validation.scaled_equilibrium_constraints()
        lb = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, self.model._FIT_EPS])
        ub = np.array([np.inf, np.inf, np.inf, -self.model._FIT_EPS, np.inf, np.inf])
        scaled = self.model._least_squares_qp(X_D, R_dot_proxy, A_ub=A_D, b_ub=b_D, lb=lb, ub=ub, ridge=self.model._FIT_RIDGE)

        updated = dict(current_conditions)
        updated.update({"I_coeff": input_coeff, "kappa_R": float(scaled[5]), "d_coeff": scaled[:5] / scaled[5], "V_proxy": V_proxy, "R_proxy": R_proxy})
        validated = WM2Validation.reduced_fit_conditions(updated, self.model.t_steps)
        updated["deterministic_state"] = self.model._solve_deterministic(validated["init_cond"], validated["U"], validated["B_O"], validated["B_I"], validated["PE"], validated["I_coeff"], validated["d_coeff"], validated["kappa_R"])
        return updated

    def fit(self, input_data, initial_conditions, baseline_feature_vec):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        init_cond = WM2Validation.initial_condition(initial_conditions)
        X_BI, X_BO = WM2Validation.baseline_features(baseline_feature_vec, self.model.t_steps)

        mean_I = float(np.mean(data["I"]))
        current = {"init_cond": init_cond.copy(), "U": data["U"].copy(), "PE": data["PE"].copy(), "B_I": np.full(self.model.t_steps, mean_I), "B_O": np.full(self.model.t_steps, mean_I), "I_coeff": np.array([0.0, 1.0, 0.0, 0.0, 0.0]), "d_coeff": np.array([0.25, 0.0, 0.0, -0.5, 0.0]), "kappa_R": 0.5 / self.model.dt, "sigma_0": self.model._FIT_EPS, "sigma_O": self.model._FIT_EPS, "B_I_weights": None, "B_O_weights": None}

        validated = WM2Validation.reduced_fit_conditions(current, self.model.t_steps)
        state = self.model._solve_deterministic(validated["init_cond"], validated["U"], validated["B_O"], validated["B_I"], validated["PE"], validated["I_coeff"], validated["d_coeff"], validated["kappa_R"])
        previous_loss = float(np.mean((data["I"] - state["I"]) ** 2))
        history, relative_tol = [previous_loss], 1e-6

        for _ in range(50):
            candidate = self._optimize_WM2_params(current, data)
            candidate = self._optimize_B_params(candidate, {"B_I": X_BI, "B_O": X_BO}, data)
            candidate = self._optimize_WM2_params(candidate, data)

            validated = WM2Validation.reduced_fit_conditions(candidate, self.model.t_steps)
            state = self.model._solve_deterministic(validated["init_cond"], validated["U"], validated["B_O"], validated["B_I"], validated["PE"], validated["I_coeff"], validated["d_coeff"], validated["kappa_R"])
            loss = float(np.mean((data["I"] - state["I"]) ** 2))
            history.append(loss)
            improvement, threshold = previous_loss - loss, relative_tol * max(1.0, previous_loss)
            if improvement < -threshold: break
            current, previous_loss = candidate, loss
            current["deterministic_state"] = state
            if improvement <= threshold: break

        current["reduced_fit_loss"] = previous_loss
        current = self._recover_alpha_beta(current, data)
        current = self._fit_uncertainty_params(current, data)
        current.update({"loss_history": np.asarray(history), "deterministic_loss": previous_loss, "converged_iterations": len(history) - 1, "fit_method": "proxy"})
        return current


class BruteParams(BaseParams):
    """Grid-refined WM2 fit over c_V, d_V and kappa_R with a convex inner C/D/state QP."""

    @staticmethod
    def _transition_stable(dt, d_v, kappa_r, tol=1e-10):
        A = np.array([[1.0, dt], [kappa_r * dt * d_v, 1.0 - kappa_r * dt]], dtype=float)
        return float(np.max(np.abs(np.linalg.eigvals(A)))) < 1.0 + tol

    def _solve_inner(self, data, init_cond, B_I, B_O, c_v, d_v, kappa_r):
        n, dt = self.model.t_steps, self.model.dt
        i_c, i_d, i_v, i_r = slice(0, 4), slice(4, 8), slice(8, 8 + n), slice(8 + n, 8 + 2 * n)
        n_var = 8 + 2 * n

        X = np.zeros((n, n_var), dtype=float)
        X[:, i_c] = np.column_stack((np.ones(n), B_I, B_O, data["U"]))
        X[np.arange(n), 8 + np.arange(n)] = c_v

        eq_rows, eq_rhs = [], []
        row = np.zeros(n_var); row[8] = 1.0; eq_rows.append(row); eq_rhs.append(float(init_cond[0]))
        row = np.zeros(n_var); row[8 + n] = 1.0; eq_rows.append(row); eq_rhs.append(float(init_cond[1]))

        for k in range(n - 1):
            row = np.zeros(n_var)
            row[8 + k + 1], row[8 + k], row[8 + n + k] = 1.0, -1.0, -dt
            eq_rows.append(row); eq_rhs.append(0.0)

            row = np.zeros(n_var)
            row[8 + n + k + 1] = 1.0
            row[8 + n + k] = -(1.0 - kappa_r * dt)
            row[8 + k] = -kappa_r * dt * d_v
            row[4:8] = -kappa_r * dt * np.array([1.0, B_I[k], B_O[k], data["U"][k]])
            eq_rows.append(row); eq_rhs.append(0.0)

        row = np.zeros(n_var); row[5], row[1] = 1.0, -1.0; eq_rows.append(row); eq_rhs.append(0.0)
        row = np.zeros(n_var); row[0], row[4], row[7], row[3] = 1.0, -1.0, -1.0, 1.0; eq_rows.append(row); eq_rhs.append(0.0)
        row = np.zeros(n_var); row[0], row[2], row[4], row[6] = 1.0, 1.0, -1.0, -1.0; eq_rows.append(row); eq_rhs.append(1.0 - (c_v - d_v))
        A_eq, b_eq = np.vstack(eq_rows), np.asarray(eq_rhs, dtype=float)

        ub_rows, ub_rhs = [X, -X], [np.ones(n), np.zeros(n)]
        for b_i, b_o, u in itertools.product((0.0, 1.0), repeat=3):
            num = np.zeros(n_var); num[4:8] = np.array([1.0, b_i, b_o, u])
            ub_rows.extend((num[None, :], (-num)[None, :])); ub_rhs.extend((np.array([-d_v]), np.array([0.0])))

        row = np.zeros(n_var); row[6], row[2] = 1.0, -1.0; ub_rows.append(row[None, :]); ub_rhs.append(np.array([0.0]))
        row = np.zeros(n_var); row[4], row[0] = 1.0, -1.0; ub_rows.append(row[None, :]); ub_rhs.append(np.array([0.0]))
        A_ub, b_ub = np.vstack(ub_rows), np.concatenate(ub_rhs)

        solution = self.model._least_squares_qp(X, data["I"], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, ridge=self.model._FIT_RIDGE)
        c_bar, d_bar = solution[i_c], solution[i_d]
        input_coeff = np.array([c_bar[0], c_bar[1], c_bar[2], c_v, c_bar[3]], dtype=float)
        drift_coeff = np.array([d_bar[0], d_bar[1], d_bar[2], d_v, d_bar[3]], dtype=float)

        current = {"init_cond": np.asarray(init_cond, dtype=float).copy(), "U": data["U"].copy(), "PE": data["PE"].copy(), "B_I": np.asarray(B_I, dtype=float).copy(), "B_O": np.asarray(B_O, dtype=float).copy(), "I_coeff": input_coeff, "d_coeff": drift_coeff, "kappa_R": float(kappa_r), "sigma_0": self.model._FIT_EPS, "sigma_O": self.model._FIT_EPS, "latent_V": solution[i_v].copy(), "latent_R": solution[i_r].copy()}
        current = self._recover_alpha_beta(current, data)
        current.update({"structural_viable": True, "brute_c_V": float(c_v), "brute_d_V": float(d_v), "brute_kappa_R": float(kappa_r)})
        return current

    @staticmethod
    def _refined_interval(values, best, global_lo, global_hi):
        values = np.unique(np.asarray(values, dtype=float))
        width = 0.1 * max(1.0, global_hi - global_lo) if values.size <= 1 else float(np.max(np.diff(values)))
        lo, hi = max(global_lo, best - width), min(global_hi, best + width)
        if hi <= lo: hi = min(global_hi, lo + max(width, 1e-8))
        return lo, hi

    def _search_branch(self, branch, data, init_cond, B_I, B_O, grid_points, refinements, c_v_range, d_v_range, ratio_eps):
        c_bounds, d_bounds = tuple(map(float, c_v_range)), tuple(map(float, d_v_range))
        rho_global = {"under": (ratio_eps, 1.0 - ratio_eps), "critical": (1.0, 1.0), "over": (1.0 + ratio_eps, 2.0)}[branch]
        c_local, d_local, rho_local = c_bounds, d_bounds, rho_global
        best, history = None, []

        for level in range(refinements):
            c_grid = np.linspace(*c_local, grid_points)
            d_grid = np.linspace(*d_local, grid_points)
            rho_grid = np.array([1.0]) if branch == "critical" else np.linspace(*rho_local, grid_points)
            level_best, tested, pruned = None, 0, 0

            for d_v in d_grid:
                if d_v >= -self.model._FIT_EPS: continue
                for c_v in c_grid:
                    beta_v = c_v - d_v
                    if beta_v < -1e-10 or beta_v > 1.0 + 1e-10:
                        pruned += rho_grid.size
                        continue
                    for rho in rho_grid:
                        kappa_r = 4.0 * float(rho) * abs(float(d_v))
                        if kappa_r <= 0.0 or not self._transition_stable(self.model.dt, d_v, kappa_r):
                            pruned += 1
                            continue
                        tested += 1
                        try:
                            candidate = self._solve_inner(data, init_cond, B_I, B_O, c_v, d_v, kappa_r)
                        except (ValueError, RuntimeError, np.linalg.LinAlgError):
                            continue

                        if not candidate.get("structural_viable", False): continue
                        score = float(candidate["recovered_model_loss"])
                        if level_best is None or score < level_best["score"]: level_best = {"score": score, "c_v": float(c_v), "d_v": float(d_v), "rho": float(rho), "kappa_R": float(kappa_r), "alpha": candidate["alpha"].copy(), "beta": candidate["beta"].copy(), "result": candidate}

            history.append({"level": level, "tested": tested, "pruned": pruned, "best_loss": np.inf if level_best is None else level_best["score"]})
            if level_best is None: break
            best = level_best
            c_local = self._refined_interval(c_grid, best["c_v"], *c_bounds)
            d_local = self._refined_interval(d_grid, best["d_v"], *d_bounds)
            if branch != "critical": rho_local = self._refined_interval(rho_grid, best["rho"], *rho_global)

        if best is not None: best.update({"history": history, "branch": branch})
        return best

    def _search_all(self, data, init_cond, B_I, B_O, grid_points, refinements, c_v_range, d_v_range, ratio_eps):
        branch_results = {branch: self._search_branch(branch, data, init_cond, B_I, B_O, grid_points, refinements, c_v_range, d_v_range, ratio_eps) for branch in ("under", "critical", "over")}
        valid = [item for item in branch_results.values() if item is not None]
        if not valid: raise RuntimeError("WM2 brute-force parameter search found no structurally admissible grid candidate.")
        return min(valid, key=lambda item: item["score"]), branch_results

    def fit(self, input_data, initial_conditions, baseline_feature_vec, *, grid_points=10, refinements=3, baseline_iterations=2, c_v_range=(-1.0, 1.0), d_v_range=(-1.0, -1e-4), ratio_eps=1e-3):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        init_cond = WM2Validation.initial_condition(initial_conditions)
        X_BI, X_BO = WM2Validation.baseline_features(baseline_feature_vec, self.model.t_steps)
        grid_points, refinements, baseline_iterations = int(grid_points), int(refinements), int(baseline_iterations)

        if grid_points < 2 or refinements < 1 or baseline_iterations < 1: raise ValueError("grid_points >= 2, refinements >= 1, and baseline_iterations >= 1 are required.")
        if not c_v_range[0] < c_v_range[1]: raise ValueError("c_v_range must be increasing.")
        if not d_v_range[0] < d_v_range[1] < 0.0: raise ValueError("d_v_range must be increasing and strictly negative.")
        if not 0.0 < ratio_eps < 0.5: raise ValueError("ratio_eps must lie in (0, 0.5).")

        mean_I = float(np.mean(data["I"]))
        B_I, B_O = np.full(self.model.t_steps, mean_I), np.full(self.model.t_steps, mean_I)
        all_searches, best = [], None
        last_BI_weights = last_BO_weights = None

        for baseline_iter in range(baseline_iterations):
            best, branches = self._search_all(data, init_cond, B_I, B_O, grid_points, refinements, c_v_range, d_v_range, ratio_eps)
            all_searches.append({"baseline_iteration": baseline_iter, "branches": branches})
            result = best["result"]
            if baseline_iter == baseline_iterations - 1: break

            try:
                updated = self._optimize_B_params(result, {"B_I": X_BI, "B_O": X_BO}, data)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                break
            B_I, B_O = updated["B_I"], updated["B_O"]
            last_BI_weights, last_BO_weights = updated.get("B_I_weights"), updated.get("B_O_weights")

        result = dict(best["result"])
        result.update({"alpha": best["alpha"].copy(), "beta": best["beta"].copy(), "B_I_weights": last_BI_weights, "B_O_weights": last_BO_weights})
        result = self._fit_uncertainty_params(result, data)
        result.update({"fit_method": "brute", "brute_branch": best["branch"], "brute_score": float(best["score"]), "brute_grid_points": grid_points, "brute_refinements": refinements, "brute_search_history": all_searches, "deterministic_loss": float(result["recovered_model_loss"])})
        return result
