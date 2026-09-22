import itertools
import numpy as np

from src.validation.model.WM2 import WM2Validation


class ProxyParams:
    """Legacy proxy-state WM2 parameter estimator.

    This class contains the previous WM2.fit_params implementation.  It fits
    reduced input coefficients from a deterministic state, constructs proxy V,
    R and R_dot trajectories, fits the scaled drift coefficients by convex QP,
    alternates the baseline fit, recovers alpha/beta, and finally estimates the
    diffusion parameters.
    """

    def __init__(self, model):
        self.model = model

    def _optimize_WM2_params(self, current_conditions, input_data):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        current = WM2Validation.reduced_fit_conditions(current_conditions, self.model.t_steps)
        state = self.model._solve_deterministic(
            current["init_cond"], current["U"], current["B_O"], current["B_I"],
            current["PE"], current["I_coeff"], current["d_coeff"], current["kappa_R"]
        )

        B_I = current["B_I"]
        B_O = current["B_O"]
        U = data["U"]

        X_I = np.column_stack((np.ones(self.model.t_steps), B_I, B_O, state["V"], U))
        A_ub = np.vstack((X_I, -X_I))
        b_ub = np.concatenate((np.ones(self.model.t_steps), np.zeros(self.model.t_steps)))
        input_coeff = self.model._least_squares_qp(
            X_I, data["I"], A_ub=A_ub, b_ub=b_ub, ridge=self.model._FIT_RIDGE
        )

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
        scaled = self.model._least_squares_qp(
            X_D, R_dot_proxy, A_ub=A_D, b_ub=b_D, lb=lb, ub=ub,
            ridge=self.model._FIT_RIDGE
        )

        updated = dict(current_conditions)
        updated["I_coeff"] = input_coeff
        updated["kappa_R"] = float(scaled[5])
        updated["d_coeff"] = scaled[:5] / scaled[5]
        updated["V_proxy"] = V_proxy
        updated["R_proxy"] = R_proxy
        validated = WM2Validation.reduced_fit_conditions(updated, self.model.t_steps)
        updated["deterministic_state"] = self.model._solve_deterministic(
            validated["init_cond"], validated["U"], validated["B_O"], validated["B_I"],
            validated["PE"], validated["I_coeff"], validated["d_coeff"], validated["kappa_R"]
        )
        return updated

    def _optimize_B_params(self, current_conditions, baseline_feature_vec, input_data):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        X_BI, X_BO = WM2Validation.baseline_features(baseline_feature_vec, self.model.t_steps)
        current = WM2Validation.reduced_fit_conditions(current_conditions, self.model.t_steps)
        state = self.model._solve_deterministic(
            current["init_cond"], current["U"], current["B_O"], current["B_I"],
            current["PE"], current["I_coeff"], current["d_coeff"], current["kappa_R"]
        )

        c = current["I_coeff"]
        n_I = X_BI.shape[1]
        n_O = X_BO.shape[1]
        fixed = c[0] + c[3] * state["V"] + c[4] * data["U"]
        X = np.column_stack((c[1] * X_BI, c[2] * X_BO))
        y = data["I"] - fixed

        A_eq = np.zeros((3, n_I + n_O))
        b_eq = np.array([1.0, 1.0, 0.0])
        A_eq[0, :n_I] = 1.0
        A_eq[1, n_I:] = 1.0
        A_eq[2, :n_I] = np.sum(X_BI, axis=0)
        A_eq[2, n_I:] = -np.sum(X_BO, axis=0)

        weights = self.model._least_squares_qp(
            X, y, A_eq=A_eq, b_eq=b_eq,
            lb=np.zeros(n_I + n_O), ub=np.ones(n_I + n_O),
            ridge=self.model._FIT_RIDGE
        )

        updated = dict(current_conditions)
        updated["B_I_weights"] = weights[:n_I]
        updated["B_O_weights"] = weights[n_I:]
        updated["B_I"] = X_BI @ weights[:n_I]
        updated["B_O"] = X_BO @ weights[n_I:]
        validated = WM2Validation.reduced_fit_conditions(updated, self.model.t_steps)
        updated["deterministic_state"] = self.model._solve_deterministic(
            validated["init_cond"], validated["U"], validated["B_O"], validated["B_I"],
            validated["PE"], validated["I_coeff"], validated["d_coeff"], validated["kappa_R"]
        )
        return updated

    def _recover_alpha_beta(self, current_conditions, input_data):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        current = WM2Validation.reduced_fit_conditions(current_conditions, self.model.t_steps)
        input_coeff, drift_coeff, beta, tol = WM2Validation.structural_recovery_inputs(
            current["I_coeff"], current["d_coeff"]
        )

        a_B = input_coeff[1]
        if beta[0] >= beta[1] and beta[0] > tol:
            a_O = input_coeff[2] / beta[0]
        else:
            a_O = (input_coeff[0] + input_coeff[3] + input_coeff[4]) / beta[1]

        a_V = a_O * beta[1] - input_coeff[3]
        a_U = -input_coeff[4] - a_O * beta[2]
        alpha = np.array([a_B, a_V, a_O, a_U], dtype=float)
        alpha, beta = WM2Validation.recovered_structural_coefficients(alpha, beta, tol=tol)

        WM2Validation.expected_equilibrium_parameters(alpha, beta)
        recovered_input, recovered_drift = self.model.reduced_coefficients(alpha, beta)
        recovered_input, recovered_drift = WM2Validation.exact_reduced_match(
            recovered_input, recovered_drift, input_coeff, drift_coeff
        )
        recovered_state = self.model._solve_deterministic(
            current["init_cond"], current["U"], current["B_O"], current["B_I"], current["PE"],
            recovered_input, recovered_drift, current["kappa_R"]
        )

        updated = dict(current_conditions)
        updated["alpha"] = alpha
        updated["beta"] = beta
        updated["reduced_fit_input_coeff"] = input_coeff.copy()
        updated["reduced_fit_d_coeff"] = drift_coeff.copy()
        updated["I_coeff"] = recovered_input
        updated["d_coeff"] = recovered_drift
        updated["recovered_input_coeff"] = recovered_input
        updated["recovered_d_coeff"] = recovered_drift
        updated["deterministic_state"] = recovered_state
        updated["recovered_model_loss"] = float(np.mean((data["I"] - recovered_state["I"]) ** 2))
        return updated

    @staticmethod
    def _local_std(residual):
        residual = np.asarray(residual, dtype=float).reshape(-1)
        n = residual.size
        window = max(5, n // 20)
        if window % 2 == 0:
            window += 1
        window = min(window, n if n % 2 else n - 1)
        half = window // 2
        result = np.empty(n)
        for k in range(n):
            lo = max(0, k - half)
            hi = min(n, k + half + 1)
            segment = residual[lo:hi]
            result[k] = np.std(segment, ddof=1) if segment.size > 1 else 0.0
        return result

    def _optimize_uncertainty_params(self, current_conditions, input_data):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        alpha, beta = WM2Validation.reduction_inputs(current_conditions["alpha"], current_conditions["beta"])
        current = WM2Validation.reduced_fit_conditions(current_conditions, self.model.t_steps)
        input_coeff, drift_coeff = self.model.reduced_coefficients(alpha, beta)
        state = self.model._solve_deterministic(
            current["init_cond"], current["U"], current["B_O"], current["B_I"], current["PE"],
            input_coeff, drift_coeff, current["kappa_R"]
        )

        residual = data["I"] - state["I"]
        local_std = self._local_std(residual)
        X = np.column_stack((np.ones(self.model.t_steps), state["O"]))
        sigma = self.model._least_squares_qp(
            X, local_std, A_ub=-X, b_ub=np.zeros(self.model.t_steps),
            lb=np.array([self.model._FIT_EPS, self.model._FIT_EPS]),
            ub=np.array([np.inf, np.inf]), ridge=self.model._FIT_RIDGE
        )

        updated = dict(current_conditions)
        updated["sigma_0"] = float(sigma[0])
        updated["sigma_O"] = float(sigma[1])
        updated["residual"] = residual
        updated["local_std"] = local_std
        updated["deterministic_state"] = state
        return updated

    def fit(self, input_data, initial_conditions, baseline_feature_vec):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        init_cond = WM2Validation.initial_condition(initial_conditions)
        X_BI, X_BO = WM2Validation.baseline_features(baseline_feature_vec, self.model.t_steps)

        mean_I = float(np.mean(data["I"]))
        current = {
            "init_cond": init_cond.copy(), "U": data["U"].copy(), "PE": data["PE"].copy(),
            "B_I": np.full(self.model.t_steps, mean_I), "B_O": np.full(self.model.t_steps, mean_I),
            "I_coeff": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            "d_coeff": np.array([0.25, 0.0, 0.0, -0.5, 0.0]),
            "kappa_R": 0.5 / self.model.dt,
            "sigma_0": self.model._FIT_EPS, "sigma_O": self.model._FIT_EPS,
            "B_I_weights": None, "B_O_weights": None,
        }

        validated = WM2Validation.reduced_fit_conditions(current, self.model.t_steps)
        state = self.model._solve_deterministic(
            validated["init_cond"], validated["U"], validated["B_O"], validated["B_I"],
            validated["PE"], validated["I_coeff"], validated["d_coeff"], validated["kappa_R"]
        )
        previous_loss = float(np.mean((data["I"] - state["I"]) ** 2))
        history = [previous_loss]
        relative_tol = 1e-6

        for _ in range(50):
            candidate = self._optimize_WM2_params(current, data)
            candidate = self._optimize_B_params(candidate, {"B_I": X_BI, "B_O": X_BO}, data)
            candidate = self._optimize_WM2_params(candidate, data)

            validated = WM2Validation.reduced_fit_conditions(candidate, self.model.t_steps)
            state = self.model._solve_deterministic(
                validated["init_cond"], validated["U"], validated["B_O"], validated["B_I"],
                validated["PE"], validated["I_coeff"], validated["d_coeff"], validated["kappa_R"]
            )
            loss = float(np.mean((data["I"] - state["I"]) ** 2))
            history.append(loss)
            improvement = previous_loss - loss
            threshold = relative_tol * max(1.0, previous_loss)
            if improvement < -threshold:
                break
            current = candidate
            current["deterministic_state"] = state
            previous_loss = loss
            if improvement <= threshold:
                break

        current["reduced_fit_loss"] = previous_loss
        current = self._recover_alpha_beta(current, data)
        current = self._optimize_uncertainty_params(current, data)
        current["loss_history"] = np.asarray(history)
        current["deterministic_loss"] = previous_loss
        current["converged_iterations"] = len(history) - 1
        current["fit_method"] = "proxy"
        return current


class BruteParams:
    """Grid-refined latent-state WM2 parameter estimator.

    V0 and R0 are supplied as statistical initial-state estimates.  The only
    outer brute-force parameters are c_V, d_V and kappa_R.  For every fixed
    grid point, all remaining reduced C/D coefficients and the complete V/R
    trajectory are solved jointly by one convex QP with exact discrete WM2
    state constraints.

    The kappa_R search is split into three damping branches:
      underdamped : 0 < kappa_R < 4 |d_V|
      critical    : kappa_R = 4 |d_V|
      overdamped  : 4 |d_V| < kappa_R <= 8 |d_V|
    """

    def __init__(self, model):
        self.model = model
        self._shared = ProxyParams(model)

    @staticmethod
    def _transition_stable(dt, d_v, kappa_r, tol=1e-10):
        A = np.array([[1.0, dt], [kappa_r * dt * d_v, 1.0 - kappa_r * dt]], dtype=float)
        return float(np.max(np.abs(np.linalg.eigvals(A)))) < 1.0 + tol

    def _solve_inner(self, data, init_cond, B_I, B_O, c_v, d_v, kappa_r):
        n = self.model.t_steps
        dt = self.model.dt
        # x = [c0,c_BI,c_BO,c_U,d0,d_BI,d_BO,d_U,V_0..V_N-1,R_0..R_N-1]
        i_c = slice(0, 4)
        i_d = slice(4, 8)
        i_v = slice(8, 8 + n)
        i_r = slice(8 + n, 8 + 2 * n)
        n_var = 8 + 2 * n

        X = np.zeros((n, n_var), dtype=float)
        X[:, i_c] = np.column_stack((np.ones(n), B_I, B_O, data["U"]))
        X[np.arange(n), 8 + np.arange(n)] = c_v

        eq_rows = []
        eq_rhs = []

        row = np.zeros(n_var); row[8] = 1.0
        eq_rows.append(row); eq_rhs.append(float(init_cond[0]))
        row = np.zeros(n_var); row[8 + n] = 1.0
        eq_rows.append(row); eq_rhs.append(float(init_cond[1]))

        for k in range(n - 1):
            row = np.zeros(n_var)
            row[8 + k + 1] = 1.0
            row[8 + k] = -1.0
            row[8 + n + k] = -dt
            eq_rows.append(row); eq_rhs.append(0.0)

            row = np.zeros(n_var)
            row[8 + n + k + 1] = 1.0
            row[8 + n + k] = -(1.0 - kappa_r * dt)
            row[8 + k] = -kappa_r * dt * d_v
            row[4:8] = -kappa_r * dt * np.array([1.0, B_I[k], B_O[k], data["U"][k]])
            eq_rows.append(row); eq_rhs.append(0.0)

        # Exact reduced-map consistency needed to recover beta.
        row = np.zeros(n_var); row[5] = 1.0; row[1] = -1.0
        eq_rows.append(row); eq_rhs.append(0.0)  # d_BI = c_BI

        row = np.zeros(n_var)
        row[0] = 1.0; row[4] = -1.0; row[7] = -1.0; row[3] = 1.0
        eq_rows.append(row); eq_rhs.append(0.0)  # c0-d0 = dU-cU

        row = np.zeros(n_var)
        row[0] = 1.0; row[2] = 1.0; row[4] = -1.0; row[6] = -1.0
        eq_rows.append(row); eq_rhs.append(1.0 - (c_v - d_v))

        A_eq = np.vstack(eq_rows)
        b_eq = np.asarray(eq_rhs, dtype=float)

        ub_rows = [X, -X]
        ub_rhs = [np.ones(n), np.zeros(n)]

        # 0 <= d0+d_BI B_I+d_BO B_O+d_U U <= -d_V at all cube vertices.
        for b_i, b_o, u in itertools.product((0.0, 1.0), repeat=3):
            num = np.zeros(n_var)
            num[4:8] = np.array([1.0, b_i, b_o, u])
            ub_rows.append(num[None, :]); ub_rhs.append(np.array([-d_v]))
            ub_rows.append((-num)[None, :]); ub_rhs.append(np.array([0.0]))

        # beta_B=c_BO-d_BO >= 0 and beta_U=c0-d0 >= 0.
        row = np.zeros(n_var); row[6] = 1.0; row[2] = -1.0
        ub_rows.append(row[None, :]); ub_rhs.append(np.array([0.0]))
        row = np.zeros(n_var); row[4] = 1.0; row[0] = -1.0
        ub_rows.append(row[None, :]); ub_rhs.append(np.array([0.0]))

        A_ub = np.vstack(ub_rows)
        b_ub = np.concatenate(ub_rhs)

        solution = self.model._least_squares_qp(
            X, data["I"], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            ridge=self.model._FIT_RIDGE
        )

        c_bar = solution[i_c]
        d_bar = solution[i_d]
        input_coeff = np.array([c_bar[0], c_bar[1], c_bar[2], c_v, c_bar[3]], dtype=float)
        drift_coeff = np.array([d_bar[0], d_bar[1], d_bar[2], d_v, d_bar[3]], dtype=float)
        V = solution[i_v].copy()
        R = solution[i_r].copy()

        current = {
            "init_cond": np.asarray(init_cond, dtype=float).copy(),
            "U": data["U"].copy(), "PE": data["PE"].copy(),
            "B_I": np.asarray(B_I, dtype=float).copy(), "B_O": np.asarray(B_O, dtype=float).copy(),
            "I_coeff": input_coeff, "d_coeff": drift_coeff, "kappa_R": float(kappa_r),
            "sigma_0": self.model._FIT_EPS, "sigma_O": self.model._FIT_EPS,
            "latent_V": V, "latent_R": R,
        }
        current = self._shared._recover_alpha_beta(current, data)
        current["brute_c_V"] = float(c_v)
        current["brute_d_V"] = float(d_v)
        current["brute_kappa_R"] = float(kappa_r)
        return current

    @staticmethod
    def _refined_interval(values, best, global_lo, global_hi):
        values = np.unique(np.asarray(values, dtype=float))
        if values.size <= 1:
            width = 0.1 * max(1.0, global_hi - global_lo)
        else:
            width = float(np.max(np.diff(values)))
        lo = max(global_lo, best - width)
        hi = min(global_hi, best + width)
        if hi <= lo:
            hi = min(global_hi, lo + max(width, 1e-8))
        return lo, hi

    def _search_branch(self, branch, data, init_cond, B_I, B_O, grid_points, refinements,
                       c_v_range, d_v_range, ratio_eps):
        c_bounds = tuple(map(float, c_v_range))
        d_bounds = tuple(map(float, d_v_range))
        rho_global = {"under": (ratio_eps, 1.0 - ratio_eps),
                      "critical": (1.0, 1.0),
                      "over": (1.0 + ratio_eps, 2.0)}[branch]
        c_local = c_bounds
        d_local = d_bounds
        rho_local = rho_global
        best = None
        history = []

        for level in range(refinements):
            c_grid = np.linspace(c_local[0], c_local[1], grid_points)
            d_grid = np.linspace(d_local[0], d_local[1], grid_points)
            rho_grid = np.array([1.0]) if branch == "critical" else np.linspace(rho_local[0], rho_local[1], grid_points)
            level_best = None
            tested = 0
            pruned = 0

            for d_v in d_grid:
                if d_v >= -self.model._FIT_EPS:
                    continue
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
                        score = float(candidate["recovered_model_loss"])
                        if level_best is None or score < level_best["score"]:
                            level_best = {"score": score, "c_v": float(c_v), "d_v": float(d_v),
                                          "rho": float(rho), "kappa_R": float(kappa_r),
                                          "result": candidate}

            history.append({"level": level, "tested": tested, "pruned": pruned,
                            "best_loss": np.inf if level_best is None else level_best["score"]})
            if level_best is None:
                break
            best = level_best
            c_local = self._refined_interval(c_grid, best["c_v"], c_bounds[0], c_bounds[1])
            d_local = self._refined_interval(d_grid, best["d_v"], d_bounds[0], d_bounds[1])
            if branch != "critical":
                rho_local = self._refined_interval(rho_grid, best["rho"], rho_global[0], rho_global[1])

        if best is not None:
            best["history"] = history
            best["branch"] = branch
        return best

    def _search_all(self, data, init_cond, B_I, B_O, grid_points, refinements,
                    c_v_range, d_v_range, ratio_eps):
        branch_results = {}
        for branch in ("under", "critical", "over"):
            branch_results[branch] = self._search_branch(
                branch, data, init_cond, B_I, B_O, grid_points, refinements,
                c_v_range, d_v_range, ratio_eps
            )
        valid = [item for item in branch_results.values() if item is not None]
        if not valid:
            raise RuntimeError("WM2 brute-force parameter search found no admissible grid candidate.")
        best = min(valid, key=lambda item: item["score"])
        return best, branch_results

    def fit(self, input_data, initial_conditions, baseline_feature_vec, *, grid_points=10,
            refinements=3, baseline_iterations=2, c_v_range=(-1.0, 1.0),
            d_v_range=(-1.0, -1e-4), ratio_eps=1e-3):
        data = WM2Validation.fit_data(input_data, self.model.t_steps)
        init_cond = WM2Validation.initial_condition(initial_conditions)
        X_BI, X_BO = WM2Validation.baseline_features(baseline_feature_vec, self.model.t_steps)
        grid_points = int(grid_points)
        refinements = int(refinements)
        baseline_iterations = int(baseline_iterations)
        if grid_points < 2 or refinements < 1 or baseline_iterations < 1:
            raise ValueError("grid_points >= 2, refinements >= 1, and baseline_iterations >= 1 are required.")
        if not (c_v_range[0] < c_v_range[1]):
            raise ValueError("c_v_range must be increasing.")
        if not (d_v_range[0] < d_v_range[1] < 0.0):
            raise ValueError("d_v_range must be increasing and strictly negative.")
        if not 0.0 < ratio_eps < 0.5:
            raise ValueError("ratio_eps must lie in (0, 0.5).")

        mean_I = float(np.mean(data["I"]))
        B_I = np.full(self.model.t_steps, mean_I)
        B_O = np.full(self.model.t_steps, mean_I)
        all_searches = []
        best = None

        for baseline_iter in range(baseline_iterations):
            best, branches = self._search_all(
                data, init_cond, B_I, B_O, grid_points, refinements,
                c_v_range, d_v_range, ratio_eps
            )
            all_searches.append({"baseline_iteration": baseline_iter, "branches": branches})
            result = best["result"]
            if baseline_iter == baseline_iterations - 1:
                break
            try:
                updated = self._shared._optimize_B_params(
                    result, {"B_I": X_BI, "B_O": X_BO}, data
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                break
            B_I = updated["B_I"]
            B_O = updated["B_O"]

        result = dict(best["result"])
        # Fit baseline weights once more for the final candidate, then re-run the
        # local search so the reported C/D and latent state correspond to them.
        try:
            baseline_result = self._shared._optimize_B_params(
                result, {"B_I": X_BI, "B_O": X_BO}, data
            )
            B_I = baseline_result["B_I"]
            B_O = baseline_result["B_O"]
            final_best, final_branches = self._search_all(
                data, init_cond, B_I, B_O, grid_points, 1,
                c_v_range, d_v_range, ratio_eps
            )
            result = dict(final_best["result"])
            result["B_I_weights"] = baseline_result.get("B_I_weights")
            result["B_O_weights"] = baseline_result.get("B_O_weights")
            best = final_best
            all_searches.append({"baseline_iteration": "final", "branches": final_branches})
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass

        result = self._shared._optimize_uncertainty_params(result, data)
        result["fit_method"] = "brute"
        result["brute_branch"] = best["branch"]
        result["brute_score"] = float(best["score"])
        result["brute_grid_points"] = grid_points
        result["brute_refinements"] = refinements
        result["brute_search_history"] = all_searches
        result["deterministic_loss"] = float(result["recovered_model_loss"])
        return result
