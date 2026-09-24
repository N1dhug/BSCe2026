import itertools
from time import perf_counter

import numpy as np

from src.validation.model.WM3 import WM3Validation


class BruteParams:
    """Reduced-coordinate brute-force estimator for the complete WM3 model.

    Only the volume-coupled nonlinear parameters (c_V, d_V, kappa_R) are
    searched on the refining grid. For each grid point, all remaining reduced
    I/D coefficients and the latent state trajectory are fitted by a convex QP.
    The initial state can be estimated inside that QP. Structural alpha/beta
    coefficients are recovered afterwards, and sigma_0/sigma_O are fitted from
    the deterministic residual.
    """

    def __init__(self, model): self.model = model

    @staticmethod
    def _transition_stable(dt, d_v, kappa_r, tol=1e-10):
        A = np.array([[1.0, dt], [kappa_r * dt * d_v, 1.0 - kappa_r * dt]], dtype=float)
        return float(np.max(np.abs(np.linalg.eigvals(A)))) < 1.0 + tol

    @staticmethod
    def _local_std(residual):
        residual = np.asarray(residual, dtype=float).reshape(-1)
        n, window = residual.size, max(5, residual.size // 20)
        if window % 2 == 0: window += 1
        window = min(window, n if n % 2 else max(1, n - 1))
        half, result = window // 2, np.empty(n)

        for k in range(n):
            segment = residual[max(0, k - half):min(n, k + half + 1)]
            result[k] = np.std(segment, ddof=1) if segment.size > 1 else 0.0
        return result

    def _prepare_data(self, input_data):
        data = WM3Validation.fit_data(input_data, self.model.t_steps)
        prepared = dict(data)
        prepared["L"] = np.array([WM3Validation.feature_value(self.model.L(value), "L(Time)") for value in data["Time"]], dtype=float)
        prepared["H"] = np.array([WM3Validation.feature_value(self.model.H(value), "H(Temp)") for value in data["Temp"]], dtype=float)
        return prepared

    def _recover_alpha_beta(self, input_coeff, drift_coeff):
        c, d, tol = np.asarray(input_coeff, dtype=float), np.asarray(drift_coeff, dtype=float), 1e-7
        beta = np.array([c[2] - d[2], c[0] - d[0], c[1] - d[1], c[4] - d[4]], dtype=float)

        if not np.isclose(d[3] - c[3], beta[1], atol=tol, rtol=0.0): raise ValueError("Reduced WM3 fit violates beta_U consistency.")
        if np.any(beta < -tol) or np.any(beta > 1.0 + tol): raise ValueError("Reduced WM3 fit implies beta outside [0, 1].")
        if not np.isclose(np.sum(beta), 1.0, atol=tol, rtol=0.0): raise ValueError("Reduced WM3 fit implies beta that does not sum to one.")

        beta = np.clip(beta, 0.0, 1.0)
        beta /= np.sum(beta)

        M = np.array([
            [1.0, 1.0, beta[1]],
            [0.0, 0.0, beta[2]],
            [-1.0, 0.0, beta[0]],
            [0.0, -1.0, -beta[1]],
            [0.0, 0.0, beta[3]],
        ], dtype=float)

        A_eq = np.vstack((np.ones(3), M[2]))
        b_eq = np.array([1.0, c[2]], dtype=float)
        alpha = self.model._least_squares_qp(
            M, c, A_eq=A_eq, b_eq=b_eq,
            lb=np.zeros(3), ub=np.ones(3), ridge=self.model._FIT_RIDGE
        )
        alpha, beta = WM3Validation.recovered_structural_coefficients(alpha, beta)
        recovered_input, recovered_drift = self.model.reduced_coefficients(alpha, beta)

        if not np.isclose(recovered_input[2], c[2], atol=tol, rtol=0.0): raise ValueError("Recovered WM3 c_V does not match the brute-force candidate.")
        if not np.isclose(recovered_drift[2], d[2], atol=tol, rtol=0.0): raise ValueError("Recovered WM3 d_V does not match the brute-force candidate.")

        projection_error = float(np.mean((recovered_input - c) ** 2) + np.mean((recovered_drift - d) ** 2))
        return alpha, beta, recovered_input, recovered_drift, projection_error

    def _solve_inner(self, data, initial_conditions, c_v, d_v, kappa_r, fit_initial_state, r0_range):
        n, dt = self.model.t_steps, self.model.dt
        i_c, i_d = slice(0, 4), slice(4, 8)
        i_v, i_r = slice(8, 8 + n), slice(8 + n, 8 + 2 * n)
        n_var = 8 + 2 * n

        X = np.zeros((n, n_var), dtype=float)
        X[:, i_c] = np.column_stack((np.ones(n), data["L"], data["U"], data["H"]))
        X[np.arange(n), 8 + np.arange(n)] = c_v

        eq_rows, eq_rhs = [], []
        if not fit_initial_state:
            row = np.zeros(n_var); row[8] = 1.0; eq_rows.append(row); eq_rhs.append(float(initial_conditions[0]))
            row = np.zeros(n_var); row[8 + n] = 1.0; eq_rows.append(row); eq_rhs.append(float(initial_conditions[1]))

        for k in range(n - 1):
            row = np.zeros(n_var)
            row[8 + k + 1], row[8 + k], row[8 + n + k] = 1.0, -1.0, -dt
            eq_rows.append(row); eq_rhs.append(0.0)

            row = np.zeros(n_var)
            row[8 + n + k + 1] = 1.0
            row[8 + n + k] = -(1.0 - kappa_r * dt)
            row[8 + k] = -kappa_r * dt * d_v
            row[i_d] = -kappa_r * dt * np.array([1.0, data["L"][k], data["U"][k], data["H"][k]])
            eq_rows.append(row); eq_rhs.append(0.0)

        row = np.zeros(n_var); row[0], row[2], row[4], row[6] = 1.0, 1.0, -1.0, -1.0
        eq_rows.append(row); eq_rhs.append(0.0)

        row = np.zeros(n_var)
        row[0], row[1], row[3], row[4], row[5], row[7] = 1.0, 1.0, 1.0, -1.0, -1.0, -1.0
        eq_rows.append(row); eq_rhs.append(1.0 - (c_v - d_v))

        if np.allclose(data["H"], 0.0, atol=1e-12, rtol=0.0):
            row = np.zeros(n_var); row[3] = 1.0; eq_rows.append(row); eq_rhs.append(0.0)
            row = np.zeros(n_var); row[7] = 1.0; eq_rows.append(row); eq_rhs.append(0.0)

        A_eq, b_eq = np.vstack(eq_rows), np.asarray(eq_rhs, dtype=float)
        ub_rows, ub_rhs = [X, -X], [np.ones(n), np.zeros(n)]

        for U_value, L_value, H_value in itertools.product((0.0, 1.0), repeat=3):
            features = np.array([1.0, L_value, U_value, H_value], dtype=float)
            row = np.zeros(n_var); row[i_d] = -features
            ub_rows.append(row[None, :]); ub_rhs.append(np.array([0.0]))
            row = np.zeros(n_var); row[i_d] = features
            ub_rows.append(row[None, :]); ub_rhs.append(np.array([-d_v]))

        for c_idx, d_idx in ((0, 4), (1, 5), (3, 7)):
            row = np.zeros(n_var); row[d_idx], row[c_idx] = 1.0, -1.0
            ub_rows.append(row[None, :]); ub_rhs.append(np.array([0.0]))

        A_ub, b_ub = np.vstack(ub_rows), np.concatenate(ub_rhs)
        lb, ub = np.full(n_var, -np.inf), np.full(n_var, np.inf)

        if fit_initial_state:
            lb[8], ub[8] = 0.0, 1.0
            lb[8 + n], ub[8 + n] = map(float, r0_range)

        qp_start = perf_counter()
        solution = self.model._least_squares_qp(
            X, data["I"], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            lb=lb, ub=ub, ridge=self.model._FIT_RIDGE
        )
        qp_solve_time = perf_counter() - qp_start

        c_bar, d_bar = solution[i_c], solution[i_d]
        input_coeff = np.array([c_bar[0], c_bar[1], c_v, c_bar[2], c_bar[3]], dtype=float)
        drift_coeff = np.array([d_bar[0], d_bar[1], d_v, d_bar[2], d_bar[3]], dtype=float)
        estimated_init = np.array([solution[8], solution[8 + n]], dtype=float)

        alpha, beta, recovered_input, recovered_drift, projection_error = self._recover_alpha_beta(input_coeff, drift_coeff)
        state = self.model._solve_deterministic(
            estimated_init, data["U"], data["PE"], data["Time"], data["L"],
            recovered_input, recovered_drift, kappa_r, H=data["H"], Temp=data["Temp"]
        )
        reduced_fit = X @ solution
        reduced_loss = float(np.mean((data["I"] - reduced_fit) ** 2))
        recovered_loss = float(np.mean((data["I"] - state["I"]) ** 2))

        return {
            "init_cond": estimated_init,
            "initial_condition_seed": np.asarray(initial_conditions, dtype=float).copy(),
            "U": data["U"].copy(), "PE": data["PE"].copy(),
            "Time": data["Time"].copy(), "Temp": data["Temp"].copy(),
            "L": data["L"].copy(), "H": data["H"].copy(),
            "alpha": alpha, "beta": beta,
            "I_coeff": recovered_input, "d_coeff": recovered_drift,
            "reduced_fit_input_coeff": input_coeff, "reduced_fit_d_coeff": drift_coeff,
            "kappa_R": float(kappa_r),
            "latent_V": solution[i_v].copy(), "latent_R": solution[i_r].copy(),
            "deterministic_state": state,
            "reduced_fit_loss": reduced_loss, "recovered_model_loss": recovered_loss,
            "structural_projection_error": projection_error, "structural_viable": True,
            "brute_c_V": float(c_v), "brute_d_V": float(d_v),
            "brute_kappa_R": float(kappa_r), "brute_alpha_O": float(alpha[2]),
            "qp_solve_time": qp_solve_time,
        }

    def _fit_uncertainty_params(self, result, data):
        state = result["deterministic_state"]
        residual = data["I"] - state["I"]
        local_std = self._local_std(residual)
        X = np.column_stack((np.ones(self.model.t_steps), state["O"]))
        sigma = self.model._least_squares_qp(
            X, local_std,
            A_ub=-X, b_ub=-np.full(self.model.t_steps, self.model._FIT_EPS),
            lb=np.array([self.model._FIT_EPS, 0.0]),
            ub=np.array([np.inf, np.inf]), ridge=self.model._FIT_RIDGE
        )

        updated = dict(result)
        updated.update({
            "sigma_0": float(sigma[0]), "sigma_O": float(sigma[1]),
            "residual": residual, "local_std": local_std,
        })
        return updated

    @staticmethod
    def _refined_interval(values, best, global_lo, global_hi):
        values = np.unique(np.asarray(values, dtype=float))
        width = 0.1 * max(1.0, global_hi - global_lo) if values.size <= 1 else float(np.max(np.diff(values)))
        lo, hi = max(global_lo, best - width), min(global_hi, best + width)
        if hi <= lo: hi = min(global_hi, lo + max(width, 1e-8))
        return lo, hi

    def _search_branch(self, branch, data, initial_conditions, grid_points, refinements, c_v_range, d_v_range, ratio_eps, fit_initial_state, r0_range, verbose, progress_every):
        c_bounds, d_bounds = tuple(map(float, c_v_range)), tuple(map(float, d_v_range))
        rho_global = {"under": (ratio_eps, 1.0 - ratio_eps), "critical": (1.0, 1.0)}[branch]
        c_local, d_local, rho_local = c_bounds, d_bounds, rho_global
        best, history, branch_start = None, [], perf_counter()

        for level in range(refinements):
            c_grid, d_grid = np.linspace(*c_local, grid_points), np.linspace(*d_local, grid_points)
            rho_grid = np.array([1.0]) if branch == "critical" else np.linspace(*rho_local, grid_points)
            level_best, tested, pruned, qp_times = None, 0, 0, []
            level_start = perf_counter()
            if verbose: print(f"[brute] branch={branch} refinement={level + 1}/{refinements} candidates={len(c_grid) * len(d_grid) * len(rho_grid)}")

            for d_v in d_grid:
                if d_v >= -self.model._FIT_EPS: continue
                for c_v in c_grid:
                    beta_v = c_v - d_v
                    if beta_v < -1e-10 or beta_v > 1.0 + 1e-10:
                        pruned += len(rho_grid)
                        continue

                    for rho in rho_grid:
                        kappa_r = 4.0 * float(rho) * abs(float(d_v))
                        if kappa_r <= 0.0 or not self._transition_stable(self.model.dt, d_v, kappa_r):
                            pruned += 1
                            continue

                        tested += 1
                        candidate_start = perf_counter()
                        try: candidate = self._solve_inner(data, initial_conditions, c_v, d_v, kappa_r, fit_initial_state, r0_range)
                        except (ValueError, RuntimeError, np.linalg.LinAlgError):
                            candidate_time = perf_counter() - candidate_start
                            if verbose and tested % progress_every == 0:
                                print(f"[brute] branch={branch} refinement={level + 1} tested={tested} pruned={pruned} elapsed={perf_counter() - level_start:.2f}s last_candidate={candidate_time:.4f}s status=failed")
                            continue

                        qp_times.append(float(candidate["qp_solve_time"]))
                        if verbose and (tested == 1 or tested % progress_every == 0):
                            print(f"[brute] branch={branch} refinement={level + 1} tested={tested} pruned={pruned} elapsed={perf_counter() - level_start:.2f}s last_qp={qp_times[-1]:.4f}s mean_qp={np.mean(qp_times):.4f}s")

                        score = float(candidate["recovered_model_loss"])
                        if level_best is None or score < level_best["score"]:
                            level_best = {
                                "score": score, "c_v": float(c_v), "d_v": float(d_v),
                                "rho": float(rho), "kappa_R": float(kappa_r),
                                "alpha": candidate["alpha"].copy(), "beta": candidate["beta"].copy(),
                                "result": candidate,
                            }

            level_elapsed = perf_counter() - level_start
            mean_qp = float(np.mean(qp_times)) if qp_times else np.nan
            history.append({
                "level": level, "tested": tested, "pruned": pruned,
                "best_loss": np.inf if level_best is None else level_best["score"],
                "elapsed": level_elapsed, "mean_qp_solve_time": mean_qp,
                "qp_solves": len(qp_times),
            })
            if verbose: print(f"[brute] branch={branch} refinement={level + 1} complete tested={tested} pruned={pruned} qp_solves={len(qp_times)} elapsed={level_elapsed:.2f}s mean_qp={mean_qp:.4f}s")
            if level_best is None: break

            best = level_best
            c_local = self._refined_interval(c_grid, best["c_v"], *c_bounds)
            d_local = self._refined_interval(d_grid, best["d_v"], *d_bounds)
            if branch != "critical": rho_local = self._refined_interval(rho_grid, best["rho"], *rho_global)

        if best is not None: best.update({"history": history, "branch": branch, "elapsed": perf_counter() - branch_start})
        return best

    def _search_all(self, data, initial_conditions, grid_points, refinements, c_v_range, d_v_range, ratio_eps, fit_initial_state, r0_range, verbose, progress_every):
        branch_results = {
            branch: self._search_branch(branch, data, initial_conditions, grid_points, refinements, c_v_range, d_v_range, ratio_eps, fit_initial_state, r0_range, verbose, progress_every)
            for branch in ("under", "critical")
        }
        valid = [item for item in branch_results.values() if item is not None]
        if not valid: raise RuntimeError("WM3 brute-force parameter search found no structurally admissible underdamped or critically damped grid candidate.")
        return min(valid, key=lambda item: item["score"]), branch_results

    def fit(self, input_data, initial_conditions, *, grid_points=4, refinements=2,
            c_v_range=(-1.0, 1.0), d_v_range=(-1.0, -1e-4), ratio_eps=1e-3,
            fit_initial_state=True, r0_range=(-1.0, 1.0), fit_diffusion=True,
            verbose=True, progress_every=25, alpha_points=None, alpha_o_range=None):
        data = self._prepare_data(input_data)
        initial_conditions = WM3Validation.initial_condition(initial_conditions)
        grid_points, refinements, progress_every = int(grid_points), int(refinements), int(progress_every)
        r0_range = tuple(map(float, r0_range))

        if grid_points < 2 or refinements < 1: raise ValueError("grid_points >= 2 and refinements >= 1 are required.")
        if progress_every < 1: raise ValueError("progress_every must be at least 1.")
        if not c_v_range[0] < c_v_range[1]: raise ValueError("c_v_range must be increasing.")
        if not d_v_range[0] < d_v_range[1] < 0.0: raise ValueError("d_v_range must be increasing and strictly negative.")
        if not 0.0 < ratio_eps < 0.5: raise ValueError("ratio_eps must lie in (0, 0.5).")
        if len(r0_range) != 2 or not r0_range[0] < r0_range[1]: raise ValueError("r0_range must contain increasing lower/upper bounds.")

        fit_start = perf_counter()
        best, branches = self._search_all(
            data, initial_conditions, grid_points, refinements, c_v_range, d_v_range,
            ratio_eps, bool(fit_initial_state), r0_range, verbose, progress_every
        )

        result = dict(best["result"])
        result.update({"alpha": best["alpha"].copy(), "beta": best["beta"].copy()})
        if fit_diffusion: result = self._fit_uncertainty_params(result, data)
        else: result.update({"sigma_0": self.model._FIT_EPS, "sigma_O": 0.0})

        result.update({
            "fit_method": "brute", "brute_branch": best["branch"],
            "brute_score": float(best["score"]), "brute_grid_points": grid_points,
            "brute_refinements": refinements, "brute_search_history": branches,
            "brute_fit_time": perf_counter() - fit_start,
            "fit_initial_state": bool(fit_initial_state),
            "deterministic_loss": float(result["recovered_model_loss"]),
        })
        return result
