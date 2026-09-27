"""Coupled Euler-Maruyama convergence study for F2-B.

Two experiments are reported:

1. An exact logistic-diffusion benchmark with the same diffusion structure
   b(x) = sigma*x*(1-x). This directly measures strong terminal error.
2. A BM1 self-convergence study. BM1 has no closed-form solution, so coupled
   successive levels h, h/2, h/4, h/8 are compared on the same Brownian paths.

Run from the repository root:

    python -B -m src.exe.run_model.bm_1.em_convergence
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.diagnostics.stochastic import (
    loglog_convergence_order,
    rms_strong_error,
    successive_convergence_orders,
)
from src.method.euler_maruyama import euler_maruyama
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder


T = (0.0, 1.0)
X0 = 0.5
SIGMA_X = 0.5
N_PATHS = 10_000
EXACT_SEED = 2026
BM1_SEED = 2027

# Exact benchmark: 1 h, 30 min, 15 min and 7.5 min.
EXACT_STEPS = np.array([24, 48, 96, 192], dtype=int)

# BM1 self-convergence: 15, 7.5, 3.75 and 1.875 min.
BM1_STEPS = np.array([96, 192, 384, 768], dtype=int)


def _logistic(values):
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _coarsened_increments(method, fine_increments, fine_steps, target_steps):
    factor = int(fine_steps // target_steps)
    if fine_steps % target_steps != 0:
        raise ValueError("target_steps must divide the finest step count.")
    if factor == 1:
        return fine_increments
    return method.coarsen_brownian_increments(fine_increments, factor=factor)


def exact_logistic_benchmark(method):
    """Return coupled EM errors against an exact logistic-diffusion solution."""
    finest_steps = int(EXACT_STEPS[-1])
    finest_t = np.linspace(T[0], T[1], finest_steps + 1)
    fine_dW = method.brownian_increments(
        finest_t,
        n_paths=N_PATHS,
        seed=EXACT_SEED,
    )

    # If Y_t = logit(x0) + sigma W_t and X_t = sigmoid(Y_t), Ito's formula gives
    # dX = 0.5*sigma^2*x*(1-x)*(1-2x) dt + sigma*x*(1-x) dW.
    logit_x0 = np.log(X0 / (1.0 - X0))
    exact_terminal = _logistic(logit_x0 + SIGMA_X * np.sum(fine_dW, axis=1))

    def drift(time, state):
        del time
        state = np.asarray(state, dtype=float)
        return (
            0.5
            * SIGMA_X**2
            * state
            * (1.0 - state)
            * (1.0 - 2.0 * state)
        )

    def diffusion(time, state):
        del time
        state = np.asarray(state, dtype=float)
        return SIGMA_X * state * (1.0 - state)

    errors = []
    diagnostics = []

    for n_steps in EXACT_STEPS:
        t = np.linspace(T[0], T[1], int(n_steps) + 1)
        dW = _coarsened_increments(method, fine_dW, finest_steps, int(n_steps))
        paths, level_diagnostics = method.solve_paths(
            t,
            X0,
            drift,
            diffusion,
            n_paths=N_PATHS,
            boundary="none",
            brownian_increments=dW,
            return_diagnostics=True,
        )
        errors.append(rms_strong_error(paths[:, -1], exact_terminal))
        diagnostics.append(level_diagnostics)

    step_sizes = (T[1] - T[0]) / EXACT_STEPS.astype(float)
    errors = np.asarray(errors, dtype=float)
    local_orders = successive_convergence_orders(errors)
    fitted_order = loglog_convergence_order(step_sizes, errors)

    return step_sizes, errors, local_orders, fitted_order, diagnostics


def bm1_self_convergence(method):
    """Return coupled successive-level terminal errors for the BM1 SDE."""
    inputs = input_builder()
    B = inputs.energy((np.array(T), np.array([0.3, 0.3])))
    U = inputs.price_intervals((np.array(T), np.array([0.25])))

    def linear_response(x):
        return 1.0 - 2.0 * np.asarray(x, dtype=float)

    parameters = [U, linear_response, linear_response, 1.0, 2.0, 1.0]
    model = bm_1(B, B, T, 2)

    def drift(time, state):
        return model._drift(time, state, parameters)

    def diffusion(time, state):
        del time
        return model.diffusion_coefficient(state, SIGMA_X)

    finest_steps = int(BM1_STEPS[-1])
    finest_t = np.linspace(T[0], T[1], finest_steps + 1)
    fine_dW = method.brownian_increments(
        finest_t,
        n_paths=N_PATHS,
        seed=BM1_SEED,
    )

    terminal_states = []
    diagnostics = []

    for n_steps in BM1_STEPS:
        t = np.linspace(T[0], T[1], int(n_steps) + 1)
        dW = _coarsened_increments(method, fine_dW, finest_steps, int(n_steps))
        paths, level_diagnostics = method.solve_paths(
            t,
            X0,
            drift,
            diffusion,
            n_paths=N_PATHS,
            boundary="none",
            brownian_increments=dW,
            return_diagnostics=True,
        )
        terminal_states.append(paths[:, -1].copy())
        diagnostics.append(level_diagnostics)

    self_errors = np.asarray(
        [
            rms_strong_error(terminal_states[i], terminal_states[i + 1])
            for i in range(len(terminal_states) - 1)
        ],
        dtype=float,
    )
    coarse_step_sizes = (T[1] - T[0]) / BM1_STEPS[:-1].astype(float)
    local_orders = successive_convergence_orders(self_errors)
    fitted_order = loglog_convergence_order(coarse_step_sizes, self_errors)

    return coarse_step_sizes, self_errors, local_orders, fitted_order, diagnostics


def _require_no_boundary_violations(name, diagnostics):
    total = sum(item["total_violations"] for item in diagnostics)
    if total:
        raise RuntimeError(
            f"{name} produced {total} raw boundary violations. "
            "The convergence comparison must not hide these with reflection or clipping."
        )


def _print_exact_results(step_sizes, errors, local_orders, fitted_order):
    print("Exact logistic-diffusion benchmark")
    print("----------------------------------")
    print("step [min]    RMS strong error    local order")
    for index, (step, error) in enumerate(zip(step_sizes, errors)):
        order = "-" if index == 0 else f"{local_orders[index - 1]:.4f}"
        print(f"{24.0 * 60.0 * step:10.4f}    {error:16.8e}    {order:>11}")
    print(f"Log-log fitted strong order: {fitted_order:.4f}")
    print("Expected Euler-Maruyama strong order: 0.5")
    print()


def _print_bm1_results(step_sizes, errors, local_orders, fitted_order):
    print("BM1 coupled self-convergence")
    print("----------------------------")
    print("coarse step [min]    RMS(X^h - X^(h/2))    local order")
    for index, (step, error) in enumerate(zip(step_sizes, errors)):
        order = "-" if index == 0 else f"{local_orders[index - 1]:.4f}"
        print(f"{24.0 * 60.0 * step:17.4f}    {error:20.8e}    {order:>11}")
    print(f"Log-log fitted self-convergence order: {fitted_order:.4f}")
    print("This is a self-convergence estimate, not an error against an exact BM1 solution.")
    print()


def _plot_results(exact_step_sizes, exact_errors, bm_step_sizes, bm_errors):
    fig, (ax_exact, ax_bm1) = plt.subplots(nrows=2, figsize=(9, 9))

    exact_minutes = 24.0 * 60.0 * exact_step_sizes
    exact_guide = exact_errors[0] * np.sqrt(exact_step_sizes / exact_step_sizes[0])
    ax_exact.loglog(exact_minutes, exact_errors, "o-", label="EM RMS strong error")
    ax_exact.loglog(exact_minutes, exact_guide, "--", label=r"$C h^{1/2}$")
    ax_exact.set_xlabel("Step size [min]")
    ax_exact.set_ylabel("Terminal RMS error")
    ax_exact.set_title("Exact logistic-diffusion benchmark")
    ax_exact.grid(True)
    ax_exact.legend(loc="best")

    bm_minutes = 24.0 * 60.0 * bm_step_sizes
    bm_guide = bm_errors[0] * np.sqrt(bm_step_sizes / bm_step_sizes[0])
    ax_bm1.loglog(bm_minutes, bm_errors, "o-", label=r"RMS$(X_T^h-X_T^{h/2})$")
    ax_bm1.loglog(bm_minutes, bm_guide, "--", label=r"$C h^{1/2}$")
    ax_bm1.set_xlabel("Coarse step size [min]")
    ax_bm1.set_ylabel("Coupled terminal difference")
    ax_bm1.set_title("BM1 coupled self-convergence")
    ax_bm1.grid(True)
    ax_bm1.legend(loc="best")

    fig.tight_layout()
    graph_dir = (
        Path(__file__).resolve().parents[4]
        / "files"
        / "output"
        / "graphs"
        / "bm_1"
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "bm_1_em_convergence.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return graph_path


def main():
    """Run the reference experiment and save its diagnostic figure."""
    method = euler_maruyama()

    (
        exact_step_sizes,
        exact_errors,
        exact_local_orders,
        exact_fitted_order,
        exact_diagnostics,
    ) = exact_logistic_benchmark(method)

    (
        bm_step_sizes,
        bm_errors,
        bm_local_orders,
        bm_fitted_order,
        bm_diagnostics,
    ) = bm1_self_convergence(method)

    _require_no_boundary_violations("Exact benchmark", exact_diagnostics)
    _require_no_boundary_violations("BM1 self-convergence", bm_diagnostics)

    if not 0.35 <= exact_fitted_order <= 0.70:
        raise RuntimeError(
            "The exact benchmark did not exhibit the expected EM strong-order regime."
        )

    if not np.all(np.diff(bm_errors) < 0.0):
        raise RuntimeError("BM1 successive coupled errors did not decrease under refinement.")

    _print_exact_results(
        exact_step_sizes,
        exact_errors,
        exact_local_orders,
        exact_fitted_order,
    )
    _print_bm1_results(
        bm_step_sizes,
        bm_errors,
        bm_local_orders,
        bm_fitted_order,
    )

    graph_path = _plot_results(
        exact_step_sizes,
        exact_errors,
        bm_step_sizes,
        bm_errors,
    )
    print(f"Saved graph to {graph_path}")


if __name__ == "__main__":
    main()
