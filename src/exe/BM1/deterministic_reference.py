"""Verify the deterministic BM1 implementation against an exact benchmark.

This is the F1 reference experiment.  The benchmark uses

    f(x) = 1 - 2x,        g(u) = 1 - 2u,

a constant baseline, and a price step aligned with every Euler grid.  Under
these assumptions BM1 has an exact, intervalwise solution.  The script
compares that solution with the existing ``solve_ivp`` interface and with
Euler-Maruyama at ``sigma_x=0`` and ``boundary="none"``.

Run this module from the repository root:

    python -m src.exe.BM1.deterministic_reference

The script is diagnostic: warnings describe behavior in the current model
implementation and do not modify BM1 or hide errors with boundary handling.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.tool.InputBuilder import InputBuilder
from src.model.BM1 import BM1


# Benchmark parameters.  Time is measured in days.
T = (0.0, 1.0)
PRICE_STEP_TIME = 0.5
PRICE_BEFORE = 0.75
PRICE_AFTER = 0.25
X0 = 0.5
BASELINE = 0.3
C = 1.0
K = 2.0
XI = 1.0

STEPS_PER_DAY = (24, 48, 96, 192)
IVP_WARNING_TOL = 1.0e-4
ENERGY_TOL = 1.0e-12
STATE_TOL = 1.0e-12


def linear_state_response(x):
    """Benchmark state response f(x) = 1 - 2x."""
    return 1.0 - 2.0 * np.asarray(x, dtype=float)


def linear_price_response(u):
    """Benchmark price response g(u) = 1 - 2u."""
    return 1.0 - 2.0 * np.asarray(u, dtype=float)


def build_inputs():
    """Build the constant baseline and right-continuous price step."""
    inputs = InputBuilder()
    B = inputs.energy(
        (np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float))
    )
    U = inputs.price(
        (
            np.array([T[0], PRICE_STEP_TIME, T[1]], dtype=float),
            np.array([PRICE_BEFORE, PRICE_AFTER, PRICE_AFTER], dtype=float),
        )
    )
    return B, U


def exact_constant_price_state(t, initial_state, initial_time, price):
    """Return the exact BM1 state while price and baseline are constant."""
    times = np.asarray(t, dtype=float)
    equilibrium = 1.0 - price

    if initial_state < equilibrium:
        demand_scale = 1.0 - BASELINE
    else:
        demand_scale = BASELINE

    decay_rate = K * XI * demand_scale / C
    initial_offset = K * (initial_state - equilibrium)
    elapsed = times - initial_time

    return equilibrium + np.arcsinh(
        np.sinh(initial_offset) * np.exp(-decay_rate * elapsed)
    ) / K


def exact_state(t):
    """Return the exact state for the two price intervals."""
    times = np.asarray(t, dtype=float)
    state = np.empty_like(times)

    before_step = times <= PRICE_STEP_TIME
    state[before_step] = exact_constant_price_state(
        times[before_step], X0, T[0], PRICE_BEFORE
    )

    state_at_step = float(
        exact_constant_price_state(
            np.array([PRICE_STEP_TIME]), X0, T[0], PRICE_BEFORE
        )[0]
    )
    state[~before_step] = exact_constant_price_state(
        times[~before_step], state_at_step, PRICE_STEP_TIME, PRICE_AFTER
    )
    return state


def energy_residual(t, state, baseline, demand):
    """Return the cumulative Euler energy-balance residual."""
    increments = np.diff(t) * (demand[:-1] - baseline[:-1])
    accumulated_energy = np.concatenate(([0.0], np.cumsum(increments)))
    return C * (state - state[0]) - accumulated_energy


def run_euler(steps_per_day, B, U):
    """Run the existing Euler-Maruyama interface as uncorrected Euler."""
    model = BM1(B, B, T, steps_per_day + 1)
    parameters = [U, linear_state_response, linear_price_response, C, K, XI]
    result = model.euler_maruyama_solver(
        [X0],
        parameters,
        sigma_x=0.0,
        n_paths=1,
        seed=0,
        boundary="none",
    )
    t, state, price, baseline, _, _, delta, demand = result
    return {
        "steps_per_day": steps_per_day,
        "dt": 1.0 / steps_per_day,
        "t": t,
        "state": state,
        "price": price,
        "baseline": baseline,
        "delta": delta,
        "demand": demand,
        "error": float(np.max(np.abs(state - exact_state(t)))),
        "energy_error": float(
            np.max(np.abs(energy_residual(t, state, baseline, demand)))
        ),
    }


def observed_orders(euler_results):
    """Compute the observed order between consecutive halved time steps."""
    orders = [np.nan]
    for coarse, fine in zip(euler_results[:-1], euler_results[1:]):
        orders.append(np.log(coarse["error"] / fine["error"]) / np.log(2.0))
    return orders


def constant_price_run(price, initial_state=X0, xi=XI, steps_per_day=192):
    """Run a small deterministic Euler case used by the sanity checks."""
    inputs = InputBuilder()
    B = inputs.energy(
        (np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float))
    )
    U = inputs.price(
        (np.array(T, dtype=float), np.array([price, price], dtype=float))
    )
    model = BM1(B, B, T, steps_per_day + 1)
    parameters = [U, linear_state_response, linear_price_response, C, K, xi]
    return model.euler_maruyama_solver(
        [initial_state],
        parameters,
        sigma_x=0.0,
        n_paths=1,
        seed=0,
        boundary="none",
    )


def boundary_drift_checks():
    """Check that the continuous benchmark drift points into [0, 1]."""
    prices = np.linspace(0.0, 1.0, 101)
    lower_drift = []
    upper_drift = []

    inputs = InputBuilder()
    B = inputs.energy(
        (np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float))
    )

    for price in prices:
        U = inputs.price(
            (np.array(T, dtype=float), np.array([price, price], dtype=float))
        )
        model = BM1(B, B, T, 2)
        parameters = [U, linear_state_response, linear_price_response, C, K, XI]
        lower_drift.append(float(model._drift(T[0], np.array([0.0]), parameters)[0]))
        upper_drift.append(float(model._drift(T[0], np.array([1.0]), parameters)[0]))

    return (
        min(lower_drift) >= -STATE_TOL
        and max(upper_drift) <= STATE_TOL
    )


def sanity_checks(euler_results):
    """Run simple model checks that are independent of solver accuracy."""
    equilibrium = constant_price_run(0.5)
    no_flexibility = constant_price_run(0.0, xi=0.0)
    low_price = constant_price_run(0.25)
    high_price = constant_price_run(0.75)

    equilibrium_state = equilibrium[1]
    no_flexibility_state = no_flexibility[1]
    low_price_state = low_price[1]
    high_price_state = high_price[1]

    all_states = np.concatenate([result["state"] for result in euler_results])
    max_energy_error = max(result["energy_error"] for result in euler_results)

    return {
        "equilibrium remains fixed": np.max(np.abs(equilibrium_state - X0))
        <= STATE_TOL,
        "Xi = 0 gives D = B and fixed X": (
            np.max(np.abs(no_flexibility_state - X0)) <= STATE_TOL
            and np.max(np.abs(no_flexibility[7] - no_flexibility[3]))
            <= ENERGY_TOL
        ),
        "low price increases X toward 0.75": (
            low_price_state[-1] > X0
            and np.all(np.diff(low_price_state) >= -STATE_TOL)
            and low_price_state[-1] < 0.75
        ),
        "high price decreases X toward 0.25": (
            high_price_state[-1] < X0
            and np.all(np.diff(high_price_state) <= STATE_TOL)
            and high_price_state[-1] > 0.25
        ),
        "continuous drift is inward at both boundaries": boundary_drift_checks(),
        "Euler states stay in [0, 1] without correction": (
            np.min(all_states) >= -STATE_TOL
            and np.max(all_states) <= 1.0 + STATE_TOL
        ),
        "Euler energy identity": max_energy_error <= ENERGY_TOL,
    }


def save_figure(ivp_result, finest_euler, B, U):
    """Save the benchmark state, inputs, equilibria, and demand plot."""
    t_ivp, state_ivp = ivp_result[0], ivp_result[1]
    exact_ivp = exact_state(t_ivp)

    plot_t = np.linspace(T[0], T[1], 1001)
    plot_price = np.asarray(U(plot_t), dtype=float)
    plot_equilibrium = 1.0 - plot_price
    plot_exact_state = exact_state(plot_t)

    model = BM1(B, B, T, len(plot_t))
    _, plot_baseline, _, _, _, plot_demand = model.eval(
        plot_t,
        plot_exact_state,
        U,
        linear_state_response,
        linear_price_response,
        K,
        XI,
    )

    hours = 24.0 * plot_t
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(hours, plot_exact_state, linewidth=2.5, label="Exact X(t)")
    axes[0].plot(
        24.0 * t_ivp,
        state_ivp,
        linestyle="--",
        label="Existing solve_ivp",
    )
    axes[0].plot(
        24.0 * finest_euler["t"],
        finest_euler["state"],
        linestyle=":",
        marker=".",
        markersize=3,
        label=f"Euler, dt = {24.0 * finest_euler['dt'] * 60.0:.1f} min",
    )
    axes[0].set_ylabel("State X")
    axes[0].set_title("BM1 deterministic F1 reference")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    axes[1].step(hours, plot_price, where="post", label="Price U(t)")
    axes[1].step(
        hours,
        plot_equilibrium,
        where="post",
        label="Equilibrium X*(t) = 1 - U(t)",
    )
    axes[1].set_ylabel("Price / state")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True)
    axes[1].legend(loc="best")

    axes[2].plot(hours, plot_demand, label="Exact demand D(t)")
    axes[2].plot(hours, plot_baseline, linestyle="--", label="Baseline B(t)")
    axes[2].set_xlabel("Time [hours]")
    axes[2].set_ylabel("Normalized power")
    axes[2].grid(True)
    axes[2].legend(loc="best")

    for axis in axes:
        axis.axvline(24.0 * PRICE_STEP_TIME, color="black", alpha=0.4, linestyle="--")

    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[3] / "files" / "output" / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "BM1_deterministic_reference.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return graph_path


def main():
    """Run the full deterministic F1 reference experiment."""
    B, U = build_inputs()
    parameters = [U, linear_state_response, linear_price_response, C, K, XI]

    # Use the finest common output grid for the existing solve_ivp interface.
    model = BM1(B, B, T, STEPS_PER_DAY[-1] + 1)
    ivp_result = model.deterministic_solver([X0], parameters)
    ivp_t, ivp_state = ivp_result[0], ivp_result[1]
    ivp_difference = np.abs(ivp_state - exact_state(ivp_t))
    ivp_error = float(np.max(ivp_difference))
    ivp_error_time = float(ivp_t[np.argmax(ivp_difference)])

    euler_results = [run_euler(steps, B, U) for steps in STEPS_PER_DAY]
    orders = observed_orders(euler_results)
    checks = sanity_checks(euler_results)
    graph_path = save_figure(ivp_result, euler_results[-1], B, U)

    print("BM1 deterministic F1 verification")
    print("==================================")
    print()
    print("Exact vs existing solve_ivp")
    print(f"max |X_ivp - X_exact| = {ivp_error:.6e}")
    print(f"largest error at t      = {ivp_error_time:.6f} days")
    if ivp_error <= IVP_WARNING_TOL:
        print(f"PASS solve_ivp error <= {IVP_WARNING_TOL:.1e}")
    else:
        print(
            f"WARN solve_ivp error > {IVP_WARNING_TOL:.1e}; "
            "the price discontinuity needs controlled integration"
        )

    print()
    print("Euler convergence")
    print("dt [days]    dt [min]    max X error    observed order")
    for result, order in zip(euler_results, orders):
        order_text = "-" if np.isnan(order) else f"{order:.6f}"
        minutes = 24.0 * 60.0 * result["dt"]
        print(
            f"{result['dt']:<12.8f} {minutes:<11.3f} "
            f"{result['error']:<14.6e} {order_text}"
        )

    finest = euler_results[-1]
    print()
    print("Finest Euler grid")
    print(f"max |energy residual| = {finest['energy_error']:.6e}")
    print(f"state range           = [{np.min(finest['state']):.6f}, "
          f"{np.max(finest['state']):.6f}]")
    print(f"demand range          = [{np.min(finest['demand']):.6f}, "
          f"{np.max(finest['demand']):.6f}]")

    print()
    print("Sanity checks")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    print()
    print(f"Saved graph to {graph_path}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print()
        print("One or more sanity checks failed; inspect these before changing BM1:")
        for name in failed:
            print(f"- {name}")


if __name__ == "__main__":
    main()
