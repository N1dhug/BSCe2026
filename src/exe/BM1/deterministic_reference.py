"""Run the deterministic BM1 F1 reference experiment.

The analytical benchmark and numerical checks live in
``src.verification.BM1``. This executable configures inputs, calls the model
solvers, reports the results, and saves the plot.

Run this module from the repository root:

    python -m src.exe.BM1.deterministic_reference
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.tool.InputBuilder import InputBuilder
from src.model.BM1 import BM1
from src.verification.BM1 import BM1DeterministicBenchmark


# Configure the one-day price-step benchmark.
BENCHMARK = BM1DeterministicBenchmark(
    time_interval=(0.0, 1.0),
    price_step_time=0.5,
    price_before=0.75,
    price_after=0.25,
    initial_state=0.5,
    baseline=0.3,
    capacity=1.0,
    k=2.0,
    flexible_share=1.0,
)

STEPS_PER_DAY = (24, 48, 96, 192)
IVP_WARNING_TOL = 1.0e-4


def build_inputs():
    """Build the constant baseline and right-continuous price step."""
    inputs = InputBuilder()
    start, end = BENCHMARK.time_interval
    baseline = inputs.energy(
        (
            np.array([start, end], dtype=float),
            np.array([BENCHMARK.baseline, BENCHMARK.baseline], dtype=float),
        )
    )
    price = inputs.price(
        (
            np.array(
                [start, BENCHMARK.price_step_time, end], dtype=float
            ),
            np.array(
                [
                    BENCHMARK.price_before,
                    BENCHMARK.price_after,
                    BENCHMARK.price_after,
                ],
                dtype=float,
            ),
        )
    )
    return baseline, price


def model_parameters(price, flexible_share=None):
    """Build BM1's parameter sequence for the configured benchmark."""
    xi = (
        BENCHMARK.flexible_share
        if flexible_share is None
        else flexible_share
    )
    return [
        price,
        BENCHMARK.state_response,
        BENCHMARK.price_response,
        BENCHMARK.capacity,
        BENCHMARK.k,
        xi,
    ]


def run_euler(steps_per_day, baseline, price):
    """Call Euler-Maruyama as uncorrected deterministic Euler."""
    model = BM1(
        baseline,
        baseline,
        BENCHMARK.time_interval,
        steps_per_day + 1,
    )
    result = model.euler_maruyama_solver(
        [BENCHMARK.initial_state],
        model_parameters(price),
        sigma_x=0.0,
        n_paths=1,
        seed=0,
        boundary="none",
    )
    t, state, price_values, baseline_values, _, _, delta, demand = result
    return {
        "steps_per_day": steps_per_day,
        "dt": 1.0 / steps_per_day,
        "t": t,
        "state": state,
        "price": price_values,
        "baseline": baseline_values,
        "delta": delta,
        "demand": demand,
        "error": float(np.max(np.abs(state - BENCHMARK.exact_state(t)))),
        "energy_error": float(
            np.max(
                np.abs(
                    BENCHMARK.energy_residual(
                        t, state, baseline_values, demand
                    )
                )
            )
        ),
    }


def constant_price_run(
    price,
    initial_state=None,
    flexible_share=None,
    steps_per_day=192,
):
    """Call Euler for a constant-price sanity-check case."""
    x0 = BENCHMARK.initial_state if initial_state is None else initial_state
    inputs = InputBuilder()
    start, end = BENCHMARK.time_interval
    baseline = inputs.energy(
        (
            np.array([start, end], dtype=float),
            np.array([BENCHMARK.baseline, BENCHMARK.baseline], dtype=float),
        )
    )
    price_signal = inputs.price(
        (
            np.array([start, end], dtype=float),
            np.array([price, price], dtype=float),
        )
    )
    model = BM1(
        baseline,
        baseline,
        BENCHMARK.time_interval,
        steps_per_day + 1,
    )
    return model.euler_maruyama_solver(
        [x0],
        model_parameters(price_signal, flexible_share),
        sigma_x=0.0,
        n_paths=1,
        seed=0,
        boundary="none",
    )


def boundary_drift_values():
    """Call BM1 drift at both state boundaries across the price range."""
    prices = np.linspace(0.0, 1.0, 101)
    lower_drift = []
    upper_drift = []
    inputs = InputBuilder()
    start, end = BENCHMARK.time_interval
    baseline = inputs.energy(
        (
            np.array([start, end], dtype=float),
            np.array([BENCHMARK.baseline, BENCHMARK.baseline], dtype=float),
        )
    )

    for price in prices:
        price_signal = inputs.price(
            (
                np.array([start, end], dtype=float),
                np.array([price, price], dtype=float),
            )
        )
        model = BM1(baseline, baseline, BENCHMARK.time_interval, 2)
        parameters = model_parameters(price_signal)
        lower_drift.append(
            float(model._drift(start, np.array([0.0]), parameters)[0])
        )
        upper_drift.append(
            float(model._drift(start, np.array([1.0]), parameters)[0])
        )

    return np.asarray(lower_drift), np.asarray(upper_drift)


def run_sanity_checks(euler_results):
    """Produce all solver results and pass them to the verification layer."""
    equilibrium = constant_price_run(0.5)
    no_flexibility = constant_price_run(0.0, flexible_share=0.0)
    low_price = constant_price_run(BENCHMARK.price_after)
    high_price = constant_price_run(BENCHMARK.price_before)
    lower_boundary_drift, upper_boundary_drift = boundary_drift_values()

    return BENCHMARK.sanity_checks(
        euler_results,
        equilibrium,
        no_flexibility,
        low_price,
        high_price,
        lower_boundary_drift,
        upper_boundary_drift,
    )


def save_figure(ivp_result, finest_euler, baseline, price):
    """Save the benchmark state, inputs, equilibria, and demand plot."""
    t_ivp, state_ivp = ivp_result[0], ivp_result[1]
    plot_t = np.linspace(*BENCHMARK.time_interval, 1001)
    plot_price = np.asarray(price(plot_t), dtype=float)
    plot_equilibrium = 1.0 - plot_price
    plot_exact_state = BENCHMARK.exact_state(plot_t)

    model = BM1(
        baseline,
        baseline,
        BENCHMARK.time_interval,
        len(plot_t),
    )
    _, plot_baseline, _, _, _, plot_demand = model.eval(
        plot_t,
        plot_exact_state,
        price,
        BENCHMARK.state_response,
        BENCHMARK.price_response,
        BENCHMARK.k,
        BENCHMARK.flexible_share,
    )

    hours = 24.0 * plot_t
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(
        hours, plot_exact_state, linewidth=2.5, label="Exact X(t)"
    )
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
    axes[2].plot(
        hours, plot_baseline, linestyle="--", label="Baseline B(t)"
    )
    axes[2].set_xlabel("Time [hours]")
    axes[2].set_ylabel("Normalized power")
    axes[2].grid(True)
    axes[2].legend(loc="best")

    step_hour = 24.0 * BENCHMARK.price_step_time
    for axis in axes:
        axis.axvline(step_hour, color="black", alpha=0.4, linestyle="--")

    fig.tight_layout()
    graph_dir = (
        Path(__file__).resolve().parents[3]
        / "files"
        / "output"
        / "graphs"
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "BM1_deterministic_reference.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return graph_path


def main():
    """Run and report the deterministic F1 reference experiment."""
    baseline, price = build_inputs()

    # Use the finest common output grid for the existing solve_ivp interface.
    model = BM1(
        baseline,
        baseline,
        BENCHMARK.time_interval,
        STEPS_PER_DAY[-1] + 1,
    )
    ivp_result = model.deterministic_solver(
        [BENCHMARK.initial_state], model_parameters(price)
    )
    ivp_t, ivp_state = ivp_result[0], ivp_result[1]
    ivp_difference = np.abs(ivp_state - BENCHMARK.exact_state(ivp_t))
    ivp_error = float(np.max(ivp_difference))
    ivp_error_time = float(ivp_t[np.argmax(ivp_difference)])

    euler_results = [
        run_euler(steps, baseline, price) for steps in STEPS_PER_DAY
    ]
    orders = BENCHMARK.observed_orders(euler_results)
    checks = run_sanity_checks(euler_results)
    graph_path = save_figure(
        ivp_result, euler_results[-1], baseline, price
    )

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
    print(
        f"state range           = [{np.min(finest['state']):.6f}, "
        f"{np.max(finest['state']):.6f}]"
    )
    print(
        f"demand range          = [{np.min(finest['demand']):.6f}, "
        f"{np.max(finest['demand']):.6f}]"
    )

    print()
    print("Sanity checks")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    print()
    print(f"Saved graph to {graph_path}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print()
        print(
            "One or more sanity checks failed; inspect these before "
            "changing BM1:"
        )
        for name in failed:
            print(f"- {name}")


if __name__ == "__main__":
    main()
