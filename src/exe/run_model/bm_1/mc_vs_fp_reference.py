"""F2-D Monte Carlo versus pure finite-volume comparison for stochastic BM1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.diagnostics.bm_1 import (
    cell_mass_cdf_sup,
    cell_mass_l1,
    cell_mass_wasserstein_1,
    empirical_cell_mass,
)
from src.exe.run_model.bm_1.f2_reference_case import (
    MAX_STEP,
    PRICE_BREAKPOINTS,
    PRICE_DROP_HOUR,
    SIGMA_X,
    T_EVAL,
    X_STEPS,
    beta_initial_cell_mass,
    build_reference_case,
    sample_beta_initial_states,
)


DEFAULT_N_PATHS = 50_000
INITIAL_SEED = 2028
BROWNIAN_SEED = 2029


def run_comparison(
    n_paths=DEFAULT_N_PATHS,
    x_steps=X_STEPS,
    max_step=MAX_STEP,
    initial_seed=INITIAL_SEED,
    brownian_seed=BROWNIAN_SEED,
):
    """Run MC and pure FV from the same Beta initial law and BM1 inputs."""
    model, parameters = build_reference_case()
    edges, _, dx = model.fokker_planc_reference.grid(x_steps)

    initial_states = sample_beta_initial_states(n_paths, initial_seed)
    integration_t, _ = model.euler_maruyama.build_time_grid(
        T_EVAL,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=max_step,
    )
    dW = model.euler_maruyama.brownian_increments(
        integration_t,
        n_paths=n_paths,
        seed=brownian_seed,
    )
    em_result = model.euler_maruyama_solver(
        initial_states,
        parameters,
        SIGMA_X,
        n_paths=n_paths,
        boundary="none",
        t_eval=T_EVAL,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=max_step,
        brownian_increments=dW,
        return_diagnostics=True,
    )
    (
        t,
        state,
        price,
        baseline,
        _,
        _,
        _,
        demand,
        em_diagnostics,
    ) = em_result

    if em_diagnostics["total_violations"]:
        raise RuntimeError(
            "F2-D Monte Carlo produced raw proposals outside [0,1]; "
            "the comparison does not apply reflection or clipping."
        )

    initial_mass = beta_initial_cell_mass(edges)
    fp_result = model.fokker_planc_reference_solver(
        initial_mass,
        parameters,
        SIGMA_X,
        t_eval=T_EVAL,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=max_step,
    )
    (
        fp_t,
        x,
        density,
        cell_mass,
        fp_mean,
        fp_variance,
        _,
        _,
        fp_expected_demand,
        fp_diagnostics,
    ) = fp_result
    np.testing.assert_array_equal(t, fp_t)

    mc_mean = np.mean(state, axis=0)
    mc_variance = np.var(state, axis=0, ddof=1)
    mc_mean_se = np.sqrt(mc_variance / n_paths)
    mc_demand = np.mean(demand, axis=0)
    mc_demand_se = np.std(demand, axis=0, ddof=1) / np.sqrt(n_paths)

    mc_cell_mass = np.empty_like(cell_mass)
    l1 = np.empty(len(t))
    cdf_sup = np.empty(len(t))
    wasserstein = np.empty(len(t))
    for index in range(len(t)):
        mc_cell_mass[index] = empirical_cell_mass(state[:, index], edges)
        l1[index] = cell_mass_l1(mc_cell_mass[index], cell_mass[index])
        cdf_sup[index] = cell_mass_cdf_sup(mc_cell_mass[index], cell_mass[index])
        wasserstein[index] = cell_mass_wasserstein_1(
            mc_cell_mass[index], cell_mass[index], dx
        )

    return {
        "t": t,
        "x": x,
        "edges": edges,
        "price": price,
        "baseline": baseline,
        "mc_state": state,
        "mc_mean": mc_mean,
        "mc_variance": mc_variance,
        "mc_mean_se": mc_mean_se,
        "mc_expected_demand": mc_demand,
        "mc_demand_se": mc_demand_se,
        "mc_cell_mass": mc_cell_mass,
        "fp_density": density,
        "fp_cell_mass": cell_mass,
        "fp_mean": fp_mean,
        "fp_variance": fp_variance,
        "fp_expected_demand": fp_expected_demand,
        "l1": l1,
        "cdf_sup": cdf_sup,
        "wasserstein_1": wasserstein,
        "em_diagnostics": em_diagnostics,
        "fp_diagnostics": fp_diagnostics,
        "dx": dx,
        "n_paths": n_paths,
    }


def main():
    result = run_comparison()

    print("F2-D Monte Carlo versus finite-volume reference")
    print("-----------------------------------------------")
    print(f"Monte Carlo paths:                {result['n_paths']}")
    print(f"Spatial cells:                    {len(result['x'])}")
    print(f"EM raw boundary violations:       {result['em_diagnostics']['total_violations']}")
    print(f"FV maximum |mass - 1|:            {result['fp_diagnostics']['maximum_abs_mass_error']:.3e}")
    print()
    print(f"Maximum |E_MC[X]-E_FV[X]|:        {np.max(np.abs(result['mc_mean'] - result['fp_mean'])):.6e}")
    print(f"Terminal |E_MC[X]-E_FV[X]|:       {abs(result['mc_mean'][-1] - result['fp_mean'][-1]):.6e}")
    print(f"Maximum |Var_MC-Var_FV|:           {np.max(np.abs(result['mc_variance'] - result['fp_variance'])):.6e}")
    print(f"Maximum |E_MC[D]-E_FV[D]|:        {np.max(np.abs(result['mc_expected_demand'] - result['fp_expected_demand'])):.6e}")
    print(f"Terminal cell-mass L1 distance:    {result['l1'][-1]:.6e}")
    print(f"Terminal CDF sup distance:         {result['cdf_sup'][-1]:.6e}")
    print(f"Terminal Wasserstein-1 distance:   {result['wasserstein_1'][-1]:.6e}")

    hours = 24.0 * result["t"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 14))
    ax_mean, ax_demand, ax_distance, ax_terminal = axes

    ax_mean.step(hours, result["price"], where="post", label="Price U(t)")
    ax_mean.plot(hours, result["mc_mean"], label="MC E[X]")
    ax_mean.plot(hours, result["fp_mean"], linestyle="--", label="FV E[X]")
    ax_mean.fill_between(
        hours,
        result["mc_mean"] - 1.96 * result["mc_mean_se"],
        result["mc_mean"] + 1.96 * result["mc_mean_se"],
        alpha=0.2,
        label="95% MC interval for E[X]",
    )
    ax_mean.axvline(PRICE_DROP_HOUR, linestyle=":")
    ax_mean.set_ylabel("Price / state")
    ax_mean.grid(True)
    ax_mean.legend(loc="best")

    ax_demand.plot(hours, result["mc_expected_demand"], label="MC E[D]")
    ax_demand.plot(hours, result["fp_expected_demand"], linestyle="--", label="FV E[D]")
    ax_demand.fill_between(
        hours,
        result["mc_expected_demand"] - 1.96 * result["mc_demand_se"],
        result["mc_expected_demand"] + 1.96 * result["mc_demand_se"],
        alpha=0.2,
        label="95% MC interval for E[D]",
    )
    ax_demand.plot(hours, result["baseline"], linestyle=":", label="B(t)")
    ax_demand.set_ylabel("Demand")
    ax_demand.grid(True)
    ax_demand.legend(loc="best")

    ax_distance.plot(hours, result["l1"], label="Cell-mass L1")
    ax_distance.plot(hours, result["cdf_sup"], label="CDF sup")
    ax_distance.plot(hours, result["wasserstein_1"], label="Wasserstein-1")
    ax_distance.set_ylabel("Distribution distance")
    ax_distance.grid(True)
    ax_distance.legend(loc="best")

    mc_terminal_density = result["mc_cell_mass"][-1] / result["dx"]
    ax_terminal.plot(result["x"], mc_terminal_density, label="MC histogram density")
    ax_terminal.plot(result["x"], result["fp_density"][-1], linestyle="--", label="FV density")
    ax_terminal.set_xlabel("State of charge x")
    ax_terminal.set_ylabel("Probability density")
    ax_terminal.grid(True)
    ax_terminal.legend(loc="best")

    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / "files" / "output" / "graphs" / "bm_1"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "bm_1_mc_vs_fp_reference.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph to {graph_path}")


if __name__ == "__main__":
    main()
