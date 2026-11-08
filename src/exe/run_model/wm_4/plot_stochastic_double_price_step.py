"""Plot the affine ARM stochastic response to a price decrease followed by an increase.

The second price step occurs only after the mean flexibility state has
settled at the first-step equilibrium. The mean is computed by the existing
wm_4 Euler solver. Pointwise Gaussian uncertainty is propagated by the same
discrete state transition (not estimated from Monte Carlo paths).

Noise enters dR through sigma dW and each observed demand through independent
tau*epsilon. Sample paths illustrate process diffusion only; the demand band
includes both sources of uncertainty.
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model.wm_4 import wm_4
from src.validation.model.wm_4 import wm_4_validation


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"wm_4"/"stochastic_double_price_step"/"stochastic_double_price_step.png"
Z_95 = 1.959963984540054


def check(args):
    values = (args.theta_f, args.theta_r, args.theta_u, args.sigma, args.tau,
              args.baseline, args.price_start, args.price_down, args.price_up,
              args.first_step, args.second_step, args.duration, args.steady_tolerance)
    if not np.isfinite(values).all(): raise ValueError("All parameters and step times must be finite.")
    if not 0. <= args.baseline <= 1.: raise ValueError("Baseline must be in [0, 1].")
    if args.sigma < 0. or args.tau < 0.: raise ValueError("sigma and tau must be nonnegative.")
    if not 0. <= args.first_step < args.second_step < args.duration:
        raise ValueError("Require 0 <= first_step < second_step < duration.")
    if args.steady_tolerance <= 0.: raise ValueError("steady_tolerance must be positive.")
    if args.steps_per_hour < 1: raise ValueError("steps_per_hour must be at least one.")
    if args.paths < 0: raise ValueError("paths must be nonnegative.")
    if not -1. <= args.price_start <= 1. or not -1. <= args.price_down <= 1. or not -1. <= args.price_up <= 1.:
        raise ValueError("Price levels must be in [-1, 1].")
    if not args.price_down < args.price_start or not args.price_up > args.price_down:
        raise ValueError("The first price step must decrease U, and the second must increase U.")
    r = wm_4_validation.r_coefficients([args.theta_f, args.theta_u, args.theta_r])
    natural_frequency = float(np.sqrt(-r[0]))
    damping_ratio = float(-r[2]/(2.*natural_frequency))
    if not 0. < damping_ratio < 1.:
        raise ValueError(f"Underdamped parameters require 0 < damping_ratio < 1; got {damping_ratio:.5f}.")
    return r, natural_frequency, damping_ratio


def price_function(args):
    def price(time):
        if time < args.first_step: return float(args.price_start)
        if time < args.second_step: return float(args.price_down)
        return float(args.price_up)
    return price


def simulate(args):
    r, natural_frequency, damping_ratio = check(args)
    grid_steps = args.duration*args.steps_per_hour
    if abs(grid_steps-round(grid_steps)) > 1e-8:
        raise ValueError("duration * steps_per_hour must be an integer.")
    model = wm_4((0., float(args.duration)), int(round(grid_steps))+1,
                 lambda _time: float(args.baseline))
    price = price_function(args)
    parameters = {"r":r, "U":price, "PE":price, "Time":lambda time:float(time)}
    state = model.deterministic_solver([0., 0.], parameters)
    t, dt = state["t"], model.dt
    transition = np.array([[1., dt], [dt*r[0], 1.+dt*r[2]]], dtype=float)
    second_index = int(np.searchsorted(t, args.second_step, side="left"))
    equilibrium_down = -r[1]*args.price_down/r[0]
    before_second = np.array([state["F"][second_index-1], state["R"][second_index-1]])
    if np.max(np.abs(before_second-[equilibrium_down, 0.])) > args.steady_tolerance:
        raise ValueError("The first response has not settled before the second price step. "
                         "Increase --second-step or --steady-tolerance.")

    covariance = np.zeros((len(t), 2, 2), dtype=float)
    process_cov = np.array([[0., 0.], [0., args.sigma**2*dt]], dtype=float)
    for i in range(len(t)-1):
        covariance[i+1] = transition@covariance[i]@transition.T+process_cov
    sd_f = np.sqrt(np.maximum(covariance[:, 0, 0], 0.))
    sd_r = np.sqrt(np.maximum(covariance[:, 1, 1], 0.))
    sd_d = np.sqrt(np.maximum(covariance[:, 1, 1]+args.tau**2, 0.))
    theta = np.array([r[0], r[2]], dtype=float)
    sd_drift = np.sqrt(np.maximum(np.einsum("i,tij,j->t", theta, covariance, theta), 0.))

    restoring = r[0]*state["F"]
    damping = r[2]*state["R"]
    price_contribution = r[1]*state["U"]
    drift = restoring+damping+price_contribution

    samples = np.empty((0, len(t), 2), dtype=float)
    if args.paths:
        rng = np.random.default_rng(args.seed)
        deviations = np.zeros((args.paths, len(t), 2), dtype=float)
        for i in range(len(t)-1):
            deviations[:, i+1, :] = deviations[:, i, :]@transition.T
            deviations[:, i+1, 1] += args.sigma*np.sqrt(dt)*rng.standard_normal(args.paths)
        samples = deviations+np.stack([state["F"], state["R"]], axis=1)[None, :, :]

    return {"t":t, "dt":dt, "state":state, "covariance":covariance,
            "sd_f":sd_f, "sd_r":sd_r, "sd_d":sd_d, "sd_drift":sd_drift,
            "restoring":restoring, "damping":damping,
            "price_contribution":price_contribution, "drift":drift,
            "samples":samples, "equilibrium_down":float(equilibrium_down),
            "equilibrium_up":float(-r[1]*args.price_up/r[0]),
            "settled_state":before_second, "natural_frequency":natural_frequency,
            "damping_ratio":damping_ratio}


def plot(result, args, destination):
    t, s = result["t"], result["state"]
    z = Z_95
    fig, axes = plt.subplots(3, 1, figsize=(13., 10.2), sharex=True,
                             gridspec_kw={"height_ratios":[1.12, 1., 1.]},
                             constrained_layout=True)
    demand = axes[0]
    demand.fill_between(t, s["O"], s["I"], color="tab:blue", alpha=.09,
                        label="Demand deviation R")
    demand.fill_between(t, s["I"]-z*result["sd_d"], s["I"]+z*result["sd_d"],
                        color="tab:blue", alpha=.23, linewidth=0,
                        label="95% pointwise demand interval")
    for i, path in enumerate(result["samples"]):
        demand.plot(t, args.baseline+path[:, 1], color="tab:blue",
                    linewidth=.8, alpha=.25,
                    label="Sample process realization" if i == 0 else None)
    demand.plot(t, s["I"], color="tab:blue", linewidth=1.8,
                label="Expected demand D = B + E[R]")
    demand.plot(t, s["O"], color="tab:orange", linestyle=":",
                linewidth=1.5, label="Statistical baseline B")
    demand.set_ylabel("Demand")
    price_ax = demand.twinx()
    price_ax.step(t, s["U"], where="post", color=".35", linestyle="--",
                  linewidth=1.4, label="Normalized price U")
    price_ax.set_ylabel("Price U")
    price_ax.set_ylim(-1.15, 1.15)
    price_ax.set_yticks([-1, 0, 1])
    handles, labels = demand.get_legend_handles_labels()
    price_handles, price_labels = price_ax.get_legend_handles_labels()
    demand.legend(handles+price_handles, labels+price_labels, loc="upper right",
                  fontsize=8, framealpha=.9)

    states = axes[1]
    states.fill_between(t, s["F"]-z*result["sd_f"], s["F"]+z*result["sd_f"],
                        color="tab:blue", alpha=.2, linewidth=0, label="F 95% interval")
    states.fill_between(t, s["R"]-z*result["sd_r"], s["R"]+z*result["sd_r"],
                        color="tab:orange", alpha=.19, linewidth=0, label="R 95% interval")
    states.plot(t, s["F"], color="tab:blue", linewidth=1.8, label="Expected flexibility F")
    states.plot(t, s["R"], color="tab:orange", linewidth=1.5, label="Expected rate R")
    for bound in (-1., 1.):
        states.axhline(bound, color="tab:blue", linestyle=":", alpha=.5, linewidth=1.)
    states.set_ylabel("States")
    states.legend(loc="upper right", fontsize=8, framealpha=.9)

    rates = axes[2]
    rates.plot(t, result["restoring"], color="tab:blue",
               label=r"Restoring $\theta_F F$")
    rates.plot(t, result["damping"], color="tab:orange",
               label=r"Damping $\theta_R g(R)$")
    rates.step(t, result["price_contribution"], where="post", color="tab:green",
               label=r"Price $\theta_U h(U)$")
    rates.fill_between(t, result["drift"]-z*result["sd_drift"],
                       result["drift"]+z*result["sd_drift"],
                       color="tab:red", alpha=.13, linewidth=0,
                       label="95% drift interval")
    rates.plot(t, result["drift"], color="tab:red", linestyle="--",
               linewidth=1.4, label=r"Total mean drift $d\mathbb{E}[R]/dt$")
    rates.set_ylabel("Rate contributions")
    rates.set_xlabel("Time (hours)")
    rates.legend(loc="lower right", fontsize=8, framealpha=.9)

    for ax in axes:
        ax.axvline(args.first_step, color=".55", linestyle=":", linewidth=1.2)
        ax.axvline(args.second_step, color=".55", linestyle=":", linewidth=1.2)
        ax.grid(True, alpha=.19)
        ax.set_xlim(0., args.duration)
    fig.suptitle(
        f"Affine stochastic ARM | price down then up after steady state"
        f" | damping ratio={result['damping_ratio']:.2f}"
        f" | sigma={args.sigma:g}, tau={args.tau:g}", fontsize=12)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_data(result, destination):
    s = result["state"]
    names = ("time_hours", "price", "baseline", "mean_demand", "sd_demand",
             "mean_F", "sd_F", "mean_R", "sd_R", "mean_rate_drift", "sd_rate_drift")
    columns = (result["t"], s["U"], s["O"], s["I"], result["sd_d"],
               s["F"], result["sd_f"], s["R"], result["sd_r"],
               result["drift"], result["sd_drift"])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, np.column_stack(columns), delimiter=",",
               header=",".join(names), comments="", fmt="%.12g")
    return destination


def run(args):
    result = simulate(args)
    image_path = Path(args.output) if args.output else OUTPUT
    image = plot(result, args, image_path)
    data = save_data(result, image_path.with_suffix(".csv"))
    f, r = result["settled_state"]
    print(f"[ARM stochastic step] underdamped zeta={result['damping_ratio']:.4f} "
          f"| omega_n={result['natural_frequency']:.4f}/h")
    print(f"[ARM stochastic step] U: {args.price_start:g} -> {args.price_down:g} "
          f"at {args.first_step:g} h -> {args.price_up:g} at {args.second_step:g} h")
    print(f"[ARM stochastic step] first equilibrium F*={result['equilibrium_down']:.6f}; "
          f"before second step F={f:.6f}, R={r:.6f}")
    print(f"[ARM stochastic step] second equilibrium F*={result['equilibrium_up']:.6f}; "
          f"final demand SD={result['sd_d'][-1]:.6f}")
    print(f"[ARM stochastic step] plot={image} | data={data}")
    return result


def main():
    parser = ArgumentParser(description="Plot an underdamped affine ARM stochastic double price step with exact Euler Gaussian uncertainty.")
    parser.add_argument("--theta-f", type=float, default=-.25)
    parser.add_argument("--theta-r", type=float, default=-.65)
    parser.add_argument("--theta-u", type=float, default=-.1875)
    parser.add_argument("--sigma", type=float, default=.03,
                        help="Process diffusion amplitude per sqrt(hour).")
    parser.add_argument("--tau", type=float, default=.012,
                        help="Independent observation-noise standard deviation.")
    parser.add_argument("--baseline", type=float, default=.5)
    parser.add_argument("--price-start", type=float, default=0.)
    parser.add_argument("--price-down", type=float, default=-1.)
    parser.add_argument("--price-up", type=float, default=1.)
    parser.add_argument("--first-step", type=float, default=2.)
    parser.add_argument("--second-step", type=float, default=24.,
                        help="Must follow settlement after the first step.")
    parser.add_argument("--duration", type=float, default=52.)
    parser.add_argument("--steady-tolerance", type=float, default=.005,
                        help="Max absolute F equilibrium error or |R| just before the second step.")
    parser.add_argument("--steps-per-hour", type=int, default=30)
    parser.add_argument("--paths", type=int, default=5,
                        help="Process-noise sample paths overlaid on the demand plot; set 0 to hide.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None, help="Output PNG path (CSV saved beside it).")
    return run(parser.parse_args())


if __name__ == "__main__": main()
