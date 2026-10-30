from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.run_model.wm_5.single_step_input import build


ROOT = Path(__file__).resolve().parents[4]


def plot_single_step(example, output):
    state = example["state"]
    F, R, U, t = (np.asarray(state[key], dtype=float) for key in ("F", "R", "U", "t"))
    demand, baseline = np.asarray(state["I"], dtype=float), np.asarray(state["O"], dtype=float)
    effects = np.asarray(example["contributions"], dtype=float)
    f, g, h = example["responses"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 13),
                             gridspec_kw={"height_ratios": (1.4, 1.1, 1.4, 1.1)})
    for ax in axes[1:3]: ax.sharex(axes[0])

    axes[0].plot(t, demand, label="Predicted demand D = B + R")
    axes[0].plot(t, baseline, ":", label="Statistical baseline B")
    axes[0].fill_between(t, baseline, demand, alpha=0.12, label="Demand deviation R")
    axes[0].set_ylabel("Demand"); axes[0].grid(alpha=0.25)
    price_axis = axes[0].twinx()
    price_axis.step(t, U, where="post", ls="--", color="0.35", label="Normalized price U")
    price_axis.set_ylim(-1.1, 1.1); price_axis.set_yticks((-1.0, 0.0, 1.0))
    price_axis.set_ylabel("Price U")
    handles, labels = axes[0].get_legend_handles_labels()
    price_handles, price_labels = price_axis.get_legend_handles_labels()
    axes[0].legend(handles + price_handles, labels + price_labels, loc="best")

    axes[1].plot(t, F, label="Flexibility F")
    axes[1].plot(t, R, label="Rate R = D - B")
    axes[1].axhline(1.0, ls=":", alpha=0.5); axes[1].axhline(-1.0, ls=":", alpha=0.5)
    axes[1].set_ylabel("States"); axes[1].legend(loc="best"); axes[1].grid(alpha=0.25)

    for col, label in enumerate(("Restoring θ_F F", "Damping θ_R g(R)", "Price θ_U h(U)")):
        axes[2].plot(t, effects[:, col], label=label)
    axes[2].plot(t, state["R_dot"], "--", label="Total dR/dt")
    axes[2].set_ylabel("Rate contributions"); axes[2].set_xlabel("Time")
    axes[2].legend(loc="best"); axes[2].grid(alpha=0.25)

    for ax in axes[:3]: ax.axvline(example["step_at"], color="gray", ls=":", alpha=0.55)
    for ax in axes[:2]: ax.tick_params(labelbottom=False)

    x = np.linspace(-1.0, 1.0, 501)
    for func, label in ((f, "f(F) = F"), (g, "g(R)"), (h, "h(U)")):
        axes[3].plot(x, [func(float(v)) for v in x], label=label)
    axes[3].set_xlim(-1.0, 1.0)
    axes[3].set_xlabel("Normalized input"); axes[3].set_ylabel("Normalized response")
    axes[3].legend(loc="best"); axes[3].grid(alpha=0.25)

    fig.suptitle(f'{example["model_name"]}: single price-step demand and flexibility response')
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'{example["model_name"]}_single_price_step.png'
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [path]


def main(default_model="wm_5"):
    parser = ArgumentParser(description="Plot a normalized single-price-step flexibility response.")
    parser.add_argument("--model", choices=("wm_4", "wm_5"), default=default_model)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--step-at", type=float, default=2.0)
    parser.add_argument("--price-before", type=float, default=0.0)
    parser.add_argument("--price-after", type=float, default=-1.0)
    parser.add_argument("--output", default=str(ROOT / "files" / "output" / "single_step"))
    args = parser.parse_args()
    example = build(args.model, duration=args.duration, dt=args.dt, step_at=args.step_at,
                    price_before=args.price_before, price_after=args.price_after)
    for path in plot_single_step(example, Path(args.output)): print(path)


if __name__ == "__main__": main()
