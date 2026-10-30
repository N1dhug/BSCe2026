from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.run_model.wm_5.single_step_input import build


ROOT = Path(__file__).resolve().parents[4]


def plot_single_step(example, output):
    state = example["state"]
    F, R, U, t = (np.asarray(state[key], dtype=float) for key in ("F", "R", "U", "t"))
    effects = example["contributions"]
    figures = []
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].step(t, U, where="post", label="Normalized price U")
    axes[0].set_ylabel("Price"); axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(t, F, label="Flexibility F")
    axes[1].plot(t, R, label="Rate R = D - B")
    axes[1].axhline(1.0, ls=":", alpha=0.5); axes[1].axhline(-1.0, ls=":", alpha=0.5)
    axes[1].set_ylabel("States"); axes[1].legend(); axes[1].grid(alpha=0.25)
    for col, label in enumerate(("Restoring θ_F F", "Damping θ_R g(R)", "Price θ_U h(U)")):
        axes[2].plot(t, effects[:, col], label=label)
    axes[2].plot(t, state["R_dot"], "--", label="Total dR/dt")
    axes[2].set_ylabel("Rate contributions"); axes[2].set_xlabel("Time")
    axes[2].legend(); axes[2].grid(alpha=0.25)
    for ax in axes: ax.axvline(example["step_at"], color="gray", ls=":", alpha=0.5)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'{example["model_name"]}_single_price_step.png'
    fig.savefig(path, dpi=200); plt.close(fig); figures.append(path)

    f, g, h = example["responses"]
    x = np.linspace(-1.0, 1.0, 501)
    fig, ax = plt.subplots(figsize=(9, 5))
    for func, label in ((f, "f(F) = F"), (g, "g(R)"), (h, "h(U)")):
        ax.plot(x, [func(v) for v in x], label=label)
    ax.set_xlabel("Normalized input"); ax.set_ylabel("Normalized response")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    path = output / f'{example["model_name"]}_response_functions.png'
    fig.savefig(path, dpi=200); plt.close(fig); figures.append(path)
    return figures


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
