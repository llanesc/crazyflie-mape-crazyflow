"""
Plot curriculum/win_rate, curriculum/level, and Reward/Total reward (mean)
for selected training runs.
"""

import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "STIXGeneral"

BASE = os.path.join(os.path.dirname(__file__), "..", "results")

RUNS = [
    {
        "path": "acmpc/action_penalty/results/run_20260219203108",
        "label": "MA-AC-MPC",
        "color": "tab:blue",
    },
    {
        "path": "ffn/action_penalty/results/run_20260221175134",
        "label": "MA-AC-MLP [256×2 / 256×2]",
        "color": "tab:orange",
    },
    {
        "path": "ffn/action_penalty/results/run_20260223130705",
        "label": "MA-AC-MLP [512×2 / 512×2]",
        "color": "tab:green",
    },
    {
        "path": "ffn/action_penalty/results/run_20260225003243",
        "label": "MA-AC-MLP [512×3 / 512×2]",
        "color": "tab:red",
    },
]

METRICS = [
    {
        "tag": "curriculum/win_rate",
        "ylabel": "Evader Win Rate",
        "title": "Evader Win Rate",
        "filename": "curriculum_win_rate.pdf",
        "ymin": 0.4,
        "ymax": 1.0,
        "mean_sigma": 10,
    },
    {
        "tag": "curriculum/level",
        "ylabel": "Curriculum Level",
        "title": "Curriculum Level",
        "filename": "curriculum_level.pdf",
        "ymin": 0.0,
        "no_raw": True,
        "no_smooth": True,
        "prepend_zero": True,
    },
    {
        "tag": "Reward / Total reward (mean)",
        "ylabel": "Total Reward (mean)",
        "title": "Total Reward (Mean)",
        "filename": "total_reward_mean.pdf",
        "band": True,
    },
]

MEAN_SIGMA  = 300   # Gaussian sigma for mean line (in samples)
BAND_SIGMA  = 2     # Gaussian pre-smooth before rolling min/max (0 = raw)
EDGE_SIGMA  = 150   # Gaussian post-smooth on lo/hi edges
BAND_WINDOW = 500   # rolling window for min/max on the pre-smoothed data
OUT_DIR = os.path.join(BASE, "plots")


def load_scalar(run_path: str, tag: str):
    """Return (steps, values) arrays for a given tag from an event directory."""
    ea = EventAccumulator(run_path, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian smooth — no lag, no asymmetric artifacts."""
    if sigma == 0:
        return values.astype(float)
    return gaussian_filter1d(values.astype(float), sigma=sigma)


def plot_metric(metric: dict, runs: list, out_dir: str):
    use_band = metric.get("band", False)
    ymin = metric.get("ymin", None)

    fig, ax = plt.subplots(figsize=(7, 2.5))

    for run in runs:
        full_path = os.path.join(BASE, run["path"])
        steps, values = load_scalar(full_path, metric["tag"])
        if len(steps) == 0:
            print(f"  [warn] tag '{metric['tag']}' not found in {run['path']}")
            continue

        if metric.get("prepend_zero", False):
            steps = np.concatenate([[0], steps])
            values = np.concatenate([[0.0], values])

        steps_m = steps / 1e6

        if use_band:
            pre = smooth(values, BAND_SIGMA)
            n = len(pre) - BAND_WINDOW + 1
            idx = np.arange(BAND_WINDOW) + np.arange(n)[:, None]
            lo = smooth(pre[idx].min(axis=1), EDGE_SIGMA)
            hi = smooth(pre[idx].max(axis=1), EDGE_SIGMA)
            offset = (BAND_WINDOW - 1) // 2
            steps_band = steps_m[offset : offset + n]
            ax.fill_between(steps_band, lo, hi, color=run["color"], alpha=0.15, linewidth=0)
            ax.plot(steps_band, lo, color=run["color"], linewidth=0.5, alpha=0.4)
            ax.plot(steps_band, hi, color=run["color"], linewidth=0.5, alpha=0.4)
            ax.plot(steps_m, smooth(values, MEAN_SIGMA), color=run["color"], linewidth=1.0, linestyle="-", label=run["label"])
        else:
            if metric.get("no_smooth", False):
                ax.plot(steps_m, values, color=run["color"], linewidth=1.0, linestyle="-", label=run["label"])
            else:
                sigma = metric.get("mean_sigma", MEAN_SIGMA)
                if not metric.get("no_raw", False):
                    ax.plot(steps_m, values, color=run["color"], alpha=0.15, linewidth=0.6)
                ax.plot(steps_m, smooth(values, sigma), color=run["color"], linewidth=1.0, linestyle="-", label=run["label"])

    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel(metric["ylabel"])
    ax.set_title(metric["title"])
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if metric.get("ymax") is not None:
        ax.set_ylim(top=metric["ymax"])
    ax.set_xlim(0, 4)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, metric["filename"])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def main():
    names = [m["filename"].replace(".pdf", "") for m in METRICS]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "plots",
        nargs="*",
        metavar="PLOT",
        help=f"Which plots to generate (default: all). Choices: {names}",
    )
    args = parser.parse_args()
    selected = set(args.plots) if args.plots else set(names)
    invalid = selected - set(names)
    if invalid:
        parser.error(f"invalid plot(s): {invalid}. Choose from {names}")

    print("Generating plots...")
    for metric in METRICS:
        if metric["filename"].replace(".pdf", "") not in selected:
            continue
        print(f"\n[{metric['title']}]")
        plot_metric(metric, RUNS, OUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
