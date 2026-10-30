"""Publication figure from saved primary metrics; no new estimation."""
from pathlib import Path
import argparse
import os

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/original-forecast/transparent_main")
    args = parser.parse_args()
    out = args.output
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    m = pd.read_csv(out / "metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), layout="constrained")
    for model, label, style, color in [
        ("lad_ar", "Linear persistence", "o-", "0.50"),
        ("lad_piecewise", "Sign-asymmetric persistence", "s--", "#0072B2"),
        ("lad_signals", "Plus eight accounting ratios", "^:", "#D55E00")]:
        t = m[m.model.eq(model) & m.partition.isin([str(y) for y in range(2015, 2020)])]
        axes[0].plot(t.partition.astype(int), t.mae, style, color=color, label=label, markersize=4, linewidth=1.25)
    axes[0].set(xlabel="Target year", ylabel="MAE (percentage points)", xticks=range(2015, 2020))
    axes[0].legend(fontsize=7, frameon=False, loc="upper left", handlelength=2.5)
    a = m[m.partition.isin([str(y) for y in range(2015, 2020)])].pivot(index="partition", columns="model", values="mae")
    improvement = 100*(a.lad_piecewise-a.lad_signals)/a.lad_piecewise
    bars = axes[1].bar(improvement.index, improvement, color="#0072B2", edgecolor="black", linewidth=.5)
    axes[1].bar_label(bars, labels=[f"{v:.2f}" for v in improvement], padding=3, fontsize=8)
    axes[1].set(xlabel="Target year", ylabel="MAE reduction from added ratios (%)", ylim=(0, 1.04))
    for ax, title in zip(axes, ["(a) Forecast accuracy", "(b) Incremental accounting information"]):
        ax.set_title(title, loc="left", fontsize=9, pad=9)
        ax.tick_params(axis="both", labelsize=8)
    fig.savefig(out / "forecast_comparison.pdf")
    fig.savefig(out / "forecast_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
