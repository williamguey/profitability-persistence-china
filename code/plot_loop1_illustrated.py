"""Render all five illustrated figures from distributed aggregate inputs.

Offline rendering requires no source workbook, record-level data, or model fits.
Histogram and hexagon counts, bin medians and saved coefficients are sufficient.
Run from any directory with the dependencies in requirements.txt installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output/illustrated-figures"
DATA = ROOT / "figure-data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "profitability-illustrated-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
from pypdf import PdfReader

PRIMARY = ROOT / "analysis/original-forecast/transparent_main"
NAVY = "#173B5F"
TEAL = "#087F8C"
ORANGE = "#CB702C"
GRAY = "#66737E"
LIGHT = "#EDF2F5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.labelsize": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelcolor": NAVY, "text.color": NAVY, "xtick.color": GRAY,
    "ytick.color": GRAY, "axes.edgecolor": "#8B959E", "axes.linewidth": .65,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.fontsize": 9, "legend.frameon": False,
    "lines.linewidth": 1.6, "patch.linewidth": .4,
})

SOURCE_PATHS = sorted(DATA.glob("*.csv")) + [
    DATA / "accounting-relationships-display.json",
    DATA / "persistence-shape-reconstruction-check.json",
    PRIMARY / "paired_bootstrap.csv",
    ROOT / "analysis/paired_bootstrap.csv",
    Path(__file__).resolve(),
]

manifest = {
    "purpose": "Illustrated profitability figures from saved analysis results",
    "script": str(Path(__file__).resolve().relative_to(ROOT)),
    "no_new_estimation": True, "no_new_bootstrap_or_confidence_intervals": True,
    "exports": "Five vector-PDF/PNG figure pairs and an input/output hash manifest; source aggregates are read-only",
    "main_target_years": [2015, 2016, 2017, 2018, 2019], "main_sample_n": 15258,
    "source_sha256": {}, "figures": {}, "validation": {},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, obj):
    (DATA / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def save(fig, name, facts):
    fig.savefig(FIG / f"{name}.pdf", metadata={
        "Title": name.replace("-", " ").title(),
        "Subject": "Frozen LOOP1 analysis; no new model estimation",
        "Creator": "code/plot_loop1_illustrated.py",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(FIG / f"{name}.png", dpi=180)
    page = PdfReader(FIG / f"{name}.pdf").pages[0]
    resources = page["/Resources"].get_object()
    objects = resources.get("/XObject", {})
    images = [str(k) for k, v in objects.items() if v.get_object().get("/Subtype") == "/Image"]
    assert not images, f"Raster embedded in vector PDF: {name}"
    facts.update({
        "pdf": str((FIG / f"{name}.pdf").relative_to(ROOT)),
        "png": str((FIG / f"{name}.png").relative_to(ROOT)),
        "width_inches": float(fig.get_figwidth()), "height_inches": float(fig.get_figheight()),
        "png_dpi": 180, "pdf_raster_image_count": len(images),
        "pdf_page_size_points": [float(page.mediabox.width), float(page.mediabox.height)],
    })
    manifest["figures"][name] = facts
    plt.close(fig)


def light_grid(ax, axis="y"):
    ax.grid(axis=axis, color="#DEE5EA", linewidth=.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6)


def load_inputs():
    for path in SOURCE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest["source_sha256"][str(path.relative_to(ROOT))] = digest(path)
    return None, None


def accounting_relationships(raw):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.7))
    fig.subplots_adjust(left=.09, right=.865, bottom=.22, top=.84, wspace=.13)
    cmap = LinearSegmentedColormap.from_list("loop1_counts", ["#E5F1F3", "#3193A0", NAVY])
    bins_out, hex_out, summaries, hbs = [], [], [], []
    display = json.loads((DATA / "accounting-relationships-display.json").read_text())
    hexbins = pd.read_csv(DATA / "accounting-relationships-hexbin-counts.csv", float_precision="round_trip")
    medians = pd.read_csv(DATA / "accounting-relationships-median-bins.csv", float_precision="round_trip")
    for idx, (ax, signal, title, xlabel) in enumerate(zip(
        axs, ["x1", "x2"], ["(a) Working capital", "(b) Share capital"],
        ["Working capital / assets", "Share capital / assets"],
    )):
        info = next(item for item in display if item["signal"] == signal)
        xlo, xhi = info["x_1st_99th_percentile_ratio"]
        ylo, yhi = info["y_1st_99th_percentile_pp"]
        cells = hexbins.loc[hexbins.signal.eq(signal)]
        hb = ax.hexbin(cells.hex_center_x_ratio, cells.hex_center_y_roe_pp,
                       C=cells["count"], reduce_C_function=np.sum,
                       gridsize=(40, 28), mincnt=1,
                       extent=(xlo, xhi, ylo, yhi), cmap=cmap, linewidths=0,
                       norm=LogNorm(vmin=1), rasterized=False, zorder=2)
        hbs.append(hb)
        assert np.isclose(hb.get_array().sum(), info["displayed_n"])
        assert len(hb.get_array()) == len(cells)
        points = medians.loc[medians.signal.eq(signal), ["x_median_ratio", "y_median_roe_pp"]].to_numpy()
        ax.plot(points[:, 0], points[:, 1], "o-", color=ORANGE, markersize=3.7,
                markeredgecolor="white", markeredgewidth=.45, linewidth=1.4, zorder=4)
        ax.set(xlim=(xlo, xhi), ylim=(min(-100, ylo), max(38, yhi)), xlabel=xlabel)
        # Exact 1st/99th percentile limits are panel specific; no artificial expansion.
        ax.set_ylim(ylo, yhi)
        ax.set_title(title, loc="left", pad=24)
        ax.text(0, 1.02, f"Complete {info['complete_n']:,}  ·  Displayed {info['displayed_n']:,}",
                transform=ax.transAxes, fontsize=8.8, color=GRAY)
        ax.axhline(0, color="#A3ABB2", lw=.65, ls="--", zorder=1)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        summaries.append(info)
    # Preserve each panel's own quantiles; one x1 input is missing.
    axs[1].tick_params(labelleft=False)
    vmax = max(float(hb.get_array().max()) for hb in hbs)
    for hb in hbs:
        hb.set_norm(LogNorm(vmin=1, vmax=vmax))
    cax = fig.add_axes([.889, .22, .021, .62])
    cb = fig.colorbar(hbs[0], cax=cax)
    cb.solids.set_rasterized(False)
    cb.set_label("Observations per hexagon", labelpad=8, fontsize=9)
    ticks = [n for n in [1, 3, 10, 30, 100, 300, 1000] if n <= vmax]
    cb.set_ticks(ticks, labels=[str(n) for n in ticks])
    cb.ax.minorticks_off()
    axs[0].set_ylabel("Next-year ROE (percentage points)")
    fig.legend([Line2D([0], [0], color=ORANGE, marker="o", markersize=4)],
               ["Median within ten signal-quantile bins (all complete observations)"],
               loc="lower center", bbox_to_anchor=(.49, .015), fontsize=9)
    save(fig, "accounting-relationships", {
        "x_units": "raw ratio to assets", "y_units": "ROE percentage points",
        "panels": summaries, "hexbin_gridsize": [40, 28], "shared_color_scale_max_count": vmax,
        "display_rule": "Panel-specific empirical 1st/99th percentiles of all finite complete pairs, both axes",
        "summary_rule": "Ten equal-frequency x bins computed from all complete pairs, including both y tails",
        "no_regression_or_confidence_band": True,
        "definition_caution": "x2 is share capital/assets, not the original thesis (working capital-share capital)/assets",
    })


def predict_from_coefficients(row, x):
    x = np.asarray(x, dtype=float)
    clipped = np.clip(x, row.origin_clip_lower_pp, row.origin_clip_upper_pp)
    return np.where(x < 0, row.negative_intercept_pp + row.negative_slope * clipped,
                    row.nonnegative_intercept_pp + row.nonnegative_slope * clipped)


def persistence_shape(mainpred):
    coefs = pd.read_csv(DATA / "persistence-shape-saved-coefficients.csv", float_precision="round_trip")
    checks = json.loads((DATA / "persistence-shape-reconstruction-check.json").read_text())
    saved_grid = pd.read_csv(DATA / "persistence-shape-function-grid.csv", float_precision="round_trip")
    curve_rows = []
    for row in coefs.itertuples(index=False):
        points = saved_grid.loc[saved_grid.target_year.eq(row.target_year) & saved_grid.model.eq(row.model)]
        assert np.allclose(predict_from_coefficients(row, points.current_roe_pp), points.predicted_roe_pp,
                           rtol=0, atol=1e-10)
    manifest["validation"]["saved_function_grid_reconstruction"] = True
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.65), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.24, top=.87, wspace=.14)
    for side, ax, bounds, title in zip(["negative", "nonnegative"], axs, [(-60., 0.), (0., 50.)],
                                     ["(a) Negative current ROE", "(b) Nonnegative current ROE"]):
        for row in coefs.sort_values("target_year").itertuples(index=False):
            x = np.linspace(bounds[0], bounds[1], 501)
            # Insert clipping knots exactly; the negative branch ends at the left-hand limit.
            knots = [v for v in [row.origin_clip_lower_pp, row.origin_clip_upper_pp] if bounds[0] < v < bounds[1]]
            x = np.unique(np.r_[x, knots])
            if side == "negative":
                x[-1] = np.nextafter(0., -1.)
            pred = predict_from_coefficients(row, x)
            color = NAVY if row.model == "lad_ar" else TEAL
            ls = "--" if row.model == "lad_ar" else "-"
            ax.plot(x, pred, color=color, ls=ls, alpha=1 if row.target_year == 2019 else .38,
                    lw=2.1 if row.target_year == 2019 else .85, zorder=3 if row.target_year == 2019 else 2)
            if row.target_year == 2019:
                endpoint = pred[-1] if side == "negative" else pred[0]
                ax.plot(0, endpoint, "o", ms=4, color=color,
                        markerfacecolor="white" if side == "negative" else color,
                        markeredgewidth=.85, clip_on=False, zorder=4)
            curve_rows.extend({"target_year": row.target_year, "model": row.model, "branch": side,
                               "current_roe_pp": a, "predicted_roe_pp": b} for a, b in zip(x, pred))
        ax.axhline(0, color="#A3ABB2", lw=.7, zorder=1)
        ax.axvline(0, color="#A3ABB2", lw=.7, zorder=1)
        ax.set(xlim=bounds, ylim=(-30, 36), xlabel="Current ROE (percentage points)")
        ax.set_title(title, loc="left", pad=12)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        light_grid(ax)
    axs[0].set_ylabel("Predicted next-year ROE (pp)")
    handles = [Line2D([0], [0], color=NAVY, ls="--", lw=2.1), Line2D([0], [0], color=TEAL, lw=2.1),
               Line2D([0], [0], color=GRAY, lw=2.1), Line2D([0], [0], color=GRAY, lw=.85, alpha=.5)]
    fig.legend(handles, ["Linear LAD", "Sign-asymmetric LAD", "2019 model", "2015–2018 models"],
               ncol=2, loc="lower center", bbox_to_anchor=(.52, .005), columnspacing=2.1)
    manifest["validation"]["historical_persistence_prediction_count_checked"] = sum(v["n_predictions"] for v in checks)
    manifest["validation"]["historical_persistence_max_abs_prediction_error_pp"] = max(v["max_abs_reconstruction_error_pp"] for v in checks)
    save(fig, "persistence-shape", {
        "units": "ROE percentage points on both axes", "target_years": list(range(2015, 2020)),
        "negative_x_limits_pp": [-60, 0], "nonnegative_x_limits_pp": [0, 50],
        "shared_y_limits_pp": [-30, 36], "bold_target_year": 2019,
        "transforms": "Original training-only clipping; raw origin sign determines branch; zero belongs to nonnegative branch",
        "sign_jump_preserved": True, "separate_negative_left_limit": True,
        "no_accounting_signals": True, "earlier_curve_spread_is_not_confidence_interval": True,
        "prediction_reconstruction": checks,
    })


def gain_decomposition():
    primary = pd.read_csv(PRIMARY / "paired_bootstrap.csv")
    primary = primary.loc[primary.partition.eq("2015-2019")]
    public = pd.read_csv(ROOT / "analysis/paired_bootstrap.csv")
    contrasts = [("lad_ar", "lad_piecewise", "Sign asymmetry", TEAL, "o"),
                 ("lad_piecewise", "lad_signals", "Added signals", ORANGE, "s")]
    rows = []
    fig, axs = plt.subplots(2, 1, figsize=(7, 4.15), sharex=True)
    fig.subplots_adjust(left=.215, right=.97, bottom=.145, top=.81, hspace=.64)
    for ax, frame, panel, count, title in zip(
        axs, [primary, public], ["accounting", "public"], [15258, 13970],
        ["(a) Accounting panel", "(b) Public panel"],
    ):
        for y, (ref, cand, label, color, marker) in zip([1., 0.], contrasts):
            row = frame.loc[frame.reference.eq(ref) & frame.candidate.eq(cand)].iloc[0]
            v, lo, hi = row.mae_reduction_pp, row.ci_low_pp, row.ci_high_pp
            ax.errorbar(v, y, xerr=[[v-lo], [hi-v]], fmt=marker, color=color,
                        ms=6, capsize=3.8, elinewidth=1.6, markeredgecolor="white", markeredgewidth=.65, zorder=3)
            ax.text(.685, y+.18, f"{v:.3f}  [{lo:.3f}, {hi:.3f}]", ha="right", va="bottom", fontsize=9)
            rows.append({"panel": panel, "n": count, **row.to_dict(), "label_3dp": f"{v:.3f} [{lo:.3f}, {hi:.3f}]"})
        ax.axvline(0, color="#9AA5AD", linewidth=.8)
        ax.set(yticks=[1, 0], yticklabels=["Sign asymmetry", "Added signals"], ylim=(-.42, 1.55), xlim=(-.025, .70))
        ax.set_title(f"{title}   ·   N = {count:,}", loc="left", pad=6)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.spines["left"].set_visible(False)
        light_grid(ax, "x")
    axs[-1].set_xlabel("Reduction in MAE (percentage points)")
    axs[-1].set_xticks(np.arange(0, .71, .1))
    fig.suptitle("Two successive gains in LAD forecast accuracy", x=.5, y=.985, fontsize=11, fontweight="bold")
    fig.legend([Line2D([0], [0], color=TEAL, marker="o", lw=1.4),
                Line2D([0], [0], color=ORANGE, marker="s", lw=1.4)],
               ["Linear → sign-asymmetric", "Sign-asymmetric → augmented"],
               ncol=2, loc="upper center", bbox_to_anchor=(.52, .935), fontsize=9)
    save(fig, "gain-decomposition", {
        "units": "MAE reduction in percentage points", "pooled_target_years": [2015, 2016, 2017, 2018, 2019],
        "shared_x_limits_pp": [-.025, .70], "contrasts": rows,
        "intervals": "Previously saved percentile 95% paired security-cluster intervals; 2000 draws, seed 20261025; conditional on fixed forecasts and realized years",
        "panels_not_pooled": True, "number_of_successive_contrasts_per_panel": 2,
    })


def annual_gains():
    plotted = pd.read_csv(DATA / "annual-gains-saved-metrics.csv", float_precision="round_trip")
    years = plotted.year.astype(int).tolist()
    table_rows = plotted.to_dict("records")
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.65))
    fig.subplots_adjust(left=.085, right=.985, bottom=.18, top=.85, wspace=.28)
    for ax in axs:
        ax.axvspan(2019.5, 2020.4, color=LIGHT, zorder=0)
        ax.set(xlim=(2014.7, 2020.35), xticks=years, xlabel="Target year")
        ax.set_xticklabels([str(y) if y in [2015, 2020] else str(y)[2:] for y in years])
        ax.text(.965, 1.035, "2020 separate", transform=ax.transAxes, ha="right", fontsize=8.5, color=GRAY)
        light_grid(ax)
    model_specs = [("primary_linear_mae_pp", "Linear", NAVY, "o", "-"),
                   ("primary_asymmetric_mae_pp", "Sign-asymmetric", TEAL, "s", "--"),
                   ("primary_augmented_mae_pp", "Augmented", ORANGE, "^", ":")]
    for col, label, color, marker, ls in model_specs:
        values = plotted[col].to_numpy()
        axs[0].plot(years[:5], values[:5], color=color, marker=marker, ms=4, lw=1.5, ls=ls, label=label, zorder=3)
        axs[0].plot(years[-2:], values[-2:], color=color, marker=marker, ms=4, lw=1.3, ls="--", zorder=3)
    for col, label, color, marker in [
        ("primary_accounting_relative_gain_pct", "Accounting panel", TEAL, "o"),
        ("public_accounting_relative_gain_pct", "Public panel", ORANGE, "s"),
    ]:
        values = plotted[col].to_numpy()
        axs[1].plot(years[:5], values[:5], color=color, marker=marker, ms=4, lw=1.5, label=label, zorder=3)
        axs[1].plot(years[-2:], values[-2:], color=color, marker=marker, ms=4, lw=1.3, ls="--", zorder=3)
    axs[0].set(ylabel="MAE (percentage points)", ylim=(0, 21), yticks=[0, 5, 10, 15, 20])
    axs[1].set(ylabel="Accounting MAE reduction (%)", ylim=(-.13, 1.12), yticks=[0, .25, .5, .75, 1.0])
    axs[1].axhline(0, color="#8E9BA4", lw=.85)
    axs[0].set_title("(a) Accounting-panel error", loc="left", pad=24)
    axs[1].set_title("(b) Relative accounting gain", loc="left", pad=24)
    axs[0].legend(loc="upper left", labelspacing=.3, handlelength=2.2, borderaxespad=.5)
    axs[1].legend(loc="upper right", labelspacing=.3, handlelength=2.2, borderaxespad=.45)
    save(fig, "annual-gains", {
        "left_units": "MAE percentage points", "right_units": "Percentage reduction in MAE",
        "relative_gain_formula": "100 * (LAD_asymmetric_MAE - LAD_augmented_MAE) / LAD_asymmetric_MAE",
        "left_y_limits_pp": [0, 21], "right_y_limits_pct": [-.13, 1.12],
        "2020_separate_evaluation": True, "2019_to_2020_segment": "dashed",
        "no_new_intervals": True, "table_values": table_rows,
        "connecting_lines_are_descriptive_only": True,
    })


def signal_distributions(raw):
    names = ["Working capital", "Share capital", "Operating profit", "Finance expense",
             "Net profit", "Net nonoperating income", "Income tax", "Depreciation"]
    definitions = ["(current_assets-current_liabilities)/total_assets", "share_capital/total_assets",
                   "operating_profit/total_assets", "financial_expenses_total/total_assets", "net_profit/total_assets",
                   "(non_operating_income-non_operating_expenses)/total_assets", "income_tax_expense/total_assets",
                   "depreciation_fixed_assets/total_assets"]
    fig, axs = plt.subplots(4, 2, figsize=(7, 7.2))
    fig.subplots_adjust(left=.09, right=.98, bottom=.075, top=.91, hspace=.65, wspace=.26)
    histrows, summaries = [], []
    histograms = pd.read_csv(DATA / "signal-distributions-histogram-bins.csv", float_precision="round_trip")
    quantiles = pd.read_csv(DATA / "signal-distributions-quantiles.csv", float_precision="round_trip")
    for i, (ax, name, definition) in enumerate(zip(axs.flat, names, definitions), start=1):
        signal = f"x{i}"
        info = quantiles.loc[quantiles.signal.eq(signal)].iloc[0]
        bins = histograms.loc[histograms.signal.eq(signal)].sort_values("bin")
        lo, median, hi = info.q01_ratio, info.median_ratio, info.q99_ratio
        edges = np.r_[bins.left_edge_ratio.to_numpy(), bins.right_edge_ratio.iloc[-1]]
        counts = bins["count"].to_numpy()
        widths = np.diff(edges)
        density = bins.density_per_unit_ratio.to_numpy()
        assert np.isclose(np.sum(density*widths), counts.sum()/info.finite_observed_n)
        color = TEAL if i % 2 else NAVY
        ax.bar(edges[:-1], density, width=widths, align="edge", color=color, alpha=.84,
               edgecolor="white", linewidth=.3, zorder=2)
        ax.axvline(median, color=ORANGE, lw=1.6, ls="--", zorder=3)
        ax.set(xlim=(lo, hi), ylim=(0, max(density)*1.22))
        ax.set_title(f"({chr(96+i)}) {name}", loc="left", pad=7)
        ax.text(.97, .91, f"N = {int(info.finite_observed_n):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=GRAY)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        if i % 2:
            ax.set_ylabel("Density")
        if i >= 7:
            ax.set_xlabel("Raw ratio to assets")
        light_grid(ax)
        summaries.append(info.to_dict())
    fig.legend([Line2D([0], [0], color=ORANGE, lw=1.6, ls="--")], ["Median of all finite observed inputs"],
               loc="upper center", bbox_to_anchor=(.53, 1), fontsize=9)
    save(fig, "signal-distributions", {
        "layout": "4 rows by 2 columns", "x_units": "raw ratio to assets", "y_units": "density per unit of raw ratio",
        "main_sample_n": 15258, "signals": summaries,
        "histogram_normalization": "count / (total finite observed N * bin width); omitted tails are not renormalized",
        "display_window": "Empirical 1st–99th percentiles of each signal's finite observed values",
        "no_imputation_or_clipping_in_descriptive_histograms": True,
        "display_does_not_remove_records_from_estimation": True,
        "no_normal_fits_or_kernel_smoothing": True,
    })


def main():
    raw, mainpred = load_inputs()
    accounting_relationships(raw)
    persistence_shape(mainpred)
    gain_decomposition()
    annual_gains()
    signal_distributions(raw)
    manifest["output_sha256"] = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted(list(FIG.glob("*.pdf")) + list(FIG.glob("*.png")) + list(DATA.glob("*.csv")))
    }
    (FIG / "render-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"figure_count": len(manifest["figures"]),
                      "main_sample_n": 15258, "validation": manifest["validation"],
                      "output_directory": str(FIG)}, indent=2))


if __name__ == "__main__":
    main()
