"""Retrospective chronological evaluation of the recovered thesis worksheet.

Run from the project with .venv/bin/python code/analyze_original.py.
The design/counts are written before any forecasting model is fitted. Private
row-level source material stays under data/original-private or analysis.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy

from analyze_forecasts import fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/original-forecast"
SOURCE = ROOT / "data/original-private/original_panel.csv"
SEED = 20261025
SIGNALS = [f"x{i}" for i in range(1, 9)]
MODELS = ["mean", "median", "random_walk"] + [
    f"{loss}_{spec}" for loss in ["ols", "lad"] for spec in ["ar", "piecewise", "signals"]]
FINANCIAL = {"Capital Market Services", "Insurance", "monetary financial services", "Other financial industries"}
LEGACY_DEFINITIONS = {
    "x1": "(current_assets-current_liabilities)/total_assets",
    "x2": "(current_assets-current_liabilities-share_capital)/total_assets",
    "x3": "(operating_profit-financial_expenses_total)/total_assets",
    "x4": "non_operating_income/non_operating_expenses",
    "x5": "net_profit/total_assets",
    "x6": "EBIT/financial_expenses_positive",
    "x7": "income_tax_expense/total_assets",
    "x8": "depreciation_fixed_assets/total_assets",
}
TRANSPARENT_DEFINITIONS = {
    "x1": "(current_assets-current_liabilities)/total_assets",
    "x2": "share_capital/total_assets",
    "x3": "operating_profit/total_assets",
    "x4": "financial_expenses_total/total_assets",
    "x5": "net_profit/total_assets",
    "x6": "(non_operating_income-non_operating_expenses)/total_assets",
    "x7": "income_tax_expense/total_assets",
    "x8": "depreciation_fixed_assets/total_assets",
}
SIGNAL_SET = "transparent"


def ratios(d):
    """Preserve missing components; require positive finite assets throughout."""
    d = d.copy()
    assets = pd.to_numeric(d.total_assets).where(lambda x: np.isfinite(x) & (x > 0))
    working_capital = d.current_assets - d.current_liabilities
    denom4 = d.non_operating_expenses.where(lambda x: np.isfinite(x) & x.ne(0))
    denom6 = d.financial_expenses_positive.where(lambda x: np.isfinite(x) & x.ne(0))
    if SIGNAL_SET == "legacy":
        values = [working_capital/assets, (working_capital-d.share_capital)/assets,
            (d.operating_profit-d.financial_expenses_total)/assets,
            d.non_operating_income/denom4, d.net_profit/assets, d.EBIT/denom6,
            d.income_tax_expense/assets, d.depreciation_fixed_assets/assets]
    else:
        values = [working_capital/assets, d.share_capital/assets, d.operating_profit/assets,
            d.financial_expenses_total/assets, d.net_profit/assets,
            (d.non_operating_income-d.non_operating_expenses)/assets,
            d.income_tax_expense/assets, d.depreciation_fixed_assets/assets]
    for name, value in zip(SIGNALS, values):
        d[name] = value.where(np.isfinite(value) & assets.notna())
    d["roe_pct"] = d.roe_decimal * 100
    return d


def prepare_pairs():
    d = pd.read_csv(SOURCE, dtype={"stock_code": str})
    assert not d.duplicated(["stock_code", "year"]).any()
    assert d.year.between(2010, 2020).all()
    d = ratios(d)
    known = d.industry.notna() & d.industry.str.strip().ne("")
    financial = d.industry.isin(FINANCIAL)
    eligible_origin = known & ~financial
    origins = d.loc[eligible_origin].copy()
    future = d[["stock_code", "year", "roe_pct"]].rename(columns={"roe_pct": "target"})
    future["year"] -= 1
    joined = origins.merge(future, on=["stock_code", "year"], validate="one_to_one")
    complete_roe = np.isfinite(joined[["roe_pct", "target"]]).all(axis=1)
    p = joined.loc[complete_roe].copy()
    p["target_year"] = p.year+1
    p["negative_roe"] = p.roe_pct.lt(0).astype(int)
    # Industries vary across years in the retrospective source. Eligibility uses
    # the origin-year label; original classification/revision vintages are unknown.
    p.to_csv(OUT / "eligible_pairs.csv", index=False)
    profile = ["year", "total_assets", "net_profit", "operating_profit", "share_capital"]
    nonmissing_roe = d[np.isfinite(d.roe_pct)]
    profile_eligible = nonmissing_roe.dropna(subset=profile)
    duplicate_eligible = profile_eligible.duplicated(profile, keep=False)
    assert not duplicate_eligible.any(), "Recovered finite-ROE securities overlap; revisit issuer deduplication."
    counts = p.groupby("target_year").agg(n=("target", "size"), firms=("stock_code", "nunique"),
        negative_origin=("negative_roe", "sum"), negative_target=("target", lambda x: int((x < 0).sum())))
    counts.to_csv(OUT / "sample_counts.csv")
    missingness = pd.DataFrame({"variable": SIGNALS,
        "origin_missing_n": [int(p[c].isna().sum()) for c in SIGNALS],
        "origin_missing_pct": [100*p[c].isna().mean() for c in SIGNALS]})
    missingness.to_csv(OUT / "ratio_missingness.csv", index=False)
    audit = {"signal_set": SIGNAL_SET, "source_rows": len(d), "source_securities": int(d.stock_code.nunique()),
        "unknown_industry_rows": int((~known).sum()), "financial_rows": int(financial.sum()),
        "known_nonfinancial_origin_rows": int(eligible_origin.sum()),
        "consecutive_pairs_before_roe_requirement": len(joined),
        "pairs_missing_or_nonfinite_roe": int((~complete_roe).sum()),
        "eligible_pairs": len(p), "eligible_securities": int(p.stock_code.nunique()),
        "same_year_duplicate_accounting_profiles_with_finite_roe": int(duplicate_eligible.sum()),
        "invalid_assets_origin_rows": int((~np.isfinite(d.total_assets) | d.total_assets.le(0)).sum()),
        "missing_working_capital_components_origin_rows": int((d.current_assets.isna() | d.current_liabilities.isna()).sum()),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "sample_by_target_year": counts.reset_index().to_dict(orient="records")}
    (OUT / "sample_audit.json").write_text(json.dumps(audit, indent=2))
    return d, p, audit


def freeze_design(audit):
    """Validate the distributed method configuration without private notes."""
    path = ROOT / "config/forecast_design.json"
    cfg = json.loads(path.read_text())
    assert audit["source_sha256"] == cfg["accounting_panel_sha256"], "Prepared source differs from the documented study snapshot."
    assert SIGNAL_SET in cfg["signal_sets"]
    expected = TRANSPARENT_DEFINITIONS if SIGNAL_SET == "transparent" else LEGACY_DEFINITIONS
    assert expected == cfg["signal_sets"][SIGNAL_SET], "Predictor definitions differ from configuration."
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design(train, test, specification, industry=False):
    cols = ["roe_pct"] + (SIGNALS if specification == "signals" else [])
    a = train[cols].to_numpy(float)
    b = test[cols].to_numpy(float)
    am, bm = ~np.isfinite(a), ~np.isfinite(b)
    lower, upper, median = [], [], []
    for j in range(len(cols)):
        observed = a[~am[:, j], j]
        if len(observed):
            lo, hi = np.quantile(observed, [.01, .99])
            fill = np.clip(np.median(observed), lo, hi)
        else:
            lo, hi, fill = 0., 0., 0.
        a[:, j] = np.where(am[:, j], fill, np.clip(a[:, j], lo, hi))
        b[:, j] = np.where(bm[:, j], fill, np.clip(b[:, j], lo, hi))
        lower.append(float(lo)); upper.append(float(hi)); median.append(float(fill))
    names = list(cols)
    if specification in ["piecewise", "signals"]:
        a = np.column_stack([a, train.negative_roe, a[:, 0]*train.negative_roe])
        b = np.column_stack([b, test.negative_roe, b[:, 0]*test.negative_roe])
        names += ["negative_roe", "roe_x_negative"]
    for j, col in enumerate(cols):
        if col in SIGNALS and am[:, j].any() and not am[:, j].all():
            a = np.column_stack([a, am[:, j]])
            b = np.column_stack([b, bm[:, j]])
            names.append(f"{col}_missing")
    categories = []
    if industry:
        categories = sorted(train.industry.unique().tolist())
        for category in categories[1:]:
            a = np.column_stack([a, train.industry.eq(category)])
            b = np.column_stack([b, test.industry.eq(category)])
            names.append("industry:"+category)
    mu, sd = a.mean(axis=0), a.std(axis=0, ddof=0)
    nonconstant = sd > 1e-12
    sd[~nonconstant] = 1
    a, b = (a-mu)/sd, (b-mu)/sd
    keep, dropped = [], []
    # Small predictor count, deterministic ordered retention. QR rank is used
    # rather than assuming the imputed source ratios are linearly independent.
    x = np.ones((len(a), 1))
    for j, name in enumerate(names):
        proposed = np.column_stack([x, a[:, j]])
        if nonconstant[j] and np.linalg.matrix_rank(proposed, tol=1e-9) == proposed.shape[1]:
            keep.append(j)
            x = proposed
        else:
            dropped.append(name)
    z = np.column_stack([np.ones(len(b)), b[:, keep]])
    assert np.isfinite(x).all() and np.isfinite(z).all()
    return x, z, {"continuous_columns": cols, "lower": lower, "upper": upper,
        "median_imputation": median, "all_columns": names, "retained_columns": [names[i] for i in keep],
        "dropped_columns": dropped, "mean": mu.tolist(), "std": sd.tolist(),
        "industry_categories": categories, "industry_reference": categories[0] if categories else None}


def forecasts(p, folder, industry=False):
    folder.mkdir(exist_ok=True, parents=True)
    predictions, records = [], []
    for year in range(2015, 2021):
        train = p[p.target_year < year].copy()
        test = p[p.target_year == year].copy()
        assert train.target_year.max() == year-1 and test.year.eq(year-1).all()
        y = train.target.to_numpy()
        row = test[["stock_code", "company", "industry", "year", "target_year", "roe_pct", "target", "negative_roe"]].copy()
        row["mean"], row["median"], row["random_walk"] = y.mean(), np.median(y), test.roe_pct.to_numpy()
        for spec in ["ar", "piecewise", "signals"]:
            x, z, transform = design(train, test, spec, industry=industry)
            for loss in ["ols", "lad"]:
                model = f"{loss}_{spec}"
                beta, diagnostic = fit(x, y, loss)
                row[model] = z@beta
                records.append({"target_year": year, "model": model, "n_train": len(train),
                    "n_test": len(test), "training_target_max": int(train.target_year.max()),
                    "beta": beta.tolist(), "transform": transform, "diagnostic": diagnostic})
        predictions.append(row)
        # Atomic per-year checkpoint makes interruption recovery cheap.
        pd.concat(predictions, ignore_index=True).to_csv(folder / "predictions.csv", index=False)
        (folder / "fit_records.json").write_text(json.dumps(records, indent=2))
        print(f"{folder.name}: completed {year}, training={len(train)}, test={len(test)}", flush=True)
    return pd.concat(predictions, ignore_index=True)


def partitions(pred):
    base = pred[pred.target_year <= 2019]
    return [(str(y), pred[pred.target_year == y]) for y in range(2015, 2021)] + [
        ("2015-2019", base), ("negative_origin", base[base.negative_roe.eq(1)]),
        ("nonnegative_origin", base[base.negative_roe.eq(0)]),
        ("without_000982_scoring", base[base.stock_code.ne("000982")])]


def metrics(pred, folder):
    rows = []
    for label, frame in partitions(pred):
        for model in MODELS:
            e = frame[model]-frame.target
            byfirm = pd.DataFrame({"firm": frame.stock_code, "abs": np.abs(e)}).groupby("firm").mean()
            rows.append({"partition": label, "model": model, "n": len(frame), "firms": frame.stock_code.nunique(),
                "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.mean(e**2))),
                "median_ae": float(np.median(np.abs(e))), "bias": float(np.mean(e)),
                "firm_balanced_mae": float(byfirm["abs"].mean())})
    m = pd.DataFrame(rows)
    m.to_csv(folder / "metrics.csv", index=False)
    return m


def uncertainty(pred, folder, full=False):
    comparisons = [("lad_ar", "lad_piecewise"), ("lad_piecewise", "lad_signals"),
        ("ols_piecewise", "ols_signals"), ("ols_signals", "lad_signals"),
        ("random_walk", "lad_signals"), ("median", "lad_signals")]
    rows = []
    selected = ["2015-2019", "2020", "negative_origin", "nonnegative_origin", "without_000982_scoring"] if full else ["2015-2019"]
    for label, frame in partitions(pred):
        if label not in selected:
            continue
        for reference, candidate in comparisons:
            temp = frame[["stock_code"]].copy()
            temp["baseline"] = np.abs(frame[reference]-frame.target)
            temp["candidate"] = np.abs(frame[candidate]-frame.target)
            g = temp.groupby("stock_code").agg(base_sum=("baseline", "sum"), new_sum=("candidate", "sum"), n=("baseline", "size"))
            rng = np.random.default_rng(SEED)
            boot = []
            v = g.to_numpy()
            for _ in range(2000):
                s = v[rng.integers(0, len(v), len(v))].sum(axis=0)
                boot.append([(s[0]-s[1])/s[2], 100*(s[0]-s[1])/s[0]])
            boot = np.asarray(boot)
            estimate = temp.baseline.mean()-temp.candidate.mean()
            rows.append({"partition": label, "reference": reference, "candidate": candidate,
                "mae_reduction_pp": estimate, "relative_reduction_pct": 100*estimate/temp.baseline.mean(),
                "ci_low_pp": np.quantile(boot[:, 0], .025), "ci_high_pp": np.quantile(boot[:, 0], .975),
                "ci_low_relative_pct": np.quantile(boot[:, 1], .025), "ci_high_relative_pct": np.quantile(boot[:, 1], .975),
                "bootstrap_draws": 2000, "seed": SEED, "clusters": len(g)})
    pd.DataFrame(rows).to_csv(folder / "paired_bootstrap.csv", index=False)
    return rows


def diagnostics(pred, folder):
    base = pred[pred.target_year <= 2019]
    rows = []
    for model in MODELS:
        se = (base[model]-base.target)**2
        largest = se.nlargest(1).index[0]
        rows.append({"model": model, "max_sse_share_pct": 100*se.max()/se.sum(),
            "max_sse_security": str(base.loc[largest, "stock_code"]),
            "max_sse_year": int(base.loc[largest, "target_year"]),
            "top_1pct_sse_share_pct": 100*se.nlargest(int(np.ceil(.01*len(se)))).sum()/se.sum()})
    (folder / "tail_diagnostics.json").write_text(json.dumps(rows, indent=2))
    return rows


def plot(m):
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.65), layout="constrained")
    for model, label, style in [("lad_ar", "Linear persistence", "o-"),
        ("lad_piecewise", "Sign-asymmetric persistence", "s--"), ("lad_signals", "Plus eight ratios", "^:")]:
        t = m[m.model.eq(model) & m.partition.isin([str(y) for y in range(2015, 2020)])]
        axes[0].plot(t.partition.astype(int), t.mae, style, color="0.25", label=label, markersize=4)
    axes[0].set(xlabel="Target year", ylabel="Mean absolute error (percentage points)", xticks=range(2015, 2020))
    axes[0].legend(fontsize=7, frameon=False)
    a = m[m.partition.isin([str(y) for y in range(2015, 2020)])].pivot(index="partition", columns="model", values="mae")
    improve = 100*(a.lad_piecewise-a.lad_signals)/a.lad_piecewise
    axes[1].bar(improve.index, improve, color="0.65", edgecolor="black", linewidth=.5)
    axes[1].axhline(0, color="black", linewidth=.6)
    axes[1].set(xlabel="Target year", ylabel="MAE reduction from added ratios (%)")
    axes[0].text(-.17, 1.04, "(a)", transform=axes[0].transAxes)
    axes[1].text(-.17, 1.04, "(b)", transform=axes[1].transAxes)
    fig.savefig(OUT / "forecast_comparison.pdf")
    fig.savefig(OUT / "forecast_comparison.png", dpi=180)
    plt.close(fig)


def checks():
    # Explicit small-frame invariants catch the material leakage/missingness risks.
    rng = np.random.default_rng(SEED)
    train = pd.DataFrame({"roe_pct": np.linspace(-4., 8., 40)})
    train["negative_roe"] = train.roe_pct.lt(0).astype(int)
    for i, c in enumerate(SIGNALS):
        train[c] = rng.normal(size=40)
        train.loc[i, c] = np.nan
    test = train.iloc[[0, 1]].copy()
    x, z, meta = design(train, test, "signals")
    altered = test.copy()
    altered[SIGNALS] = 1e20
    x2, z2, meta2 = design(train, altered, "signals")
    assert np.array_equal(x, x2) and meta == meta2
    assert np.isfinite(z).all() and np.isfinite(z2).all()
    assert "x1_missing" in meta["retained_columns"]
    # Sparse LP LAD solver is checked against a known intercept-only median.
    y = np.array([-10., 1., 3., 4., 9.])
    beta, _ = fit(np.ones((5, 1)), y, "lad")
    assert np.isclose(beta[0], 3.)
    return {"train_design_unchanged_by_test_extremes": True, "missing_inputs_imputed_finitely": True,
        "nonconstant_missing_indicator_retained": True, "lad_exact_median_solution": True}


def main():
    global OUT, SIGNAL_SET
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["primary", "exclude", "industry", "all"], default="all")
    parser.add_argument("--signal-set", choices=["transparent", "legacy"], default="transparent")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    SIGNAL_SET = args.signal_set
    if args.output is not None:
        OUT = args.output.resolve()
    elif SIGNAL_SET == "legacy":
        OUT = OUT / "legacy_ratios"
    OUT.mkdir(parents=True, exist_ok=True)
    test_results = checks()
    d, p, audit = prepare_pairs()
    design_sha = freeze_design(audit)
    (OUT / "implementation_checks.json").write_text(json.dumps(test_results, indent=2))
    configs = [("primary", OUT, p, False), ("exclude", OUT / "without_000982_refit", p[p.stock_code.ne("000982")], False),
        ("industry", OUT / "industry_controls", p, True)]
    for label, folder, pairs, industry in configs:
        if args.mode not in ["all", label]:
            continue
        pred = forecasts(pairs, folder, industry)
        m = metrics(pred, folder)
        ci = uncertainty(pred, folder, full=(label == "primary"))
        tails = diagnostics(pred, folder)
        summary = {"analysis": label, "signal_set": SIGNAL_SET, "sample_audit": audit, "test_n": int(pred.target_year.le(2019).sum()),
            "test_firms": int(pred.loc[pred.target_year.le(2019), "stock_code"].nunique()),
            "primary_comparison": next(v for v in ci if v["partition"] == "2015-2019" and v["reference"] == "lad_piecewise"),
            "design_sha256": design_sha, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2))
        print(m[m.partition.eq("2015-2019")][["model", "mae", "rmse"]].to_string(index=False), flush=True)
        print(json.dumps(summary["primary_comparison"], indent=2), flush=True)
        if label == "primary":
            plot(m)


if __name__ == "__main__":
    main()
