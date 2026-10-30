"""Chronological next-year ROE benchmark; no model selection on evaluation years."""
from pathlib import Path
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)
SEED = 20261025
SIGNALS = ["sell_margin_pct", "income_growth_pct", "profit_growth_pct"]


def prepare_pairs():
    d = pd.read_csv(ROOT / "data/processed/peng2022_panel_rawvalues.csv", dtype={"stock_code": str})
    assert not d.duplicated(["stock_code", "year"]).any()
    d = d.loc[d.listed_by_year_end.eq(1)].copy()
    future = d[["stock_code", "year", "roe_pct"]].rename(columns={"roe_pct": "target"})
    future["year"] -= 1
    p = d.merge(future, on=["stock_code", "year"], validate="one_to_one")
    p["target_year"] = p.year + 1
    p = p.loc[np.isfinite(p[["roe_pct", "target"]]).all(axis=1)].copy()
    p["negative_roe"] = p.roe_pct.lt(0).astype(int)
    assert np.isfinite(p[SIGNALS].to_numpy()).all()
    p.to_csv(OUT / "eligible_pairs.csv", index=False)
    counts = p.groupby("target_year").agg(n=("target", "size"), firms=("stock_code", "nunique"),
        negative_origin=("negative_roe", "sum"), negative_target=("target", lambda x: int((x < 0).sum())))
    counts.to_csv(OUT / "sample_counts.csv")
    return d, p


def design(train, test, specification, clip=True):
    cols = ["roe_pct"] + (SIGNALS if specification == "signals" else [])
    a = train[cols].to_numpy(float)
    b = test[cols].to_numpy(float)
    lo, hi = np.quantile(a, [.01, .99], axis=0)
    if clip:
        a = np.clip(a, lo, hi)
        b = np.clip(b, lo, hi)
    # Negative-ROE indicator is based on the original sign, not on a fitted threshold.
    names = list(cols)
    if specification in ["piecewise", "signals"]:
        a = np.column_stack([a, train.negative_roe, a[:, 0] * train.negative_roe])
        b = np.column_stack([b, test.negative_roe, b[:, 0] * test.negative_roe])
        names += ["negative_roe", "roe_x_negative"]
    mu, sd = a.mean(axis=0), a.std(axis=0, ddof=0)
    sd[sd == 0] = 1
    x = np.column_stack([np.ones(len(a)), (a - mu) / sd])
    z = np.column_stack([np.ones(len(b)), (b - mu) / sd])
    return x, z, {"columns": names, "lower": lo.tolist(), "upper": hi.tolist(),
        "mean": mu.tolist(), "std": sd.tolist(), "clip": clip}


def fit(x, y, loss):
    if loss == "ols":
        beta, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
        assert rank == x.shape[1]
        return beta, {"rank": int(rank), "condition": float(singular[0] / singular[-1])}
    # Exact least absolute deviations via sparse linear programming.
    # min sum(u+v), subject X beta + u - v = y, u,v >= 0.
    n, k = x.shape
    constraint = sparse.hstack([sparse.csc_matrix(x), sparse.eye(n), -sparse.eye(n)], format="csc")
    solution = linprog(np.r_[np.zeros(k), np.ones(2*n)], A_eq=constraint, b_eq=y,
        bounds=[(None, None)]*k + [(0, None)]*(2*n), method="highs")
    if not solution.success:
        raise RuntimeError(solution.message)
    beta = solution.x[:k]
    observed = np.abs(y-x@beta).sum()
    assert np.isclose(observed, solution.fun, rtol=1e-7, atol=1e-5)
    return beta, {"solver_status": int(solution.status), "objective": float(solution.fun),
        "iterations": int(solution.nit), "solver": "scipy.optimize.linprog/highs"}


def forecasts(p):
    predictions, records = [], []
    for year in range(2015, 2021):
        train = p.loc[p.target_year < year].copy()
        test = p.loc[p.target_year == year].copy()
        assert train.target_year.max() == year - 1
        assert test.year.eq(year - 1).all()
        y = train.target.to_numpy()
        row = test[["stock_code", "stock_name", "year", "target_year", "roe_pct", "target", "negative_roe"]].copy()
        row["mean"] = y.mean()
        row["median"] = np.median(y)
        row["random_walk"] = test.roe_pct.to_numpy()
        qlo, qhi = np.quantile(y, [.01, .99])
        row["train_y_q01"], row["train_y_q99"] = qlo, qhi
        for spec in ["ar", "piecewise", "signals"]:
            x, z, transform = design(train, test, spec)
            for loss in ["ols", "lad"]:
                model = f"{loss}_{spec}"
                beta, diagnostic = fit(x, y, loss)
                row[model] = z@beta
                records.append({"target_year": year, "model": model, "n_train": len(train),
                    "n_test": len(test), "training_target_max": int(train.target_year.max()),
                    "beta": beta.tolist(), "transform": transform, "diagnostic": diagnostic})
            beta, diagnostic = fit(x, np.clip(y, qlo, qhi), "ols")
            row[f"winsor_y_{spec}"] = z@beta
            records.append({"target_year": year, "model": f"winsor_y_{spec}", "n_train": len(train),
                "n_test": len(test), "beta": beta.tolist(), "transform": transform,
                "response_clip": [float(qlo), float(qhi)], "diagnostic": diagnostic})
        predictions.append(row)
        print(f"Completed target {year}, training={len(train)}, test={len(test)}", flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUT / "predictions.csv", index=False)
    (OUT / "fit_records.json").write_text(json.dumps(records, indent=2))
    return pred


def metrics(pred):
    models = ["mean", "median", "random_walk"] + [f"{loss}_{spec}"
        for loss in ["ols", "lad", "winsor_y"] for spec in ["ar", "piecewise", "signals"]]
    rows = []
    partitions = [(str(y), pred[pred.target_year == y]) for y in range(2015, 2021)]
    base = pred[pred.target_year <= 2019]
    partitions += [("2015-2019", base), ("negative_origin", base[base.negative_roe.eq(1)]),
        ("nonnegative_origin", base[base.negative_roe.eq(0)]),
        ("without_000982", base[base.stock_code.ne("000982")]),
        ("zero_excluded", base[base.roe_pct.ne(0) & base.target.ne(0)])]
    for label, frame in partitions:
        for model in models:
            e = frame[model] - frame.target
            byfirm = pd.DataFrame({"firm": frame.stock_code, "abs": np.abs(e), "sq": e**2}).groupby("firm").mean()
            clipped_y = np.clip(frame.target, frame.train_y_q01, frame.train_y_q99)
            rows.append({"partition": label, "model": model, "n": len(frame), "firms": frame.stock_code.nunique(),
                "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.mean(e**2))),
                "median_ae": float(np.median(np.abs(e))), "bias": float(np.mean(e)),
                "firm_balanced_mae": float(byfirm["abs"].mean()),
                "clipped_target_mae": float(np.mean(np.abs(frame[model]-clipped_y))),
                "clipped_target_rmse": float(np.sqrt(np.mean((frame[model]-clipped_y)**2)))})
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "metrics.csv", index=False)
    return m


def uncertainty(pred):
    base = pred[pred.target_year <= 2019].copy()
    comparisons = [("lad_ar", "lad_piecewise"), ("lad_piecewise", "lad_signals"),
        ("ols_piecewise", "ols_signals"), ("ols_signals", "lad_signals"),
        ("random_walk", "lad_signals"), ("median", "lad_signals")]
    rows = []
    for reference, candidate in comparisons:
        temp = base[["stock_code"]].copy()
        temp["baseline"] = np.abs(base[reference]-base.target)
        temp["candidate"] = np.abs(base[candidate]-base.target)
        g = temp.groupby("stock_code").agg(base_sum=("baseline", "sum"), new_sum=("candidate", "sum"), n=("baseline", "size"))
        rng = np.random.default_rng(SEED)
        boot = []
        v = g.to_numpy()
        for _ in range(2000):
            s = v[rng.integers(0, len(v), len(v))].sum(axis=0)
            boot.append([(s[0]-s[1])/s[2], 100*(s[0]-s[1])/s[0]])
        boot = np.asarray(boot)
        estimate = temp.baseline.mean()-temp.candidate.mean()
        rows.append({"reference": reference, "candidate": candidate, "mae_reduction_pp": estimate,
            "relative_reduction_pct": 100*estimate/temp.baseline.mean(),
            "ci_low_pp": np.quantile(boot[:,0], .025), "ci_high_pp": np.quantile(boot[:,0], .975),
            "ci_low_relative_pct": np.quantile(boot[:,1], .025), "ci_high_relative_pct": np.quantile(boot[:,1], .975),
            "bootstrap_draws": 2000, "seed": SEED, "clusters": len(g)})
    pd.DataFrame(rows).to_csv(OUT / "paired_bootstrap.csv", index=False)
    return rows


def diagnostics(pred):
    base = pred[pred.target_year <= 2019].copy()
    rows = []
    for m in ["random_walk", "ols_piecewise", "ols_signals", "lad_piecewise", "lad_signals"]:
        se = (base[m]-base.target)**2
        largest = se.nlargest(1).index[0]
        rows.append({"model": m, "max_sse_share_pct": 100*se.max()/se.sum(),
            "max_sse_firm": str(base.loc[largest,"stock_code"]), "max_sse_year": int(base.loc[largest,"target_year"]),
            "top_1pct_sse_share_pct": 100*se.nlargest(int(np.ceil(.01*len(se)))).sum()/se.sum()})
    (OUT / "tail_diagnostics.json").write_text(json.dumps(rows, indent=2))
    return rows


def plots(m):
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.65), layout="constrained")
    for model, label, style in [("lad_ar", "Linear persistence", "o-"),
        ("lad_piecewise", "Sign-asymmetric persistence", "s--"), ("lad_signals", "Plus accounting signals", "^:")]:
        t = m[m.model.eq(model) & m.partition.isin([str(y) for y in range(2015, 2020)])]
        axes[0].plot(t.partition.astype(int), t.mae, style, color="0.25", label=label, markersize=4)
    axes[0].set(xlabel="Target year", ylabel="Mean absolute error (percentage points)", xticks=range(2015,2020))
    axes[0].legend(fontsize=7, frameon=False)
    a = m[m.partition.isin([str(y) for y in range(2015,2020)])].pivot(index="partition",columns="model",values="mae")
    improve = 100*(a.lad_piecewise-a.lad_signals)/a.lad_piecewise
    axes[1].bar(improve.index, improve, color="0.65", edgecolor="black", linewidth=.5)
    axes[1].axhline(0, color="black", linewidth=.6)
    axes[1].set(xlabel="Target year", ylabel="MAE reduction from added signals (%)")
    axes[0].text(-.17,1.04,"(a)",transform=axes[0].transAxes)
    axes[1].text(-.17,1.04,"(b)",transform=axes[1].transAxes)
    fig.savefig(OUT / "forecast_comparison.pdf")
    fig.savefig(OUT / "forecast_comparison.png", dpi=180)
    plt.close(fig)


def main():
    d,p = prepare_pairs()
    pred = forecasts(p)
    m = metrics(pred)
    ci = uncertainty(pred)
    tails = diagnostics(pred)
    plots(m)
    summary = {"raw_listed_rows": len(d), "raw_firms": int(d.stock_code.nunique()),
        "eligible_pairs": len(p), "test_n": int((pred.target_year <= 2019).sum()),
        "test_firms": int(pred.loc[pred.target_year <= 2019,"stock_code"].nunique()),
        "primary": ci[1], "tails": tails,
        "python_dependencies": {"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(m[m.partition.eq("2015-2019")][["model","mae","rmse"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
