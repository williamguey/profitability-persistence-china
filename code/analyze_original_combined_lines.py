"""Post-analysis seven-signal LAD sensitivity combining two accounting lines."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

import analyze_original as core

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "analysis/original-forecast/transparent_main"
OUT = ROOT / "analysis/original-forecast/combined_line_sensitivity"
DESIGN = ROOT / "config/original-combined-line-sensitivity.txt"
SIGNALS = ["x1", "x2", "x3_combined", "x4", "x5", "x7", "x8"]
CELLS = ["lad_piecewise", "lad_signals", "lad_combined_signals"]
SEED = 20261025


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    p = pd.read_csv(MAIN / "eligible_pairs.csv", dtype={"stock_code": str})
    raw = pd.read_csv(ROOT / "data/original-private/original_panel.csv", dtype={"stock_code": str})
    key = ["stock_code", "year"]
    assert not p.duplicated(key).any() and not raw.duplicated(key).any()
    columns = ["total_assets", "operating_profit", "non_operating_income", "non_operating_expenses",
               "current_assets", "current_liabilities", "share_capital", "financial_expenses_total",
               "net_profit", "income_tax_expense", "depreciation_fixed_assets"]
    d = p[key].merge(raw[key+columns], on=key, how="left", validate="one_to_one", indicator=True)
    assert d._merge.eq("both").all() and np.array_equal(d[key].to_numpy(), p[key].to_numpy())
    assets = d.total_assets.where(np.isfinite(d.total_assets) & d.total_assets.gt(0))
    unchanged = {"x1": (d.current_assets-d.current_liabilities)/assets,
                 "x2": d.share_capital/assets, "x4": d.financial_expenses_total/assets,
                 "x5": d.net_profit/assets, "x7": d.income_tax_expense/assets,
                 "x8": d.depreciation_fixed_assets/assets}
    for name, value in unchanged.items():
        value = value.where(np.isfinite(value))
        np.testing.assert_allclose(p[name], value, rtol=1e-12, atol=1e-12, equal_nan=True)
    finite_components = np.isfinite(d[["operating_profit", "non_operating_income", "non_operating_expenses"]]).all(axis=1)
    combined = (d.operating_profit+d.non_operating_income-d.non_operating_expenses)/assets
    p["x3_combined"] = combined.where(finite_components & np.isfinite(combined))
    assert len(p) == 28035 and p.target_year.eq(p.year+1).all()
    return p


def fit_lad(x, y):
    n, k = x.shape
    constraints = sparse.hstack([sparse.csc_matrix(x), sparse.eye(n), -sparse.eye(n)], format="csc")
    solution = linprog(np.r_[np.zeros(k), np.ones(2*n)], A_eq=constraints, b_eq=y,
                       bounds=[(None, None)]*k+[(0, None)]*(2*n), method="highs")
    assert solution.success, solution.message
    beta, u, v = solution.x[:k], solution.x[k:k+n], solution.x[k+n:]
    dual = np.asarray(solution.eqlin.marginals)
    observed = float(np.abs(y-x@beta).sum())
    assert np.isclose(observed, solution.fun, rtol=1e-7, atol=1e-5)
    diagnostic = {"solver": "scipy.optimize.linprog/highs", "solver_status": int(solution.status),
                  "iterations": int(solution.nit), "objective": float(solution.fun),
                  "observed_absolute_loss_sum": observed, "dual_objective": float(y@dual),
                  "primal_dual_gap": float(solution.fun-y@dual),
                  "max_constraint_residual": float(np.max(np.abs(x@beta+u-v-y))),
                  "max_dual_stationarity_residual": float(np.max(np.abs(x.T@dual))),
                  "max_dual_bound_violation": float(max(0., np.max(np.abs(dual))-1.)),
                  "max_complementarity_residual": float(max(np.max(np.abs(u*(1-dual))), np.max(np.abs(v*(1+dual)))))}
    assert diagnostic["max_constraint_residual"] < 1e-6
    assert diagnostic["max_dual_stationarity_residual"] < 1e-6
    assert abs(diagnostic["primal_dual_gap"]) < max(1e-5, abs(solution.fun)*1e-10)
    return beta, diagnostic, {"dual": dual, "positive_residual": u, "negative_residual": v}


def forecasts(p):
    saved = pd.read_csv(MAIN / "predictions.csv", dtype={"stock_code": str})
    records, rows = [], []
    # Process-local feature list only; this does not edit the primary helper file.
    core.SIGNALS = SIGNALS
    for year in range(2015, 2021):
        train, test = p[p.target_year.lt(year)], p[p.target_year.eq(year)]
        assert train.target_year.max() == year-1 and test.year.eq(year-1).all()
        previous = saved[saved.target_year.eq(year)]
        key = ["stock_code", "year", "target_year"]
        assert np.array_equal(test[key].to_numpy(), previous[key].to_numpy())
        np.testing.assert_allclose(test[["roe_pct", "target"]], previous[["roe_pct", "target"]], atol=1e-10, rtol=0)
        x, z, transform = core.design(train, test, "signals")
        beta, diagnostic, certificate = fit_lad(x, train.target.to_numpy())
        np.savez_compressed(OUT / f"lad_certificate_{year}.npz", **certificate)
        row = previous[["stock_code", "year", "target_year", "roe_pct", "target", "negative_roe", "lad_piecewise", "lad_signals"]].copy()
        row["lad_combined_signals"] = z@beta
        rows.append(row)
        records.append({"target_year": year, "model": "lad_combined_signals", "n_train": len(train), "n_test": len(test),
                        "training_target_max": int(train.target_year.max()),
                        "combined_ratio_missing_train_n": int(train.x3_combined.isna().sum()),
                        "combined_ratio_missing_test_n": int(test.x3_combined.isna().sum()),
                        "beta": beta.tolist(), "transform": transform, "diagnostic": diagnostic,
                        "certificate_file": f"lad_certificate_{year}.npz"})
        pd.concat(rows, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
        (OUT / "fit_records.json").write_text(json.dumps(records, indent=2))
        print(f"Combined seven-signal LAD: target {year}, train {len(train)}, test {len(test)}", flush=True)
    return pd.concat(rows, ignore_index=True)


def comparisons(pred):
    metrics, contrasts = [], []
    specs = [("eight_signals_beyond_asymmetry", np.array([1., -1., 0.]), "lad_piecewise", "lad_signals"),
             ("combined_seven_beyond_asymmetry", np.array([1., 0., -1.]), "lad_piecewise", "lad_combined_signals"),
             ("combined_seven_vs_original_eight", np.array([0., 1., -1.]), "lad_signals", "lad_combined_signals")]
    parts = [("2015-2019", pred[pred.target_year.between(2015, 2019)])]+[(str(y), pred[pred.target_year.eq(y)]) for y in range(2015, 2021)]
    for label, frame in parts:
        e = frame[CELLS].subtract(frame.target, axis=0)
        ae = e.abs()
        for model in CELLS:
            metrics.append({"partition": label, "model": model, "n": len(frame), "securities": frame.stock_code.nunique(),
                            "mae": ae[model].mean(), "rmse": np.sqrt(np.mean(e[model]**2)),
                            "median_ae": ae[model].median(), "bias": e[model].mean()})
        aggregate = ae.copy()
        aggregate["stock_code"], aggregate["n"] = frame.stock_code, 1
        values = aggregate.groupby("stock_code")[CELLS+["n"]].sum().to_numpy()
        rng = np.random.default_rng(SEED)
        bootstrap = np.empty((2000, 3))
        for b in range(2000):
            total = values[rng.integers(0, len(values), len(values))].sum(axis=0)
            bootstrap[b] = total[:3]/total[3]
        observed = ae.mean().to_numpy()
        for name, weights, reference, candidate in specs:
            gain, draws = float(observed@weights), bootstrap@weights
            ref = CELLS.index(reference)
            relative = 100*draws/bootstrap[:, ref]
            ci = np.quantile(draws, [.025, .975]); rci = np.quantile(relative, [.025, .975])
            contrasts.append({"partition": label, "comparison": name, "reference": reference, "candidate": candidate,
                "mae_reduction_pp": gain, "ci_low_pp": ci[0], "ci_high_pp": ci[1],
                "relative_reduction_pct": 100*gain/observed[ref], "ci_low_relative_pct": rci[0], "ci_high_relative_pct": rci[1],
                "n": len(frame), "clusters": len(values), "bootstrap_draws": 2000, "seed": SEED})
    m, c = pd.DataFrame(metrics), pd.DataFrame(contrasts)
    m.to_csv(OUT / "metrics.csv", index=False)
    c.to_csv(OUT / "paired_bootstrap.csv", index=False)
    previous = pd.read_csv(ROOT / "analysis/original-forecast/nesting_order_diagnostic/paired_bootstrap.csv")
    previous.partition = previous.partition.astype(str)
    for _, row in c[c.comparison.eq("eight_signals_beyond_asymmetry")].iterrows():
        old = previous[previous.partition.eq(row.partition) & previous.comparison.eq("accounting_second")].iloc[0]
        for col in ["mae_reduction_pp", "ci_low_pp", "ci_high_pp", "relative_reduction_pct", "ci_low_relative_pct", "ci_high_relative_pct"]:
            assert np.isclose(row[col], old[col], atol=1e-10)
    return m, c


def main():
    assert DESIGN.exists()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [MAIN / "eligible_pairs.csv", MAIN / "predictions.csv", MAIN / "fit_records.json",
              ROOT / "data/original-private/original_panel.csv",
              ROOT / "analysis/original-forecast/nesting_order_diagnostic/paired_bootstrap.csv"]
    metadata = {"status": "post-analysis seven-signal sensitivity; six new LAD fits only",
                "started_utc": datetime.now(timezone.utc).isoformat(), "executing_code_sha256": sha(Path(__file__)),
                "helper_sha256": {"code/analyze_original.py": sha(ROOT/"code/analyze_original.py")},
                "design_sha256": sha(DESIGN), "input_sha256": {str(f.relative_to(ROOT)): sha(f) for f in inputs},
                "combined_signal": "(operating_profit+non_operating_income-non_operating_expenses)/positive_finite_total_assets",
                "signal_order": SIGNALS, "primary_models_modified": False,
                "interpretation": "Sensitivity to combining reported lines; not immunity to all accounting changes or a causal accounting-rule effect."}
    (OUT / "run_manifest.json").write_text(json.dumps(metadata, indent=2))
    p = prepare()
    pred = forecasts(p)
    m, c = comparisons(pred)
    assert metadata["input_sha256"] == {str(f.relative_to(ROOT)): sha(f) for f in inputs}
    assert metadata["executing_code_sha256"] == sha(Path(__file__))
    assert metadata["helper_sha256"]["code/analyze_original.py"] == sha(ROOT/"code/analyze_original.py")
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["inputs_unchanged"] = True
    metadata["main_comparisons"] = c[c.partition.eq("2015-2019")].to_dict(orient="records")
    metadata["2020_comparisons"] = c[c.partition.eq("2020")].to_dict(orient="records")
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print(c[c.partition.isin(["2015-2019", "2020"])].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
