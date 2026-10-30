"""Post-review public-panel LAD 2x2 diagnostic; never changes primary outputs."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import analyze_forecasts as primary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/public-factorial"
SEED = 20261025
DRAWS = 2000
CELLS = ["lad_ar", "lad_linear_signals", "lad_piecewise", "lad_signals"]
TRACKED = [ROOT / "analysis" / f for f in ["predictions.csv", "fit_records.json", "metrics.csv", "paired_bootstrap.csv", "eligible_pairs.csv"]]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data():
    d = pd.read_csv(ROOT / "data/processed/peng2022_panel_rawvalues.csv", dtype={"stock_code": str})
    assert not d.duplicated(["stock_code", "year"]).any()
    listed = d[d.listed_by_year_end.eq(1)].copy()
    future = listed[["stock_code", "year", "roe_pct"]].rename(columns={"roe_pct": "target"})
    future["year"] -= 1
    p = listed.merge(future, on=["stock_code", "year"], validate="one_to_one")
    p["target_year"] = p.year + 1
    p = p.loc[np.isfinite(p[["roe_pct", "target"]]).all(axis=1)].copy()
    p["negative_roe"] = p.roe_pct.lt(0).astype(int)
    assert np.isfinite(p[primary.SIGNALS].to_numpy()).all()
    old = pd.read_csv(ROOT / "analysis/eligible_pairs.csv", dtype={"stock_code": str})
    cols = ["stock_code", "year", "target_year", "roe_pct", "target", "negative_roe"] + primary.SIGNALS
    pd.testing.assert_frame_equal(p[cols].reset_index(drop=True), old[cols].reset_index(drop=True), check_dtype=False)
    return d, listed, p


def fit_new(p, pred):
    saved = json.loads((ROOT / "analysis/fit_records.json").read_text())
    records = []
    rows = []
    for year in range(2015, 2021):
        train = p[p.target_year.lt(year)].copy()
        test = p[p.target_year.eq(year)].copy()
        x, z, transform = primary.design(train, test, "signals")
        rec = next(r for r in saved if r["target_year"] == year and r["model"] == "lad_signals")
        assert transform["columns"] == rec["transform"]["columns"]
        assert transform["clip"] == rec["transform"]["clip"]
        for key in ["lower", "upper", "mean", "std"]:
            np.testing.assert_allclose(transform[key], rec["transform"][key], rtol=1e-13, atol=1e-13)
        expected = pred[pred.target_year.eq(year)].set_index("stock_code").loc[test.stock_code]
        np.testing.assert_allclose(test.target, expected.target, rtol=0, atol=1e-10)
        np.testing.assert_allclose(z @ np.array(rec["beta"]), expected.lad_signals, rtol=1e-12, atol=1e-10)
        keep = [0] + [i+1 for i, name in enumerate(transform["columns"]) if name not in ["negative_roe", "roe_x_negative"]]
        assert keep == [0, 1, 2, 3, 4]
        x, z = x[:, keep], z[:, keep]
        assert np.linalg.matrix_rank(x) == x.shape[1]
        beta, diagnostic = primary.fit(x, train.target.to_numpy(), "lad")
        row = test[["stock_code", "year", "target_year"]].copy()
        row["lad_linear_signals"] = z @ beta
        rows.append(row)
        records.append({"target_year": year, "model": "lad_linear_signals", "n_train": len(train), "n_test": len(test),
                        "training_target_max": int(train.target_year.max()), "beta": beta.tolist(),
                        "source_design": "lad_signals", "kept_design_column_indices": keep,
                        "columns": ["intercept"] + transform["columns"][:4], "transform": transform,
                        "diagnostic": diagnostic})
        print(f"Completed public linear-plus-signals LAD target {year}: train={len(train)}, test={len(test)}", flush=True)
    new = pd.concat(rows, ignore_index=True)
    out = pred.merge(new, on=["stock_code", "year", "target_year"], validate="one_to_one", how="left")
    assert len(out) == len(pred) and out.lad_linear_signals.notna().all()
    out.to_csv(OUT / "predictions.csv", index=False)
    (OUT / "fit_records.json").write_text(json.dumps(records, indent=2))
    return out


def partitions(pred):
    return [("2015-2019", pred[pred.target_year.le(2019)])] + [(str(y), pred[pred.target_year.eq(y)]) for y in range(2015, 2021)]


def compare(pred):
    contrasts = [
        ("accounting_first", np.array([1., -1., 0., 0.]), 0),
        ("asymmetry_second", np.array([0., 1., 0., -1.]), 1),
        ("asymmetry_first", np.array([1., 0., -1., 0.]), 0),
        ("accounting_second", np.array([0., 0., 1., -1.]), 2),
        ("average_accounting", np.array([.5, -.5, .5, -.5]), None),
        ("average_asymmetry", np.array([.5, .5, -.5, -.5]), None),
    ]
    stats, metrics = [], []
    for label, frame in partitions(pred):
        ae = np.abs(frame[CELLS].to_numpy() - frame.target.to_numpy()[:, None])
        losses = pd.DataFrame(ae, columns=CELLS)
        losses["stock_code"] = frame.stock_code.to_numpy()
        losses["n"] = 1
        g = losses.groupby("stock_code")[CELLS + ["n"]].sum()
        v = g.to_numpy()
        rng = np.random.default_rng(SEED)
        boots = np.empty((DRAWS, len(CELLS)))
        for b in range(DRAWS):
            sums = v[rng.integers(0, len(v), len(v))].sum(axis=0)
            boots[b] = sums[:4] / sums[4]
        mae = ae.mean(axis=0)
        for j, model in enumerate(CELLS):
            errors = frame[model].to_numpy() - frame.target.to_numpy()
            metrics.append({"partition": label, "model": model, "n": len(frame), "securities": len(g),
                            "mae": mae[j], "rmse": float(np.sqrt(np.mean(errors**2)))})
        for name, weights, baseline in contrasts:
            gains = boots @ weights
            estimate = float(mae @ weights)
            row = {"partition": label, "contrast": name, "mae_reduction_pp": estimate,
                   "ci_low_pp": np.quantile(gains, .025), "ci_high_pp": np.quantile(gains, .975),
                   "n": len(frame), "clusters": len(g), "bootstrap_draws": DRAWS, "seed": SEED}
            if baseline is not None:
                rel = 100*gains / boots[:, baseline]
                row.update(relative_reduction_pct=100*estimate/mae[baseline],
                           ci_low_relative_pct=np.quantile(rel, .025), ci_high_relative_pct=np.quantile(rel, .975),
                           reference_model=CELLS[baseline])
            stats.append(row)
        assert np.isclose(mae[0]-mae[3], mae @ contrasts[4][1] + mae @ contrasts[5][1], atol=1e-12)
    result = pd.DataFrame(stats)
    result.to_csv(OUT / "paired_bootstrap.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    old = pd.read_csv(ROOT / "analysis/paired_bootstrap.csv")
    old = old[old.reference.eq("lad_piecewise") & old.candidate.eq("lad_signals")].iloc[0]
    new = result[result.partition.eq("2015-2019") & result.contrast.eq("accounting_second")].iloc[0]
    for key in ["mae_reduction_pp", "ci_low_pp", "ci_high_pp", "relative_reduction_pct", "ci_low_relative_pct", "ci_high_relative_pct"]:
        np.testing.assert_allclose(old[key], new[key], rtol=0, atol=1e-10)
    return result


def zeros(raw, listed, pairs):
    groups = [("all_source_rows", raw), ("post_listing_rows", listed), ("eligible_adjacent_pairs", pairs),
              ("main_evaluation_origins", pairs[pairs.target_year.between(2015, 2019)])]
    groups += [(f"training_origins_target_{y}", pairs[pairs.target_year.lt(y)]) for y in range(2015, 2021)]
    groups += [(f"evaluation_origins_target_{y}", pairs[pairs.target_year.eq(y)]) for y in range(2015, 2021)]
    groups += [(f"post_listing_source_year_{y}", listed[listed.year.eq(y)]) for y in range(2009, 2021)]
    rows = []
    for label, frame in groups:
        row = {"partition": label, "n": len(frame), "securities": frame.stock_code.nunique()}
        for col in ["roe_pct"] + primary.SIGNALS:
            row[col + "_zero_n"] = int(frame[col].eq(0).sum())
            row[col + "_nonfinite_n"] = int((~np.isfinite(frame[col])).sum())
        z = frame[primary.SIGNALS].eq(0)
        row["any_signal_zero_n"] = int(z.any(axis=1).sum())
        row["all_three_signals_zero_n"] = int(z.all(axis=1).sum())
        row["revenue_and_profit_growth_zero_n"] = int(z[["income_growth_pct", "profit_growth_pct"]].all(axis=1).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "zero_audit.csv", index=False)
    # The package-ready zero audit is aggregate only; do not export security-level zero records.
    return rows


def extreme(pred):
    models = ["mean", "median", "random_walk"] + [f"{loss}_{spec}" for loss in ["ols", "lad", "winsor_y"] for spec in ["ar", "piecewise", "signals"]]
    rows = []
    for label, frame in [("2015-2019", pred[pred.target_year.le(2019)]), ("2019", pred[pred.target_year.eq(2019)])]:
        case = frame[frame.stock_code.eq("000982") & frame.target_year.eq(2019)]
        assert len(case) == 1
        case = case.iloc[0]
        for model in models:
            ae = abs(case[model] - case.target)
            total = np.abs(frame[model] - frame.target).sum()
            rows.append({"partition": label, "model": model, "stock_code": "000982", "target_year": 2019,
                         "target_pct": case.target, "prediction_pct": case[model], "absolute_error_pp": ae,
                         "n": len(frame), "absolute_error_sum_pp": total, "mae_pp": total/len(frame),
                         "single_row_mae_contribution_pp": ae/len(frame), "single_row_absolute_error_share_pct": 100*ae/total})
    pd.DataFrame(rows).to_csv(OUT / "000982_2019_absolute_error.csv", index=False)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(ROOT)): sha(p) for p in TRACKED}
    raw, listed, pairs = data()
    pred = pd.read_csv(ROOT / "analysis/predictions.csv", dtype={"stock_code": str})
    # The primary predictions are read and reused exactly; only cell B is newly fitted.
    pred = fit_new(pairs, pred)
    comparisons = compare(pred)
    zero_rows = zeros(raw, listed, pairs)
    extreme_rows = extreme(pred)
    after = {str(p.relative_to(ROOT)): sha(p) for p in TRACKED}
    assert before == after, "Primary analysis output changed"
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "code_sha256": sha(Path(__file__)),
               "design_sha256": sha(ROOT / "config/public-factorial.txt"), "primary_hashes_before": before,
               "primary_hashes_after": after, "primary_files_unchanged": True, "existing_interval_reproduction": "passed within 1e-10",
               "pooled_main": json.loads(comparisons[comparisons.partition.eq("2015-2019")].to_json(orient="records", double_precision=15)),
               "primary_code_sha256": sha(ROOT / "code/analyze_forecasts.py"),
               "source_panel_sha256": sha(ROOT / "data/processed/peng2022_panel_rawvalues.csv"),
               "zero_interpretation": "Exact numerical zeros retained. Counts cannot establish whether any zero is a missing-data placeholder.",
               "uncertainty_scope": "Paired security bootstrap of fixed forecasts, conditional on deposited cohort and realized calendar years; no refitting or independent calendar resampling.",
               "attribution_scope": "Two-order average gains are descriptive allocations of MAE reduction, not causal contributions."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(comparisons[comparisons.partition.eq("2015-2019")].to_string(index=False))


if __name__ == "__main__":
    main()
