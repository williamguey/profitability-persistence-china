"""Post-analysis LAD sensitivity omitting the named net-profit/assets ratio."""
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

import analyze_original as core
from analyze_original_combined_lines import fit_lad, sha

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "analysis/original-forecast/transparent_main"
OUT = ROOT / "analysis/original-forecast/drop_np_sensitivity"
DESIGN = ROOT / "config/original-income-identity-drop-np.txt"
IDENTITY = ROOT / "analysis/original-forecast/income_identity_diagnostic/summary.json"
SIGNALS = ["x1", "x2", "x3", "x4", "x6", "x7", "x8"]
CELLS = ["lad_piecewise", "lad_signals", "lad_drop_np"]
SEED = 20261025


def forecasts(pairs):
    saved = pd.read_csv(MAIN/"predictions.csv", dtype={"stock_code": str})
    original_records = json.loads((MAIN/"fit_records.json").read_text())
    rows, records = [], []
    for year in range(2015, 2021):
        train, test = pairs[pairs.target_year.lt(year)], pairs[pairs.target_year.eq(year)]
        assert train.target_year.max() == year-1 and test.year.eq(year-1).all()
        previous = saved[saved.target_year.eq(year)]
        keys = ["stock_code", "year", "target_year"]
        assert np.array_equal(test[keys].to_numpy(), previous[keys].to_numpy())
        np.testing.assert_allclose(test[["roe_pct", "target"]], previous[["roe_pct", "target"]], rtol=0, atol=1e-10)
        old = next(r for r in original_records if r["target_year"] == year and r["model"] == "lad_signals")
        core.SIGNALS = [f"x{i}" for i in range(1, 9)]
        full_x, full_z, full_transform = core.design(train, test, "signals")
        assert full_transform["retained_columns"] == old["transform"]["retained_columns"]
        reconstruction = float(np.max(np.abs(full_z@np.asarray(old["beta"])-previous.lad_signals)))
        assert reconstruction < 1e-8
        # Change only this process's feature list; the primary helper is read-only.
        core.SIGNALS = SIGNALS
        x, z, transform = core.design(train, test, "signals")
        for field in ["lower", "upper", "median_imputation"]:
            for j, name in enumerate(transform["continuous_columns"]):
                k = full_transform["continuous_columns"].index(name)
                np.testing.assert_allclose(transform[field][j], full_transform[field][k], rtol=0, atol=1e-12)
        for field in ["mean", "std"]:
            for j, name in enumerate(transform["all_columns"]):
                k = full_transform["all_columns"].index(name)
                np.testing.assert_allclose(transform[field][j], full_transform[field][k], rtol=0, atol=1e-12)
        shared_errors = []
        for j, name in enumerate(transform["retained_columns"]):
            if name in full_transform["retained_columns"]:
                k = full_transform["retained_columns"].index(name)
                shared_errors += [float(np.max(np.abs(x[:,j+1]-full_x[:,k+1]))),
                                  float(np.max(np.abs(z[:,j+1]-full_z[:,k+1])))]
        assert max(shared_errors, default=0.) < 1e-10
        assert not any(n in transform["all_columns"] for n in ["x5", "x5_missing"])
        beta, diagnostic, certificate = fit_lad(x, train.target.to_numpy())
        np.savez_compressed(OUT/f"lad_certificate_{year}.npz", **certificate)
        row = previous[["stock_code", "year", "target_year", "roe_pct", "target", "negative_roe", "lad_piecewise", "lad_signals"]].copy()
        row["lad_drop_np"] = z@beta
        rows.append(row)
        records.append({"target_year": year, "model": "lad_drop_np", "n_train": len(train), "n_test": len(test),
            "training_target_max": int(train.target_year.max()), "beta": beta.tolist(), "transform": transform,
            "diagnostic": diagnostic, "certificate_file": f"lad_certificate_{year}.npz",
            "primary_prediction_reconstruction_max_error": reconstruction,
            "shared_processed_column_max_difference": max(shared_errors, default=0.),
            "primary_retained_columns": full_transform["retained_columns"],
            "primary_dropped_columns": full_transform["dropped_columns"]})
        pd.concat(rows, ignore_index=True).to_csv(OUT/"predictions.csv", index=False)
        (OUT/"fit_records.json").write_text(json.dumps(records, indent=2))
        print(f"Drop-NP LAD completed target {year}: train {len(train)}, evaluation {len(test)}", flush=True)
    return pd.concat(rows, ignore_index=True)


def comparisons(pred):
    metrics, contrasts = [], []
    specs = [("eight_signals_beyond_asymmetry", np.array([1., -1., 0.]), "lad_piecewise", "lad_signals"),
             ("drop_np_beyond_asymmetry", np.array([1., 0., -1.]), "lad_piecewise", "lad_drop_np"),
             ("drop_np_vs_original_eight", np.array([0., 1., -1.]), "lad_signals", "lad_drop_np")]
    parts = [("2015-2019", pred[pred.target_year.between(2015, 2019)])]+[(str(y), pred[pred.target_year.eq(y)]) for y in range(2015, 2021)]
    for label, frame in parts:
        e = frame[CELLS].subtract(frame.target, axis=0)
        ae = e.abs()
        for model in CELLS:
            security_loss = pd.DataFrame({"stock_code": frame.stock_code, "loss": ae[model]}).groupby("stock_code").loss.mean()
            metrics.append({"partition": label, "model": model, "n": len(frame), "securities": frame.stock_code.nunique(),
                "mae": ae[model].mean(), "rmse": np.sqrt(np.mean(e[model]**2)), "median_ae": ae[model].median(),
                "bias": e[model].mean(), "security_balanced_mae": security_loss.mean()})
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
            relative = 100*draws/bootstrap[:,ref]
            ci, rci = np.quantile(draws,[.025,.975]), np.quantile(relative,[.025,.975])
            contrasts.append({"partition": label, "comparison": name, "reference": reference, "candidate": candidate,
                "mae_reduction_pp": gain, "ci_low_pp": ci[0], "ci_high_pp": ci[1],
                "relative_reduction_pct": 100*gain/observed[ref], "ci_low_relative_pct": rci[0], "ci_high_relative_pct": rci[1],
                "n": len(frame), "clusters": len(values), "bootstrap_draws": 2000, "seed": SEED})
    m, c = pd.DataFrame(metrics), pd.DataFrame(contrasts)
    m.to_csv(OUT/"metrics.csv", index=False)
    c.to_csv(OUT/"paired_bootstrap.csv", index=False)
    previous = pd.read_csv(ROOT/"analysis/original-forecast/nesting_order_diagnostic/paired_bootstrap.csv", dtype={"partition":str})
    for _, row in c[c.comparison.eq("eight_signals_beyond_asymmetry")].iterrows():
        old = previous[previous.partition.eq(row.partition) & previous.comparison.eq("accounting_second")].iloc[0]
        for col in ["mae_reduction_pp", "ci_low_pp", "ci_high_pp", "relative_reduction_pct", "ci_low_relative_pct", "ci_high_relative_pct"]:
            np.testing.assert_allclose(row[col], old[col], rtol=0, atol=1e-10)
    return m, c


def main():
    assert DESIGN.exists()
    gate = json.loads(IDENTITY.read_text())["conditional_sensitivity_decision"]
    assert gate["drop_np_sensitivity_gate_met"]
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [MAIN/"eligible_pairs.csv", MAIN/"predictions.csv", MAIN/"fit_records.json", IDENTITY,
              ROOT/"analysis/original-forecast/nesting_order_diagnostic/paired_bootstrap.csv"]
    helpers = [ROOT/"code/analyze_original.py", ROOT/"code/analyze_original_combined_lines.py"]
    metadata = {"status": "post-analysis omission sensitivity; six exact LAD fits only",
        "started_utc": datetime.now(timezone.utc).isoformat(), "executing_code_sha256": sha(Path(__file__)),
        "design_sha256": sha(DESIGN), "input_sha256": {str(p.relative_to(ROOT)):sha(p) for p in inputs},
        "helper_sha256": {str(p.relative_to(ROOT)):sha(p) for p in helpers}, "signal_order":SIGNALS,
        "conditional_sensitivity_decision":gate,"primary_models_modified":False,
        "interpretation":"Seven named ratios after dropping NP/assets; no assertion of seven independent dimensions, causal earnings-component contributions, or primary model selection."}
    (OUT/"run_manifest.json").write_text(json.dumps(metadata,indent=2))
    pairs = pd.read_csv(MAIN/"eligible_pairs.csv",dtype={"stock_code":str})
    assert len(pairs)==28035
    pred = forecasts(pairs)
    m,c = comparisons(pred)
    assert metadata["input_sha256"]=={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    assert metadata["helper_sha256"]=={str(p.relative_to(ROOT)):sha(p) for p in helpers}
    assert metadata["executing_code_sha256"]==sha(Path(__file__))
    metadata["completed_utc"]=datetime.now(timezone.utc).isoformat()
    metadata["inputs_unchanged"]=True
    metadata["main_comparisons"]=c[c.partition.eq("2015-2019")].to_dict(orient="records")
    metadata["2020_comparisons"]=c[c.partition.eq("2020")].to_dict(orient="records")
    (OUT/"summary.json").write_text(json.dumps(metadata,indent=2,allow_nan=False))
    print(c[c.partition.isin(["2015-2019","2020"])].to_string(index=False),flush=True)


if __name__=="__main__":main()
