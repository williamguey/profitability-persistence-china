"""Post-review 2x2 nesting-order diagnostic; one new LAD specification only."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from analyze_forecasts import fit
from analyze_original import SEED

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "analysis/original-forecast/transparent_main"
OUT = ROOT / "analysis/original-forecast/nesting_order_diagnostic"
NEW_MODEL = "lad_linear_signals"
CELLS = ["lad_ar", NEW_MODEL, "lad_piecewise", "lad_signals"]


def project_saved_design(train, test, transform):
    """Reuse saved primary signal transformations, excluding the two sign terms."""
    continuous = transform["continuous_columns"]
    full_names = transform["all_columns"]
    primary_names = transform["retained_columns"]
    excluded = {"negative_roe", "roe_x_negative"}
    selected = [name for name in primary_names if name not in excluded]

    def matrix(frame):
        raw = frame[continuous].to_numpy(float)
        missing = ~np.isfinite(raw)
        clipped = np.clip(raw, np.asarray(transform["lower"]), np.asarray(transform["upper"]))
        clipped = np.where(missing, np.asarray(transform["median_imputation"]), clipped)
        values = {name: clipped[:, j] for j, name in enumerate(continuous)}
        values["negative_roe"] = frame.negative_roe.to_numpy(float)
        values["roe_x_negative"] = clipped[:, 0]*frame.negative_roe.to_numpy(float)
        values.update({f"{name}_missing": missing[:, j].astype(float) for j, name in enumerate(continuous)})
        full = np.column_stack([values[name] for name in full_names])
        full = (full-np.asarray(transform["mean"]))/np.asarray(transform["std"])
        new = np.column_stack([np.ones(len(frame)), full[:, [full_names.index(name) for name in selected]]])
        primary = np.column_stack([np.ones(len(frame)), full[:, [full_names.index(name) for name in primary_names]]])
        return new, primary, full

    x, primary_x, full_train = matrix(train)
    z, primary_z, _ = matrix(test)
    assert np.linalg.matrix_rank(x, tol=1e-9) == x.shape[1]
    assert np.isfinite(x).all() and np.isfinite(z).all()
    # No signal discarded by the original rank check may become informative
    # after removing the sign terms. In this source the sole dropped flag is
    # x4_missing, identical to x1_missing; validate rather than assume it.
    dropped_checks = {}
    for name in transform["dropped_columns"]:
        if name in excluded:
            continue
        y = full_train[:, full_names.index(name)]
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        err = float(np.max(np.abs(y-x@beta)))
        assert err < 1e-7, f"Dropped feature {name} is relevant after removing sign terms."
        dropped_checks[name] = err
    return x, z, primary_z, selected, dropped_checks


def fit_new_cell():
    p = pd.read_csv(MAIN / "eligible_pairs.csv", dtype={"stock_code": str})
    previous = pd.read_csv(MAIN / "predictions.csv", dtype={"stock_code": str})
    records = json.loads((MAIN / "fit_records.json").read_text())
    records = {r["target_year"]: r for r in records if r["model"] == "lad_signals"}
    new_rows, diagnostics = [], []
    for year in range(2015, 2021):
        train, test = p[p.target_year.lt(year)].copy(), p[p.target_year.eq(year)].copy()
        record = records[year]
        assert len(train) == record["n_train"] and len(test) == record["n_test"]
        x, z, primary_z, names, dropped = project_saved_design(train, test, record["transform"])
        existing = previous[previous.target_year.eq(year)]
        assert np.array_equal(test.stock_code.to_numpy(), existing.stock_code.to_numpy())
        primary_error = float(np.max(np.abs(primary_z@np.asarray(record["beta"])-existing.lad_signals.to_numpy())))
        assert primary_error < 1e-8
        beta, diagnostic = fit(x, train.target.to_numpy(), "lad")
        row = existing.copy()
        row[NEW_MODEL] = z@beta
        new_rows.append(row)
        diagnostics.append({"target_year": year, "model": NEW_MODEL, "n_train": len(train), "n_test": len(test),
            "training_target_max": int(train.target_year.max()), "beta": beta.tolist(),
            "retained_columns": names, "saved_primary_transform": record["transform"],
            "dropped_columns_redundancy_error": dropped,
            "reconstructed_primary_prediction_max_error": primary_error, "diagnostic": diagnostic})
        pd.concat(new_rows, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
        (OUT / "fit_records.json").write_text(json.dumps(diagnostics, indent=2))
        print(f"Linear+signals LAD complete: {year}, train={len(train)}, test={len(test)}", flush=True)
    return pd.concat(new_rows, ignore_index=True)


def partitions(pred):
    return [("2015-2019", pred[pred.target_year.between(2015, 2019)])] + [
        (str(y), pred[pred.target_year.eq(y)]) for y in range(2015, 2021)]


def compare(pred):
    metrics, comparisons = [], []
    definitions = [
        ("accounting_first", np.array([1., -1., 0., 0.]), "lad_ar", NEW_MODEL),
        ("asymmetry_second", np.array([0., 1., 0., -1.]), NEW_MODEL, "lad_signals"),
        ("asymmetry_first", np.array([1., 0., -1., 0.]), "lad_ar", "lad_piecewise"),
        ("accounting_second", np.array([0., 0., 1., -1.]), "lad_piecewise", "lad_signals"),
        ("average_accounting_marginal", np.array([.5, -.5, .5, -.5]), "two_order_average", "two_order_average"),
        ("average_asymmetry_marginal", np.array([.5, .5, -.5, -.5]), "two_order_average", "two_order_average")]
    for label, frame in partitions(pred):
        errors = frame[CELLS].subtract(frame.target, axis=0)
        ae = abs(errors)
        for name in CELLS:
            metrics.append({"partition": label, "model": name, "n": len(frame), "firms": frame.stock_code.nunique(),
                "mae": ae[name].mean(), "rmse": np.sqrt(np.mean(errors[name]**2)), "median_ae": ae[name].median(),
                "bias": errors[name].mean()})
        temp = ae.copy()
        temp["stock_code"] = frame.stock_code
        temp["n"] = 1
        g = temp.groupby("stock_code")[CELLS+["n"]].sum()
        values = g.to_numpy()
        rng = np.random.default_rng(SEED)
        boot_means = np.empty((2000, 4))
        for draw in range(2000):
            sampled = values[rng.integers(0, len(values), len(values))].sum(axis=0)
            boot_means[draw] = sampled[:4]/sampled[4]
        observed = ae.mean().to_numpy()
        for name, weights, reference, candidate in definitions:
            estimate = float(observed@weights)
            boot = boot_means@weights
            if reference in CELLS:
                index = CELLS.index(reference)
                relative = 100*estimate/observed[index]
                relative_boot = 100*boot/boot_means[:, index]
                relative_lo, relative_hi = np.quantile(relative_boot, [.025, .975])
            else:
                relative = relative_lo = relative_hi = None
            comparisons.append({"partition": label, "comparison": name, "reference": reference, "candidate": candidate,
                "mae_reduction_pp": estimate, "ci_low_pp": np.quantile(boot, .025), "ci_high_pp": np.quantile(boot, .975),
                "relative_reduction_pct": relative, "ci_low_relative_pct": relative_lo, "ci_high_relative_pct": relative_hi,
                "n": len(frame), "clusters": len(values), "bootstrap_draws": 2000, "seed": SEED})
    m, c = pd.DataFrame(metrics), pd.DataFrame(comparisons)
    m.to_csv(OUT / "metrics.csv", index=False)
    c.to_csv(OUT / "paired_bootstrap.csv", index=False)
    c[c.comparison.eq("accounting_second")].to_csv(OUT / "existing_accounting_annual_intervals.csv", index=False)
    # The post-review diagnostic must reproduce the pre-existing primary gain.
    old = pd.read_csv(MAIN / "paired_bootstrap.csv")
    old = old[old.partition.eq("2015-2019") & old.reference.eq("lad_piecewise")].iloc[0]
    new = c[c.partition.eq("2015-2019") & c.comparison.eq("accounting_second")].iloc[0]
    for col in ["mae_reduction_pp", "ci_low_pp", "ci_high_pp"]:
        assert np.isclose(old[col], new[col], atol=1e-10)
    return m, c


def attribution():
    pred = pd.read_csv(MAIN / "predictions.csv", dtype={"stock_code": str})
    base = pred[pred.target_year.between(2015, 2019)]
    target = base[base.stock_code.eq("000982") & base.target_year.eq(2019)]
    assert len(target) == 1
    models = ["mean", "median", "random_walk"] + [f"{loss}_{spec}" for loss in ["ols", "lad"] for spec in ["ar", "piecewise", "signals"]]
    rows = []
    for model in models:
        errors = np.abs(base[model]-base.target)
        contribution = float(np.abs(target[model]-target.target).iloc[0])
        rows.append({"model": model, "security": "000982", "target_year": 2019,
            "target_roe_pp": float(target.target.iloc[0]), "prediction_pp": float(target[model].iloc[0]),
            "absolute_error_pp": contribution, "total_absolute_error_pp": float(errors.sum()),
            "absolute_error_share_pct": 100*contribution/errors.sum(), "pooled_mae": errors.mean(),
            "contribution_to_pooled_mae_pp": contribution/len(base), "n": len(base)})
    pd.DataFrame(rows).to_csv(OUT / "000982_2019_absolute_error_attribution.csv", index=False)
    primary = pd.read_csv(MAIN / "metrics.csv")
    refit = pd.read_csv(MAIN / "without_000982_refit/metrics.csv")
    rmse = []
    for label, data, partition in [("full_original", primary, "2015-2019"),
        ("without_000982_scoring_only", primary, "without_000982_scoring"),
        ("without_000982_training_and_scoring", refit, "2015-2019")]:
        for spec in ["ar", "piecewise", "signals"]:
            a = data[data.partition.eq(partition) & data.model.eq("ols_"+spec)].iloc[0]
            b = data[data.partition.eq(partition) & data.model.eq("lad_"+spec)].iloc[0]
            rmse.append({"sample": label, "specification": spec, "n": int(a.n), "ols_rmse": a.rmse,
                "lad_rmse": b.rmse, "lad_minus_ols_rmse": b.rmse-a.rmse,
                "lower_rmse_loss": "ols" if a.rmse < b.rmse else "lad"})
    pd.DataFrame(rmse).to_csv(OUT / "exclusion_ols_lad_rmse.csv", index=False)


def main():
    design = ROOT / "config/original-nesting-order.txt"
    assert design.exists()
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {"status": "post-review diagnostic; one additional model only",
        "design_sha256": hashlib.sha256(design.read_bytes()).hexdigest(),
        "executing_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [MAIN / "eligible_pairs.csv", MAIN / "predictions.csv", MAIN / "fit_records.json"]},
        "primary_models_modified": False, "average_marginal_gains_are_causal": False}
    pred = fit_new_cell()
    m, c = compare(pred)
    attribution()
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2))
    print(m[m.partition.eq("2015-2019")].to_string(index=False), flush=True)
    print(c[c.partition.eq("2015-2019")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
