"""Post-analysis income-identity and existing-forecast interpretation diagnostics."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

import analyze_original as core

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "analysis/original-forecast/transparent_main"
OUT = ROOT / "analysis/original-forecast/income_identity_diagnostic"
DESIGN = ROOT / "config/original-income-identity-drop-np.txt"
FIELDS = ["net_profit", "operating_profit", "non_operating_income", "non_operating_expenses", "income_tax_expense"]
SIGNS = np.array([1., -1., -1., 1., 1.])
COMPONENTS = ["x3", "x5", "x6", "x7"]
WEIGHTS = np.array([-1., 1., -1., 1.])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = np.asarray(values, float)
    v = values[np.isfinite(values)]
    a = np.abs(v)
    result = {"rows": len(values), "finite_n": len(v), "unavailable_n": len(values)-len(v)}
    result.update({"signed_mean": float(v.mean()) if len(v) else None,
                   "signed_median": float(np.median(v)) if len(v) else None,
                   "absolute_mean": float(a.mean()) if len(v) else None})
    for name, q in [("median", .5), ("q90", .9), ("q95", .95), ("q99", .99), ("maximum", 1.)]:
        result["absolute_"+name] = float(np.quantile(a, q)) if len(a) else None
    for name, tol in [("exact_zero", 0.), ("le_1e_10", 1e-10), ("le_1e_8", 1e-8), ("le_1e_6", 1e-6), ("le_0p01", .01), ("le_1", 1.)]:
        result[name+"_n"] = int((a <= tol).sum())
    return result


def raw_identity(raw, pairs, pred):
    known = raw.industry.notna() & raw.industry.str.strip().ne("") & ~raw.industry.isin(core.FINANCIAL)
    key = ["stock_code", "year"]
    main = pred[pred.target_year.between(2015, 2019)]
    scopes = [("all_source", raw), ("known_nonfinancial_source", raw[known]),
              ("eligible_origins", pairs[key].merge(raw, on=key, validate="one_to_one")),
              ("main_2015_2019_origins", main[key].merge(raw, on=key, validate="one_to_one"))]
    rows, counts = [], []
    for scope, frame in scopes:
        values = frame[FIELDS].to_numpy(float)
        complete = np.isfinite(values).all(axis=1)
        asset = frame.total_assets.to_numpy(float)
        positive_asset = np.isfinite(asset) & (asset > 0)
        residual = np.where(complete, values@SIGNS, np.nan)
        scaled = np.divide(residual, asset, out=np.full(len(frame), np.nan), where=positive_asset)
        rows += [{"scope": scope, "units": "source_monetary_units", **stats(residual)},
                 {"scope": scope, "units": "residual_over_total_assets", **stats(scaled)}]
        counts.append({"scope": scope, "rows": len(frame), "securities": frame.stock_code.nunique(),
            "five_lines_complete_n": int(complete.sum()), "positive_finite_assets_n": int(positive_asset.sum()),
            "common_complete_positive_assets_n": int((complete & positive_asset).sum()),
            **{name+"_unavailable_n": int((~np.isfinite(frame[name])).sum()) for name in FIELDS}})
    pd.DataFrame(rows).to_csv(OUT/"raw_identity_residuals.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT/"raw_identity_availability.csv", index=False)
    main_row = next(r for r in rows if r["scope"] == "main_2015_2019_origins" and r["units"] == "residual_over_total_assets")
    fraction = main_row["le_1e_8_n"]/main_row["finite_n"]
    return {"main_complete_rows": main_row["finite_n"], "main_asset_scaled_le_1e_8_n": main_row["le_1e_8_n"],
            "fraction": fraction, "predeclared_minimum_fraction": .9, "drop_np_sensitivity_gate_met": fraction >= .9}


def rank_row(matrix, **labels):
    matrix = np.asarray(matrix, float)
    complete = np.isfinite(matrix).all(axis=1)
    matrix = matrix[complete]
    if not len(matrix):
        return {**labels, "rows": 0, "columns_including_intercept": matrix.shape[1]+1,
                "rank_tol_1e_9": 0, "smallest_singular": None, "largest_singular": None, "singular_ratio": None}
    x = np.column_stack([np.ones(len(matrix)), matrix])
    s = np.linalg.svd(x, compute_uv=False)
    return {**labels, "rows": len(x), "columns_including_intercept": x.shape[1],
            "rank_tol_1e_9": int((s > 1e-9).sum()), "smallest_singular": float(s[-1]),
            "largest_singular": float(s[0]), "singular_ratio": float(s[-1]/s[0])}


def preprocessing(pairs, pred, records):
    distributions, counts, ranks, checks = [], [], [], []
    for year in range(2015, 2021):
        train, test = pairs[pairs.target_year.lt(year)], pairs[pairs.target_year.eq(year)]
        record = next(r for r in records if r["target_year"] == year and r["model"] == "lad_signals")
        x, z, transform = core.design(train, test, "signals")
        old = record["transform"]
        for field in ["continuous_columns", "all_columns", "retained_columns", "dropped_columns"]:
            assert transform[field] == old[field], (year, field)
        for field in ["lower", "upper", "median_imputation", "mean", "std"]:
            np.testing.assert_allclose(transform[field], old[field], rtol=1e-12, atol=1e-12)
        saved = pred[pred.target_year.eq(year)]
        assert np.array_equal(saved[["stock_code", "year"]].to_numpy(), test[["stock_code", "year"]].to_numpy())
        error = float(np.max(np.abs(z@np.asarray(record["beta"])-saved.lad_signals)))
        assert error < 1e-8
        checks.append({"target_year": year, "prediction_reconstruction_max_error": error,
            "full_design_columns_including_intercept": x.shape[1], "full_design_rank_tol_1e_9": int(np.linalg.matrix_rank(x, tol=1e-9)),
            "retained_columns": ";".join(transform["retained_columns"]), "dropped_columns": ";".join(transform["dropped_columns"]),
            "all_four_income_components_retained": all(c in transform["retained_columns"] for c in COMPONENTS)})
        columns = transform["continuous_columns"]
        indexes = [columns.index(c) for c in COMPONENTS]
        lo, hi, fill = [np.asarray(transform[field]) for field in ["lower", "upper", "median_imputation"]]
        # Training-affine standardized versions keep the intercept in rank tests.
        for split, frame in [("train", train), ("evaluation", test)]:
            allraw = frame[columns].to_numpy(float)
            raw = allraw[:, indexes]
            missing = ~np.isfinite(raw)
            clipped = np.clip(raw, lo[indexes], hi[indexes])
            imputed = np.where(missing, fill[indexes], raw)
            processed = np.where(missing, fill[indexes], clipped)
            anymissing = missing.any(axis=1)
            anyclip = ((raw < lo[indexes]) | (raw > hi[indexes])).any(axis=1)
            groups = {"all": np.ones(len(frame), bool), "complete_interior": ~anymissing & ~anyclip,
                      "complete_clipped": ~anymissing & anyclip, "missing_component": anymissing}
            counts.append({"target_year": year, "split": split, "rows": len(frame),
                "complete_interior_n": int(groups["complete_interior"].sum()),
                "complete_clipped_n": int(groups["complete_clipped"].sum()), "missing_component_n": int(anymissing.sum()),
                "any_identity_component_clipped_n": int(anyclip.sum()),
                "any_continuous_predictor_clipped_n": int(((allraw < lo) | (allraw > hi)).any(axis=1).sum()),
                "any_continuous_predictor_missing_n": int((~np.isfinite(allraw)).any(axis=1).sum())})
            for stage, matrix in [("raw", raw), ("clipping_only", clipped), ("imputation_only", imputed), ("clipping_and_imputation", processed)]:
                residual = matrix@WEIGHTS
                for group, mask in groups.items():
                    distributions.append({"target_year": year, "split": split, "stage": stage, "group": group, **stats(residual[mask])})
                ranks.append(rank_row(matrix, target_year=year, split=split, stage=stage, representation="four_asset_scaled_components"))
                affine = (matrix-np.asarray(transform["mean"])[indexes])/np.asarray(transform["std"])[indexes]
                ranks.append(rank_row(affine, target_year=year, split=split, stage=stage, representation="four_training_standardized_components"))
            ranks.append(rank_row(raw[groups["complete_interior"]], target_year=year, split=split, stage="raw_complete_interior", representation="four_asset_scaled_components"))
    pd.DataFrame(distributions).to_csv(OUT/"preprocessing_identity_residuals.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT/"preprocessing_row_groups.csv", index=False)
    pd.DataFrame(ranks).to_csv(OUT/"identity_rank_singular_values.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT/"primary_design_reconstruction.csv", index=False)


def tail_and_asymmetry(raw, pred, records):
    main = pred[pred.target_year.between(2015, 2019)]
    target = main[main.stock_code.eq("000982") & main.target_year.eq(2019)]
    assert len(target) == 1
    source = raw[raw.stock_code.eq("000982") & raw.year.eq(2018)]
    assert len(source) == 1
    case = {"stock_code": "000982", "origin_year": 2018, "target_year": 2019,
        "origin_roe_pct": float(target.roe_pct.iloc[0]), "origin_net_profit": float(source.net_profit.iloc[0]),
        "saved_negative_origin": int(target.negative_roe.iloc[0]), "target_roe_pct": float(target.target.iloc[0])}
    assert np.isclose(case["origin_roe_pct"], float(source.roe_decimal.iloc[0])*100)
    (OUT/"private").mkdir(exist_ok=True)
    (OUT/"private/zhongyin_2019_case.json").write_text(json.dumps(case, indent=2))
    loss = main[main.negative_roe.eq(1)]
    groups = [("all_negative_origins", loss), ("exclude_000982_target2019", loss[~(loss.stock_code.eq("000982") & loss.target_year.eq(2019))]),
              ("exclude_all_000982", loss[~loss.stock_code.eq("000982")])]
    rows = [{"sample": name, "n": len(frame), "securities": frame.stock_code.nunique(),
             "target_mean_pp": frame.target.mean(), "target_median_pp": frame.target.median(),
             "saved_full_lad_mae_pp": np.abs(frame.lad_signals-frame.target).mean()} for name, frame in groups]
    pd.DataFrame(rows).to_csv(OUT/"negative_origin_tail_summary.csv", index=False)
    coefficients = []
    for year in range(2015, 2021):
        record = next(r for r in records if r["target_year"] == year and r["model"] == "lad_piecewise")
        transform = record["transform"]
        names, kept = transform["all_columns"], transform["retained_columns"]
        beta = np.asarray(record["beta"])
        slopes = {name: beta[j+1]/transform["std"][names.index(name)] for j, name in enumerate(kept)}
        intercept = beta[0]-sum(slopes[name]*transform["mean"][names.index(name)] for name in kept)
        loss_intercept = intercept+slopes.get("negative_roe", 0.)
        loss_slope = slopes["roe_pct"]+slopes.get("roe_x_negative", 0.)
        lo, hi = transform["lower"][0], min(0., transform["upper"][0])
        evaluation = pred[pred.target_year.eq(year)]
        clipped = np.clip(evaluation.roe_pct, transform["lower"][0], transform["upper"][0])
        reconstructed = intercept+slopes["roe_pct"]*clipped+evaluation.negative_roe*(slopes.get("negative_roe",0.)+slopes.get("roe_x_negative",0.)*clipped)
        error = float(np.max(np.abs(reconstructed-evaluation.lad_piecewise)))
        assert error < 1e-8
        negative = evaluation[evaluation.negative_roe.eq(1)]
        loss_range = loss_intercept+loss_slope*np.array([lo, hi])
        coefficients.append({"target_year": year, "positive_intercept_pp": intercept, "positive_slope": slopes["roe_pct"],
            "negative_intercept_pp": loss_intercept, "negative_slope": loss_slope,
            "training_clip_lower_roe_pp": lo, "negative_interval_upper_roe_pp": hi,
            "negative_side_interval_min_prediction_pp": loss_range.min(), "negative_side_interval_max_prediction_pp": loss_range.max(),
            "negative_evaluation_n": len(negative), "negative_evaluation_min_prediction_pp": negative.lad_piecewise.min(),
            "negative_evaluation_max_prediction_pp": negative.lad_piecewise.max(), "prediction_reconstruction_max_error": error})
    pd.DataFrame(coefficients).to_csv(OUT/"existing_asymmetric_lad_coefficients.csv", index=False)
    return case, rows


def main():
    assert DESIGN.exists()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT/"data/original-private/original_panel.csv", MAIN/"eligible_pairs.csv", MAIN/"predictions.csv", MAIN/"fit_records.json"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    raw, pairs, pred = [pd.read_csv(p, dtype={"stock_code": str}) for p in inputs[:3]]
    records = json.loads(inputs[3].read_text())
    gate = raw_identity(raw, pairs, pred)
    preprocessing(pairs, pred, records)
    case, tail = tail_and_asymmetry(raw, pred, records)
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    summary = {"status": "post-analysis deterministic diagnostics; no fits", "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha(Path(__file__)), "design_sha256": sha(DESIGN), "helper_sha256": sha(ROOT/"code/analyze_original.py"),
        "input_sha256": hashes, "inputs_unchanged": True, "conditional_sensitivity_decision": gate,
        "source_identity": "NP - OP - NOI + NOE + TAX; no source correction imposed",
        "rank_scope": "Numerical ranks at stated tolerance, not an assertion of economic dimension or universal exact identity.",
        "individual_security_record": "Stored privately; excluded from any aggregate-only distribution."}
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(gate, indent=2), flush=True)
    print(json.dumps(case, indent=2), flush=True)
    print(pd.DataFrame(tail).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
