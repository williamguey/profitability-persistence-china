"""Post-analysis source coverage and fixed-forecast group descriptions; no fits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

from decompose_original_fixed_forecasts import prepare, groups, CELLS, CONTRASTS, PROVENANCE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/original-forecast/provenance_coverage_diagnostic"
DESIGN = ROOT / "config/original-provenance-coverage.txt"
FINANCIAL = {"Capital Market Services", "Insurance", "monetary financial services", "Other financial industries"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def netprofit_sign(x):
    return np.select([x.isna(), np.isinf(x), x.lt(0), x.eq(0)],
                     ["missing", "nonfinite", "negative", "zero"], default="positive")


def fixed_groups(p):
    existing = pd.read_csv(ROOT / "analysis/original-forecast/fixed_forecast_decomposition/group_contrasts.csv")
    rows = []
    for scope, provenance, state, sub in groups(p):
        n = len(sub)
        loss = sub[CELLS].subtract(sub.target, axis=0).abs()
        means = loss.mean().to_numpy()
        row = {"group_scope": scope, "provenance": provenance, "origin_roe_group": state,
               "n": n, "securities": sub.stock_code.nunique(), "sample_share_pct": 100*n/len(p)}
        for model in CELLS:
            row[model+"_mae_pp"] = loss[model].mean() if n else None
        for name, contrast in CONTRASTS.items():
            value = means@contrast if n else np.nan
            old = existing[existing.group_scope.eq(scope) & existing.provenance.eq(provenance)
                           & existing.origin_roe_group.eq(state) & existing.comparison.eq(name)]
            assert len(old) == 1
            old = old.iloc[0]
            assert int(old.n) == n and int(old.clusters) == sub.stock_code.nunique()
            assert np.isclose(value, old.mae_reduction_pp, atol=1e-10, equal_nan=True)
            row[name+"_gain_pp"] = value
            row[name+"_weighted_contribution_pp"] = value*n/len(p) if n else 0.
            row[name+"_within_group_ci_low_pp"] = old.ci_low_pp
            row[name+"_within_group_ci_high_pp"] = old.ci_high_pp
        rows.append(row)
    frame = pd.DataFrame(rows)
    for scope in ["provenance", "origin_sign", "provenance_x_origin_sign"]:
        sub = frame[frame.group_scope.eq(scope)]
        assert sub.n.sum() == len(p)
        for name in CONTRASTS:
            total = sub[name+"_weighted_contribution_pp"].sum()
            overall = frame.loc[frame.group_scope.eq("overall"), name+"_gain_pp"].iloc[0]
            assert np.isclose(total, overall, atol=1e-10)
    frame.to_csv(OUT / "sign_provenance_fixed_forecasts.csv", index=False)
    return frame


def recovery(p, raw):
    prior = raw[["stock_code", "year", "net_profit"]].rename(columns={"net_profit": "prior_net_profit"})
    prior["year"] += 1
    d = p.merge(prior, on=["stock_code", "year"], how="left", validate="one_to_one")
    d["prior_profit_sign"] = netprofit_sign(d.prior_net_profit)
    rows = []
    scopes = [("overall", "all", "all", d)]
    scopes += [("provenance", status, "all", d[d.provenance.eq(status)]) for status in PROVENANCE]
    scopes += [("prior_profit_sign", "all", status, d[d.prior_profit_sign.eq(status)])
               for status in ["negative", "zero", "positive", "missing", "nonfinite"]]
    scopes += [("provenance_x_prior_profit_sign", provenance, status,
                d[d.provenance.eq(provenance) & d.prior_profit_sign.eq(status)])
               for provenance in PROVENANCE for status in ["negative", "zero", "positive", "missing", "nonfinite"]]
    for scope, provenance, prior_state, sub in scopes:
        negative = sub[sub.origin_net_profit.lt(0) & np.isfinite(sub.origin_net_profit)]
        finite = negative[np.isfinite(negative.target_net_profit)]
        positive, zero = int(finite.target_net_profit.gt(0).sum()), int(finite.target_net_profit.eq(0).sum())
        rows.append({"group_scope": scope, "provenance": provenance, "prior_profit_sign": prior_state,
                     "all_group_pairs": len(sub), "negative_origin_n": len(negative),
                     "negative_origin_securities": negative.stock_code.nunique(),
                     "target_profit_missing_n": int(negative.target_net_profit.isna().sum()),
                     "target_profit_infinite_n": int(np.isinf(negative.target_net_profit).sum()),
                     "finite_transition_denominator": len(finite), "next_profit_positive_n": positive,
                     "next_profit_zero_n": zero, "next_profit_negative_n": int(finite.target_net_profit.lt(0).sum()),
                     "negative_to_positive_pct": 100*positive/len(finite) if len(finite) else None,
                     "negative_to_nonnegative_pct": 100*(positive+zero)/len(finite) if len(finite) else None})
    result = pd.DataFrame(rows)
    for scope in ["provenance", "prior_profit_sign", "provenance_x_prior_profit_sign"]:
        part = result[result.group_scope.eq(scope)]
        assert part.negative_origin_n.sum() == result.iloc[0].negative_origin_n
        assert part.next_profit_positive_n.sum() == result.iloc[0].next_profit_positive_n
    result.to_csv(OUT / "profit_recovery_by_provenance.csv", index=False)
    return result


def coverage(p, raw):
    assert len(raw) == 34799 and raw.stock_code.nunique() == 4445
    codes = set(p.loc[p.provenance.eq("unmatched"), "stock_code"])
    assert len(codes) == 347
    table_rows, year_rows, joint_rows, prefix_rows = [], [], [], []
    for scope, sub in [("all_source_codes", raw), ("unmatched_main_codes", raw[raw.stock_code.isin(codes)])]:
        code = sub.groupby("stock_code").agg(first_year=("year", "min"), last_year=("year", "max"),
                   observed_years=("year", "nunique"), rows=("year", "size"))
        assert code.observed_years.eq(code.rows).all()
        code["internal_missing_years"] = code.last_year-code.first_year+1-code.observed_years
        finite_roe = sub[np.isfinite(sub.roe_decimal)]
        finite_profit = sub[np.isfinite(sub.net_profit)]
        main = p[p.stock_code.isin(code.index)]
        table_rows.append({"scope": scope, "codes": len(code), "source_rows": len(sub),
            "codes_last_export_before_2020": int(code.last_year.lt(2020).sum()),
            "codes_in_2020_export": int(code.last_year.eq(2020).sum()),
            "codes_with_internal_year_gaps": int(code.internal_missing_years.gt(0).sum()),
            "internal_missing_years_total": int(code.internal_missing_years.sum()),
            "source_finite_roe_n": len(finite_roe), "source_negative_roe_n": int(finite_roe.roe_decimal.lt(0).sum()),
            "source_negative_roe_pct": 100*finite_roe.roe_decimal.lt(0).mean(),
            "source_missing_roe_n": int(sub.roe_decimal.isna().sum()),
            "source_finite_profit_n": len(finite_profit), "source_negative_profit_n": int(finite_profit.net_profit.lt(0).sum()),
            "source_negative_profit_pct": 100*finite_profit.net_profit.lt(0).mean(),
            "main_pairs": len(main), "main_codes": main.stock_code.nunique(),
            "main_negative_origins_n": int(main.roe_pct.lt(0).sum()),
            "main_negative_origin_pct": 100*main.roe_pct.lt(0).mean()})
        for field in ["first_year", "last_year", "observed_years", "internal_missing_years"]:
            for value, n in code.groupby(field).size().items():
                year_rows.append({"scope": scope, "field": field, "value": int(value), "codes": int(n),
                                  "share_scope_codes_pct": 100*n/len(code)})
        for (first, last), n in code.groupby(["first_year", "last_year"]).size().items():
            joint_rows.append({"scope": scope, "first_export_year": int(first), "last_export_year": int(last),
                               "codes": int(n)})
        for prefix, code_part in code.groupby(code.index.str[:3]):
            history = sub[sub.stock_code.isin(code_part.index)]
            eval_part = main[main.stock_code.isin(code_part.index)]
            finite = history[np.isfinite(history.roe_decimal)]
            prefix_rows.append({"scope": scope, "code_prefix_3": prefix, "codes": len(code_part),
                "source_rows": len(history), "source_finite_roe_n": len(finite),
                "source_negative_roe_n": int(finite.roe_decimal.lt(0).sum()),
                "source_negative_roe_pct": 100*finite.roe_decimal.lt(0).mean() if len(finite) else None,
                "codes_last_export_before_2020": int(code_part.last_year.lt(2020).sum()),
                "main_pairs": len(eval_part), "main_negative_origins_n": int(eval_part.roe_pct.lt(0).sum()),
                "main_negative_origin_pct": 100*eval_part.roe_pct.lt(0).mean() if len(eval_part) else None})
    for name, rows in [("source_code_coverage", table_rows), ("export_year_distributions", year_rows),
                       ("first_last_export_year_counts", joint_rows), ("code_prefix_counts", prefix_rows)]:
        pd.DataFrame(rows).to_csv(OUT / (name+".csv"), index=False)
    return pd.DataFrame(table_rows)


def missing_roe(raw):
    origin = raw[raw.industry.notna() & raw.industry.str.strip().ne("") & ~raw.industry.isin(FINANCIAL)]
    future = raw[["stock_code", "year", "roe_decimal", "source_excel_row", "net_profit"]].rename(
        columns={"roe_decimal": "target_roe", "source_excel_row": "target_source_excel_row", "net_profit": "target_net_profit"})
    future["year"] -= 1
    pairs = origin.merge(future, on=["stock_code", "year"], validate="one_to_one")
    mo, mt = pairs.roe_decimal.isna(), pairs.target_roe.isna()
    inf = np.isinf(pairs[["roe_decimal", "target_roe"]]).any(axis=1)
    counts = {"consecutive_known_nonfinancial_pairs": len(pairs), "origin_missing_only_pairs": int((mo & ~mt).sum()),
              "target_missing_only_pairs": int((~mo & mt).sum()), "both_missing_pairs": int((mo & mt).sum()),
              "any_missing_pairs": int((mo|mt).sum()), "pairs_with_infinite_roe": int(inf.sum()),
              "retained_finite_pairs": int(np.isfinite(pairs[["roe_decimal", "target_roe"]]).all(axis=1).sum())}
    assert list(counts.values()) == [28439, 97, 153, 154, 404, 0, 28035]
    source_rows = set(pairs.loc[mo, "source_excel_row"].astype(int)) | set(pairs.loc[mt, "target_source_excel_row"].astype(int))
    counts["unique_missing_source_cells"] = len(source_rows)
    assert len(source_rows) == 361
    prior_audit = json.loads((ROOT / "analysis/original-forecast/roe_missingness_audit.json").read_text())
    assert prior_audit["counts"]["unique_missing_source_cells_across_both_roles"] == len(source_rows)
    for mode in ["stored_cells", "cached_values"]:
        assert prior_audit["cell_audit"][mode]["checked_unique_cells"] == len(source_rows)
        assert prior_audit["cell_audit"][mode]["cell_values"] == {"None (blank)": len(source_rows)}
    sign_rows = []
    for scope, values in [("missing_origin_slots", pairs.loc[mo, "net_profit"]),
                          ("missing_target_slots", pairs.loc[mt, "target_net_profit"]),
                          ("unique_missing_source_cells", raw.loc[raw.source_excel_row.isin(source_rows), "net_profit"])]:
        states = pd.Series(netprofit_sign(values)).value_counts()
        for state in ["negative", "zero", "positive", "missing", "nonfinite"]:
            sign_rows.append({"scope": scope, "net_profit_sign": state, "n": int(states.get(state, 0))})
    pd.DataFrame(sign_rows).to_csv(OUT / "missing_roe_profit_sign_counts.csv", index=False)
    result = {"counts": counts, "existing_direct_workbook_audit": prior_audit["cell_audit"],
              "reason_for_blanks": "Unknown; neither a non-positive-equity rule nor a database convention is established.",
              "source_table_identifiers": "Not established by supplied records.",
              "source_extraction_date": "Not established by supplied records; document/filesystem timestamps are not extraction evidence."}
    (OUT / "missing_roe_audit_summary.json").write_text(json.dumps(result, indent=2))
    return result


def annual_gains():
    predictions = pd.read_csv(ROOT / "analysis/original-forecast/nesting_order_diagnostic/predictions.csv", dtype={"stock_code": str})
    intervals = pd.read_csv(ROOT / "analysis/original-forecast/nesting_order_diagnostic/paired_bootstrap.csv")
    intervals.partition = intervals.partition.astype(str)
    selected = intervals[intervals.comparison.eq("asymmetry_first") & intervals.partition.isin([str(y) for y in range(2015, 2021)])].copy()
    assert len(selected) == 6
    for _, row in selected.iterrows():
        sub = predictions[predictions.target_year.eq(int(row.partition))]
        gain = (sub.lad_ar-sub.target).abs().mean()-(sub.lad_piecewise-sub.target).abs().mean()
        assert np.isclose(gain, row.mae_reduction_pp, atol=1e-10)
        assert len(sub) == row.n and sub.stock_code.nunique() == row.clusters
    selected.to_csv(OUT / "annual_asymmetry_first_gains.csv", index=False)
    return selected


def main():
    assert DESIGN.exists()
    inputs = ["analysis/original-forecast/nesting_order_diagnostic/predictions.csv",
              "analysis/original-forecast/nesting_order_diagnostic/paired_bootstrap.csv",
              "analysis/original-forecast/fixed_forecast_decomposition/group_contrasts.csv",
              "analysis/original-forecast/roe_missingness_audit.json",
              "data/original-private/original_panel.csv", "data/processed/peng2022_panel_rawvalues.csv"]
    hashes = {path: digest(ROOT/path) for path in inputs}
    OUT.mkdir(parents=True, exist_ok=True)
    p, raw = prepare()
    fixed = fixed_groups(p)
    transitions = recovery(p, raw)
    coverage_table = coverage(p, raw)
    missing = missing_roe(raw)
    annual = annual_gains()
    assert hashes == {path: digest(ROOT/path) for path in inputs}
    manifest = {"status": "post-analysis descriptive fixed-forecast diagnostics; no refits",
                "created_utc": datetime.now(timezone.utc).isoformat(), "design_sha256": digest(DESIGN),
                "executing_code_sha256": digest(Path(__file__)),
                "helper_code_sha256": digest(ROOT/"code/decompose_original_fixed_forecasts.py"),
                "input_sha256": hashes, "inputs_unchanged": True,
                "interval_scope": "Existing within-group fixed-forecast security bootstrap; no intervals for weighted contributions.",
                "interpretation": "Code prefixes and export years describe source coverage, not legal boards, delisting, survival, or reasons for source inclusion."}
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(coverage_table.to_string(index=False))
    print(transitions[transitions.group_scope.isin(["overall", "provenance", "prior_profit_sign"])].to_string(index=False))
    print(annual.to_string(index=False))


if __name__ == "__main__":
    main()
