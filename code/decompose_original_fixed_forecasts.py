"""Post-review aggregation of fixed original forecasts; no estimation or refits."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/original-forecast/fixed_forecast_decomposition"
SEED = 20261025
CELLS = ["lad_ar", "lad_linear_signals", "lad_piecewise", "lad_signals"]
PROVENANCE = ["matched_at_or_after_listing", "prelisting", "unmatched"]
SIGNS = ["negative", "nonnegative"]
CONTRASTS = {
    "accounting_first": np.array([1., -1., 0., 0.]),
    "asymmetry_second": np.array([0., 1., 0., -1.]),
    "asymmetry_first": np.array([1., 0., -1., 0.]),
    "accounting_second": np.array([0., 0., 1., -1.]),
}


def sign(values):
    values = pd.Series(values)
    return pd.Series(np.select([values.isna(), ~np.isfinite(values), values.lt(0), values.eq(0)],
        ["missing", "nonfinite", "negative", "zero"], default="positive"), index=values.index)


def prepare():
    pred = pd.read_csv(ROOT / "analysis/original-forecast/nesting_order_diagnostic/predictions.csv", dtype={"stock_code": str})
    p = pred[pred.target_year.between(2015, 2019)].copy()
    raw = pd.read_csv(ROOT / "data/original-private/original_panel.csv", dtype={"stock_code": str})
    public = pd.read_csv(ROOT / "data/processed/peng2022_panel_rawvalues.csv", dtype={"stock_code": str})
    dates = public[["stock_code", "listing_date"]].drop_duplicates()
    assert not dates.stock_code.duplicated().any()
    dates["listing_year"] = pd.to_datetime(dates.listing_date, errors="coerce").dt.year
    assert dates.listing_year.notna().all()
    p = p.merge(dates, on="stock_code", how="left", validate="many_to_one")
    p["provenance"] = np.select([p.listing_year.isna(), p.year.lt(p.listing_year)],
        ["unmatched", "prelisting"], default="matched_at_or_after_listing")
    p["origin_sign"] = np.where(p.roe_pct.lt(0), "negative", "nonnegative")
    origin_fields = raw[["stock_code", "year", "net_profit", "roe_decimal"]].rename(
        columns={"net_profit": "origin_net_profit", "roe_decimal": "origin_roe_decimal"})
    target_fields = raw[["stock_code", "year", "net_profit", "roe_decimal"]].rename(
        columns={"year": "target_year", "net_profit": "target_net_profit", "roe_decimal": "target_roe_decimal"})
    p = p.merge(origin_fields, on=["stock_code", "year"], validate="one_to_one")
    p = p.merge(target_fields, on=["stock_code", "target_year"], validate="one_to_one")
    assert len(p) == 15258
    assert np.allclose(p.roe_pct, 100*p.origin_roe_decimal, atol=1e-9, rtol=1e-12)
    assert np.allclose(p.target, 100*p.target_roe_decimal, atol=1e-9, rtol=1e-12)
    for role, roe in [("origin", "roe_pct"), ("target", "target")]:
        p[f"{role}_roe_sign"] = sign(p[roe])
        p[f"{role}_profit_sign"] = sign(p[f"{role}_net_profit"])
    return p, raw


def groups(p):
    yield "overall", "all", "all", p
    for provenance in PROVENANCE:
        yield "provenance", provenance, "all", p[p.provenance.eq(provenance)]
    for state in SIGNS:
        yield "origin_sign", "all", state, p[p.origin_sign.eq(state)]
    for provenance, state in itertools.product(PROVENANCE, SIGNS):
        yield "provenance_x_origin_sign", provenance, state, p[p.provenance.eq(provenance) & p.origin_sign.eq(state)]


def group_analysis(p, raw):
    total_ae = p[CELLS].subtract(p.target, axis=0).abs().sum()
    codes_2020 = set(raw.loc[raw.year.eq(2020), "stock_code"])
    metrics, contrasts, signs, agreement = [], [], [], []
    sign_categories = ["negative", "zero", "positive", "missing", "nonfinite"]
    for scope, provenance, state, frame in groups(p):
        keys = {"group_scope": scope, "provenance": provenance, "origin_roe_group": state}
        n, codes = len(frame), set(frame.stock_code)
        ae = frame[CELLS].subtract(frame.target, axis=0).abs()
        observed = ae.mean().to_numpy() if n else np.full(4, np.nan)
        row = {**keys, "n": n, "securities": len(codes), "sample_share_pct": 100*n/len(p),
            "negative_origin_roe_n": int(frame.roe_pct.lt(0).sum()),
            "negative_origin_roe_share_pct": 100*frame.roe_pct.lt(0).mean() if n else None,
            "codes_absent_from_raw_2020_export": len(codes-codes_2020),
            "codes_absent_from_raw_2020_export_pct": 100*len(codes-codes_2020)/len(codes) if codes else None}
        for model in CELLS:
            row[f"{model}_mae"] = ae[model].mean() if n else None
            row[f"{model}_absolute_error_sum"] = float(ae[model].sum())
            row[f"{model}_share_total_absolute_error_pct"] = 100*ae[model].sum()/total_ae[model]
        metrics.append(row)
        if n:
            temp = ae.copy()
            temp["stock_code"], temp["n"] = frame.stock_code, 1
            values = temp.groupby("stock_code")[CELLS+["n"]].sum().to_numpy()
            rng = np.random.default_rng(SEED)
            boot_means = np.empty((2000, 4))
            for draw in range(2000):
                s = values[rng.integers(0, len(values), len(values))].sum(axis=0)
                boot_means[draw] = s[:4]/s[4]
        for name, weights in CONTRASTS.items():
            value = float(observed@weights) if n else None
            ci = np.quantile(boot_means@weights, [.025, .975]) if n else [None, None]
            contrasts.append({**keys, "comparison": name, "n": n, "clusters": len(codes),
                "mae_reduction_pp": value, "ci_low_pp": ci[0], "ci_high_pp": ci[1],
                "weighted_contribution_to_pooled_gain_pp": value*n/len(p) if n else 0.,
                "bootstrap_draws": 2000 if n else 0, "seed": SEED})
        for role in ["origin", "target"]:
            roe_sign, profit_sign = frame[f"{role}_roe_sign"], frame[f"{role}_profit_sign"]
            counts = pd.crosstab(roe_sign, profit_sign)
            for rs, ns in itertools.product(sign_categories, repeat=2):
                count = int(counts.loc[rs, ns]) if rs in counts.index and ns in counts.columns else 0
                signs.append({**keys, "role": role, "roe_sign": rs, "net_profit_sign": ns, "n": count})
            opposite = ((roe_sign.eq("negative") & profit_sign.eq("positive")) |
                (roe_sign.eq("positive") & profit_sign.eq("negative")))
            agreement.append({**keys, "role": role, "n": n,
                "strict_opposite_sign_n": int(opposite.sum()),
                "same_nonzero_sign_n": int((roe_sign.eq(profit_sign) & roe_sign.isin(["negative", "positive"])).sum()),
                "both_zero_n": int((roe_sign.eq("zero") & profit_sign.eq("zero")).sum()),
                "roe_zero_profit_nonzero_n": int((roe_sign.eq("zero") & profit_sign.isin(["negative", "positive"])).sum()),
                "profit_zero_roe_nonzero_n": int((profit_sign.eq("zero") & roe_sign.isin(["negative", "positive"])).sum()),
                "roe_missing_n": int(roe_sign.eq("missing").sum()), "profit_missing_n": int(profit_sign.eq("missing").sum()),
                "roe_nonfinite_n": int(roe_sign.eq("nonfinite").sum()), "profit_nonfinite_n": int(profit_sign.eq("nonfinite").sum())})
    m, c = pd.DataFrame(metrics), pd.DataFrame(contrasts)
    m.to_csv(OUT / "group_metrics.csv", index=False)
    c.to_csv(OUT / "group_contrasts.csv", index=False)
    pd.DataFrame(signs).to_csv(OUT / "roe_netprofit_sign_counts.csv", index=False)
    pd.DataFrame(agreement).to_csv(OUT / "sign_agreement_summary.csv", index=False)
    checks = {}
    for scope in ["provenance", "origin_sign", "provenance_x_origin_sign"]:
        part = m[m.group_scope.eq(scope)]
        assert part.n.sum() == len(p)
        for model in CELLS:
            weighted = np.nansum(part.n*part[f"{model}_mae"])/len(p)
            true = total_ae[model]/len(p)
            assert np.isclose(weighted, true, atol=1e-10)
            assert np.isclose(part[f"{model}_share_total_absolute_error_pct"].sum(), 100, atol=1e-10)
        for contrast in CONTRASTS:
            weighted = c[c.group_scope.eq(scope) & c.comparison.eq(contrast)].weighted_contribution_to_pooled_gain_pp.sum()
            true = c[c.group_scope.eq("overall") & c.comparison.eq(contrast)].mae_reduction_pp.iloc[0]
            assert np.isclose(weighted, true, atol=1e-10)
        checks[scope+"_weighted_decomposition"] = True
    old = pd.read_csv(ROOT / "analysis/original-forecast/nesting_order_diagnostic/paired_bootstrap.csv")
    old = old[old.partition.eq("2015-2019") & old.comparison.isin(CONTRASTS)]
    for _, row in old.iterrows():
        new = c[c.group_scope.eq("overall") & c.comparison.eq(row.comparison)].iloc[0]
        for col in ["mae_reduction_pp", "ci_low_pp", "ci_high_pp"]:
            assert np.isclose(row[col], new[col], atol=1e-10)
    checks["pooled_joint_bootstrap_matches_previous"] = True
    return m, c, checks


def transitions_and_sanity(p, raw):
    rows = []
    for name, frame in [("overall", p), ("negative_roe_origins", p[p.roe_pct.lt(0)])]:
        negative = frame[frame.origin_net_profit.lt(0) & np.isfinite(frame.origin_net_profit)]
        observed = negative[np.isfinite(negative.target_net_profit)]
        positive = int(observed.target_net_profit.gt(0).sum())
        zero = int(observed.target_net_profit.eq(0).sum())
        rows.append({"group": name, "n": len(frame),
            "origin_net_profit_negative_n": len(negative),
            "origin_net_profit_positive_n": int((frame.origin_net_profit.gt(0) & np.isfinite(frame.origin_net_profit)).sum()),
            "origin_net_profit_zero_n": int(frame.origin_net_profit.eq(0).sum()),
            "origin_net_profit_missing_n": int(frame.origin_net_profit.isna().sum()),
            "origin_net_profit_infinite_n": int(np.isinf(frame.origin_net_profit).sum()),
            "negative_origin_next_profit_missing_n": int(negative.target_net_profit.isna().sum()),
            "negative_origin_next_profit_infinite_n": int(np.isinf(negative.target_net_profit).sum()),
            "observed_negative_origin_transition_denominator": len(observed),
            "negative_to_positive_n": positive, "negative_to_zero_n": zero,
            "negative_to_negative_n": int(observed.target_net_profit.lt(0).sum()),
            "negative_to_nonnegative_pct": 100*(positive+zero)/len(observed) if len(observed) else None})
    pd.DataFrame(rows).to_csv(OUT / "net_profit_transitions.csv", index=False)
    finite = np.isfinite(raw.roe_decimal) & np.isfinite(raw.net_profit)
    a, b = sign(raw.loc[finite, "roe_decimal"]), sign(raw.loc[finite, "net_profit"])
    sanity = {"raw_rows": len(raw), "joint_finite_roe_netprofit_rows": int(finite.sum()),
        "joint_finite_three_way_sign_disagreements": int(a.ne(b).sum()),
        "negative_roe_and_negative_netprofit_rows": int((a.eq("negative") & b.eq("negative")).sum()),
        "zero_roe_joint_finite_rows": int(a.eq("zero").sum()), "zero_netprofit_joint_finite_rows": int(b.eq("zero").sum()),
        "meaning": "Observed sign correspondence only; no equity, denominator, legal status or cause inferred."}
    (OUT / "raw_source_sign_sanity.json").write_text(json.dumps(sanity, indent=2))


def screened_existing():
    folder = ROOT / "analysis/original-forecast/verified_listing_sensitivity"
    m = pd.read_csv(folder / "metrics.csv")
    c = pd.read_csv(folder / "paired_bootstrap.csv")
    m[m.partition.isin(["2015-2019", "2020"]) & m.model.isin(["lad_ar", "lad_piecewise", "lad_signals"])].to_csv(
        OUT / "screened_existing_metrics.csv", index=False)
    c[c.partition.isin(["2015-2019", "2020"]) & c.reference.eq("lad_ar") & c.candidate.eq("lad_piecewise")].to_csv(
        OUT / "screened_existing_asymmetry_gain.csv", index=False)


def main():
    design = ROOT / "config/original-fixed-decomposition.txt"
    assert design.exists()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / "analysis/original-forecast/nesting_order_diagnostic/predictions.csv",
        ROOT / "data/original-private/original_panel.csv", ROOT / "data/processed/peng2022_panel_rawvalues.csv"]
    initial_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    p, raw = prepare()
    m, c, checks = group_analysis(p, raw)
    transitions_and_sanity(p, raw)
    screened_existing()
    for path, digest in initial_hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    manifest = {"status": "post-review fixed-forecast aggregation; no refits", "created_utc": datetime.now(timezone.utc).isoformat(),
        "design_sha256": hashlib.sha256(design.read_bytes()).hexdigest(),
        "executing_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "input_sha256": initial_hashes,
        "checks": checks, "primary_outputs_modified": False,
        "code_absence_interpretation": "Absent from raw2020 source export, not a legal-survival classification",
        "fixed_vs_screened": "Fixed full-sample forecasts stratified here; screened table separately reuses earlier refitted cohort models"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(m[m.group_scope.isin(["overall", "provenance", "provenance_x_origin_sign"])].to_string(index=False))
    print(c[c.group_scope.isin(["provenance", "provenance_x_origin_sign"]) & c.comparison.eq("accounting_second")].to_string(index=False))
    print(pd.read_csv(OUT / "net_profit_transitions.csv").to_string(index=False))


if __name__ == "__main__":
    main()
