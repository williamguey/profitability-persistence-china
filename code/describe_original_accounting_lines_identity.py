"""Post-analysis annual accounting-line and security/name duplication descriptions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/original-forecast/accounting_line_identity_diagnostic"
DESIGN = ROOT / "config/original-accounting-lines-identity.txt"
SOURCE = ROOT / "data/original-private/original_panel.csv"
ELIGIBLE = ROOT / "analysis/original-forecast/transparent_main/eligible_pairs.csv"
PREDICTIONS = ROOT / "analysis/original-forecast/transparent_main/predictions.csv"
KEY = ["stock_code", "year"]
REDUCED = ["year", "total_assets", "net_profit", "operating_profit", "share_capital"]
FULL = ["year", "ROA", "ROE", "current_assets", "liabilities", "net_assets", "total_assets", "share_capital",
        "operating_profit", "financial_expenses_positive", "financial_expenses_total", "net_profit",
        "non_operating_income", "non_operating_expenses", "total_profit", "income_tax_expense",
        "depreciation_fixed_assets", "EBIT", "profit_before", "EBITDA"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_from_keys(raw, path, main=False):
    keys = pd.read_csv(path, dtype={"stock_code": str})
    if main:
        keys = keys[keys.target_year.between(2015, 2019)]
    assert not keys.duplicated(KEY).any()
    out = keys[KEY].merge(raw, on=KEY, how="left", validate="one_to_one", indicator=True)
    assert out._merge.eq("both").all()
    return out.drop(columns="_merge")


def annual(eligible):
    assert len(eligible) == 28035 and eligible.year.between(2010, 2019).all()
    a = eligible.total_assets.where(np.isfinite(eligible.total_assets) & eligible.total_assets.gt(0))
    op = eligible.operating_profit / a
    no = (eligible.non_operating_income-eligible.non_operating_expenses) / a
    frame = eligible.copy()
    frame["operating_profit_over_assets"] = op.where(np.isfinite(op))
    frame["net_nonoperating_over_assets"] = no.where(np.isfinite(no))
    frame["sum_operating_and_net_nonoperating_over_assets"] = (op+no).where(np.isfinite(op) & np.isfinite(no))
    rows = []
    for year, sub in frame.groupby("year", sort=True):
        assert sub.stock_code.nunique() == len(sub)
        for variable in ["operating_profit_over_assets", "net_nonoperating_over_assets", "sum_operating_and_net_nonoperating_over_assets"]:
            finite = sub.loc[np.isfinite(sub[variable]), variable]
            q = finite.quantile([.01, .5, .99])
            rows.append({"origin_year": int(year), "variable": variable, "eligible_n": len(sub),
                "securities": sub.stock_code.nunique(), "finite_ratio_n": len(finite), "missing_ratio_n": len(sub)-len(finite),
                "raw_mean_assets_fraction": finite.mean(), "raw_median_assets_fraction": q.loc[.5],
                "raw_q01_assets_fraction": q.loc[.01], "raw_q99_assets_fraction": q.loc[.99],
                "negative_ratio_n": int(finite.lt(0).sum()), "zero_ratio_n": int(finite.eq(0).sum())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "annual_accounting_line_summaries.csv", index=False)
    return result


def duplicate_counts(frame, columns, complete_only):
    sample = frame.dropna(subset=columns) if complete_only else frame
    dup = sample[sample.duplicated(columns, keep=False)]
    return {"evaluated_rows": len(sample), "duplicate_rows": len(dup),
            "duplicate_groups": len(dup[columns].drop_duplicates()), "distinct_codes_in_duplicate_rows": dup.stock_code.nunique()}


def identities(raw, eligible, main):
    rows, profiles = [], []
    for scope, frame in [("all_source", raw), ("finite_roe_source", raw[np.isfinite(raw.roe_decimal)]),
                         ("all_eligible_origins", eligible), ("main_2015_2019_origins", main)]:
        assert not frame.duplicated(KEY).any()
        names = frame.groupby("company", dropna=False).stock_code.nunique()
        shared_names = set(names[names.gt(1)].index)
        name_rows = frame[frame.company.isin(shared_names)]
        names_year = frame.groupby(["company", "year"], dropna=False).stock_code.nunique()
        shared_year_names = names_year[names_year.gt(1)]
        sameyear = frame.set_index(["company", "year"]).index.isin(shared_year_names.index)
        rows.append({"scope": scope, "rows": len(frame), "security_codes": frame.stock_code.nunique(),
                     "nonmissing_company_name_strings": frame.company.nunique(dropna=True),
                     "company_name_categories_including_missing": frame.company.nunique(dropna=False),
                     "missing_company_names_n": int(frame.company.isna().sum()),
                     "names_with_multiple_codes": len(shared_names), "codes_using_shared_names": name_rows.stock_code.nunique(),
                     "rows_using_shared_names": len(name_rows), "same_year_shared_name_groups": len(shared_year_names),
                     "same_year_shared_name_rows": int(sameyear.sum()),
                     "same_year_shared_name_codes": frame.loc[sameyear, "stock_code"].nunique()})
        for label, columns, complete in [("reduced_complete", REDUCED, True),
                                         ("full_including_missing_patterns", FULL, False), ("full_complete", FULL, True)]:
            profiles.append({"scope": scope, "profile": label, **duplicate_counts(frame, columns, complete)})
    identity, duplicate = pd.DataFrame(rows), pd.DataFrame(profiles)
    identity.to_csv(OUT / "security_name_overlap_counts.csv", index=False)
    duplicate.to_csv(OUT / "financial_profile_duplicate_counts.csv", index=False)
    assert duplicate[duplicate.scope.isin(["finite_roe_source", "all_eligible_origins", "main_2015_2019_origins"])].duplicate_rows.eq(0).all()
    return identity, duplicate


def main():
    assert DESIGN.exists()
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in [SOURCE, ELIGIBLE, PREDICTIONS]}
    raw = pd.read_csv(SOURCE, dtype={"stock_code": str})
    eligible = rows_from_keys(raw, ELIGIBLE)
    main = rows_from_keys(raw, PREDICTIONS, main=True)
    OUT.mkdir(parents=True, exist_ok=True)
    annual_results = annual(eligible)
    names, duplicates = identities(raw, eligible, main)
    assert hashes == {path: sha(ROOT/path) for path in hashes}
    result = {"status": "post-analysis descriptive accounting-line and security-identity diagnostics; no fits",
              "created_utc": datetime.now(timezone.utc).isoformat(), "code_sha256": sha(Path(__file__)),
              "design_sha256": sha(DESIGN), "input_sha256": hashes, "inputs_unchanged": True,
              "annual_sample": "Exact saved eligible original-source security/origin-year pairs, 2010--2019, each key once; known nonfinancial origin and finite adjacent ROEs.",
              "ratio_units": "Fractions of positive finite total assets; raw unweighted summaries, no clipping, trimming or imputation.",
              "net_nonoperating_definition": "non_operating_income minus non_operating_expenses",
              "reduced_profile_columns": REDUCED, "full_profile_columns": FULL,
              "interpretation": "Changes in annual cross-sections do not identify accounting-rule effects. Company names are source labels, not verified issuer IDs; no board or legal-status inference."}
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2))
    print(annual_results.to_string(index=False))
    print(names.to_string(index=False))
    print(duplicates.to_string(index=False))


if __name__ == "__main__":
    main()
