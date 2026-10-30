"""Post-analysis original-source sensitivity screened on deposited listing dates.

No primary outputs are overwritten. Every training/evaluation origin must match
the existing public panel listing reference and be at or after the listing year.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import scipy

from analyze_original import forecasts, metrics, uncertainty, diagnostics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/original-forecast/verified_listing_sensitivity"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    executing_code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    p = pd.read_csv(ROOT / "analysis/original-forecast/transparent_main/eligible_pairs.csv", dtype={"stock_code": str})
    public = pd.read_csv(ROOT / "data/processed/peng2022_panel_rawvalues.csv", dtype={"stock_code": str})
    dates = public[["stock_code", "listing_date"]].drop_duplicates()
    assert not dates.stock_code.duplicated().any()
    dates["listing_year"] = pd.to_datetime(dates.listing_date, errors="coerce").dt.year
    joined = p.merge(dates, on="stock_code", how="left", validate="many_to_one")
    keep = joined.listing_year.notna() & joined.year.ge(joined.listing_year)
    restricted = joined[keep].copy()
    restricted.to_csv(OUT / "eligible_pairs.csv", index=False)
    public_predictions = pd.read_csv(ROOT / "analysis/predictions.csv", dtype={"stock_code": str})
    count_rows = []
    for label, before in [("all_eligible", joined), ("2015-2019", joined[joined.target_year.between(2015, 2019)]),
        ("2020", joined[joined.target_year.eq(2020)])] + [(str(y), joined[joined.target_year.eq(y)]) for y in range(2011, 2020)]:
        after = before[before.listing_year.notna() & before.year.ge(before.listing_year)]
        pub = public_predictions if label == "all_eligible" else public_predictions[
            public_predictions.target_year.between(2015, 2019) if label == "2015-2019" else public_predictions.target_year.eq(int(label))]
        overlap_pairs = after[["stock_code", "target_year"]].merge(pub[["stock_code", "target_year"]], on=["stock_code", "target_year"], validate="one_to_one")
        count_rows.append({"partition": label, "before_n": len(before), "after_n": len(after),
            "after_securities": after.stock_code.nunique(),
            "dropped_unknown_date_n": int(before.listing_year.isna().sum()),
            "dropped_prelisting_n": int((before.listing_year.notna() & before.year.lt(before.listing_year)).sum()),
            "same_year_listing_n": int(after.year.eq(after.listing_year).sum()),
            "new_codes_relative_public_reference": len(set(after.stock_code)-set(dates.stock_code)),
            "overlap_codes_with_public_predictions": len(set(after.stock_code)&set(pub.stock_code)),
            "new_codes_relative_public_predictions": len(set(after.stock_code)-set(pub.stock_code)),
            "overlap_security_target_pairs_with_public_predictions": len(overlap_pairs)})
    counts = pd.DataFrame(count_rows)
    counts.to_csv(OUT / "sample_counts.csv", index=False)
    design_path = ROOT / "config/listing-reference-sensitivity.txt"
    design = """POST-ANALYSIS sensitivity requested after listing-date audit and primary results.

This is a narrower deposited-reference cohort, not a correction silently replacing
the primary population. Starting with the original-source transparent-eight eligible
pairs, retain an origin only when its stock_code matches the existing public panel
listing-date reference, its date parses successfully, and origin year >= listing
year. Drop unmatched/missing dates and origins before the listing year from BOTH
training and evaluation. No new external listing dataset is acquired. A year-level
screen does not establish point-in-time availability or exclude post-delisting
accounts; the word verified here means screened against this deposited reference,
not an independent official exchange/date audit.

All other primary rules remain: finite raw current/next-year ROE, positive finite
assets for ratios, training-only median imputation and 1/99 clipping, standardizing
and missing flags, raw targets, origin-year known nonfinancial classifications,
expanding training targets strictly before each evaluation target. Origins begin
2010 and target folds are 2015--2020. Every model uses the same restricted rows.
Run historical mean/median, raw current-ROE persistence, and OLS/LAD with linear,
sign-asymmetric and sign-asymmetric-plus-transparent-eight designs. No industry
controls or additional model selection. Primary evaluation remains 2015--2019;
2020 is separate. Pairwise security-cluster bootstrap uses 2,000 draws and seed
20261025, conditional on fitted predictions, including sign groups and 2020.

The comparison with the full original sample changes both listing eligibility and
reference coverage; it cannot isolate only the influence of pre-IPO rows. Its
codes come from the public deposited cohort, but its outcome and predictors remain
the original-source accounting definitions. Do not call it independent replication.

Pre-fit counts:
""" + counts.to_csv(index=False)
    if design_path.exists():
        assert "POST-ANALYSIS" in design_path.read_text()
    else:
        design_path.write_text("Frozen UTC: "+datetime.now(timezone.utc).isoformat()+"\n\n"+design)
    print(counts[counts.partition.isin(["all_eligible", "2015-2019", "2020"])].to_string(index=False), flush=True)
    pred = forecasts(restricted, OUT)
    met = metrics(pred, OUT)
    intervals = uncertainty(pred, OUT, full=True)
    diagnostics(pred, OUT)
    summary = {"status": "post-analysis listing-reference sensitivity", "design_path": str(design_path.relative_to(ROOT)),
        "executing_code_sha256": executing_code_hash,
        "helper_code_sha256": hashlib.sha256((ROOT / "code/analyze_original.py").read_bytes()).hexdigest(),
        "primary_source_predictions_changed": False,
        "sample_counts": count_rows,
        "primary_comparison": next(x for x in intervals if x["partition"] == "2015-2019" and x["reference"] == "lad_piecewise"),
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(met[met.partition.eq("2015-2019")].to_string(index=False), flush=True)
    print(json.dumps(summary["primary_comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
