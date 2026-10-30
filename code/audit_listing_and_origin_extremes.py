"""Post-analysis audits of listing-date coverage and ex-ante origin extremes.

Reads existing source/prediction artifacts; does not modify them or refit models.
Design frozen in config/listing-and-origin-extreme.txt.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/original-listing-audit"
MODELS = ["lad_ar", "lad_piecewise", "lad_signals"]


def listing_audit():
    original = pd.read_csv(ROOT / "analysis/original-forecast/transparent_main/eligible_pairs.csv", dtype={"stock_code": str})
    public = pd.read_csv(ROOT / "data/processed/peng2022_panel_rawvalues.csv", dtype={"stock_code": str})
    dates = public[["stock_code", "listing_date"]].drop_duplicates()
    assert not dates.stock_code.duplicated().any(), "Listing-date reference has inconsistent dates within code."
    dates["listing_year"] = pd.to_datetime(dates.listing_date, errors="coerce").dt.year
    joined = original.merge(dates, on="stock_code", how="left", validate="many_to_one", indicator=True)
    joined["listing_status"] = np.select([
        joined._merge.eq("left_only"), joined.listing_year.isna(), joined.year.lt(joined.listing_year),
        joined.year.eq(joined.listing_year)],
        ["unmatched_security", "matched_missing_or_invalid_date", "origin_before_listing_year", "origin_in_listing_year"],
        default="origin_after_listing_year")
    partitions = [("all_eligible", joined), ("2015-2019", joined[joined.target_year.between(2015, 2019)]),
        ("2020", joined[joined.target_year.eq(2020)])]
    partitions += [(str(year), joined[joined.target_year.eq(year)]) for year in range(2011, 2020)]
    rows = []
    for partition, frame in partitions:
        for status in ["origin_before_listing_year", "origin_in_listing_year", "origin_after_listing_year",
            "unmatched_security", "matched_missing_or_invalid_date"]:
            sub = frame[frame.listing_status.eq(status)]
            rows.append({"partition": partition, "listing_status": status, "observations": len(sub),
                "securities": sub.stock_code.nunique(), "partition_observations": len(frame),
                "share_partition_pct": 100*len(sub)/len(frame)})
    counts = pd.DataFrame(rows)
    counts.to_csv(OUT / "listing_aggregate_counts.csv", index=False)
    # Securities are public identifiers; selected examples document the mismatch.
    examples = joined[joined.listing_status.eq("origin_before_listing_year")].sort_values(["target_year", "stock_code"])
    prelisting = examples.copy()
    prelisting["origin_minus_listing_year"] = prelisting.year-prelisting.listing_year
    prelisting.groupby("origin_minus_listing_year").agg(n=("stock_code", "size"), securities=("stock_code", "nunique")).to_csv(OUT / "prelisting_gap_counts.csv")
    examples = examples[["stock_code", "company", "year", "target_year", "listing_date", "roe_pct"]].head(12)
    examples.to_csv(OUT / "prelisting_examples.csv", index=False)
    research_sample = pd.read_csv(ROOT / "data/original-private/05_research_sample_cached.csv", header=None)
    sample_text = research_sample.fillna("").astype(str)
    terms = r"(?i)listing|listed|IPO|listing.date|上市|上市日期|上市时间"
    listing_mentions = sample_text.apply(lambda col: col.str.contains(terms, regex=True)).any(axis=1)
    sample_audit = {"rows": len(research_sample), "columns": len(research_sample.columns),
        "first_row": sample_text.iloc[0].tolist(), "second_row": sample_text.iloc[1].tolist(),
        "listing_keyword_rows": np.flatnonzero(listing_mentions).tolist(),
        "last_rows": sample_text.tail(3).values.tolist(),
        "public_reference_security_count": len(dates),
        "public_reference_invalid_listing_dates": int(dates.listing_year.isna().sum()),
        "original_observed_roe_does_not_imply_listed": bool(joined.listing_status.eq("origin_before_listing_year").any())}
    (OUT / "research_sample_sheet_audit.json").write_text(json.dumps(sample_audit, ensure_ascii=False, indent=2))
    return counts, examples, sample_audit


def origin_extreme_audit():
    rows, thresholds = [], []
    sources = [("original", ROOT / "analysis/original-forecast/transparent_main"), ("public", ROOT / "analysis")]
    for source, folder in sources:
        p = pd.read_csv(folder / "predictions.csv", dtype={"stock_code": str})
        fit_records = json.loads((folder / "fit_records.json").read_text())
        bounds = {}
        for record in fit_records:
            if record["model"] != "ols_ar":
                continue
            lo, hi = record["transform"]["lower"][0], record["transform"]["upper"][0]
            bounds[record["target_year"]] = (lo, hi)
            thresholds.append({"source": source, "target_year": record["target_year"],
                "origin_roe_q01": lo, "origin_roe_q99": hi, "n_train": record["n_train"]})
        p["lower"] = p.target_year.map(lambda x: bounds[x][0])
        p["upper"] = p.target_year.map(lambda x: bounds[x][1])
        p["origin_stratum"] = np.where(p.roe_pct.lt(p.lower) | p.roe_pct.gt(p.upper), "outside_training_1_99", "inside_training_1_99")
        for period, frame in [("2015-2019", p[p.target_year.between(2015, 2019)]), ("2020", p[p.target_year.eq(2020)])]:
            for model in MODELS:
                error = frame[model]-frame.target
                totalsse = float(np.sum(error**2))
                totalae = float(np.sum(np.abs(error)))
                for group, sub in frame.groupby("origin_stratum"):
                    e = sub[model]-sub.target
                    sse = float(np.sum(e**2))
                    ae = float(np.sum(np.abs(e)))
                    rows.append({"source": source, "partition": period, "origin_stratum": group,
                        "model": model, "n": len(sub), "securities": sub.stock_code.nunique(),
                        "sample_share_pct": 100*len(sub)/len(frame), "mae": np.abs(e).mean(),
                        "rmse": np.sqrt(np.mean(e**2)), "sse": sse, "sse_share_pct": 100*sse/totalsse,
                        "absolute_error_share_pct": 100*ae/totalae,
                        "negative_origin_n": int(sub.roe_pct.lt(0).sum()),
                        "negative_target_n": int(sub.target.lt(0).sum())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "origin_extreme_metrics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "origin_extreme_training_thresholds.csv", index=False)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    design = ROOT / "config/listing-and-origin-extreme.txt"
    assert design.exists(), "Freeze the post-analysis diagnostic design first."
    listing, examples, sheet = listing_audit()
    extreme = origin_extreme_audit()
    print("LISTING AUDIT")
    print(listing[listing.partition.isin(["all_eligible", "2015-2019", "2020"])].to_string(index=False))
    print("RESEARCH SAMPLE SHEET", json.dumps(sheet, ensure_ascii=False))
    print("PRELISTING EXAMPLES")
    print(examples.to_string(index=False))
    print("ORIGIN EXTREMES")
    print(extreme.to_string(index=False))


if __name__ == "__main__":
    main()
