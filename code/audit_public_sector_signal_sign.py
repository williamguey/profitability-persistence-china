"""Aggregate source-sector audit of public zeros and raw primary signal signs."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/public-sector-signal-sign"
ORIGINAL = ROOT / "data/original-private/original_panel.csv"
PUBLIC = ROOT / "data/processed/peng2022_panel_rawvalues.csv"
FINANCIAL = {"Capital Market Services", "Insurance", "monetary financial services", "Other financial industries"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def public_metadata():
    source = next((ROOT / "data/raw/public").glob("wx8nwsrrbr__*.xlsx"))
    workbook = load_workbook(source, read_only=True, data_only=False)
    sheets = []
    for ws in workbook.worksheets:
        header = next(ws.iter_rows(max_row=1, values_only=True))
        matches = [str(h) for h in header if h is not None and re.search(r"industry|sector|行业|产业", str(h), re.I)]
        sheets.append({"worksheet": ws.title, "column_count": len(header), "sector_header_matches": matches,
                       "first_four_headers": [str(h) if h is not None else None for h in header[:4]]})
    workbook.close()
    result = {"source_sha256": sha(source), "worksheets": sheets,
              "explicit_sector_field_found_in_scanned_headers": any(r["sector_header_matches"] for r in sheets),
              "inspection_scope": "Worksheet names and first-row column headers; no name-based sector inference.",
              "external_classification": "Original accounting workbook industry, matched on exact security code and origin year; unknowns retained."}
    (OUT / "public_sector_metadata.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def public_zeros(original, public):
    pred = pd.read_csv(ROOT / "analysis/predictions.csv", dtype={"stock_code": str})
    pred = pred[pred.target_year.between(2015, 2019)]
    key = ["stock_code", "year"]
    assert not pred.duplicated(key).any()
    joined = pred[key + ["target_year"]].merge(public[key + ["sell_margin_pct"]], on=key, how="left", validate="one_to_one")
    assert len(joined) == 13970 and joined.sell_margin_pct.notna().all()
    joined = joined.merge(original[key + ["industry"]], on=key, how="left", validate="one_to_one", indicator=True)
    known = joined.industry.notna()
    joined["industry_status"] = np.select(
        [joined._merge.ne("both"), ~known, joined.industry.isin(FINANCIAL)],
        ["unknown_unmatched_code_year", "unknown_missing_industry", "financial"], default="nonfinancial")
    joined["sector"] = joined.industry_status.where(joined.industry_status.isin(["financial", "nonfinancial"]), "unknown")
    assert joined.sell_margin_pct.eq(0).sum() == 274
    rows, details = [], []
    for period, frame in [("2015-2019", joined)] + [(str(y), joined[joined.target_year.eq(y)]) for y in range(2015, 2020)]:
        for status in ["financial", "nonfinancial", "unknown_missing_industry", "unknown_unmatched_code_year"]:
            subset = frame[frame.industry_status.eq(status)]
            zero = subset[subset.sell_margin_pct.eq(0)]
            rows.append({"partition": period, "industry_status": status, "n": len(subset),
                         "securities": subset.stock_code.nunique(), "zero_margin_n": len(zero),
                         "zero_margin_securities": zero.stock_code.nunique(),
                         "zero_margin_pct_of_status": 100*len(zero)/len(subset) if len(subset) else None,
                         "share_of_all_zero_margins_pct": 100*len(zero)/frame.sell_margin_pct.eq(0).sum(),
                         "all_n": len(frame)})
        for (sector, industry), subset in frame[frame.sell_margin_pct.eq(0)].assign(
                industry=frame.industry.fillna("[unavailable]")).groupby(["sector", "industry"], dropna=False):
            details.append({"partition": period, "sector": sector, "original_origin_industry": industry,
                            "zero_margin_n": len(subset), "securities": subset.stock_code.nunique()})
    pd.DataFrame(rows).to_csv(OUT / "public_margin_zero_by_industry_status.csv", index=False)
    pd.DataFrame(details).to_csv(OUT / "public_margin_zero_by_original_industry.csv", index=False)
    return {"main_evaluation_origins": len(joined), "exact_zero_margin_origins": int(joined.sell_margin_pct.eq(0).sum()),
            "classification_key": "security code and origin year", "public_primary_sample_changed": False}


def signal_signs(original):
    key = ["stock_code", "year"]
    scopes = [
        ("main_evaluation_2015_2019", ROOT / "analysis/original-forecast/transparent_main/predictions.csv"),
        ("all_eligible_origins", ROOT / "analysis/original-forecast/transparent_main/eligible_pairs.csv")]
    counts, summaries = [], []
    for scope, path in scopes:
        frame = pd.read_csv(path, dtype={"stock_code": str})
        if scope == "main_evaluation_2015_2019":
            frame = frame[frame.target_year.between(2015, 2019)]
        frame = frame[key + ["roe_pct", "target_year"]].merge(
            original[key + ["roe_decimal", "total_assets", "net_profit", "operating_profit"]], on=key, how="left", validate="one_to_one")
        np.testing.assert_allclose(frame.roe_pct, 100*frame.roe_decimal, atol=1e-10, rtol=0)
        for label, numerator in [("net_profit_over_assets", "net_profit"), ("operating_profit_over_assets", "operating_profit")]:
            valid = np.isfinite(frame.total_assets) & frame.total_assets.gt(0) & np.isfinite(frame[numerator])
            ratio = frame[numerator] / frame.total_assets.where(valid)
            for group, selection in [("all", np.ones(len(frame), dtype=bool)),
                                     ("negative_roe", frame.roe_pct.lt(0)), ("nonnegative_roe", frame.roe_pct.ge(0))]:
                use = valid & selection
                same = np.sign(ratio[use]).eq(np.sign(frame.loc[use, "roe_pct"]))
                summaries.append({"scope": scope, "signal": label, "roe_group": group,
                                  "group_n": int(np.sum(selection)), "finite_positive_assets_ratio_n": int(use.sum()),
                                  "missing_ratio_n": int(np.sum(selection)-use.sum()), "same_sign_n": int(same.sum()),
                                  "different_sign_n": int((~same).sum()), "same_sign_pct": 100*same.mean() if len(same) else None})
            for rs in [-1, 0, 1]:
                for xs in [-1, 0, 1]:
                    mask = valid & np.sign(frame.roe_pct).eq(rs) & np.sign(ratio).eq(xs)
                    counts.append({"scope": scope, "signal": label, "roe_sign": rs, "raw_ratio_sign": xs,
                                   "n": int(mask.sum()), "securities": int(frame.loc[mask, "stock_code"].nunique())})
    pd.DataFrame(counts).to_csv(OUT / "primary_signal_roe_sign_crosstab.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "primary_signal_roe_sign_summary.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ORIGINAL, PUBLIC, ROOT / "analysis/predictions.csv",
             ROOT / "analysis/original-forecast/transparent_main/predictions.csv",
             ROOT / "analysis/original-forecast/transparent_main/eligible_pairs.csv"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    original = pd.read_csv(ORIGINAL, dtype={"stock_code": str})
    public = pd.read_csv(PUBLIC, dtype={"stock_code": str})
    assert not original.duplicated(["stock_code", "year"]).any()
    assert not public.duplicated(["stock_code", "year"]).any()
    metadata = public_metadata()
    zero = public_zeros(original, public)
    signal_signs(original)
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    result = {"public_zero_audit": zero, "source_input_sha256": hashes, "inputs_unchanged": True,
              "plan_sha256": sha(ROOT / "config/public-sector-signal-sign.txt"), "code_sha256": sha(Path(__file__)),
              "interpretation": "Descriptive exact-year industry matching and raw sign relationships. Unknown sectors retained. No inference from company names, no exclusions, no refitting, and no claim that a linear ratio reproduces an ROE sign kink."}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
