"""Targeted accounting-construct checks; no edits to the deposited panel.

Six deliberately selected firm-years corroborate candidate formulas. These
checks do not assert that every observation or vendor field follows them.
Input quantities were visually checked against company-authored annual reports.
"""
from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "source" / "construct-check"
PANEL = ROOT / "data" / "processed" / "peng2022_panel_rawvalues.csv"
TOLERANCE = Decimal("0.00005")  # half the last displayed 4-decimal unit


def run() -> None:
    with PANEL.open(newline="", encoding="utf-8-sig") as handle:
        panel = {(r["stock_code"], r["year"]): r for r in csv.DictReader(handle)}
    with (EVIDENCE / "primary-accounting-inputs.csv").open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    results = []
    for row in cases:
        deposited = panel[(row["stock_code"], row["year"])]
        profit = Decimal(row["net_income_attributable"])
        opening = Decimal(row["opening_equity_attributable"])
        closing = Decimal(row["closing_equity_attributable"])
        liabilities = Decimal(row["total_liabilities"])
        average = (opening + closing) / Decimal(2)
        roe = Decimal(100) * profit / average
        debt = liabilities / closing
        roe_source = Decimal(deposited["roe_pct"])
        debt_source = Decimal(deposited["debt_ratio_source"])
        roe_error = abs(roe - roe_source)
        debt_error = abs(debt - debt_source)
        assert roe_error <= TOLERANCE, (row["stock_code"], row["year"], "ROE", roe_error)
        assert debt_error <= TOLERANCE, (row["stock_code"], row["year"], "debt", debt_error)
        results.append({
            "stock_code": row["stock_code"], "company": row["company"], "year": row["year"],
            "monetary_unit": row["monetary_unit"], "mean_equity_attributable": str(average),
            "roe_reconstructed_pct": str(roe), "roe_deposited_pct": str(roe_source),
            "roe_absolute_error_pct": str(roe_error),
            "debt_reconstructed": str(debt), "debt_deposited": str(debt_source),
            "debt_absolute_error": str(debt_error),
            "reported_weighted_roe_pct": row["reported_weighted_roe_pct"],
            "roe_matches_to_four_decimals": f"{roe:.4f}" == f"{roe_source:.4f}",
            "debt_matches_to_four_decimals": f"{debt:.4f}" == f"{debt_source:.4f}",
            "report_file": row["report_file"], "roe_pdf_pages": row["roe_pdf_pages"],
            "debt_pdf_pages": row["debt_pdf_pages"],
        })
    target = EVIDENCE / "construct-check-results.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(EVIDENCE.glob("*annual-report.pdf"))}
    (EVIDENCE / "primary-report-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Verified {len(results)} firm-years: all 12 constructed ratios match within rounding.")
    print("Candidate ROE formula: 100 * attributable net income / mean(opening, closing attributable equity).")
    print("Candidate debt formula: total liabilities / closing attributable equity.")
    print("Scope: targeted corroboration; not a full-panel audit or an authoritative vendor codebook.")


if __name__ == "__main__":
    run()
