"""Reshape Peng (2022), Mendeley DOI 10.17632/wx8nwsrrbr.2, without recoding.

The first worksheet is the full supplied sample; other worksheets are portfolios
selected using full-period returns and must not be treated as independent data.
All numeric values and zeros are preserved. Eligibility is left to analysis code.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import re
from openpyxl import load_workbook
from openpyxl.utils.datetime import from_excel

ROOT = Path(__file__).resolve().parents[1]
SOURCE = next((ROOT / "data/raw/public").glob("wx8nwsrrbr__*.xlsx"))
EXPECTED_HASH = "9cb04bd1d5ed44383d7e54351c8e8c7271797ccd74ecac3934e06302141dd75f"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_HASH
wb = load_workbook(SOURCE, read_only=True, data_only=True)
ws = wb["全部数据"]
rows = list(ws.iter_rows(values_only=True))
headers = rows[0]
BLOCKS = {
    "stock_return_pct": 4,
    "sell_margin_pct": 16,
    "profit_growth_pct": 28,
    "income_growth_pct": 40,
    "roe_pct": 52,
    "pledge_ratio_source": 64,
    "market_value_100m_cny": 76,
    "pe_source": 88,
    "debt_ratio_source": 100,
    "dividend_block1_source": 112,
    "research_pct": 124,
    "seasoned_offering_count": 136,
    "dividend_block2_source": 148,
}
schema = {
    "source_doi": "10.17632/wx8nwsrrbr.2", "source_sha256": EXPECTED_HASH,
    "worksheet": ws.title, "transformations": "Wide-to-long reshape only; no numeric transformations or exclusions beyond non-company footer rows.",
    "fields": {},
}
for field, start in BLOCKS.items():
    schema["fields"][field] = {"zero_based_first_column": start, "headers": list(headers[start:start + 12])}
records = []
excluded = []
for excel_row, row in enumerate(rows[1:], 2):
    code = str(row[0])
    if not re.fullmatch(r"\d{6}", code):
        excluded.append({"excel_row": excel_row, "first_cell": row[0]})
        continue
    listing = row[2] if isinstance(row[2], datetime.datetime) else from_excel(row[2], wb.epoch)
    assert isinstance(listing, datetime.datetime), (code, row[2])
    for year in range(2009, 2021):
        record = {"stock_code": code, "stock_name": row[1], "listing_date": listing.date().isoformat(),
                  "year": year, "source_excel_row": excel_row,
                  "listed_by_year_end": int(listing.year <= year),
                  "full_period_stock_return_pct": row[3]}
        for field, start in BLOCKS.items():
            record[field] = row[start + year - 2009]
        records.append(record)
assert len({(r["stock_code"], r["year"]) for r in records}) == len(records)
out = ROOT / "data/processed/peng2022_panel_rawvalues.csv"
out.parent.mkdir(parents=True, exist_ok=True)
(ROOT / "literature").mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]))
    w.writeheader(); w.writerows(records)
schema.update({"records": len(records), "firms": len({r["stock_code"] for r in records}),
               "excluded_footer_rows": excluded})
(ROOT / "literature/peng2022_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2))
print(json.dumps({"panel_path": str(out), "firms": schema["firms"], "records": len(records)}, indent=2))
