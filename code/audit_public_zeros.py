"""Aggregate-only provenance audit of deposited public-panel numerical zeros."""
from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/public-factorial"
FIELDS = {"roe_pct": 52, "sell_margin_pct": 16, "income_growth_pct": 40, "profit_growth_pct": 28}


def main():
    panel = pd.read_csv(ROOT / "data/processed/peng2022_panel_rawvalues.csv", dtype={"stock_code": str})
    listed = panel[panel.listed_by_year_end.eq(1)].copy()
    source = next((ROOT / "data/raw/public").glob("wx8nwsrrbr__*.xlsx"))
    requested = {}
    result = {}
    for field, zero_based_start in FIELDS.items():
        z = listed[listed[field].eq(0)]
        counts = z.groupby("stock_code").size()
        group_years = z.groupby("stock_code").year.apply(lambda a: sorted(map(int, a)))
        result[field] = {
            "zero_rows": len(z), "securities_with_zeros": len(counts),
            "zero_rows_by_source_year": {str(k): int(v) for k, v in z.groupby("year").size().items()},
            "securities_by_number_of_zero_years": {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
            "securities_whose_zero_spell_is_consecutive": int(sum(ys == list(range(ys[0], ys[-1]+1)) for ys in group_years)),
            "securities_whose_first_zero_is_2009": int(sum(ys[0] == 2009 for ys in group_years)),
            "negative_debt_ratio_source_at_zero": int(z.debt_ratio_source.lt(0).sum()),
            "zero_profit_growth_at_zero": int(z.profit_growth_pct.eq(0).sum()),
            "nonzero_profit_growth_at_zero": int(z.profit_growth_pct.ne(0).sum()),
            "zero_revenue_growth_at_zero": int(z.income_growth_pct.eq(0).sum()),
            "zero_pe_source_at_zero": int(z.pe_source.eq(0).sum()),
        }
        for _, r in z.iterrows():
            requested[(int(r.source_excel_row), zero_based_start + int(r.year) - 2009 + 1)] = field
    wb = load_workbook(source, read_only=True, data_only=False)
    ws = wb["全部数据"]
    raw_counts = {f: Counter() for f in FIELDS}
    found = 0
    for row_number, row in enumerate(ws.iter_rows(), 1):
        for column_number, cell in enumerate(row, 1):
            field = requested.get((row_number, column_number))
            if field is not None:
                raw_counts[field][(cell.data_type, repr(cell.value))] += 1
                found += 1
    wb.close()
    assert found == len(requested)
    for field in FIELDS:
        result[field]["raw_xlsx_cell_type_value_counts"] = [
            {"cell_type": k[0], "raw_value_repr": k[1], "count": v} for k, v in raw_counts[field].items()
        ]
        assert sum(raw_counts[field].values()) == result[field]["zero_rows"]
    result["provenance"] = {
        "source_doi": "10.17632/wx8nwsrrbr.2", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "worksheet": "全部数据", "raw_cell_read_mode": "openpyxl read_only=True, data_only=False",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Numerical zero patterns only; no value is recoded and no new forecast is fitted.",
        "interpretation": "ROE zeros occur in initial, consecutive 2009--2013 spells, with nonzero profit-growth values and frequently negative source debt ratios. This raises a missing-value or measurement-convention concern but cannot establish the generating rule. Persistent sales-margin zeros include company names identifying financial institutions, consistent with inapplicable gross-margin fields; this is a name-based observation, not a verified sector classification. Numeric XLSX zeros provide no separate missing-value flags."
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "zero_provenance.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
