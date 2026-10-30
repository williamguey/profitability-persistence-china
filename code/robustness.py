"""Frozen denominator sensitivity: refit after removing firm 000982 everywhere.

This is an exclusion sensitivity for a valid but near-zero-equity ROE ratio,
not a data-error correction. Main-model source and primary outputs are untouched.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analyze_forecasts import design, fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
EXCLUDED = "000982"


def main():
    p = pd.read_csv(OUT / "eligible_pairs.csv", dtype={"stock_code": str})
    p = p[p.stock_code.ne(EXCLUDED)].copy()
    predictions, records = [], []
    for year in range(2015, 2021):
        train = p[p.target_year.lt(year)].copy()
        test = p[p.target_year.eq(year)].copy()
        assert train.target_year.max() == year - 1
        assert test.year.eq(year - 1).all()
        assert not train.stock_code.eq(EXCLUDED).any()
        row = test[["stock_code", "stock_name", "year", "target_year", "roe_pct", "target", "negative_roe"]].copy()
        y = train.target.to_numpy()
        for spec in ["piecewise", "signals"]:
            x, z, transform = design(train, test, spec)
            for loss in ["ols", "lad"]:
                model = f"{loss}_{spec}"
                beta, diagnostic = fit(x, y, loss)
                row[model] = z @ beta
                records.append({"target_year": year, "model": model, "excluded_firm": EXCLUDED,
                    "n_train": len(train), "n_test": len(test), "beta": beta.tolist(),
                    "transform": transform, "diagnostic": diagnostic})
        predictions.append(row)
        print(f"Exclusion refit completed target {year}: training={len(train)}, test={len(test)}", flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(OUT / "robustness_refit_predictions.csv", index=False)
    (OUT / "robustness_refit_records.json").write_text(json.dumps(records, indent=2))
    rows = []
    partitions = [(str(year), pred[pred.target_year.eq(year)]) for year in range(2015, 2021)]
    partitions.append(("2015-2019", pred[pred.target_year.le(2019)]))
    for partition, frame in partitions:
        for model in ["ols_piecewise", "ols_signals", "lad_piecewise", "lad_signals"]:
            error = frame[model] - frame.target
            rows.append({"partition": partition, "model": model, "excluded_firm": EXCLUDED,
                "n": len(frame), "firms": frame.stock_code.nunique(),
                "mae": np.abs(error).mean(), "rmse": np.sqrt(np.mean(error**2)),
                "median_ae": np.median(np.abs(error)), "bias": error.mean()})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "robustness_refit.csv", index=False)
    report = {"excluded_firm": EXCLUDED, "scope": "Excluded from every training and evaluation fold before preprocessing",
        "reason": "Near-zero average equity audit; exclusion selected before forecast comparisons",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "main_model_code_sha256": hashlib.sha256((ROOT / "code/analyze_forecasts.py").read_bytes()).hexdigest(),
        "pooled_metrics": metrics[metrics.partition.eq("2015-2019")].to_dict("records")}
    (OUT / "robustness_refit_summary.json").write_text(json.dumps(report, indent=2))
    print(metrics[metrics.partition.eq("2015-2019")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
