"""Descriptive tables for the recovered worksheet; no fitted specification choices."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from analyze_original import TRANSPARENT_DEFINITIONS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/original-forecast")
    args = parser.parse_args()
    out = args.output
    p = pd.read_csv(out / "eligible_pairs.csv", dtype={"stock_code": str})
    main = p[p.target_year.between(2015, 2019)]
    rows = []
    for label, frame in [("all_eligible", p), ("2015-2019", main), ("2020", p[p.target_year.eq(2020)])]:
        for name in ["roe_pct", "target"] + list(TRANSPARENT_DEFINITIONS):
            v = frame[name].dropna()
            rows.append({"partition": label, "variable": name, "observed_n": len(v), "missing_n": len(frame)-len(v),
                "mean": v.mean(), "std": v.std(), "min": v.min(), "q01": v.quantile(.01),
                "q25": v.quantile(.25), "median": v.median(), "q75": v.quantile(.75),
                "q99": v.quantile(.99), "max": v.max()})
    pd.DataFrame(rows).to_csv(out / "descriptive_statistics.csv", index=False)
    transition = []
    for sign, frame in main.groupby("negative_roe"):
        transition.append({"negative_origin": int(sign), "n": len(frame), "firms": frame.stock_code.nunique(),
            "negative_target_n": int(frame.target.lt(0).sum()),
            "negative_target_pct": 100*frame.target.lt(0).mean(),
            "median_origin_roe_pp": frame.roe_pct.median(), "median_target_roe_pp": frame.target.median(),
            "mean_origin_roe_pp": frame.roe_pct.mean(), "mean_target_roe_pp": frame.target.mean()})
    pd.DataFrame(transition).to_csv(out / "sign_transitions.csv", index=False)
    extremes = main.loc[main.target.abs().nlargest(10).index,
        ["stock_code", "company", "year", "target_year", "roe_pct", "target"]]
    extremes.to_csv(out / "largest_absolute_targets.csv", index=False)
    labels = {"roe_pct": "Origin ROE (percentage points)", "target": "Next-year ROE (percentage points)", **TRANSPARENT_DEFINITIONS}
    (out / "descriptive_variable_labels.json").write_text(json.dumps(labels, indent=2))
    records_path = out / "fit_records.json"
    if records_path.exists():
        slopes = []
        for record in json.loads(records_path.read_text()):
            if record["model"] not in ["ols_ar", "lad_ar", "ols_piecewise", "lad_piecewise"]:
                continue
            transform = record["transform"]
            names = transform["retained_columns"]
            index = {name: transform["all_columns"].index(name) for name in names}
            raw = {name: record["beta"][j+1]/transform["std"][index[name]] for j, name in enumerate(names)}
            intercept = record["beta"][0]-sum(raw[name]*transform["mean"][index[name]] for name in names)
            slopes.append({"target_year": record["target_year"], "model": record["model"],
                "nonnegative_intercept_pp": intercept,
                "negative_intercept_pp": intercept+raw.get("negative_roe", 0),
                "nonnegative_slope": raw["roe_pct"],
                "negative_slope": raw["roe_pct"]+raw.get("roe_x_negative", 0),
                "origin_clip_lower_pp": transform["lower"][0], "origin_clip_upper_pp": transform["upper"][0],
                "interpretation": "ROE slopes within training clipping thresholds; slopes are zero outside thresholds"})
        pd.DataFrame(slopes).to_csv(out / "persistence_coefficients.csv", index=False)
    print(pd.DataFrame(transition).to_string(index=False))


if __name__ == "__main__":
    main()
