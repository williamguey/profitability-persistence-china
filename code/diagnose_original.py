"""Post-analysis diagnostic: a train-clipped current-ROE persistence forecast.

This was added after observing the primary results, at the research lead's
request. It reuses the main folds' saved clipping limits; it changes no fitted
model, main benchmark definition, sample, or primary estimand.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from analyze_original import partitions, SEED

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/original-forecast/transparent_main")
    args = parser.parse_args()
    out = args.output
    folder = out / "clipped_persistence_diagnostic"
    folder.mkdir(exist_ok=True)
    p = pd.read_csv(out / "predictions.csv", dtype={"stock_code": str})
    records = json.loads((out / "fit_records.json").read_text())
    thresholds = {r["target_year"]: (r["transform"]["lower"][0], r["transform"]["upper"][0])
        for r in records if r["model"] == "ols_ar"}
    p["clipped_random_walk"] = [np.clip(roe, *thresholds[year]) for roe, year in zip(p.roe_pct, p.target_year)]
    p[["stock_code", "target_year", "roe_pct", "target", "clipped_random_walk"]].to_csv(folder / "predictions.csv", index=False)
    rows = []
    for label, frame in partitions(p):
        e = frame.clipped_random_walk-frame.target
        byfirm = pd.DataFrame({"firm": frame.stock_code, "abs": abs(e)}).groupby("firm").mean()
        rows.append({"partition": label, "model": "clipped_random_walk", "n": len(frame), "firms": frame.stock_code.nunique(),
            "mae": abs(e).mean(), "rmse": np.sqrt(np.mean(e**2)), "median_ae": np.median(abs(e)),
            "bias": np.mean(e), "firm_balanced_mae": byfirm["abs"].mean()})
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    intervals = []
    for label, frame in partitions(p):
        if label not in ["2015-2019", "2020", "negative_origin", "nonnegative_origin"]:
            continue
        for candidate in ["lad_ar", "lad_piecewise", "lad_signals"]:
            temp = frame[["stock_code"]].copy()
            temp["baseline"] = abs(frame.clipped_random_walk-frame.target)
            temp["candidate"] = abs(frame[candidate]-frame.target)
            g = temp.groupby("stock_code").agg(base_sum=("baseline", "sum"), new_sum=("candidate", "sum"), n=("baseline", "size"))
            v = g.to_numpy()
            rng = np.random.default_rng(SEED)
            boot = []
            for _ in range(2000):
                s = v[rng.integers(0, len(v), len(v))].sum(axis=0)
                boot.append([(s[0]-s[1])/s[2], 100*(s[0]-s[1])/s[0]])
            boot = np.asarray(boot)
            delta = temp.baseline.mean()-temp.candidate.mean()
            intervals.append({"partition": label, "reference": "clipped_random_walk", "candidate": candidate,
                "mae_reduction_pp": delta, "relative_reduction_pct": 100*delta/temp.baseline.mean(),
                "ci_low_pp": np.quantile(boot[:, 0], .025), "ci_high_pp": np.quantile(boot[:, 0], .975),
                "ci_low_relative_pct": np.quantile(boot[:, 1], .025), "ci_high_relative_pct": np.quantile(boot[:, 1], .975),
                "bootstrap_draws": 2000, "seed": SEED, "clusters": len(g)})
    pd.DataFrame(intervals).to_csv(folder / "paired_bootstrap.csv", index=False)
    (folder / "provenance.json").write_text(json.dumps({
        "status": "Post-analysis diagnostic requested after primary results were observed",
        "construction": "Current ROE clipped at the existing fold's training-only observed 1st/99th percentiles; no fitting",
        "thresholds_by_target_year": thresholds,
        "primary_predictions_modified": False}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(intervals).query("partition == '2015-2019'").to_string(index=False))


if __name__ == "__main__":
    main()
