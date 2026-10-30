"""Aggregate saved public-forecast sign decomposition and cross-source ROE gaps."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/public-sign-constructs"
CELLS = ["lad_ar", "lad_linear_signals", "lad_piecewise", "lad_signals"]
CONTRASTS = {
    "accounting_first": np.array([1., -1., 0., 0.]),
    "asymmetry_second": np.array([0., 1., 0., -1.]),
    "asymmetry_first": np.array([1., 0., -1., 0.]),
    "accounting_second": np.array([0., 0., 1., -1.]),
    "average_accounting": np.array([.5, -.5, .5, -.5]),
    "average_asymmetry": np.array([.5, .5, -.5, -.5]),
}
SEED, DRAWS = 20261025, 2000
PUBLIC_PRED = ROOT / "analysis/public-factorial/predictions.csv"
ORIGINAL_PRED = ROOT / "analysis/original-forecast/transparent_main/predictions.csv"
PUBLIC_PANEL = ROOT / "data/processed/peng2022_panel_rawvalues.csv"
ORIGINAL_PANEL = ROOT / "data/original-private/original_panel.csv"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def interval(a, prefix):
    return {prefix + "_ci_low": float(np.quantile(a, .025)), prefix + "_ci_high": float(np.quantile(a, .975))}


def sign_groups(public):
    frame = public[public.target_year.between(2015, 2019)].copy()
    codes = sorted(frame.stock_code.unique())
    index = {code: j for j, code in enumerate(codes)}
    # Security x origin-sign-group x [four absolute-error sums, observation count].
    values = np.zeros((len(codes), 2, 5))
    for g, mask in enumerate([frame.roe_pct.lt(0), frame.roe_pct.ge(0)]):
        subset = frame[mask]
        t = pd.DataFrame(np.abs(subset[CELLS].to_numpy() - subset.target.to_numpy()[:, None]), columns=CELLS)
        t["n"] = 1
        t["stock_code"] = subset.stock_code.to_numpy()
        totals = t.groupby("stock_code")[CELLS + ["n"]].sum()
        for code, row in totals.iterrows():
            values[index[code], g] = row.to_numpy()
    total = values.sum(axis=0)
    rng = np.random.default_rng(SEED)
    boot = np.stack([values[rng.integers(0, len(codes), len(codes))].sum(axis=0) for _ in range(DRAWS)])
    assert (boot[:, :, 4] > 0).all()
    n_all = total[:, 4].sum()
    boot_n_all = boot[:, :, 4].sum(axis=1)
    mae_all = total[:, :4].sum(axis=0) / n_all
    boot_mae_all = boot[:, :, :4].sum(axis=1) / boot_n_all[:, None]
    metrics, comparisons = [], []
    for g, label in enumerate(["negative_origin", "nonnegative_origin"]):
        n = total[g, 4]
        n_codes = int((values[:, g, 4] > 0).sum())
        mae = total[g, :4] / n
        boot_mae = boot[:, g, :4] / boot[:, g, 4, None]
        for j, model in enumerate(CELLS):
            share = 100*total[g, j] / total[:, j].sum()
            share_boot = 100*boot[:, g, j] / boot[:, :, j].sum(axis=1)
            metrics.append({"partition": "2015-2019", "origin_sign": label, "model": model,
                            "n": int(n), "securities": n_codes, "overall_n": int(n_all),
                            "observation_share_pct": 100*n/n_all, "absolute_error_sum_pp": total[g, j],
                            "mae_pp": mae[j], "share_total_absolute_error_pct": share,
                            **interval(boot_mae[:, j], "mae_pp"), **interval(share_boot, "share_total_absolute_error_pct")})
        for name, weight in CONTRASTS.items():
            group_gain = float(mae @ weight)
            group_boot = boot_mae @ weight
            overall_gain = float(mae_all @ weight)
            overall_boot = boot_mae_all @ weight
            contribution = float(total[g, :4] @ weight / n_all)
            contribution_boot = (boot[:, g, :4] @ weight) / boot_n_all
            share_boot = 100*contribution_boot / overall_boot
            comparisons.append({"partition": "2015-2019", "origin_sign": label, "contrast": name,
                                "n": int(n), "securities": n_codes, "group_mae_gain_pp": group_gain,
                                "overall_mae_gain_pp": overall_gain, "weighted_contribution_pp": contribution,
                                "share_overall_gain_pct": 100*contribution/overall_gain,
                                **interval(group_boot, "group_mae_gain_pp"),
                                **interval(contribution_boot, "weighted_contribution_pp"),
                                **interval(share_boot, "share_overall_gain_pct"),
                                "bootstrap_draws": DRAWS, "seed": SEED})
    metrics = pd.DataFrame(metrics)
    comparisons = pd.DataFrame(comparisons)
    for name, g in comparisons.groupby("contrast"):
        np.testing.assert_allclose(g.weighted_contribution_pp.sum(), g.overall_mae_gain_pp.iloc[0], atol=1e-12, rtol=0)
    prior = pd.read_csv(ROOT / "analysis/public-factorial/paired_bootstrap.csv")
    prior = prior[prior.partition.eq("2015-2019")].set_index("contrast")
    for name, weight in CONTRASTS.items():
        np.testing.assert_allclose(mae_all @ weight, prior.loc[name, "mae_reduction_pp"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(np.quantile(boot_mae_all @ weight, [.025, .975]),
                                   prior.loc[name, ["ci_low_pp", "ci_high_pp"]].to_numpy(float), atol=1e-10, rtol=0)
    metrics.to_csv(OUT / "public_origin_sign_metrics.csv", index=False)
    comparisons.to_csv(OUT / "public_origin_sign_contrasts.csv", index=False)
    return {"overall_n": int(n_all), "overall_securities": len(codes),
            "pooled_factorial_metrics_and_intervals_reproduced": True}


def describe(values, variable):
    q = np.quantile(values, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {"variable": variable, "n": len(values), "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)), **dict(zip(["min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max"], q))}


def construct_comparison(public):
    a = pd.read_csv(ORIGINAL_PRED, dtype={"stock_code": str})
    a = a[a.target_year.between(2015, 2019)].copy()
    b = public[public.target_year.between(2015, 2019)].copy()
    key = ["stock_code", "target_year"]
    assert not a.duplicated(key).any() and not b.duplicated(key).any()
    merged = a[key + ["target"]].merge(b[key + ["target"]], on=key, how="outer", validate="one_to_one", suffixes=("_A", "_B"), indicator=True)
    counts = []
    for name, subset in [("source_A_main", a), ("source_B_main", b)]:
        counts.append({"sample": name, "n": len(subset), "securities": subset.stock_code.nunique()})
    for status in ["both", "left_only", "right_only"]:
        subset = merged[merged._merge.eq(status)]
        counts.append({"sample": {"both": "matched", "left_only": "A_only", "right_only": "B_only"}[status],
                       "n": len(subset), "securities": subset.stock_code.nunique()})
    common = merged[merged._merge.eq("both")].copy()
    pa = pd.read_csv(ORIGINAL_PANEL, dtype={"stock_code": str})
    pb = pd.read_csv(PUBLIC_PANEL, dtype={"stock_code": str})
    assert not pa.duplicated(["stock_code", "year"]).any()
    assert not pb.duplicated(["stock_code", "year"]).any()
    check = common.merge(pa[["stock_code", "year", "roe_decimal"]].rename(columns={"year": "target_year"}), on=key, validate="one_to_one")
    check = check.merge(pb[["stock_code", "year", "roe_pct"]].rename(columns={"year": "target_year"}), on=key, validate="one_to_one")
    assert len(check) == len(common)
    np.testing.assert_allclose(check.target_A, 100*check.roe_decimal, rtol=0, atol=1e-10)
    np.testing.assert_allclose(check.target_B, check.roe_pct, rtol=0, atol=1e-10)
    diff = common.target_A.to_numpy() - common.target_B.to_numpy()
    absolute = np.abs(diff)
    pd.DataFrame([describe(diff, "signed_A_minus_B_pp"), describe(absolute, "absolute_A_minus_B_pp")]).to_csv(OUT / "matched_roe_difference_distribution.csv", index=False)
    tolerances = []
    for t in [1e-10, .0001, .01, .1, 1., 10., 100.]:
        within = int((absolute <= t + (1e-10 if t > 1e-10 else 0)).sum())
        tolerances.append({"absolute_tolerance_pp": t, "within_n": within, "within_pct": 100*within/len(common),
                           "beyond_n": len(common)-within, "beyond_pct": 100*(len(common)-within)/len(common), "n": len(common)})
    pd.DataFrame(tolerances).to_csv(OUT / "matched_roe_difference_tolerances.csv", index=False)
    signs = pd.crosstab(np.sign(common.target_A), np.sign(common.target_B)).stack().reset_index()
    signs.columns = ["source_A_target_sign", "source_B_target_sign", "n"]
    signs["pct_matched"] = 100*signs.n/len(common)
    signs.to_csv(OUT / "matched_roe_sign_counts.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "matched_roe_coverage.csv", index=False)
    # Existing six deliberately selected annual-report cases: verify parent construct and coverage only.
    cases = pd.read_csv(ROOT / "source/construct-check/primary-accounting-inputs.csv", dtype={"stock_code": str})
    cases["reconstructed_parent_roe_pct"] = 100*cases.net_income_attributable / ((cases.opening_equity_attributable+cases.closing_equity_attributable)/2)
    case_public = cases.merge(pb[["stock_code", "year", "roe_pct"]], on=["stock_code", "year"], validate="one_to_one")
    assert len(case_public) == 6
    max_case_gap = float(np.abs(case_public.reconstructed_parent_roe_pct-case_public.roe_pct).max())
    assert max_case_gap <= .00005
    common_keys = set(map(tuple, common[key].to_numpy()))
    case_matched = sum((r.stock_code, r.year) in common_keys for r in cases.itertuples())
    return {"matched_rows": len(common), "matched_securities": int(common.stock_code.nunique()),
            "matches_use_exact_security_code_and_target_year": True, "targets_verified_against_raw_source_units": True,
            "annual_report_sanity_checks": {"public_parent_construct_cases": len(case_public),
                                            "max_abs_reconstructed_minus_deposited_pp": max_case_gap,
                                            "cases_in_actual_matched_main_sample": int(case_matched),
                                            "cases_outside_actual_matched_main_sample": int(len(cases)-case_matched)},
            "scope": "Descriptive differences between source ROE constructions; not validation against a common target or evidence that large gaps are errors."}


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    inputs = [PUBLIC_PRED, ORIGINAL_PRED, PUBLIC_PANEL, ORIGINAL_PANEL,
              ROOT / "analysis/public-factorial/paired_bootstrap.csv"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    public = pd.read_csv(PUBLIC_PRED, dtype={"stock_code": str})
    sign = sign_groups(public)
    constructs = construct_comparison(public)
    after = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    assert before == after
    summary = {"sign_decomposition": sign, "construct_comparison": constructs,
               "inputs_unchanged": True, "input_sha256": before, "code_sha256": sha(Path(__file__)),
               "plan_sha256": sha(ROOT / "config/public-sign-constructs.txt"),
               "scope": "Aggregate-only diagnostics; no model refitting, source recoding or row-level output."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
