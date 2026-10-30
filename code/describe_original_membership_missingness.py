"""Post-analysis source-membership, size, denominator and availability descriptions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

from describe_original_accounting_lines_identity import REDUCED, FULL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/original-forecast/membership_missingness_diagnostic"
DESIGN = ROOT / "config/original-membership-missingness.txt"
STATUSES = ["matched_at_or_after_listing", "prelisting", "unmatched"]
FINANCIAL = {"Capital Market Services", "Insurance", "monetary financial services", "Other financial industries"}
KEY = ["stock_code", "year"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign(x):
    return np.select([x.isna(), np.isinf(x), x.lt(0), x.eq(0)],
                     ["missing", "nonfinite", "negative", "zero"], default="positive")


def provenance(frame, dates):
    d = frame.merge(dates, on="stock_code", how="left", validate="many_to_one")
    d["provenance"] = np.select([d.listing_year.isna(), d.year.lt(d.listing_year)],
                               ["unmatched", "prelisting"], default="matched_at_or_after_listing")
    return d


def load():
    raw = pd.read_csv(ROOT / "data/original-private/original_panel.csv", dtype={"stock_code": str})
    public = pd.read_csv(ROOT / "data/processed/peng2022_panel_rawvalues.csv", dtype={"stock_code": str})
    dates = public[["stock_code", "listing_date"]].drop_duplicates()
    assert not dates.stock_code.duplicated().any()
    dates["listing_year"] = pd.to_datetime(dates.listing_date, errors="coerce").dt.year
    assert dates.listing_year.notna().all()
    assert not raw.duplicated(KEY).any() and raw.stock_code.str.fullmatch(r"[0-9]{6}").all()
    raw = provenance(raw, dates[["stock_code", "listing_year"]])
    pred = pd.read_csv(ROOT / "analysis/original-forecast/transparent_main/predictions.csv", dtype={"stock_code": str})
    pred = pred[pred.target_year.between(2015, 2019)]
    main = pred[KEY+["target_year", "roe_pct", "target"]].merge(raw, on=KEY, how="left", validate="one_to_one", indicator=True)
    assert main._merge.eq("both").all() and len(main) == 15258
    main = main.drop(columns="_merge")
    np.testing.assert_allclose(main.roe_pct, 100*main.roe_decimal, rtol=0, atol=1e-10)
    records = json.loads((ROOT / "analysis/original-forecast/transparent_main/fit_records.json").read_text())
    bounds = {r["target_year"]: (r["transform"]["lower"][0], r["transform"]["upper"][0])
              for r in records if r["model"] == "ols_ar"}
    old = pd.read_csv(ROOT / "analysis/original-listing-audit/origin_extreme_training_thresholds.csv")
    for _, r in old[old.source.eq("original")].iterrows():
        np.testing.assert_allclose(bounds[int(r.target_year)], [r.origin_roe_q01, r.origin_roe_q99], atol=1e-10, rtol=0)
    main["tail_low"] = main.target_year.map(lambda y: bounds[y][0])
    main["tail_high"] = main.target_year.map(lambda y: bounds[y][1])
    main["origin_tail_flag"] = main.roe_pct.lt(main.tail_low) | main.roe_pct.gt(main.tail_high)
    main["origin_sign"] = np.where(main.roe_pct.lt(0), "negative", "nonnegative")
    a = main.total_assets.where(np.isfinite(main.total_assets) & main.total_assets.gt(0))
    valid = np.isfinite(main.net_profit) & np.isfinite(main.roe_pct) & main.roe_pct.ne(0) & a.notna()
    implied = ((main.net_profit/a)/(main.roe_pct/100)).where(valid)
    implied = implied.where(np.isfinite(implied))
    alternative = (main.net_profit/(main.roe_decimal*a)).where(valid)
    np.testing.assert_allclose(implied, alternative, rtol=1e-12, atol=1e-12, equal_nan=True)
    main["positive_total_assets"] = a
    main["implied_average_equity_over_assets"] = implied
    assert main.origin_tail_flag.sum() == 365
    return raw, main


def groups(main):
    yield "overall", "all", "all", main
    for status in STATUSES:
        yield "provenance", status, "all", main[main.provenance.eq(status)]
    for state in ["negative", "nonnegative"]:
        yield "origin_sign", "all", state, main[main.origin_sign.eq(state)]
    for status, state in itertools.product(STATUSES, ["negative", "nonnegative"]):
        yield "provenance_x_origin_sign", status, state, main[main.provenance.eq(status) & main.origin_sign.eq(state)]


def main_descriptions(main):
    summary, distribution, prefixes = [], [], []
    partitions = [("2015-2019", main)]+[(str(y), main[main.target_year.eq(y)]) for y in range(2015, 2020)]
    for partition, part in partitions:
        for scope, status, state, sub in groups(part):
            keys = {"partition": partition, "group_scope": scope, "provenance": status, "origin_sign": state}
            n = len(sub)
            summary.append({**keys, "n": n, "securities": sub.stock_code.nunique(),
                "negative_origin_n": int(sub.roe_pct.lt(0).sum()), "negative_origin_pct": 100*sub.roe_pct.lt(0).mean() if n else None,
                "origin_training_tail_n": int(sub.origin_tail_flag.sum()), "origin_training_tail_pct": 100*sub.origin_tail_flag.mean() if n else None,
                "implied_equity_ratio_finite_n": int(np.isfinite(sub.implied_average_equity_over_assets).sum()),
                "implied_equity_ratio_nonpositive_n": int(sub.implied_average_equity_over_assets.le(0).sum()),
                "unavailable_reference_listing_date_n": int(sub.listing_year.isna().sum())})
            for variable in ["positive_total_assets", "implied_average_equity_over_assets"]:
                finite = sub.loc[np.isfinite(sub[variable]), variable]
                quantiles = finite.quantile([.01,.1,.25,.5,.75,.9,.99]) if len(finite) else pd.Series(np.nan,index=[.01,.1,.25,.5,.75,.9,.99])
                distribution.append({**keys, "variable": variable, "group_n": n, "finite_n": len(finite), "missing_n": n-len(finite),
                    "mean": finite.mean(), "minimum": finite.min(), "q01": quantiles.loc[.01], "q10": quantiles.loc[.1],
                    "q25": quantiles.loc[.25], "median": quantiles.loc[.5], "q75": quantiles.loc[.75],
                    "q90": quantiles.loc[.9], "q99": quantiles.loc[.99], "maximum": finite.max()})
            if partition == "2015-2019" and scope in ["overall", "provenance"]:
                for prefix, frame in sub.groupby(sub.stock_code.str[:3]):
                    prefixes.append({**keys, "code_prefix_3": prefix, "n": len(frame), "securities": frame.stock_code.nunique(),
                                     "negative_origin_n": int(frame.roe_pct.lt(0).sum()), "origin_training_tail_n": int(frame.origin_tail_flag.sum())})
    pd.DataFrame(summary).to_csv(OUT / "main_group_counts.csv", index=False)
    pd.DataFrame(distribution).to_csv(OUT / "main_size_implied_equity_distributions.csv", index=False)
    pd.DataFrame(prefixes).to_csv(OUT / "main_code_prefix_counts.csv", index=False)
    return pd.DataFrame(summary), pd.DataFrame(distribution)


def unmatched_history(raw, main):
    codes = set(main.loc[main.provenance.eq("unmatched"), "stock_code"])
    assert len(codes) == 347
    history = raw[raw.stock_code.isin(codes)].groupby("stock_code").agg(first_export_year=("year", "min"),last_export_year=("year", "max"))
    rows = []
    for column in ["first_export_year", "last_export_year"]:
        for year,n in history.groupby(column).size().items():
            rows.append({"field": column, "year": int(year), "securities": int(n), "share_of_347_pct":100*n/len(history)})
    pd.DataFrame(rows).to_csv(OUT / "unmatched_export_year_counts.csv", index=False)
    joint = history.groupby(["first_export_year", "last_export_year"]).size().rename("securities").reset_index()
    joint.to_csv(OUT / "unmatched_first_last_year_counts.csv", index=False)
    return {"unmatched_codes":len(history), "first_export_2010_n":int(history.first_export_year.eq(2010).sum()),
            "first_export_after_2010_n":int(history.first_export_year.gt(2010).sum()),
            "first_2010_and_last_2020_n":int((history.first_export_year.eq(2010)&history.last_export_year.eq(2020)).sum()),
            "last_export_2020_n":int(history.last_export_year.eq(2020).sum()), "listing_dates_known_in_available_reference":0,
            "meaning":"First and last exported year only; unmatched IPO dates and legal listing status remain unidentified."}


def identity(raw, main):
    mappings, profiles = [], []
    for scope, frame in [("all_source",raw),("finite_roe_source",raw[np.isfinite(raw.roe_decimal)]),("main_2015_2019_origins",main)]:
        bycode=frame.groupby("stock_code").company.nunique(dropna=True)
        byname=frame.dropna(subset=["company"]).groupby("company").stock_code.nunique()
        sameyear=frame.dropna(subset=["company"]).groupby(["company","year"]).stock_code.nunique()
        mappings.append({"scope":scope,"rows":len(frame),"securities":frame.stock_code.nunique(),
            "nonmissing_name_strings":frame.company.nunique(),"missing_name_rows":int(frame.company.isna().sum()),
            "codes_with_no_observed_name":int(bycode.eq(0).sum()),"codes_with_exactly_one_observed_name":int(bycode.eq(1).sum()),
            "codes_with_multiple_observed_names":int(bycode.gt(1).sum()),
            "names_with_one_code":int(byname.eq(1).sum()),"names_with_multiple_codes":int(byname.gt(1).sum()),
            "same_year_name_groups_with_multiple_codes":int(sameyear.gt(1).sum())})
        for label,columns,complete in [("reduced_complete",REDUCED,True),("full_including_missing_patterns",FULL,False),("full_complete",FULL,True)]:
            sub=frame.dropna(subset=columns) if complete else frame
            d=sub[sub.duplicated(columns,keep=False)]
            profiles.append({"scope":scope,"profile":label,"evaluated_rows":len(sub),"duplicate_rows":len(d),
                "duplicate_groups":len(d[columns].drop_duplicates()),"distinct_codes_in_duplicate_rows":d.stock_code.nunique()})
    a,b=pd.DataFrame(mappings),pd.DataFrame(profiles)
    a.to_csv(OUT / "code_name_mapping_counts.csv",index=False)
    b.to_csv(OUT / "economic_profile_duplicate_counts.csv",index=False)
    old=pd.read_csv(ROOT / "analysis/original-forecast/accounting_line_identity_diagnostic/financial_profile_duplicate_counts.csv")
    for _,r in b.iterrows():
        previous=old[old.scope.eq(r.scope)&old.profile.eq(r.profile)].iloc[0]
        assert all(r[c]==previous[c] for c in ['evaluated_rows','duplicate_rows','duplicate_groups','distinct_codes_in_duplicate_rows'])
    return a


def counted(frame, columns, filename):
    output=frame.groupby(columns,dropna=False).agg(n=("stock_code","size"),securities=("stock_code","nunique")).reset_index()
    assert output.n.sum()==len(frame)
    output.to_csv(OUT/filename,index=False)
    return output


def availability(raw):
    known=raw.industry.notna()&raw.industry.str.strip().ne("")
    origins=raw[known&~raw.industry.isin(FINANCIAL)]
    future=raw[["stock_code","year","roe_decimal","net_profit","source_excel_row"]].rename(columns={"roe_decimal":"target_roe_decimal","net_profit":"target_net_profit","source_excel_row":"target_source_excel_row"})
    future["year"]-=1
    pairs=origins.merge(future,on=KEY,validate="one_to_one")
    pairs["target_year"]=pairs.year+1
    mo,mt=pairs.roe_decimal.isna(),pairs.target_roe_decimal.isna()
    missing=pairs[mo|mt].copy()
    missing["blank_pattern"]=np.select([mo.loc[missing.index]&mt.loc[missing.index],mo.loc[missing.index]],['both','origin_only'],default='target_only')
    missing["origin_profit_sign"]=sign(missing.net_profit)
    missing["target_profit_sign"]=sign(missing.target_net_profit)
    assert len(pairs)==28439 and len(missing)==404
    assert missing.blank_pattern.value_counts().to_dict()=={'both':154,'target_only':153,'origin_only':97}
    assert not np.isinf(pairs[["roe_decimal","target_roe_decimal"]]).any().any()
    counted(missing,["year","target_year","provenance","blank_pattern","origin_profit_sign","target_profit_sign"],"blank_roe_pairs_by_year_membership_profit_sign.csv")
    rows=[]
    for status,sub in [("all",missing)]+[(s,missing[missing.provenance.eq(s)]) for s in STATUSES]:
        row={"provenance":status,"excluded_pairs":len(sub),"securities":sub.stock_code.nunique()}
        for pattern in ['origin_only','target_only','both']:row[pattern+"_pairs"]=int(sub.blank_pattern.eq(pattern).sum())
        for role,column in [('origin','net_profit'),('target','target_net_profit')]:
            states=pd.Series(sign(sub[column])).value_counts()
            for state in ['negative','zero','positive','missing','nonfinite']:row[role+'_profit_'+state+'_n']=int(states.get(state,0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/"blank_roe_pair_membership_summary.csv",index=False)
    source_rows=set(pairs.loc[mo,'source_excel_row'].astype(int))|set(pairs.loc[mt,'target_source_excel_row'].astype(int))
    cells=raw[raw.source_excel_row.isin(source_rows)].copy();assert len(cells)==361 and cells.roe_decimal.isna().all()
    cells['net_profit_sign']=sign(cells.net_profit)
    counted(cells,['year','provenance','net_profit_sign'],'unique_blank_roe_cells_by_year_membership_profit_sign.csv')
    old=json.loads((ROOT/'analysis/original-forecast/roe_missingness_audit.json').read_text())
    assert old['counts']['excluded_pairs']==len(missing)
    for v in old['cell_audit'].values():assert v['checked_unique_cells']==361 and v['cell_values']=={'None (blank)':361}
    unknown=raw[~known].copy();assert len(unknown)==1495
    unknown['net_profit_sign']=sign(unknown.net_profit)
    unknown['roe_availability']=np.select([unknown.roe_decimal.isna(),np.isinf(unknown.roe_decimal)],['missing','nonfinite'],default='finite')
    counted(unknown,['year','provenance','net_profit_sign','roe_availability'],'unknown_industry_by_year_membership_profit_sign.csv')
    unknown_summary=[]
    for status,sub in [("all",unknown)]+[(s,unknown[unknown.provenance.eq(s)]) for s in STATUSES]:
        row={"provenance":status,"source_rows":len(sub),"securities":sub.stock_code.nunique(),
             "finite_roe_n":int(np.isfinite(sub.roe_decimal).sum()),"missing_roe_n":int(sub.roe_decimal.isna().sum())}
        for state in ['negative','zero','positive','missing','nonfinite']:row['net_profit_'+state+'_n']=int(sub.net_profit_sign.eq(state).sum())
        unknown_summary.append(row)
    pd.DataFrame(unknown_summary).to_csv(OUT/'unknown_industry_membership_summary.csv',index=False)
    return {'adjacent_known_nonfinancial_pairs':len(pairs),'blank_roe_pairs':len(missing),'unique_blank_roe_cells':len(cells),
            'unknown_industry_source_rows':len(unknown),'source_cell_audit_reused':True,'reasons_for_blank_roe_or_unknown_industry':'Not established by these source fields; no equity rule or causal missingness inference.'}


def main():
    assert DESIGN.exists()
    inputs=['data/original-private/original_panel.csv','data/processed/peng2022_panel_rawvalues.csv',
        'analysis/original-forecast/transparent_main/predictions.csv','analysis/original-forecast/transparent_main/fit_records.json',
        'analysis/original-listing-audit/origin_extreme_training_thresholds.csv','analysis/original-forecast/roe_missingness_audit.json',
        'analysis/original-forecast/accounting_line_identity_diagnostic/financial_profile_duplicate_counts.csv']
    hashes={p:sha(ROOT/p) for p in inputs}
    OUT.mkdir(parents=True,exist_ok=True)
    raw,p=load()
    counts,distributions=main_descriptions(p)
    history=unmatched_history(raw,p)
    mappings=identity(raw,p)
    missing=availability(raw)
    assert hashes=={p:sha(ROOT/p) for p in inputs}
    result={'status':'post-analysis bounded descriptive audit; no data acquisition or model fits',
        'created_utc':datetime.now(timezone.utc).isoformat(),'design_sha256':sha(DESIGN),'code_sha256':sha(Path(__file__)),
        'definition_helper_sha256':sha(ROOT/'code/describe_original_accounting_lines_identity.py'),'input_sha256':hashes,'inputs_unchanged':True,
        'unmatched_export_history':history,'availability':missing,
        'size_units':'Total assets in unchanged source currency units; distributions use positive finite assets.',
        'implied_equity_definition':'(net_profit/total_assets)/(roe_pct/100), an implied average-equity/assets ratio under the checked ROE construction, not directly observed equity.',
        'tail_definition':'Origin ROE strictly outside the saved evaluation-fold training q01/q99 thresholds; not target-conditioned.',
        'identity_scope':'Source code/name mapping and exact financial-profile duplicates only; legal issuer and share-class identity not independently verified.',
        'security_count_scope':'Codes may occur in different sign/year/provenance groups, so subgroup distinct-security counts need not sum.',
        'distribution_scope':'All outputs are aggregate; no security-level observation or company-name lists.'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(counts[counts.partition.eq('2015-2019')&counts.group_scope.eq('provenance')].to_string(index=False))
    print(distributions[distributions.partition.eq('2015-2019')&distributions.group_scope.eq('provenance')][['provenance','variable','finite_n','q10','median','q90']].to_string(index=False))
    print(json.dumps(history,indent=2))
    print(mappings.to_string(index=False))
    for name in ['blank_roe_pair_membership_summary.csv','unknown_industry_membership_summary.csv']:
        print(pd.read_csv(OUT/name).to_string(index=False))


if __name__=='__main__':
    main()
