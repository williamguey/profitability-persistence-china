"""Post-analysis profit-transition descriptions by next-year ROE availability."""
from pathlib import Path
from datetime import datetime,timezone
import json

import numpy as np
import pandas as pd

from describe_original_membership_missingness import FINANCIAL, STATUSES, provenance, sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"analysis/original-forecast/loss_recovery_availability_diagnostic"
DESIGN=ROOT/"config/original-loss-recovery-availability.txt"
AVAILABILITY=["finite_target_roe_retained","blank_target_roe_existing_record","nonfinite_target_roe_existing_record","no_next_year_source_record"]


def main():
    assert DESIGN.exists()
    inputs=["data/original-private/original_panel.csv","data/processed/peng2022_panel_rawvalues.csv"]
    hashes={p:sha(ROOT/p) for p in inputs}
    raw=pd.read_csv(ROOT/inputs[0],dtype={"stock_code":str})
    pub=pd.read_csv(ROOT/inputs[1],dtype={"stock_code":str})
    dates=pub[["stock_code","listing_date"]].drop_duplicates()
    assert not dates.stock_code.duplicated().any()
    dates['listing_year']=pd.to_datetime(dates.listing_date,errors='coerce').dt.year
    assert dates.listing_year.notna().all()
    known=raw.industry.notna()&raw.industry.str.strip().ne('')&~raw.industry.isin(FINANCIAL)
    origins=raw[known&raw.year.between(2014,2018)&np.isfinite(raw.roe_decimal)&raw.roe_decimal.lt(0)].copy()
    origins=provenance(origins,dates[['stock_code','listing_year']])
    future=raw[['stock_code','year','roe_decimal','net_profit']].rename(columns={'roe_decimal':'target_roe_decimal','net_profit':'target_net_profit'})
    future['year']-=1
    pairs=origins.merge(future,on=['stock_code','year'],how='left',validate='one_to_one',indicator=True)
    pairs['target_year']=pairs.year+1
    pairs['target_availability']=np.select([pairs._merge.eq('left_only'),pairs.target_roe_decimal.isna(),np.isinf(pairs.target_roe_decimal)],AVAILABILITY[3:4]+AVAILABILITY[1:3],default=AVAILABILITY[0])
    assert len(pairs)==len(origins)
    retained=pairs[pairs.target_availability.eq(AVAILABILITY[0])]
    assert len(retained)==1520 and retained.target_net_profit.gt(0).sum()==1058 and retained.target_net_profit.lt(0).sum()==462
    rows=[]
    partitions=[('2015-2019',pairs)]+[(str(y),pairs[pairs.target_year.eq(y)]) for y in range(2015,2020)]
    for period,part in partitions:
        for group,frame in [('all',part)]+[(s,part[part.provenance.eq(s)]) for s in STATUSES]:
            for state in AVAILABILITY:
                sub=frame[frame.target_availability.eq(state)]
                finite=sub[np.isfinite(sub.target_net_profit)]
                positive,zero,negative=(int(finite.target_net_profit.gt(0).sum()),int(finite.target_net_profit.eq(0).sum()),int(finite.target_net_profit.lt(0).sum()))
                rows.append({'partition':period,'provenance':group,'target_roe_availability':state,'origin_n':len(sub),
                    'securities':sub.stock_code.nunique(),'finite_target_profit_n':len(finite),'positive_target_profit_n':positive,
                    'zero_target_profit_n':zero,'negative_target_profit_n':negative,'missing_target_profit_n':int(sub.target_net_profit.isna().sum()),
                    'nonfinite_target_profit_n':int(np.isinf(sub.target_net_profit).sum()),
                    'positive_recovery_pct_among_observed_profit':100*positive/len(finite) if len(finite) else None,
                    'nonnegative_recovery_pct_among_observed_profit':100*(positive+zero)/len(finite) if len(finite) else None})
    result=pd.DataFrame(rows)
    for period,part in partitions:
        assert result[result.partition.eq(period)&result.provenance.eq('all')].origin_n.sum()==len(part)
        for state in AVAILABILITY:
            a=result[result.partition.eq(period)&result.target_roe_availability.eq(state)]
            for column in ['origin_n','finite_target_profit_n','positive_target_profit_n','negative_target_profit_n','missing_target_profit_n']:
                assert a.loc[a.provenance.eq('all'),column].iloc[0]==a.loc[a.provenance.isin(STATUSES),column].sum()
    OUT.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUT/'recovery_by_target_roe_availability.csv',index=False)
    assert hashes=={p:sha(ROOT/p) for p in inputs}
    metadata={'status':'post-analysis descriptive extension; no model fits or imputations','created_utc':datetime.now(timezone.utc).isoformat(),
        'code_sha256':sha(Path(__file__)),'design_sha256':sha(DESIGN),
        'definition_helper_sha256':sha(ROOT/'code/describe_original_membership_missingness.py'),'input_sha256':hashes,'inputs_unchanged':True,
        'population':'Known nonfinancial 2014--2018 origins with finite negative ROE, exact next-calendar-year same-security join.',
        'candidate_origins':len(origins),'candidate_securities':int(origins.stock_code.nunique()),
        'finite_target_roe_retained_origins':len(retained),'availability_categories':AVAILABILITY,
        'no_next_year_record_interpretation':'Source coverage absence, distinct from an existing blank ROE cell; no next-year profit inferred.',
        'recovery_denominator':'Finite observed next-year net profit within each category; unavailable values explicitly counted, never imputed.',
        'scope':'Conditional observed profit transitions only; no identified missingness cause, equity rule, population recovery rate, or regulatory mechanism.'}
    (OUT/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    print(result[result.partition.eq('2015-2019')].to_string(index=False))


if __name__=='__main__':main()
