"""Post-analysis deterministic loss-difference bounds for missing target ROE; no fits."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT/'analysis/original-forecast/transparent_main'
OUT = ROOT/'analysis/original-forecast/missing_target_loss_bound'
DESIGN = ROOT/'config/original-missing-target-loss-bound.txt'
RAW = ROOT/'data/original-private/original_panel.csv'
AVAIL = ROOT/'analysis/original-forecast/loss_recovery_availability_diagnostic/recovery_by_target_roe_availability.csv'
FINANCIAL = {'Capital Market Services','Insurance','monetary financial services','Other financial industries'}
MODELS = ['lad_ar','lad_piecewise','lad_signals']
CONTRASTS = [('asymmetry_first','lad_ar','lad_piecewise'),('accounting_second','lad_piecewise','lad_signals')]
STATES = ['finite_target_roe_retained','blank_target_roe_existing_record','nonfinite_target_roe_existing_record','no_next_year_source_record']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def origin_ratios(frame):
    """Original transparent ratio definitions, without target or future inputs."""
    a=frame.total_assets.where(np.isfinite(frame.total_assets)&frame.total_assets.gt(0))
    r=pd.DataFrame({'roe_pct':frame.roe_decimal*100,
        'x1':(frame.current_assets-frame.current_liabilities)/a,
        'x2':frame.share_capital/a,'x3':frame.operating_profit/a,
        'x4':frame.financial_expenses_total/a,'x5':frame.net_profit/a,
        'x6':(frame.non_operating_income-frame.non_operating_expenses)/a,
        'x7':frame.income_tax_expense/a,'x8':frame.depreciation_fixed_assets/a},index=frame.index)
    return r.where(np.isfinite(r))


def apply_saved(frame, record):
    """Apply frozen thresholds/flags/scales/column retention and coefficients only."""
    t=record['transform'];names=t['continuous_columns']
    values=frame[names].to_numpy(float);missing=~np.isfinite(values)
    lo,hi,fill=[np.asarray(t[k],float) for k in ['lower','upper','median_imputation']]
    assert np.isfinite([lo,hi,fill]).all() and (lo<=hi).all()
    processed=np.where(missing,fill,np.clip(values,lo,hi))
    negative=frame.roe_pct.lt(0).to_numpy()
    cols=[]
    assert not t['industry_categories']
    for name in t['all_columns']:
        if name in names:
            col=processed[:,names.index(name)]
        elif name=='negative_roe':
            col=negative
        elif name=='roe_x_negative':
            col=processed[:,names.index('roe_pct')]*negative
        elif name.endswith('_missing') and name.removesuffix('_missing') in names:
            col=missing[:,names.index(name.removesuffix('_missing'))]
        else:
            raise ValueError('Unsupported saved column: '+name)
        cols.append(col)
    x=np.column_stack(cols)
    mean,sd=np.asarray(t['mean']),np.asarray(t['std'])
    assert len(mean)==x.shape[1] and np.isfinite(mean).all() and np.isfinite(sd).all() and (sd>0).all()
    z=(x-mean)/sd
    keep=[t['all_columns'].index(name) for name in t['retained_columns']]
    z=np.column_stack([np.ones(len(frame)),z[:,keep]])
    beta=np.asarray(record['beta'])
    assert z.shape[1]==len(beta) and np.isfinite(beta).all()
    return z@beta


def main():
    assert DESIGN.exists()
    started=datetime.now(timezone.utc).isoformat()
    inputs=[RAW,MAIN/'eligible_pairs.csv',MAIN/'predictions.csv',MAIN/'fit_records.json',MAIN/'metrics.csv',MAIN/'paired_bootstrap.csv',AVAIL]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    code_hash,design_hash=sha(Path(__file__)),sha(DESIGN)
    raw=pd.read_csv(RAW,dtype={'stock_code':str})
    eligible=pd.read_csv(MAIN/'eligible_pairs.csv',dtype={'stock_code':str})
    saved=pd.read_csv(MAIN/'predictions.csv',dtype={'stock_code':str})
    saved=saved[saved.target_year.between(2015,2019)].copy()
    records=json.loads((MAIN/'fit_records.json').read_text())
    old_metrics=pd.read_csv(MAIN/'metrics.csv',dtype={'partition':str})
    old_contrasts=pd.read_csv(MAIN/'paired_bootstrap.csv',dtype={'partition':str})
    prior_availability=pd.read_csv(AVAIL,dtype={'partition':str})
    keys=['stock_code','year']
    assert not raw.duplicated(keys).any() and len(saved)==15258 and not saved.duplicated(keys).any()
    assert saved.target_year.eq(saved.year+1).all()
    # Verify all retained predictions before constructing any excluded forecast.
    raw_index=raw.set_index(keys)
    source=raw_index.loc[pd.MultiIndex.from_frame(saved[keys])].reset_index()
    np.testing.assert_array_equal(source[keys],saved[keys])
    ratios=origin_ratios(source)
    eligible_main=eligible.set_index(keys).loc[pd.MultiIndex.from_frame(saved[keys])].reset_index()
    for c in ratios:
        np.testing.assert_allclose(ratios[c],eligible_main[c],rtol=1e-12,atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(source.roe_decimal*100,saved.roe_pct,rtol=0,atol=1e-10)
    recon_rows=[]
    for year in range(2015,2020):
        mask=saved.target_year.eq(year).to_numpy()
        for model in MODELS:
            record=next(r for r in records if r['target_year']==year and r['model']==model)
            assert record['training_target_max']==year-1 and record['n_test']==mask.sum()
            reconstructed=apply_saved(ratios.loc[mask],record)
            error=float(abs(reconstructed-saved.loc[mask,model].to_numpy()).max())
            assert error<1e-8,(year,model,error)
            actual_mae=float(abs(saved.loc[mask,model]-saved.loc[mask,'target']).mean())
            old=old_metrics[old_metrics.partition.eq(str(year))&old_metrics.model.eq(model)]
            assert len(old)==1
            np.testing.assert_allclose(actual_mae,old.mae.iloc[0],rtol=0,atol=1e-10)
            recon_rows.append({'target_year':year,'model':model,'retained_n':int(mask.sum()),
                'max_absolute_prediction_difference':error,'saved_mae_pp':actual_mae,
                'reconstructed_mae_pp':float(abs(reconstructed-saved.loc[mask,'target'].to_numpy()).mean())})
    retained_verified=datetime.now(timezone.utc).isoformat()
    known=raw.industry.notna()&raw.industry.str.strip().ne('')&~raw.industry.isin(FINANCIAL)
    origins=raw[known&raw.year.between(2014,2018)&np.isfinite(raw.roe_decimal)&raw.roe_decimal.lt(0)].copy()
    future=raw[['stock_code','year','roe_decimal']].rename(columns={'roe_decimal':'target_roe_decimal'})
    future['year']-=1
    candidates=origins.merge(future,on=keys,how='left',validate='one_to_one',indicator=True)
    candidates['target_year']=candidates.year+1
    candidates['availability']=np.select([candidates._merge.eq('left_only'),candidates.target_roe_decimal.isna(),np.isinf(candidates.target_roe_decimal)],
        [STATES[3],STATES[1],STATES[2]],default=STATES[0])
    assert len(candidates)==1583
    retained=candidates[candidates.availability.eq(STATES[0])]
    expected=saved[saved.negative_roe.eq(1)]
    assert len(retained)==1520 and set(map(tuple,retained[keys].to_numpy()))==set(map(tuple,expected[keys].to_numpy()))
    missing=candidates[candidates.availability.isin([STATES[1],STATES[3]])].copy().reset_index(drop=True)
    assert len(missing)==63 and missing.availability.eq(STATES[1]).sum()==62 and missing.availability.eq(STATES[3]).sum()==1
    assert not candidates.availability.eq(STATES[2]).any()
    availability_rows=[]
    partitions=[('2015-2019',candidates)]+[(str(y),candidates[candidates.target_year.eq(y)]) for y in range(2015,2020)]
    for label,frame in partitions:
        for state in STATES:
            f=frame[frame.availability.eq(state)]
            old=prior_availability[prior_availability.partition.eq(label)&prior_availability.provenance.eq('all')&prior_availability.target_roe_availability.eq(state)]
            assert len(old)==1 and old.origin_n.iloc[0]==len(f) and old.securities.iloc[0]==f.stock_code.nunique()
            availability_rows.append({'partition':label,'target_roe_availability':state,'origin_n':len(f),'securities':f.stock_code.nunique()})
    omitted_ratios=origin_ratios(missing)
    omitted_forecasts=pd.DataFrame(index=missing.index,columns=MODELS,dtype=float)
    for year in range(2015,2020):
        mask=missing.target_year.eq(year)
        for model in MODELS:
            rec=next(r for r in records if r['target_year']==year and r['model']==model)
            omitted_forecasts.loc[mask,model]=apply_saved(omitted_ratios.loc[mask],rec)
    input_rows=[]
    for c in omitted_ratios:
        finite=np.isfinite(omitted_ratios[c])
        input_rows.append({'variable':c,'excluded_n':len(missing),'finite_origin_input_n':int(finite.sum()),'unavailable_origin_input_n':int((~finite).sum())})
    forecast_rows=[]
    for model in MODELS:
        finite=np.isfinite(omitted_forecasts[model]);v=omitted_forecasts.loc[finite,model]
        forecast_rows.append({'model':model,'excluded_n':len(missing),'finite_forecast_n':int(finite.sum()),
            'unavailable_forecast_n':int((~finite).sum()),'forecast_minimum_pp':float(v.min()) if len(v) else None,
            'forecast_median_pp':float(v.median()) if len(v) else None,'forecast_maximum_pp':float(v.max()) if len(v) else None})
    gap_rows,bounds=[],[]
    for contrast,benchmark,candidate in CONTRASTS:
        finite=np.isfinite(omitted_forecasts[[benchmark,candidate]]).all(axis=1)
        gap=abs(omitted_forecasts[candidate]-omitted_forecasts[benchmark])
        total_gap=float(gap[finite].sum())
        for year in range(2015,2020):
            mask=missing.target_year.eq(year)
            gap_rows.append({'target_year':year,'contrast':contrast,'excluded_n':int(mask.sum()),
                'finite_forecast_pair_n':int((mask&finite).sum()),'unresolved_forecast_pair_n':int((mask&~finite).sum()),
                'sum_absolute_forecast_gap_pp':float(gap[mask&finite].sum())})
        yearly_sum=sum(r['sum_absolute_forecast_gap_pp'] for r in gap_rows if r['contrast']==contrast)
        np.testing.assert_allclose(total_gap,yearly_sum,rtol=0,atol=1e-10)
        observed=abs(saved[benchmark]-saved.target)-abs(saved[candidate]-saved.target)
        loss_sum=float(observed.sum());gain=float(observed.mean())
        old=old_contrasts[old_contrasts.partition.eq('2015-2019')&old_contrasts.reference.eq(benchmark)&old_contrasts.candidate.eq(candidate)]
        assert len(old)==1
        np.testing.assert_allclose(gain,old.mae_reduction_pp.iloc[0],rtol=0,atol=1e-10)
        complete=bool(finite.all());denom=len(saved)+len(missing)
        # Unresolved predictions cannot silently disappear from the requested bound.
        lower=(loss_sum-total_gap)/denom if complete else None
        upper=(loss_sum+total_gap)/denom if complete else None
        covered_denominator=len(saved)+int(finite.sum())
        bounds.append({'contrast':contrast,'benchmark':benchmark,'candidate':candidate,
            'retained_n':len(saved),'excluded_n':len(missing),'enlarged_n':denom,
            'finite_excluded_forecast_pair_n':int(finite.sum()),'unresolved_excluded_forecast_pair_n':int((~finite).sum()),
            'original_mae_gain_pp':gain,'retained_paired_loss_sum_pp':loss_sum,
            'sum_absolute_excluded_forecast_gap_pp':total_gap,'enlarged_sample_half_width_pp':total_gap/denom if complete else None,
            'enlarged_sample_lower_gain_pp':lower,'enlarged_sample_upper_gain_pp':upper,
            'complete_requested_bound':complete,'covered_only_enlarged_n':covered_denominator,
            'covered_only_lower_gain_pp':(loss_sum-total_gap)/covered_denominator,
            'covered_only_upper_gain_pp':(loss_sum+total_gap)/covered_denominator})
    assert hashes=={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    assert code_hash==sha(Path(__file__)) and design_hash==sha(DESIGN)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in [('availability_counts.csv',availability_rows),('retained_forecast_reconstruction.csv',recon_rows),
        ('excluded_input_availability.csv',input_rows),('excluded_forecast_summaries.csv',forecast_rows),
        ('annual_forecast_gap_sums.csv',gap_rows),('gain_bounds.csv',bounds)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False)
    summary={'status':'post-analysis deterministic bound using original fixed forecasts; no fitting or outcome imputation',
        'started_utc':started,'retained_predictions_verified_utc':retained_verified,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'code_sha256':code_hash,'design_sha256':design_hash,'input_sha256':hashes,'inputs_unchanged':True,
        'candidate_negative_origins':len(candidates),'retained_negative_origins':len(retained),'excluded_negative_origins':len(missing),
        'blank_target_roe_existing_record_n':int(missing.availability.eq(STATES[1]).sum()),'no_next_year_source_record_n':int(missing.availability.eq(STATES[3]).sum()),
        'retained_full_sample_n':len(saved),'enlarged_full_sample_n':len(saved)+len(missing),
        'retained_models_checked':len(recon_rows),'retained_forecast_max_difference':max(r['max_absolute_prediction_difference'] for r in recon_rows),
        'forecast_coverage_complete':bool(np.isfinite(omitted_forecasts).all().all()),
        'formula':'For each contrast: (sum_retained(|y-b|-|y-a|) +/- sum_excluded(|a-b|))/(15258+63).',
        'scope':'Adds exactly 63 known-nonfinancial negative-current-ROE origins with missing next-year ROE to the original evaluation, holding fitted models fixed. Does not cover other excluded rows, missingness causes, nonpositive-equity assumptions, refitting, family selection, or unrestricted populations.',
        'interval_type':'Deterministic contrast-wise envelope for arbitrary finite missing targets; not a confidence interval and not necessarily jointly attainable across contrasts.',
        'row_level_outputs_distributed':False,'new_model_fits':False,'target_imputation':False,'primary_outputs_modified':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(pd.DataFrame(bounds).to_string(index=False))
    print(pd.DataFrame(input_rows).to_string(index=False))
    print(json.dumps({k:summary[k] for k in ['retained_forecast_max_difference','forecast_coverage_complete','excluded_negative_origins']},indent=2))


if __name__=='__main__':
    main()
