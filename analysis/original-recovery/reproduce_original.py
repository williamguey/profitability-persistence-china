"""Translate supplied R literally, then audit definitions and printed differences.

Run with project .venv/bin/python. Never executes original R or changes sources.
The workbook's cached values are inputs; all transformed columns are separate.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT/'data/original-private'
df = pd.read_csv(RAW/'01_base_tese_cached.csv')
df['source_excel_row'] = np.arange(2, len(df)+2)
assert not df.duplicated(['codigo_empresa','years']).any()
numeric = [c for c in df if c not in ('company','industry')]
summary = {'rows':len(df),'codes':int(df.codigo_empresa.nunique()),'company_names':int(df.company.nunique()),
           'years':sorted(df.years.unique().tolist()),'industry_nonmissing':int(df.industry.nunique()),
           'industry_missing':int(df.industry.isna().sum()), 'duplicate_code_year':int(df.duplicated(['codigo_empresa','years']).sum())}
df[numeric].describe(percentiles=[.01,.5,.99]).T.to_csv(OUT/'raw_descriptives.csv')
df.groupby('years').size().to_csv(OUT/'raw_year_counts.csv',header=['n'])
pd.DataFrame({'missing':df.isna().sum(),'zeros':df.eq(0).sum()}).to_csv(OUT/'raw_missing_zeros.csv')

def identity(name, a, b):
    ok = np.isfinite(a)&np.isfinite(b)
    e = a[ok]-b[ok]
    return {'identity':name,'n':int(ok.sum()),'within_cent':int((abs(e)<=.011).sum()),
            'within_relative_1e_8':int(np.isclose(a[ok],b[ok],rtol=1e-8,atol=.011).sum()),
            'median_abs_error':float(abs(e).median()),'max_abs_error':float(abs(e).max())}
identities = [identity('net_assets = current_assets - liabilities',df.net_assets,df.current_assets-df.liabilities),
              identity('net_assets = total_assets - liabilities',df.net_assets,df.total_assets-df.liabilities),
              identity('net_profit = total_profit - income_tax_expense',df.net_profit,df.total_profit-df.income_tax_expense),
              identity('total_profit = operating_profit + non_operating_income - non_operating_expenses',df.total_profit,df.operating_profit+df.non_operating_income-df.non_operating_expenses),
              identity('EBIT = total_profit + financial_expenses_total',df.EBIT,df.total_profit+df.financial_expenses_total),
              identity('EBIT = total_profit + financial_expenses_positive',df.EBIT,df.total_profit+df.financial_expenses_positive),
              identity('EBITDA = EBIT + depreciation_fixed_assets',df.EBITDA,df.EBIT+df.depreciation_fixed_assets)]
pd.DataFrame(identities).to_csv(OUT/'accounting_identities.csv',index=False)
df.assign(zero_liabilities=df.liabilities.eq(0),zero_current_assets=df.current_assets.eq(0),negative_liabilities=df.liabilities.lt(0)).groupby('industry',dropna=False).agg(n=('years','size'),zero_liabilities=('zero_liabilities','sum'),zero_current_assets=('zero_current_assets','sum'),negative_liabilities=('negative_liabilities','sum')).to_csv(OUT/'industry_zero_balances.csv')

# R's subset(ROE > -50) removes missing comparisons as well as extremes.
d = df.loc[df.ROE.gt(-50)].copy()
summary['ROE_missing_dropped'] = int(df.ROE.isna().sum())
summary['ROE_le_minus50_dropped'] = int(df.ROE.le(-50).sum())
summary['after_source_ROE_filter'] = len(d)
df.loc[~df.ROE.gt(-50)].to_csv(OUT/'source_roe_filter_dropped.csv',index=False)
with np.errstate(divide='ignore',invalid='ignore'):
    d['X1'] = (d.current_assets-d.liabilities)/d.total_assets
    d['X2'] = (d.net_assets-d.share_capital)/d.total_assets
    d['X3'] = (d.operating_profit-d.financial_expenses_total)/d.total_assets
    d['X4'] = (d.non_operating_income/d.non_operating_expenses)/1e6
    d['X5'] = d.net_profit/d.total_assets
    d['X6'] = (d.EBIT/d.financial_expenses_positive)/1e6
    d['X7'] = d.income_tax_expense/d.total_assets
    d['X8'] = d.depreciation_fixed_assets/d.total_assets
    d.loc[d.non_operating_expenses.eq(0),'X4'] = np.nan
    d.loc[d.financial_expenses_positive.eq(0),'X6'] = np.nan
    # Alternative CSMAR ROE proxy diagnostics only; liabilities field is unresolved.
    d['profit_div_end_assets_less_reported_liabilities'] = d.net_profit/(d.total_assets-d.liabilities)
aux = d[['codigo_empresa','years','ROE','company','source_excel_row']].rename(columns={'ROE':'diff_ROE','company':'outcome_company','source_excel_row':'outcome_excel_row'})
aux['years'] -= 1
joined = d.merge(aux,on=['codigo_empresa','years'],validate='one_to_one')
summary['calendar_adjacent_pairs_after_filter'] = len(joined)
joined.to_csv(RAW/'original_script_pairs.csv',index=False)
xs = [f'X{i}' for i in range(1,9)]
joined[xs+['ROE','diff_ROE']].describe(percentiles=[.01,.5,.99]).T.to_csv(OUT/'script_ratios_descriptives.csv')
finite = np.isfinite(joined[xs+['diff_ROE']]).all(axis=1)
summary['infinite_model_values'] = int(np.isinf(joined[xs+['diff_ROE']]).any(axis=1).sum())
summary['finite_ratios_and_outcome'] = int(finite.sum())
summary['finite_plus_industry'] = int((finite & joined.industry.notna()).sum())
models=[]; coefficients=[]
def fit(name,data,terms):
    req = [t for t in terms if not t.startswith('C(')] + ['diff_ROE']
    if 'C(industry)' in terms: req += ['industry']
    use = data.dropna(subset=req).copy()
    assert np.isfinite(use[[c for c in req if c!='industry']]).all().all()
    m = smf.ols('diff_ROE ~ '+' + '.join(terms), data=use).fit()
    hc = m.get_robustcov_results(cov_type='HC1')
    cl = m.get_robustcov_results(cov_type='cluster',groups=use.codigo_empresa)
    models.append({'model':name,'n':int(m.nobs),'r2':float(m.rsquared),'r2_adj':float(m.rsquared_adj),'resid_se':float(np.sqrt(m.mse_resid)),
                   'df_model':int(m.df_model),'df_resid':int(m.df_resid),'rank':int(m.model.rank),'industries':int(use.industry.nunique()),'years':sorted(use.years.unique().tolist()),
                   'min_target':float(use.diff_ROE.min()),'max_target':float(use.diff_ROE.max())})
    for k,(term,b) in enumerate(m.params.items()):
        coefficients.append({'model':name,'term':term,'estimate':float(b),'se_classical':float(m.bse.iloc[k]),'se_HC1':float(hc.bse[k]),'se_firm_cluster':float(cl.bse[k]),
                            'p_classical':float(m.pvalues.iloc[k]),'p_HC1':float(hc.pvalues[k]),'p_firm_cluster':float(cl.pvalues[k])})
    return m

fit('script_2019_cross_section',joined.loc[joined.years.eq(2019)],xs)
fit('script_model1',joined,xs+['C(industry)','C(years)'])
# Extensions correspond structurally to thesis models 2 and 3, but are NOT
# contained in supplied R. They use raw script transforms, not undocumented ones.
fit('script_transforms_thesis_model2',joined,xs[1:]+['C(industry)','C(years)'])
fit('script_transforms_thesis_model3',joined,xs[1:]+['ROE','C(industry)','C(years)'])
pd.DataFrame(models).to_csv(OUT/'model_statistics.csv',index=False)
pd.DataFrame(coefficients).to_csv(OUT/'coefficients.csv',index=False)
summary['source_script_filter_note'] = 'ROE > -50 applied before self-join, so both t and t+1 must pass. Exact code-year match, no row-based lead.'
summary['net_assets_formula_all_rows'] = 'The original formula export verifies =G{row}-H{row} in all 34,799 data rows.'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print(pd.DataFrame(models).to_string(index=False))
print(pd.DataFrame(coefficients).query("term.str.match('^X[1-8]$')",engine='python').to_string(index=False))
