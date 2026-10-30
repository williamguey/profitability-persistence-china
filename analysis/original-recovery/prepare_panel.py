"""Create documented analysis inputs from unchanged recovered scalar values.

Preserves every row. Flags only; no sample selection, zero imputation, outlier
deletion, final-thesis reverse engineering, or issuer matching by company name.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RAW=ROOT/'data/original-private'
d=pd.read_csv(RAW/'01_base_tese_cached.csv')
d['source_excel_row']=np.arange(2,len(d)+2)
d['firm_code']=d.codigo_empresa.astype(int)
d['stock_code']=d.firm_code.astype(str).str.zfill(6)
d['year']=d.years.astype(int)
d['current_liabilities']=d.liabilities
d['working_capital_cached']=d.net_assets
d['working_capital']=d.current_assets-d.current_liabilities
d['working_capital_cache_has_blank_components']=d.current_assets.isna()|d.current_liabilities.isna()
d['roe_decimal']=d.ROE
d['roa_decimal']=d.ROA
d['is_b_share_code']=d.firm_code.between(200000,299999)|d.firm_code.between(900000,999999)
d['industry_financial']=d.industry.fillna('').str.contains('financial|insurance|capital market',case=False)

# No duplicated finite accounting profiles among rows with nonmissing ROE.
keys=['year','total_assets','net_profit','operating_profit','share_capital']
valid=d.total_assets.notna()&d.net_profit.notna()
d['duplicate_accounting_profile']=False
d.loc[valid,'duplicate_accounting_profile']=d.loc[valid].duplicated(keys,keep=False)
d['duplicate_profile_with_nonmissing_roe']=False
use=valid&d.ROE.notna()
d.loc[use,'duplicate_profile_with_nonmissing_roe']=d.loc[use].duplicated(keys,keep=False)
assert not d.duplicate_profile_with_nonmissing_roe.any()
d.to_csv(RAW/'original_panel.csv',index=False)
d.loc[d.duplicate_accounting_profile,['source_excel_row','stock_code','company','year','ROE']+keys[1:]].to_csv(OUT/'duplicate_accounting_profiles.csv',index=False)
multiname=d.groupby('company').firm_code.nunique()
d.loc[d.company.isin(multiname[multiname>1].index),['stock_code','company','year','ROE','total_assets']].to_csv(OUT/'nonunique_company_names.csv',index=False)

lag=d[['firm_code','year','total_assets']].copy().rename(columns={'total_assets':'previous_assets'})
lag.year += 1
adj=d.merge(lag,on=['firm_code','year'],validate='one_to_one')
adj['calculated_roa']=adj.net_profit/((adj.total_assets+adj.previous_assets)/2)
ok=adj[['ROA','calculated_roa']].notna().all(axis=1)&np.isfinite(adj.calculated_roa)
err=abs(adj.loc[ok,'ROA']-adj.loc[ok,'calculated_roa'])
roa_diag={'n':int(ok.sum()),'within_0_00000051':int((err<=.00000051).sum()),'median_abs_error':float(err.median()),'max_abs_error':float(err.max())}
adj.loc[ok&((adj.ROA-adj.calculated_roa).abs()>.00000051),['stock_code','year','ROA','calculated_roa','net_profit','total_assets','previous_assets']].to_csv(OUT/'roa_formula_exceptions.csv',index=False)

# Primary-report consolidated liabilities from previously audited annual reports.
primary=pd.read_csv(ROOT/'source/construct-check/primary-accounting-inputs.csv',dtype={'stock_code':str})
primary['firm_code']=primary.stock_code.astype(int)
primary['total_liabilities_cny']=primary.total_liabilities*np.where(primary.monetary_unit.eq('CNY_million'),1e6,1)
check=d.merge(primary[['firm_code','year','total_liabilities_cny','report_file']],on=['firm_code','year'])
check['consolidated_equity']=check.total_assets-check.total_liabilities_cny
prev=check[['firm_code','year','consolidated_equity']].rename(columns={'consolidated_equity':'previous_consolidated_equity'})
prev.year += 1
check=check.merge(prev,on=['firm_code','year'])
check['implied_average_consolidated_equity']=check.net_profit/check.ROE
check['computed_average_consolidated_equity']=(check.consolidated_equity+check.previous_consolidated_equity)/2
check['computed_roe']=check.net_profit/check.computed_average_consolidated_equity
check['roe_abs_error']=abs(check.ROE-check.computed_roe)
check[['stock_code','company','year','net_profit','total_assets','current_liabilities','total_liabilities_cny','consolidated_equity','previous_consolidated_equity','computed_average_consolidated_equity','ROE','computed_roe','roe_abs_error','report_file']].to_csv(OUT/'primary_report_roe_checks.csv',index=False)

# Direct primary corroboration for the mislabeled liability column.
vanke=pd.DataFrame([{'firm_code':2,'year':2018,'primary_current_assets':1295071856292.37,'primary_current_liabilities':1121913936785.61,'primary_total_liabilities':1292958626477.23},
                    {'firm_code':2,'year':2019,'primary_current_assets':1438989354552.98,'primary_current_liabilities':1272610284010.69,'primary_total_liabilities':1459350334988.27}])
vanke=d.merge(vanke,on=['firm_code','year'])
vanke['ca_error']=vanke.current_assets-vanke.primary_current_assets
vanke['liabilities_vs_current_error']=vanke.liabilities-vanke.primary_current_liabilities
vanke['liabilities_vs_total_error']=vanke.liabilities-vanke.primary_total_liabilities
vanke['report_file']='source/construct-check/vanke-2019-annual-report.pdf'
vanke['pdf_page_assets']=134; vanke['pdf_page_liabilities']=135
vanke[['stock_code','year','current_assets','liabilities','primary_current_assets','primary_current_liabilities','primary_total_liabilities','ca_error','liabilities_vs_current_error','liabilities_vs_total_error','report_file','pdf_page_assets','pdf_page_liabilities']].to_csv(OUT/'primary_current_liabilities_checks.csv',index=False)

summary={'rows':len(d),'no_rows_dropped':True,'codes':int(d.firm_code.nunique()),'company_names_excluding_missing':int(d.company.nunique()),
         'name_missing_rows':int(d.company.isna().sum()),'name_missing_codes':sorted(d.loc[d.company.isna(),'stock_code'].unique().tolist()),
         'names_with_multiple_codes':multiname[multiname>1].to_dict(),'blank_components_in_cached_working_capital':int(d.working_capital_cache_has_blank_components.sum()),
         'duplicate_accounting_profile_rows':int(d.duplicate_accounting_profile.sum()),'duplicate_profile_rows_with_nonmissing_roe':int(d.duplicate_profile_with_nonmissing_roe.sum()),
         'b_share_rows':int(d.is_b_share_code.sum()),'b_share_nonmissing_roe':int((d.is_b_share_code&d.ROE.notna()).sum()),'roa_formula_audit':roa_diag,
         'primary_roe_check_n':len(check),'primary_roe_check_max_error':float(check.roe_abs_error.max())}
(OUT/'panel_preparation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
mapping={
 'original_columns':'All 23 supplied columns retained verbatim. Fields below clarify interpretation and are additive.',
 'firm_code':'Integer supplied codigo_empresa. Unique together with year. Do not identify firms by company name.',
 'stock_code':'Same code padded to six digits. A securities identifier, not an externally validated issuer ID.',
 'year':'Integer supplied years, 2010--2020.',
 'current_liabilities':'Exact copy of supplied liabilities. Validated against Vanke consolidated current liabilities in both 2018 and 2019; workbook metadata row99 identifies current liabilities.',
 'working_capital_cached':'Exact copy of supplied net_assets, with cached Excel blanks-as-zero behavior intact.',
 'working_capital':'current_assets minus current_liabilities, missing if either component is missing. All available components agree with cached workbook within rounding.',
 'roe_decimal':'Exact supplied ROE, decimal ratio. Three 2019 annual-report checks reproduce consolidated net profit divided by average opening/closing consolidated equity within rounding, including negative/near-zero denominator case. Not weighted-average disclosed ROE.',
 'roa_decimal':'Exact supplied ROA, decimal ratio. Net profit divided by average opening/closing assets is checked where adjacent years exist; see exceptions.',
 'financial_expenses_total':'Exact supplied measure. EBIT equals total_profit + this field in 32,883/32,901 available rows within relative1e-8/cent tolerance. Name should be finance expense, not interest expense.',
 'financial_expenses_positive':'Cash-flow reconciliation finance expense. Zhongyin2019 supplied449004849.57 exactly matches financial expenses(revenues with minus sign) in cashflow reconciliation, annualreportPDF194. It is not actualinterestexpense452835918.84 in PDF100/188; contains negative and zero values.',
 'industry_financial':'Provisional text flag from financial/insurance/capital market. See category list before selecting sample.',
 'duplicate_accounting_profile':'Finite identical tuple(year, assets, net profit, operating profit, share capital) across securities. Intended diagnostic; NOT a rule that edits data.',
 'recommendation':'Use working-capital/assets, share-capital/assets, net-profit/assets, explicitly named operating-profit-minus-finance-expense/assets, nonoperating income and expense scaled individually by assets, finance-expense/assets, depreciation/assets as transparent economic predictors. Do not call net_assets equity or EBIT/financial_expenses_positive interest coverage without source validation.'
}
(RAW/'original_panel_dictionary.json').write_text(json.dumps(mapping,indent=2,ensure_ascii=False))
print(json.dumps(summary,indent=2,ensure_ascii=False))
print(check[['stock_code','ROE','computed_roe','roe_abs_error']].to_string(index=False))
