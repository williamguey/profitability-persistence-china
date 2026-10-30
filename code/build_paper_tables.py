from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/paper-assets'; OUT.mkdir(parents=True,exist_ok=True)
A=ROOT/'analysis/original-forecast/transparent_main'
B=ROOT/'analysis'
a=pd.read_csv(A/'metrics.csv').set_index(['partition','model'])
b=pd.read_csv(B/'metrics.csv').set_index(['partition','model'])
ca=pd.read_csv(A/'paired_bootstrap.csv').set_index(['partition','reference','candidate'])
cb=pd.read_csv(B/'paired_bootstrap.csv').assign(partition='2015-2019').set_index(['partition','reference','candidate'])
fa=pd.read_csv(ROOT/'analysis/original-forecast/nesting_order_diagnostic/metrics.csv').set_index(['partition','model'])
fb=pd.read_csv(ROOT/'analysis/public-factorial/metrics.csv').set_index(['partition','model'])
na=pd.read_csv(ROOT/'analysis/original-forecast/nesting_order_diagnostic/paired_bootstrap.csv').rename(columns={'comparison':'contrast'}).set_index(['partition','contrast'])
nb=pd.read_csv(ROOT/'analysis/public-factorial/paired_bootstrap.csv').set_index(['partition','contrast'])
models=[('mean','Historical mean'),('median','Historical median'),('random_walk','Current ROE'),('ols_ar','OLS: linear'),('ols_piecewise','OLS: sign-asymmetric'),('ols_signals','OLS: asymmetry plus signals'),('lad_ar','LAD: linear'),('lad_linear_signals','LAD: linear plus signals'),('lad_piecewise','LAD: sign-asymmetric'),('lad_signals','LAD: asymmetry plus signals')]
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Forecast accuracy, target years 2015--2019}\label{tab:main}',r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{Accounting panel} & \multicolumn{2}{c}{Public panel}\\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',r'Model & MAE & RMSE & MAE & RMSE\\',r'\midrule']
for model,label in models:
    va=(fa if model=='lad_linear_signals' else a).loc['2015-2019',model]; vb=(fb if model=='lad_linear_signals' else b).loc['2015-2019',model]
    lines.append(f'{label} & {va.mae:.3f} & {va.rmse:.3f} & {vb.mae:.3f} & {vb.rmse:.3f}'+r'\\')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\caption*{Errors are ROE percentage points. Common evaluation rows: accounting panel 15,258 forecasts/3,542 securities; public panel 13,970/3,270. Estimation uses earlier target years. Current ROE is unmodified; panels have distinct targets and predictors.}',r'\end{table}'])
(OUT/'main-table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[H]',r'\centering\small',r'\caption{Annual LAD forecast gains in the persistence-first ordering}\label{tab:annual}',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrll}',r'\toprule',r'Year & Full MAE & Asymmetry gain [95\% interval] & Accounting gain [95\% interval]\\',r'\midrule']
for data,metric,panel in [(na,a,'Accounting panel'),(nb,b,'Public panel')]:
    lines.append(r'\multicolumn{4}{l}{'+panel+r'}\\')
    for y in range(2015,2021):
        ix=data.loc[str(y),'asymmetry_first']; iz=data.loc[str(y),'accounting_second']; full=metric.loc[str(y),'lad_signals'].mae
        if y==2020:lines.append(r'\addlinespace[2pt]')
        lines.append(f'{y} & {full:.3f} & {ix.mae_reduction_pp:.3f} [{ix.ci_low_pp:.3f}, {ix.ci_high_pp:.3f}] & {iz.mae_reduction_pp:.3f} [{iz.ci_low_pp:.3f}, {iz.ci_high_pp:.3f}]'+r'\\')
    lines.append(r'\midrule' if panel=='Accounting panel' else r'\bottomrule')
lines.extend([r'\end{tabular}',r'\caption*{All quantities are ROE percentage points. Asymmetry is added to linear LAD; accounting signals are then added to asymmetry. Full includes both. Intervals use 2,000 security-cluster draws. Each 2020 model is re-estimated on targets through 2019.}',r'\end{table}'])
(OUT/'annual-table.tex').write_text('\n'.join(lines)+'\n')
# Fixed forecasts: the two partitions are distinct views of the same primary sample.
group=pd.read_csv(ROOT/'analysis/original-forecast/fixed_forecast_decomposition/group_metrics.csv')
pub=pd.read_csv(ROOT/'analysis/public-sign-constructs/public_origin_sign_metrics.csv').set_index(['origin_sign','model'])
lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Four LAD forecasts by origin sign and provenance, 2015--2019}\label{tab:contrasts}',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Group & $N$ & $L$ & $L+Z$ & $A$ & $A+Z$ & AE share\%\\',r'\midrule',r'\multicolumn{7}{l}{Accounting panel: origin sign}\\']
for key,label in [('negative','Negative ROE'),('nonnegative','Nonnegative ROE')]:
    r=group[(group.group_scope=='origin_sign')&(group.origin_roe_group==key)].iloc[0]
    vals=[r[m+'_mae'] for m in ['lad_ar','lad_linear_signals','lad_piecewise','lad_signals']]
    lines.append(f'{label} & {int(r.n):,} & '+' & '.join(f'{v:.3f}' for v in vals)+f' & {r.lad_signals_share_total_absolute_error_pct:.2f}'+r'\\')
lines.extend([r'\midrule',r'\multicolumn{7}{l}{Accounting panel: deposit membership $\times$ origin sign}\\'])
for provenance,sign,label in [('matched_at_or_after_listing','negative','Matched / negative'),('matched_at_or_after_listing','nonnegative','Matched / nonnegative'),('prelisting','nonnegative','Prelisting / nonnegative'),('unmatched','negative','Unmatched / negative'),('unmatched','nonnegative','Unmatched / nonnegative')]:
    r=group[(group.group_scope=='provenance_x_origin_sign')&(group.provenance==provenance)&(group.origin_roe_group==sign)].iloc[0]
    vals=[r[m+'_mae'] for m in ['lad_ar','lad_linear_signals','lad_piecewise','lad_signals']]
    lines.append(f'{label} & {int(r.n):,} & '+' & '.join(f'{v:.3f}' for v in vals)+f' & {r.lad_signals_share_total_absolute_error_pct:.2f}'+r'\\')
lines.extend([r'\midrule',r'\multicolumn{7}{l}{Public panel: origin sign}\\'])
for groupname,label,neg in [('negative_origin','Negative ROE',100),('nonnegative_origin','Nonnegative ROE',0)]:
    vals=[pub.loc[groupname,m].mae_pp for m in ['lad_ar','lad_linear_signals','lad_piecewise','lad_signals']];r=pub.loc[groupname,'lad_signals']
    lines.append(f'{label} & {int(r.n):,} & '+' & '.join(f'{v:.3f}' for v in vals)+f' & {r.share_total_absolute_error_pct:.2f}'+r'\\')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\caption*{$L$ is linear persistence, $A$ sign-asymmetric persistence, and $Z$ accounting signals. Model columns are MAE in ROE percentage points. AE share is the share of full-model absolute error in that source; each partition sums to 100\%. Matched origins are at or after IPO year; prelisting origins have no negative ROE. Predictions are fixed pooled forecasts, without subgroup refitting.}',r'\end{table}'])
(OUT/'contrasts-table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[H]',r'\centering\small',r'\caption{Sensitivity of the primary 2015--2019 accounting comparison}\label{tab:sensitivity}',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Specification & $N$ & Linear & Asym. & Full & LAD gain & OLS gain\\',r'\midrule']
for folder,label in [(A,'Primary'),(A/'industry_controls','Industry controls'),(A/'without_000982_refit','Zhongyin excluded'),(A.parent/'legacy_ratios','Alternative ratios'),(A.parent/'verified_listing_sensitivity','Deposit-matched subsample')]:
    m=pd.read_csv(folder/'metrics.csv').set_index(['partition','model'])
    base=m.loc['2015-2019','lad_piecewise'];full=m.loc['2015-2019','lad_signals']
    og=m.loc['2015-2019','ols_piecewise'].mae-m.loc['2015-2019','ols_signals'].mae
    lines.append(f'{label} & {int(base["n"]):,} & {m.loc["2015-2019","lad_ar"].mae:.3f} & {base.mae:.3f} & {full.mae:.3f} & {base.mae-full.mae:.4f} & {og:.4f}'+r'\\')
lines.extend([r'\bottomrule',r'\end{tabular}',r'\caption*{Errors and gains are ROE percentage points. The three MAE columns use LAD; full adds signals to asymmetry. Gains compare full with asymmetric under the named estimator. Controls enter both specifications. Exclusion and deposit-matching restrictions apply to training and evaluation before preprocessing; deposit matching retains at-or-after-IPO origins in the end-period-observed public cohort. Alternative formulas appear in Appendix B.}',r'\end{table}'])
(OUT/'sensitivity-table.tex').write_text('\n'.join(lines)+'\n')
# Both paths through the four LAD specifications, on identical pp scales.
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(6.9,3.1),sharey=True)
contrasts=['asymmetry_first','asymmetry_second','accounting_first','accounting_second']
labels=['Asymmetry first','Asymmetry second','Accounting first','Accounting second']
for ax,data,title in [(axs[0],na,'A. Accounting panel'),(axs[1],nb,'B. Public panel')]:
    for pos,key in enumerate(contrasts):
        r=data.loc['2015-2019',key]
        ax.errorbar(r.mae_reduction_pp,3-pos,xerr=[[r.mae_reduction_pp-r.ci_low_pp],[r.ci_high_pp-r.mae_reduction_pp]],fmt=('o' if pos%2==0 else 's'),mfc=('black' if pos%2==0 else 'white'),color='black',capsize=3,ms=5)
    ax.axvline(0,color='.7',lw=.7);ax.axhline(1.5,color='.85',lw=.7)
    ax.set_xlim(-.025,.7);ax.set_ylim(-.6,3.6);ax.set_yticks([3,2,1,0],labels);ax.set_title(title,fontsize=10,loc='left');ax.set_xlabel('MAE reduction (percentage points)')
fig.tight_layout(w_pad=1.8)
fig.savefig(OUT/'gain-decomposition.pdf',bbox_inches='tight');fig.savefig(OUT/'gain-decomposition.png',dpi=180,bbox_inches='tight')


# Rebuild the sample appendix directly from the distributed aggregate counts.
sample_a=pd.read_csv(A/'sample_counts.csv').set_index('target_year')
sample_b=pd.read_csv(B/'sample_counts.csv').set_index('target_year')
lines=[r'\begin{table}[htbp]',r'\centering\small',
       r'\caption{Eligible forecast pairs by target year}\label{tab:sample}',
       r'\begin{tabular}{lrrrr}',r'\toprule',
       r' & \multicolumn{2}{c}{Accounting panel} & \multicolumn{2}{c}{Public panel}\\',
       r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
       r'Target year & Pairs & Negative origin & Pairs & Negative origin\\',r'\midrule']
for year in sorted(set(sample_a.index)|set(sample_b.index)):
    values=[]
    for frame in [sample_a,sample_b]:
        values.extend([f'{int(frame.loc[year,c]):,}' if year in frame.index else '--' for c in ['n','negative_origin']])
    lines.append(str(year)+' & '+' & '.join(values)+r'\\')
lines.extend([r'\bottomrule',r'\end{tabular}',
              r'\caption*{Each pair links the same security in consecutive calendar years. Main evaluation targets are 2015--2019; preceding targets enter initial training and 2020 is reported separately. Negative origin means current ROE below zero.}',
              r'\end{table}'])
(OUT/'sample-table.tex').write_text('\n'.join(lines)+'\n')
