# Full reproduction guide

These materials accompany *Profitability Persistence and Accounting Information in Chinese Firms*. The reproduction code, aggregate results and figure materials are distributed through this GitHub repository only. Run every command below from the repository root. No paper DOI or arXiv identifier is assigned here.

The package contains Python analysis code, pinned dependencies, method configurations, aggregate outputs, five illustrated figures with their aggregate plotting inputs, and six explicitly cited public annual-report cases. The full previously audited analysis bundle is retained for transparency, including later diagnostics not all used in this econ.GN revision. Its later table builder and historical paper mapping are identified below. No manuscript or TeX source, source workbook, original R source file, private worksheet, individual forecast, eligible-pair record, fitted-model record, optimality certificate, private correspondence or review is distributed.

## Offline figure reproduction

The tested environment is Python 3.12.14. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/plot_loop1_illustrated.py
```

After dependencies are installed, the final command needs neither network access nor source data. It regenerates five PDF/PNG figure pairs under `output/illustrated-figures/`: `accounting-relationships`, `persistence-shape`, `gain-decomposition`, `annual-gains` and `signal-distributions`. It also writes `render-manifest.json` there. The script reads the frozen `figure-data/` inputs, leaves those inputs unchanged, and writes no manuscript files. It does not estimate models or draw bootstrap samples. The gain-interval panel uses the included aggregate paired-bootstrap results.

The 10,040 rows of `persistence-shape-function-grid.csv` are evaluations of ten saved model functions on a chosen plotting grid; they are not company observations or individual forecasts. Hexagon centers/counts, quantile-bin medians, histogram bins and empirical quantiles are aggregate display inputs. The one-dimensional histograms retain the original finite-observation denominator, so omitted tails are not renormalized. The negative-ROE function branch preserves its separate left-hand limit at zero. Stored reconstruction checks describe checks performed when the source aggregates were prepared; the offline renderer separately verifies the function grid against the saved aggregate coefficients.

The optional command below regenerates five LaTeX fragments and a different, later-analysis gain figure under `output/paper-assets/`:

```sh
python code/build_paper_tables.py
```

Those later assets are retained for the extended analysis workflow and are not a complete rebuild of this econ.GN manuscript. No `.tex` files are included in the archive. This builder derives its output root from its own package location and cannot overwrite the separately maintained manuscript when run from the extracted package.

## Scope and data access

Exact primary accounting-panel re-estimation requires authorized access to the same CSMAR-derived `Dados Tese.xlsx` snapshot, SHA256 `759de2c9a3ecd6ab03f56b7ebc17ebf6d8b937d25bc9bb21bcb20b29273ee8ab`. The package supplies the extraction, preparation, estimation and diagnostic code but does not grant access to or redistribute that workbook. A different workbook vintage can change observations and estimates. Public-panel re-estimation requires downloading the cited deposited workbook and verifying its hash, as documented below.

After reproducing both primary pipelines and running `code/describe_original.py`, `python code/prepare_illustrated_figure_data.py` rebuilds the illustrative aggregates and figures from the locally generated eligible pairs, saved individual forecasts and fitted-model records. This preparation command requires those local source-derived files; the offline figure command above does not. The portable preparation script preserves the original scientific calculations, removes internal review/manuscript dependencies, and writes only to this package's `figure-data/` and `output/illustrated-figures/` directories. No re-estimation was performed to assemble this attachment.

The full analysis commands below create local company-level intermediates, predictions and fitted-model records. These are excluded from the distributed archive and must not be mistaken for redistributable package content. The six public annual-report cases in `source/construct-check/` are the only distributed company-level source table, with citations and page locators.

`config/environment.json` records the runtime; `config/forecast_design.json` and the other method files specify the primary analysis and extended diagnostics. `config/source-checksums.json` records the original code and source provenance. `base-package-validation.json` preserves the historical audit of the 176-file base bundle; `ancillary-validation.json` describes the 196-file ancillary attachment from which this repository was prepared. `MANIFEST.sha256` covers all distributed files except itself. Pinned dependencies include NumPy, pandas, SciPy/HiGHS, Matplotlib, openpyxl, statsmodels and pypdf. No R installation, API key or AI service is needed. A fresh network installation was not rerun for this repository release.

## Public panel: download and estimate

The source is Peng (2022), *Data including roe and revenue growth rate of listed companies in China’s A-share market from 2009 to 2020*, Mendeley Data V2, [doi:10.17632/wx8nwsrrbr.2](https://doi.org/10.17632/wx8nwsrrbr.2), deposited under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The depositor identifies Wind as its source. The downloader retrieves the unchanged workbook and verifies SHA256 `9cb04bd1d5ed44383d7e54351c8e8c7271797ccd74ecac3934e06302141dd75f`. Only the full-sample worksheet is used; the other worksheets contain portfolios selected using full-period returns.

```sh
python code/download_public_data.py
python code/extract_public_panel.py
python analysis/check_primary_constructs.py
python code/analyze_forecasts.py
python code/robustness.py
python code/analyze_public_factorial.py
python code/audit_public_zeros.py
```

Primary results are written to `analysis/metrics.csv`, `paired_bootstrap.csv`, `sample_counts.csv` and `robustness_refit.csv`. The four-cell and zero diagnostics write to `analysis/public-factorial/`. The four-cell diagnostic fits only the additional linear-ROE-plus-signals LAD model and reuses all existing predictions. Zero audits retain values as deposited; numerical zeros do not establish a missing-value rule. This deposited cohort is not an independently verified market census or point-in-time database.

## Author-supplied accounting panel: authorized source required

The CSMAR-derived workbook is not redistributed. Exact reproduction requires authorized access to the same `Dados Tese.xlsx` snapshot, SHA256 `759de2c9a3ecd6ab03f56b7ebc17ebf6d8b937d25bc9bb21bcb20b29273ee8ab`. A different data vintage may produce different observations or estimates.

```sh
python analysis/original-recovery/extract_original.py --workbook "/path/to/Dados Tese.xlsx"
python analysis/original-recovery/prepare_panel.py
python code/analyze_original.py --mode all --output analysis/original-forecast/transparent_main
python code/analyze_original.py --mode primary --signal-set legacy
python code/describe_original.py --output analysis/original-forecast/transparent_main
python code/diagnose_original.py --output analysis/original-forecast/transparent_main
python code/analyze_original_nesting_order.py
```

The extractor accepts `--input-dir` and `--output-dir`; its default input location is `data/original-private/`. Original R files are optional and never executed. Preparation retains source rows and columns while clarifying current liabilities and working capital. The verified prepared-panel hash is `b6290b3629760ef272f75c95e78d1b9f4029e1d4dfd152617135def5a3152b23`; the forecast configuration checks it. The three modes `primary`, `exclude` and `industry` run individually, while `all` runs all three. `--signal-set legacy` preserves the earlier literal ratio family as a separate sensitivity. All extraction/preparation scripts are identical to those independently checked in the project.

After **both** public and original primary pipelines have run:

```sh
python code/analyze_original_listing_sensitivity.py
python code/audit_listing_and_origin_extremes.py
python code/decompose_original_fixed_forecasts.py
python code/describe_public_sign_and_constructs.py
python code/describe_original_provenance_coverage.py
python code/describe_original_accounting_lines_identity.py
python code/audit_public_sector_signal_sign.py
python code/analyze_original_combined_lines.py
python code/describe_original_membership_missingness.py
python code/describe_original_loss_recovery_availability.py
python code/analyze_original_income_identity.py
python code/analyze_original_drop_np.py
python code/analyze_original_sign_interaction.py
python code/bound_original_missing_target_losses.py
python code/build_paper_tables.py
```

These commands regenerate the deposited-listing-reference sensitivity, listing/origin-tail diagnostics, fixed-forecast subgroups, source-construct comparisons, provenance/coverage and accounting-line summaries, public-sector/sign audits, the combined-income sensitivity, membership/missingness and recovery-availability descriptions, the income-identity and net-profit omission diagnostics, and final presentation assets. The provenance diagnostic checks an included aggregate audit of 361 blank source ROE cells; that prior physical-cell audit is preserved as `analysis/original-forecast/roe_missingness_audit.json`. It does not infer why source values are unavailable.

The combined-income sensitivity fits six additional LAD models, replacing operating-profit/assets and net-nonoperating-income/assets with their raw sum before training transformations; the other six ratios and eligible forecast rows are unchanged. Its full local output includes forecasts and optimality certificates, which are intentionally excluded from this distribution. It is a post-analysis sensitivity to the income-line split, not a replacement selected for the primary model or an identified effect of the 2017 accounting reform. The listing screen applies to every training and evaluation origin, retaining matched valid reference dates with fiscal origin year at least the listing year. It changes reference coverage as well as listing eligibility; it does not prove information availability at a historical trading date. The origin-tail diagnostic uses each fold's training-origin ROE percentiles, without refitting or changing primary predictions.

The membership/missingness diagnostic adds source code/name mappings, literal code prefixes, full unmatched export-entry histories, raw asset and implied average-equity/assets distributions, training-tail shares, and availability summaries. Its implied denominator is `(net_profit/total_assets)/(ROE_percent/100)` and is not an independently observed equity field. The name checks distinguish nonmissing names from a missing-name category; the new `code_name_mapping_counts.csv` provides that explicit distinction. The recovery-availability diagnostic starts from all known-nonfinancial, negative-ROE origins in 2014–2018 and separately counts finite future ROE, an existing blank future ROE cell, an infinite value, and absence of a next-year source record. Recovery percentages use only finite observed next-year profit within each group, with unavailable profits counted separately. Both diagnostics use existing source fields and saved forecasts, require no new fits, and leave primary inputs unchanged. They identify conditional associations, not reasons for missing values or verified legal issuer/listing status. Their two method configurations retain the exact original UTC freeze timestamps. Included summaries preserve the validated project code hashes; `config/source-checksums.json` also records the distributed code hashes after portable configuration-path substitutions.

The income-identity diagnostic checks `NP − OP − NOI + NOE + TAX` in the raw source, its asset-scaled counterpart, and the effect of separate training-only clipping and imputation. It records availability, numerical rank and singular-value summaries, reconstructs existing asymmetric forecasts, and reports aggregate loss-target exclusions. It imposes no accounting correction. The following drop-net-profit sensitivity runs only when the frozen diagnostic criterion is met (at least 90% of complete main origins have absolute asset-scaled residual at most `1e-8`). It fits six additional LAD models omitting `NP/assets`, with unchanged eligible rows and verified shared transformations, and compares them with the saved asymmetric and full forecasts. These are seven named ratios, not a claim of seven independent dimensions. The original full-model estimates remain primary. The identity diagnostic must precede the omission command. Its individual source case, the new row-level forecasts and the six optimality certificates are generated locally but are not distributed. Aggregate summaries preserve original run timestamps and code hashes; distributed hashes are recorded separately in the provenance configuration.

The within-sign interaction diagnostic uses the saved four LAD forecasts, with no refitting. It calculates accounting-second minus accounting-first separately for negative and nonnegative origin ROE, using jointly paired security-cluster draws within each group. It reproduces the eight existing component intervals and verifies the identical asymmetry-second-minus-first reading on every draw. The three distributed CSVs contain only group MAEs, component checks and the two interaction intervals. The source plan was frozen before this calculation; its distributed copy omits only an internal assessment-file routing line. Original and distributed hashes are recorded separately. Full execution requires the existing four-cell predictions created by the earlier pipeline; those individual predictions are not distributed.

The missing-target-loss bound applies the saved primary training transformations and coefficients to exactly 63 known-nonfinancial negative-ROE origins excluded because next-year ROE is missing (62 blank cells and one absent next-year record). It first reconstructs all 45,774 retained forecasts across the three models. For each contrast, the retained loss-difference sum plus or minus the sum of the 63 absolute forecast gaps, divided by 15,321, gives a deterministic envelope for arbitrary finite missing target values. These are contrast-wise bounds, not confidence intervals; their endpoints need not be jointly attainable. They hold the fitted models fixed, do not impute outcomes, and cover neither other exclusions nor refitted models. Full execution requires the authorized prepared panel, eligible pairs, saved primary predictions and fit records created by the preceding pipeline. Only six aggregate CSVs and a provenance summary are distributed. The frozen plan retains its exact UTC timestamp and scientific specification; only its internal assessment-file routing line is omitted.

Optional source-history reconstruction:

```sh
python analysis/original-recovery/reproduce_original.py
```

This recreates earlier workbook regressions using the supplied R formulas; it does not reconstruct the final thesis PDF's undocumented transformations and is not the paper's forecasting model. An optional additional annual comparison figure can be regenerated with `python code/plot_original.py`.

## Extended-analysis outputs and historical paper mapping

The following mapping describes the broader analysis bundle retained for completeness. Later diagnostics and its generated table/figure designs are not all used by this econ.GN manuscript. Generated TeX assets are excluded from the attachment; the builder recreates them locally.

`A/` means `analysis/original-forecast/transparent_main/`; `P/` means `analysis/`; `F/` means `analysis/original-forecast/nesting_order_diagnostic/`; `G/` means `analysis/public-factorial/`.

| Paper result or generated asset | Aggregate input |
|---|---|
| Forecast accuracy (`tab:main`, `main-table.tex`) | `A/metrics.csv`, `P/metrics.csv`, `F/metrics.csv`, `G/metrics.csv`, main partition |
| Annual errors and asymmetry/accounting intervals (`tab:annual`, `annual-table.tex`) | Primary metrics plus `F/paired_bootstrap.csv`, `G/paired_bootstrap.csv`, annual asymmetry-first and accounting-second contrasts |
| Fixed four-cell sign and provenance groups (`tab:contrasts`, `contrasts-table.tex`) | `analysis/original-forecast/fixed_forecast_decomposition/group_metrics.csv`, `analysis/public-sign-constructs/public_origin_sign_metrics.csv` |
| Both nesting orders (`fig:gains`, `gain-decomposition.*`) | `F/paired_bootstrap.csv`, `G/paired_bootstrap.csv` |
| Sample appendix (`tab:sample`, `sample-table.tex`) | `A/sample_counts.csv`, `P/sample_counts.csv` |
| Sensitivities (`tab:sensitivity`, `sensitivity-table.tex`) | Primary, industry, exclusion-refit, legacy and listing-reference `metrics.csv` |
| Fixed-group contrasts, profit signs and transitions | Aggregate tables under `analysis/original-forecast/fixed_forecast_decomposition/` |
| Public sign contributions and cross-source ROE differences | Aggregate tables under `analysis/public-sign-constructs/` |
| Listing-reference gain and coverage | `analysis/original-forecast/verified_listing_sensitivity/{metrics,paired_bootstrap,sample_counts}.csv` |
| Listing audit and origin-extreme strata | Four aggregate CSVs under `analysis/original-listing-audit/` |
| Public zero counts and raw-cell provenance | `G/zero_audit.csv`, `G/zero_provenance.json` |
| Sign-by-provenance gains, profit recovery and source coverage | Aggregate tables under `analysis/original-forecast/provenance_coverage_diagnostic/` |
| Annual income-line and security-name/profile summaries | Aggregate tables under `analysis/original-forecast/accounting_line_identity_diagnostic/` |
| Combined-income sensitivity | `analysis/original-forecast/combined_line_sensitivity/{metrics,paired_bootstrap}.csv` |
| Public zero-margin sectors and raw signal/ROE signs | Aggregate tables and workbook-header metadata under `analysis/public-sector-signal-sign/` |
| Membership, assets, implied denominator, code/name mappings and ROE/industry availability | Aggregate tables under `analysis/original-forecast/membership_missingness_diagnostic/` |
| Profit recovery conditional on next-year ROE availability | `analysis/original-forecast/loss_recovery_availability_diagnostic/recovery_by_target_roe_availability.csv` |
| Income identity, transformations, fitted loss-branch summaries and tail-mean exclusions | Aggregate tables under `analysis/original-forecast/income_identity_diagnostic/` |
| Net-profit/assets omission sensitivity | `analysis/original-forecast/drop_np_sensitivity/{metrics,paired_bootstrap}.csv` |
| Within-sign change in accounting gains | `analysis/original-forecast/sign_interaction_diagnostic/interaction_intervals.csv` |
| Fixed-model gain bounds for 63 missing loss targets | `analysis/original-forecast/missing_target_loss_bound/gain_bounds.csv` |
| OLS/LAD RMSE under original exclusions | `F/exclusion_ols_lad_rmse.csv` |
| Public exclusion comparison | `P/robustness_refit.csv` |
| ROE construct checks (`tab:roecheck`) | Six public cases in `source/construct-check/primary-accounting-inputs.csv`; source-loading checks use the two construct/preparation scripts |
| Distribution, persistence and transitions | `A/descriptive_statistics.csv`, `A/persistence_coefficients.csv`, `A/sign_transitions.csv` |
| Tail concentration | Included `tail_concentration.csv` tables; generated from each pipeline's `tail_diagnostics.json` |
| Clipped carry-forward comparison | `A/clipped_persistence_diagnostic/{metrics,paired_bootstrap}.csv` |

Exact single-security error-attribution records are omitted; the two four-cell scripts regenerate these locally from predictions. The six explicitly cited annual-report cases are the only company-level numerical source table distributed. Report URLs and page locators accompany that file.

Errors are ROE percentage points. The two panels use different target definitions and populations and must not be pooled. Security overlap prevents interpreting them as independent population replications. Bootstrap intervals condition on fitted predictions and realized calendar years, retaining each sampled security's evaluation history without refitting or resampling calendar shocks. Two-order average gains are descriptive allocations, not causal contributions. Original fixed-group contrast intervals resample security histories within each group; their weighted pooled contributions are point estimates. The sign-by-provenance extension retains those same verified intervals and point-contribution scope. Public sign-contribution intervals instead resample both sign-group histories jointly across the entire security sample. The two procedures have distinct scopes. Cross-source ROE differences compare different accounting definitions and are not record validation. Observed profit transitions are conditional descriptions; absence from the raw 2020 export is not a legal survival or delisting classification. Clipped persistence, listing/reference screens, origin strata, four-cell nesting, zero and sector/sign audits, fixed-forecast subgroup decompositions, source-coverage/accounting-line summaries, cross-source comparisons, the combined-income sensitivity, membership/missingness descriptions, recovery-availability comparisons, income-identity checks, net-profit omission, within-sign interaction intervals and missing-target-loss bounds are explicitly post-analysis diagnostics; the primary estimates are preserved. Public financial companies remain included. Cross-source sector matches retain missing and unmatched classifications as unknown; company names are not used to infer sectors or legal issuer identities.

## Validation and distribution boundary

`base-package-validation.json` is the historical validation record for the audited 176-file base bundle. It records syntax, privacy, source-byte-identity, portable-path and presentation checks. Independent numerical QA covered 180 original fits (four configurations plus listing sensitivity), their 450 metric rows, and 90 LAD optimality certificates; the additional four-cell diagnostics add 12 LAD fits, 56 metric rows and 84 jointly regenerated bootstrap comparisons. Listing/origin-tail aggregates, public numerical zeros and 732 raw XLSX zero cells were independently checked. The later fixed-group analyses add checks of all 12 original group rows, 48 original contrasts, 600 profit-sign crosscounts, 24 agreement rows, two transition summaries and 20 public sign/model/contrast rows. Every reported subgroup interval was independently regenerated; cross-source tolerance counts were checked using exact decimal source arithmetic. The additional combined-income sensitivity was checked by independently reconstructing all six training designs, verifying all six primal/dual optimality certificates, and regenerating 21 metric rows and 21 paired-bootstrap comparisons. Independent checks also reproduced the provenance/coverage, annual-line/identity and public-sector/signal-sign aggregates. The membership/missingness checks independently reproduced all 72 group-count rows, 144 distribution rows, 29 prefix rows and all export-history, name/profile and missingness tables. A separate source-key checker reproduced all 96 recovery-availability rows and their observed-profit denominators. The income-identity analysis independently reproduced all source and preprocessing aggregates, the saved asymmetry coefficient summaries and tail means. The net-profit omission analysis independently reconstructed all six designs and primal/dual certificates and all 21 metric rows and 21 paired-bootstrap contrasts, without production-code imports or refitting. The within-sign interaction check independently regenerated both new intervals and all eight previous component intervals from saved forecasts. An isolated portable replay reproduced its three aggregate CSVs byte-for-byte and all scientific summary fields without fitting models. The missing-target-loss bound was independently checked using separate source-key joins, raw-ratio construction and scalar application of the saved models. The portable replay regenerated all six aggregate CSVs byte-for-byte and all scientific summary fields without fitting models or imputing outcomes. This is implementation verification, not a claim of publication acceptance or real-time forecasting validity.

To check the package manifest on macOS or Linux:

```sh
shasum -a 256 -c MANIFEST.sha256
```
