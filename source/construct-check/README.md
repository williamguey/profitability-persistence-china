# Public annual-report cases

`primary-accounting-inputs.csv` contains six deliberately selected published cases: Vanke, Ping An Bank and Ningxia Zhongyin Cashmere in 2018 and 2019. Monetary units and report page locators are explicit. These quantities support accounting-definition checks, not a general audit of either panel. `primary-annual-reports.bib` supplies report URLs and attribution.

The raw annual-report PDFs and both research workbooks are not bundled. Run `python analysis/check_primary_constructs.py` after the public-panel extraction to verify the parent-attributable ROE and liability/equity checks. The accounting-panel preparation separately reconstructs consolidated ROE for the three 2019 cases, using the licensed source workbook's consolidated values and these public liabilities.
