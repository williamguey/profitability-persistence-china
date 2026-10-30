# Profitability Persistence and Accounting Information in Chinese Firms

Reproduction materials for the study by **Lilian Chien Jiang, William Guey, Ziyi Qiao, Vitor D. de Moura, Yushi Ma, and José O. Gomes**.

The repository contains analysis code, pinned dependencies, method configurations, aggregate results, and the inputs and code for five figures. The empirical study uses historical Chinese company data through 2020. Full commands, result mappings and diagnostic scope are in the [reproduction guide](docs/REPRODUCTION.md).

## Reproduce the five figures

Tested with Python 3.12.14. Run from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/plot_loop1_illustrated.py
```

After installing dependencies, the plotting command works offline using the included aggregates. It writes PDF/PNG pairs and a rendering manifest to `output/illustrated-figures/`, without fitting models or drawing bootstrap samples.

| Figure | Output filename |
|---|---|
| Accounting ratios and next-year ROE | `accounting-relationships` |
| Two successive forecast gains | `gain-decomposition` |
| Fitted profitability-persistence functions | `persistence-shape` |
| Annual forecast errors and accounting gains | `annual-gains` |
| Distributions of the eight accounting signals | `signal-distributions` |

## Re-estimate the models

- **Public panel:** the scripts download and verify Peng's [Mendeley Data V2 deposit](https://doi.org/10.17632/wx8nwsrrbr.2), made available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Follow the [public-panel commands](docs/REPRODUCTION.md#public-panel-download-and-estimate).
- **Primary accounting panel:** exact re-estimation requires authorized access to the same CSMAR-derived workbook. The private workbook and company-level forecasts are not distributed. Extraction and estimation instructions, including source hashes, are in the [primary-panel guide](docs/REPRODUCTION.md#author-supplied-accounting-panel-authorized-source-required).

The two panels overlap and use different ROE definitions and predictor sets. Their results should not be pooled. Extended diagnostics are retained with their scope stated in the guide; the later table builder is not a complete rebuild of the paper.

## Files and integrity

- `code/`, `analysis/`: estimation and diagnostic code, plus aggregate outputs.
- `config/`: method specifications, environment and source checksums.
- `figure-data/`, `output/`: aggregate figure inputs and rendered figures.
- `source/construct-check/`: six cited public annual-report cases and source locators.
- `CITATION.cff`: the study's confirmed authors and repository citation metadata.

Verify the distributed files with:

```sh
shasum -a 256 -c MANIFEST.sha256
```

The repository excludes the manuscript, private source data, individual forecasts, model-fit records and reviews. A curated `.gitignore` allowlist keeps newly generated files out of commits until their contents are reviewed.
