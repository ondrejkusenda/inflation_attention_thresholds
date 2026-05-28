# Replication package — *Inflation attention thresholds before and after the inflation peak*

Replication code for:

> Kusenda, O. and Marenčák, M. *Inflation attention thresholds before and
> after the inflation peak.* National Bank of Slovakia.

The paper studies whether the inflation threshold that triggers household
attention survives during disinflation. Using Google Trends search data for
21 Eurozone countries plus the United States, the United Kingdom, and
Switzerland over the 2021–2023 inflation surge, it estimates attention
thresholds separately before and after each country's inflation peak, and
tests whether the pre-peak structure carries over. GDELT news-coverage data
are used as a robustness check.

This package reproduces all figures, tables, and statistics in the paper.

---

## Headline findings reproduced

1. **Pre-peak threshold is universal.** The Hansen test rejects linearity in
   the pre-peak window for all 24 countries; cross-country variation in the
   threshold replicates the habituation result (higher average inflation →
   higher threshold).
2. **Post-peak threshold dissolves for ~half the sample.** For roughly half
   the countries no post-peak threshold is identified, and attention decays
   linearly during disinflation rather than switching off at a discrete point.
3. **Dissolution is genuine, not low power.** A residual-bootstrap power
   analysis confirms the Hansen test would detect the pre-peak threshold if it
   were still active.
4. **Symmetry is rejected everywhere.** A stacked Chow-type structural-break
   test rejects pre-/post-peak symmetry for every country.
5. **US is a fragile exception.** Under Google Trends the US shows a *higher*
   post-peak threshold (Δ = +2.95 pp), but the result rests on 4
   above-threshold observations and does not replicate under GDELT.

---

## How the code maps to the paper

| Paper element | Produced by |
|---------------|-------------|
| Eq. (1) single-threshold regression (Hansen 2000) | `threshold_model.threshold_regression_full` |
| Asymptotic Hansen p-value (Monte Carlo, Gaussian null) | `threshold_model.asymptotic_pvalue` |
| Bootstrap Hansen p-value (B = 1,999) | `threshold_model.bootstrap_pvalue` |
| Korenok–Munro filter (test post-peak only if inflation fell below γ_pre) | `threshold_model.run_threshold_analysis` |
| Threshold gap Δ = γ_post − γ_pre | `threshold_model.run_threshold_analysis` |
| Stacked Chow test (App. A.5) | `threshold_model.chow_test_stacked` |
| Power analysis (App. A.6) | `threshold_model.power_analysis_post_peak` |
| Figure 1 (Germany / UK threshold panels) | `plotting.plot_single_window` |
| Figure 2 (γ_pre vs γ_post scatter) | `plotting.plot_threshold_scatter` |
| Figures A.1–A.2 (time-series with threshold) | `plotting.plot_timeseries_single` |
| Tables A.1 / A.4 (country-level results, Google / GDELT) | `results/<source>/summary_<SOURCE>.xlsx` |

---

## Repository layout

```
inflation_attention_thresholds/
├── data/                     # Input CSVs (see data/README.md for schema)
│   ├── INFLATION_DATA.csv    # OECD MEI headline inflation (HICP / CPI)
│   ├── GOOGLE_DATA.csv        # Google Trends "inflation" search index
│   └── GDELT_DATA.csv         # GDELT news-coverage share
├── results/                  # Generated outputs (created on run)
│   ├── google/
│   │   ├── raw_results/      # Per-country JSON
│   │   ├── plots/            # Per-country diagnostic PDFs
│   │   └── summary_GOOGLE.xlsx
│   ├── gdelt/                # Same structure as google/
│   └── single_window/        # Figure 1 / A.1 / A.2 style figures
├── src/
│   ├── config.py             # Palette, plot style, country metadata, constants
│   ├── data_loading.py       # CSV readers with schema validation
│   ├── preprocessing.py      # prepare_data: merge + split at the inflation peak
│   ├── threshold_model.py    # Hansen model, Chow test, power analysis
│   ├── plotting.py           # All publication figures
│   └── pipeline.py           # Per-country and full-run drivers
├── main.py                   # Replication entry point
├── requirements.txt
└── README.md
```

---

## Data

The three input files share the schema `GEO` / `TIME` / `VALUE` (details in
`data/README.md`). As in the paper:

- **Inflation** is year-on-year headline inflation from the OECD Main Economic
  Indicators database (HICP for Eurozone members; national CPI for the US, UK,
  and Switzerland).
- **Google Trends** is the monthly search index for "inflation" in the local
  language, from January 2014. Belgium sums the Dutch and French series. The
  per-country query strings are listed in the paper's appendix (A.11).
- **GDELT** is the share of news articles mentioning inflation in the local
  language, from January 2017.

The sample runs through December 2025.

---

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the replication

1. Place the three input CSVs in `data/`.
2. Run:

   ```bash
   python main.py
   ```

This regenerates everything under `results/`: per-country JSON and diagnostic
plots, the Google and GDELT summary spreadsheets (Tables A.1 and A.4), the
pre-vs-post threshold scatter (Figure 2), and the Germany / UK figures
(Figures 1, A.1, A.2).

A full run with the paper's settings (`n_bootstrap = 1999`, 24 countries, both
sources) is computationally heavy because of the bootstrap and Monte-Carlo
p-values. For a quick check, lower the bootstrap count when calling the
pipeline directly (see below).

---

## Reproducing a single country

```python
import sys; sys.path.insert(0, "src")

from src.config import COUNTRIES_LANGUAGE
from src.data_loading import load_all
from src.pipeline import run_full_pipeline

data = load_all("data")

out = run_full_pipeline(
    country="Germany",             # paper's dissolution example
    index="GOOGLE",                # or "GDELT"
    countries_language=COUNTRIES_LANGUAGE,
    eurostat_data=data["eurostat"],
    google_data=data["google"],
    gdelt_data=data["gdelt"],
    n_bootstrap=200,               # lower than the paper for a fast run
)
print(out["summary_row"])
```

---

## Methodology references (as cited in the paper)

- Hansen, B. E. (1996). *Inference when a nuisance parameter is not identified
  under the null hypothesis.* Econometrica. (bootstrap inference)
- Hansen, B. E. (1997). *Approximate asymptotic p-values for structural-change
  tests.* JBES. (asymptotic p-values)
- Hansen, B. E. (2000). *Sample splitting and threshold estimation.*
  Econometrica. (threshold regression, Eq. 1)
- Korenok, O., Munro, D., Chen, J. (2023). *Inflation and attention
  thresholds.* REStat.
- Korenok, O., Munro, D. (2024). *The rockets and feathers of inflation
  attention.* GLO Discussion Paper. (post-peak filter)
- Pfäuti, O. (2026). *The inflation attention threshold and inflation surges.*
  AER, forthcoming.

Standard errors are heteroskedasticity-robust (HC1). The grid search uses a
10/90 rank-based trim on the unique inflation values with a minimum of three
observations per regime. The asymptotic Hansen p-value uses 5,000 Monte-Carlo
draws; the bootstrap uses B = 1,999.

---

## Reproducibility notes

- All random procedures (bootstrap, Monte-Carlo, power simulation) are seeded
  (`seed = 42`).
- Sample window, trim fraction, bootstrap count, and significance level are
  centralised in `src/config.py`.

---

## Citation

If you use this code, please cite the paper:

> Kusenda, O. and Marenčák, M. *Inflation attention thresholds before and
> after the inflation peak.* National Bank of Slovakia.