# Inflation, Attention, and Thresholds

A replication package for estimating threshold (regime-switching)
relationships between **inflation** and **public attention to inflation**,
measured by Google Trends and GDELT news indices, across 24 advanced
economies.

The analysis asks whether attention responds to inflation only once inflation
crosses a country-specific threshold, and whether that threshold shifts
between the run-up to the 2021–2023 inflation peak and the disinflation that
followed.

---

## What the analysis does

For each country and each attention source, the pipeline:

1. **Stacked Chow test** — a gateway structural-break test of whether the
   inflation–attention relationship differs before vs. after the inflation
   peak.
2. **Pre-peak threshold model** — Hansen sample-splitting threshold
   regression with a sup-F test (bootstrap and asymptotic p-values) and a
   Hansen (2000) likelihood-ratio confidence interval for the threshold.
3. **Korenok–Munro filter** — the post-peak window is analysed only if
   inflation actually fell back below the pre-peak threshold.
4. **Post-peak threshold model** — the same Hansen procedure on the
   post-peak window.
5. **Threshold shift Δ** — reported as `γ_post − γ_pre` when both windows
   reject linearity.
6. **Power analysis** — when the post-peak window fails to reject linearity,
   a residual-bootstrap power study under the pre-peak data-generating
   process distinguishes genuine linearity from low power.

---

## Repository layout

```
inflation_attention_thresholds/
├── data/                     # Input CSVs (see data/README.md for schema)
│   ├── INFLATION_DATA.csv
│   ├── GOOGLE_DATA.csv
│   └── GDELT_DATA.csv
├── results/                  # Generated outputs (created on run)
│   ├── google/
│   │   ├── raw_results/      # Per-country JSON
│   │   ├── plots/            # Per-country diagnostic PDFs
│   │   └── summary_GOOGLE.xlsx
│   ├── gdelt/                # Same structure as google/
│   └── single_window/        # Example standalone figures
├── src/
│   ├── config.py             # Palette, plot style, country metadata, constants
│   ├── data_loading.py       # CSV readers with schema validation
│   ├── preprocessing.py      # prepare_data: merge + split at the peak
│   ├── threshold_model.py    # Hansen model, tests, Chow, power, orchestration
│   ├── plotting.py           # All publication figures
│   └── pipeline.py           # Per-country and full-run drivers
├── main.py                   # Replication entry point
├── requirements.txt
└── README.md
```

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

1. Put the three input CSVs in `data/` (schema in `data/README.md`).
2. Run:

   ```bash
   python main.py
   ```

This regenerates everything under `results/`: per-country JSON and diagnostic
plots, the two summary spreadsheets, the cross-country pre-vs-post threshold
scatter plots, and the example single-window / time-series figures for the
United Kingdom and Germany.

A full run with the default `n_bootstrap = 1999` over all 24 countries and
both sources is computationally heavy. For a quick check, lower the bootstrap
count when calling the pipeline directly (see below).

---

## Using the package programmatically

```python
import sys; sys.path.insert(0, "src")

from src.config import COUNTRIES_LANGUAGE
from src.data_loading import load_all
from src.pipeline import run_full_pipeline

data = load_all("data")

out = run_full_pipeline(
    country="Slovakia",
    index="GOOGLE",                 # or "GDELT"
    countries_language=COUNTRIES_LANGUAGE,
    eurostat_data=data["eurostat"],
    google_data=data["google"],
    gdelt_data=data["gdelt"],
    n_bootstrap=200,                # lower for a fast run
    make_plots=True,
)

print(out["summary_row"])
```



## Outputs

| Output                                      | Description                                            |
|---------------------------------------------|--------------------------------------------------------|
| `results/<source>/raw_results/<C>.json`     | All test statistics and estimates for country `<C>`.   |
| `results/<source>/plots/<C>.pdf`            | Three-panel diagnostic (time series + two scatters).   |
| `results/<source>/summary_<SOURCE>.xlsx`    | One flat row per country with the headline results.    |
| `results/threshold_scatter_<source>.pdf`    | Cross-country γ_pre vs γ_post scatter.                  |
| `results/single_window/*.pdf`               | Standalone example figures (UK, Germany).              |

Each summary row reports the Chow test, the pre- and post-peak thresholds
with p-values and confidence intervals, the threshold shift Δ (when
reportable), and the power-analysis verdict.

---

## Methodology references

- Hansen, B. E. (1996). *Inference when a nuisance parameter is not
  identified under the null hypothesis.* Econometrica.
- Hansen, B. E. (1997). *Approximate asymptotic p-values for
  structural-change tests.* Journal of Business & Economic Statistics.
- Hansen, B. E. (2000). *Sample splitting and threshold estimation.*
  Econometrica. (Confidence-interval critical values: Table 1.)
- Andrews, D. W. K. (1993). *Tests for parameter instability and structural
  change with unknown change point.* Econometrica.

Standard errors are heteroskedasticity-robust (HC1). Threshold p-values use
both a fixed-regressor bootstrap and an asymptotic Monte-Carlo approximation;
rejections are evaluated at the 10% level.

---

## Reproducibility notes

- All random procedures (bootstrap, Monte-Carlo, power simulation) are seeded
  (`seed = 42`) so runs are reproducible.
- Sample window, trimming fraction, bootstrap count, and significance level
  are centralised in `src/config.py`.
