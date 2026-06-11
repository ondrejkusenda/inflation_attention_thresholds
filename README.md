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

This package reproduces all figures, tables, and statistics in the paper, and
runs end-to-end with a single command (`python main.py`).

---

## Headline findings reproduced

1. **Pre-peak threshold is universal.** The Hansen test rejects linearity in
   the pre-peak window for all 24 countries. The cross-country **habituation
   regression** (§3.1) confirms higher pre-2021 average inflation → higher
   threshold (Google: β = +2.06, t = +3.58).
2. **Post-peak threshold dissolves for ~half the sample.** For roughly half
   the countries no post-peak threshold is identified, and attention decays
   linearly during disinflation rather than switching off at a discrete point.
3. **Dissolution is genuine, not low power.** A residual-bootstrap power
   analysis confirms the Hansen test would detect the pre-peak threshold if it
   were still active. Each simulated series is judged by a proper 10 % Hansen
   test (the per-window **simulated** sup-F critical value), not a fixed
   cut-off — so the power figures share the level of the main test.
4. **Symmetry is rejected everywhere.** A stacked Chow-type structural-break
   test (classic homoskedastic F) rejects pre-/post-peak symmetry for every
   country.
5. **US is a fragile exception.** Under Google Trends the US shows a *higher*
   post-peak threshold (Δ = +2.95 pp), but the result rests on 4
   above-threshold observations and does not replicate under GDELT. The
   **peak-date sensitivity** check (§A.9) shows shifting the split +2 months
   flips the sign of the US Δ.
6. **Supply vs demand (A.10).** Pooled post-peak regressions of attention on
   inflation and GDELT news coverage decompose the decline: where news
   coverage absorbs the inflation effect the decline is supply-driven
   (e.g. FI/ES/IT/LT), where inflation stays significant it is demand-driven
   (e.g. US/DE/FR/NL).

---

## How the code maps to the paper

| Paper element | Produced by |
|---------------|-------------|
| Eq. (1) single-threshold regression (Hansen 2000) | `threshold_model.threshold_regression_full` |
| Asymptotic Hansen p-value (Monte Carlo, Gaussian null) | `threshold_model.asymptotic_pvalue` |
| Bootstrap Hansen p-value (residual bootstrap) | `threshold_model.bootstrap_pvalue` |
| Korenok–Munro filter (test post-peak only if inflation fell below γ_pre) | `threshold_model.run_threshold_analysis` |
| Threshold gap Δ = γ_post − γ_pre | `threshold_model.run_threshold_analysis` |
| Stacked Chow test (App. A.5) | `threshold_model.chow_test_stacked` |
| Power analysis, simulated 10 % critical value (App. A.6) | `threshold_model.power_analysis_post_peak`, `threshold_model.simulated_critical_value` |
| Peak-date sensitivity (App. A.9) | `pipeline.run_peak_sensitivity` (via `preprocessing.prepare_data(..., peak_offset=k)`) |
| Supply/demand decomposition (App. A.10) | `supply_demand_decomposition.run_supply_demand_decomposition` |
| Habituation regression (§3.1) | `threshold_model.habituation_regression` (driven across countries in `pipeline.run_all_sources`) |
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
│   ├── single_window/        # Figure 1 / A.1 / A.2 style figures
│   ├── sensitivity/          # Peak-date sensitivity JSON, per country (A.9)
│   ├── habituation.json       # Cross-country habituation regression (§3.1)
│   ├── supply_demand_decomposition.json        # A.10 pooled + classification
│   ├── supply_demand_country_regressions.csv   # A.10 per-country table
│   ├── threshold_scatter_*.pdf                  # Figure 2
│   └── SUMMARY.txt            # Consolidated console summary of every result
├── src/
│   ├── config.py             # Palette, plot style, country metadata, constants
│   ├── data_loading.py       # CSV readers with schema validation
│   ├── preprocessing.py      # prepare_data: merge + split at the inflation peak
│   ├── threshold_model.py    # Hansen model, Chow test, power, habituation
│   ├── supply_demand_decomposition.py  # Appendix A.10
│   ├── plotting.py           # All publication figures
│   └── pipeline.py           # Per-country, full-run, and sensitivity drivers
├── main.py                   # Single-command replication entry point
├── requirements.txt
└── README.md
```

---

## Data

The three input files share the schema `GEO` / `TIME` / `VALUE`. As in the paper:

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

Place the three input CSVs in `data/` and run the single entry point:

```bash
python main.py                 # full run: all countries, both sources, all figures
```

This regenerates everything under `results/`: per-country JSON and diagnostic
plots, the Google and GDELT summary spreadsheets (Tables A.1 and A.4), the
pre-vs-post threshold scatter (Figure 2), the Germany / UK / US figures
(Figures 1, A.1, A.2), the habituation regression (§3.1), the supply/demand
decomposition (A.10), and the peak-date sensitivity tables (A.9). Every
headline number is also printed to the console and written to
`results/SUMMARY.txt`.

### Command-line options

| Flag | Effect |
|------|--------|
| *(none)* | Full run: `--bootstrap 5000`, all figures. Use this for the paper's numbers. |
| `--quick` | Fast sanity run (`--bootstrap 200`). Headline coefficients are stable; borderline p-values are noisier. |
| `--bootstrap N` | Set the bootstrap / Monte-Carlo replication count explicitly. |
| `--no-plots` | Skip all figures (tables and JSON still produced). Fastest. |

The full run is computationally heavy because of the bootstrap and
Monte-Carlo p-values; `--quick` is recommended for a first pass.

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

Standard errors are heteroskedasticity-robust (HC1) throughout, including the
supply/demand regressions. The grid search uses a 10/90 rank-based trim on the
unique inflation values with a minimum of three observations per regime.

**P-values.** The Hansen test reports two p-values per window: a residual
bootstrap and an asymptotic (Gaussian-null Monte-Carlo) p-value. At the paper
scale these use **B = 1,999** bootstrap draws and **5,000** asymptotic draws
respectively. The `n_bootstrap` setting (`config.py`; `--quick` lowers it to
200) acts as a budget: a full run caps the bootstrap at 1,999 and runs the
asymptotic at 5,000, while a smaller budget scales both down together so quick
runs stay fast.

**Power analysis (A.6).** The post-peak power simulation rejects each simulated
series when its sup-F exceeds the **simulated 10 % critical value** of the
sup-F null on that window's design (`simulated_critical_value`), i.e. a proper
Hansen test at the same 10 % level as the main analysis — not a fixed numeric
cut-off.

**Supply/demand (A.10).** Pooled post-peak regression of Google attention on
inflation and GDELT with country fixed effects and HC1 SEs reproduces the
headline numbers from the shipped CSVs: N = 931, inflation 1.97 → 1.32, GDELT
403.7. Note: under HC1 the correct t-statistics are inflation-alone 15.01,
inflation|GDELT 7.40, GDELT 6.09; the published text transposes the
inflation-alone and GDELT t-values (6.09 / 15.0). Coefficients and N are
correct.

---

## Reproducibility notes

- All random procedures (bootstrap, Monte-Carlo, power simulation, simulated
  critical values) are seeded (`seed = 42`).
- Sample window, the pre-2021 cutoff for habituation, trim fraction, bootstrap
  count, and significance level are centralised in `src/config.py`.
- Figures render through a headless Matplotlib backend, so `python main.py`
  works on machines without a display server.

---

## Citation

If you use this code, please cite the paper:

> Kusenda, O. and Marenčák, M. *Inflation attention thresholds before and
> after the inflation peak.* National Bank of Slovakia.