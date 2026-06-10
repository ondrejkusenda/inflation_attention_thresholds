"""
Replication entry point — run EVERYTHING with one command.

    python main.py

This single script reproduces the whole analysis and prints every headline
result to the console (and to ``results/SUMMARY.txt``):

    1. Load the three input datasets from ``data/``.
    2. Threshold pipeline for every country and both attention sources:
       per-country JSON, diagnostic plots, the two summary spreadsheets, and
       the cross-country habituation regression (section 3.1).
    3. Supply/demand decomposition (Appendix A.10): pooled FE regression and
       per-country mediation table with the supply/demand classification.
    4. Peak-date sensitivity (Appendix A.9) for the headline countries.
    5. Cross-country threshold scatter plots and the single-window / time-series
       example figures.
    6. A consolidated summary of all of the above.

All paths are relative to this file, so it can be launched from any directory.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Headless: render figures without a display server.
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.config import COUNTRIES_LANGUAGE, DEFAULT_N_BOOTSTRAP
from src.data_loading import load_all
from src.pipeline import run_all_sources, run_peak_sensitivity
from src.supply_demand_decomposition import run_supply_demand_decomposition
from src.plotting import (
    plot_threshold_scatter,
    plot_single_window,
    plot_timeseries_single,
)

DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
SINGLE_DIR = os.path.join(RESULTS_DIR, "single_window")

SENSITIVITY_COUNTRIES = ("United States", "Germany", "United Kingdom")


class _Tee:
    """Collect summary lines so they are both printed and saved to a file."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


def _fmt(x, spec="{:.2f}", na="NA"):
    return na if x is None else spec.format(x)


def _print_habituation(out, hab):
    out()
    out("=" * 72)
    out("HABITUATION REGRESSION  (section 3.1):  gamma_pre ~ pre-2021 inflation")
    out("=" * 72)
    out(f"{'source':<10}{'beta':>9}{'t':>9}{'R2':>8}{'n':>5}")
    out("-" * 41)
    for source in ("GOOGLE", "GDELT"):
        res = hab.get(source, {})
        if "error" in res:
            out(f"{source:<10}  {res['error']}")
        else:
            out(f"{source:<10}{res['beta']:>+9.2f}{res['t_stat']:>+9.2f}"
                f"{res['r_squared']:>8.2f}{res['n']:>5}")
    out("(paper: Google beta=+2.06 t=+3.58 ; GDELT beta=+2.21 t=+3.14)")


def _print_supply_demand(out, sd):
    pooled = sd["pooled"]
    out()
    out("=" * 72)
    out("SUPPLY / DEMAND DECOMPOSITION  (Appendix A.10)")
    out("=" * 72)
    out(f"Pooled post-peak regression with country FE, HC1 SE  "
        f"(N={pooled['n']}, {pooled['n_countries']} countries)")
    out(f"  inflation (alone)    : {pooled['beta_inf_alone']:>7.2f}  "
        f"(t={pooled['t_inf_alone']:>6.2f})")
    out(f"  inflation (w/ GDELT) : {pooled['beta_inf_with_gdelt']:>7.2f}  "
        f"(t={pooled['t_inf_with_gdelt']:>6.2f})")
    out(f"  GDELT                : {pooled['beta_gdelt']:>7.1f}  "
        f"(t={pooled['t_gdelt']:>6.2f})")
    out(f"  mediation by GDELT   : {pooled['mediation_pct']:.1f}%")
    out("  NOTE: the paper text transposes the inflation-alone and GDELT")
    out("        t-statistics (prints 6.09 / 15.0); the correct HC1 values are")
    out("        15.01 / 6.09. Coefficients and N are correct.")
    out()
    out("Per-country mediation (* = significant at 10%):")
    out(f"{'country':<18}{'b_inf':>8}{'b_inf|G':>9}{'b_gdelt':>10}"
        f"{'%med':>7}{'R2(2)':>7}  label")
    out("-" * 72)
    for r in sorted(sd["by_country"], key=lambda d: d["country"]):
        s1 = "*" if r["p_inf_alone"] < 0.10 else " "
        s2 = "*" if r["p_inf_with_gdelt"] < 0.10 else " "
        s3 = "*" if r["p_gdelt"] < 0.10 else " "
        med = _fmt(r["mediation_pct"], "{:.0f}")
        out(f"{r['country']:<18}"
            f"{r['beta_inf_alone']:>7.2f}{s1}"
            f"{r['beta_inf_with_gdelt']:>8.2f}{s2}"
            f"{r['beta_gdelt']:>9.1f}{s3}"
            f"{med:>7}{r['r2_model2']:>7.2f}  {r['label']}")
    out()
    out(f"supply-driven ({len(sd['supply_driven'])}): {', '.join(sd['supply_driven'])}")
    out(f"demand-driven ({len(sd['demand_driven'])}): {', '.join(sd['demand_driven'])}")


def _print_peak_sensitivity(out, sensitivity):
    out()
    out("=" * 72)
    out("PEAK-DATE SENSITIVITY  (Appendix A.9):  split shifted +-1 / +-2 months")
    out("=" * 72)
    for country, rows in sensitivity.items():
        out(f"\n{country} (Google):")
        out(f"{'offset':>7}{'peak':>10}{'gamma_pre':>11}{'gamma_post':>12}"
            f"{'delta':>9}{'sign':>6}")
        out("-" * 55)
        for r in rows:
            out(f"{r['offset_months']:>+7}{r['peak_date']:>10}"
                f"{_fmt(r['gamma_pre']):>11}{_fmt(r['gamma_post']):>12}"
                f"{_fmt(r['delta'], '{:+.2f}'):>9}{str(r['delta_sign']):>6}")


def _print_threshold_overview(out, results):
    out()
    out("=" * 72)
    out("THRESHOLD OVERVIEW  (Hansen test rejections at 10%)")
    out("=" * 72)
    for source in ("GOOGLE", "GDELT"):
        rows = results[source]
        n = len(rows)
        rej_pre = sum(1 for r in rows if r.get("reject_before"))
        rej_post = sum(1 for r in rows if r.get("reject_after"))
        out(f"{source:<10} countries={n:>3}   "
            f"pre-peak rejections={rej_pre:>3}/{n}   "
            f"post-peak rejections={rej_post:>3}/{n}")
    out("(full per-country detail is in results/summary_*.xlsx)")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the entire inflation-attention replication in one go.",
    )
    p.add_argument(
        "--bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
        help=f"Bootstrap / Monte-Carlo replications for the Hansen p-values "
             f"(default {DEFAULT_N_BOOTSTRAP}).",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Fast sanity run: sets --bootstrap to 200. Use the default for "
             "the final, paper-matching numbers.",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip all figure generation (faster; tables still produced).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    """Run the complete replication, write all artefacts, and print a summary."""
    args = _parse_args(argv)
    n_bootstrap = 200 if args.quick else args.bootstrap
    make_plots = not args.no_plots

    os.makedirs(SINGLE_DIR, exist_ok=True)
    out = _Tee()
    out(f"Inflation-attention replication  "
        f"(bootstrap={n_bootstrap}, plots={'on' if make_plots else 'off'})")

    # 1. Load data.
    data = load_all(DATA_DIR)
    eurostat_data = data["eurostat"]
    google_data = data["google"]
    gdelt_data = data["gdelt"]

    # 2. Full threshold pipeline (all countries, both sources) + habituation.
    results = run_all_sources(
        countries_language=COUNTRIES_LANGUAGE,
        eurostat_data=eurostat_data,
        google_data=google_data,
        gdelt_data=gdelt_data,
        output_dir=RESULTS_DIR,
        n_bootstrap=n_bootstrap,
        make_plots=make_plots,
    )
    results_all_google = results["GOOGLE"]
    results_all_gdelt = results["GDELT"]
    habituation = results.get("_habituation", {})

    # 2b. Supply/demand decomposition (Appendix A.10), reproduced from the
    #     shipped CSVs (post-peak window, country FE, HC1 SE).
    sd = run_supply_demand_decomposition(
        countries_language=COUNTRIES_LANGUAGE,
        eurostat_data=eurostat_data,
        google_data=google_data,
        gdelt_data=gdelt_data,
        output_dir=RESULTS_DIR,
    )

    # 2c. Peak-date sensitivity (Appendix A.9): the US sign flip lives here.
    sensitivity = {}
    for country in SENSITIVITY_COUNTRIES:
        sensitivity[country] = run_peak_sensitivity(
            country=country, index="GOOGLE",
            countries_language=COUNTRIES_LANGUAGE,
            eurostat_data=eurostat_data,
            google_data=google_data, gdelt_data=gdelt_data,
            n_bootstrap=n_bootstrap,
            output_dir=RESULTS_DIR,
        )

    # 3. Cross-country threshold scatter plots.
    if make_plots:
        iso_labels = {
            row["country"]: COUNTRIES_LANGUAGE[row["country"]]["id"]
            for row in results_all_google + results_all_gdelt
            if row["country"] in COUNTRIES_LANGUAGE
        }

        fig1, ax1 = plt.subplots(figsize=(7, 7))
        plot_threshold_scatter(results_all_google, "Google Trends", ax1, iso_labels)
        fig1.tight_layout()
        fig1.savefig(os.path.join(RESULTS_DIR, "threshold_scatter_google.pdf"),
                     dpi=300, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(7, 7))
        plot_threshold_scatter(results_all_gdelt, "GDELT", ax2, iso_labels)
        fig2.tight_layout()
        fig2.savefig(os.path.join(RESULTS_DIR, "threshold_scatter_gdelt.pdf"),
                     dpi=300, bbox_inches="tight")
        plt.close(fig2)

        # 4. Single-window and time-series example figures (UK, Germany, US).
        for country, code in [("United Kingdom", "UK"), ("Germany", "DE")]:
            for window in ("Before", "After"):
                plot_single_window(
                    country=country, index="GOOGLE", window=window,
                    countries_language=COUNTRIES_LANGUAGE,
                    eurostat_data=eurostat_data, google_data=google_data,
                    save_path=os.path.join(SINGLE_DIR, f"{code}_{window.lower()}.pdf"),
                )
                plt.close("all")

            plot_timeseries_single(
                country=country, index="GOOGLE",
                countries_language=COUNTRIES_LANGUAGE,
                eurostat_data=eurostat_data, google_data=google_data,
                save_path=os.path.join(SINGLE_DIR, f"{code}_timeseries.pdf"),
            )
            plt.close("all")

        plot_timeseries_single(
            country="United States", index="GOOGLE",
            countries_language=COUNTRIES_LANGUAGE,
            eurostat_data=eurostat_data, google_data=google_data,
            save_path=os.path.join(SINGLE_DIR, "US_timeseries.pdf"),
        )
        plt.close("all")

    # 5. Consolidated summary of every headline result.
    out()
    out("#" * 72)
    out("#  CONSOLIDATED RESULTS")
    out("#" * 72)
    _print_threshold_overview(out, results)
    _print_habituation(out, habituation)
    _print_supply_demand(out, sd)
    _print_peak_sensitivity(out, sensitivity)

    out()
    out("Replication complete. Figures, tables and SUMMARY.txt are in results/.")
    out.save(os.path.join(RESULTS_DIR, "SUMMARY.txt"))


if __name__ == "__main__":
    main()
