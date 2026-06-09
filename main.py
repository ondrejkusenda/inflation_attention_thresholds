"""
Replication entry point.

Running ``python main.py`` reproduces the full analysis:

    1. Load the three input datasets from ``data/``.
    2. Run the threshold pipeline for every country and both attention
       sources, writing per-country JSON, diagnostic plots, and the two
       summary spreadsheets under ``results/``.
    3. Produce the cross-country pre-vs-post threshold scatter plots.
    4. Produce the single-window and single time-series example figures
       (United Kingdom and Germany) under ``results/single_window/``.

All paths are relative to this file, so the script can be launched from any
working directory.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.config import COUNTRIES_LANGUAGE
from src.data_loading import load_all
from src.pipeline import run_all_sources
from src.plotting import (
    plot_threshold_scatter,
    plot_single_window,
    plot_timeseries_single,
)

DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
SINGLE_DIR = os.path.join(RESULTS_DIR, "single_window")


def main() -> None:
    """Run the complete replication and write all figures and tables."""
    os.makedirs(SINGLE_DIR, exist_ok=True)

    # 1. Load data.
    data = load_all(DATA_DIR)
    eurostat_data = data["eurostat"]
    google_data = data["google"]
    gdelt_data = data["gdelt"]

    # 2. Full pipeline over all countries and both sources.
    results = run_all_sources(
        countries_language=COUNTRIES_LANGUAGE,
        eurostat_data=eurostat_data,
        google_data=google_data,
        gdelt_data=gdelt_data,
        output_dir=RESULTS_DIR,
        make_plots=True,
    )
    results_all_google = results["GOOGLE"]
    results_all_gdelt = results["GDELT"]

    # 3. Cross-country threshold scatter plots.
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

    # 4. Single-window and time-series figures (UK and Germany).
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
        save_path=os.path.join(SINGLE_DIR, f"US_timeseries.pdf"),
    )
    plt.close("all")

    print("\nReplication complete. See the results/ directory.")


if __name__ == "__main__":
    main()
