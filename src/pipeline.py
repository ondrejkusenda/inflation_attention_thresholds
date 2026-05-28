"""
Pipeline.

End-to-end orchestration: for a country and attention source, prepare the
data, run the threshold analysis, persist raw JSON and a diagnostic plot, and
return a flat summary row. :func:`run_all_sources` loops this over every
country and both sources and writes the summary spreadsheets.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    COUNTRIES_LIST, SIGNIFICANCE_LEVEL,
    DEFAULT_TRIM, DEFAULT_N_BOOTSTRAP,
)
from preprocessing import prepare_data
from threshold_model import run_threshold_analysis
from plotting import plot_threshold_results


def run_full_pipeline(
    country,
    index,
    countries_language,
    eurostat_data,
    google_data=None,
    gdelt_data=None,
    output_dir="results",
    trim=DEFAULT_TRIM,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    make_plots=True,
):
    """Run the analysis for one country / source and persist the outputs.

    Writes ``results/<source>/raw_results/<country>.json`` and, optionally,
    ``results/<source>/plots/<country>.pdf``.

    Parameters
    ----------
    country : str
        Country name.
    index : {"GOOGLE", "GDELT"}
        Attention source; also selects the output sub-directory.
    countries_language : dict
        Country metadata.
    eurostat_data : pandas.DataFrame
        Inflation data.
    google_data, gdelt_data : pandas.DataFrame, optional
        Attention-index data.
    output_dir : str, optional
        Root output directory. Defaults to ``"results"``.
    trim : float, optional
        Trimming fraction for the threshold estimation.
    n_bootstrap : int, optional
        Replications for the Hansen p-values.
    make_plots : bool, optional
        Whether to render and save the diagnostic plot. Defaults to ``True``.

    Returns
    -------
    dict
        Keys ``country``, ``source``, ``meta``, ``summary_row``, and
        ``files`` (paths to the saved JSON and plot).
    """
    source = index.lower()  # "google" or "gdelt"

    raw_dir = os.path.join(output_dir, source, "raw_results")
    plot_dir = os.path.join(output_dir, source, "plots")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    safe_country = country.replace(" ", "_")
    json_path = os.path.join(raw_dir, f"{safe_country}.json")
    plot_path = os.path.join(plot_dir, f"{safe_country}.pdf")

    # 1. Data preparation.
    prep = prepare_data(
        country=country, index=index,
        countries_language=countries_language,
        eurostat_data=eurostat_data,
        google_data=google_data, gdelt_data=gdelt_data,
    )

    # 2. Threshold analysis.
    results = run_threshold_analysis(prep, trim=trim, n_bootstrap=n_bootstrap)

    # 3. Persist raw JSON.
    with open(json_path, "w") as f:
        json.dump(
            {
                "country": country,
                "source": source,
                "index": index,
                "meta": prep["out_data"],
                "results": results,
            },
            f, default=str, indent=4,
        )

    # 4. Optional diagnostic plot.
    if make_plots:
        plot_threshold_results(country=country, index=index,
                               prep=prep, results=results)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plot_path = None

    # 5. Flat summary row for the cross-country spreadsheet.
    bp = results.get("before_peak", {}) or {}
    ap = results.get("after_peak", {}) or {}
    ch = results.get("chow_test", {}) or {}
    pw = results.get("power", {}) or {}
    dl = results.get("delta", {}) or {}

    summary_row = {
        "country": country,
        "source": source,

        "peak_date": str(prep["out_data"].get("peak_date"))[0:7],
        "inf_peak": prep["out_data"].get("inf_peak"),
        "inf_mean": prep["out_data"].get("inf_mean"),
        "inf_mean_pre-peak": prep["out_data"].get("inf_mean_pre-peak"),
        "inf_mean_post-peak": prep["out_data"].get("inf_mean_post-peak"),

        # Stacked Chow test (gateway).
        "chow_f": ch.get("f_chow"),
        "chow_p": ch.get("p_chow"),
        "chow_reject_10": ch.get("reject_chow_10"),
        "chow_gamma_null": ch.get("gamma_null"),

        # Pre-peak.
        "n_before": prep["out_data"].get("n_before_peak"),
        "threshold_before": bp.get("threshold"),
        "p_boot_before": bp.get("p_value_boot"),
        "p_asym_before": bp.get("p_value_asym"),
        "reject_before": bp.get("p_value_asym", 1) <= SIGNIFICANCE_LEVEL,
        "ci_low_before": bp.get("ci_low"),
        "ci_high_before": bp.get("ci_high"),

        # Post-peak.
        "n_after": prep["out_data"].get("n_after_peak"),
        "fell_below": ap.get("fell_below", False),
        "threshold_after": ap.get("threshold"),
        "p_boot_after": ap.get("p_value_boot"),
        "p_asym_after": ap.get("p_value_asym"),
        "reject_after": ap.get("p_value_asym", 1) <= SIGNIFICANCE_LEVEL,
        "ci_low_after": ap.get("ci_low"),
        "ci_high_after": ap.get("ci_high"),

        # Δ (reported only if both windows reject linearity).
        "delta": dl.get("value"),
        "delta_reported": dl.get("reported", False),

        # Power analysis (only when post-peak fails to reject).
        "power_10": pw.get("power_10"),
        "power_n_below": pw.get("n_below_post"),
        "power_n_above": pw.get("n_above_post"),
        "power_interp": pw.get("interpretation"),
    }

    return {
        "country": country,
        "source": source,
        "meta": prep["out_data"],
        "summary_row": summary_row,
        "files": {"raw": json_path, "plot": plot_path},
    }


def run_all_sources(
    countries_language,
    eurostat_data,
    google_data,
    gdelt_data,
    countries_list=None,
    output_dir="results",
    trim=DEFAULT_TRIM,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    make_plots=True,
):
    """Run the pipeline for every country and both attention sources.

    Saves per-source summary spreadsheets to
    ``results/google/summary_GOOGLE.xlsx`` and
    ``results/gdelt/summary_GDELT.xlsx``.

    Parameters
    ----------
    countries_language : dict
        Country metadata.
    eurostat_data, google_data, gdelt_data : pandas.DataFrame
        Input datasets.
    countries_list : list of str, optional
        Countries to process. Defaults to :data:`config.COUNTRIES_LIST`.
    output_dir : str, optional
        Root output directory. Defaults to ``"results"``.
    trim : float, optional
        Trimming fraction.
    n_bootstrap : int, optional
        Replications for the Hansen p-values.
    make_plots : bool, optional
        Whether to save diagnostic plots. Defaults to ``True``.

    Returns
    -------
    dict
        ``{"GOOGLE": [...rows...], "GDELT": [...rows...]}``.
    """
    if countries_list is None:
        countries_list = COUNTRIES_LIST

    sources = {"GOOGLE": google_data, "GDELT": gdelt_data}
    results_by_source = {"GOOGLE": [], "GDELT": []}

    for source_name in sources:
        print(f"\n{'=' * 30}\nRUNNING SOURCE: {source_name}\n{'=' * 30}")

        for i, country in enumerate(countries_list):
            print(f"\n[{i + 1}/{len(countries_list)}] {country}")
            try:
                out = run_full_pipeline(
                    country=country,
                    index=source_name,
                    countries_language=countries_language,
                    eurostat_data=eurostat_data,
                    google_data=google_data,
                    gdelt_data=gdelt_data,
                    output_dir=output_dir,
                    trim=trim,
                    n_bootstrap=n_bootstrap,
                    make_plots=make_plots,
                )
                row = out["summary_row"]
                row["source"] = source_name
                results_by_source[source_name].append(row)
            except Exception as exc:  # noqa: BLE001 - log and continue the batch.
                print(f"Error in {country}: {exc}")

    # Save the summary spreadsheets.
    google_dir = os.path.join(output_dir, "google")
    gdelt_dir = os.path.join(output_dir, "gdelt")
    os.makedirs(google_dir, exist_ok=True)
    os.makedirs(gdelt_dir, exist_ok=True)

    google_path = os.path.join(google_dir, "summary_GOOGLE.xlsx")
    gdelt_path = os.path.join(gdelt_dir, "summary_GDELT.xlsx")
    pd.DataFrame(results_by_source["GOOGLE"]).to_excel(google_path, index=False)
    pd.DataFrame(results_by_source["GDELT"]).to_excel(gdelt_path, index=False)

    print(f"\nSaved GOOGLE summary -> {google_path}")
    print(f"Saved GDELT summary -> {gdelt_path}")

    return results_by_source
