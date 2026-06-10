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
from threshold_model import run_threshold_analysis, habituation_regression
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
        "inf_mean_pre_2021": prep["out_data"].get("inf_mean_pre_2021"),

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

    # Cross-country habituation regression (section 3.1): regress the
    # identified pre-peak threshold on each country's pre-2021 mean inflation.
    habituation = {}
    for source_name in ("GOOGLE", "GDELT"):
        rows = results_by_source[source_name]
        gammas, means, labels = [], [], []
        for row in rows:
            g = row.get("threshold_before")
            m = row.get("inf_mean_pre_2021")
            if g is None or m is None:
                continue
            gammas.append(g)
            means.append(m)
            labels.append(row.get("country"))
        habituation[source_name] = habituation_regression(gammas, means, labels)

    hab_path = os.path.join(output_dir, "habituation.json")
    with open(hab_path, "w") as f:
        json.dump(habituation, f, default=str, indent=4)
    print(f"Saved habituation regression -> {hab_path}")
    for source_name, res in habituation.items():
        if "error" in res:
            print(f"  {source_name}: {res['error']}")
        else:
            print(f"  {source_name}: beta={res['beta']:+.2f} "
                  f"(t={res['t_stat']:+.2f}, n={res['n']}, R2={res['r_squared']:.2f})")

    results_by_source["_habituation"] = habituation
    return results_by_source


def run_peak_sensitivity(
    country,
    index,
    countries_language,
    eurostat_data,
    google_data=None,
    gdelt_data=None,
    offsets=(-2, -1, 0, 1, 2),
    trim=DEFAULT_TRIM,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    output_dir=None,
):
    """Peak-date sensitivity analysis (Appendix A.9).

    Re-runs the full threshold analysis with the pre/post split shifted by
    each value in ``offsets`` (in observed months) relative to the inflation
    peak, and reports how the estimated thresholds and their gap move. This
    reproduces results such as "+2 months flips the US sign", where the sign
    of Δ = γ_post − γ_pre reverses under a shifted split.

    Parameters
    ----------
    country : str
        Country name.
    index : {"GOOGLE", "GDELT"}
        Attention source.
    countries_language : dict
        Country metadata.
    eurostat_data : pandas.DataFrame
        Inflation data.
    google_data, gdelt_data : pandas.DataFrame, optional
        Attention-index data.
    offsets : iterable of int, optional
        Month shifts applied to the split point. Defaults to ``(-2, -1, 0, 1, 2)``.
    trim : float, optional
        Trimming fraction.
    n_bootstrap : int, optional
        Replications for the Hansen p-values.
    output_dir : str, optional
        If given, the per-offset rows are written to
        ``<output_dir>/sensitivity/<source>_<country>.json``.

    Returns
    -------
    list of dict
        One row per offset with peak date, both thresholds, their rejection
        flags, Δ, and the sign of Δ.
    """
    source = index.lower()
    rows = []

    for off in offsets:
        prep = prepare_data(
            country=country, index=index,
            countries_language=countries_language,
            eurostat_data=eurostat_data,
            google_data=google_data, gdelt_data=gdelt_data,
            peak_offset=off,
        )
        res = run_threshold_analysis(prep, trim=trim, n_bootstrap=n_bootstrap)

        bp = res.get("before_peak", {}) or {}
        ap = res.get("after_peak", {}) or {}
        dl = res.get("delta", {}) or {}

        g_pre = bp.get("threshold")
        g_post = ap.get("threshold")
        reject_pre = bp.get("p_value_asym", 1.0) <= SIGNIFICANCE_LEVEL
        reject_post = ap.get("p_value_asym", 1.0) <= SIGNIFICANCE_LEVEL

        delta = None
        if g_pre is not None and g_post is not None:
            delta = float(g_post - g_pre)
        delta_sign = None if delta is None else ("+" if delta > 0 else
                                                 ("-" if delta < 0 else "0"))

        rows.append({
            "country": country,
            "source": source,
            "offset_months": int(off),
            "peak_date": str(prep["out_data"].get("peak_date"))[0:7],
            "n_before": prep["out_data"].get("n_before_peak"),
            "n_after": prep["out_data"].get("n_after_peak"),
            "gamma_pre": g_pre,
            "reject_pre": bool(reject_pre),
            "gamma_post": g_post,
            "reject_post": bool(reject_post),
            "fell_below": ap.get("fell_below", False),
            "delta": delta,
            "delta_sign": delta_sign,
            "delta_reported": dl.get("reported", False),
        })

    if output_dir is not None:
        sens_dir = os.path.join(output_dir, "sensitivity")
        os.makedirs(sens_dir, exist_ok=True)
        safe_country = country.replace(" ", "_")
        path = os.path.join(sens_dir, f"{source}_{safe_country}.json")
        with open(path, "w") as f:
            json.dump(rows, f, default=str, indent=4)
        print(f"Saved peak sensitivity ({country}, {source}) -> {path}")

    return rows
