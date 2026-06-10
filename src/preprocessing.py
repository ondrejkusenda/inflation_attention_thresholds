"""
Pre-processing.

Turns the raw datasets into the per-country analysis frame: a single table
of (inflation, attention-index) pairs, split into a pre-peak and a post-peak
window around the country's inflation peak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import START_DATE, MAX_DATE, PRE_2021_END


def prepare_data(
    country: str,
    index: str,
    countries_language: dict,
    eurostat_data: pd.DataFrame,
    google_data: pd.DataFrame | None = None,
    gdelt_data: pd.DataFrame | None = None,
    start_date: str = START_DATE,
    max_date: str = MAX_DATE,
    clean_outliers: bool = False,
    peak_offset: int = 0,
) -> dict:
    """Build the merged analysis dataset for one country and attention source.

    The attention index (Google Trends or GDELT) is merged with the Eurostat
    inflation series on the monthly date, the sample is restricted to
    ``[start_date, max_date]``, and the result is split at the inflation peak.

    Parameters
    ----------
    country : str
        Country name, must be a key of ``countries_language``.
    index : {"GOOGLE", "GDELT"}
        Which attention source to use for the ``INDEX`` column.
    countries_language : dict
        Country metadata (see :data:`config.COUNTRIES_LANGUAGE`).
    eurostat_data : pandas.DataFrame
        Inflation data with ``GEO`` / ``TIME`` / ``VALUE`` columns.
    google_data, gdelt_data : pandas.DataFrame, optional
        Attention-index data; the one matching ``index`` is required.
    start_date, max_date : str, optional
        Inclusive sample bounds (ISO date strings).
    clean_outliers : bool, optional
        If ``True``, drop inflation observations outside the 5th–95th
        percentile range. Defaults to ``False``.
    peak_offset : int, optional
        Shift the pre/post split point by this many observed months relative
        to the inflation peak (Appendix A.9 peak-date sensitivity). ``0`` is
        the baseline; ``+1`` / ``+2`` move the split later, ``-1`` / ``-2``
        earlier. The shift is applied over observed months and clipped to the
        sample so neither window becomes empty. Defaults to ``0``.

    Returns
    -------
    dict
        Keys:
            ``dataset_clean``    – full merged frame (TIME, INDEX, INFLATION).
            ``data_before_peak`` – observations up to and including the peak.
            ``data_after_peak``  – observations strictly after the peak.
            ``out_data``         – summary statistics and the peak date.
    """
    country_id = countries_language[country]["id"]
    start = pd.Timestamp(start_date)

    # Locate the inflation peak within the sample window.
    df_inf_window = eurostat_data[
        (eurostat_data["GEO"] == country_id) & (eurostat_data["TIME"] >= start)
    ].copy()
    peak = df_inf_window.loc[df_inf_window["VALUE"].idxmax(), "TIME"]

    # Select the attention-index series.
    if index == "GOOGLE":
        google_window = google_data[
            (google_data["GEO"] == country_id) & (google_data["TIME"] >= start)
        ]
        if country_id == "BE":
            # Belgium has two language queries; sum them per month.
            data_index = (
                google_window.groupby("TIME")["VALUE"].sum().reset_index()
            )
        else:
            data_index = google_window.reset_index()
    else:
        data_index = gdelt_data[
            (gdelt_data["GEO"] == country_id) & (gdelt_data["TIME"] >= start)
        ].copy()

    data_inf = eurostat_data[
        (eurostat_data["GEO"] == country_id) & (eurostat_data["TIME"] >= start)
    ].copy()

    # Merge index and inflation on the monthly date.
    dataset = (
        data_index[["TIME", "VALUE"]]
        .rename(columns={"VALUE": "INDEX"})
        .merge(
            data_inf[["TIME", "VALUE"]].rename(columns={"VALUE": "INFLATION"}),
            on="TIME",
            how="left",
        )
    )

    dataset = dataset[dataset["TIME"] <= pd.Timestamp(max_date)]
    dataset.sort_values("TIME", inplace=True)
    dataset.dropna(inplace=True)

    # Optional symmetric outlier trimming on inflation.
    if clean_outliers:
        dataset = dataset[
            dataset["INFLATION"].between(
                dataset["INFLATION"].quantile(0.05),
                dataset["INFLATION"].quantile(0.95),
            )
        ]

    # Optional peak-date shift (Appendix A.9). Move the split point by
    # ``peak_offset`` observed months, clipped so both windows stay non-empty.
    if peak_offset != 0:
        observed = dataset["TIME"].sort_values().unique()
        pos = int(np.searchsorted(observed, np.datetime64(pd.Timestamp(peak))))
        new_pos = int(np.clip(pos + peak_offset, 0, len(observed) - 1))
        peak = pd.Timestamp(observed[new_pos])

    # Split into pre-peak (inclusive) and post-peak windows.
    before = dataset[dataset["TIME"] <= peak].reset_index(drop=True)
    after = dataset[dataset["TIME"] > peak].reset_index(drop=True)

    # "Normal-times" average inflation over [start, PRE_2021_END), i.e. the
    # pre-2021 mean used as the regressor in the habituation regression
    # (section 3.1). This is distinct from inf_mean_pre-peak, which averages
    # over the whole pre-peak window up to the ~2022 inflation peak.
    pre_2021_inf = data_inf[data_inf["TIME"] < pd.Timestamp(PRE_2021_END)]["VALUE"]
    inf_mean_pre_2021 = float(pre_2021_inf.mean()) if len(pre_2021_inf) else float("nan")

    return {
        "dataset_clean": dataset,
        "data_before_peak": before,
        "data_after_peak": after,
        "out_data": {
            "inf_peak": dataset["INFLATION"].max(),
            "inf_mean": dataset["INFLATION"].mean(),
            "inf_mean_pre-peak": before["INFLATION"].mean(),
            "inf_mean_post-peak": after["INFLATION"].mean(),
            "inf_mean_pre_2021": inf_mean_pre_2021,
            "n_before_peak": len(before),
            "n_after_peak": len(after),
            "peak_date": peak,
        },
    }
