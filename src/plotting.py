"""
Plotting.

Publication-quality figures for the threshold analysis. All functions use
the house style applied by :mod:`config` and draw a two-regime fit only when
the relevant Hansen test rejects linearity; otherwise they show a single OLS
line.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    C_INDEX, C_INFLATION, C_SCATTER, C_THR_PRE, C_THR_POST, C_PEAK,
    C_FIT_LOW, C_FIT_HIGH, C_LINEAR, NBS,
    SIGNIFICANCE_LEVEL, DEFAULT_TRIM, DEFAULT_N_BOOTSTRAP,
)
from preprocessing import prepare_data
from threshold_model import threshold_regression_full, fit_ols


def _draw_scatter_fit(ax, df, res, index, is_after=False, gamma_pre_ref=None):
    """Draw a regime / linear scatter fit onto an existing axis.

    Plots the points and then either the two-regime threshold fit (when the
    Hansen test rejects) or a single linear OLS line. On the post-peak panel
    the pre-peak threshold can be shown as a dashed reference line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    df : pandas.DataFrame or None
        Window data with ``INFLATION`` / ``INDEX`` columns.
    res : dict
        Threshold-regression result for this window.
    index : str
        Attention-source label used in axis titles.
    is_after : bool, optional
        Whether this is the post-peak window. Defaults to ``False``.
    gamma_pre_ref : float or None, optional
        Pre-peak threshold to draw as a reference on the post-peak panel.
    """
    if df is None or len(df) == 0:
        return

    x = df["INFLATION"].values.astype(float)
    y = df["INDEX"].values.astype(float)

    ax.scatter(x, y, color=C_SCATTER, s=35, alpha=0.8,
               edgecolor="white", linewidth=0.5, zorder=3)

    # Reference pre-peak threshold on the post-peak panel.
    if is_after and gamma_pre_ref is not None:
        ax.axvline(gamma_pre_ref, color=C_THR_PRE, linestyle=(0, (5, 3)),
                   alpha=0.7, lw=1.5, zorder=2,
                   label=fr"$\gamma^{{pre}}$ ({gamma_pre_ref:.2f})")

    rejected = res.get("p_value_asym", 1.0) <= SIGNIFICANCE_LEVEL

    if "threshold" in res and rejected:
        thr = res["threshold"]
        mask_low = x <= thr
        mask_high = x > thr

        vline_color = C_THR_POST if is_after else C_THR_PRE
        thr_label = (fr"$\gamma^{{post}}$ ({thr:.2f})" if is_after
                     else fr"$\gamma^{{pre}}$ ({thr:.2f})")

        ax.axvline(thr, color=vline_color, linestyle="dashed",
                   alpha=0.85, lw=1.8, zorder=2, label=thr_label)

        if "regime_low" in res and mask_low.sum() > 0:
            r = res["regime_low"]
            xs_low = np.append(np.sort(x[mask_low]), thr)
            ax.plot(xs_low, r["coef"] * xs_low + r["intercept"],
                    color=C_FIT_LOW, lw=2.5, zorder=4,
                    label="Fit below threshold")

        if "regime_high" in res and mask_high.sum() > 0:
            r = res["regime_high"]
            xs_high = np.insert(np.sort(x[mask_high]), 0, thr)
            ax.plot(xs_high, r["coef"] * xs_high + r["intercept"],
                    color=C_FIT_HIGH, lw=2.5, zorder=4,
                    label="Fit above threshold")
    else:
        if len(x) >= 2:
            lin = fit_ols(x, y)
            if not np.isnan(lin["coef"]):
                xs = np.linspace(x.min(), x.max(), 100)
                ax.plot(xs, lin["coef"] * xs + lin["intercept"],
                        color=C_LINEAR, lw=2.5, zorder=4, label="Linear fit")

    ax.set_xlabel("Inflation (y-o-y %)")
    ax.set_ylabel(f"{index} index")
    ax.legend(loc="best")


def plot_threshold_results(country, index, prep, results):
    """Three-panel figure: time series, pre-peak scatter, post-peak scatter.

    Parameters
    ----------
    country : str
        Country name (used in titles).
    index : str
        Attention-source label.
    prep : dict
        Output of :func:`preprocessing.prepare_data`.
    results : dict
        Output of :func:`threshold_model.run_threshold_analysis`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dataset = prep["dataset_clean"]
    data_before_peak = prep["data_before_peak"]
    data_after_peak = prep["data_after_peak"]
    peak = prep["out_data"]["peak_date"]

    res_before = results.get("before_peak", {})
    res_after = results.get("after_peak", {})

    fig, axs = plt.subplots(1, 3, figsize=(22, 6))

    # 1. Time series with peak marker and (rejected) thresholds.
    axs[0].plot(dataset["TIME"], dataset["INDEX"],
                label=f"{index} index", color=C_INDEX, lw=2)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(f"{index} index", color=C_INDEX)
    axs[0].tick_params(axis="y", labelcolor=C_INDEX)

    ax2 = axs[0].twinx()
    ax2.plot(dataset["TIME"], dataset["INFLATION"],
             label="Inflation", color=C_INFLATION, lw=2)
    ax2.axvline(x=peak, color=C_PEAK, linestyle="dotted", alpha=0.8,
                label=f"Peak ({str(peak)[:7]})")

    rejected_before = res_before.get("p_value_asym", 1.0) <= SIGNIFICANCE_LEVEL
    rejected_after = res_after.get("p_value_asym", 1.0) <= SIGNIFICANCE_LEVEL

    if "threshold" in res_before and rejected_before:
        ax2.axhline(y=res_before["threshold"], color=C_THR_PRE,
                    linestyle="dashed", alpha=0.85,
                    label=fr"$\gamma^{{pre}}$ ({res_before['threshold']:.2f})")
    if "threshold" in res_after and rejected_after:
        ax2.axhline(y=res_after["threshold"], color=C_THR_POST,
                    linestyle="dashed", alpha=0.85,
                    label=fr"$\gamma^{{post}}$ ({res_after['threshold']:.2f})")

    ax2.set_ylabel("Inflation (y-o-y %)", color=C_INFLATION)
    ax2.tick_params(axis="y", labelcolor=C_INFLATION)
    ax2.grid(False)

    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    axs[0].set_title(f"{index} vs inflation — {country}")

    # 2. Pre-peak scatter.
    _draw_scatter_fit(axs[1], data_before_peak, res_before, index,
                      is_after=False)
    axs[1].set_title(f"{country} — Before peak")

    # 3. Post-peak scatter, with the pre-peak threshold as a reference.
    _draw_scatter_fit(axs[2], data_after_peak, res_after, index,
                      is_after=True,
                      gamma_pre_ref=res_before.get("threshold"))
    axs[2].set_title(f"{country} — After peak")

    plt.tight_layout()
    return fig


def plot_single_window(
    country,
    index,
    window,
    countries_language,
    eurostat_data,
    google_data=None,
    gdelt_data=None,
    trim=DEFAULT_TRIM,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    figsize=(7, 6),
    show_pre_threshold=True,
    save_path=None,
):
    """Standalone scatter for one country and one window.

    Parameters
    ----------
    country : str
        Country name.
    index : {"GOOGLE", "GDELT"}
        Attention source.
    window : {"Before", "After"}
        Which window to plot.
    countries_language : dict
        Country metadata.
    eurostat_data : pandas.DataFrame
        Inflation data.
    google_data, gdelt_data : pandas.DataFrame, optional
        Attention-index data.
    trim : float, optional
        Trimming fraction for the threshold estimation.
    n_bootstrap : int, optional
        Replications for the Hansen p-values.
    figsize : tuple, optional
        Figure size. Defaults to ``(7, 6)``.
    show_pre_threshold : bool, optional
        On the post-peak window, draw the pre-peak threshold as a reference.
    save_path : str or None, optional
        If given, save the figure to this path at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``window`` is invalid or the window has fewer than 3 observations.
    """
    if window not in ("Before", "After"):
        raise ValueError("window must be 'Before' or 'After'")

    prep = prepare_data(
        country=country, index=index,
        countries_language=countries_language,
        eurostat_data=eurostat_data,
        google_data=google_data, gdelt_data=gdelt_data,
    )

    is_after = (window == "After")
    df = prep["data_after_peak"] if is_after else prep["data_before_peak"]
    if df is None or len(df) < 3:
        raise ValueError(f"Not enough data for {country} ({window} peak)")

    res = threshold_regression_full(df, trim=trim, n_bootstrap=n_bootstrap)

    gamma_pre_ref = None
    if is_after and show_pre_threshold:
        res_pre = threshold_regression_full(
            prep["data_before_peak"], trim=trim, n_bootstrap=n_bootstrap)
        gamma_pre_ref = res_pre.get("threshold")

    fig, ax = plt.subplots(figsize=figsize)
    _draw_scatter_fit(ax, df, res, index, is_after=is_after,
                      gamma_pre_ref=gamma_pre_ref)
    ax.set_title(f"{country} - {window} peak")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_timeseries_single(
    country,
    index,
    countries_language,
    eurostat_data,
    google_data=None,
    gdelt_data=None,
    trim=DEFAULT_TRIM,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    figsize=(10, 5),
    show_thresholds=True,
    save_path=None,
):
    """Single dual-axis time-series plot (attention index vs inflation).

    The left axis shows the attention index and the right axis shows
    inflation, with the peak marked and the pre/post thresholds drawn only
    when their Hansen test rejects.

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
    trim : float, optional
        Trimming fraction for the threshold estimation.
    n_bootstrap : int, optional
        Replications for the Hansen p-values.
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 5)``.
    show_thresholds : bool, optional
        Whether to draw the threshold lines. Defaults to ``True``.
    save_path : str or None, optional
        If given, save the figure to this path at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    prep = prepare_data(
        country=country, index=index,
        countries_language=countries_language,
        eurostat_data=eurostat_data,
        google_data=google_data, gdelt_data=gdelt_data,
    )
    dataset = prep["dataset_clean"]
    peak = prep["out_data"]["peak_date"]

    res_pre = threshold_regression_full(
        prep["data_before_peak"], trim=trim, n_bootstrap=n_bootstrap)
    after_df = prep["data_after_peak"]
    res_post = (
        threshold_regression_full(after_df, trim=trim, n_bootstrap=n_bootstrap)
        if (after_df is not None and len(after_df) >= 3) else {}
    )

    rej_pre = res_pre.get("p_value_asym", 1.0) <= SIGNIFICANCE_LEVEL
    rej_post = res_post.get("p_value_asym", 1.0) <= SIGNIFICANCE_LEVEL

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(dataset["TIME"], dataset["INDEX"],
             label=f"{index} index", color=C_INDEX, lw=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"{index} index", color=C_INDEX)
    ax1.tick_params(axis="y", labelcolor=C_INDEX)

    ax2 = ax1.twinx()
    ax2.plot(dataset["TIME"], dataset["INFLATION"],
             label="Inflation", color=C_INFLATION, lw=2)
    ax2.axvline(x=peak, color=C_PEAK, linestyle="dotted", alpha=0.8,
                label=f"Peak ({str(peak)[:7]})")

    if show_thresholds and "threshold" in res_pre and rej_pre:
        ax2.axhline(res_pre["threshold"], color=C_THR_PRE,
                    linestyle="dashed", alpha=0.85, lw=1.8,
                    label=fr"$\gamma^{{pre}}$ ({res_pre['threshold']:.2f})")
    if show_thresholds and "threshold" in res_post and rej_post:
        ax2.axhline(res_post["threshold"], color=C_THR_POST,
                    linestyle="dashed", alpha=0.85, lw=1.8,
                    label=fr"$\gamma^{{post}}$ ({res_post['threshold']:.2f})")

    ax2.set_ylabel("Inflation (y-o-y %)", color=C_INFLATION)
    ax2.tick_params(axis="y", labelcolor=C_INFLATION)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"{country} - {index} index vs Inflation")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_threshold_scatter(results_list, source_name, ax, iso_labels):
    """Cross-country scatter of pre-peak vs post-peak thresholds.

    Each country is a point at ``(γ_pre, γ_post)``, coloured by whether the
    post-peak Hansen test rejects, with a 45° symmetry reference line.

    Parameters
    ----------
    results_list : list of dict
        Summary rows from the pipeline for one attention source.
    source_name : str
        Display name for the source (e.g. "Google Trends").
    ax : matplotlib.axes.Axes
        Target axis.
    iso_labels : dict
        Mapping from country name to ISO code for point annotations.
    """
    df = pd.DataFrame(results_list).copy()
    df["reject_after"] = df["reject_after"].fillna(False).astype(bool)

    df_reject = df[df["reject_after"]]
    df_no_reject = df[~df["reject_after"]]

    all_vals = pd.concat([df["threshold_before"], df["threshold_after"]]).dropna()
    if len(all_vals) == 0:
        ax.set_title(f"{source_name}: no data")
        return

    lim_lo = max(0, all_vals.min() - 0.5)
    lim_hi = all_vals.max() + 0.5

    # 45° symmetry line.
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="black",
            linestyle="dashed", alpha=0.5, lw=1, label="45° line (symmetric)")

    categories = [
        (df_reject, NBS["dark_blue"], "Reject post-peak (p ≤ 0.10)"),
        (df_no_reject, NBS["gold"], "No reject post-peak"),
    ]
    for df_cat, color, label in categories:
        if len(df_cat) == 0:
            continue
        ax.scatter(df_cat["threshold_before"], df_cat["threshold_after"],
                   color=color, s=80, zorder=5, label=label)
        for _, row in df_cat.iterrows():
            iso = iso_labels.get(row["country"], row["country"][:2])
            ax.annotate(iso, (row["threshold_before"], row["threshold_after"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r"$\hat{\gamma}^{pre}$ (pre-peak threshold, %)", fontsize=11)
    ax.set_ylabel(r"$\hat{\gamma}^{post}$ (post-peak threshold, %)", fontsize=11)
    ax.set_title(f"{source_name}: pre-peak vs post-peak thresholds", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
