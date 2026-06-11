"""
Supply/demand decomposition of the post-peak attention decline (Appendix A.10).

Question: does household search (Google) fall after the inflation peak because
the media stopped covering inflation (supply-side), or because households
stopped searching even though coverage continued (demand-side)?

Test: for each country's post-peak window, regress Google on inflation alone
versus Google on inflation + GDELT. If GDELT absorbs the inflation coefficient
(high "mediation"), the decline is supply-driven; if inflation stays
significant regardless of GDELT, it is demand-driven.

Specification (matches the paper / the standalone colleague script):
    * sample restricted to dates <= MAX_DATE,
    * the inflation peak is located within 2021-01..2023-12,
    * only the post-peak window is used, countries with < 10 post-peak months
      are dropped,
    * pooled regressions add country fixed effects,
    * standard errors are heteroskedasticity-robust (HC1).

This reproduces the headline A.10 numbers from the three shipped CSVs (Belgium
summed across its two language queries), so no external panel file is needed:
    N = 931, inflation 1.97 -> 1.32, GDELT 403.7.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from config import COUNTRIES_LIST, START_DATE, MAX_DATE

# Peak-search window (the inflation surge), as in the paper.
PEAK_WINDOW = ("2021-01-01", "2023-12-01")
# Minimum post-peak observations for a country to enter the analysis.
MIN_POST_OBS = 10
# Per-country supply/demand classification thresholds.
MEDIATION_SUPPLY = 50.0   # % of the inflation coefficient absorbed by GDELT
MEDIATION_DEMAND = 50.0
SIG = 0.10


def build_panel(
    countries_language,
    eurostat_data,
    google_data,
    gdelt_data,
    countries_list=None,
    max_date=MAX_DATE,
):
    """Build the pooled country-month panel from the three shipped CSVs.

    Google attention, inflation, and GDELT are merged on the monthly date for
    each country and stacked. Belgium's two language queries are summed,
    mirroring :func:`preprocessing.prepare_data`. Only rows with all three
    series present are kept. Equivalent to the colleague's ``panel.parquet``.

    Returns
    -------
    pandas.DataFrame
        Columns ``country`` / ``GEO`` / ``date`` / ``google`` / ``inflation`` /
        ``gdelt``.
    """
    if countries_list is None:
        countries_list = COUNTRIES_LIST
    end = pd.Timestamp(max_date)

    frames = []
    for country in countries_list:
        geo = countries_language[country]["id"]

        g = google_data[google_data["GEO"] == geo]
        if geo == "BE":
            g = g.groupby("TIME")["VALUE"].sum().reset_index()
        else:
            g = g[["TIME", "VALUE"]]
        g = g.rename(columns={"VALUE": "google"})

        inf = eurostat_data[eurostat_data["GEO"] == geo][["TIME", "VALUE"]] \
            .rename(columns={"VALUE": "inflation"})
        nd = gdelt_data[gdelt_data["GEO"] == geo][["TIME", "VALUE"]] \
            .rename(columns={"VALUE": "gdelt"})

        m = g.merge(inf, on="TIME", how="inner").merge(nd, on="TIME", how="inner")
        m = m[m["TIME"] <= end].dropna().sort_values("TIME").reset_index(drop=True)
        m["country"] = country
        m["GEO"] = geo
        frames.append(m.rename(columns={"TIME": "date"}))

    panel = pd.concat(frames, ignore_index=True)
    return panel[["country", "GEO", "date", "google", "inflation", "gdelt"]]


def find_peak(df):
    """Locate the inflation peak within :data:`PEAK_WINDOW`."""
    lo, hi = PEAK_WINDOW
    mask = (df["date"] >= lo) & (df["date"] <= hi)
    if mask.sum() == 0:
        return None
    return df.loc[df.loc[mask, "inflation"].idxmax(), "date"]


def ols_robust(y, X, var_names):
    """OLS with HC1 heteroskedasticity-robust standard errors.

    Returns a dict with ``beta`` / ``se`` / ``t`` / ``pval`` arrays, ``r2``,
    ``n``, ``k``, and ``var_names``.
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    ssr = float(resid @ resid)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan

    # HC1 sandwich (the (X*resid) form avoids a dense diag matrix).
    Xr = X * resid[:, None]
    meat = Xr.T @ Xr
    cov = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pval = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))

    return {"beta": beta, "se": se, "t": t, "pval": pval,
            "r2": r2, "n": int(n), "k": int(k), "var_names": list(var_names)}


def _classify(mediation_pct, p_gdelt, p_inf_with_gdelt):
    """Per-country supply/demand label.

    Supply-driven: GDELT is significant and absorbs > MEDIATION_SUPPLY % of the
    inflation coefficient. Demand-driven: GDELT is insignificant and mediates
    little, so the inflation effect survives. Otherwise mixed. This rule
    reproduces the paper's named examples (FI/ES/IT/LT supply; US/DE/FR/NL
    demand).
    """
    if np.isnan(mediation_pct):
        return "mixed"
    if p_gdelt < SIG and mediation_pct > MEDIATION_SUPPLY:
        return "supply-driven"
    if p_gdelt >= SIG and mediation_pct < MEDIATION_DEMAND:
        return "demand-driven"
    return "mixed"


def country_regressions(panel):
    """Per-country post-peak Model 1 / Model 2 regressions and classification.

    Returns
    -------
    pandas.DataFrame
        One row per analysed country with both models' inflation coefficients,
        the GDELT coefficient, mediation %, robust t / p values, and the
        supply/demand label.
    """
    rows = []
    for country in sorted(panel["country"].unique()):
        df = panel[panel["country"] == country].sort_values("date").reset_index(drop=True)
        peak = find_peak(df)
        if peak is None:
            continue
        post = df[df["date"] > peak]
        if len(post) < MIN_POST_OBS:
            continue

        y = post["google"].values.astype(float)
        xi = post["inflation"].values.astype(float)
        xg = post["gdelt"].values.astype(float)
        n = len(y)

        fit1 = ols_robust(y, np.column_stack([np.ones(n), xi]), ["const", "inflation"])
        fit2 = ols_robust(y, np.column_stack([np.ones(n), xi, xg]),
                          ["const", "inflation", "gdelt"])

        b_inf_alone = fit1["beta"][1]
        b_inf_with = fit2["beta"][1]
        b_gdelt = fit2["beta"][2]
        mediation = (100.0 * (1.0 - b_inf_with / b_inf_alone)
                     if abs(b_inf_alone) > 1e-8 else np.nan)
        label = _classify(mediation, fit2["pval"][2], fit2["pval"][1])

        rows.append({
            "country": country,
            "n_post": n,
            "beta_inf_alone": float(b_inf_alone),
            "t_inf_alone": float(fit1["t"][1]),
            "p_inf_alone": float(fit1["pval"][1]),
            "beta_inf_with_gdelt": float(b_inf_with),
            "t_inf_with_gdelt": float(fit2["t"][1]),
            "p_inf_with_gdelt": float(fit2["pval"][1]),
            "beta_gdelt": float(b_gdelt),
            "t_gdelt": float(fit2["t"][2]),
            "p_gdelt": float(fit2["pval"][2]),
            "mediation_pct": float(mediation) if not np.isnan(mediation) else None,
            "r2_model1": float(fit1["r2"]),
            "r2_model2": float(fit2["r2"]),
            "label": label,
        })
    return pd.DataFrame(rows)


def pooled_regression(panel):
    """Pooled post-peak regressions with country fixed effects (A.10 headline).

    Model 1: google = const + beta * inflation + country FE
    Model 2: google = const + beta * inflation + gamma * GDELT + country FE
    Standard errors are HC1. Only the post-peak windows enter the pool.

    Returns
    -------
    dict
        Inflation coefficient with/without GDELT, the GDELT coefficient, their
        robust t / p values, mediation %, sample size, and country count.
    """
    post_frames = []
    for country in sorted(panel["country"].unique()):
        df = panel[panel["country"] == country].sort_values("date").reset_index(drop=True)
        peak = find_peak(df)
        if peak is None:
            continue
        post = df[df["date"] > peak]
        if len(post) < MIN_POST_OBS:
            continue
        post_frames.append(post)

    post_all = pd.concat(post_frames, ignore_index=True)
    y = post_all["google"].values.astype(float)
    xi = post_all["inflation"].values.astype(float)
    xg = post_all["gdelt"].values.astype(float)
    fe = pd.get_dummies(post_all["country"], drop_first=True).values.astype(float)
    n = len(y)

    fit1 = ols_robust(y, np.column_stack([np.ones(n), xi, fe]),
                      ["const", "inflation"])
    fit2 = ols_robust(y, np.column_stack([np.ones(n), xi, xg, fe]),
                      ["const", "inflation", "gdelt"])

    b_alone = fit1["beta"][1]
    b_with = fit2["beta"][1]
    b_gdelt = fit2["beta"][2]
    mediation = 100.0 * (1.0 - b_with / b_alone) if abs(b_alone) > 1e-8 else np.nan

    return {
        "n": int(n),
        "n_countries": int(post_all["country"].nunique()),
        "beta_inf_alone": float(b_alone),
        "t_inf_alone": float(fit1["t"][1]),
        "p_inf_alone": float(fit1["pval"][1]),
        "beta_inf_with_gdelt": float(b_with),
        "t_inf_with_gdelt": float(fit2["t"][1]),
        "p_inf_with_gdelt": float(fit2["pval"][1]),
        "beta_gdelt": float(b_gdelt),
        "t_gdelt": float(fit2["t"][2]),
        "p_gdelt": float(fit2["pval"][2]),
        "mediation_pct": float(mediation) if not np.isnan(mediation) else None,
        "r2_model1": float(fit1["r2"]),
        "r2_model2": float(fit2["r2"]),
    }


def run_supply_demand_decomposition(
    countries_language,
    eurostat_data,
    google_data,
    gdelt_data,
    countries_list=None,
    output_dir=None,
):
    """Run the A.10 decomposition end-to-end and optionally persist it.

    Returns
    -------
    dict
        ``pooled`` (headline FE regression), ``by_country`` (list of per-country
        rows), ``supply_driven`` / ``demand_driven`` country lists, and
        ``meta``.
    """
    panel = build_panel(countries_language, eurostat_data, google_data,
                        gdelt_data, countries_list=countries_list)

    pooled = pooled_regression(panel)
    country_df = country_regressions(panel)

    supply = sorted(country_df.loc[country_df["label"] == "supply-driven", "country"])
    demand = sorted(country_df.loc[country_df["label"] == "demand-driven", "country"])

    result = {
        "pooled": pooled,
        "by_country": country_df.to_dict(orient="records"),
        "supply_driven": supply,
        "demand_driven": demand,
        "meta": {
            "n_panel": int(len(panel)),
            "peak_window": PEAK_WINDOW,
            "min_post_obs": MIN_POST_OBS,
            "spec": "post-peak; google ~ inflation (+ gdelt) + country FE; HC1 SE",
        },
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        country_df.to_csv(
            os.path.join(output_dir, "supply_demand_country_regressions.csv"),
            index=False,
        )
        with open(os.path.join(output_dir, "supply_demand_decomposition.json"), "w") as f:
            json.dump(result, f, default=str, indent=4)
        print(f"Saved supply/demand decomposition -> {output_dir}/")
        print(f"  pooled N={pooled['n']}  "
              f"inflation {pooled['beta_inf_alone']:.2f} -> "
              f"{pooled['beta_inf_with_gdelt']:.2f}  "
              f"GDELT {pooled['beta_gdelt']:.1f}")
        print(f"  supply-driven: {supply}")
        print(f"  demand-driven: {demand}")

    return result