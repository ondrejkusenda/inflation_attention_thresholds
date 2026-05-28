"""
Threshold-regression econometrics.

Implements Hansen's (1996, 1997, 2000) sample-splitting threshold model for
the relationship between inflation and attention, plus the supporting tests
used in the paper:

    * fast grid search for the threshold via cumulative-sum RSS,
    * OLS with HC1 (heteroskedasticity-robust) standard errors,
    * Hansen sup-F test with bootstrap and asymptotic p-values,
    * Hansen (2000) likelihood-ratio confidence interval for the threshold,
    * a stacked Chow-type structural-break test (pre- vs post-peak),
    * a residual-bootstrap power analysis for the post-peak window,
    * an orchestration routine that runs the tests in the paper's order.

Throughout, ``x`` is inflation (the regressor and the threshold variable) and
``y`` is the attention index.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from config import DEFAULT_TRIM, DEFAULT_N_BOOTSTRAP, SIGNIFICANCE_LEVEL



def fast_rss_all_thresholds(x, y, tvar, candidates, min_obs=3):
    """Compute the two-regime RSS for every candidate threshold.

    Uses cumulative sums so the cost is O(n + m) rather than O(n·m) of a
    naive loop that re-fits OLS at each candidate.

    Parameters
    ----------
    x, y : numpy.ndarray
        Regressor and response.
    tvar : numpy.ndarray
        Threshold variable (here, equal to ``x``).
    candidates : numpy.ndarray
        Candidate threshold values to evaluate.
    min_obs : int, optional
        Minimum observations required in each regime. Defaults to 3.

    Returns
    -------
    numpy.ndarray
        RSS for each candidate; ``np.nan`` where a candidate is infeasible.
    """
    order = np.argsort(tvar)
    x = x[order]
    y = y[order]
    tvar = tvar[order]

    n = len(y)

    cum_x = np.cumsum(x)
    cum_y = np.cumsum(y)
    cum_x2 = np.cumsum(x ** 2)
    cum_xy = np.cumsum(x * y)
    cum_y2 = np.cumsum(y ** 2)

    total_x = cum_x[-1]
    total_y = cum_y[-1]
    total_x2 = cum_x2[-1]
    total_xy = cum_xy[-1]
    total_y2 = cum_y2[-1]

    rss = np.full(len(candidates), np.nan)

    for i, g in enumerate(candidates):
        k = np.searchsorted(tvar, g, side="right")

        if k < min_obs or (n - k) < min_obs:
            continue

        # --- Below-threshold regime (first k observations) ---
        n_b = k
        sx = cum_x[k - 1]
        sy = cum_y[k - 1]
        sxx = cum_x2[k - 1]
        sxy = cum_xy[k - 1]
        syy = cum_y2[k - 1]

        det = n_b * sxx - sx * sx
        if abs(det) < 1e-12:
            continue

        b = (n_b * sxy - sx * sy) / det
        a = (sy - b * sx) / n_b
        rss_b = syy - 2 * a * sy - 2 * b * sxy + a * a * n_b + 2 * a * b * sx + b * b * sxx

        # --- Above-threshold regime (remaining observations) ---
        n_a = n - k
        sx = total_x - sx
        sy = total_y - sy
        sxx = total_x2 - sxx
        sxy = total_xy - sxy
        syy = total_y2 - syy

        det = n_a * sxx - sx * sx
        if abs(det) < 1e-12:
            continue

        b = (n_a * sxy - sx * sy) / det
        a = (sy - b * sx) / n_a
        rss_a = syy - 2 * a * sy - 2 * b * sxy + a * a * n_a + 2 * a * b * sx + b * b * sxx

        rss[i] = rss_b + rss_a

    return rss


def linear_rss(x, y):
    """Return the RSS of the single-regime (linear, no-threshold) model.

    Parameters
    ----------
    x, y : numpy.ndarray
        Regressor and response.

    Returns
    -------
    float
        Residual sum of squares of ``y = a·x + b``.
    """
    A = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coef
    return float(np.sum((y - y_hat) ** 2))



def build_candidates(y, trim=DEFAULT_TRIM):
    """Build the trimmed grid of candidate thresholds.

    Drops the lowest and highest ``trim`` fraction of the unique sorted
    values so every candidate leaves enough observations in each regime.

    Parameters
    ----------
    y : numpy.ndarray
        Values of the threshold variable.
    trim : float, optional
        Fraction trimmed from each tail. Defaults to :data:`config.DEFAULT_TRIM`.

    Returns
    -------
    numpy.ndarray
        Candidate threshold values.
    """
    vals = np.sort(np.unique(y))
    lo = int(trim * len(vals))
    hi = int((1 - trim) * len(vals)) + 1
    return vals[lo:hi]


def grid_search_fast(x, y, candidates):
    """Find the RSS-minimising threshold over the candidate grid.

    Parameters
    ----------
    x, y : numpy.ndarray
        Regressor and response.
    candidates : numpy.ndarray
        Candidate threshold values.

    Returns
    -------
    tuple
        ``(best_gamma, best_rss, rss_grid)``. ``best_gamma`` is ``None`` and
        ``best_rss`` is ``np.inf`` when no feasible threshold exists.
    """
    rss = fast_rss_all_thresholds(x, y, x, candidates)

    valid = ~np.isnan(rss)
    if not valid.any():
        return None, np.inf, rss

    idx = np.nanargmin(rss)
    return candidates[idx], rss[idx], rss


def hansen_f_stat(rss_full, rss_thr, n):
    """Hansen's sup-F statistic comparing the linear and threshold models.

    Parameters
    ----------
    rss_full : float
        RSS of the single-regime linear model.
    rss_thr : float
        RSS of the best two-regime threshold model.
    n : int
        Sample size.

    Returns
    -------
    float
        ``n · (rss_full - rss_thr) / rss_thr``.
    """
    return n * (rss_full - rss_thr) / rss_thr



def fit_ols(x, y):
    """Fit ``y = a·x + b`` by OLS with HC1 robust standard errors.

    The HC1 sandwich estimator is
    ``V = (X'X)^-1 X' diag(resid^2) X (X'X)^-1 · n/(n-k)`` with ``k = 2``.

    Parameters
    ----------
    x, y : numpy.ndarray
        Regressor and response.

    Returns
    -------
    dict
        Slope (``coef``), intercept, their robust standard errors
        (``se_coef``, ``se_intercept``), RSS, and sample size ``n``. All
        numeric fields are ``np.nan`` when the fit is infeasible.
    """
    n = len(x)
    k = 2

    nan_result = {
        "coef": np.nan, "intercept": np.nan,
        "se_coef": np.nan, "se_intercept": np.nan,
        "rss": np.nan, "n": n,
    }

    if n <= k:
        return nan_result

    X = np.column_stack([x, np.ones(n)])

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return nan_result

    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    rss = float(resid @ resid)

    # HC1 sandwich; the (X·resid) form avoids building a dense diag matrix.
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    cov_hc1 = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov_hc1))

    return {
        "coef": float(beta[0]),
        "intercept": float(beta[1]),
        "se_coef": float(se[0]),
        "se_intercept": float(se[1]),
        "rss": rss,
        "n": n,
    }


def bootstrap_pvalue(x, y, f_obs, candidates, n_bootstrap=1000, seed=42):
    """Fixed-regressor bootstrap p-value for the Hansen sup-F test.

    Resamples residuals around the linear-model fit (Hansen 1996), so the
    regressors are held fixed and only the disturbances are bootstrapped.

    Parameters
    ----------
    x, y : numpy.ndarray
        Regressor and response.
    f_obs : float
        Observed sup-F statistic.
    candidates : numpy.ndarray
        Candidate threshold grid.
    n_bootstrap : int, optional
        Number of bootstrap replications. Defaults to 1000.
    seed : int, optional
        RNG seed for reproducibility. Defaults to 42.

    Returns
    -------
    float
        Bootstrap p-value (share of replications with ``F* >= f_obs``).
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    A = np.column_stack([x, np.ones(n)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coef
    resid = y - y_hat

    count = 0
    for _ in range(n_bootstrap):
        y_boot = y_hat + rng.choice(resid, size=n, replace=True)

        rss_full = linear_rss(x, y_boot)
        _, rss_thr, _ = grid_search_fast(x, y_boot, candidates)
        if rss_thr == np.inf:
            continue

        f_boot = hansen_f_stat(rss_full, rss_thr, n)
        if f_boot >= f_obs:
            count += 1

    return count / n_bootstrap


def asymptotic_pvalue(x, y, f_obs, candidates, n_asymptotic=5000, seed=42):
    """Hansen (1997) asymptotic p-value via Monte-Carlo under an iid Gaussian null.

    Parameters
    ----------
    x, y : numpy.ndarray
        Regressor and response.
    f_obs : float
        Observed sup-F statistic.
    candidates : numpy.ndarray
        Candidate threshold grid.
    n_asymptotic : int, optional
        Number of Monte-Carlo replications. Defaults to 5000.
    seed : int, optional
        RNG seed. Defaults to 42.

    Returns
    -------
    float
        Simulated asymptotic p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    A = np.column_stack([x, np.ones(n)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coef
    resid = y - y_hat
    sigma = np.sqrt((resid @ resid) / (n - 2))

    count = 0
    for _ in range(n_asymptotic):
        y_sim = y_hat + sigma * rng.standard_normal(n)

        rss_full = linear_rss(x, y_sim)
        _, rss_thr, _ = grid_search_fast(x, y_sim, candidates)
        if rss_thr == np.inf:
            continue

        f_sim = hansen_f_stat(rss_full, rss_thr, n)
        if f_sim >= f_obs:
            count += 1

    return count / n_asymptotic


def lr_confidence_interval(rss_grid, rss_best, candidates, n, ci_level=0.90):
    """Hansen (2000) likelihood-ratio confidence interval for the threshold.

    The LR statistic ``LR(γ) = n·(SSR(γ) − SSR(γ̂)) / SSR(γ̂)`` has a
    nonstandard limiting distribution (Hansen 2000, Theorem 3). Critical
    values are taken from Hansen (2000), Table 1; the standard chi-squared
    inversion ``-2·log(1-α)`` does **not** apply.

    Parameters
    ----------
    rss_grid : numpy.ndarray
        RSS at each candidate threshold.
    rss_best : float
        RSS at the estimated threshold.
    candidates : numpy.ndarray
        Candidate threshold grid (aligned with ``rss_grid``).
    n : int
        Sample size.
    ci_level : {0.90, 0.95, 0.99}, optional
        Confidence level. Defaults to 0.90.

    Returns
    -------
    tuple
        ``(ci_low, ci_high)``; ``(np.nan, np.nan)`` if the interval is empty.

    Raises
    ------
    ValueError
        If ``ci_level`` is not a tabulated value.
    """
    cv_table = {0.90: 7.35, 0.95: 8.18, 0.99: 10.59}
    if ci_level not in cv_table:
        raise ValueError(f"ci_level must be one of {list(cv_table)}, got {ci_level}")
    c = cv_table[ci_level]

    lr = n * (rss_grid - rss_best) / rss_best
    inside = candidates[lr <= c]

    if len(inside) == 0:
        return np.nan, np.nan

    return float(inside.min()), float(inside.max())



def threshold_regression_full(df, trim=DEFAULT_TRIM, n_bootstrap=1000):
    """Estimate the threshold model and run the Hansen test on one window.

    Combines the fast grid search, the sup-F test (bootstrap and asymptotic
    p-values), the LR confidence interval, and the two regime regressions.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame with ``INFLATION`` (x) and ``INDEX`` (y) columns.
    trim : float, optional
        Trimming fraction for the candidate grid.
    n_bootstrap : int, optional
        Replications for both the bootstrap and asymptotic p-values.

    Returns
    -------
    dict
        Threshold estimate, test statistics, p-values, rejection flags,
        confidence interval, and the two regime OLS fits. On failure returns
        a dict with an ``"error"`` key.
    """
    x = df["INFLATION"].values.astype(float)
    y = df["INDEX"].values.astype(float)
    n = len(y)

    candidates = build_candidates(x, trim)
    if len(candidates) < 3:
        return {"error": "Too few candidates"}

    best_gamma, best_rss, rss_grid = grid_search_fast(x, y, candidates)
    if best_gamma is None:
        return {"error": "No threshold found"}

    rss_full = linear_rss(x, y)

    f_stat = hansen_f_stat(rss_full, best_rss, n)
    p_value = bootstrap_pvalue(x, y, f_stat, candidates, n_bootstrap)
    p_value_asym = asymptotic_pvalue(x, y, f_stat, candidates, n_bootstrap)

    ci_low, ci_high = lr_confidence_interval(rss_grid, best_rss, candidates, n)

    mask_low = x <= best_gamma
    mask_high = x > best_gamma
    reg_low = fit_ols(x[mask_low], y[mask_low])
    reg_high = fit_ols(x[mask_high], y[mask_high])

    return {
        "threshold": best_gamma,
        "f_stat": f_stat,
        "p_value_boot": p_value,
        "threshold_exists": p_value < SIGNIFICANCE_LEVEL,
        "p_value_asym": p_value_asym,
        "threshold_exists_asym": p_value_asym < SIGNIFICANCE_LEVEL,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "regime_low": reg_low,
        "regime_high": reg_high,
        "n": n,
    }



def chow_test_stacked(before_df, after_df, trim=DEFAULT_TRIM):
    """Stacked Chow-type structural-break test (paper equation 2).

    Tests whether the threshold relationship differs between the pre- and
    post-peak windows:

    1. Pool both windows and estimate a common threshold ``γ*`` under H0.
    2. Restricted model at ``γ*``:
       ``y = α1·1(π≤γ*) + β1·π·1(π≤γ*) + α2·1(π>γ*) + β2·π·1(π>γ*)``.
    3. Unrestricted model adds four interactions with a post-peak dummy.
    4. Standard F-test on the four restrictions.

    Parameters
    ----------
    before_df, after_df : pandas.DataFrame
        Pre- and post-peak windows with ``INFLATION`` / ``INDEX`` columns.
    trim : float, optional
        Trimming fraction for the pooled candidate grid.

    Returns
    -------
    dict
        F-statistic, p-value, degrees of freedom, rejection flag, the pooled
        null threshold, and the two window sizes. On failure returns a dict
        with an ``"error"`` key.
    """
    y_pre = before_df["INDEX"].values.astype(float)
    x_pre = before_df["INFLATION"].values.astype(float)
    y_post = after_df["INDEX"].values.astype(float)
    x_post = after_df["INFLATION"].values.astype(float)

    n_pre, n_post = len(y_pre), len(y_post)
    n_all = n_pre + n_post

    if n_pre < 10 or n_post < 6:
        return {"error": "Too few observations for Chow test"}

    # Pooled threshold under H0.
    y_pool = np.concatenate([y_pre, y_post])
    x_pool = np.concatenate([x_pre, x_post])
    cand_pool = build_candidates(x_pool, trim)
    if len(cand_pool) < 3:
        return {"error": "Too few candidates in pooled sample"}
    g_null, _, _ = grid_search_fast(x_pool, y_pool, cand_pool)
    if g_null is None:
        return {"error": "Pooled threshold estimation failed"}

    # Restricted and unrestricted design matrices.
    below_pool = (x_pool <= g_null).astype(float)
    above_pool = 1.0 - below_pool
    post_dummy = np.concatenate([np.zeros(n_pre), np.ones(n_post)])

    X_r = np.column_stack([
        below_pool, x_pool * below_pool,
        above_pool, x_pool * above_pool,
    ])
    X_u = np.column_stack([
        X_r,
        post_dummy * below_pool,
        post_dummy * x_pool * below_pool,
        post_dummy * above_pool,
        post_dummy * x_pool * above_pool,
    ])

    try:
        beta_r = np.linalg.lstsq(X_r, y_pool, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y_pool, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"error": "OLS failed in Chow test"}

    ssr_r = float((y_pool - X_r @ beta_r) @ (y_pool - X_r @ beta_r))
    ssr_u = float((y_pool - X_u @ beta_u) @ (y_pool - X_u @ beta_u))

    df_num = X_u.shape[1] - X_r.shape[1]   # 4 restrictions
    df_denom = n_all - X_u.shape[1]
    if df_denom <= 0 or ssr_u <= 0:
        return {"error": "Degenerate F-test"}

    f_chow = ((ssr_r - ssr_u) / df_num) / (ssr_u / df_denom)
    p_chow = 1.0 - stats.f.cdf(f_chow, df_num, df_denom)

    return {
        "gamma_null": float(g_null),
        "f_chow": float(f_chow),
        "p_chow": float(p_chow),
        "df_num": int(df_num),
        "df_denom": int(df_denom),
        "reject_chow_10": bool(p_chow <= SIGNIFICANCE_LEVEL),
        "n_pre": n_pre,
        "n_post": n_post,
    }



def power_analysis_post_peak(
    before_df,
    after_df,
    gamma_pre,
    trim=DEFAULT_TRIM,
    n_sim=500,
    andrews_critical=8.5,
    seed=42,
):
    """Power of the Hansen test on the post-peak window under the pre-peak DGP.

    Procedure:
      1. Fit the pre-peak two-regime model at ``gamma_pre`` and keep its
         coefficients and residuals.
      2. For each simulation, resample residuals regime-specifically (below /
         above ``gamma_pre``) to preserve heteroskedasticity, build a
         simulated ``y*`` at the realised post-peak inflation, and run the
         Hansen sup-F test.
      3. Reject when ``F > andrews_critical`` (Andrews 1993 fast rule, roughly
         the 10% level for the sup-F distribution).
      4. Return the rejection frequency.

    Andrews' fast rule replaces a nested bootstrap with near-identical
    conclusions at a large speed-up.

    Parameters
    ----------
    before_df, after_df : pandas.DataFrame
        Pre- and post-peak windows.
    gamma_pre : float
        Estimated pre-peak threshold (the assumed DGP threshold).
    trim : float, optional
        Trimming fraction for the post-window candidate grid.
    n_sim : int, optional
        Number of simulations. Defaults to 500.
    andrews_critical : float, optional
        sup-F rejection cut-off. Defaults to 8.5.
    seed : int, optional
        RNG seed. Defaults to 42.

    Returns
    -------
    dict
        Estimated power, simulation settings, regime counts, and a textual
        interpretation. On failure returns a dict with an ``"error"`` key.
    """
    y_pre = before_df["INDEX"].values.astype(float)
    x_pre = before_df["INFLATION"].values.astype(float)
    x_post = after_df["INFLATION"].values.astype(float)
    n_post = len(x_post)

    if n_post < 6:
        return {"error": "Post window too small for power"}

    # Fit the pre-peak two-regime model at gamma_pre.
    below_pre = (x_pre <= gamma_pre).astype(float)
    above_pre = 1.0 - below_pre
    X_pre = np.column_stack([
        below_pre, x_pre * below_pre,
        above_pre, x_pre * above_pre,
    ])
    try:
        beta_pre = np.linalg.lstsq(X_pre, y_pre, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"error": "Pre-peak OLS failed"}
    resid_pre = y_pre - X_pre @ beta_pre

    # Regime-specific residual pools (fall back to the full pool if empty).
    resid_below = resid_pre[below_pre.astype(bool)]
    resid_above = resid_pre[above_pre.astype(bool)]
    if len(resid_below) == 0:
        resid_below = resid_pre
    if len(resid_above) == 0:
        resid_above = resid_pre

    # Deterministic component at the realised post-peak inflation.
    below_post = (x_post <= gamma_pre).astype(float)
    above_post = 1.0 - below_post
    X_post_det = np.column_stack([
        below_post, x_post * below_post,
        above_post, x_post * above_post,
    ])
    yhat_post = X_post_det @ beta_pre

    # Candidate grid on x_post is constant across simulations.
    cand_post = build_candidates(x_post, trim)
    if len(cand_post) < 3:
        return {"error": "Too few candidates in post window"}

    below_mask = below_post.astype(bool)
    above_mask = above_post.astype(bool)
    n_below = int(below_mask.sum())
    n_above = int(above_mask.sum())

    rng = np.random.default_rng(seed)
    n_reject = 0

    for _ in range(n_sim):
        eps = np.empty(n_post)
        if n_below > 0:
            eps[below_mask] = rng.choice(resid_below, size=n_below, replace=True)
        if n_above > 0:
            eps[above_mask] = rng.choice(resid_above, size=n_above, replace=True)
        y_sim = yhat_post + eps

        rss_full = linear_rss(x_post, y_sim)
        _, rss_thr, _ = grid_search_fast(x_post, y_sim, cand_post)
        if rss_thr == np.inf or rss_thr <= 0 or np.isnan(rss_full):
            continue

        f_sim = hansen_f_stat(rss_full, rss_thr, n_post)
        if f_sim > andrews_critical:
            n_reject += 1

    power = n_reject / n_sim
    if power >= 0.80:
        interp = "GENUINE LINEARITY (high power, fail to reject)"
    elif power >= 0.50:
        interp = "AMBIGUOUS"
    else:
        interp = "LOW POWER (cannot distinguish)"

    return {
        "power_10": power,
        "n_sim": n_sim,
        "n_post": n_post,
        "n_below_post": n_below,
        "n_above_post": n_above,
        "gamma_dgp": float(gamma_pre),
        "andrews_crit": andrews_critical,
        "interpretation": interp,
    }




def run_threshold_analysis(prep, trim=DEFAULT_TRIM, n_bootstrap=1000):
    """Run the full per-country analysis in the paper's fixed order.

    Steps:
        1. Stacked Chow test (gateway: did the relationship change?).
        2. Pre-peak threshold + Hansen test.
        3. Korenok–Munro filter (inflation must fall back below ``γ_pre``).
        4. Post-peak threshold + Hansen test.
        5. If both windows reject linearity, report ``Δ = γ_post − γ_pre``.
        6. If the post-peak window fails to reject, run the power analysis
           under the pre-peak DGP.

    Parameters
    ----------
    prep : dict
        Output of :func:`preprocessing.prepare_data`.
    trim : float, optional
        Trimming fraction passed to the sub-routines.
    n_bootstrap : int, optional
        Replications for the Hansen p-values.

    Returns
    -------
    dict
        Keys ``chow_test``, ``before_peak``, ``after_peak``, ``delta``, and
        ``power``, each holding the corresponding result dict.
    """
    results = {}
    before_df = prep["data_before_peak"]
    after_df = prep["data_after_peak"]

    # 1. Stacked Chow test (gateway, runs first).
    results["chow_test"] = chow_test_stacked(before_df, after_df, trim=trim)

    # 2. Pre-peak threshold + Hansen.
    before = threshold_regression_full(before_df, trim, n_bootstrap)
    results["before_peak"] = before

    if "error" in before or "threshold" not in before:
        results["after_peak"] = {"skipped": "No valid before threshold"}
        results["power"] = {"skipped": "No valid before threshold"}
        results["delta"] = {"value": None, "reported": False,
                            "note": "No valid before threshold"}
        return results

    gamma_pre = before["threshold"]
    reject_pre = before.get("p_value_asym", 1.0) <= SIGNIFICANCE_LEVEL

    # 3. Korenok–Munro filter + post-peak threshold.
    fell_below = bool(after_df["INFLATION"].min() < gamma_pre)

    if not fell_below:
        results["after_peak"] = {
            "skipped": "Inflation never dropped below pre-peak threshold",
            "fell_below": False,
        }
        after = None
        reject_post = False
    else:
        after = threshold_regression_full(after_df, trim, n_bootstrap)
        after["fell_below"] = True
        results["after_peak"] = after
        reject_post = ("threshold" in after) and \
                      (after.get("p_value_asym", 1.0) <= SIGNIFICANCE_LEVEL)

    # 4. Δ when both windows reject linearity.
    if reject_pre and reject_post and after is not None and "threshold" in after:
        results["delta"] = {
            "value": after["threshold"] - gamma_pre,
            "gamma_pre": gamma_pre,
            "gamma_post": after["threshold"],
            "reported": True,
            "note": "Both pre- and post-peak Hansen tests reject at 10%.",
        }
    else:
        reason = []
        if not reject_pre:
            reason.append("pre-peak does not reject linearity")
        if not fell_below:
            reason.append("inflation never fell back below γ_pre")
        elif not reject_post:
            reason.append("post-peak does not reject linearity")
        results["delta"] = {
            "value": None,
            "reported": False,
            "note": "Δ not reported: " + "; ".join(reason),
        }

    # 5. Power analysis only when the post-peak window fails to reject.
    if reject_pre and fell_below and not reject_post:
        results["power"] = power_analysis_post_peak(
            before_df=before_df,
            after_df=after_df,
            gamma_pre=gamma_pre,
            trim=trim,
            n_sim=500,
        )
    else:
        if not reject_pre:
            note = "pre-peak threshold not identified — power test moot"
        elif not fell_below:
            note = "Korenok–Munro filter excluded country"
        else:
            note = "post-peak rejects linearity — power test unnecessary"
        results["power"] = {"skipped": note}

    return results
