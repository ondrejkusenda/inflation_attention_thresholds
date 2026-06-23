"""Pfauti (2026), equation (4), estimated exactly -- full sample and split at the peak.

Expectations appendix (Appendix A.8) of the inflation-attention package. The
script is self-contained -- it imports nothing from the rest of the project --
but lives in ``src/`` alongside the other modules. It downloads (and caches in
``data/expectations/``) the three public series it needs and writes all outputs
to ``results/expectations/``.

It produces the figure used in the paper's expectations appendix
(``results/expectations/pfauti_eq4_gains.pdf``): the attention/updating gain in
Pfauti's equation (4), estimated pre- and post-peak for the US.

Pfauti's threshold-updating regression (his eq. 4, p. 12):

    pi^e_{t+3|t} = 1{pi_{t-1} <= pibar} (b0L + b1L pi^e_{t|t-3} + b2L (pi_t - pi^e_{t|t-3}))
                 + (1 - 1{...})         (b0H + b1H pi^e_{t|t-3} + b2H (pi_t - pi^e_{t|t-3}))
                 + eps_t

where, per regime r in {L,H}:
    b0r = (1 - rho_r) pibar_r      (intercept)
    b1r = rho_r                    (perceived persistence)
    b2r / b1r = gamma_{pi,r}       (ATTENTION / updating gain)  <-- his attention measure

Definitions (Pfauti, sec. 2.2):
  * pi_t          quarter-on-quarter CPI inflation, (P_t - P_{t-3}) / P_{t-3}.
  * pi^e_{t+3|t}  one-quarter-ahead expectation = one-year-ahead survey / 4.
  * pi^e_{t|t-3}  the same survey expectation lagged three months (the prior).
  * pi_{t-1}      lagged inflation -> the threshold variable.
  * pibar         threshold, chosen jointly with the betas by minimising SSR
                  over all candidate thresholds (Gonzalo-Pitarakis 2002; Hansen 2011).
  * Inference: Newey-West HAC, 12 lags; headline test H0: gamma_L = gamma_H.

We work in ANNUALISED percent throughout (q-o-q change x4, survey expectations
left in annual units), so the estimated threshold and gains are directly
comparable to Pfauti's Table 1 (pibar = 3.91%, gamma_L = 0.18, gamma_H = 0.35).

This script (i) replicates his baseline on the full Michigan sample, then
(ii) re-estimates the SAME equation separately before and after the US
inflation peak -- the symmetric-threshold question of our paper, asked inside
Pfauti's own attention-updating specification.

Data sources (downloaded once, then cached in data/expectations/):
  * Michigan median 1y expected inflation -- FRED series MICH.
  * US CPI level (CPIAUCSL) -- FRED.
  * NY Fed Survey of Consumer Expectations -- newyorkfed.org xlsx.

Run:  python3 src/pfauti_eq4.py
Outputs land in results/expectations/ (figure PDF, CSV, SUMMARY_eq4.txt).
"""

from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # koreň repa (src/ -> ..)
DATA = ROOT / "data" / "expectations"       # cachnuté vstupné série
OUT = ROOT / "results" / "expectations"     # figúra, CSV, summary
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

MICH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MICH"
CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
SCE_URL = ("https://www.newyorkfed.org/medialibrary/interactives/sce/"
           "sce/downloads/data/FRBNY-SCE-Data.xlsx")

PEAK = pd.Timestamp("2022-06-01")   # US inflation peak used throughout the paper
TRIM = 0.15                          # threshold grid trim (each tail)
MIN_REGIME = 15                      # min obs per regime for a candidate threshold
NW_LAGS = 12                         # Newey-West lags (matches Pfauti)


# ---------------------------------------------------------------------------
# Data downloads (cached locally so the package is replicable offline)
# ---------------------------------------------------------------------------
def _fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  cached  {dest.name}")
        return dest
    print(f"  downloading {dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    # Some macOS Python builds ship without a usable CA bundle; fall back to an
    # unverified context for these public data endpoints if verification fails.
    contexts = [ssl.create_default_context()]
    try:
        import certifi
        contexts.insert(0, ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    contexts.append(ssl._create_unverified_context())
    last_err = None
    for ctx in contexts:
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                dest.write_bytes(r.read())
            return dest
        except Exception as e:                                   # noqa: BLE001
            last_err = e
    raise RuntimeError(f"failed to download {url}: {last_err}")


def load_michigan() -> pd.DataFrame:
    """Michigan median 1-year-ahead expected inflation (FRED MICH), monthly."""
    path = _fetch(MICH_URL, DATA / "MICH.csv")
    df = pd.read_csv(path)
    df.columns = ["date", "michigan"]
    df["date"] = pd.to_datetime(df["date"])
    df["michigan"] = pd.to_numeric(df["michigan"], errors="coerce")
    return df.dropna()


def load_nyfed() -> pd.DataFrame:
    """NY Fed SCE median 1-year-ahead expected inflation, monthly (since 2013-06)."""
    path = _fetch(SCE_URL, DATA / "FRBNY-SCE-Data.xlsx")
    raw = pd.read_excel(path, sheet_name="Inflation expectations", header=3)
    date_col = raw.columns[0]
    col = next(c for c in raw.columns
               if isinstance(c, str)
               and "Median one-year ahead expected inflation" in c)
    df = raw[[date_col, col]].dropna()
    df["date"] = pd.to_datetime(df[date_col].astype(int).astype(str),
                                format="%Y%m")
    return df.rename(columns={col: "nyfed"})[["date", "nyfed"]]


def load_cpi() -> pd.DataFrame:
    path = _fetch(CPI_URL, DATA / "CPIAUCSL.csv")
    df = pd.read_csv(path)
    df.columns = ["date", "cpi"]
    df["date"] = pd.to_datetime(df["date"])
    df["cpi"] = pd.to_numeric(df["cpi"], errors="coerce")
    return df.dropna()


def build(survey: pd.DataFrame, scol: str) -> pd.DataFrame:
    """Construct Pfauti's eq.(4) variables (annualised %) for one survey series."""
    cpi = load_cpi().sort_values("date").reset_index(drop=True)
    # q-o-q inflation, annualised: pi_t = 4 * (P_t - P_{t-3}) / P_{t-3} * 100
    cpi["pi"] = 4.0 * (cpi["cpi"] / cpi["cpi"].shift(3) - 1.0) * 100.0
    cpi["pi_lag1"] = cpi["pi"].shift(1)          # pi_{t-1}  -> threshold variable

    s = survey.rename(columns={scol: "S"})[["date", "S"]].sort_values("date")
    s["S_lag3"] = s["S"].shift(3)                # pi^e_{t|t-3} (annualised survey, lag 3)

    df = cpi.merge(s, on="date", how="inner").sort_values("date")
    df["y"] = df["S"]                            # pi^e_{t+3|t} (annualised)
    df["x1"] = df["S_lag3"]                       # prior expectation
    df["x2"] = df["pi"] - df["S_lag3"]            # forecast error / news
    df["z"] = df["pi_lag1"]                       # threshold variable
    return df.dropna(subset=["y", "x1", "x2", "z"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def _nw_cov(X, resid, lags):
    """Newey-West HAC covariance of the OLS coefficients."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]                         # score contributions
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    return XtX_inv @ S @ XtX_inv


def _design(df, below):
    low = below.astype(float)
    high = 1.0 - low
    one = np.ones(len(df))
    x1, x2 = df["x1"].values, df["x2"].values
    # columns: b0L, b1L, b2L, b0H, b1H, b2H
    return np.column_stack([low, low * x1, low * x2,
                            high, high * x1, high * x2])


def hansen_supF(df: pd.DataFrame, n_boot: int = 1999, seed: int = 20260615) -> dict:
    """Hansen (1996) sup-F test for a threshold in eq.(4): H0 linear vs H1 threshold.

    This is the SAME kind of linearity test used in the main text (Hansen's
    sup-F with the threshold unidentified under the null), adapted to the
    expectations-updating regression. Because the updating residuals are
    serially correlated, we report two bootstrap p-values: i.i.d. residual
    resampling (the convention in the main text) and a moving-block bootstrap
    that preserves the dependence.
    """
    z = df["z"].values
    y = df["y"].values
    n = len(df)
    Xlin = np.column_stack([np.ones(n), df["x1"].values, df["x2"].values])

    lo_q, hi_q = np.quantile(z, [TRIM, 1 - TRIM])
    cands = [c for c in np.unique(z[(z >= lo_q) & (z <= hi_q)])
             if (z <= c).sum() >= MIN_REGIME and (z > c).sum() >= MIN_REGIME]
    if len(cands) < 2 or n <= 7:
        return {"supF": np.nan, "p_iid": np.nan, "p_block": np.nan}
    if len(cands) > 100:                     # thin grid for the bootstrap
        sel = np.unique(np.linspace(0, len(cands) - 1, 100).round().astype(int))
        cands = [cands[i] for i in sel]

    # Designs depend only on z and the candidate threshold, not on y -> precompute.
    Xs = [_design(df, z <= c) for c in cands]

    def supF(yv):
        b0, r0 = _ols(Xlin, yv)
        s0 = float(r0 @ r0)
        best = 0.0
        for X in Xs:
            _, r1 = _ols(X, yv)
            s1 = float(r1 @ r1)
            F = ((s0 - s1) / 3.0) / (s1 / (n - 6))
            if F > best:
                best = F
        return best

    supF_obs = supF(y)

    # Bootstrap under H0: y* = linear fit + resampled restricted residuals.
    b0, r0 = _ols(Xlin, y)
    fit0 = Xlin @ b0
    rng = np.random.default_rng(seed)
    L = max(4, int(round(n ** (1.0 / 3.0))))     # block length
    n_blocks = int(np.ceil(n / L))
    iid_hits = blk_hits = 0
    for _ in range(n_boot):
        e_iid = r0[rng.integers(0, n, n)]
        if supF(fit0 + e_iid) >= supF_obs:
            iid_hits += 1
        starts = rng.integers(0, n - L + 1, n_blocks)
        e_blk = np.concatenate([r0[s:s + L] for s in starts])[:n]
        if supF(fit0 + e_blk) >= supF_obs:
            blk_hits += 1
    return {"supF": supF_obs,
            "p_iid": iid_hits / n_boot,
            "p_block": blk_hits / n_boot,
            "block_len": L, "n_boot": n_boot}


def estimate_eq4(df: pd.DataFrame, label: str, n_boot: int = 1999) -> dict:
    z = df["z"].values
    y = df["y"].values
    n = len(df)

    # Candidate thresholds: unique z within the trimmed interior, with at least
    # MIN_REGIME observations on each side.
    lo_q, hi_q = np.quantile(z, [TRIM, 1 - TRIM])
    cands = np.unique(z[(z >= lo_q) & (z <= hi_q)])
    best = None
    for c in cands:
        below = z <= c
        if below.sum() < MIN_REGIME or (~below).sum() < MIN_REGIME:
            continue
        X = _design(df, below)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        beta, resid = _ols(X, y)
        ssr = float(resid @ resid)
        if best is None or ssr < best["ssr"]:
            best = {"c": float(c), "ssr": ssr, "beta": beta,
                    "resid": resid, "below": below, "X": X}
    if best is None:
        return {"label": label, "n": n, "converged": False}

    beta, X, resid = best["beta"], best["X"], best["resid"]
    V = _nw_cov(X, resid, NW_LAGS)
    b0L, b1L, b2L, b0H, b1H, b2H = beta
    gL, gH = b2L / b1L, b2H / b1H

    # Delta-method covariance of (gamma_L, gamma_H) over (b1L,b2L,b1H,b2H).
    # gamma = b2/b1 ; d/db1 = -b2/b1^2 ; d/db2 = 1/b1
    idx = [1, 2, 4, 5]                      # positions of b1L,b2L,b1H,b2H
    Vg_sub = V[np.ix_(idx, idx)]
    J = np.array([
        [-b2L / b1L**2, 1.0 / b1L, 0.0, 0.0],          # d gamma_L
        [0.0, 0.0, -b2H / b1H**2, 1.0 / b1H],          # d gamma_H
    ])
    Vgamma = J @ Vg_sub @ J.T
    se_gL, se_gH = np.sqrt(np.diag(Vgamma))

    # Wald test H0: gamma_L = gamma_H
    d = gL - gH
    var_d = Vgamma[0, 0] + Vgamma[1, 1] - 2 * Vgamma[0, 1]
    wald = d**2 / var_d
    from scipy import stats
    p_equal = float(stats.chi2.sf(wald, 1))

    # No-threshold (single-regime) benchmark: ONE updating gain for all inflation
    # levels. Its size is what tells us whether agents stay attentive (high gain)
    # or go inattentive (low gain) once the threshold is gone.
    Xlin = np.column_stack([np.ones(n), df["x1"].values, df["x2"].values])
    blin, rlin = _ols(Xlin, y)
    ssr_lin = float(rlin @ rlin)
    Vlin = _nw_cov(Xlin, rlin, NW_LAGS)
    g_lin = blin[2] / blin[1]                 # single updating gain = beta2/beta1
    Jlin = np.array([-blin[2] / blin[1]**2, 1.0 / blin[1]])
    se_glin = float(np.sqrt(Jlin @ Vlin[np.ix_([1, 2], [1, 2])] @ Jlin))
    bic_thr = n * np.log(best["ssr"] / n) + 7 * np.log(n)   # 6 betas + threshold
    bic_lin = n * np.log(ssr_lin / n) + 3 * np.log(n)

    # Main-text-style linearity test: Hansen sup-F (H0 linear vs threshold)
    hs = hansen_supF(df, n_boot=n_boot)

    return {
        "label": label, "n": n, "converged": True,
        "supF": hs["supF"], "supF_p_iid": hs["p_iid"],
        "supF_p_block": hs["p_block"],
        "threshold_ann": best["c"],
        "n_below": int(best["below"].sum()),
        "n_above": int((~best["below"]).sum()),
        "rho_L": b1L, "rho_H": b1H,
        "gamma_L": gL, "se_gamma_L": se_gL,
        "gamma_H": gH, "se_gamma_H": se_gH,
        "gamma_linear": g_lin, "se_gamma_linear": se_glin, "rho_linear": blin[1],
        "delta_gamma": d, "p_gammaL_eq_gammaH": p_equal,
        "ssr_threshold": best["ssr"], "ssr_linear": ssr_lin,
        "bic_threshold": bic_thr, "bic_linear": bic_lin,
        "sample_start": df["date"].min().date().isoformat(),
        "sample_end": df["date"].max().date().isoformat(),
    }


# ---------------------------------------------------------------------------
def fmt(r: dict) -> str:
    if not r.get("converged"):
        return f"  {r['label']}: not identified (N={r.get('n')})"
    pref = "preferred" if r["bic_threshold"] < r["bic_linear"] else "NOT preferred"
    return "\n".join([
        f"  {r['label']}",
        f"    sample            : {r['sample_start']} .. {r['sample_end']}  (N={r['n']})",
        f"    threshold pibar   : {r['threshold_ann']:.2f}%  (annualised)   "
        f"[below/above N = {r['n_below']}/{r['n_above']}]",
        f"    gamma_L (low att) : {r['gamma_L']:.3f}  (se {r['se_gamma_L']:.3f})   "
        f"rho_L={r['rho_L']:.3f}",
        f"    gamma_H (high att): {r['gamma_H']:.3f}  (se {r['se_gamma_H']:.3f})   "
        f"rho_H={r['rho_H']:.3f}",
        f"    Delta gamma (H-L) : {-r['delta_gamma']:+.3f}   "
        f"H0 gamma_L=gamma_H: p={r['p_gammaL_eq_gammaH']:.3f}",
        f"    single gain (no threshold): {r['gamma_linear']:.3f}  "
        f"(se {r['se_gamma_linear']:.3f})   rho={r['rho_linear']:.3f}",
        f"    LINEARITY TESTS:",
        f"      Hansen sup-F (main-text style): F={r['supF']:.2f}  "
        f"p_iid={r['supF_p_iid']:.3f}  p_block={r['supF_p_block']:.3f}",
        f"      Pfauti BIC (thr vs linear): {r['bic_threshold']:.1f} vs "
        f"{r['bic_linear']:.1f} -> threshold {pref};  "
        f"Wald gamma_L=gamma_H p={r['p_gammaL_eq_gammaH']:.3f}",
    ])


COLOR_BLUE = "#1C355E"
COLOR_ACCENT = "#A5835A"
COLOR_GREY = "#6E6E6E"


def figure_gains(pre: dict, post: dict, title: str, ax) -> None:
    """Pre-peak two-regime gains vs post-peak single gain, with 95% CIs."""
    z = 1.96
    pts = [
        (0, pre["gamma_L"], pre["se_gamma_L"], COLOR_BLUE,
         r"$\widehat{\gamma}_L$ (pre, low)"),
        (1, pre["gamma_H"], pre["se_gamma_H"], COLOR_ACCENT,
         r"$\widehat{\gamma}_H$ (pre, high)"),
        (2, post["gamma_linear"], post["se_gamma_linear"], COLOR_GREY,
         r"$\widehat{\gamma}$ (post, single)"),
    ]
    for x, g, se, c, lab in pts:
        ax.errorbar(x, g, yerr=z * se, fmt="o", color=c, ecolor=c,
                    elinewidth=1.4, capsize=4, markersize=7, label=lab)
    # span the pre-peak regimes to show the jump
    ax.plot([0, 1], [pre["gamma_L"], pre["gamma_H"]], color=COLOR_GREY,
            linestyle=":", linewidth=1.0, zorder=0)
    ax.axhline(0, color=COLOR_GREY, linewidth=0.7, linestyle="-", alpha=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["pre\n$\\gamma_L$", "pre\n$\\gamma_H$", "post\nsingle"],
                       fontsize=9)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylabel(r"updating gain $\gamma_\pi=\beta_2/\beta_1$")
    ax.set_title(title, fontsize=10)


PFAUTI_END = pd.Timestamp("2024-05-01")   # Pfauti's baseline sample ends 2024M5


def main() -> None:
    print("Loading data ...")
    mich = load_michigan()
    nyfed = load_nyfed()
    m_df = build(mich, "michigan")
    n_df = build(nyfed, "nyfed")

    # (label, full-variable frame, date filter)
    specs = [
        ("Michigan median (Pfauti window 1978M1-2024M5, replication)",
         m_df, lambda d: d[d["date"] <= PFAUTI_END]),
        ("Michigan median (FULL, through latest)",
         m_df, lambda d: d),
        ("Michigan median (pre-peak <= 2022-06)",
         m_df, lambda d: d[d["date"] <= PEAK]),
        ("Michigan median (post-peak > 2022-06)",
         m_df, lambda d: d[d["date"] > PEAK]),
        ("NY Fed SCE median (FULL 2013-)",
         n_df, lambda d: d),
        ("NY Fed SCE median (pre-peak <= 2022-06)",
         n_df, lambda d: d[d["date"] <= PEAK]),
        ("NY Fed SCE median (post-peak > 2022-06)",
         n_df, lambda d: d[d["date"] > PEAK]),
    ]

    results = []
    for label, df, flt in specs:
        results.append(estimate_eq4(flt(df).reset_index(drop=True), label))

    print("\n" + "=" * 74)
    print("PFAUTI (2026) EQUATION 4 -- US, FULL SAMPLE AND SPLIT AT THE PEAK")
    print("=" * 74)
    body = "\n\n".join(fmt(r) for r in results)
    print(body)
    print("\nReference (Pfauti Table 1, Michigan mean, 1978-2024): "
          "pibar=3.91%, gamma_L=0.18, gamma_H=0.35, p(equal)=0.000")

    # Figure: pre-peak two regimes vs post-peak single gain (Michigan + NY Fed)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {r["label"]: r for r in results}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    figure_gains(by["Michigan median (pre-peak <= 2022-06)"],
                 by["Michigan median (post-peak > 2022-06)"],
                 "Michigan (median 1y) inflation expectations", axes[0])
    figure_gains(by["NY Fed SCE median (pre-peak <= 2022-06)"],
                 by["NY Fed SCE median (post-peak > 2022-06)"],
                 "NY Fed SCE (median 1y) inflation expectations", axes[1])
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "pfauti_eq4_gains.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {(OUT/'pfauti_eq4_gains.pdf').relative_to(ROOT)}")

    pd.DataFrame(results).to_csv(OUT / "pfauti_eq4_results.csv", index=False)
    (OUT / "SUMMARY_eq4.txt").write_text(
        "Pfauti (2026) eq. (4), estimated exactly. US.\n"
        "Annualised %. Threshold variable = lagged q-o-q CPI inflation.\n"
        "gamma_r = beta2_r / beta1_r is the attention/updating gain.\n\n"
        + body
        + "\n\nReference (Pfauti Table 1, Michigan mean, 1978-2024): "
        "pibar=3.91%, gamma_L=0.18, gamma_H=0.35, p(equal)=0.000\n")
    print(f"\n  wrote {(OUT/'pfauti_eq4_results.csv').relative_to(ROOT)}")
    print(f"  wrote {(OUT/'SUMMARY_eq4.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
