"""
Project-wide configuration: plotting style, colour palette, and the
country / language metadata used throughout the analysis.

Importing this module also applies the publication-quality Matplotlib
defaults (via :func:`apply_plot_style`), so a simple ``import config`` is
enough to give every figure a consistent look.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
START_DATE = "2014-01-01"   # First month included in the sample.
MAX_DATE = "2025-12-01"     # Last month included in the sample.
DEFAULT_TRIM = 0.10         # Trimming fraction for the threshold candidate grid.
DEFAULT_N_BOOTSTRAP = 1999  # Bootstrap / Monte-Carlo replications for p-values.
SIGNIFICANCE_LEVEL = 0.10   # Level at which the Hansen / Chow tests reject.

# ---------------------------------------------------------------------------
# NBS brand palette
# ---------------------------------------------------------------------------
NBS = {
    "dark_blue":   "#1C355E",
    "darker_blue": "#112039",
    "bright_blue": "#0067AB",
    "medium_blue": "#00518A",
    "pale_blue":   "#CCE1EE",
    "light_blue":  "#99C2DD",
    "gold":        "#A5835A",
    "brown":       "#7A5029",
    "wine":        "#74253E",
    "dark_wine":   "#561C2F",
    "teal":        "#00594F",
    "dark_teal":   "#003D36",
    "grey":        "#6E6E6E",
}

# ---------------------------------------------------------------------------
# Semantic colour roles (kept visually distinguishable)
# ---------------------------------------------------------------------------
C_INDEX = NBS["dark_blue"]        # Attention-index series / scatter fit.
C_INFLATION = NBS["light_blue"]   # Inflation series.
C_SCATTER = NBS["grey"]           # Scatter points.
C_THR_PRE = NBS["gold"]           # Pre-peak threshold.
C_THR_POST = NBS["brown"]         # Post-peak threshold.
C_PEAK = NBS["grey"]              # Peak vertical line.
C_FIT_LOW = NBS["light_blue"]     # Below-threshold regime fit.
C_FIT_HIGH = NBS["dark_blue"]     # Above-threshold regime fit.
C_LINEAR = NBS["medium_blue"]     # Single linear fit (no threshold).


def apply_plot_style() -> None:
    """Apply the project's publication-quality Matplotlib defaults.

    Called automatically on import so that every figure produced by the
    package shares the same serif fonts, white background, and trimmed
    spines.
    """
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.edgecolor": NBS["dark_blue"],
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


# ---------------------------------------------------------------------------
# Country metadata
# ---------------------------------------------------------------------------
# For each country: ``lan`` = Google-Trends query language, ``id`` = Eurostat
# / ISO geo code, ``inflation`` = the localised search term for "inflation".
COUNTRIES_LANGUAGE = {
    # Euro area
    "Austria":        {"lan": "de", "id": "AT", "inflation": "inflation"},
    "Belgium":        {"lan": "nl", "id": "BE", "inflation": "inflatie"},
    "Belgium_FR":     {"lan": "fr", "id": "BE", "inflation": "inflation"},
    "Bulgaria":       {"lan": "bg", "id": "BG", "inflation": "инфлация"},
    "Croatia":        {"lan": "hr", "id": "HR", "inflation": "inflacija"},
    "Cyprus":         {"lan": "en", "id": "CY", "inflation": "inflation"},
    "Estonia":        {"lan": "et", "id": "EE", "inflation": "inflation"},
    "Finland":        {"lan": "fi", "id": "FI", "inflation": "inflaatio"},
    "France":         {"lan": "fr", "id": "FR", "inflation": "inflation"},
    "Germany":        {"lan": "de", "id": "DE", "inflation": "inflation"},
    "Greece":         {"lan": "el", "id": "GR", "inflation": "inflation"},
    "Ireland":        {"lan": "en", "id": "IE", "inflation": "inflation"},
    "Italy":          {"lan": "it", "id": "IT", "inflation": "inflazione"},
    "Latvia":         {"lan": "lv", "id": "LV", "inflation": "inflācija"},
    "Lithuania":      {"lan": "lt", "id": "LT", "inflation": "infliacija"},
    "Luxembourg":     {"lan": "fr", "id": "LU", "inflation": "inflation"},
    "Malta":          {"lan": "en", "id": "MT", "inflation": "inflation"},
    "Netherlands":    {"lan": "nl", "id": "NL", "inflation": "inflatie"},
    "Portugal":       {"lan": "pt", "id": "PT", "inflation": "inflação"},
    "Slovakia":       {"lan": "sk", "id": "SK", "inflation": "inflacia"},
    "Slovenia":       {"lan": "sl", "id": "SI", "inflation": "inflacija"},
    "Spain":          {"lan": "es", "id": "ES", "inflation": "inflacion"},

    # Extra advanced economies
    "Switzerland":    {"lan": "de", "id": "CH", "inflation": "inflation"},
    "United Kingdom": {"lan": "en", "id": "GB", "inflation": "inflation"},
    "United States":  {"lan": "en", "id": "US", "inflation": "inflation"},
}

# Countries analysed in the main run, in display order.
COUNTRIES_LIST = [
    # Euro area
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Estonia",
    "Finland", "France", "Germany", "Greece", "Ireland", "Italy",
    "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands",
    "Portugal", "Slovakia", "Slovenia", "Spain",
    # Extra advanced economies
    "Switzerland", "United States", "United Kingdom",
]

# Apply the house style as soon as the configuration is imported.
apply_plot_style()
