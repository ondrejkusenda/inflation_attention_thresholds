"""
Data loading.

Reads the three raw input files shipped in ``data/`` and returns clean
``pandas`` frames. Every loader guarantees that the ``TIME`` column is a
proper ``datetime64`` so that downstream date comparisons work, and that the
expected ``GEO`` / ``TIME`` / ``VALUE`` columns are present.

Expected schema (all three files share it):
    GEO    : str    ISO / Eurostat geo code (e.g. "SK", "DE").
    TIME   : date   Monthly observation date.
    VALUE  : float  Inflation (y-o-y %) or attention-index level.
"""

from __future__ import annotations

import os

import pandas as pd

# Column layout shared by every input file.
REQUIRED_COLUMNS = ("GEO", "TIME", "VALUE")

# Default file names inside the ``data/`` directory.
GOOGLE_FILE = "GOOGLE_DATA.csv"
INFLATION_FILE = "INFLATION_DATA.csv"
GDELT_FILE = "GDELT_DATA.csv"


def _read_csv(path: str) -> pd.DataFrame:
    """Read one input CSV and normalise its schema.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Frame with a parsed ``TIME`` column and validated columns.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If any of the required columns is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)} is missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Parse dates once, here, so the rest of the pipeline can compare against
    # ``pd.Timestamp`` safely.
    df["TIME"] = pd.to_datetime(df["TIME"])
    return df


def load_all(data_dir: str = "data") -> dict[str, pd.DataFrame]:
    """Load the Google, inflation, and GDELT datasets.

    Parameters
    ----------
    data_dir : str, optional
        Directory holding the three CSV files. Defaults to ``"data"``.

    Returns
    -------
    dict
        Mapping with keys ``"google"``, ``"eurostat"``, and ``"gdelt"``,
        each holding the corresponding :class:`pandas.DataFrame`.
    """
    return {
        "google": _read_csv(os.path.join(data_dir, GOOGLE_FILE)),
        "eurostat": _read_csv(os.path.join(data_dir, INFLATION_FILE)),
        "gdelt": _read_csv(os.path.join(data_dir, GDELT_FILE)),
    }
