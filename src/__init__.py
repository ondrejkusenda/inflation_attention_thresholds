"""
Inflation–attention threshold analysis package.

Re-exports the most commonly used entry points so callers can simply do
``from src import run_all_sources`` (or import the relevant module directly).
"""

from data_loading import load_all
from preprocessing import prepare_data
from threshold_model import (
    threshold_regression_full,
    chow_test_stacked,
    power_analysis_post_peak,
    run_threshold_analysis,
)
from pipeline import run_full_pipeline, run_all_sources
from plotting import (
    plot_threshold_results,
    plot_single_window,
    plot_timeseries_single,
    plot_threshold_scatter,
)

__all__ = [
    "load_all",
    "prepare_data",
    "threshold_regression_full",
    "chow_test_stacked",
    "power_analysis_post_peak",
    "run_threshold_analysis",
    "run_full_pipeline",
    "run_all_sources",
    "plot_threshold_results",
    "plot_single_window",
    "plot_timeseries_single",
    "plot_threshold_scatter",
]
