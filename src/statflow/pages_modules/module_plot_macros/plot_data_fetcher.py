"""
Plot data fetching logic for visualization macros.

This module handles fetching data for plotting and visualization.

plot_data_fetcher.py
├── fetch_plot_data()  # Fetches data for plotting based on filters.
└── Plot data retrieval logic
"""

import polars as pl

from statflow.functional.mlflow.mlflow_client import fetch_all_datasets_parallel


def fetch_plot_data(
    selected_mpf: list, selected_beta: list, selected_pinflate: list, selected_datasets: list[str]
) -> pl.DataFrame | None:
    """Fetch data for plotting.

    Args:
        selected_mpf: Selected MPF.
        selected_beta: Selected beta.
        selected_pinflate: Selected pinflate.
        selected_datasets: Selected datasets.

    Returns:
        DataFrame or None.
    """
    all_runs_df = fetch_all_datasets_parallel(
        tuple(selected_mpf), tuple(selected_beta), tuple(selected_pinflate), selected_datasets
    )

    if all_runs_df.is_empty():
        return None

    if not selected_pinflate:
        slim_variants = ["slim_gsgp", "slim"]
        all_runs_df = all_runs_df.filter(~pl.col("params.variant").is_in(slim_variants))

    return all_runs_df