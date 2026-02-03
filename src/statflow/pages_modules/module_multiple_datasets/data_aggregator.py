"""
Data aggregation logic for multiple datasets comparison.

This module handles fetching and aggregating data across multiple datasets.

data_aggregator.py
├── fetch_and_process_multiple_datasets()  # Fetches and processes data for multiple datasets.
└── Data aggregation and filtering logic
"""

import polars as pl

from statflow.functional.mlflow.mlflow_client import fetch_all_datasets_parallel


def fetch_and_process_multiple_datasets(
    selected_mpf: tuple, selected_beta: tuple, selected_pinflate: tuple, selected_datasets: list[str], selected_metrics: list[str] | None = None
) -> pl.DataFrame | None:
    """Fetch and process data for multiple datasets.

    Args:
        selected_mpf: Selected MPF values.
        selected_beta: Selected beta values.
        selected_pinflate: Selected pinflate values.
        selected_datasets: List of selected datasets.
        selected_metrics: Optional list of metric names to include (without 'metrics.' prefix).

    Returns:
        Processed DataFrame with all runs, or None if no data.
    """
    all_runs_df = fetch_all_datasets_parallel(
        selected_mpf, selected_beta, selected_pinflate, selected_datasets
    )

    if all_runs_df.is_empty():
        return None

    # Filter out SLIM if no pinflate values selected
    if not selected_pinflate:
        slim_variants = ["slim_gsgp", "slim"]
        all_runs_df = all_runs_df.filter(~pl.col("params.variant").is_in(slim_variants))

    # Filter by selected metrics if specified
    if selected_metrics:
        metric_cols = [f"metrics.{m}" for m in selected_metrics if f"metrics.{m}" in all_runs_df.columns]
        if metric_cols:
            # Keep only selected metric columns plus essential columns
            essential_cols = [col for col in all_runs_df.columns if not col.startswith("metrics.") or col in metric_cols]
            all_runs_df = all_runs_df.select(essential_cols)

    return all_runs_df