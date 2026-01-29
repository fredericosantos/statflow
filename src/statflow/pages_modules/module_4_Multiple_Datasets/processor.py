"""
Processor for multiple datasets comparison.

This module handles data fetching, processing, and analysis for comparing multiple datasets.

processor.py
├── fetch_and_process_multiple_datasets()  # Fetches and processes data for multiple datasets.
└── prepare_comparison_tables()            # Prepares tables for comparison display.
"""

from typing import List, Optional, Tuple
import polars as pl
import streamlit as st

from statflow.utils.mlflow_client import fetch_all_datasets_parallel


def fetch_and_process_multiple_datasets(
    selected_mpf: Tuple, selected_beta: Tuple, selected_pinflate: Tuple, selected_datasets: List[str], selected_metrics: Optional[List[str]] = None
) -> Optional[pl.DataFrame]:
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


def prepare_comparison_tables(all_runs_df: pl.DataFrame) -> Tuple[pl.DataFrame, dict, pl.DataFrame]:
    """Prepare tables for comparison display.

    Args:
        all_runs_df: DataFrame with all runs.

    Returns:
        Tuple of (rmse_df, significance_info, nodes_df).
    """
    from statflow.utils.table_builders.rmse_table_builder import build_rmse_table
    from statflow.utils.table_builders.nodes_table_builder import build_nodes_table

    rmse_df, significance_info = build_rmse_table(all_runs_df)
    nodes_df = build_nodes_table(all_runs_df)

    return rmse_df, significance_info, nodes_df