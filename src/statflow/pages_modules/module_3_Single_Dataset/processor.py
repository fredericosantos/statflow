"""
Processor for single dataset analysis.

This module handles data fetching, processing, and analysis for individual datasets.

processor.py
├── fetch_and_process_single_dataset()  # Fetches and processes data for a single dataset.
├── extract_filter_values()             # Extracts available values for filters from runs data.
└── prepare_dataset_summary()           # Prepares summary statistics for the dataset.
"""

from typing import Tuple, Optional, List
import polars as pl
import streamlit as st

from statflow.utils.mlflow_client import get_filtered_runs


def fetch_and_process_single_dataset(dataset_name: str) -> Optional[pl.DataFrame]:
    """Fetch and process data for a single dataset.

    Args:
        dataset_name: Name of the dataset to fetch.

    Returns:
        Processed DataFrame with runs data, or None if no data.
    """
    runs_df = get_filtered_runs(dataset_name)

    if runs_df.is_empty():
        return None

    runs_df = runs_df.with_columns(pl.lit(dataset_name).alias("dataset_name"))

    return runs_df


def extract_filter_values(runs_df: pl.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Extract available values for filters from runs data.

    Args:
        runs_df: DataFrame with runs data.

    Returns:
        Tuple of (available_mpf, available_beta, available_pinflate).
    """
    available_mpf = sorted(runs_df.select(pl.col("params.mutation_pool_factor").drop_nulls().unique().cast(pl.Utf8)).to_series().to_list())
    available_beta = sorted(runs_df.select(pl.col("params.arc_beta").drop_nulls().unique().cast(pl.Utf8)).to_series().to_list())
    available_pinflate = sorted(runs_df.filter(pl.col("params.variant") == "slim_gsgp").select(pl.col("params.arc_beta").drop_nulls().unique().cast(pl.Utf8)).to_series().to_list())

    return available_mpf, available_beta, available_pinflate


def prepare_dataset_summary(runs_df: pl.DataFrame) -> pl.DataFrame:
    """Prepare summary statistics for the dataset.

    Args:
        runs_df: DataFrame with runs data.

    Returns:
        DataFrame with summary statistics.
    """
    # Implement summary logic here
    # For now, return empty or basic summary
    return pl.DataFrame()