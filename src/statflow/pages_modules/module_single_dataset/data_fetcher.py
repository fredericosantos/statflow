"""
Data fetching logic for single dataset analysis.

This module handles fetching and initial processing of data for individual datasets.

data_fetcher.py
├── fetch_and_process_single_dataset()  # Fetches and processes data for a single dataset.
└── Data retrieval and initial processing logic
"""

import polars as pl

from statflow.functional.mlflow.mlflow_client import get_filtered_runs


def fetch_and_process_single_dataset(dataset_name: str) -> pl.DataFrame | None:
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