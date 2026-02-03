"""
Metrics data fetching logic for metrics overview.

This module handles fetching metrics data from MLflow experiments.

metrics_fetcher.py
├── fetch_metrics_data()  # Fetch metrics data from MLflow
└── Metrics data retrieval logic
"""

import polars as pl

from statflow.functional.dataframes.data_processing import fetch_experiment_data


def fetch_metrics_data() -> pl.DataFrame:
    """Fetch metrics data for selected experiments and datasets.

    Returns:
        DataFrame with metrics information.
    """
    return fetch_experiment_data('metrics.')