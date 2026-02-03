"""
Parameter data fetching logic for parameter exploration.

This module handles fetching parameter data from MLflow experiments.

parameter_fetcher.py
├── fetch_parameter_data()  # Fetch parameter data from MLflow
└── Parameter data retrieval logic
"""

import polars as pl

from statflow.functional.dataframes.data_processing import fetch_experiment_data


def fetch_parameter_data() -> pl.DataFrame:
    """Fetch parameter data for selected experiments and datasets.

    Returns:
        DataFrame with parameter information.
    """
    return fetch_experiment_data('params.')