"""
Processor for Parameters page.

This module handles data processing for parameter exploration and configuration.

module_1_Parameters/processor.py
├── fetch_parameter_data()    # Fetch parameter data from MLflow
└── prepare_parameter_summary() # Prepare summary statistics for parameters
"""

import polars as pl
import streamlit as st

from statflow.utils.data_processing import fetch_experiment_data


def fetch_parameter_data() -> pl.DataFrame:
    """Fetch parameter data for selected experiments and datasets.

    Returns:
        DataFrame with parameter information.
    """
    return fetch_experiment_data('params.')


def prepare_parameter_summary(param_df: pl.DataFrame) -> pl.DataFrame:
    """Prepare summary statistics for parameters.

    Args:
        param_df: DataFrame with parameter data.

    Returns:
        DataFrame with parameter summary statistics.
    """
    if param_df.is_empty():
        return pl.DataFrame()

    summary_data = []

    # Exclude dataset_name from parameter analysis
    param_cols = [col for col in param_df.columns if col != 'dataset_name']

    for param in param_cols:
        if param in param_df.columns:
            unique_values = param_df.select(pl.col(param).n_unique()).item()
            mode_series = param_df.select(pl.col(param).mode())
            most_common = mode_series.select(pl.first(pl.col(param))).item() if not mode_series.is_empty() else None
            missing_count = param_df.select(pl.col(param).is_null().sum()).item()

            summary_data.append({
                'Parameter': param,
                'Unique Values': unique_values,
                'Most Common': most_common,
                'Missing Values': missing_count,
                'Total Records': param_df.height
            })

    return pl.DataFrame(summary_data)