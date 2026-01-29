"""
Processor for Metrics page.

This module handles data processing for metrics overview and selection.

module_2_Metrics/processor.py
├── fetch_metrics_data()    # Fetch metrics data from MLflow
└── prepare_metrics_summary() # Prepare summary statistics for metrics
"""

import polars as pl
import streamlit as st

from statflow.utils.data_processing import fetch_experiment_data


def fetch_metrics_data() -> pl.DataFrame:
    """Fetch metrics data for selected experiments and datasets.

    Returns:
        DataFrame with metrics information.
    """
    return fetch_experiment_data('metrics.')


def prepare_metrics_summary(metrics_df: pl.DataFrame) -> pl.DataFrame:
    """Prepare summary statistics for metrics.

    Args:
        metrics_df: DataFrame with metrics data.

    Returns:
        DataFrame with metrics summary statistics.
    """
    if metrics_df.is_empty():
        return pl.DataFrame()

    summary_data = []

    # Exclude dataset_name and experiment_name from metrics analysis
    metrics_cols = [col for col in metrics_df.columns if col not in ['dataset_name', 'experiment_name']]

    # Check if experiment_name exists for grouping
    has_experiment = 'experiment_name' in metrics_df.columns

    if has_experiment:
        # Group by experiment
        for exp in metrics_df.select('experiment_name').unique().to_series().to_list():
            exp_df = metrics_df.filter(pl.col('experiment_name') == exp)
            for metric in metrics_cols:
                if metric in exp_df.columns:
                    col_expr = pl.col(metric)
                    values = exp_df.select(col_expr).drop_nulls()
                    if not values.is_empty():
                        count = values.height
                        mean_val = values.select(col_expr.mean()).item()
                        median_val = values.select(col_expr.median()).item()
                        std_val = values.select(col_expr.std()).item()
                        min_val = values.select(col_expr.min()).item()
                        max_val = values.select(col_expr.max()).item()
                        missing_count = exp_df.select(col_expr.is_null().sum()).item()

                        summary_data.append({
                            'Experiment': exp,
                            'Metric': metric,
                            'Count': count,
                            'Mean': mean_val,
                            'Median': median_val,
                            'Std Dev': std_val,
                            'Min': min_val,
                            'Max': max_val,
                            'Missing Values': missing_count
                        })
    else:
        # No experiment grouping
        for metric in metrics_cols:
            if metric in metrics_df.columns:
                col_expr = pl.col(metric)
                values = metrics_df.select(col_expr).drop_nulls()
                if not values.is_empty():
                    count = values.height
                    mean_val = values.select(col_expr.mean()).item()
                    median_val = values.select(col_expr.median()).item()
                    std_val = values.select(col_expr.std()).item()
                    min_val = values.select(col_expr.min()).item()
                    max_val = values.select(col_expr.max()).item()
                    missing_count = metrics_df.select(col_expr.is_null().sum()).item()

                    summary_data.append({
                        'Metric': metric,
                        'Count': count,
                        'Mean': mean_val,
                        'Median': median_val,
                        'Std Dev': std_val,
                        'Min': min_val,
                        'Max': max_val,
                        'Missing Values': missing_count
                    })

    return pl.DataFrame(summary_data)