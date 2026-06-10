"""
Data processing utilities for experiment results.

This module contains functions for transforming, labeling, and processing
experiment data from the cached (provider-agnostic) runs.

data_processing.py
├── calculate_pareto_front()        # Calculate Pareto front for points
├── fetch_experiment_data()         # Fetch filtered experiment data from cache
└── apply_metric_filters()          # Apply saved metric range/NaN filters
"""

import polars as pl
import streamlit as st

from statflow.loggers.runs_cache import RunsCache


def calculate_pareto_front(df: pl.DataFrame, x_col: str, y_col: str) -> pl.DataFrame:
    """Calculate the Pareto front for a set of points (minimizing both objectives).

    Efficient Polars-based implementation.
    """
    if df.is_empty():
        return df

    sorted_df = df.sort([x_col, y_col])
    rows = sorted_df.to_dicts()
    pareto_indices = []
    min_y = float("inf")

    for i, row in enumerate(rows):
        if row[y_col] < min_y:
            min_y = row[y_col]
            pareto_indices.append(i)

    return sorted_df[pareto_indices].sort(x_col)


def fetch_experiment_data(column_prefix: str, clean_prefix: bool = True) -> pl.DataFrame:
    """Fetch experiment data for selected experiments and datasets from cache.

    Args:
        column_prefix: Prefix to filter columns (e.g., 'metrics.', 'params.').
        clean_prefix: If True, remove the prefix from column names.

    Returns:
        Polars DataFrame with filtered and renamed columns.
    """
    selected_datasets = st.session_state["selected_datasets"]
    dataset_param = st.session_state["dataset_param"]

    if not selected_datasets:
        return pl.DataFrame()

    # Get filtered runs from cache
    all_runs_df = RunsCache.filter_by_datasets(dataset_param, selected_datasets)

    if all_runs_df.is_empty():
        return pl.DataFrame()

    # Extract columns with the specified prefix
    cols = [col for col in all_runs_df.columns if col.startswith(column_prefix)]
    if not cols:
        return pl.DataFrame()

    # Build list of columns to select
    select_cols = cols.copy()

    # Always include run_id for proper joins
    if "run_id" in all_runs_df.columns and "run_id" not in select_cols:
        select_cols.append("run_id")

    # Determine dataset column name - only add if not already in cols
    dataset_col = f"params.{dataset_param}" if dataset_param else None
    if dataset_col and dataset_col in all_runs_df.columns and dataset_col not in select_cols:
        select_cols.append(dataset_col)

    df = all_runs_df.select(select_cols)

    # Rename dataset column to standard name first
    if dataset_col and dataset_col in df.columns:
        df = df.rename({dataset_col: "dataset_name"})

    # Clean column names if requested
    if clean_prefix:
        # Build rename map, excluding dataset_col (already renamed above) and run_id
        new_names = {col: col.replace(column_prefix, "") for col in cols if col != dataset_col}

        # Only rename columns that exist in df
        new_names = {k: v for k, v in new_names.items() if k in df.columns}

        if new_names:
            df = df.rename(new_names)

    return df


def apply_metric_filters(df: pl.DataFrame) -> pl.DataFrame:
    """Apply saved metric filters from session state to a DataFrame.

    This function reads active filters, their range values, and NaN preferences
    from st.session_state and applies them sequentially to the input DataFrame.
    """
    active_filters = st.session_state.get("active_metric_filters", [])
    filter_values = st.session_state.get("metric_filter_values", {})
    filter_nans = st.session_state.get("metric_filter_nans", {})

    if not active_filters or df.is_empty():
        return df

    filtered_df = df
    for metric in active_filters:
        if metric not in filtered_df.columns:
            continue

        selected_range = filter_values.get(metric)
        include_nans = filter_nans.get(metric, False)

        if selected_range is not None:
            # Range filter naturally excludes nulls/NaNs (float comparisons)
            cond = (pl.col(metric) >= selected_range[0]) & (pl.col(metric) <= selected_range[1])

            # If user wants NaNs/nulls, we explicitly OR them back in
            if include_nans:
                cond = cond | pl.col(metric).is_null() | pl.col(metric).is_nan()

            filtered_df = filtered_df.filter(cond)

    return filtered_df
