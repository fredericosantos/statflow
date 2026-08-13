"""
Data processing utilities for experiment results.

This module contains functions for transforming, labeling, and processing
experiment data from the cached (provider-agnostic) runs.

data_processing.py
├── calculate_pareto_front()        # Calculate Pareto front for points
├── fetch_experiment_data()         # Fetch filtered experiment data from cache
├── tags_from_values()              # Split comma-joined `tags` values into unique tags
├── available_tags()                # Distinct W&B tags across cached runs
├── add_tag_columns()               # Derive `tag:<name>` true/false columns
├── grouping_params()               # selected_params + selected tags (as `tag:<name>`)
└── apply_metric_filters()          # Apply saved metric range/NaN filters
"""

import polars as pl
import streamlit as st

from statflow.loggers.runs_cache import RunsCache

TAG_PARAM_PREFIX = "tag:"  # `tag:<name>` columns derived from selected W&B tags


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

    # Surface selected W&B tags as `tag:<name>` boolean params (no-op for metrics
    # data or when no tags are selected).
    df = add_tag_columns(df, st.session_state.get("selected_tags", []))

    return df


def tags_from_values(values: list) -> list[str]:
    """Split comma-joined `params.tags` values into a sorted list of unique tags."""
    tags: set[str] = set()
    for v in values:
        if v is None:
            continue
        tags.update(t for t in str(v).split(",") if t)
    return sorted(tags)


def available_tags() -> list[str]:
    """Distinct individual W&B tags across all cached runs."""
    return tags_from_values(RunsCache.get_param_values("tags"))


def add_tag_columns(df: pl.DataFrame, tags: list[str]) -> pl.DataFrame:
    """Add a `tag:<name>` "true"/"false" column per tag, by membership in `tags`.

    The `tags` column holds the comma-joined tag set per run (or null). Each selected
    tag becomes a binary categorical param so it slots into the existing
    `param=value` group-label machinery.
    """
    if not tags or "tags" not in df.columns:
        return df

    split = pl.col("tags").str.split(",")
    exprs = [
        pl.when(split.list.contains(t).fill_null(False))
        .then(pl.lit("true"))
        .otherwise(pl.lit("false"))
        .alias(f"{TAG_PARAM_PREFIX}{t}")
        for t in tags
    ]
    return df.with_columns(exprs)


def grouping_params() -> list[str]:
    """Params that define a group: explicit `selected_params` plus selected tags.

    Selected tags are appended as `tag:<name>` columns (created by `add_tag_columns`)
    so they behave exactly like any other selected parameter in grouping/comparison.
    """
    base = list(st.session_state["selected_params"])
    tags = st.session_state.get("selected_tags", [])
    return base + [f"{TAG_PARAM_PREFIX}{t}" for t in tags]


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
