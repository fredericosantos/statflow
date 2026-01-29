"""
Data processing utilities for experiment results.

This module contains functions for transforming, labeling, and processing
experiment data from MLflow runs.

data_processing.py
├── Configuration labeling (create_group_label, get_pareto_group_label)
├── Sorting utilities (get_sort_key)
├── Pareto front calculation (calculate_pareto_front)
├── Variant extraction (get_variant_from_config)
└── Data validation and cleaning

Usage:
    from statflow.utils.data_processing import (
        create_group_label, get_sort_key, calculate_pareto_front
    )
"""

import pandas as pd
from typing import Tuple
import polars as pl
import streamlit as st

from statflow.utils.mlflow_client import fetch_all_datasets_parallel


def create_group_label(row: pd.Series) -> str:
    """Create a label for grouping based on configuration parameters.

    Args:
        row: DataFrame row containing parameter values.

    Returns:
        String label representing the configuration group.
    """
    variant = row.get("params.variant", "N/A")

    # Handle different variants with specific formatting
    if variant == "gsgp":
        use_oms = row.get("params.use_oms", "False")
        if use_oms == "True":
            return "GSGP-OMS"
        else:
            return "GSGP-std"

    elif variant == "slim_gsgp" or variant == "slim":
        p_inflate = row.get("params.arc_beta", "N/A")
        return f"SLIM"

    elif variant == "arc":
        beta = row.get("params.arc_beta", "N/A")
        # Show beta with the beta symbol on the boxplot x-axis
        return f"ARC (β = {beta})"

    else:
        # Fallback for unknown variants
        return variant


def get_sort_key(row: pd.Series) -> Tuple:
    """Generate a sort key for ordering configurations.

    Order: Variant -> arc_beta (low to high) -> MPF

    Args:
        row: DataFrame row containing parameter values.

    Returns:
        Tuple for sorting.
    """
    variant = row.get("params.variant", "zzz")  # Unknown variants go last

    # Variant order: arc, slim_gsgp, gsgp (reversed to A-Z)
    variant_order = {"arc": 0, "slim_gsgp": 1, "gsgp": 2}
    variant_rank = variant_order.get(variant, 999)

    # Get beta and MPF (parse as floats/ints)
    beta = float(row.get("params.arc_beta", "0"))
    mpf = int(row.get("params.mutation_pool_factor", "0"))

    # For GSGP, use use_oms as secondary sort
    if variant == "gsgp":
        # Put GSGP-OMS before GSGP-std: OMS should have lower sort key
        use_oms = 0 if row.get("params.use_oms", "False") == "True" else 1
        return (variant_rank, use_oms, 0, 0)

    # For SLIM-GSGP, use p_inflate
    elif variant == "slim_gsgp":
        p_inflate = float(row.get("params.arc_beta", "0.5"))
        return (variant_rank, p_inflate, 0, 0)

    # For ARC, sort by beta (low to high), then MPF
    else:
        return (variant_rank, beta, mpf, 0)


def calculate_pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Calculate the Pareto front for a set of points (minimizing both objectives).

    Args:
        df: DataFrame containing the points.
        x_col: Column name for x-axis (to minimize).
        y_col: Column name for y-axis (to minimize).

    Returns:
        DataFrame containing only Pareto-optimal points.
    """
    pareto_points = []

    for idx, row in df.iterrows():
        is_dominated = False
        for other_idx, other_row in df.iterrows():
            # Skip comparing with itself
            if idx == other_idx:
                continue
            # Check if other_row dominates current row (both <= and at least one <)
            if (other_row[x_col] <= row[x_col] and other_row[y_col] < row[y_col]) or (
                other_row[x_col] < row[x_col] and other_row[y_col] <= row[y_col]
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_points.append(idx)

    return df.loc[pareto_points].sort_values(x_col)


def get_pareto_group_label(row: pd.Series) -> str:
    """Create a label for Pareto front grouping.

    Args:
        row: DataFrame row containing parameter values.

    Returns:
        String label for Pareto grouping.
    """
    variant = row.get("params.variant", "N/A")

    if variant == "gsgp":
        use_oms = row.get("params.use_oms", "False")
        if use_oms == "True":
            return "GSGP-OMS"
        else:
            return "GSGP-std"
    elif variant == "slim_gsgp":
        return "SLIM"
    elif variant == "arc":
        beta = row.get("params.arc_beta", "N/A")
        return f"ARC $\\beta={beta}$"
    else:
        return variant


def get_variant_from_config(config_label: str) -> str:
    """Extract variant name from config label.

    Args:
        config_label: Configuration label string.

    Returns:
        Variant name (ARC, SLIM, GSGP, or GSGP-OMS).
    """
    if config_label.startswith("ARC"):
        return "ARC"
    elif config_label.startswith("SLIM"):
        return "SLIM"
    elif config_label.startswith("GSGP-OMS"):
        return "GSGP-OMS"
    elif config_label.startswith("GSGP"):
        return "GSGP"
    else:
        return "Other"


def get_dataset_info(datasets_path: str, dataset_name: str) -> Tuple[int, int]:
    """Get number of samples and features for a dataset.

    Args:
        datasets_path: Path to the datasets directory.
        dataset_name: Name of the dataset (without .csv extension).

    Returns:
        Tuple of (num_samples, num_features).
        num_features = num_columns - 1 (last column is target).
    """
    import os
    
    csv_path = os.path.join(datasets_path, f"{dataset_name}.csv")
    
    if not os.path.exists(csv_path):
        return 0, 0
    
    try:
        df = pd.read_csv(csv_path)
        num_samples = df.shape[0]
        num_features = df.shape[1] - 1  # Last column is target
        return num_samples, num_features
    except Exception:
        return 0, 0


@st.cache_data(ttl=600, show_spinner=False)
def fetch_experiment_data(column_prefix: str, clean_prefix: bool = True) -> pl.DataFrame:
    """Fetch experiment data for selected experiments and datasets, extracting specific column types.

    Args:
        column_prefix: Prefix to filter columns (e.g., 'params.', 'metrics.').
        clean_prefix: Whether to remove the prefix from column names.

    Returns:
        DataFrame with the requested data type.
    """
    if not st.session_state.get('selected_experiments') or not st.session_state.get('selected_datasets'):
        return pl.DataFrame()

    # Get filtered runs
    all_runs_df = fetch_all_datasets_parallel(
        selected_datasets=tuple(st.session_state['selected_datasets']),
        selected_experiments=tuple(st.session_state['selected_experiments'])
    )

    if all_runs_df.is_empty():
        return pl.DataFrame()

    # Extract columns with the specified prefix
    cols = [col for col in all_runs_df.columns if col.startswith(column_prefix)]
    if not cols:
        return pl.DataFrame()

    # Create DataFrame with selected columns plus dataset_name
    df = all_runs_df.select(cols + ['dataset_name'])

    # Clean column names if requested
    if clean_prefix:
        new_names = {col: col.replace(column_prefix, '') for col in cols}
        df = df.rename(new_names)

    return df