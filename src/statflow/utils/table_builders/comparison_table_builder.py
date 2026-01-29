"""
Comparison table builder utilities.

This module provides functions for constructing cross-dataset comparison tables.

comparison_table_builder.py
├── _calculate_median_std()     # Helper for median ± std formatting
├── _calculate_nodes_median_std() # Helper for nodes median ± std
└── build_comparison_table()    # Main function for cross-dataset comparisons

Usage:
    from statflow.utils.table_builders.comparison_table_builder import build_comparison_table
"""

import pandas as pd
import streamlit as st
from statflow.utils.data_processing import create_group_label, get_sort_key


def _calculate_median_std(values: pd.Series) -> str:
    """Calculate median ± std formatted string."""
    median = values.median()
    std = values.std()
    return f"{median:.4f} ± {std:.4f}"


def _calculate_nodes_median_std(values: pd.Series) -> str:
    """Calculate median ± std for nodes formatted string."""
    median = values.median()
    std = values.std()
    return f"{median:.1f} ± {std:.1f}"


@st.cache_data(show_spinner=False)
def build_comparison_table(all_runs_df: pd.DataFrame) -> pd.DataFrame:
    """Build cross-dataset comparison table with multi-level headers.

    Args:
        all_runs_df: DataFrame containing runs from all datasets.

    Returns:
        DataFrame with datasets as rows and configs as columns, showing median ± std.
    """
    if all_runs_df.empty:
        return pd.DataFrame()

    # Add grouping labels and sort keys
    all_runs_df = all_runs_df.copy()
    all_runs_df["config_group"] = all_runs_df.apply(create_group_label, axis=1)
    all_runs_df["sort_key"] = all_runs_df.apply(get_sort_key, axis=1)

    # Get unique configurations sorted
    config_order = (
        all_runs_df.groupby("config_group")["sort_key"]
        .first()
        .sort_values()
        .index.tolist()
    )

    # Build table data
    table_data = []

    available_datasets = st.session_state['available_datasets'] if 'available_datasets' in st.session_state else []
    for dataset in available_datasets:
        dataset_runs = all_runs_df[all_runs_df["dataset_name"] == dataset]

        if dataset_runs.empty:
            continue

        # Row 1: RMSE (best_test_fitness)
        rmse_row = {"Dataset": dataset, "Metric": "RMSE"}
        # Row 2: Tree Size (best_n_nodes)
        nodes_row = {"Dataset": dataset, "Metric": "Tree Size"}

        for config in config_order:
            config_runs = dataset_runs[dataset_runs["config_group"] == config]

            if config_runs.empty:
                rmse_row[config] = "—"
                nodes_row[config] = "—"
                continue

            # RMSE statistics
            rmse_values = config_runs["metrics.best_test_fitness"]
            rmse_row[config] = _calculate_median_std(rmse_values)

            # Tree size statistics
            nodes_values = config_runs["metrics.best_n_nodes"]
            nodes_row[config] = _calculate_nodes_median_std(nodes_values)

        table_data.append(rmse_row)
        table_data.append(nodes_row)

    if not table_data:
        return pd.DataFrame()

    # Create DataFrame
    table_df = pd.DataFrame(table_data)

    # Set multi-level columns
    table_df = table_df.set_index(["Dataset", "Metric"])

    return table_df