"""
Nodes table builder utilities.

This module provides functions for constructing tree size comparison tables.

nodes_table_builder.py
└── build_nodes_table()    # Main function for nodes tables

Usage:
    from statflow.utils.table_builders.nodes_table_builder import build_nodes_table
"""

import pandas as pd
import streamlit as st

from statflow.utils.data_processing import create_group_label, get_sort_key


@st.cache_data(show_spinner=False)
def build_nodes_table(all_runs_df: pd.DataFrame) -> pd.DataFrame:
    """Build tree size comparison table across datasets.

    Args:
        all_runs_df: DataFrame containing runs from all datasets.

    Returns:
        DataFrame with datasets as rows and configs as columns, showing median ± std for tree size.
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

        row = {"Dataset": dataset}

        for config in config_order:
            config_runs = dataset_runs[dataset_runs["config_group"] == config]

            if config_runs.empty:
                row[config] = "—"
                continue

            # Tree size statistics
            nodes_values = config_runs["metrics.best_n_nodes"]
            nodes_median = nodes_values.median()
            nodes_std = nodes_values.std()
            row[config] = f"{nodes_median:.1f} ± {nodes_std:.1f}"

        table_data.append(row)

    if not table_data:
        return pd.DataFrame()

    # Create DataFrame
    table_df = pd.DataFrame(table_data)

    # Reorder columns: Dataset, then configs in order
    col_order = ["Dataset"] + [c for c in config_order if c in table_df.columns]
    table_df = table_df[col_order]

    return table_df