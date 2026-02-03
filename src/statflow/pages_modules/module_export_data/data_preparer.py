"""
Data preparation logic for export functionality.

This module handles fetching and preparing data for bulk export operations.

data_preparer.py
├── prepare_export_data()  # Fetches and prepares data for export based on filters.
└── Data preparation and filtering logic
"""

import pandas as pd

from statflow.functional.mlflow.mlflow_client import fetch_all_datasets_parallel


def prepare_export_data(
    selected_mpf: list, selected_beta: list, selected_pinflate: list, selected_datasets: list[str]
) -> pd.DataFrame | None:
    """Prepare data for export based on selected filters.

    Args:
        selected_mpf: Selected MPF values.
        selected_beta: Selected beta values.
        selected_pinflate: Selected pinflate values.
        selected_datasets: Selected datasets.

    Returns:
        Prepared DataFrame or None if no data.
    """
    all_runs_df = fetch_all_datasets_parallel(
        tuple(selected_mpf), tuple(selected_beta), tuple(selected_pinflate), selected_datasets
    )

    if all_runs_df.empty:
        return None

    # Filter SLIM if needed
    if not selected_pinflate:
        slim_variants = ["slim_gsgp", "slim"]
        all_runs_df = all_runs_df[~all_runs_df["params.variant"].isin(slim_variants)]

    return all_runs_df