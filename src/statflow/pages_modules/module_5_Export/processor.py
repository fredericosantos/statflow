"""
Processor for data export functionality.

This module handles data fetching, processing, and export operations for bulk downloads.

processor.py
├── prepare_export_data()     # Fetches and prepares data for export based on filters.
├── create_export_files()     # Generates CSV, LaTeX, Markdown files.
└── generate_zip_archive()    # Creates ZIP archive with all export files.
"""

from typing import List, Optional
import pandas as pd
import streamlit as st

from statflow.utils.mlflow_client import fetch_all_datasets_parallel


def prepare_export_data(
    selected_mpf: List, selected_beta: List, selected_pinflate: List, selected_datasets: List[str]
) -> Optional[pd.DataFrame]:
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


def create_export_files(all_runs_df: pd.DataFrame, filename_prefix: str) -> dict:
    """Create export files (CSV, LaTeX, Markdown).

    Args:
        all_runs_df: DataFrame with runs.
        filename_prefix: Prefix for filenames.

    Returns:
        Dict of file contents.
    """
    from statflow.utils.export import export_table_to_csv, export_table_to_latex, export_table_to_markdown

    # Implement file creation logic
    files = {}
    # Example: files['csv'] = export_table_to_csv(all_runs_df, filename_prefix)
    return files


def generate_zip_archive(files: dict, zip_name: str) -> bytes:
    """Generate ZIP archive from files.

    Args:
        files: Dict of file contents.
        zip_name: Name of ZIP file.

    Returns:
        ZIP file bytes.
    """
    import zipfile
    import io

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for name, content in files.items():
            zip_file.writestr(name, content)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()