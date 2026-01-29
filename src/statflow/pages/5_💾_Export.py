"""
Data Export page.

This page provides bulk export functionality for downloading raw data,
ZIP archives, and formatted tables for research and reporting.

4_💾_Export.py
├── Bulk data export
├── ZIP archive creation
├── Format selection (CSV, LaTeX, Markdown)
├── Filtered data export
├── Batch processing
└── Export history

Usage:
    Accessed via navigation from Home.py
"""

import streamlit as st
import pandas as pd

from statflow.config import (
    initialize_session_state,
    save_session_state_to_config, setup_sidebar
)
from statflow.utils.mlflow_client import fetch_all_datasets_parallel
from statflow.utils.table_builders.rmse_table_builder import build_rmse_table
from statflow.utils.table_builders.statistical_table_builder import build_statistical_table
from statflow.utils.table_builders.nodes_table_builder import build_nodes_table
from statflow.components.downloads import render_download_section
from statflow.components.filters import render_dataset_selector, render_global_filters
from statflow.components.tables import render_table_with_downloads
from statflow.utils.export import (
    export_table_to_csv,
    export_table_to_latex,
    export_table_to_markdown,
)
from statflow.pages_modules.module_5_Export.processor import (
    prepare_export_data, create_export_files
)


st.set_page_config(
    page_title=f"Data Export - {st.session_state.get('app_name', 'Experiment Analysis')}",
    page_icon="💾",
    layout="wide",
)

# Initialize session state
initialize_session_state()


def main():
    st.title("💾 Data Export")

    # Navigation
    if st.button("← Back to Home", key="back_home"):
        st.switch_page("Home.py")

    # Sidebar for export configuration
    with st.sidebar:
        st.header("💾 Data Export")

        # Dataset selection
        st.markdown("### Dataset Selection")
        selected_datasets = render_dataset_selector(selection_mode="multi")
        st.markdown("---")

        # Global filters
        available_mpf = ["1", "2", "5", "10"]
        available_beta = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "1.0"]
        available_pinflate = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        
        _, selected_mpf_values, selected_beta_values, selected_pinflate_values = render_global_filters(
            available_mpf, available_beta, available_pinflate
        )

    # Convert to tuples
    selected_mpf_tuple = tuple(selected_mpf_values) if selected_mpf_values else None
    selected_beta_tuple = tuple(selected_beta_values) if selected_beta_values else None
    selected_pinflate_tuple = tuple(selected_pinflate_values) if selected_pinflate_values else None

    # Update session state
    st.session_state.selected_mpf_values = selected_mpf_tuple
    st.session_state.selected_beta_values = selected_beta_tuple
    st.session_state.selected_pinflate_values = selected_pinflate_tuple
    st.session_state.selected_datasets = selected_datasets

    st.markdown("---")

    # Check if datasets are selected
    if not selected_datasets:
        st.info("Please select at least one dataset to export.")
        return

    # Export type selection using tabs
    tab_raw, tab_comparison = st.tabs([
        "📦 Raw Fitness Data",
        "📊 Comparison Tables", 
    ])

    # Update session state (filters are always available now)
    st.session_state.selected_mpf_values = selected_mpf_tuple
    st.session_state.selected_beta_values = selected_beta_tuple
    st.session_state.selected_pinflate_values = selected_pinflate_tuple

    with tab_raw:
        st.markdown("""
        Export raw fitness data for all datasets as a ZIP archive containing CSV files.
        Each dataset will have its own CSV file with all runs and their fitness values.
        """)

        # ZIP download section
        render_download_section(
            pd.DataFrame(),  # Empty dataframe since ZIP is handled separately
            prefix="raw_fitness_data",
            include_zip=True,
            selected_mpf_values=selected_mpf_tuple,
            selected_beta_values=selected_beta_tuple,
            selected_pinflate_values=selected_pinflate_tuple,
        )

        # Preview data size
        with st.spinner("Calculating data size..."):
            sample_data = prepare_export_data(
                selected_mpf_tuple, selected_beta_tuple, selected_pinflate_tuple, selected_datasets
            )

            if sample_data is not None and not sample_data.empty:
                total_runs = len(sample_data)
                total_datasets = len(sample_data["dataset_name"].unique())

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Runs", f"{total_runs:,}")
                with col2:
                    st.metric("Datasets", total_datasets)
                with col3:
                    avg_runs_per_dataset = total_runs / total_datasets
                    st.metric("Avg Runs/Dataset", ".0f")
            else:
                st.warning("No data available with current filters.")

    with tab_comparison:
        st.markdown("""
        Export comparison tables across all datasets in multiple formats.
        Tables include RMSE comparisons and tree size analysis.
        """)

        # Fetch data for tables
        with st.spinner("Fetching data for comparison tables..."):
            all_runs_df = prepare_export_data(
                selected_mpf_tuple, selected_beta_tuple, selected_pinflate_tuple, selected_datasets
            )

            if all_runs_df is None or all_runs_df.empty:
                st.error("No data available for comparison tables.")
                return

        # RMSE Comparison Table
        rmse_df, _ = build_rmse_table(all_runs_df)
        render_table_with_downloads(
            rmse_df,
            "RMSE Comparison Table",
            "rmse_comparison"
        )

        # Tree Size Comparison Table
        nodes_df = build_nodes_table(all_runs_df)
        render_table_with_downloads(
            nodes_df,
            "Tree Size Comparison Table", 
            "tree_size_comparison"
        )

        # Statistical Significance Table
        datasets_path = st.session_state.get('datasets_path', 'datasets')
        stats_df = build_statistical_table(all_runs_df, datasets_path)
        render_table_with_downloads(
            stats_df,
            "Statistical Significance Table",
            "statistical_significance", 
            description="""
            Export statistical significance analysis comparing variants across datasets.
            Includes p-values from Mann-Whitney U tests with significance levels: *** (p<0.001), ** (p<0.01), * (p<0.05).
            """
        )

    # Save configuration changes
    save_session_state_to_config()


if __name__ == "__main__":
    main()
