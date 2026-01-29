"""
Single Dataset Analysis page.

This page provides detailed analysis for individual datasets including
boxplots, Pareto fronts, and statistical summaries.

1_🔬_Single_Dataset.py
├── Dataset selection
├── Boxplot visualization
├── Pareto front plot
├── Statistics table
├── Raw data viewer
├── Download options
└── Filter controls

Usage:
    Accessed via navigation from Home.py
"""

import streamlit as st
import pandas as pd

from statflow.config import (
    initialize_session_state, save_session_state_to_config, setup_sidebar
)
from statflow.utils.mlflow_client import get_filtered_runs
from statflow.components.filters import (
    render_dataset_selector, render_mpf_filter, render_beta_filter,
    render_pinflate_filter, render_display_options, render_graph_config,
    render_filter_summary, render_global_filters
)
from statflow.components.graphs import (
    render_boxplot, render_pareto_front, render_statistics_table, render_raw_data_table
)
from statflow.components.tables import render_table_with_downloads
from statflow.utils.export import (
    export_table_to_csv, export_table_to_latex, export_table_to_markdown
)
from statflow.pages_modules.module_3_Single_Dataset.processor import (
    fetch_and_process_single_dataset, extract_filter_values
)


st.set_page_config(
    page_title=f"Single Dataset Analysis - {st.session_state.get('app_name', 'Experiment Analysis')}",
    page_icon="🔬",
    layout="wide",
)

# Initialize session state
initialize_session_state()

# Setup sidebar
setup_sidebar()


def main():
    st.title("🔬 Single Dataset Analysis")
    st.markdown("Detailed analysis of individual datasets")

    # Sidebar for filters
    with st.sidebar:
        # Dataset selection
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")

    # Update session state
    st.session_state.selected_dataset = dataset_name

    # Main content area
    if not dataset_name:
        st.info("Please select a dataset to analyze.")
        return

    # Fetch and process data for selected dataset
    with st.spinner(f"Fetching runs for dataset: {dataset_name}..."):
        runs_df = fetch_and_process_single_dataset(dataset_name)

        if runs_df is None:
            st.error(f"No data found for dataset: {dataset_name}")
            return

    # Extract available values for filters
    available_mpf, available_beta, available_pinflate = extract_filter_values(runs_df)

    # Continue with sidebar filters
    with st.sidebar:
        # Global filters
        _, selected_mpf_values, selected_beta_values, selected_pinflate_values = render_global_filters(
            available_mpf, available_beta, available_pinflate
        )

        # Display options
        show_mean, use_custom_colors = render_display_options()

    # Apply filtering to runs_df
    filtered_runs_df = runs_df.copy()

    # Filter ARC by MPF
    if available_mpf and selected_mpf_values is not None:
        # Keep non-ARC runs + ARC runs with selected MPF values
        mask = (filtered_runs_df["params.variant"] != "arc") | (
            filtered_runs_df["params.mutation_pool_factor"].astype(str).isin(selected_mpf_values)
        )
        filtered_runs_df = filtered_runs_df[mask]
    elif available_mpf and selected_mpf_values is None:
        # If ARC exists but no MPF selected, exclude all ARC runs
        filtered_runs_df = filtered_runs_df[filtered_runs_df["params.variant"] != "arc"]

    # Filter ARC by Beta
    if available_beta and selected_beta_values is not None:
        mask = (filtered_runs_df["params.variant"] != "arc") | (
            filtered_runs_df["params.arc_beta"].astype(str).isin(selected_beta_values)
        )
        filtered_runs_df = filtered_runs_df[mask]
    elif available_beta and selected_beta_values is None:
        filtered_runs_df = filtered_runs_df[filtered_runs_df["params.variant"] != "arc"]

    # Filter SLIM-GSGP by P_inflate
    if available_pinflate and selected_pinflate_values is not None:
        mask = (filtered_runs_df["params.variant"] != "slim_gsgp") | (
            filtered_runs_df["params.arc_beta"].astype(str).isin(selected_pinflate_values)
        )
        filtered_runs_df = filtered_runs_df[mask]
    elif available_pinflate and selected_pinflate_values is None:
        filtered_runs_df = filtered_runs_df[filtered_runs_df["params.variant"] != "slim_gsgp"]

    # Update session state
    st.session_state.selected_mpf_values = selected_mpf_values
    st.session_state.selected_beta_values = selected_beta_values
    st.session_state.selected_pinflate_values = selected_pinflate_values
    st.session_state.show_mean = show_mean
    st.session_state.use_custom_colors = use_custom_colors

    # Main content tabs
    tab_plots, tab_data = st.tabs(["📊 Plots", "📋 Data"])

    with tab_plots:
        # Graph configuration (render first to get config values)
        graph_width, graph_height, points_display, show_error_bars = render_graph_config()

        # Update graph config session state
        st.session_state.graph_width = graph_width
        st.session_state.graph_height = graph_height
        st.session_state.points_display = points_display
        st.session_state.show_error_bars = show_error_bars

        # Boxplot and Pareto front side by side
        col1, col2 = st.columns(2)
        
        with col1:
            render_boxplot(
                filtered_runs_df,
                show_mean=show_mean,
                points_display=points_display,
                use_custom_colors=use_custom_colors,
                custom_colors=st.session_state.get('custom_colors'),
                width=graph_width,
                height=graph_height
            )

        with col2:
            render_pareto_front(
                filtered_runs_df,
                show_error_bars=show_error_bars,
                use_custom_colors=use_custom_colors,
                custom_colors=st.session_state.get('custom_colors'),
                custom_symbols=st.session_state.get('custom_symbols'),
                width=graph_width,
                height=graph_height
            )

    with tab_data:
        # Create statistics table
        stats_df = render_statistics_table(filtered_runs_df, show_mean=show_mean, return_df=True)

        # Statistics table
        if stats_df is not None:
            render_table_with_downloads(
                stats_df,
                "Statistics Summary",
                f"{dataset_name}_statistics",
                description=""
            )

        st.markdown("---")

        # Raw data
        with st.expander("🔍 Raw Run Data", expanded=False):
            selected_params = st.session_state.get('selected_params', [])
            render_raw_data_table(filtered_runs_df, selected_params)

    # Save configuration changes
    save_session_state_to_config()


if __name__ == "__main__":
    main()