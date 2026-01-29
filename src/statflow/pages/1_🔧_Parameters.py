"""
Parameters page for the Statflow application.

This page allows users to explore, filter, and configure experiment parameters.

1_🔧_Parameters.py
├── Parameter exploration and visualization
├── Parameter filtering and selection
└── Session state integration for parameter choices

Usage:
    Streamlit page for parameter management.
"""

import streamlit as st
import pandas as pd
import polars as pl

from statflow.config import (
    initialize_session_state, save_session_state_to_config, setup_sidebar
)
from statflow.utils.mlflow_client import get_filtered_runs
from statflow.components.filters import (
    render_dataset_selector, render_mpf_filter, render_beta_filter,
    render_pinflate_filter, render_display_options, render_graph_config,
    render_filter_summary, render_global_filters
)
from statflow.components.tables import render_table_with_downloads
from statflow.components.graphs import render_parameter_distributions
from statflow.pages_modules.module_1_Parameters.processor import (
    fetch_parameter_data, prepare_parameter_summary
)


def main():
    # Setup sidebar with server status
    setup_sidebar()

    st.title(":material/tune: Parameters")
    st.markdown("Explore and configure experiment parameters for analysis.")

    # Check if experiments and datasets are selected
    if not st.session_state.get('selected_experiments'):
        st.warning("Please select experiments on the Home page first.")
        return

    if not st.session_state.get('selected_datasets'):
        st.warning("Please select datasets on the Home page first.")
        return

    # Fetch parameter data
    with st.spinner("Loading parameter data..."):
        param_df = fetch_parameter_data()

    if param_df.empty:
        st.error("No parameter data found for the selected experiments and datasets.")
        return
    # Parameter Filters
    with st.expander("Parameter Filters", expanded=False, icon=":material/filter_list:"):
        st.markdown("Filter parameters to focus on specific values:")

        # Use Polars for filtering
        filtered_df = param_df.clone()

        param_cols = [col for col in param_df.columns if col != 'dataset_name']

        for param in param_cols[:5]:  # Limit to 5 to avoid clutter
            if param_df[param].dtype in [pl.Int64, pl.Float64]:
                min_val = float(param_df[param].min())
                max_val = float(param_df[param].max())
                selected_range = st.slider(
                    f"Filter {param}",
                    min_val, max_val, (min_val, max_val),
                    key=f"filter_{param}"
                )
                filtered_df = filtered_df.filter(
                    (pl.col(param) >= selected_range[0]) & (pl.col(param) <= selected_range[1])
                )
            elif param_df[param].n_unique() < 20:  # Categorical with few values
                unique_vals = sorted(param_df[param].drop_nulls().unique().to_list())
                selected_vals = st.multiselect(
                    f"Filter {param}",
                    unique_vals, unique_vals,
                    key=f"filter_{param}"
                )
                if selected_vals:
                    filtered_df = filtered_df.filter(pl.col(param).is_in(selected_vals))

        # Update param_df with filtered data
        param_df = filtered_df

    # Check for empty filtered data
    if param_df.is_empty():
        st.warning("⚠️ No data matches the current filter criteria. Please adjust your filters to see parameter data.")
        return

    # Parameter Summary
    with st.expander("Parameter Summary", expanded=True, icon=":material/summarize:"):
        summary_df = prepare_parameter_summary(param_df)
        if not summary_df.empty:
            render_table_with_downloads(summary_df, "Parameter Summary")
        else:
            st.info("No parameter summary available.")

    # Parameter Distributions
    with st.expander("Parameter Distributions", expanded=False, icon=":material/bar_chart:"):
        render_parameter_distributions(param_df)

    # Parameter Filtering
    with st.expander("Parameter Filtering", expanded=False, icon=":material/filter_list:"):
        st.markdown("Configure parameter filters for analysis:")

        # Use existing filter components
        render_global_filters()

        # Additional parameter-specific filters can be added here

    # Save configuration
    if st.button("Save Parameter Configuration", icon=":material/save:"):
        save_session_state_to_config()
        st.success("Parameter configuration saved!")


if __name__ == "__main__":
    main()