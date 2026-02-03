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

from statflow.config import State
from statflow.pages_modules.shared.server_status import ServerStatusManager
from statflow.functional.mlflow.mlflow_client import get_filtered_runs
from statflow.components.filters import (
    render_dataset_selector, render_mpf_filter, render_beta_filter,
    render_pinflate_filter, render_display_options, render_graph_config,
    render_filter_summary, render_global_filters
)
from statflow.components.tables import render_table_with_downloads
from statflow.components.graphs import (
    render_parameter_distributions, render_animated_scatter_plot, render_network_graph
)
from statflow.pages_modules.module_get_started import (
    parameter_selection,
)


st.set_page_config(
    page_title=f"Parameters - {st.session_state['app_name']}",
    page_icon=":material/tune:",
    layout="wide",
)


def main():
    # Initialize session state
    State.initialize()

    # Setup sidebar with server status
    status_manager = ServerStatusManager()
    server_running = status_manager.display_sidebar()

    # Check if MLflow server is running
    if not server_running:
        st.error("MLflow server is not running. This page requires an active MLflow server connection.", icon=":material/power_off:")
        st.info("Please start your MLflow server and refresh this page.")
        st.markdown("""
        **To start MLflow server:**
        ```bash
        mlflow server --host 0.0.0.0 --port 5000
        ```
        """)
        return

    st.title(":material/tune: Parameters")
    st.markdown("Explore and configure experiment parameters for analysis.")

    # Help section
    with st.expander("ℹ️ How to use this page", expanded=False):
        st.markdown("""
        **Parameters Page Help:**

        1. **Parameter Distributions**: View histograms, box plots, and correlations of experiment parameters
        2. **Advanced 3D Visualizations**: Use 3D scatter plots and radar charts for deeper parameter analysis
        3. **Parameter Filters**: Narrow down your analysis by filtering parameter ranges
        4. **Interactive Charts**: Zoom, pan, and select data points in all visualizations

        **Tips:**
        - Use the tabs to switch between different visualization types
        - 3D plots require at least 3 numeric parameters
        - Filters are applied in real-time to all visualizations
        - Data is cached for 10 minutes to improve performance
        """)

    # Check if experiments and datasets are selected
    if not st.session_state['selected_experiments']:
        st.warning("Please select experiments on the Home page first.")
        return

    if not st.session_state['selected_datasets']:
        st.warning("Please select datasets on the Home page first.")
        return

    # Parameter Setup (moved from Get Started)
    if not st.session_state['selected_params']:
        st.info("Please configure parameters for analysis.")
        # Get dataset_param from session state (set in get_started)
        dataset_param = st.session_state['dataset_param']
        if dataset_param:
            parameter_selection.render_parameter_selection(st.session_state['selected_experiments'], dataset_param)
            # Check if parameters were selected
            if not st.session_state['selected_params']:
                st.stop()  # Wait for parameter selection
        else:
            st.error("Dataset parameter not configured. Please complete setup on the Home page.")
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

    # Dataset Renaming
    with st.expander("Dataset Renaming", expanded=False, icon=":material/edit:"):
        st.markdown("Customize display names for datasets (used in exports and visualizations):")

        # Get current renames from session state, merged with defaults
        from statflow.config import DEFAULT_DATASET_RENAMES
        saved_renames = st.session_state['dataset_renames']
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state

        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names_params"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            State.save_to_config()
            st.success("Dataset names reset to defaults!")
            st.rerun()

        if selected_datasets:
            st.markdown("**Selected Datasets:**")
            
            # Create a form for renaming
            with st.form("dataset_renaming_form_params"):
                new_renames = {}
                for dataset in selected_datasets:
                    current_name = current_renames.get(dataset, dataset)
                    new_name = st.text_input(
                        f"Display name for '{dataset}':",
                        value=current_name,
                        key=f"params_rename_{dataset}"
                    )
                    new_renames[dataset] = new_name
                
                if st.form_submit_button("Save Dataset Names"):
                    # Update session state
                    updated_renames = current_renames.copy()
                    updated_renames.update(new_renames)
                    st.session_state['dataset_renames'] = updated_renames
                    # Save to config
                    State.save_to_config()
                    st.success("Dataset names saved!")
                    st.rerun()  # Refresh to show updated names
        else:
            st.info("No datasets selected. Please select datasets on the Home page first.")

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
        State.save_to_config()
        st.success("Parameter configuration saved!")


if __name__ == "__main__":
    main()