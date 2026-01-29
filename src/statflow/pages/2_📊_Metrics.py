"""
Metrics page for the Statflow application.

This page provides an overview of available metrics, their distributions, and selection.

2_📊_Metrics.py
├── Metrics overview and summary
├── Metrics distribution visualization
└── Metrics selection for analysis

Usage:
    Streamlit page for metrics management.
"""

import streamlit as st
import pandas as pd

from statflow.config import (
    initialize_session_state, save_session_state_to_config, setup_sidebar
)
from statflow.utils.mlflow_client import get_filtered_runs
from statflow.components.tables import render_table_with_downloads
from statflow.components.graphs import render_metrics_distributions
from statflow.pages_modules.module_2_Metrics.processor import (
    fetch_metrics_data, prepare_metrics_summary
)


def main():
    # Setup sidebar with server status
    setup_sidebar()

    st.title(":material/bar_chart: Metrics")
    st.markdown("Explore and select metrics for analysis.")

    # Check if experiments and datasets are selected
    if not st.session_state.get('selected_experiments'):
        st.warning("Please select experiments on the Home page first.")
        return

    if not st.session_state.get('selected_datasets'):
        st.warning("Please select datasets on the Home page first.")
        return

    # Fetch metrics data
    with st.spinner("Loading metrics data..."):
        metrics_df = fetch_metrics_data()

    if metrics_df.empty:
        st.error("No metrics data found for the selected experiments and datasets.")
        return

    # Metrics Summary
    with st.expander("Metrics Summary", expanded=True, icon=":material/summarize:"):
        summary_df = prepare_metrics_summary(metrics_df)
        if not summary_df.empty:
            render_table_with_downloads(summary_df, "Metrics Summary")
        else:
            st.info("No metrics summary available.")

    # Metrics Distributions
    with st.expander("Metrics Distributions", expanded=False, icon=":material/bar_chart:"):
        render_metrics_distributions(metrics_df)

    # Metrics Selection
    with st.expander("Metrics Selection", expanded=False, icon=":material/checklist:"):
        st.markdown("Select metrics to include in analysis:")

        available_metrics = [col for col in metrics_df.columns if col != 'dataset_name']

        if 'selected_metrics' not in st.session_state:
            st.session_state.selected_metrics = available_metrics

        selected_metrics = st.multiselect(
            "Choose metrics for analysis",
            options=available_metrics,
            default=st.session_state.selected_metrics,
            key="metrics_selector"
        )

        st.session_state.selected_metrics = selected_metrics

        if selected_metrics:
            st.success(f"Selected {len(selected_metrics)} metric{'s' if len(selected_metrics) != 1 else ''} for analysis")
        else:
            st.warning("No metrics selected. Analysis may be limited.")

    # Save configuration
    if st.button("Save Metrics Configuration", icon=":material/save:"):
        save_session_state_to_config()
        st.success("Metrics configuration saved!")


if __name__ == "__main__":
    main()