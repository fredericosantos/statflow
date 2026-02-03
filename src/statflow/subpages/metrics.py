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

from statflow.config import State, DEFAULT_APP_NAME
from statflow.pages_modules.shared.server_status import ServerStatusManager
from statflow.functional.mlflow.mlflow_client import get_filtered_runs
from statflow.components.tables import render_table_with_downloads
from statflow.components.graphs import render_metrics_distributions
from statflow.pages_modules.module_metrics.metrics_fetcher import fetch_metrics_data
from statflow.pages_modules.module_metrics.metrics_analyzer import prepare_metrics_summary


st.set_page_config(
    page_title=f"Metrics - {st.session_state['app_name'] if 'app_name' in st.session_state else DEFAULT_APP_NAME}",
    page_icon=":material/bar_chart:",
    layout="wide",
)


def main():
    # Setup sidebar with server status
    manager = ServerStatusManager()
    server_running = manager.display_sidebar()

    # Check if MLflow server is running
    if not server_running:
        st.error("🚫 MLflow server is not running. This page requires an active MLflow server connection.", icon=":material/power_off:")
        st.info("Please start your MLflow server and refresh this page.")
        st.markdown("""
        **To start MLflow server:**
        ```bash
        mlflow server --host 0.0.0.0 --port 5000
        ```
        """)
        return

    st.title(":material/bar_chart: Metrics")
    st.markdown("Explore and select metrics for analysis.")

    # Help section
    with st.expander("ℹ️ How to use this page", expanded=False):
        st.markdown("""
        **Metrics Page Help:**

        1. **Metrics Distributions**: View histograms, box plots, and correlations of performance metrics
        2. **Advanced 3D Visualizations**: Use 3D scatter plots and radar charts for multi-metric analysis
        3. **Experiment Comparison**: Compare metrics across different experiments
        4. **Interactive Charts**: Zoom, pan, and select data points in all visualizations

        **Tips:**
        - Use the tabs to explore different aspects of your metrics
        - 3D plots help visualize trade-offs between multiple objectives
        - Correlation matrices show relationships between metrics
        - Data is cached for 10 minutes to improve performance
        """)

    # Check if experiments and datasets are selected
    if not (st.session_state['selected_experiments'] if 'selected_experiments' in st.session_state else []):
        st.warning("Please select experiments on the Home page first.")
        return

    if not (st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else []):
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
        State.save_to_config()
        st.success("Metrics configuration saved!")


if __name__ == "__main__":
    main()