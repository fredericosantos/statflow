"""
Metrics page for the Statflow application.

This page allows users to explore, filter, and configure experiment metrics.

metrics.py
├── main()                          # Main page entry point
└── render_metric_filters()         # Dynamic metric filter UI with sliders
"""

import streamlit as st
import polars as pl

from statflow.config import SessionState
from statflow.shared.server_status import ServerStatusManager
from statflow.functional.dataframes.data_processing import fetch_experiment_data


st.set_page_config(
    page_title=f"Metrics - {st.session_state['app_name']}",
    page_icon=":material/bar_chart:",
    layout="wide",
)


def render_metric_filters(metric_df: pl.DataFrame) -> pl.DataFrame:
    """Render dynamic metric filters with sliders for numerical filtering.

    Args:
        metric_df: DataFrame with metric columns.

    Returns:
        Filtered DataFrame.
    """
    metric_cols = [
        col for col in metric_df.columns if col not in ["dataset_name"]
    ]

    if not metric_cols:
        return metric_df

    # Initialize active filters in session state
    if "active_metric_filters" not in st.session_state:
        st.session_state.active_metric_filters = []

    # Add filter using a form to prevent rerun on selectbox change
    available_to_add = [
        m for m in metric_cols if m not in st.session_state.active_metric_filters
    ]

    if available_to_add:
        with st.form("add_metric_filter_form", clear_on_submit=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_filter = st.selectbox(
                    "Select metric to filter",
                    options=available_to_add,
                    key="new_metric_filter_select",
                    label_visibility="collapsed",
                )
            with col2:
                submitted = st.form_submit_button(
                    "Add Filter", width='content'
                )

            if submitted and new_filter:
                if new_filter not in st.session_state.active_metric_filters:
                    st.session_state.active_metric_filters.append(new_filter)
                    st.rerun()

    # Render active filters with sliders
    filtered_df = metric_df.clone()

    for metric in st.session_state.active_metric_filters:
        if metric not in metric_df.columns:
            continue

        col_data = metric_df.get_column(metric).drop_nulls()
        if col_data.is_empty():
            continue

        min_val = float(col_data.min())
        max_val = float(col_data.max())

        # Avoid slider error when min == max
        if min_val == max_val:
            max_val = min_val + 1.0

        filter_col, remove_col = st.columns([5, 1])
        with filter_col:
            selected_range = st.slider(
                f"Filter: {metric}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                key=f"metric_filter_{metric}",
            )
        with remove_col:
            if st.button(
                "✕", key=f"remove_metric_filter_{metric}", help=f"Remove {metric} filter"
            ):
                st.session_state.active_metric_filters.remove(metric)
                st.rerun()

        # Apply filter
        filtered_df = filtered_df.filter(
            (pl.col(metric) >= selected_range[0]) & (pl.col(metric) <= selected_range[1])
        )

    return filtered_df


def main():
    SessionState.initialize()

    status_manager = ServerStatusManager()
    status_manager.display_sidebar()

    st.title(":material/bar_chart: Metrics")
    st.markdown("Explore and configure experiment metrics for analysis.")

    if not st.session_state["selected_experiments"]:
        st.warning("Please select experiments first in Get Started.")
        return

    selected_datasets = st.session_state["selected_datasets"]
    if not selected_datasets:
        st.warning("Please select datasets first in Get Started.")
        return

    # Metric Selection using SelectionManager
    from statflow.components.selection_ui import SelectionManager
    
    available_metrics = st.session_state.get("available_metrics", [])
    if not available_metrics:
        st.warning("No metrics found. Please load experiment data first.")
        return
    
    manager = SelectionManager(
        options=available_metrics,
        session_key="selected_metrics",
        label="Metric Selection",
        enable_ordering=False,
        enable_renaming=True,
        renames_session_key="metric_renames",
    )
    manager.render()
    
    selected_metrics = st.session_state.get("selected_metrics", [])
    if not selected_metrics:
        return

    # Fetch metric data
    with st.spinner("Loading metric data..."):
        metric_df = fetch_experiment_data("metrics.")

    if metric_df.is_empty():
        st.error("No metric data found for the selected experiments and datasets.")
        return

    # Preview data
    st.caption(f"Found {len(metric_df)} runs with metrics")

    # Metric Filters
    with st.expander(
        "Metric Filters", expanded=False, icon=":material/filter_list:"
    ):
        st.markdown("Add filters to focus on specific metric ranges:")
        metric_df = render_metric_filters(metric_df)

    # Check for empty filtered data
    if metric_df.is_empty():
        st.warning(
            "No data matches the current filter criteria. Please adjust your filters."
        )
        return

    # Data preview
    with st.expander("Data Preview", expanded=False, icon=":material/table:"):
        st.dataframe(metric_df.head(100), width='content')


if __name__ == "__main__":
    main()
