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
from statflow.pages_modules.module_get_started.server_status import ServerStatusManager
from statflow.functional.dataframes.data_processing import (
    fetch_experiment_data,
    apply_metric_filters,
)
from statflow.managers.naming import NamingManager


st.set_page_config(
    page_title=f"Metrics - {st.session_state['app_name']}",
    page_icon=":material/bar_chart:",
)


def render_metric_filters(metric_df: pl.DataFrame) -> pl.DataFrame:
    """Render dynamic metric filters with sliders for numerical filtering.

    Args:
        metric_df: DataFrame with metric columns.

    Returns:
        Filtered DataFrame.
    """
    # Filter to only numeric columns (exclude dataset_name and any string IDs)
    metric_cols = [
        col for col in metric_df.columns 
        if col != "dataset_name" and metric_df[col].dtype.is_numeric()
    ]

    if not metric_cols:
        return metric_df

    # Initialize active filters in session state
    if "active_metric_filters" not in st.session_state or st.session_state.active_metric_filters is None:
        st.session_state.active_metric_filters = []
    if "metric_filter_values" not in st.session_state:
        st.session_state.metric_filter_values = {}

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
                submitted = st.form_submit_button("Add Filter", width="content")

            if submitted and new_filter:
                if new_filter not in st.session_state.active_metric_filters:
                    st.session_state.active_metric_filters.append(new_filter)
                    SessionState.save_to_config()
                    st.rerun()

    # Render active filters with sliders
    if "metric_filter_nans" not in st.session_state:
        st.session_state.metric_filter_nans = {}

    for metric in st.session_state.active_metric_filters:
        if metric not in metric_df.columns:
            continue

        # Get valid data for range calculation
        col_data = metric_df.get_column(metric).drop_nulls().drop_nans()
        if col_data.is_empty():
            continue

        min_val = float(col_data.min())
        max_val = float(col_data.max())

        # Avoid slider error when min == max
        if min_val == max_val:
            max_val = min_val + 1.0

        # Get saved value or default to full range
        saved_val = st.session_state.metric_filter_values.get(metric)
        if saved_val:
            safe_min = max(min_val, saved_val[0])
            safe_max = min(max_val, saved_val[1])
            default_val = (safe_min, safe_max)
        else:
            default_val = (min_val, max_val)
        
        # Get Nan preference
        include_nans = st.session_state.metric_filter_nans.get(metric, False)

        filter_col, remove_col = st.columns([5, 1])
        with filter_col:
            display_name = NamingManager.get_metric_name(metric)
            selected_range = st.slider(
                f":material/filter_alt: {display_name}",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=f"metric_filter_{metric}",
            )
            
            # NaNs toggle
            new_nans = st.toggle(
                "Include NaNs",
                value=include_nans,
                key=f"metric_nans_{metric}",
            )

            # Save if changed
            if selected_range != saved_val or new_nans != include_nans:
                st.session_state.metric_filter_values[metric] = selected_range
                st.session_state.metric_filter_nans[metric] = new_nans
                # We update two keys here
                SessionState.save_to_config()
                if new_nans != include_nans:
                    st.rerun() # Rerun to apply nan change immediately

        with remove_col:
            st.write("") # Spacer to align button
            st.write("")
            if st.button(
                "Metric",
                key=f"remove_metric_filter_{metric}",
                help=f"Remove {metric} filter",
                icon=":material/delete:",
                type="primary", # Trying primary to make it stand out
            ):
                st.session_state.active_metric_filters.remove(metric)
                if metric in st.session_state.metric_filter_values:
                    del st.session_state.metric_filter_values[metric]
                if metric in st.session_state.metric_filter_nans:
                    del st.session_state.metric_filter_nans[metric]
                SessionState.save_to_config()
                st.rerun()

        # The logic has been shifted to a shared utility for consistency
        pass

    return apply_metric_filters(metric_df)


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
        enable_ordering=True,
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
    with st.expander("Metric Filters", expanded=False, icon=":material/filter_list:"):
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
        st.dataframe(metric_df.head(100), width="content")

    # Navigation to next page
    st.space()
    _, col_next = st.columns([6, 1])
    with col_next:
        if st.button(
            "Next",
            type="primary",
            key="next_to_results",
            icon=":material/arrow_forward:",
            icon_position="right",
            width="content",
        ):
            st.switch_page("subpages/results.py")


if __name__ == "__main__":
    main()
