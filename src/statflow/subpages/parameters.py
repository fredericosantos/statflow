"""
Parameters page for the Statflow application.

This page allows users to explore, filter, and configure experiment parameters.

parameters.py
├── main()                          # Main page entry point
├── handle_parameter_selection()    # Parameter selection UI
└── render_parameter_filters()      # Dynamic parameter filter UI
"""

import polars as pl
import streamlit as st

from statflow.config import SessionState
from statflow.functional.dataframes.data_processing import fetch_experiment_data
from statflow.shared.server_status import ServerStatusManager

st.set_page_config(
    page_title=f"Parameters - {st.session_state['app_name']}",
    page_icon=":material/tune:",
)


def render_parameter_filters(param_df: pl.DataFrame) -> pl.DataFrame:
    """Render dynamic parameter filters with Add Filter button.

    Args:
        param_df: DataFrame with parameter columns.

    Returns:
        Filtered DataFrame.
    """
    param_cols = [col for col in param_df.columns if col not in ["dataset_name", "group_label"]]

    if not param_cols:
        return param_df

    # Initialize active filters in session state
    if "active_param_filters" not in st.session_state:
        st.session_state.active_param_filters = []

    # Add filter using a form to prevent rerun on selectbox change
    available_to_add = [p for p in param_cols if p not in st.session_state.active_param_filters]

    if available_to_add:
        with st.form("add_filter_form", clear_on_submit=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                new_filter = st.selectbox(
                    "Select parameter to filter",
                    options=available_to_add,
                    key="new_filter_select",
                    label_visibility="collapsed",
                )
            with col2:
                submitted = st.form_submit_button("Add Filter", width="content")

            if submitted and new_filter:
                if new_filter not in st.session_state.active_param_filters:
                    st.session_state.active_param_filters.append(new_filter)
                    st.rerun()

    # Render active filters
    filtered_df = param_df.clone()

    for param in st.session_state.active_param_filters:
        if param not in param_df.columns:
            continue

        col_data = param_df.get_column(param)
        unique_vals = sorted(col_data.drop_nulls().unique().to_list())

        filter_col, remove_col = st.columns([5, 1])
        with filter_col:
            selected_vals = st.multiselect(
                f"Filter: {param}",
                unique_vals,
                default=unique_vals,
                key=f"filter_{param}",
            )
        with remove_col:
            if st.button("✕", key=f"remove_filter_{param}", help=f"Remove {param} filter"):
                st.session_state.active_param_filters.remove(param)
                st.rerun()

        if selected_vals:
            filtered_df = filtered_df.filter(pl.col(param).is_in(selected_vals))

    return filtered_df


def reset_active_group_filters():
    """Reset the transient group filter session state."""
    st.session_state.active_group_filters = []


def handle_parameter_selection(
    selected_experiments: list[str], dataset_param: str
) -> list[str] | None:
    """Handle parameter selection UI."""
    from statflow.components.selection_ui import (
        render_item_ordering,
        render_selection_pills,
    )

    available_params = st.session_state["available_params"]
    selectable_params = [p for p in available_params if p != dataset_param]

    if not selectable_params:
        st.warning(
            "No parameters available yet. Select experiments/datasets on Get Started "
            "(and make sure runs loaded) first."
        )
        return None

    st.markdown("Select parameters to include in comparisons and analysis:")

    selected_params = render_selection_pills(
        selectable_params,
        "selected_params",
        label="Parameters",
        on_change=reset_active_group_filters,
        label_visibility="collapsed",
    )

    if selected_params:
        # Allow reordering of selected parameters
        selected_params = render_item_ordering(
            items=selected_params,
            session_key="selected_params",  # Use same key to overwrite order
            label="Order parameters",
            key_suffix="_param_order",
        )

    return selected_params


def main():
    SessionState.initialize()

    status_manager = ServerStatusManager()
    status_manager.display_sidebar()

    st.title(":material/tune: Parameters")
    st.markdown("Explore and configure experiment parameters for analysis.")

    if not st.session_state["selected_experiments"]:
        st.warning("Please select experiments first.")
        return

    selected_datasets = st.session_state["selected_datasets"]
    if not selected_datasets:
        st.warning("Please select datasets first.")
        return

    with st.expander("Parameter Configuration", expanded=True, icon=":material/settings:"):
        selected_params = handle_parameter_selection(
            st.session_state["selected_experiments"], st.session_state["dataset_param"]
        )

    if not selected_params:
        return

    # Fetch parameter data
    with st.spinner("Loading parameter data..."):
        param_df = fetch_experiment_data("params.")

    if param_df.is_empty():
        st.error("No parameter data found for the selected experiments and datasets.")
        return

    comparison_params = st.session_state["selected_params"]
    if not comparison_params:
        return

    with st.expander("Filters", expanded=False, icon=":material/filter_list:"):
        st.markdown("Add filters to focus on specific parameter values:")
        param_df = render_parameter_filters(param_df)

    # Check for empty filtered data
    if param_df.is_empty():
        st.warning("No data matches the current filter criteria. Please adjust your filters.")
        return

    # Create group labels
    exprs = []
    for i, p in enumerate(comparison_params):
        if i > 0:
            exprs.append(pl.lit(", "))
        exprs.append(pl.lit(f"{p}="))
        exprs.append(pl.col(p).cast(pl.Utf8))

    param_df = param_df.with_columns(pl.concat_str(exprs).alias("group_label"))

    available_groups = sorted(param_df.get_column("group_label").drop_nulls().unique().to_list())

    # Use SORTED key for cache (so A->B and B->A share same cache)
    sorted_cache_key = ",".join(sorted(comparison_params))

    # Use SelectionManager for group selection with caching
    from statflow.components.selection_ui import SelectionManager

    manager = SelectionManager(
        options=available_groups,
        session_key="selected_groups",
        label="Group Selection",
        enable_ordering=True,
        enable_renaming=True,
        renames_session_key="group_renames",
        use_fragment=True,
        cache_key=sorted_cache_key,
        cache_param_order=comparison_params,
        on_change=reset_active_group_filters,
    )
    manager.render()

    selected_groups = st.session_state.get("selected_groups", [])

    if selected_groups:
        param_df = param_df.filter(pl.col("group_label").is_in(st.session_state["selected_groups"]))

    # Navigation to next page
    st.space()
    _, col_next = st.columns([6, 1])
    with col_next:
        if st.button(
            "Next",
            type="primary",
            key="next_to_metrics",
            icon=":material/arrow_forward:",
            icon_position="right",
            width="content",
        ):
            st.switch_page("subpages/metrics.py")


if __name__ == "__main__":
    main()
