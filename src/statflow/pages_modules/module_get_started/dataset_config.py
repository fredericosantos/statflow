"""
Dataset configuration logic for get started page.

This module handles dataset parameter selection, dataset discovery,
and dataset ordering functionality using the centralized RunsCache.

dataset_config.py
├── handle_dataset_selection()              # Handle dataset selection and ordering
├── _get_available_datasets()               # Get available datasets based on mode
├── _render_dataset_selector()              # Render dataset selection UI
├── _render_dataset_ordering()              # Render dataset ordering UI
├── render_rename_datasets_ui()             # Render dataset renaming UI
└── _render_multiple_datasets_ui()          # Render multiple datasets mode UI
"""

import streamlit as st

from statflow.components.selection_ui import (
    render_item_ordering,
    render_item_selector,
    render_renaming_ui,
)
from statflow.config import SessionState
from statflow.loggers.runs_cache import RunsCache

from .constants import DatasetParamMode


def _sort_datasets(datasets: list[str]) -> list[str]:
    """Sort datasets by their parameter values, handling numeric values correctly."""

    def sort_key(value: str) -> tuple[bool, float | str]:
        try:
            return (False, float(value))
        except ValueError:
            return (True, value.lower())

    return sorted(datasets, key=sort_key)


def _get_available_datasets(
    selected_experiments: list[str], dataset_param: DatasetParamMode | str
) -> list[str]:
    """Get available datasets based on the dataset parameter mode."""
    if dataset_param == DatasetParamMode.EXPERIMENT_NAMES_AS_DATASETS:
        return selected_experiments
    elif dataset_param == DatasetParamMode.SINGLE_DATASET_MODE:
        return ["default"]
    else:
        # Multiple datasets mode - get from cache
        values = RunsCache.get_param_values(dataset_param)
        return _sort_datasets(values)


def _render_dataset_selector(available_datasets: list[str]) -> list[str]:
    """Render the dataset selector UI."""
    return render_item_selector(
        options=available_datasets,
        session_key="selected_datasets",
        label="Select Datasets",
        renames_session_key="dataset_renames",
    )


def _render_dataset_ordering(selected_datasets: list[str]) -> list[str]:
    """Render the dataset ordering UI."""
    return render_item_ordering(
        items=selected_datasets,
        session_key="selected_datasets",
        label="Order Selected Datasets",
        renames_session_key="dataset_renames",
    )


def render_rename_datasets_ui(selected_datasets: list[str]) -> None:
    """Render the UI for renaming datasets."""
    render_renaming_ui(
        items=selected_datasets,
        session_key_renames="dataset_renames",
        label="Rename Datasets",
    )


def _render_multiple_datasets_ui(selected_experiments: list[str]) -> list[str] | None:
    """Render the UI for multiple datasets mode."""
    available_params = RunsCache.get_available_params()
    batch_size = st.session_state.get("max_results", 1000)
    run_count = RunsCache.get_run_count()
    new_runs_found = 0

    # Handle search action
    if st.session_state.get("_trigger_search", False):
        st.session_state._trigger_search = False
        with st.spinner("Searching for more runs..."):
            new_runs_found = RunsCache.load_more_runs(selected_experiments, max_results=batch_size)
            run_count = RunsCache.get_run_count()  # Refresh count

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1], vertical_alignment="bottom")

    with col1:
        if available_params:
            current_param = st.session_state.get("dataset_param", "")
            default_index = (
                available_params.index(current_param) if current_param in available_params else 0
            )
            dataset_param = st.selectbox(
                "Select parameter that defines dataset names",
                options=available_params,
                index=default_index,
                key="dataset_param_selector",
                help="Runs without a value in the selected parameter will be filtered out",
            )
            st.session_state.dataset_param = dataset_param
        else:
            st.warning("No parameters found. Search for runs first.")
            dataset_param = st.session_state.get("dataset_param", "")

    with col2:
        st.number_input(
            "Batch size",
            min_value=100,
            max_value=50000,
            step=500,
            help="Number of runs to fetch per search.",
            key="max_results",
        )

        # Historical max runs logic
        historical_max = st.session_state.get("historical_max_run_count", 0)

        # Update historical max if current runs are higher
        if run_count > historical_max:
            st.session_state["historical_max_run_count"] = run_count
            historical_max = run_count
            SessionState.save_to_config()

    with col3:
        button_label = (
            f"Search {batch_size} more runs" if run_count > 0 else f"Search {batch_size} runs"
        )
        if st.button(button_label, key="search_datasets", width="stretch"):
            st.session_state._trigger_search = True
            st.rerun()

    with col4:
        # Show button to load max runs if applicable
        if historical_max > batch_size:

            def _load_max_runs():
                st.session_state["max_results"] = historical_max
                st.session_state._trigger_search = True

            st.button(
                f"Load {historical_max} runs",
                help=f"Set batch size to your historical maximum of {historical_max} runs",
                width="stretch",
                on_click=_load_max_runs,
            )

    # Show run count status
    if run_count > 0:
        if new_runs_found > 0:
            st.caption(
                f":material/analytics: {run_count} runs loaded (+{new_runs_found} new)",
                text_alignment="center",
            )
        else:
            st.caption(f":material/analytics: {run_count} runs loaded", text_alignment="center")
    elif new_runs_found == 0 and st.session_state.get("_searched_once", False):
        st.caption(":material/analytics: No additional runs found", text_alignment="center")

    # Get available datasets from cache
    available_datasets = _get_available_datasets(selected_experiments, dataset_param)

    if not available_datasets:
        if run_count == 0:
            st.info("Click 'Search runs' to load experiment data.")
        else:
            st.info(f"No datasets found for parameter '{dataset_param}'.")
        return None

    selected_datasets = _render_dataset_selector(available_datasets)
    ordered_datasets = _render_dataset_ordering(selected_datasets)

    # Warning moved to selectbox help

    return ordered_datasets


def handle_dataset_selection(
    selected_experiments: list[str], dataset_param: DatasetParamMode | str
) -> list[str] | None:
    """Handle dataset selection and ordering.

    Args:
        selected_experiments: List of selected experiment names.
        dataset_param: The parameter that defines dataset names.

    Returns:
        List of selected dataset names, or None if no datasets available.
    """
    if not selected_experiments:
        st.info("Please select at least one experiment to see available datasets.")
        return None

    if dataset_param == DatasetParamMode.EXPERIMENT_NAMES_AS_DATASETS:
        available_datasets = _get_available_datasets(selected_experiments, dataset_param)
        selected_datasets = _render_dataset_selector(available_datasets)
        ordered_datasets = _render_dataset_ordering(selected_datasets)
        return ordered_datasets
    elif dataset_param == DatasetParamMode.SINGLE_DATASET_MODE:
        available_datasets = _get_available_datasets(selected_experiments, dataset_param)
        selected_datasets = _render_dataset_selector(available_datasets)
        ordered_datasets = _render_dataset_ordering(selected_datasets)
        return ordered_datasets
    else:
        # Multiple datasets mode
        return _render_multiple_datasets_ui(selected_experiments)
