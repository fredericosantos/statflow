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
    render_item_selector,
    render_item_ordering,
    render_renaming_ui,
)
from statflow.loggers.mlflow.runs_cache import RunsCache
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
    )


def _render_dataset_ordering(selected_datasets: list[str]) -> list[str]:
    """Render the dataset ordering UI."""
    return render_item_ordering(
        items=selected_datasets,
        session_key="selected_datasets",
        label="Order Selected Datasets",
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
    batch_size = st.session_state.get("max_results", 2000)
    run_count = RunsCache.get_run_count()

    # Handle search action
    if st.session_state.get("_trigger_search", False):
        st.session_state._trigger_search = False
        with st.spinner("Searching for more runs..."):
            new_count = RunsCache.load_more_runs(selected_experiments, max_results=batch_size)
        if new_count > 0:
            st.success(f"Found {new_count} new runs.")
        else:
            st.info("No additional runs found.")

    col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment="bottom")

    with col1:
        if available_params:
            current_param = st.session_state.get("dataset_param", "")
            default_index = (
                available_params.index(current_param)
                if current_param in available_params
                else 0
            )
            dataset_param = st.selectbox(
                "Select parameter that defines dataset names",
                options=available_params,
                index=default_index,
                key="dataset_param_selector",
            )
            st.session_state.dataset_param = dataset_param
        else:
            st.warning("No parameters found. Search for runs first.")
            dataset_param = st.session_state.get("dataset_param", "")

    with col2:
        st.number_input(
            "Batch size",
            min_value=100,
            max_value=10000,
            value=batch_size,
            step=500,
            help="Number of runs to fetch per search.",
            key="max_results",
        )

    with col3:
        button_label = (
            f"Search {batch_size} more runs"
            if run_count > 0
            else f"Search {batch_size} runs"
        )
        if st.button(button_label, key="search_datasets"):
            st.session_state._trigger_search = True
            st.rerun()

    # Show current run count
    if run_count > 0:
        st.caption(f"📊 {run_count} runs loaded")

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

    if dataset_param:
        st.warning(
            f"Runs without a value in `{dataset_param}` will be filtered out",
            icon=":material/warning:",
        )

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
        available_datasets = _get_available_datasets(
            selected_experiments, dataset_param
        )
        selected_datasets = _render_dataset_selector(available_datasets)
        ordered_datasets = _render_dataset_ordering(selected_datasets)
        return ordered_datasets
    elif dataset_param == DatasetParamMode.SINGLE_DATASET_MODE:
        available_datasets = _get_available_datasets(
            selected_experiments, dataset_param
        )
        selected_datasets = _render_dataset_selector(available_datasets)
        ordered_datasets = _render_dataset_ordering(selected_datasets)
        return ordered_datasets
    else:
        # Multiple datasets mode
        return _render_multiple_datasets_ui(selected_experiments)
