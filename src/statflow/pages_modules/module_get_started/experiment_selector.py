"""
Experiment selection logic for get started page.

This module handles the logic for selecting experiments from MLflow,
including experiment discovery and selection UI.

experiment_selector.py
├── select_experiment()              # Handle experiment selection with custom label
├── select_experiment_as_datasets()  # Handle experiment selection as datasets
"""

import streamlit as st

from statflow.loggers.mlflow.mlflow_client import get_experiment_names
from statflow.loggers.mlflow.runs_cache import RunsCache


def select_experiment_as_datasets() -> list[str] | None:
    """Handle dataset selection when datasets are experiment names.

    Returns:
        List of selected dataset names (experiment names), or None if no experiments available.
    """
    experiment_names = get_experiment_names()
    if not experiment_names:
        return None

    selected_datasets = st.pills(
        "Select Datasets from MLFlow",
        options=experiment_names,
        default=st.session_state["selected_datasets"],
        key="dataset_selector_from_experiments",
        selection_mode="multi",
    )

    # Update session state
    st.session_state.selected_datasets = selected_datasets

    # Also update experiments for metadata
    selected_experiments = selected_datasets
    # Load if selection changed OR if we have selection but no data (initial load)
    if (
        selected_experiments != st.session_state["selected_experiments"]
        or (selected_experiments and RunsCache.get_run_count() == 0)
    ):
        st.session_state.selected_experiments = selected_experiments
        # Clear cache and load initial batch
        RunsCache.clear_cache()
        if selected_experiments:
            max_results = st.session_state.get("max_results", 2000)
            with st.spinner("Loading experiment runs..."):
                RunsCache.load_runs(selected_experiments, max_results=max_results)

    return selected_datasets


def select_experiment(
    label: str = "Select Experiments from MLFlow",
) -> list[str] | None:
    """Handle experiment selection UI and logic.

    Args:
        label: Custom label for the selection component.

    Returns:
        List of selected experiment names, or None if no experiments available.
    """
    experiment_names = get_experiment_names()
    if not experiment_names:
        return None

    st.markdown(f"#### {label}")
    selected_experiments = st.pills(
        label,
        options=experiment_names,
        default=st.session_state["selected_experiments"],
        key="experiment_selector",
        selection_mode="multi",
        label_visibility="collapsed",
    )
    st.space()

    # Check if selection changed - load initial runs
    # Load if selection changed OR if we have selection but no data (initial load)
    if (
        selected_experiments != st.session_state["selected_experiments"]
        or (selected_experiments and RunsCache.get_run_count() == 0)
    ):
        st.session_state.selected_experiments = selected_experiments
        # Clear cache and load initial batch
        RunsCache.clear_cache()
        if selected_experiments:
            max_results = st.session_state.get("max_results", 1000)
            with st.spinner("Loading experiment runs..."):
                RunsCache.load_runs(selected_experiments, max_results=max_results)

    return selected_experiments
