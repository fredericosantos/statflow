"""
Experiment selection logic for get started page.

This module handles the logic for selecting experiments from MLflow,
including experiment discovery, selection UI, and metadata updates.

experiment_selector.py
├── handle_experiment_selection()           # Handle experiment selection with custom label
├── handle_experiment_selection_as_datasets() # Handle experiment selection as datasets
└── _update_experiment_metadata()           # Update session state with experiment metadata
"""

import streamlit as st
from statflow.functional.mlflow.mlflow_client import (
    get_experiment_names,
    get_metadata_from_experiments
)


def handle_experiment_selection_as_datasets() -> list[str] | None:
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
        default=st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else [],
        key="dataset_selector_from_experiments",
        selection_mode="multi",
    )

    # Update session state
    st.session_state.selected_datasets = selected_datasets

    # Also update experiments for metadata
    selected_experiments = selected_datasets
    if selected_experiments != (st.session_state['selected_experiments'] if 'selected_experiments' in st.session_state else []):
        _update_experiment_metadata(selected_experiments)
        st.session_state.selected_experiments = selected_experiments

    return selected_datasets


def handle_experiment_selection(label: str = "Select Experiments from MLFlow") -> list[str] | None:
    """Handle experiment selection UI and logic.

    Args:
        label: Custom label for the selection component.

    Returns:
        List of selected experiment names, or None if no experiments available.
    """
    experiment_names = get_experiment_names()
    if not experiment_names:
        return None

    selected_experiments = st.pills(
        label,
        options=experiment_names,
        default=st.session_state['selected_experiments'] if 'selected_experiments' in st.session_state else [],
        key="experiment_selector",
        selection_mode="multi",
    )

    # Check if selection changed
    if selected_experiments != (st.session_state['selected_experiments'] if 'selected_experiments' in st.session_state else []):
        _update_experiment_metadata(selected_experiments)
        st.session_state.selected_experiments = selected_experiments

    return selected_experiments


def _update_experiment_metadata(selected_experiments: list[str]) -> None:
    """Update session state with metadata for selected experiments."""
    if selected_experiments:
        with st.spinner("Loading experiment metadata..."):
            metadata = get_metadata_from_experiments(tuple(selected_experiments))
            st.session_state.available_params = metadata["params"]
            st.session_state.available_param_values = metadata["param_values"]
            st.session_state.available_metrics = metadata["metrics"]
    else:
        st.session_state.available_params = []
        st.session_state.available_param_values = {}
        st.session_state.available_metrics = []