"""
Dataset mode selection for get started page.

This module handles the selection of how datasets are defined
and coordinates the appropriate selection flows.

dataset_mode.py
├── render_dataset_mode_and_selections()    # Main entry point for dataset mode UI
├── handle_dataset_parameter_selection()    # Handle dataset parameter selection
└── _find_dataset_parameter_suggestion()    # Find best dataset parameter suggestion
"""

import streamlit as st

from statflow.pages_modules.module_get_started.experiment_selector import (
    select_experiment,
    select_experiment_as_datasets,
)
from statflow.pages_modules.module_get_started.dataset_config import (
    handle_dataset_selection,
)
from statflow.loggers.mlflow.runs_cache import RunsCache
from .constants import DatasetParamMode


def render_dataset_mode_and_selections():
    """Main entry point for dataset mode selection and related flows."""
    options = [
        "Dataset names are experiment names",
        "Single dataset",
        "Multiple datasets defined by parameter",
    ]

    current = st.session_state.get("dataset_mode", options[0])
    if current not in options:
        current = options[0]

    dataset_mode = st.radio(
        "Choose how datasets are identified in your experiments. ",
        options=options,
        index=options.index(current),
        key="dataset_mode_radio",
    )
    st.session_state["dataset_mode"] = dataset_mode

    if dataset_mode == "Dataset names are experiment names":
        selected_datasets = select_experiment_as_datasets()
        selected_experiments = selected_datasets
        dataset_param = DatasetParamMode.EXPERIMENT_NAMES_AS_DATASETS
    elif dataset_mode == "Single dataset":
        selected_experiments = select_experiment()
        selected_datasets = ["Default"]
        dataset_param = DatasetParamMode.SINGLE_DATASET_MODE
    else:
        selected_experiments = select_experiment()
        if not selected_experiments:
            st.info("Please select at least one experiment to proceed.")
            return None, None, None

        dataset_param = handle_dataset_parameter_selection(
            selected_experiments, dataset_mode
        )
        if not dataset_param:
            st.info("Please select a dataset parameter to proceed.")
            return None, None, None

        selected_datasets = handle_dataset_selection(
            selected_experiments, dataset_param
        )

    return selected_experiments, selected_datasets, dataset_param


def handle_dataset_parameter_selection(
    selected_experiments: list[str], dataset_mode: str
) -> str | None:
    """Handle dataset parameter selection.

    Args:
        selected_experiments: List of selected experiment names.
        dataset_mode: The mode selected for dataset definition.

    Returns:
        Selected dataset parameter name, or None if not found.
    """
    if not selected_experiments:
        return None

    available_params = RunsCache.get_available_params()

    if not available_params:
        st.error("No parameters found in the selected experiments.")
        st.session_state.dataset_param = None
        return None

    if dataset_mode == "Multiple datasets defined by parameter":
        suggested_default = _find_dataset_parameter_suggestion(available_params)
        st.session_state.dataset_param = suggested_default
        return suggested_default
    else:
        dataset_param = None

    st.session_state.dataset_param = dataset_param
    return dataset_param


def _find_dataset_parameter_suggestion(available_params: list[str]) -> str:
    """Find the best suggestion for dataset parameter."""
    if "dataset_name" in available_params:
        return "dataset_name"
    elif "dataset" in available_params:
        return "dataset"
    elif any("dataset" in p.lower() and "name" in p.lower() for p in available_params):
        for p in available_params:
            if "dataset" in p.lower() and "name" in p.lower():
                return p
    return available_params[0] if available_params else ""
