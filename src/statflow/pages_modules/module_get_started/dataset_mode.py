"""
Dataset mode selection for get started page.

This module handles the selection of how datasets are defined
and coordinates the appropriate selection flows.

dataset_mode.py
├── Dataset definition mode selection
├── Conditional logic for different modes
└── Coordination of experiment/dataset selection
"""

import streamlit as st

from statflow.pages_modules.module_get_started.experiment_selector import (
    handle_experiment_selection,
    handle_experiment_selection_as_datasets,
)
from statflow.pages_modules.module_get_started.dataset_config import (
    handle_dataset_parameter_selection,
    handle_dataset_selection,
)
from .constants import DatasetParamMode


def render_dataset_mode_and_selections():
    """Render dataset mode selection and handle conditional logic.

    Returns:
        tuple: (selected_experiments, selected_datasets, dataset_param)
    """
    options = [
        "Dataset names are experiment names",
        "Single dataset",
        "Multiple datasets defined by parameter",
    ]

    # Handle legacy session state values
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
        # Step 1: Dataset Selection (experiments are datasets)
        selected_datasets = handle_experiment_selection_as_datasets()
        selected_experiments = selected_datasets  # experiments are datasets
        dataset_param = DatasetParamMode.EXPERIMENT_NAMES_AS_DATASETS
    elif dataset_mode == "Single dataset":
        # Step 1: Experiment Selection
        selected_experiments = handle_experiment_selection()
        selected_datasets = ["default"]
        dataset_param = DatasetParamMode.SINGLE_DATASET_MODE
        # dataset_param = "SINGLE_DATASET_MODE" # This line was redundant/incorrect in previous view
    else:
        # Step 1: Experiment Selection
        selected_experiments = handle_experiment_selection()
        if not selected_experiments:
            st.info("Please select at least one experiment to proceed.")
            return None, None, None

        # Step 2: Dataset Parameter Selection
        dataset_param = handle_dataset_parameter_selection(
            selected_experiments, dataset_mode
        )
        if not dataset_param:
            st.info("Please select a dataset parameter to proceed.")
            return None, None, None

        # Step 3: Dataset Selection
        selected_datasets = handle_dataset_selection(
            selected_experiments, dataset_param
        )

    return selected_experiments, selected_datasets, dataset_param