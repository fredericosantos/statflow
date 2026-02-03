"""
Parameter selection UI for get started page.

This module handles the parameter selection interface,
including parameter pills and linking configuration.

parameter_selection.py
├── Parameter selection with pills
├── Parameter linking configuration
└── UI rendering for parameter management
"""

import streamlit as st

from statflow.pages_modules.module_get_started.parameter_config import handle_parameter_selection


def render_parameter_selection(selected_experiments: list[str], dataset_param: str) -> list[str] | None:
    """Render the parameter selection interface.

    Args:
        selected_experiments: List of selected experiment names
        dataset_param: Dataset parameter configuration

    Returns:
        List of selected parameters, or None
    """
    with st.expander("Parameter Selection", expanded=False, icon=":material/tune:"):
        selected_params = handle_parameter_selection(
            selected_experiments, dataset_param
        )
    return selected_params