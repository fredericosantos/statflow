"""
Get Started page for the Statflow application.

This page provides experiment and dataset selection for analysis.

get_started.py
├── Server status check
├── Experiment selection with pills
├── Dataset selection based on experiments
└── Session state management

Usage:
    Accessed via navigation from app.py
"""

import streamlit as st

from statflow.config import SessionState
from statflow.pages_modules.module_get_started import (
    dataset_mode,
)
from statflow.pages_modules.module_get_started.server_status import handle_server_status
from statflow.pages_modules.module_get_started.provider_config import (
    render_provider_config,
)
from statflow.components.selection_ui import render_renaming_ui
from statflow.shared.server_status import ServerStatusManager


def main():
    SessionState.initialize()

    st.set_page_config(
        page_title=st.session_state["app_name"],
        page_icon=":material/home:",
    )
    status_manager = ServerStatusManager()
    render_provider_config()
    server_running = status_manager.display_sidebar()

    if not server_running:
        handle_server_status(status_manager)
        return

    # Page title
    st.title(":material/experiment: Initial Setup")

    # Main setup interface
    selected_experiments, selected_datasets, dataset_param = (
        dataset_mode.render_dataset_mode_and_selections()
    )

    if selected_experiments is None:
        return

    if selected_datasets:
        render_renaming_ui(
            items=selected_datasets,
            session_key_renames="dataset_renames",
            label="Rename Datasets",
        )

    # Navigation to next page
    st.space()
    _, col_next = st.columns([6, 1])
    with col_next:
        if st.button(
            "Next",
            type="primary",
            key="next_to_parameters",
            icon=":material/arrow_forward:",
            icon_position="right",
            width='content'
        ):
            st.switch_page("subpages/parameters.py")


if __name__ == "__main__":
    main()
