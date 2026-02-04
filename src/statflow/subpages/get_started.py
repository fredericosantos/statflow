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
import mlflow

from statflow.config import SessionState
from statflow.pages_modules.module_get_started import (
    dataset_mode,
)
from statflow.pages_modules.module_get_started.server_status import handle_server_status
from statflow.components.selection_ui import (
    render_item_selector,
    render_item_ordering,
    render_renaming_ui,
)
from statflow.shared.server_status import ServerStatusManager


def main():
    SessionState.initialize()
    mlflow.set_tracking_uri(st.session_state["mlflow_server_url"])

    st.set_page_config(
        page_title=st.session_state["app_name"],
        page_icon=":material/home:",
        layout="wide",
    )
    status_manager = ServerStatusManager()
    server_running = status_manager.display_sidebar()


    if not server_running:
        server_running = handle_server_status(status_manager)
        return

    with st.expander("Initial Setup", expanded=True, icon=":material/experiment:"):
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

    # Save configuration button
    st.divider()
    if st.button("Save Configuration", icon=":material/save:", width='content'):
        SessionState.save_to_config()
        st.success("Configuration saved!")


if __name__ == "__main__":
    main()
