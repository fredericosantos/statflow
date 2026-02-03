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

from statflow.config import State
from statflow.pages_modules.module_get_started import (
    dataset_mode,
)
from statflow.pages_modules.module_get_started.dataset_config import (
    render_rename_datasets_ui,
)
from statflow.pages_modules.shared.server_status import ServerStatusManager
import mlflow


def main():
    # Initialize session state first
    State.initialize()

    # Set MLflow tracking URI globally
    mlflow.set_tracking_uri(st.session_state["mlflow_server_url"])

    st.set_page_config(
        page_title=st.session_state["app_name"],
        page_icon=":material/home:",
        layout="wide",
    )

    # Initialize server status manager
    status_manager = ServerStatusManager()

    # Check server status in sidebar
    server_running = status_manager.display_sidebar()

    if not server_running:
        # Show connection options in main area
        status_manager.handle_connection_options()
        return

    with st.expander("Initial Setup", expanded=True, icon=":material/experiment:"):
        # Main setup interface
        selected_experiments, selected_datasets, dataset_param = dataset_mode.render_dataset_mode_and_selections()
        if selected_experiments is None:
            return

    # Save configuration after selections
    State.save_to_config()

    if selected_datasets:
        render_rename_datasets_ui(selected_datasets)
        # selected_datasets remain as original names
        # rename mapping is stored in st.session_state['dataset_renames']
        State.save_to_config()


if __name__ == "__main__":
    main()
