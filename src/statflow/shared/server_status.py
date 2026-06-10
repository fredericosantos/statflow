"""
Shared backend status checking functionality.

Centralized status checking + sidebar UI for the active run provider. The actual
reachability/auth probe is delegated to the provider (`check_status`), so this
module stays backend-agnostic: MLflow probes `/health`, W&B probes its API auth.

server_status.py
├── ServerStatusManager  # Class for managing provider status and UI
"""

import streamlit as st

from statflow.loggers.registry import get_provider


class ServerStatusManager:
    """Manager class for run-provider status checking and sidebar UI."""

    def _provider(self):
        return get_provider(st.session_state["provider"])

    def check_status(self) -> bool:
        """Check if the active provider is reachable; cache the result per session.

        The status is checked only once per session and cached under
        `server_running`. Clear that key to force a recheck.

        Returns:
            bool: True if the provider is reachable/authenticated.
        """
        if "server_running" not in st.session_state:
            st.session_state["server_running"] = self._provider().check_status()

        return st.session_state["server_running"]

    def display_sidebar(self) -> bool:
        """Display provider status in the sidebar.

        Returns:
            bool: True if the provider is reachable, False otherwise.
        """
        server_running = self.check_status()
        label = self._provider().label

        with st.sidebar:
            if server_running:
                st.success(f"{label} Connected", icon=":material/bolt:")
            else:
                st.error(f"{label} Not Reachable", icon=":material/power_off:")

        return server_running
