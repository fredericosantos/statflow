"""
Shared server status checking functionality.

This module provides centralized MLflow server status checking and UI components
that can be used across multiple pages and modules.

server_status.py
├── ServerStatusManager  # Class for managing server status and UI
"""

import requests
import streamlit as st
import polars as pl
from pathlib import Path
import sqlite3


class ServerStatusManager:
    """Manager class for MLflow server status checking and UI handling."""

    def check_status(self) -> bool:
        """Check if MLflow server is running and cache the result in session state.

        The status is checked only once per session and cached.

        Returns:
            bool: True if server is running, False otherwise.
        """
        if "server_running" not in st.session_state:
            tracking_uri = st.session_state["mlflow_server_url"]
            if not tracking_uri.startswith("http"):
                # If not HTTP, assume it's running (file:// or other)
                server_running = True
            else:
                try:
                    # Try to connect to the MLflow server health endpoint
                    health_url = tracking_uri.rstrip("/") + "/health"
                    response = requests.get(health_url, timeout=2)
                    server_running = response.status_code == 200
                except:
                    server_running = False
            st.session_state["server_running"] = server_running

        return st.session_state["server_running"]

    def display_sidebar(self) -> bool:
        """Display server status in the sidebar.

        Returns:
            bool: True if server is running, False otherwise.
        """
        server_running = self.check_status()

        with st.sidebar:
            if server_running:
                st.success("MLFlow Server Running", icon=":material/bolt:")
            else:
                st.error("Server Not Running", icon=":material/power_off:")

        return server_running

    def _render_table_details(self, table_name: str, conn) -> None:
        """Render schema and data for a selected table."""
        # Show schema
        st.markdown(f"#### Schema for table: {table_name}")
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        if columns:
            df_schema = pl.DataFrame(
                columns,
                schema=[
                    "cid",
                    "name",
                    "type",
                    "notnull",
                    "dflt_value",
                    "pk",
                ],
            )
            st.dataframe(df_schema.select(["name", "type", "notnull", "pk"]))

            # Show row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            st.write(f"**Rows:** {row_count}")

            # Load and show data
            st.markdown(f"#### Data from table: {table_name}")
            try:
                df = pl.read_database(
                    f"SELECT * FROM {table_name}",
                    conn,
                    infer_schema_length=None,
                )
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error loading table data with Polars: {e}")
        else:
            st.write("No schema information available")
