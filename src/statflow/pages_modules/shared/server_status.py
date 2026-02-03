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
        if 'server_running' not in st.session_state:
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
            st.session_state['server_running'] = server_running

        return st.session_state['server_running']

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

    def handle_connection_options(self) -> None:
        """Handle connection options when server is not running."""
        tab_mlflow, tab_db = st.tabs(["Connect to MLflow", "Database Viewer"])

        with tab_mlflow:
            st.error("MLflow server is not running", icon=":material/power_off:")
            if st.button(
                icon=":material/refresh:",
                label="Recheck server status",
                key="recheck_mlflow_server_status",
            ):
                # Force recheck by clearing cached status
                if 'server_running' in st.session_state:
                    del st.session_state['server_running']
                st.rerun()

        with tab_db:
            self._render_database_viewer()

    def _render_database_viewer(self) -> None:
        """Render the database viewer interface."""
        st.markdown("### Database Viewer")
        db_path = st.text_input(
            "Database file path", placeholder="/path/to/database.db"
        )

        if db_path:
            if not Path(db_path).exists():
                st.error(f"Database file does not exist: {db_path}")
            else:
                try:
                    conn = sqlite3.connect(db_path)

                    # Get all tables
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table';"
                    )
                    tables = cursor.fetchall()
                    table_names = [table[0] for table in tables]

                    if table_names:
                        # Get row counts for all tables and filter out empty ones
                        non_empty_tables = []
                        for table in table_names:
                            cursor.execute(f"SELECT COUNT(*) FROM {table}")
                            row_count = cursor.fetchone()[0]
                            if row_count > 0:
                                non_empty_tables.append(table)

                        if non_empty_tables:
                            # Table selector
                            selected_table = st.pills(
                                "Select a table to view",
                                options=non_empty_tables,
                                key="table_selector",
                            )

                            if selected_table:
                                self._render_table_details(selected_table, conn)
                        else:
                            st.warning("No tables with data found in the database")
                    else:
                        st.warning("No tables found in the database")

                    conn.close()
                except Exception as e:
                    st.error(f"Error inspecting database: {e}")

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