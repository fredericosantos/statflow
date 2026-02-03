"""
Server status handling for get started page.

This module handles MLflow server status checks and provides
a database viewer fallback when the server is unavailable.

server_status.py
├── MLflow server status checking
├── Database viewer functionality
└── Error handling and user feedback
"""

import streamlit as st
import polars as pl
from pathlib import Path

from statflow.pages_modules.shared.server_status import ServerStatusManager


def handle_server_status() -> bool:
    """Handle MLflow server status checking and database viewer.

    Returns:
        bool: True if server is running, False otherwise
    """
    # Check MLflow server status, use cached if available
    manager = ServerStatusManager()
    server_running = manager.check_status()

    # If server is not running, show connection options
    if not server_running:
        tab_mlflow, tab_db = st.tabs(["Connect to MLflow", "Database Viewer"])

        with tab_mlflow:
            st.error("MLflow server is not running", icon=":material/power_off:")
            if st.button(
                icon=":material/refresh:",
                label="Recheck server status",
                key="recheck_mlflow_server_status",
            ):
                server_running = manager.check_status()
                st.session_state['server_running'] = server_running
                st.rerun()

        with tab_db:
            _render_database_viewer()

        return False

    return True


def _render_database_viewer() -> None:
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
                import sqlite3

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
                            selection_mode="single",
                        )

                        if selected_table:
                            _render_table_details(selected_table, conn)
                    else:
                        st.warning("No tables with data found in the database")
                else:
                    st.warning("No tables found in the database")

                conn.close()
            except Exception as e:
                st.error(f"Error inspecting database: {e}")


def _render_table_details(table_name: str, conn) -> None:
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