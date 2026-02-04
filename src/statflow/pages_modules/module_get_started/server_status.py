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
import sqlite3

from statflow.shared.server_status import ServerStatusManager
from statflow.config import SessionState


def handle_server_status(server_status_manager: ServerStatusManager) -> bool:
    """Handle MLflow server status checking and database viewer.

    Returns:
        bool: True if server is running, False otherwise
    """
    # Check MLflow server status, use cached if available
    server_running = server_status_manager.check_status()

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
                # Force recheck by clearing cached status
                if "server_running" in st.session_state:
                    del st.session_state["server_running"]
                st.rerun()

        with tab_db:
            _render_database_viewer()

        return False

    return True


def _render_database_viewer() -> None:
    """Render the database viewer interface."""
    # Try to get path from session state/config first
    current_db_path = SessionState.get("mlflow_db_path", "")

    # If empty, search for the database file in the directory where the app was called from
    if not current_db_path:
        cwd = Path.cwd()
        db_files = list(cwd.glob("*.db"))
        db_filepath = str(db_files[0]) if db_files else None
    else:
        db_filepath = current_db_path

    db_path = st.text_input(
        "Database file path",
        placeholder=db_filepath if db_filepath else "/path/to/database.db",
        value=db_filepath,
        key="mlflow_db_path_input",
    )

    # Save to config if it changed
    if db_path and db_path != current_db_path:
        SessionState.set("mlflow_db_path", db_path)
        SessionState.save_to_config()
        st.success(f"Saved database path to config: {db_path}")

    if db_path:
        if not Path(db_path).exists():
            st.error(f"Database file does not exist: {db_path}")
        else:
            try:
                conn = sqlite3.connect(db_path)

                # Get all tables
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
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
        # Robust Schema Rendering
        schema_cols = ["cid", "name", "type", "notnull", "dflt_value", "pk"]
        dicts = [dict(zip(schema_cols, row)) for row in columns]
        df_schema = pl.from_dicts(dicts, strict=False)
        
        st.dataframe(df_schema.select(["name", "type", "notnull", "pk"]))

        # Show row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        st.write(f"**Rows:** {row_count}")

        # Row limit management
        limit_key = f"db_limit_{table_name}"
        if limit_key not in st.session_state:
            st.session_state[limit_key] = 100
        
        current_limit = st.session_state[limit_key]

        # Load and show data
        st.markdown(f"#### Data from table: {table_name}")
        
        query = f"SELECT * FROM {table_name} LIMIT {current_limit}"
        
        try:
            df = pl.read_database(
                query,
                conn,
                infer_schema_length=None,
            )
        except Exception:
            # Fallback for mixed types (e.g. SQLite flexible typing)
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            dicts = [dict(zip(columns, row)) for row in data]
            df = pl.from_dicts(dicts, strict=False)
        
        st.dataframe(df, width="stretch")

        # Load more button
        if row_count > current_limit:
            if st.button(
                f"Load more rows (Showing {current_limit} of {row_count})",
                key=f"load_more_{table_name}",
                icon=":material/add:",
            ):
                st.session_state[limit_key] += 100
                st.rerun()
    else:
        st.write("No schema information available")
