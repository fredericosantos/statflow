"""
Table display components with download functionality.

This module provides Streamlit components for rendering data tables
with integrated download buttons for various formats.

tables.py
├── Table with downloads component
├── Copy to clipboard helper
└── Table rendering helpers

Usage:
    from statflow.components.tables import (
        render_table_with_downloads
    )
"""

import streamlit as st
import pandas as pd
import polars as pl
from pandas.io.formats.style import Styler

from statflow.utils.export import (
    export_table_to_csv, export_table_to_latex, export_table_to_markdown
)


def render_table_with_downloads(
    table_data: pl.DataFrame | pd.DataFrame | object,
    table_title: str,
    filename_prefix: str,
    description: str = "",
    latex_rotation_angle: int = 90,
    latex_font_size: str = "normalsize"
) -> None:
    """Render a comparison table with download buttons and dataframe display.

    Args:
        table_data: DataFrame or Styler containing the table data to display
        table_title: Title to display for the table
        filename_prefix: Prefix for download filenames
        description: Optional description text to show below the table
        latex_rotation_angle: Rotation angle for dataset names in LaTeX (default 90)
        latex_font_size: Font size for LaTeX table (default "normalsize")
    """
    # Streamlit supports Polars DataFrames natively

    # Handle both DataFrame and Styler objects
    if isinstance(table_data, Styler):
        table_df = table_data.data
        display_df = table_data
    else:
        table_df = table_data
        display_df = table_data
    
    if not table_df.empty:
        # Get dataset renames from session state
        from statflow.config import DEFAULT_DATASET_RENAMES
        dataset_renames = st.session_state.get('dataset_renames', DEFAULT_DATASET_RENAMES)
        
        # Generate LaTeX once for both download and copy
        latex_data = export_table_to_latex(
            table_df, table_title, f"{filename_prefix}-table",
            rotation_angle=latex_rotation_angle,
            font_size=latex_font_size,
            dataset_renames=dataset_renames
        )
        
        # Table with download buttons - added extra column for copy button
        col1, col2, col3, col4, col5 = st.columns([5, 1, 1, 1, 1])
        with col1:
            st.markdown(f"#### {table_title}")
        with col2:
            csv_data = export_table_to_csv(table_df, f"{filename_prefix}.csv")
            st.download_button(
                label="CSV",
                icon=":material/table:",
                data=csv_data,
                file_name=f"{filename_prefix}.csv",
                mime="text/csv",
                key=f"{filename_prefix}_csv",
            )
        with col3:
            st.download_button(
                label="LaTeX",
                icon=":material/code:",
                data=latex_data,
                file_name=f"{filename_prefix}.tex",
                mime="text/plain",
                key=f"{filename_prefix}_latex",
            )
        with col4:
            # Copy LaTeX button using popover with code block (has built-in copy)
            with st.popover("Copy", icon=":material/content_copy:", help="Click to view LaTeX code, then use the copy button"):
                st.code(latex_data, language="latex")
        with col5:
            markdown_data = export_table_to_markdown(table_df)
            st.download_button(
                label="MD",
                icon=":material/description:",
                data=markdown_data,
                file_name=f"{filename_prefix}.md",
                mime="text/markdown",
                key=f"{filename_prefix}_markdown",
            )

        st.dataframe(display_df, width="stretch", height="content")

        if description:
            st.markdown(description)
    else:
        st.warning(f"No {table_title.lower()} data available.")