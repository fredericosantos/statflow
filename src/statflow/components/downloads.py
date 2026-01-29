"""
Download components for exporting data and results.

This module provides Streamlit components for downloading various data
formats including CSV, ZIP archives, LaTeX, and Markdown.

downloads.py
├── CSV download component
├── ZIP download component
├── LaTeX table download component
├── Markdown table download component
└── Download button helpers

Usage:
    from statflow.components.downloads import (
        render_csv_download, render_zip_download
    )
"""

import streamlit as st
import pandas as pd
from typing import Optional, Any, Tuple

from statflow.utils.export import (
    create_fitness_zip, export_table_to_csv, export_table_to_latex, export_table_to_markdown
)


def render_csv_download(df: pd.DataFrame, filename: str = "data.csv", label: str = "Download CSV") -> None:
    """Render CSV download button.

    Args:
        df: DataFrame to download.
        filename: Default filename.
        label: Button label.
    """
    if df.empty:
        st.warning("No data available for download.")
        return

    csv_data = export_table_to_csv(df, filename)

    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key=f"download_csv_{filename}"
    )


def render_zip_download(
    selected_mpf_values: Optional[Tuple[str, ...]] = None,
    selected_beta_values: Optional[Tuple[str, ...]] = None,
    selected_pinflate_values: Optional[Tuple[str, ...]] = None,
) -> None:
    """Render ZIP download button for raw fitness data.

    Args:
        selected_mpf_values: Selected MPF values to filter.
        selected_beta_values: Selected beta values to filter.
        selected_pinflate_values: Selected p_inflate values to filter.
    """
    # Check if ZIP creation was triggered
    if st.session_state.get('zip_clicked', False):
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, text="Starting ZIP creation...")

        zip_data = create_fitness_zip(
            selected_mpf_values, selected_beta_values, selected_pinflate_values, progress_bar
        )

        progress_placeholder.empty()

        if zip_data:
            st.download_button(
                label="📦 Download ZIP (Raw Fitness Data)",
                data=zip_data,
                file_name="fitness_data.zip",
                mime="application/zip",
                key="download_zip"
            )
        else:
            st.error("No data available for ZIP download.")

        # Reset the trigger
        st.session_state.zip_clicked = False
    else:
        # Show the trigger button
        if st.button("📦 Prepare ZIP Download", key="prepare_zip"):
            st.session_state.zip_clicked = True
            st.rerun()


def render_latex_download(
    df: pd.DataFrame,
    filename: str = "table.tex",
    caption: str = "",
    label: str = "Download LaTeX"
) -> None:
    """Render LaTeX table download button.

    Args:
        df: DataFrame to convert to LaTeX.
        filename: Default filename.
        caption: Table caption.
        label: Button label.
    """
    if df.empty:
        st.warning("No data available for LaTeX export.")
        return

    latex_data = export_table_to_latex(df, caption, label)

    st.download_button(
        label=label,
        data=latex_data,
        file_name=filename,
        mime="application/x-tex",
        key=f"download_latex_{filename}"
    )


def render_markdown_download(
    df: pd.DataFrame,
    filename: str = "table.md",
    label: str = "Download Markdown"
) -> None:
    """Render Markdown table download button.

    Args:
        df: DataFrame to convert to Markdown.
        filename: Default filename.
        label: Button label.
    """
    if df.empty:
        st.warning("No data available for Markdown export.")
        return

    markdown_data = export_table_to_markdown(df)

    st.download_button(
        label=label,
        data=markdown_data,
        file_name=filename,
        mime="text/markdown",
        key=f"download_md_{filename}"
    )


def render_download_section(
    df: pd.DataFrame,
    prefix: str = "data",
    include_zip: bool = False,
    selected_mpf_values: Optional[Tuple[str, ...]] = None,
    selected_beta_values: Optional[Tuple[str, ...]] = None,
    selected_pinflate_values: Optional[Tuple[str, ...]] = None,
) -> None:
    """Render a complete download section with multiple format options.

    Args:
        df: DataFrame to download.
        prefix: Prefix for filenames.
        include_zip: Whether to include ZIP download option.
        selected_mpf_values: Selected MPF values for ZIP filtering.
        selected_beta_values: Selected beta values for ZIP filtering.
        selected_pinflate_values: Selected p_inflate values for ZIP filtering.
    """
    st.markdown("### 📥 Downloads")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_csv_download(df, f"{prefix}.csv", "📄 CSV")

    with col2:
        render_latex_download(df, f"{prefix}.tex", "LaTeX Table", "📖 LaTeX")

    with col3:
        render_markdown_download(df, f"{prefix}.md", "📝 Markdown")

    if include_zip:
        st.markdown("---")
        render_zip_download(selected_mpf_values, selected_beta_values, selected_pinflate_values)