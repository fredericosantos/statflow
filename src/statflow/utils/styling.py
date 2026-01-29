"""
Table styling utilities for formatting and highlighting.

This module provides functions for styling pandas DataFrames for display,
including highlighting significant values and formatting.

styling.py
├── RMSE table styling (style_rmse_table)
├── Statistical significance table styling (style_statistical_table)
└── General table formatting utilities

Usage:
    from statflow.utils.styling import (
        style_rmse_table, style_statistical_table
    )
"""

import pandas as pd
from typing import Dict
import streamlit as st


def style_statistical_table(stat_df: pd.DataFrame) -> pd.DataFrame:
    """Apply styling to statistical significance table to highlight significant p-values.

    Args:
        stat_df: Statistical significance DataFrame to style.

    Returns:
        Styled DataFrame.
    """

    def highlight_significance(val):
        """Highlight cell based on p-value significance."""
        if (
            isinstance(val, str)
            and val not in ["—", "N/A"]
            and "Dataset" not in str(val)
            and "ARC" not in str(val)
        ):
            if "***" in val:
                return "font-weight: bold; color: red;"
            elif "**" in val:
                return "font-weight: bold; color: orange;"
            elif "*" in val:
                return "font-weight: bold; color: blue;"
            elif val == "Best":
                return "font-weight: bold; color: green;"
        return ""

    return stat_df.style.map(highlight_significance)


def style_rmse_table(rmse_df: pd.DataFrame, significance_info: Dict) -> pd.DataFrame:
    """Apply styling to RMSE table to bold the best value and underline if significant.

    Args:
        rmse_df: RMSE DataFrame to style.
        significance_info: Dict with statistical significance per dataset.

    Returns:
        Styled DataFrame.
    """

    def highlight_min(row):
        # Check if Dataset column exists in this row
        if "Dataset" not in row:
            return [""] * len(row)

        dataset = row["Dataset"]
        styles = [""]
        values = []
        indices = []

        for i, (col, val) in enumerate(list(row.items())[1:], start=1):
            if isinstance(val, str) and " ± " in val:
                # Extract median value from "median ± std" format
                median_str = val.split(" ± ")[0]
                median_val = float(median_str)
                values.append(median_val)
                indices.append(i)
                styles.append("")
            else:
                styles.append("")

        if values:
            min_idx = values.index(min(values))
            actual_col_idx = indices[min_idx]
            styles[actual_col_idx] = "font-weight: bold;"

            # Check for statistical significance
            config_cols = list(row.index)[1:]
            if dataset in significance_info and min_idx < len(config_cols):
                config_name = config_cols[min_idx]
                if significance_info[dataset].get(config_name, False):
                    styles[actual_col_idx] += " text-decoration: underline;"

        return styles

    return rmse_df.style.apply(highlight_min, axis=1)


def handle_empty_data(data, message: str = "No data available.") -> bool:
    """Handle empty data with consistent messaging.

    Args:
        data: Data to check (DataFrame, list, etc.)
        message: Message to display if empty.

    Returns:
        True if data is empty, False otherwise.
    """
    if hasattr(data, 'is_empty') and data.is_empty():
        st.info(message)
        return True
    elif hasattr(data, '__len__') and len(data) == 0:
        st.info(message)
        return True
    return False


def show_user_warning(message: str, icon: str = "⚠️") -> None:
    """Show a user-friendly warning message.

    Args:
        message: Warning message to display.
        icon: Icon to prepend to message.
    """
    st.warning(f"{icon} {message}")


def show_user_success(message: str, icon: str = "✅") -> None:
    """Show a user-friendly success message.

    Args:
        message: Success message to display.
        icon: Icon to prepend to message.
    """
    st.success(f"{icon} {message}")