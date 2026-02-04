"""
Reusable filter components for the sidebar.

This module provides Streamlit components for filtering experiment data
by various parameters like MPF, beta values, and dataset selection.

filters.py
├── Dataset selector component
├── MPF filter component
├── Beta filter component
├── P_inflate filter component
├── Display options component
└── Filter state management

Usage:
    from statflow.components.filters import (
        render_dataset_selector, render_mpf_filter
    )
"""

import streamlit as st

from statflow.config import SessionState

# Default values
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True


def render_pills_filter(
    label: str,
    options: list[str],
    session_key: str,
    selection_mode: str = "multi",
    help_text: str = "",
) -> tuple[str, ...] | None:
    """Generic pills filter widget.

    Args:
        label: Label for the filter widget.
        options: List of available options.
        session_key: Session state key for storing selections.
        selection_mode: "single" or "multi".
        help_text: Help text for the widget.

    Returns:
        Tuple of selected values, or None if all selected (multi) or no selection.
    """
    if not options:
        return None

    # Get default from session state
    default_values = (
        st.session_state[session_key]
        if session_key in st.session_state
        else tuple(options)
    )

    if selection_mode == "single":
        # For single select, if multiple are selected, take the first one
        if default_values and len(default_values) > 1:
            default_value = default_values[0]
        elif default_values:
            default_value = default_values[0]
        else:
            default_value = options[0] if options else None

        selected = st.pills(
            label,
            options=sorted(options),
            selection_mode="single",
            default=default_value,
            help=help_text,
        )

        if selected is None:
            return None
        return (selected,)

    else:  # multi
        selected = st.pills(
            label,
            options=sorted(options),
            selection_mode="multi",
            default=list(default_values) if default_values else options,
            help=help_text,
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(
    selection_mode: str = "single",
) -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = SessionState.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state["selected_dataset"]
            if "selected_dataset" in st.session_state
            else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = (
            st.session_state["selected_datasets"]
            if "selected_datasets" in st.session_state
            else available
        )
        selected = st.pills(
            "Select Datasets",
            options=available,
            selection_mode="multi",
            default=default_datasets,
            label_visibility="collapsed",
            key="dataset_selector_multi",
        )
        return tuple(selected) if selected else None
    else:
        raise ValueError(
            f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'."
        )
