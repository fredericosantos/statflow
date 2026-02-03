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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True


def render_pills_filter(
    label: str,
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
    session_key: str,
    selection_mode: str = "multi",
    help_text: str = ""
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
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
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
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
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
            selection_mode="single",
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
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
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str, min_val: float, max_val: float, session_key: str, help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = (
        st.session_state[session_key]
        if session_key in st.session_state
        else (min_val, max_val)
    )

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}",
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True,
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = (
        st.session_state[session_key] if session_key in st.session_state else None
    )
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}",
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state["show_mean"]
            if "show_mean" in st.session_state
            else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state["use_custom_colors"]
            if "use_custom_colors" in st.session_state
            else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state["graph_width"]
                if "graph_width" in st.session_state
                else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels",
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state["graph_height"]
                if "graph_height" in st.session_state
                else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels",
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = (
                st.session_state["points_display"]
                if "points_display" in st.session_state
                else DEFAULT_POINTS_DISPLAY
            )
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state["show_error_bars"]
                if "show_error_bars" in st.session_state
                else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str],
    available_beta: list[str],
    available_pinflate: list[str],
    include_dataset_selector: bool = False,
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
    include_dataset_selector: bool = False
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str],
    available_beta: list[str],
    available_pinflate: list[str],
    include_dataset_selector: bool = False,
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES

    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = (
            st.session_state["dataset_renames"]
            if "dataset_renames" in st.session_state
            else {}
        )
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = (
                current_renames[dataset] if dataset in current_renames else dataset
            )
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = (
                    current_renames[dataset] if dataset in current_renames else dataset
                )
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
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

from statflow.config import State, DEFAULT_DATASET_RENAMES

# Default values (inline, removed from config.py)
DEFAULT_SHOW_MEAN = False
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_SHOW_ERROR_BARS = True

# Default values (inline, removed from config.py for simplicity)
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
    help_text: str = ""
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
    default_values = st.session_state[session_key] if session_key in st.session_state else tuple(options)

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
            help=help_text
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
            help=help_text
        )

        # Return None if all are selected (no filtering)
        if set(selected) == set(options):
            return None
        return tuple(selected)


def render_dataset_selector(selection_mode: str = "single") -> str | tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    available = State.get("available_datasets", [])
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=available,
            selection_mode="single",
            default=st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else (available[0] if available else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state['selected_datasets'] if 'selected_datasets' in st.session_state else available
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
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: list[str]) -> tuple[str, ...] | None:
    """Render MPF (Mutation Pool Factor) filter widget.

    Args:
        available_mpf: List of available MPF values.

    Returns:
        Tuple of selected MPF values, or None if all selected.
    """
    return render_pills_filter(
        "ARC MPF Values",
        available_mpf,
        "selected_mpf_values",
        selection_mode="single",
        help_text="Select Mutation Pool Factor value for ARC variant."
    )


def render_beta_filter(available_beta: list[str]) -> tuple[str, ...] | None:
    """Render beta filter widget.

    Args:
        available_beta: List of available beta values.

    Returns:
        Tuple of selected beta values, or None if all selected.
    """
    return render_pills_filter(
        "ARC Beta Values",
        available_beta,
        "selected_beta_values",
        selection_mode="multi",
        help_text="Select beta values for ARC variant. Leave all selected to include all values."
    )


def render_pinflate_filter(available_pinflate: list[str]) -> tuple[str, ...] | None:
    """Render P_inflate filter widget for SLIM-GSGP.

    Args:
        available_pinflate: List of available P_inflate values.

    Returns:
        Tuple of selected P_inflate values, or None if all selected.
    """
    return render_pills_filter(
        "SLIM-GSGP P_inflate Values",
        available_pinflate,
        "selected_pinflate_values",
        selection_mode="single",
        help_text="Select P_inflate value for SLIM-GSGP variant."
    )


# Generic filter functions for consolidation
def render_range_filter(
    label: str,
    min_val: float,
    max_val: float,
    session_key: str,
    help_text: str = ""
) -> tuple[float, float]:
    """Render a generic range slider filter.

    Args:
        label: Label for the filter.
        min_val: Minimum value for the range.
        max_val: Maximum value for the range.
        session_key: Session state key for storing the range.
        help_text: Help text for the filter.

    Returns:
        Tuple of (min_selected, max_selected).
    """
    # Get current range from session state
    current_range = st.session_state[session_key] if session_key in st.session_state else (min_val, max_val)

    selected_range = st.slider(
        label,
        min_val,
        max_val,
        current_range,
        help=help_text,
        key=f"slider_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = selected_range

    return selected_range


def render_multiselect_filter(
    label: str,
    options: list[str],
    session_key: str,
    help_text: str = "",
    default_all: bool = True
) -> tuple[str, ...] | None:
    """Render a generic multiselect filter.

    Args:
        label: Label for the filter.
        options: List of available options.
        session_key: Session state key for storing selections.
        help_text: Help text for the filter.
        default_all: Whether to select all options by default.

    Returns:
        Tuple of selected values, or None if all selected (when default_all=True).
    """
    # Get current selection from session state
    current_selection = st.session_state[session_key] if session_key in st.session_state else None
    if current_selection is None:
        current_selection = tuple(options) if default_all else ()

    selected = st.multiselect(
        label,
        options=sorted(options),
        default=list(current_selection),
        help=help_text,
        key=f"multiselect_{session_key}"
    )

    # Store in session state
    st.session_state[session_key] = tuple(selected)

    # Return None if all are selected (no filtering needed)
    if default_all and set(selected) == set(options):
        return None
    return tuple(selected)


def render_display_options() -> tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state['show_mean'] if 'show_mean' in st.session_state else DEFAULT_SHOW_MEAN,
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> tuple[int, int, str, bool]:
    """Render graph configuration widgets.

    Returns:
        Tuple of (width, height, points_display, show_error_bars).
    """
    with st.expander("⚙️ Graph config", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            width = st.slider(
                "Graph Width",
                min_value=400,
                max_value=1200,
                value=st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY
            if current_points_display == "outliers":
                current_points_display = "Outliers only"
            elif current_points_display not in ["Hide", "Outliers only", "All points"]:
                current_points_display = "Outliers only"
            
            points_display = st.pills(
                "Show points in boxplot",
                options=["Hide", "Outliers only", "All points"],
                selection_mode="single",
                default=current_points_display,
                help="Choose whether to display data points on the boxplot",
            )

            show_error_bars = st.checkbox(
                "Show Error Bars (Pareto Front)",
                value=st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
                help="Toggle to display standard deviation error bars in Pareto front plot",
            )

            # Map radio selection to Plotly parameter
            points_param = {
                "Hide": False,
                "Outliers only": "outliers",
                "All points": "all",
            }[points_display]

    return width, height, points_param, show_error_bars


def render_global_filters(
    available_mpf: list[str], 
    available_beta: list[str], 
    available_pinflate: list[str],
    include_dataset_selector: bool = False
) -> tuple[
    str | None, tuple[str, ...] | None, tuple[str, ...] | None, tuple[str, ...] | None
]:
    """Render global filters in main content area.
    
    Args:
        available_mpf: Available MPF values.
        available_beta: Available beta values.
        available_pinflate: Available P_inflate values.
        include_dataset_selector: Whether to include dataset selector.
        
    Returns:
        Tuple of (dataset_name, selected_mpf, selected_beta, selected_pinflate)
    """
    dataset_name = None
    
    if include_dataset_selector:
        st.markdown("### Dataset Selection")
        dataset_name = render_dataset_selector()
        st.markdown("---")
    
    st.markdown("### Global Filters")
    
    # MPF filter
    selected_mpf_values = render_mpf_filter(available_mpf)
    
    # Beta filter  
    selected_beta_values = render_beta_filter(available_beta)
    
    # P_inflate filter
    selected_pinflate_values = render_pinflate_filter(available_pinflate)
    
    st.markdown("---")
    
    return dataset_name, selected_mpf_values, selected_beta_values, selected_pinflate_values


def render_filter_summary() -> None:
    """Render summary of applied filters."""
    st.markdown("---")
    st.markdown("### Filter Applied")
    st.code(
        """
crossover_prob = 0.0
mutation_prob = 1.0
activation_fn_init = IDENTITY
best_test_fitness > 0
scale_dataset = True
scale_target = True
arc_v2 = True
        """,
        language="sql",
    )

    st.markdown("---")
    st.markdown("### Grouping Parameters")
    st.markdown("""
**GSGP:**
- GSGP-std (use_oms=False)
- GSGP-OMS (use_oms=True)

**SLIM-GSGP:**
- P_inflate value

**ARC:**
- Beta (β)
- Mutation Pool Factor
        """)


def render_dataset_names_expander() -> None:
    """Render dataset names customization in an expander for sidebar.
    
    Updates `session_state['dataset_renames']` with custom display names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    available_datasets = State.get("available_datasets", [])
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else {}
        current_renames = DEFAULT_DATASET_RENAMES.copy()
        current_renames.update(saved_renames)  # User customizations override defaults
        st.session_state['dataset_renames'] = current_renames  # Ensure merged version is in state
        
        # Reset button
        if st.button("Reset to Defaults", icon=":material/restart_alt:", key="reset_dataset_names"):
            st.session_state['dataset_renames'] = DEFAULT_DATASET_RENAMES.copy()
            st.rerun()
        
        # Show text inputs for blackbox datasets (most need renaming)
        st.markdown("**Blackbox:**")
        for dataset in [d for d in available_datasets if d.startswith("blackbox_")]:
            current_name = current_renames[dataset] if dataset in current_renames else dataset
            new_name = st.text_input(
                dataset,
                value=current_name,
                key=f"sidebar_rename_{dataset}",
                label_visibility="visible"
            )
            if new_name != current_name:
                current_renames[dataset] = new_name
                st.session_state['dataset_renames'] = current_renames
        
        # Collapsible for real-life datasets
        with st.popover("Real-life datasets"):
            for dataset in [d for d in available_datasets if not d.startswith("blackbox_")]:
                current_name = current_renames[dataset] if dataset in current_renames else dataset
                new_name = st.text_input(
                    dataset,
                    value=current_name,
                    key=f"sidebar_rename_{dataset}",
                    label_visibility="visible"
                )
                if new_name != current_name:
                    current_renames[dataset] = new_name
                    st.session_state['dataset_renames'] = current_renames


def get_dataset_display_name(dataset: str) -> str:
    """Get the display name for a dataset from session state.
    
    Args:
        dataset: Original dataset name.
        
    Returns:
        Display name (renamed if customized, otherwise original).
    """
    from statflow.config import DEFAULT_DATASET_RENAMES

    renames = (
        st.session_state["dataset_renames"]
        if "dataset_renames" in st.session_state
        else DEFAULT_DATASET_RENAMES
    )
    rename_entry = renames.get(dataset)
    if rename_entry is None:
        return dataset
    elif isinstance(rename_entry, dict):
        return rename_entry.get("display_name", dataset)
    else:
        return rename_entry