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
from typing import List, Optional, Tuple

from statflow.config import DEFAULT_DATASET_RENAMES, get_all_datasets


def render_pills_filter(
    label: str,
    options: List[str],
    session_key: str,
    selection_mode: str = "multi",
    help_text: str = ""
) -> Optional[Tuple[str, ...]]:
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
    default_values = st.session_state.get(session_key, tuple(options))

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


def render_dataset_selector(selection_mode: str = "single") -> str | Tuple[str, ...] | None:
    """Render dataset selection widget.

    Args:
        selection_mode: Either "single" or "multi" for selection mode.

    Returns:
        Selected dataset name(s) - string for single mode, tuple for multi mode, None if nothing selected.
    """
    if selection_mode == "single":
        return st.pills(
            "Select Dataset",
            options=get_all_datasets(),
            selection_mode="single",
            default=st.session_state.get('selected_dataset', get_all_datasets()[0] if get_all_datasets() else None),
            label_visibility="collapsed",
            key="dataset_selector_single",
        )
    elif selection_mode == "multi":
        default_datasets = st.session_state.get('selected_datasets', get_all_datasets())
        selected = st.pills(
            "Select Datasets",
            options=get_all_datasets(),
            selection_mode="multi",
            default=default_datasets,
            label_visibility="collapsed",
            key="dataset_selector_multi",
        )
        return tuple(selected) if selected else None
    else:
        raise ValueError(f"Invalid selection_mode: {selection_mode}. Must be 'single' or 'multi'.")


def render_mpf_filter(available_mpf: List[str]) -> Optional[Tuple[str, ...]]:
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


def render_beta_filter(available_beta: List[str]) -> Optional[Tuple[str, ...]]:
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


def render_pinflate_filter(available_pinflate: List[str]) -> Optional[Tuple[str, ...]]:
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


def render_display_options() -> Tuple[bool, bool]:
    """Render display options widgets.

    Returns:
        Tuple of (show_mean, use_custom_colors).
    """
    col1, col2 = st.columns(2)

    with col1:
        show_mean = st.checkbox(
            "Show Mean (default: Median)",
            value=st.session_state.get('show_mean', False),
            help="Toggle to display mean instead of median in statistics",
        )

    with col2:
        use_custom_colors = st.checkbox(
            "Use Custom Colors",
            value=st.session_state.get('use_custom_colors', True),
            help="Toggle to use custom color palette or default Streamlit colors",
        )

    return show_mean, use_custom_colors


def render_graph_config() -> Tuple[int, int, str, bool]:
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
                value=st.session_state.get('graph_width', 800),
                step=50,
                help="Width of the graphs in pixels"
            )

            height = st.slider(
                "Graph Height",
                min_value=300,
                max_value=800,
                value=st.session_state.get('graph_height', 600),
                step=50,
                help="Height of the graphs in pixels"
            )

        with col2:
            # Handle legacy session state values
            current_points_display = st.session_state.get('points_display', "Outliers only")
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
                value=st.session_state.get('show_error_bars', True),
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
    available_mpf: List[str], 
    available_beta: List[str], 
    available_pinflate: List[str],
    include_dataset_selector: bool = False
) -> Tuple[
    Optional[str], Optional[Tuple[str, ...]], Optional[Tuple[str, ...]], Optional[Tuple[str, ...]]
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
    from statflow.config import DEFAULT_DATASET_RENAMES, get_all_datasets
    
    available_datasets = get_all_datasets()
    
    with st.expander("📝 Dataset Names", expanded=False):
        st.caption("Customize display names for LaTeX export")
        
        # Get current renames from session state, merged with defaults
        saved_renames = st.session_state.get('dataset_renames', {})
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
            current_name = current_renames.get(dataset, dataset)
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
                current_name = current_renames.get(dataset, dataset)
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
    
    renames = st.session_state.get('dataset_renames', DEFAULT_DATASET_RENAMES)
    return renames.get(dataset, dataset)