"""
Parameter configuration logic for get started page.

This module handles parameter selection for comparison and analysis,
including parameter linking functionality.

parameter_config.py
├── handle_parameter_selection()            # Handle parameter selection UI and logic
├── _handle_single_parameter_linking()      # Handle parameter linking for single parameter
└── Parameter linking and filtering logic
"""

import streamlit as st


def handle_parameter_selection(selected_experiments: list[str], dataset_param: str) -> list[str] | None:
    """Handle parameter selection for comparison and analysis.

    Args:
        selected_experiments: List of selected experiment names.
        dataset_param: The parameter that defines dataset names.

    Returns:
        List of selected parameter names, or None if no parameters available.
    """
    if not selected_experiments or not dataset_param:
        return None

    # Use cached parameters instead of fetching again
    available_params = st.session_state['available_params'] if 'available_params' in st.session_state else []
    if not available_params:
        st.error("No parameters found in the selected experiments.")
        return None

    # Filter out the dataset parameter
    selectable_params = [p for p in available_params if p != dataset_param]

    if not selectable_params:
        st.warning("No parameters available for selection after filtering out the dataset parameter.")
        return []

    st.markdown("Select parameters to include in comparisons and analysis:")

    # Use pills for parameter selection
    selected_params = st.pills(
        "Parameters",
        options=selectable_params,
        default=st.session_state['selected_params'] if 'selected_params' in st.session_state else selectable_params,  # Default to all
        key="parameter_selector",
        selection_mode="multi",
    )

    st.session_state.selected_params = selected_params

    # Handle parameter linking for selected parameters
    if selected_params:
        st.markdown("### Parameter Linking (Optional)")
        # Initialize parameter value links if not already set
        if 'param_value_links' not in st.session_state:
            st.session_state.param_value_links = {}  # param -> {linked_param: [values]}

        for param in selected_params:
            _handle_single_parameter_linking(param, available_params, dataset_param)

    if not selected_params:
        st.warning("No parameters selected for comparison. Some analysis features may not work.")
    else:
        st.success(f"Selected {len(selected_params)} parameter{'s' if len(selected_params) != 1 else ''} for comparison")

    return selected_params


def _handle_single_parameter_linking(param: str, available_params: list[str], dataset_param: str) -> None:
    """Handle parameter linking UI for a selected parameter."""
    st.markdown(f"**{param}**")

    # Get other parameters for linking
    other_params = [p for p in available_params if p != param and p != dataset_param]

    if other_params:
        col_link_param, col_link_val = st.columns([1, 2])

        with col_link_param:
            link_param = st.selectbox(
                f"Link {param} to:",
                options=["None"] + other_params,
                key=f"link_param_{param}",
                help=f"Select parameter to link {param} to specific values of"
            )

        with col_link_val:
            if link_param != "None":
                param_values = st.session_state['available_param_values'] if 'available_param_values' in st.session_state else {}
                link_values = param_values[link_param] if link_param in param_values else []
                if link_values:
                    current_linked_values = []
                    if ('param_value_links' in st.session_state and
                        param in st.session_state['param_value_links'] and
                        link_param in st.session_state['param_value_links'][param]):
                        current_linked_values = st.session_state['param_value_links'][param][link_param]

                    selected_link_values = st.multiselect(
                        f"Values to link to:",
                        options=link_values,
                        default=current_linked_values,
                        key=f"link_values_{param}",
                        help=f"{param} will only appear when {link_param} has any of these values"
                    )

                    # Update session state
                    if param not in st.session_state.param_value_links:
                        st.session_state.param_value_links[param] = {}

                    if selected_link_values:
                        st.session_state.param_value_links[param][link_param] = selected_link_values
                    elif link_param in st.session_state.param_value_links[param]:
                        del st.session_state.param_value_links[param][link_param]
                        if not st.session_state.param_value_links[param]:
                            del st.session_state.param_value_links[param]
                else:
                    st.text("(No values found)")
            else:
                st.text("(Select parameter first)")
    else:
        st.text("No other parameters available for linking")

    st.divider()