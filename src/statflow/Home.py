"""
Home page for the GSGP-ARC multi-page Streamlit application.

This is the main entry point that provides experiment and dataset selection for analysis.

Home.py
├── Server status check
├── Experiment selection with pills
├── Dataset selection based on experiments
└── Session state management

Usage:
    uv run streamlit run src/statflow/Home.py --server.address 0.0.0.0
"""

import streamlit as st
import polars as pl

from statflow.config import (
    MLFLOW_TRACKING_URI,
    initialize_session_state,
    setup_sidebar,
)
from statflow.utils.mlflow_client import get_experiment_names, get_metadata_from_experiments, get_datasets_from_experiments
from streamlit_sortables import sort_items

# Initialize session state first
initialize_session_state()

# Set MLflow tracking URI globally
import mlflow
mlflow.set_tracking_uri(st.session_state['mlflow_server_url'])

st.set_page_config(
    page_title=st.session_state["app_name"],
    page_icon=":material/home:",
    layout="wide",
)


def main():
    # Setup sidebar with server status
    setup_sidebar()

    # Experiment Selection
    experiment_names = get_experiment_names()
    if experiment_names:
        with st.expander("Initial Setup", expanded=True, icon=":material/experiment:"):
            selected_experiments = st.pills(
                "Select Experiments from MLFlow",
                options=experiment_names,
                default=st.session_state['selected_experiments'],
                key="experiment_selector",
                selection_mode="multi",
            )
            
            # Check if selection changed
            if selected_experiments != st.session_state.selected_experiments:
                if selected_experiments:
                    with st.spinner("Loading experiment metadata..."):
                        metadata = get_metadata_from_experiments(tuple(selected_experiments))
                        st.session_state.available_params = metadata["params"]
                        st.session_state.available_param_values = metadata["param_values"]
                        st.session_state.available_metrics = metadata["metrics"]
                else:
                    st.session_state.available_params = []
                    st.session_state.available_param_values = {}
                    st.session_state.available_metrics = []
                
                st.session_state.selected_experiments = selected_experiments
            
            # Dataset Parameter Selection
            if selected_experiments:
                available_params = st.session_state.available_params
                if available_params:
                    # Suggest default based on common patterns
                    suggested_default = 'dataset_name'
                    if 'dataset_name' in available_params:
                        suggested_default = 'dataset_name'
                    elif 'dataset' in available_params:
                        suggested_default = 'dataset'
                    elif any('dataset' in p.lower() and 'name' in p.lower() for p in available_params):
                        # Find one that contains both
                        for p in available_params:
                            if 'dataset' in p.lower() and 'name' in p.lower():
                                suggested_default = p
                                break
                    
                    dataset_param = st.selectbox(
                        "Select parameter that defines dataset names",
                        options=available_params,
                        index=available_params.index(suggested_default) if suggested_default in available_params else 0,
                        key="dataset_param_selector",
                    )
                    st.session_state.dataset_param = dataset_param
                    st.warning("⚠️ Runs without a value in this field will be filtered out")
                else:
                    st.error("No parameters found in the selected experiments.")
                    st.session_state.dataset_param = None  # fallback
            
            # Dataset Selection
            if selected_experiments and st.session_state['dataset_param']:
                available_datasets = get_datasets_from_experiments(tuple(selected_experiments), st.session_state.dataset_param)
                st.session_state.available_datasets = available_datasets
                if available_datasets:
                    selected_datasets = st.pills(
                        "Select Datasets",
                        options=available_datasets,
                        default=st.session_state['selected_datasets'] or available_datasets,  # Default to all available
                        key="dataset_selector",
                        selection_mode="multi",
                    )
                    st.session_state.selected_datasets = selected_datasets
                    
                    # Dataset Ordering
                    if selected_datasets:
                        st.space()
                        st.markdown("Order Selected Datasets")
                        # Create a unique key based on the current dataset selection
                        sort_key = f"dataset_order_{len(selected_datasets)}_{hash(tuple(sorted(selected_datasets)))}"
                        ordered_datasets = sort_items(
                            selected_datasets,
                            key=sort_key
                        )
                        st.session_state.selected_datasets = ordered_datasets
                else:
                    st.session_state.selected_datasets = []
                    st.info("No datasets found in the selected experiments.")
            else:
                st.info("Please select at least one experiment to see available datasets.")
        
            # Parameter Selection for Comparison
            if selected_experiments and st.session_state['dataset_param']:
                with st.expander("Parameter Selection", expanded=False, icon=":material/tune:"):
                    available_params = get_parameters_from_experiments(tuple(selected_experiments))
                    if available_params:
                        st.markdown("Select parameters to include in comparisons and analysis:")
                        
                        # Initialize selected parameters if not already set
                        if 'selected_params' not in st.session_state:
                            # Default to all parameters except the dataset parameter
                            dataset_param = st.session_state['dataset_param']
                            st.session_state.selected_params = [p for p in available_params if p != dataset_param]
                        
                        selected_params = []
                        # Initialize parameter value links if not already set
                        if 'param_value_links' not in st.session_state:
                            st.session_state.param_value_links = {}  # param -> {linked_param: [values]}
                        
                        for param in available_params:
                            # Skip the dataset parameter since it's used for grouping
                            if param == st.session_state['dataset_param']:
                                continue
                                
                            # Create columns for the parameter row (vertically centered)
                            col_cb, col_link_param, col_link_val = st.columns([1, 2, 3])
                            
                            with col_cb:
                                st.write("")  # Spacer for vertical centering
                                is_selected = st.checkbox(
                                    f"{param}",
                                    value=param in st.session_state['selected_params'],
                                    key=f"param_{param}",
                                )
                                if is_selected:
                                    selected_params.append(param)
                                st.write("")  # Spacer for vertical centering
                            
                            if is_selected:
                                # Get other parameters for linking
                                other_params = [p for p in available_params if p != param and p != st.session_state['dataset_param']]
                                
                                with col_link_param:
                                    st.markdown("**Link to other parameter**")
                                    if other_params:
                                        link_param = st.selectbox(
                                            f"Link {param} to:",
                                            options=["None"] + other_params,
                                            key=f"link_param_{param}",
                                            label_visibility="collapsed",
                                            help=f"Select parameter to link {param} to specific values of"
                                        )
                                    else:
                                        st.text("(No other params)")
                                        link_param = "None"
                                
                                with col_link_val:
                                    st.markdown("**Values to link to**")
                                    if link_param != "None":
                                        link_values = st.session_state.available_param_values.get(link_param, [])
                                        if link_values:
                                            current_linked_values = []
                                            if param in st.session_state.param_value_links and link_param in st.session_state.param_value_links[param]:
                                                current_linked_values = st.session_state.param_value_links[param][link_param]
                                            
                                            selected_link_values = st.multiselect(
                                                f"Link {param} when {link_param} has these values:",
                                                options=link_values,
                                                default=current_linked_values,
                                                key=f"link_values_{param}",
                                                label_visibility="collapsed",
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
                                        st.text("(Select param first)")
                            else:
                                # Parameter not selected - show empty columns
                                with col_link_param:
                                    st.text("")
                                with col_link_val:
                                    st.text("")
                        
                        st.session_state.selected_params = selected_params
                        
                        if not selected_params:
                            st.warning("No parameters selected for comparison. Some analysis features may not work.")
                        else:
                            st.success(f"Selected {len(selected_params)} parameter{'s' if len(selected_params) != 1 else ''} for comparison")
                    else:
                        st.error("No parameters found in the selected experiments.")
            else:
                st.warning("No experiments found. Please ensure the MLflow server is running and has experiments.")
    else:
        st.info(f"MLflow server must be running at {st.session_state['mlflow_server_url']} to continue.")


if __name__ == "__main__":
    main()
