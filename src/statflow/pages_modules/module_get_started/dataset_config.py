"""
Dataset configuration logic for get started page.

This module handles dataset parameter selection, dataset discovery,
and dataset ordering functionality.

dataset_config.py
├── handle_dataset_parameter_selection()    # Handle dataset parameter selection UI
├── handle_dataset_selection()              # Handle dataset selection and ordering
├── _find_dataset_parameter_suggestion()    # Find best dataset parameter suggestion
├── _get_available_datasets()               # Get available datasets based on mode
├── _render_dataset_selector()              # Render dataset selection UI
├── _render_dataset_ordering()              # Render dataset ordering UI
├── render_rename_datasets_ui()             # Render dataset renaming UI
├── _search_datasets()                      # Perform dataset search
└── _render_multiple_datasets_ui()          # Render multiple datasets mode UI
"""

import streamlit as st
import mlflow
from streamlit_sortables import sort_items

from .constants import DatasetParamMode


def _sort_datasets(datasets: list[str]) -> list[str]:
    """Sort datasets by their parameter values, handling numeric values correctly."""
    def sort_key(value: str) -> tuple[bool, float | str]:
        try:
            # Try to convert to float for numeric sorting
            return (False, float(value))
        except ValueError:
            # Fall back to string sorting
            return (True, value.lower())
    
    return sorted(datasets, key=sort_key)


def _get_available_datasets(
    selected_experiments: list[str], dataset_param: DatasetParamMode
) -> list[str]:
    """Get available datasets based on the dataset parameter mode."""
    if dataset_param == DatasetParamMode.EXPERIMENT_NAMES_AS_DATASETS:
        return selected_experiments
    elif dataset_param == DatasetParamMode.SINGLE_DATASET_MODE:
        return ["default"]
    else:
        # Multiple datasets mode
        if st.session_state.get("datasets_searched", False):
            return _sort_datasets(list(st.session_state.accumulated_datasets))
        else:
            return []


def _render_dataset_selector(available_datasets: list[str]) -> list[str]:
    """Render the dataset selector UI."""
    if not available_datasets:
        st.info("No datasets available.")
        return []

    selected_datasets = st.pills(
        "Select Datasets",
        options=available_datasets,
        default=st.session_state.get("selected_datasets", available_datasets),
        key="dataset_selector",
        selection_mode="multi",
    )
    st.session_state.selected_datasets = selected_datasets
    return selected_datasets


def _render_dataset_ordering(selected_datasets: list[str]) -> list[str]:
    """Render the dataset ordering UI."""
    if not selected_datasets:
        return selected_datasets

    st.space()
    st.markdown("Order Selected Datasets")
    sort_key = f"dataset_order_{len(selected_datasets)}_{hash(tuple(sorted(selected_datasets)))}"
    ordered_datasets = sort_items(selected_datasets, key=sort_key)
    st.session_state.selected_datasets = ordered_datasets
    return ordered_datasets


def render_rename_datasets_ui(selected_datasets: list[str]) -> None:
    """Render the UI for renaming datasets with display and LaTeX names.
    
    Stores mapping in st.session_state['dataset_renames'] as:
    {original_name: {"display_name": str, "latex_name": str}}
    
    Does NOT modify selected_datasets - they remain as original names.
    """
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    # Get current renames, merging defaults with user customizations
    saved_renames: dict = st.session_state.get('dataset_renames', {})
    current_renames = DEFAULT_DATASET_RENAMES.copy()
    current_renames.update(saved_renames)
    
    with st.expander("Rename Datasets", expanded=False, icon=":material/edit:"):
        st.markdown("Customize dataset names for display and LaTeX export:")
        
        # Header row
        cols = st.columns([2, 2, 2, 1])
        cols[0].markdown("**Original**")
        cols[1].markdown("**Display Name**")
        cols[2].markdown("**LaTeX Name**")
        cols[3].markdown("**Preview**")
        
        for dataset in selected_datasets:
            # Get current values from mapping
            entry = current_renames.get(dataset)
            if isinstance(entry, dict):
                current_display = entry.get("display_name", dataset)
                current_latex = entry.get("latex_name", dataset)
            elif isinstance(entry, str):
                # Backward compatibility: single string means both
                current_display = entry
                current_latex = entry
            else:
                current_display = dataset
                current_latex = dataset
            
            cols = st.columns([2, 2, 2, 1])
            
            # Column 1: Original name (read-only)
            cols[0].text(dataset)
            
            # Column 2: Display name input
            new_display = cols[1].text_input(
                "Display",
                value=current_display,
                key=f"rename_display_{dataset}",
                label_visibility="collapsed"
            )
            
            # Column 3: LaTeX name input
            new_latex = cols[2].text_input(
                "LaTeX",
                value=current_latex,
                key=f"rename_latex_{dataset}",
                label_visibility="collapsed"
            )
            
            # Column 4: LaTeX preview
            cols[3].latex(new_latex)
            
            # Update mapping if changed
            if new_display != current_display or new_latex != current_latex:
                current_renames[dataset] = {
                    "display_name": new_display,
                    "latex_name": new_latex
                }
        
        st.session_state['dataset_renames'] = current_renames



def _search_datasets(
    selected_experiments: list[str], dataset_param: str, batch_size: int
) -> int:
    """Search for datasets in the selected experiments."""
    initial_count = len(st.session_state.accumulated_datasets)
    datasets = st.session_state.accumulated_datasets.copy()
    mlflow.set_tracking_uri(
        st.session_state.get("mlflow_server_url", "http://localhost:5000")
    )
    client = mlflow.tracking.MlflowClient()

    for exp_name in selected_experiments:
        filter_string = ""
        if exp_name in st.session_state.last_start_time_per_exp:
            last_time = st.session_state.last_start_time_per_exp[exp_name]
            filter_string = f"attributes.start_time < {last_time}"

        exp = client.get_experiment_by_name(exp_name)
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=filter_string,
                max_results=batch_size,
                order_by=["attributes.start_time DESC"],
            )
            for run in runs:
                if dataset_param in run.data.params:
                    value = run.data.params[dataset_param]
                    if value and value.strip():
                        datasets.add(value)
            if runs:
                st.session_state.last_start_time_per_exp[exp_name] = runs[
                    -1
                ].info.start_time

    st.session_state.accumulated_datasets = datasets
    st.session_state.datasets_searched = True
    new_count = len(datasets) - initial_count
    return new_count


def _render_multiple_datasets_ui(selected_experiments: list[str]) -> list[str] | None:
    """Render the UI for multiple datasets mode."""
    batch_size = 5000

    if st.session_state.get("is_searching", False):
        with st.spinner("Searching for datasets..."):
            new_count = _search_datasets(
                selected_experiments, st.session_state.dataset_param, batch_size
            )
        st.session_state.is_searching = False
    else:
        new_count = 0
    col1, col2 = st.columns([3, 2], vertical_alignment="bottom")
    with col1:
        available_params = st.session_state.get("available_params", [])
        dataset_param = st.selectbox(
            "Select parameter that defines dataset names",
            options=available_params,
            index=available_params.index(st.session_state.dataset_param)
            if st.session_state.dataset_param in available_params
            else 0,
            key="dataset_param_selector",
        )
        if (
            "previous_dataset_param" not in st.session_state
            or st.session_state.previous_dataset_param != dataset_param
        ):
            st.session_state.accumulated_datasets = set()
            st.session_state.last_start_time_per_exp = {}
            st.session_state.datasets_searched = False
        st.session_state.previous_dataset_param = dataset_param
        st.session_state.dataset_param = dataset_param

    with col2:
        button_label = (
            f"Search {batch_size} more runs"
            if st.session_state.get("datasets_searched", False)
            else f"Search {batch_size} runs"
        )
        if st.button(button_label, key="search_datasets"):
            # Set flag and force a clean rerun
            st.session_state.is_searching = True
            st.rerun()  # <-- KEY FIX: This forces a new script run

    available_datasets = (
        _sort_datasets(list(st.session_state.accumulated_datasets))
        if st.session_state.get("datasets_searched", False)
        else []
    )
    if new_count > 0:
        st.success(f"Found {new_count} new datasets.")

    if available_datasets:
        selected_datasets = _render_dataset_selector(available_datasets)
        ordered_datasets = _render_dataset_ordering(selected_datasets)
        st.warning(
            f"Runs without a value in `{st.session_state.dataset_param}` will be filtered out",
            icon=":material/warning:",
        )
        return ordered_datasets
    else:
        st.info("Click 'Search' to find available datasets.")
        return None


def handle_dataset_parameter_selection(
    selected_experiments: list[str], dataset_mode: str
) -> str | None:
    """Handle dataset parameter selection.

    Args:
        selected_experiments: List of selected experiment names.
        dataset_mode: The mode selected for dataset definition.

    Returns:
        Selected dataset parameter name, or None if not found.
    """
    if not selected_experiments:
        return None

    available_params = (
        st.session_state["available_params"]
        if "available_params" in st.session_state
        else []
    )
    if not available_params:
        st.error("No parameters found in the selected experiments.")
        st.session_state.dataset_param = None
        return None

    if dataset_mode == "Multiple datasets defined by parameter":
        # Suggest default based on common patterns
        suggested_default = _find_dataset_parameter_suggestion(available_params)
        st.session_state.dataset_param = suggested_default
        return suggested_default
    else:
        # For other modes, dataset_param is set elsewhere
        dataset_param = None

    st.session_state.dataset_param = dataset_param
    return dataset_param


def _find_dataset_parameter_suggestion(available_params: list[str]) -> str:
    """Find the best suggestion for dataset parameter."""
    if "dataset_name" in available_params:
        return "dataset_name"
    elif "dataset" in available_params:
        return "dataset"
    elif any("dataset" in p.lower() and "name" in p.lower() for p in available_params):
        # Find one that contains both
        for p in available_params:
            if "dataset" in p.lower() and "name" in p.lower():
                return p
    return available_params[0] if available_params else ""


def handle_dataset_selection(
    selected_experiments: list[str], dataset_param: DatasetParamMode
) -> list[str] | None:
    """Handle dataset selection and ordering.

    Args:
        selected_experiments: List of selected experiment names.
        dataset_param: The parameter that defines dataset names.

    Returns:
        List of selected dataset names, or None if no datasets available.
    """
    if not selected_experiments:
        st.info("Please select at least one experiment to see available datasets.")
        return None

    if dataset_param == DatasetParamMode.EXPERIMENT_NAMES_AS_DATASETS:
        available_datasets = _get_available_datasets(
            selected_experiments, dataset_param
        )
        selected_datasets = _render_dataset_selector(available_datasets)
        ordered_datasets = _render_dataset_ordering(selected_datasets)
        return ordered_datasets
    elif dataset_param == DatasetParamMode.SINGLE_DATASET_MODE:
        available_datasets = _get_available_datasets(
            selected_experiments, dataset_param
        )
        selected_datasets = _render_dataset_selector(available_datasets)
        ordered_datasets = _render_dataset_ordering(selected_datasets)
        return ordered_datasets
    else:
        # Multiple datasets mode
        return _render_multiple_datasets_ui(selected_experiments)
