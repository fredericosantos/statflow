"""
Experiment selection logic for get started page.

Lists experiments from the active provider and renders the selection pills.
Persisted selections are clamped to the current options so a stale
`.statflow_config.yaml` (or a different provider/server) never crashes the
pills with a "default not in options" error.

experiment_selector.py
├── get_experiment_names()           # List experiments from the active provider
├── select_experiment()              # Handle experiment selection with custom label
└── select_experiment_as_datasets()  # Handle experiment selection as datasets
"""

from typing import cast

import streamlit as st

from statflow.loggers.registry import get_provider
from statflow.loggers.runs_cache import RunsCache


def get_experiment_names() -> list[str]:
    """List selectable experiments from the active provider."""
    return get_provider(st.session_state["provider"]).list_experiments()


def select_experiment_as_datasets() -> list[str] | None:
    """Handle dataset selection when datasets are experiment names.

    Returns:
        List of selected dataset names (experiment names), or None if no experiments available.
    """
    experiment_names = get_experiment_names()
    if not experiment_names:
        return None

    valid_default = [d for d in st.session_state["selected_datasets"] if d in experiment_names]
    # st.pills with selection_mode="multi" returns list[V]; cast to resolve type.
    selected_datasets: list[str] = cast(
        list[str],
        st.pills(
            "Select Datasets",
            options=experiment_names,
            default=valid_default,
            key="dataset_selector_from_experiments",
            selection_mode="multi",
        )
        or [],
    )

    # Update session state
    st.session_state.selected_datasets = selected_datasets

    # Also update experiments for metadata
    selected_experiments: list[str] = selected_datasets
    # Load if selection changed OR if we have selection but no data (initial load)
    if selected_experiments != st.session_state["selected_experiments"] or (
        selected_experiments and RunsCache.get_run_count() == 0
    ):
        st.session_state.selected_experiments = selected_experiments
        # Clear cache and load initial batch
        RunsCache.clear_cache()
        if selected_experiments:
            max_results = st.session_state.get("max_results", 2000)
            with st.spinner("Loading experiment runs..."):
                RunsCache.load_runs(selected_experiments, max_results=max_results)

    return selected_datasets


def select_experiment(
    label: str = "Select Experiments",
) -> list[str] | None:
    """Handle experiment selection UI and logic.

    Args:
        label: Custom label for the selection component.

    Returns:
        List of selected experiment names, or None if no experiments available.
    """
    experiment_names = get_experiment_names()
    if not experiment_names:
        return None

    st.markdown(f"#### {label}")
    valid_default = [e for e in st.session_state["selected_experiments"] if e in experiment_names]
    # st.pills with selection_mode="multi" returns list[V]; cast to resolve type.
    selected_experiments: list[str] = cast(
        list[str],
        st.pills(
            label,
            options=experiment_names,
            default=valid_default,
            key="experiment_selector",
            selection_mode="multi",
            label_visibility="collapsed",
        )
        or [],
    )
    st.space()

    # Check if selection changed - load initial runs
    # Load if selection changed OR if we have selection but no data (initial load)
    if selected_experiments != st.session_state["selected_experiments"] or (
        selected_experiments and RunsCache.get_run_count() == 0
    ):
        st.session_state.selected_experiments = selected_experiments
        # Clear cache and load initial batch
        RunsCache.clear_cache()
        if selected_experiments:
            max_results = st.session_state.get("max_results", 1000)
            with st.spinner("Loading experiment runs..."):
                RunsCache.load_runs(selected_experiments, max_results=max_results)

    return selected_experiments
