"""
Data-source (provider) selection and connection config for Get Started.

Renders the sidebar control that picks the active run provider (MLflow, W&B,
...) and its connection settings, persisting the choice. Switching providers
clears the run cache and the provider-scoped selections (experiments, datasets,
params, metrics) because their valid options differ per backend.

provider_config.py
├── render_provider_config()   # sidebar: provider picker + connection settings
├── _switch_provider()         # apply a provider change and reset scoped state
├── _mlflow_connection()       # MLflow tracking-URI input
├── _wandb_connection()        # W&B entity input
├── _render_reset()            # "Reset selections" button
└── _reset_selections()        # wipe saved selections, keep connection settings
"""

import copy

import streamlit as st

from statflow.config import DEFAULT_STATE, SessionState
from statflow.loggers.registry import available_providers, get_provider
from statflow.loggers.runs_cache import RunsCache

# Keys kept across a "Reset selections": connection + global display prefs.
# Everything else in DEFAULT_STATE (selections, filters, renames, comparison
# choices, caches) is restored to its default and re-saved.
_PRESERVE_ON_RESET = {
    "provider",
    "mlflow_server_url",
    "wandb_entity",
    "app_name",
    "max_results",
    "historical_max_run_count",
    "show_mean",
    "show_median",
    "show_std",
    "show_count",
    "show_error_bars",
    "use_custom_colors",
    "custom_colors",
    "custom_symbols",
    "points_display",
}

# Selection state that only makes sense for one backend; reset on provider switch
# so stale values never become invalid widget defaults.
_PROVIDER_SCOPED_KEYS = {
    "selected_experiments": [],
    "selected_datasets": [],
    "available_datasets": [],
    "selected_params": [],
    "available_params": [],
    "selected_groups": [],
    "selected_metrics": [],
    "available_metrics": [],
    "active_param_filters": [],
    "active_metric_filters": [],
    "dataset_param": "",
}


def render_provider_config() -> None:
    """Sidebar control: pick the data-source provider and its connection."""
    current = st.session_state["provider"]
    labels = {name: get_provider(name).label for name in available_providers()}

    with st.sidebar:
        with st.expander("Data Source", expanded=True, icon=":material/database:"):
            chosen = st.pills(
                "Provider",
                options=list(labels),
                format_func=lambda name: labels.get(name, name),
                default=current,
                selection_mode="single",
                key="provider_selector",
                label_visibility="collapsed",
            )
            if chosen and chosen != current:
                _switch_provider(chosen)

            if current == "mlflow":
                _mlflow_connection()
            elif current == "wandb":
                _wandb_connection()

            _render_reset()


def _switch_provider(name: str) -> None:
    """Activate provider `name`, dropping caches and provider-scoped selections."""
    st.session_state["provider"] = name
    RunsCache.clear_cache()
    st.session_state.pop("server_running", None)
    for key, empty in _PROVIDER_SCOPED_KEYS.items():
        st.session_state[key] = empty
    SessionState.save_key_to_config("provider")
    st.rerun()


def _mlflow_connection() -> None:
    url = st.text_input(
        "Tracking URI",
        value=st.session_state["mlflow_server_url"],
        key="mlflow_url_input",
        help="MLflow tracking server, e.g. http://0.0.0.0:5000",
    )
    if url != st.session_state["mlflow_server_url"]:
        st.session_state["mlflow_server_url"] = url
        st.session_state.pop("server_running", None)
        RunsCache.clear_cache()
        SessionState.save_key_to_config("mlflow_server_url")
        st.rerun()


def _wandb_connection() -> None:
    entity = st.text_input(
        "Entity",
        value=st.session_state["wandb_entity"],
        placeholder="(your default entity)",
        key="wandb_entity_input",
        help="W&B entity (user or team). Leave blank to use your default. "
        "Auth uses the api.wandb.ai key in ~/.netrc.",
    )
    if entity != st.session_state["wandb_entity"]:
        st.session_state["wandb_entity"] = entity
        st.session_state.pop("server_running", None)
        RunsCache.clear_cache()
        SessionState.save_key_to_config("wandb_entity")
        st.rerun()


def _render_reset() -> None:
    """Button to wipe saved selections (keeps connection settings)."""
    if st.button(
        "Reset selections",
        icon=":material/restart_alt:",
        width="stretch",
        help="Clear saved experiments, datasets, params, groups, metrics, filters, "
        "renames, and comparison choices (keeps your provider/connection). Use this "
        "when pages come up empty after switching data source — usually stale state "
        "in .statflow_config.yaml from a previous setup.",
    ):
        _reset_selections()


def _reset_selections() -> None:
    """Restore every non-connection key to its default and persist the clean state."""
    RunsCache.clear_cache()
    st.session_state.pop("server_running", None)
    for key, default in DEFAULT_STATE.items():
        if key not in _PRESERVE_ON_RESET:
            st.session_state[key] = copy.deepcopy(default)
    SessionState.save_to_config()
    st.rerun()
