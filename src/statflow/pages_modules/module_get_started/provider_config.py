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
└── _wandb_connection()        # W&B entity input
"""

import streamlit as st

from statflow.config import SessionState
from statflow.loggers.registry import available_providers, get_provider
from statflow.loggers.runs_cache import RunsCache

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
