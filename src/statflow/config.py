"""
Configuration and state management for Streamlit MLflow visualization application.

Provides centralized default state management and YAML persistence.

config.py
├── Constants (MLFLOW_TRACKING_URI, CONFIG_FILE)
├── PERSISTABLE_KEYS              # Keys that get saved to YAML
├── DEFAULT_STATE                 # Default session state schema
├── load_config()                 # Load config from YAML
├── save_config()                 # Save config to YAML
└── SessionState                  # State manager class
    ├── initialize()              # Initialize with defaults + saved config
    ├── save_to_config()          # Save all persistable state to YAML
    ├── get() / set() / has()     # State access methods
    └── Properties                # Convenience accessors
"""

from pathlib import Path
from typing import Any

import yaml


# ==============================================================================
# Core Constants
# ==============================================================================

MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"
CONFIG_DIR = Path.cwd()
CONFIG_FILE = CONFIG_DIR / ".statflow_config.yaml"

DEFAULT_GRAPH_CONFIG = {
    "width": 800,
    "height": 600,
    "show_error_bars": True,
    "points_display": "outliers",
}


# ==============================================================================
# Keys to persist to YAML (user preferences that should survive restarts)
# ==============================================================================

PERSISTABLE_KEYS = [
    # Get Started page
    "dataset_mode",
    "selected_experiments",
    "selected_datasets",
    "dataset_param",
    "dataset_renames",
    "max_results",
    "historical_max_run_count",
    # Parameters page
    "selected_params",
    "active_param_filters",
    "param_filter_values",
    "selected_groups",
    "group_selections_cache",  # Caches group selections per parameter combo
    "group_renames",
    # Metrics page
    "selected_metrics",
    "active_metric_filters",
    "metric_filter_values",
    "metric_filter_nans",
    "metric_renames",
    "plot_height",
    # Comparison page
    "comparison_dataset_filter",
    "comparison_our_groups",
    "comparison_decimals",
    # Application settings
    "app_name",
    "mlflow_server_url",
    # Visualization settings
    "show_error_bars",
    "show_mean",
    "show_median",
    "show_std",
    "show_count",
    "use_custom_colors",
    "custom_colors",
    "custom_symbols",
    "points_display",
]


# ==============================================================================
# Default State Schema
# ==============================================================================

DEFAULT_STATE: dict[str, Any] = {
    # UI state (transient - not persisted)
    "zip_clicked": False,
    "zip_data": None,
    "active_group_filters": [],
    # Get Started page
    "dataset_mode": "Dataset names are experiment names",
    "selected_experiments": [],
    "selected_datasets": [],
    "available_datasets": [],
    "dataset_param": "",
    "dataset_renames": {},
    "max_results": 1000,
    "historical_max_run_count": 0,
    # Parameters page
    "selected_params": [],
    "available_params": [],
    "available_param_values": {},
    "active_param_filters": [],
    "param_filter_values": {},
    "selected_groups": [],
    "group_selections_cache": {},  # {param_combo_key: [selected_groups]}
    "group_renames": {},
    # Metrics page
    "selected_metrics": [],
    "available_metrics": [],
    "active_metric_filters": [],
    "metric_renames": {},
    "metric_filter_values": {},
    "metric_filter_nans": {},
    "plot_height": 400,
    # Comparison page
    "comparison_dataset_filter": [],
    "comparison_our_groups": [],
    "comparison_decimals": 4,
    # Visualization settings
    "show_mean": True,
    "show_median": False,
    "show_std": True,
    "show_count": False,
    "show_error_bars": True,
    "use_custom_colors": True,
    "custom_colors": {},
    "custom_symbols": {},
    "points_display": "outliers",
    # Application settings
    "app_name": "Experiment Viewer",
    "mlflow_server_url": MLFLOW_TRACKING_URI,
}


def load_config() -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        return {}

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        print(f"Warning: Could not load config from {CONFIG_FILE}: {e}")
        return {}


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to YAML file."""
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    except OSError as e:
        print(f"Warning: Could not save config to {CONFIG_FILE}: {e}")


# ==============================================================================
# State Manager
# ==============================================================================


class SessionState:
    """State manager that operates directly on st.session_state with YAML persistence."""

    _initialized_key = "_state_initialized"

    @classmethod
    def initialize(cls) -> None:
        """Initialize session state with defaults and saved config. Idempotent."""
        import streamlit as st

        if st.session_state.get(cls._initialized_key, False):
            return

        saved_config = load_config()

        # Initialize all keys with: saved value > default value
        for key, default_value in DEFAULT_STATE.items():
            if key in st.session_state:
                continue

            if key in saved_config:
                st.session_state[key] = saved_config[key]
            else:
                st.session_state[key] = default_value

        st.session_state[cls._initialized_key] = True

    @classmethod
    def save_to_config(cls) -> None:
        """Save all persistable state to YAML configuration."""
        import streamlit as st

        config = {}
        for key in PERSISTABLE_KEYS:
            if key in st.session_state:
                value = st.session_state[key]
                # Only save serializable values
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    config[key] = value

        save_config(config)

    @classmethod
    def save_key_to_config(cls, key: str) -> None:
        """Save a single key from session state to the configuration file."""
        import streamlit as st

        if key not in PERSISTABLE_KEYS:
            return

        if key not in st.session_state:
            return

        value = st.session_state[key]
        # Only save serializable values
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            return

        # Load existing, update, and save
        config = load_config()
        config[key] = value
        save_config(config)

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get value from session state."""
        import streamlit as st
        return st.session_state.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set value in session state."""
        import streamlit as st
        st.session_state[key] = value

    @classmethod
    def has(cls, key: str) -> bool:
        """Check if key exists in session state."""
        import streamlit as st
        return key in st.session_state

    @classmethod
    def update(cls, **kwargs: Any) -> None:
        """Update multiple session state values."""
        import streamlit as st
        for key, value in kwargs.items():
            st.session_state[key] = value

    # Convenience properties
    @property
    def app_name(self) -> str:
        return self.get("app_name", "Experiment Viewer")

    @property
    def selected_experiments(self) -> list[str]:
        return self.get("selected_experiments", [])

    @property
    def selected_datasets(self) -> list[str]:
        return self.get("selected_datasets", [])