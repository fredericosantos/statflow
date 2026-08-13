"""
Configuration and state management for Streamlit MLflow visualization application.

Provides centralized default state management and YAML persistence.

Config is stored at ``~/.statflow/config.yaml``.  On first run, if the legacy
CWD config (``<cwd>/.statflow_config.yaml``) exists and the new path does not,
the legacy file is **copied** to the new path (one-time migration; the original
is left untouched).  All reads and writes thereafter go to the new path.

config.py
├── Constants (MLFLOW_TRACKING_URI, CONFIG_FILE, _LEGACY_CONFIG_FILE)
├── PERSISTABLE_KEYS              # Keys that get saved to YAML
├── DEFAULT_STATE                 # Default session state schema
├── _resolve_config_file()        # Resolve canonical path, run migration once
├── load_config()                 # Load config from YAML
├── save_config()                 # Save config to YAML
└── SessionState                  # State manager class
    ├── initialize()              # Initialize with defaults + saved config
    ├── save_to_config()          # Save all persistable state to YAML
    ├── save_key_to_config()      # Save a single key to YAML
    ├── get() / set() / has()     # State access methods
    └── Properties                # Convenience accessors
"""

import shutil
from pathlib import Path
from typing import Any

import yaml

# ==============================================================================
# Core Constants
# ==============================================================================

MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"

# Canonical config location: ~/.statflow/config.yaml
_STATFLOW_DIR = Path.home() / ".statflow"
CONFIG_FILE = _STATFLOW_DIR / "config.yaml"

# Legacy location (CWD-relative) — kept only for one-time migration detection.
_LEGACY_CONFIG_FILE = Path.cwd() / ".statflow_config.yaml"


def _resolve_config_file() -> Path:
    """Return the canonical config path, creating the directory and migrating if needed.

    Migration rules (run once at import time):
    - If ``~/.statflow/config.yaml`` already exists → use it, nothing to do.
    - Else if ``./.statflow_config.yaml`` (legacy CWD path) exists → **copy** it
      to ``~/.statflow/config.yaml`` (the original is left completely untouched).
    - Else → just ensure ``~/.statflow/`` exists and return the new path.
    """
    _STATFLOW_DIR.mkdir(parents=True, exist_ok=True)

    if CONFIG_FILE.exists():
        return CONFIG_FILE

    if _LEGACY_CONFIG_FILE.exists():
        try:
            shutil.copy2(_LEGACY_CONFIG_FILE, CONFIG_FILE)
            print(f"Statflow: migrated config from {_LEGACY_CONFIG_FILE} → {CONFIG_FILE}")
        except OSError as e:
            print(f"Warning: Could not migrate config: {e}")

    return CONFIG_FILE


# Run migration once at import time so CONFIG_FILE is always the active path.
_resolve_config_file()


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
    "selected_tags",
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
    "metric_directions",
    # Overall page
    "cross_dataset_agg",
    # Plots page
    "plot_agg",
    "plot_dataset_scope",
    "plot_axis_limits",
    "plot_log_x",
    "plot_log_y",
    # Application settings
    "app_name",
    "provider",
    "mlflow_server_url",
    "wandb_entity",
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
    "selected_tags": [],  # W&B tags chosen to act as grouping parameters
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
    "metric_directions": {},  # {metric: "Minimize"|"Maximize"} — better-is direction
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
    # Overall page
    "cross_dataset_agg": "median",
    # Plots page
    "plot_agg": "median",
    "plot_dataset_scope": "Aggregate across datasets",
    "plot_axis_limits": {},
    "plot_log_x": False,
    "plot_log_y": False,
    # Application settings
    "app_name": "Experiment Viewer",
    "provider": "mlflow",
    "mlflow_server_url": MLFLOW_TRACKING_URI,
    "wandb_entity": "",  # empty -> use the viewer's default W&B entity
}


def load_config() -> dict[str, Any]:
    """Load configuration from ``~/.statflow/config.yaml``."""
    if not CONFIG_FILE.exists():
        return {}

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        print(f"Warning: Could not load config from {CONFIG_FILE}: {e}")
        return {}


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to ``~/.statflow/config.yaml``."""
    try:
        _STATFLOW_DIR.mkdir(parents=True, exist_ok=True)
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
