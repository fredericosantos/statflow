"""
Configuration and state management for Streamlit MLflow visualization application.

Provides centralized default state management and YAML persistence while
using st.session_state directly (no separate singleton).

config.py
├── Constants
│   ├── MLFLOW_TRACKING_URI     # MLflow tracking server URL.
│   ├── CONFIG_DIR              # Configuration directory path.
│   ├── CONFIG_FILE             # Configuration YAML file path.
│   └── DEFAULT_GRAPH_CONFIG    # Default graph configuration settings.
├── DEFAULT_STATE               # Default session state schema.
├── load_config()               # Loads project-specific config from YAML.
├── save_config()               # Persists configuration to YAML file.
├── State                       # State manager class.
│   ├── initialize()            # Initializes session state with defaults.
│   ├── get()                   # Gets value from session state.
│   ├── set()                   # Sets value in session state.
│   ├── has()                   # Checks if key exists in session state.
│   ├── update()                # Updates multiple session state values.
│   ├── save_to_config()        # Saves current state to YAML config.
│   ├── save_session()          # Saves experiment-specific session state.
│   └── Properties              # Convenience property accessors.
└── get_default_config()        # Returns default configuration dict.
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
# Default State Schema
# ==============================================================================

DEFAULT_STATE: dict[str, Any] = {
    # UI state (transient)
    "zip_clicked": False,
    "zip_data": None,
    # Dataset selection
    "selected_dataset": "",
    "selected_datasets": [],
    "available_datasets": [],
    "dataset_mode": "single",
    "dataset_param": "",
    # Parameter filtering
    "selected_params": [],
    "available_params": [],
    "available_param_values": {},
    "param_value_links": {},
    # Experiment/metric selection
    "selected_experiments": [],
    "available_metrics": [],
    # Visualization settings
    "show_mean": None,
    "show_error_bars": True,
    "use_custom_colors": True,
    "custom_colors": {},
    "custom_symbols": {},
    "graph_width": 800,
    "graph_height": 600,
    "points_display": "outliers",
    # Application settings
    "app_name": "Experiment Viewer",
    "mlflow_db_path": "mlruns.db",
    "datasets_path": "datasets",
    "mlflow_server_url": MLFLOW_TRACKING_URI,
    "dataset_renames": {},
}


def load_config() -> dict[str, Any]:
    """Load project-specific configuration from YAML file."""
    if not CONFIG_FILE.exists():
        return get_default_config()

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        print(f"Warning: Could not load config from {CONFIG_FILE}: {e}")
        return get_default_config()

    config = get_default_config()
    config.update(user_config)
    return config


def save_config(config: dict[str, Any]) -> None:
    """Persist configuration to YAML file."""
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    except OSError as e:
        print(f"Warning: Could not save config to {CONFIG_FILE}: {e}")


# ==============================================================================
# State Manager (operates directly on st.session_state)
# ==============================================================================


class State:
    """
    State manager that operates directly on st.session_state.

    Provides a cleaner API for accessing session state with default values
    and YAML persistence. All data lives in st.session_state.

    Usage:
        # In main app or any page
        import streamlit as st
        from statflow.config import State

        State.initialize()  # Call once (safe to call multiple times)

        # Use class methods directly
        experiments = State.get("selected_experiments")
        State.set("selected_datasets", ["ds1", "ds2"])

        # Or create instance for property access
        state = State()
        print(state.app_name)
        print(state.selected_experiments)
    """

    _initialized_key = "_state_initialized"

    def __init__(self) -> None:
        """Create state helper. Just a thin wrapper - no state stored here."""
        import streamlit as st

        self._st = st

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize st.session_state with defaults from YAML and schema.

        Safe to call multiple times (idempotent). Should be called at app startup.
        """
        import streamlit as st

        # Skip if already initialized
        if st.session_state.get(cls._initialized_key, False):
            return

        user_config = load_config()

        # Initialize all default keys
        for key, default_value in DEFAULT_STATE.items():
            if key in st.session_state:
                continue

            # Check user config first
            if key in user_config:
                st.session_state[key] = user_config[key]
                continue

            # Special handling for nested config values
            if key == "show_error_bars":
                st.session_state[key] = user_config.get("graph_config", {}).get(
                    "show_error_bars", default_value
                )
            elif key in ("graph_width", "graph_height", "points_display"):
                config_key = key.replace("graph_", "")
                st.session_state[key] = user_config.get("graph_config", {}).get(
                    config_key, default_value
                )
            elif key == "selected_dataset":
                st.session_state[key] = user_config.get("last_dataset", default_value)
            elif key in ("app_name", "mlflow_db_path", "datasets_path"):
                st.session_state[key] = user_config.get(key, default_value)
            elif key in ("custom_colors", "custom_symbols", "dataset_renames"):
                st.session_state[key] = user_config.get(key, {})
            else:
                st.session_state[key] = default_value

        # Mark as initialized
        st.session_state[cls._initialized_key] = True

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get value from session state.

        Args:
            key: Session state key.
            default: Default value if key doesn't exist.

        Returns:
            Value from session state or default.
        """
        import streamlit as st

        return st.session_state.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """
        Set value in session state.

        Args:
            key: Session state key.
            value: Value to set.
        """
        import streamlit as st

        st.session_state[key] = value

    @classmethod
    def has(cls, key: str) -> bool:
        """
        Check if key exists in session state.

        Args:
            key: Session state key.

        Returns:
            True if key exists.
        """
        import streamlit as st

        return key in st.session_state

    @classmethod
    def update(cls, **kwargs: Any) -> None:
        """
        Update multiple session state values.

        Args:
            **kwargs: Key-value pairs to update.
        """
        import streamlit as st

        for key, value in kwargs.items():
            st.session_state[key] = value

    @classmethod
    def save_to_config(cls) -> None:
        """Save current session state to YAML configuration."""
        import streamlit as st

        config = {
            "use_custom_colors": st.session_state.get("use_custom_colors", True),
            "custom_colors": st.session_state.get("custom_colors", {}),
            "custom_symbols": st.session_state.get("custom_symbols", {}),
            "graph_config": {
                "width": st.session_state.get("graph_width", 800),
                "height": st.session_state.get("graph_height", 600),
                "show_error_bars": st.session_state.get("show_error_bars", True),
                "points_display": st.session_state.get("points_display", "outliers"),
            },
            "last_dataset": st.session_state.get("selected_dataset", ""),
            "app_name": st.session_state.get("app_name", "Experiment Viewer"),
            "mlflow_db_path": st.session_state.get("mlflow_db_path", "mlruns.db"),
            "datasets_path": st.session_state.get("datasets_path", "datasets"),
            "dataset_renames": st.session_state.get("dataset_renames", {}),
        }

        save_config(config)

    @classmethod
    def save_session(cls) -> None:
        """Save experiment-specific session state."""
        import streamlit as st

        keys_to_persist = [
            "selected_experiments",
            "available_params",
            "available_param_values",
            "available_metrics",
            "selected_datasets",
            "dataset_param",
            "available_datasets",
            "selected_dataset",
            "selected_params",
            "param_value_links",
            "dataset_renames",
            "custom_colors",
            "custom_symbols",
            "app_name",
        ]

        config = {
            k: st.session_state[k] for k in keys_to_persist if k in st.session_state
        }
        save_config(config)

    # Instance method: Get value from session state
    def get_value(self, key: str, default: Any = None) -> Any:
        """Instance method: Get value from session state."""
        return self._st.session_state.get(key, default)

    # Instance method: Set value in session state
    def set_value(self, key: str, value: Any) -> None:
        """Instance method: Set value in session state."""
        self._st.session_state[key] = value

    # Instance method: Check if key exists
    def has_key(self, key: str) -> bool:
        """Instance method: Check if key exists."""
        return key in self._st.session_state

    # Instance method: Update multiple values
    def update_values(self, **kwargs: Any) -> None:
        """Instance method: Update multiple values."""
        for key, value in kwargs.items():
            self._st.session_state[key] = value

    # Property-style access for common attributes
    @property
    def app_name(self) -> str:
        """Get application name."""
        return self._st.session_state.get("app_name", "Experiment Viewer")

    @property
    def selected_experiments(self) -> list[str]:
        """Get selected experiments."""
        return self._st.session_state.get("selected_experiments", [])

    @property
    def selected_datasets(self) -> list[str]:
        """Get selected datasets."""
        return self._st.session_state.get("selected_datasets", [])

    @property
    def available_datasets(self) -> list[str]:
        """Get available datasets."""
        return self._st.session_state.get("available_datasets", [])

    @property
    def custom_colors(self) -> dict:
        """Get custom color mappings."""
        return self._st.session_state.get("custom_colors", {})

    @property
    def custom_symbols(self) -> dict:
        """Get custom symbol mappings."""
        return self._st.session_state.get("custom_symbols", {})


# ==============================================================================
# Default Constants (imported by other modules)
# ==============================================================================

DEFAULT_ZIP_CLICKED = False

DEFAULT_DATASET_RENAMES: dict[str, dict[str, str]] = {
    # Real-life datasets
    "airfoil": {"display_name": "Airfoil", "latex_name": "Airfoil"},
    "yacht": {"display_name": "Yacht", "latex_name": "Yacht"},
    "slump": {"display_name": "Slump", "latex_name": "Slump"},
    "toxicity": {"display_name": "Toxicity", "latex_name": "Toxicity"},
    "istanbul": {"display_name": "Istanbul", "latex_name": "Istanbul"},
    "qsaraquatic": {"display_name": "QSAR", "latex_name": "QSAR"},
    # Blackbox datasets
    "blackbox_1199_BNG_echoMonths": {"display_name": "Echo", "latex_name": "Echo"},
    "blackbox_1193_BNG_lowbwt": {"display_name": "Lowbwt", "latex_name": "Lowbwt"},
    "blackbox_1089_USCrime": {"display_name": "USCrime", "latex_name": "USCrime"},
    "blackbox_1028_SWD": {"display_name": "SWD", "latex_name": "SWD"},
    "blackbox_678_visualizing_environmental": {"display_name": "Enviro", "latex_name": "Enviro"},
    "blackbox_650_fri_c0_500_50": {"display_name": "FRI-1", "latex_name": "FRI-1"},
    "blackbox_606_fri_c2_1000_10": {"display_name": "FRI-2", "latex_name": "FRI-2"},
    "blackbox_579_fri_c0_250_5": {"display_name": "FRI-3", "latex_name": "FRI-3"},
    "blackbox_557_analcatdata_apnea1": {"display_name": "Apnea", "latex_name": "Apnea"},
    "blackbox_522_pm10": {"display_name": "PM10", "latex_name": "PM10"},
    "blackbox_210_cloud": {"display_name": "Cloud", "latex_name": "Cloud"},
    "blackbox_192_vineyard": {"display_name": "Vineyard", "latex_name": "Vineyard"},
}

DEFAULT_PARETO_COLORS = {
    "GSGP-OMS": "#5A6C7D",
    "GSGP-std": "#B8C5D0",
    "SLIM": "#D4A574",
    "arc_beta": {
        "0.0": "#6B8CAE",
        "0.1": "#5B9AA8",
        "0.2": "#4D9B82",
        "0.3": "#6AA56D",
        "0.4": "#8FAE6C",
        "0.5": "#B8A05C",
        "0.6": "#C4976C",
        "0.7": "#B8846C",
        "0.8": "#9B7FA0",
        "0.9": "#8B6B8F",
        "0.95": "#C85C5C",
        "1.0": "#704545",
    },
}

DEFAULT_PARETO_SYMBOLS = {
    "gsgp_oms": "circle",
    "gsgp_std": "square",
    "slim": "diamond",
    "arc_beta": {
        "0.0": "circle",
        "0.1": "square",
        "0.2": "diamond",
        "0.3": "cross",
        "0.4": "x",
        "0.5": "triangle-up",
        "0.6": "triangle-down",
        "0.7": "pentagon",
        "0.8": "hexagon",
        "0.9": "star",
        "1.0": "hexagram",
    },
}


# ==============================================================================
# Utility Functions
# ==============================================================================


def get_default_config() -> dict[str, Any]:
    """Get default configuration values."""
    return {
        "use_custom_colors": True,
        "custom_colors": DEFAULT_PARETO_COLORS.copy(),
        "custom_symbols": DEFAULT_PARETO_SYMBOLS.copy(),
        "dataset_renames": DEFAULT_DATASET_RENAMES.copy(),
        "graph": DEFAULT_GRAPH_CONFIG.copy(),
        "app_name": "Experiment Viewer",
        "mlflow_db_path": "mlruns.db",
        "datasets_path": "datasets",
    }