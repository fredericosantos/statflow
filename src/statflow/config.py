"""
Configuration management and constants for the GSGP-ARC Streamlit application.

This module centralizes all configuration constants, dataset definitions,
and provides YAML-based persistence for user preferences and settings.

config.py
├── Constants and dataset definitions
├── Configuration persistence (YAML)
├── Default color/symbol mappings
├── Session state management helpers
└── Configuration loading/saving utilities

Usage:
    from statflow.config import (
        MLFLOW_TRACKING_URI, get_all_datasets,
        load_user_config, save_user_config
    )
"""

import os
import yaml
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from typing import Any, Dict, Optional

# MLflow configuration
MLFLOW_TRACKING_URI = "http://0.0.0.0:5000"

# Dataset definitions
DATASETS_REAL_LIFE = [
    "parkinson",
    "concrete",
    "airfoil",
    "yacht",
    "slump",
    "toxicity",
    "istanbul",
    "qsaraquatic",
]

DATASETS_BLACKBOX = [
    "blackbox_1199_BNG_echoMonths",
    "blackbox_1193_BNG_lowbwt",
    "blackbox_1089_USCrime",
    "blackbox_1028_SWD",
    "blackbox_678_visualizing_environmental",
    "blackbox_650_fri_c0_500_50",
    "blackbox_606_fri_c2_1000_10",
    "blackbox_579_fri_c0_250_5",
    "blackbox_557_analcatdata_apnea1",
    "blackbox_522_pm10",
    "blackbox_210_cloud",
    "blackbox_192_vineyard",
]

# Default configuration constants
DEFAULT_USE_CUSTOM_COLORS = True
DEFAULT_SHOW_ERROR_BARS = True
DEFAULT_GRAPH_WIDTH = 800
DEFAULT_GRAPH_HEIGHT = 600
DEFAULT_POINTS_DISPLAY = "outliers"
DEFAULT_APP_NAME = "Experiment"
DEFAULT_MLFLOW_DB_PATH = "mlruns.db"
DEFAULT_DATASETS_PATH = "datasets"
DEFAULT_DATASETS = []  # Fallback empty list for dynamic datasets
DEFAULT_LAST_DATASET = DEFAULT_DATASETS[0] if DEFAULT_DATASETS else "" if DEFAULT_DATASETS else ""
DEFAULT_SELECTED_MPF_VALUES = None
DEFAULT_SELECTED_BETA_VALUES = None
DEFAULT_SELECTED_PINFLATE_VALUES = None
DEFAULT_SHOW_MEAN = False
DEFAULT_ZIP_CLICKED = False
DEFAULT_ZIP_DATA = None


def get_all_datasets() -> List[str]:
    """Get all available datasets based on session state."""
    import streamlit as st
    if 'available_datasets' in st.session_state:
        return st.session_state['available_datasets']
    return DEFAULT_DATASETS

# Default dataset renames for cleaner LaTeX export
DEFAULT_DATASET_RENAMES = {
    # Real-life datasets
    "airfoil": "Airfoil",
    "yacht": "Yacht",
    "slump": "Slump",
    "toxicity": "Toxicity",
    "istanbul": "Istanbul",
    "qsaraquatic": "QSAR",
    # Blackbox datasets
    "blackbox_1199_BNG_echoMonths": "Echo",
    "blackbox_1193_BNG_lowbwt": "Lowbwt",
    "blackbox_1089_USCrime": "USCrime",
    "blackbox_1028_SWD": "SWD",
    "blackbox_678_visualizing_environmental": "Enviro",
    "blackbox_650_fri_c0_500_50": "FRI-1",
    "blackbox_606_fri_c2_1000_10": "FRI-2",
    "blackbox_579_fri_c0_250_5": "FRI-3",
    "blackbox_557_analcatdata_apnea1": "Apnea",
    "blackbox_522_pm10": "PM10",
    "blackbox_210_cloud": "Cloud",
    "blackbox_192_vineyard": "Vineyard",
}

# Default color mappings for Pareto front visualization
DEFAULT_PARETO_COLORS = {
    "GSGP-OMS": "#5A6C7D",      # Dark slate
    "GSGP-std": "#B8C5D0",      # Light slate
    "SLIM": "#D4A574",          # Muted yellow/gold
    "arc_beta": {
        "0.0": "#6B8CAE",       # Steel blue
        "0.1": "#5B9AA8",       # Muted turquoise
        "0.2": "#4D9B82",       # Seafoam green
        "0.3": "#6AA56D",       # Sage green
        "0.4": "#8FAE6C",       # Olive green
        "0.5": "#B8A05C",       # Muted gold
        "0.6": "#C4976C",       # Tan/beige
        "0.7": "#B8846C",       # Muted brown
        "0.8": "#9B7FA0",       # Muted purple/mauve
        "0.9": "#8B6B8F",       # Darker purple
        "0.95": "#C85C5C",      # Red
        "1.0": "#704545",       # Deep maroon
    }
}

# Default symbol mappings for Pareto front visualization
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
    }
}

# Default graph configuration
DEFAULT_GRAPH_CONFIG = {
    "width": 800,
    "height": 600,
    "show_error_bars": True,
    "points_display": "outliers",  # "hide", "outliers", "all"
}

# Session state keys
SESSION_STATE_KEYS = [
    'zip_clicked',
    'zip_data',
    'selected_dataset',
    'selected_mpf_values',
    'selected_beta_values',
    'selected_pinflate_values',
    'show_mean',
    'show_error_bars',
    'use_custom_colors',
    'custom_colors',
    'custom_symbols',
    'graph_width',
    'graph_height',
    'points_display',
    'app_name',
    'mlflow_db_path',
    'datasets_path',
    'dataset_renames',
]

# Configuration file path
CONFIG_DIR = Path.cwd()
CONFIG_FILE = CONFIG_DIR / ".statflow_config.yaml"


def load_user_config() -> Dict[str, Any]:
    """Load user configuration from YAML file.

    Returns:
        Dictionary containing user configuration, merged with defaults.
    """

    if not CONFIG_FILE.exists():
        return get_default_config()

    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f) or {}
    except (yaml.YAMLError, IOError) as e:
        print(f"Warning: Could not load config file {CONFIG_FILE}: {e}")
        return get_default_config()

    # Merge with defaults
    default_config = get_default_config()
    default_config.update(user_config)
    return default_config


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for YAML serialization."""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_user_config(config: Dict[str, Any]) -> None:
    """Save user configuration to YAML file.

    Args:
        config: Configuration dictionary to save.
    """

    # Convert numpy types to Python types for YAML serialization
    config = convert_numpy_types(config)

    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    except IOError as e:
        print(f"Warning: Could not save config file {CONFIG_FILE}: {e}")


def save_session_state() -> None:
    """Save current session state to config file."""
    import streamlit as st

    config = {}
    session_keys = [
        'selected_experiments', 'available_params', 'available_param_values', 'available_metrics',
        'selected_datasets', 'dataset_param', 'available_datasets', 'selected_dataset',
        'selected_params', 'param_value_links', 'dataset_renames'
    ]
    for key in session_keys:
        if key in st.session_state:
            config[key] = st.session_state[key]
    save_user_config(config)


def load_session_state() -> None:
    """Load session state from config file."""
    import streamlit as st

    config = load_user_config()
    for key, value in config.items():
        st.session_state[key] = value


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values.

    Returns:
        Dictionary with default configuration.
    """
    return {
        "use_custom_colors": True,
        "custom_colors": DEFAULT_PARETO_COLORS.copy(),
        "custom_symbols": DEFAULT_PARETO_SYMBOLS.copy(),
        "graph_config": DEFAULT_GRAPH_CONFIG.copy(),
        "last_dataset": DEFAULT_DATASETS[0],
        "selected_mpf_values": None,  # Will be populated dynamically
        "selected_beta_values": None,  # Will be populated dynamically
        "selected_pinflate_values": None,  # Will be populated dynamically
        "dataset_renames": DEFAULT_DATASET_RENAMES.copy(),
    }


def initialize_session_state() -> None:
    """Initialize Streamlit session state with default values."""
    import streamlit as st

    # Load user config
    user_config = load_user_config()

    # Initialize session state keys if not present
    for key in SESSION_STATE_KEYS:
        if key not in st.session_state:
            if key in user_config:
                # Special handling for dataset_renames: always merge with defaults
                if key == 'dataset_renames':
                    merged_renames = DEFAULT_DATASET_RENAMES.copy()
                    merged_renames.update(user_config[key])  # User customizations override
                    st.session_state[key] = merged_renames
                else:
                    st.session_state[key] = user_config[key]
            else:
                # Set defaults for keys not in config
                if key == 'zip_clicked':
                    st.session_state[key] = DEFAULT_ZIP_CLICKED
                elif key == 'zip_data':
                    st.session_state[key] = DEFAULT_ZIP_DATA
                elif key == 'selected_dataset':
                    if 'last_dataset' in user_config:
                        st.session_state[key] = user_config['last_dataset']
                    else:
                        st.session_state[key] = DEFAULT_LAST_DATASET
                elif key in ['selected_mpf_values', 'selected_beta_values', 'selected_pinflate_values']:
                    if key in user_config:
                        st.session_state[key] = user_config[key]
                    else:
                        if key == 'selected_mpf_values':
                            st.session_state[key] = DEFAULT_SELECTED_MPF_VALUES
                        elif key == 'selected_beta_values':
                            st.session_state[key] = DEFAULT_SELECTED_BETA_VALUES
                        elif key == 'selected_pinflate_values':
                            st.session_state[key] = DEFAULT_SELECTED_PINFLATE_VALUES
                elif key == 'show_mean':
                    st.session_state[key] = DEFAULT_SHOW_MEAN
                elif key == 'show_error_bars':
                    if 'graph_config' in user_config:
                        graph_config = user_config['graph_config']
                    else:
                        graph_config = {}
                    if 'show_error_bars' in graph_config:
                        st.session_state[key] = graph_config['show_error_bars']
                    else:
                        st.session_state[key] = DEFAULT_SHOW_ERROR_BARS
                elif key == 'use_custom_colors':
                    if 'use_custom_colors' in user_config:
                        st.session_state[key] = user_config['use_custom_colors']
                    else:
                        st.session_state[key] = DEFAULT_USE_CUSTOM_COLORS
                elif key == 'custom_colors':
                    if 'custom_colors' in user_config:
                        st.session_state[key] = user_config['custom_colors']
                    else:
                        st.session_state[key] = DEFAULT_PARETO_COLORS.copy()
                elif key == 'custom_symbols':
                    if 'custom_symbols' in user_config:
                        st.session_state[key] = user_config['custom_symbols']
                    else:
                        st.session_state[key] = DEFAULT_PARETO_SYMBOLS.copy()
                elif key == 'graph_width':
                    if 'graph_config' in user_config:
                        graph_config = user_config['graph_config']
                    else:
                        graph_config = {}
                    if 'width' in graph_config:
                        st.session_state[key] = graph_config['width']
                    else:
                        st.session_state[key] = DEFAULT_GRAPH_WIDTH
                elif key == 'graph_height':
                    if 'graph_config' in user_config:
                        graph_config = user_config['graph_config']
                    else:
                        graph_config = {}
                    if 'height' in graph_config:
                        st.session_state[key] = graph_config['height']
                    else:
                        st.session_state[key] = DEFAULT_GRAPH_HEIGHT
                elif key == 'points_display':
                    if 'graph_config' in user_config:
                        graph_config = user_config['graph_config']
                    else:
                        graph_config = {}
                    if 'points_display' in graph_config:
                        st.session_state[key] = graph_config['points_display']
                    else:
                        st.session_state[key] = DEFAULT_POINTS_DISPLAY
                elif key == 'app_name':
                    if 'app_name' in user_config:
                        st.session_state[key] = user_config['app_name']
                    else:
                        st.session_state[key] = DEFAULT_APP_NAME
                elif key == 'mlflow_db_path':
                    if 'mlflow_db_path' in user_config:
                        st.session_state[key] = user_config['mlflow_db_path']
                    else:
                        st.session_state[key] = DEFAULT_MLFLOW_DB_PATH
                elif key == 'datasets_path':
                    if 'datasets_path' in user_config:
                        st.session_state[key] = user_config['datasets_path']
                    else:
                        st.session_state[key] = DEFAULT_DATASETS_PATH
                elif key == 'dataset_renames':
                    # Merge defaults with saved config to ensure new datasets are included
                    if 'dataset_renames' in user_config:
                        saved_renames = user_config['dataset_renames']
                    else:
                        saved_renames = {}
                    merged_renames = DEFAULT_DATASET_RENAMES.copy()
                    merged_renames.update(saved_renames)  # User customizations override defaults
                    st.session_state[key] = merged_renames

    # Load saved session state
    load_session_state()


def save_session_state_to_config() -> None:
    """Save current session state to user configuration file."""
    import streamlit as st

    config = {
        "use_custom_colors": st.session_state['use_custom_colors'] if 'use_custom_colors' in st.session_state else DEFAULT_USE_CUSTOM_COLORS,
        "custom_colors": st.session_state['custom_colors'] if 'custom_colors' in st.session_state else DEFAULT_PARETO_COLORS.copy(),
        "custom_symbols": st.session_state['custom_symbols'] if 'custom_symbols' in st.session_state else DEFAULT_PARETO_SYMBOLS.copy(),
        "graph_config": {
            "width": st.session_state['graph_width'] if 'graph_width' in st.session_state else DEFAULT_GRAPH_WIDTH,
            "height": st.session_state['graph_height'] if 'graph_height' in st.session_state else DEFAULT_GRAPH_HEIGHT,
            "show_error_bars": st.session_state['show_error_bars'] if 'show_error_bars' in st.session_state else DEFAULT_SHOW_ERROR_BARS,
            "points_display": st.session_state['points_display'] if 'points_display' in st.session_state else DEFAULT_POINTS_DISPLAY,
        },
        "last_dataset": st.session_state['selected_dataset'] if 'selected_dataset' in st.session_state else DEFAULT_LAST_DATASET,
        "selected_mpf_values": st.session_state['selected_mpf_values'] if 'selected_mpf_values' in st.session_state else DEFAULT_SELECTED_MPF_VALUES,
        "selected_beta_values": st.session_state['selected_beta_values'] if 'selected_beta_values' in st.session_state else DEFAULT_SELECTED_BETA_VALUES,
        "selected_pinflate_values": st.session_state['selected_pinflate_values'] if 'selected_pinflate_values' in st.session_state else DEFAULT_SELECTED_PINFLATE_VALUES,
        "app_name": st.session_state['app_name'] if 'app_name' in st.session_state else DEFAULT_APP_NAME,
        "mlflow_db_path": st.session_state['mlflow_db_path'] if 'mlflow_db_path' in st.session_state else DEFAULT_MLFLOW_DB_PATH,
        "dataset_renames": st.session_state['dataset_renames'] if 'dataset_renames' in st.session_state else DEFAULT_DATASET_RENAMES.copy(),
    }

    save_user_config(config)


# Shared sidebar utilities
def check_mlflow_server_status() -> bool:
    """Check if MLflow server is running and accessible."""
    try:
        # Try to connect to the MLflow server
        response = requests.get("http://0.0.0.0:5000/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def setup_sidebar():
    """Setup the sidebar with MLflow server status."""
    import streamlit as st
    
    with st.sidebar:
        if check_mlflow_server_status():
            st.success("MLFlow Server Running", icon=":material/bolt:")
        else:
            st.error("Server Not Running", icon=":material/power_off:")
        st.markdown("---")