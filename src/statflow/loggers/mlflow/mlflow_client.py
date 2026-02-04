"""
MLflow client utilities for fetching experiment data.

This module provides functions for connecting to MLflow and retrieving
experiment names. Run data fetching is handled by runs_cache.py.

mlflow_client.py
├── get_experiment_names()  # Get list of available experiment names
"""

import mlflow
import streamlit as st


@st.cache_data(ttl=600, show_spinner=False)
def get_experiment_names() -> list[str]:
    """Get list of available experiment names from MLflow.

    Returns:
        List of experiment names.
    """
    client = mlflow.tracking.MlflowClient(
        tracking_uri=st.session_state["mlflow_server_url"]
    )
    experiments = client.search_experiments()
    return [exp.name for exp in experiments if exp.lifecycle_stage == "active"]
