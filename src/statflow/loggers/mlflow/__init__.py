"""
MLflow client utilities and caching for fetching experiment data.

This package provides functions for connecting to MLflow, retrieving
experiment runs, and caching data for efficient access.

mlflow/
├── __init__.py        # Package initialization
├── mlflow_client.py   # Core MLflow query functions (get_experiment_names)
└── runs_cache.py      # Centralized RunsCache for Polars DataFrame caching
"""