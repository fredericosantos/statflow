"""
MLflow client utilities for fetching experiment data.

This module provides functions for connecting to MLflow and retrieving
experiment runs with appropriate filtering and caching.

mlflow_client.py
├── MLflow connection setup
├── Single dataset fetching (get_filtered_runs)
├── Parallel multi-dataset fetching (fetch_all_datasets_parallel)
├── Data filtering and validation
└── Caching decorators for performance

Usage:
    from statflow.utils.mlflow_client import (
        get_filtered_runs, fetch_all_datasets_parallel
    )
"""

import mlflow
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, List
import streamlit as st

from statflow.config import MLFLOW_TRACKING_URI, DATASETS_REAL_LIFE, DATASETS_BLACKBOX

ALL_DATASETS = DATASETS_REAL_LIFE + DATASETS_BLACKBOX


@st.cache_data(ttl=600, show_spinner=False)
def get_filtered_runs(dataset_name: str, experiment_ids: Optional[List[str]] = None) -> pl.DataFrame:
    """Fetch runs from MLflow with specific filters.

    Args:
        dataset_name: Name of the dataset to filter by.
        experiment_ids: List of experiment IDs to search in.

    Returns:
        DataFrame containing filtered run data.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Build filter string
    filter_string = (
        f'params.dataset_name = "{dataset_name}" '
        f'and params.crossover_probability = "0.0" '
        f'and params.mutation_probability = "1.0" '
        f'and params.activation_fn_init = "ActFn.IDENTITY" '
        f"and metrics.best_test_fitness > 0 "
        f'and params.scale_dataset = "True" '
        f'and params.scale_target = "True" '
        f'and params.arc_v2 = "True"'
    )

    # Search for runs across specified experiments
    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        max_results=10000,
    )

    # Convert to Polars DataFrame
    runs = pl.from_pandas(runs)

    # Apply additional filters in Polars (complex logic not supported in MLflow filter)
    # For ARC: only keep runs where (new_oms = "False" OR new_oms is missing) AND pool_refresh_interval = "0"
    # For other variants: keep all runs
    runs = runs.filter(
        (pl.col("params.variant") != "arc") | 
        (
            ((pl.col("params.new_oms") == "False") | pl.col("params.new_oms").is_null()) & 
            (pl.col("params.pool_refresh_interval") == "0")
        )
    )

    return runs


@st.cache_data(ttl=600, show_spinner=True)
def fetch_all_datasets_parallel(
    selected_mpf_values: Optional[Tuple[str, ...]] = None,
    selected_beta_values: Optional[Tuple[str, ...]] = None,
    selected_pinflate_values: Optional[Tuple[str, ...]] = None,
    selected_datasets: Optional[Tuple[str, ...]] = None,
    selected_experiments: Optional[Tuple[str, ...]] = None,
) -> pl.DataFrame:
    """Fetch runs for selected datasets in parallel.

    Args:
        selected_mpf_values: Optional tuple of MPF values to filter ARC runs.
        selected_beta_values: Optional tuple of Beta values to filter ARC runs.
        selected_pinflate_values: Optional tuple of P_inflate values to filter SLIM-GSGP runs.
        selected_datasets: Optional tuple of dataset names to fetch. If None, fetches all datasets.

    Returns:
        DataFrame containing all runs across selected datasets.
    """
    all_runs = []

    # Get experiment metadata
    if selected_experiments:
        metadata = get_metadata_from_experiments(selected_experiments)
        experiment_ids = [metadata[exp]['experiment_id'] for exp in selected_experiments if exp in metadata]
        id_to_name = {v['experiment_id']: k for k, v in metadata.items()}
    else:
        # Default to arc-gsgp
        metadata = get_metadata_from_experiments(("arc-gsgp",))
        experiment_ids = [metadata.get("arc-gsgp", {}).get('experiment_id', "0")]
        id_to_name = {experiment_ids[0]: "arc-gsgp"}

    def fetch_dataset(dataset_name: str) -> Tuple[str, pl.DataFrame]:
        """Fetch runs for a single dataset."""
        try:
            runs_df = get_filtered_runs(dataset_name, experiment_ids)
            return (dataset_name, runs_df)
        except Exception as e:
            print(f"Error fetching data for {dataset_name}: {e}")
            return (dataset_name, pl.DataFrame())

    # Use ThreadPoolExecutor for parallel fetching
    datasets_to_fetch = selected_datasets if selected_datasets else ALL_DATASETS
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_dataset, ds): ds for ds in datasets_to_fetch}

        for future in as_completed(futures):
            dataset_name, runs_df = future.result()
            if not runs_df.is_empty():
                # Add experiment_name and dataset_name
                runs_df = runs_df.with_columns(
                    pl.col('experiment_id').map_dict(id_to_name).alias('experiment_name'),
                    pl.lit(dataset_name).alias("dataset_name")
                )
                all_runs.append(runs_df)

    if not all_runs:
        return pl.DataFrame()

    # Combine all runs
    combined_df = pl.concat(all_runs)

    # Apply MPF filtering if provided
    if selected_mpf_values is not None:
        if selected_mpf_values:
            # Keep non-ARC runs + ARC runs with selected MPF values
            combined_df = combined_df.filter(
                (pl.col("params.variant") != "arc") | 
                pl.col("params.mutation_pool_factor").is_in(selected_mpf_values)
            )
        else:
            # If ARC exists but no MPF selected, exclude all ARC runs
            combined_df = combined_df.filter(pl.col("params.variant") != "arc")

    # Apply Beta filtering if provided
    if selected_beta_values is not None:
        if selected_beta_values:
            combined_df = combined_df.filter(
                (pl.col("params.variant") != "arc") | 
                pl.col("params.arc_beta").is_in(selected_beta_values)
            )
        else:
            combined_df = combined_df.filter(pl.col("params.variant") != "arc")

    # Apply P_inflate filtering if provided
    if selected_pinflate_values is not None:
        if selected_pinflate_values:
            combined_df = combined_df.filter(
                (pl.col("params.variant") != "slim_gsgp") | 
                pl.col("params.p_inflate").is_in(selected_pinflate_values)
            )
        else:
            combined_df = combined_df.filter(pl.col("params.variant") != "slim_gsgp")

    return combined_df


@st.cache_data(ttl=600, show_spinner=False)
def get_experiment_names() -> list[str]:
    """Get list of available experiment names from MLflow.

    Returns:
        List of experiment names.
    """
    mlflow.set_tracking_uri(st.session_state.get('mlflow_server_url', MLFLOW_TRACKING_URI))
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    return [exp.name for exp in experiments if exp.lifecycle_stage == 'active']


@st.cache_data(ttl=600, show_spinner=False)
def get_metadata_from_experiments(experiment_names: tuple[str, ...]) -> dict:
    """Get metadata for selected experiments.

    Args:
        experiment_names: Tuple of experiment names.

    Returns:
        Dictionary with experiment metadata.
    """
    mlflow.set_tracking_uri(st.session_state.get('mlflow_server_url', MLFLOW_TRACKING_URI))
    client = mlflow.tracking.MlflowClient()
    
    metadata = {}
    for exp_name in experiment_names:
        try:
            exp = client.get_experiment_by_name(exp_name)
            if exp:
                metadata[exp_name] = {
                    'experiment_id': exp.experiment_id,
                    'lifecycle_stage': exp.lifecycle_stage,
                    'artifact_location': exp.artifact_location
                }
        except Exception:
            continue
    
    return metadata


@st.cache_data(ttl=600, show_spinner=False)
def get_datasets_from_experiments(experiment_names: tuple[str, ...], dataset_param: str) -> list[str]:
    """Get unique dataset names from selected experiments.

    Args:
        experiment_names: Tuple of experiment names.
        dataset_param: Name of the parameter containing dataset names.

    Returns:
        List of unique dataset names.
    """
    mlflow.set_tracking_uri(st.session_state.get('mlflow_server_url', MLFLOW_TRACKING_URI))
    client = mlflow.tracking.MlflowClient()
    
    datasets = set()
    for exp_name in experiment_names:
        try:
            exp = client.get_experiment_by_name(exp_name)
            if exp:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"params.{dataset_param} != ''",
                    max_results=1000
                )
                for run in runs:
                    if f"params.{dataset_param}" in run.data.params:
                        datasets.add(run.data.params[f"params.{dataset_param}"])
        except Exception:
            continue
    
    return sorted(list(datasets))


@st.cache_data(ttl=600, show_spinner=False)
def get_parameters_from_experiments(experiment_names: tuple[str, ...]) -> list[str]:
    """Get unique parameter names from selected experiments.

    Args:
        experiment_names: Tuple of experiment names.

    Returns:
        List of unique parameter names (without 'params.' prefix).
    """
    mlflow.set_tracking_uri(st.session_state.get('mlflow_server_url', MLFLOW_TRACKING_URI))
    client = mlflow.tracking.MlflowClient()
    
    parameters = set()
    for exp_name in experiment_names:
        try:
            exp = client.get_experiment_by_name(exp_name)
            if exp:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=10  # Sample a few runs
                )
                for run in runs:
                    for param_key in run.data.params.keys():
                        if param_key.startswith('params.'):
                            parameters.add(param_key.replace('params.', ''))
        except Exception:
            continue
    
    return sorted(list(parameters))