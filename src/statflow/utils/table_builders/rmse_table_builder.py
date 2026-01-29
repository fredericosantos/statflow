"""
RMSE table builder utilities.

This module provides functions for constructing RMSE comparison tables with significance testing.

rmse_table_builder.py
├── _calculate_config_stats()     # Helper for config statistics
├── _identify_arc_configs()       # Helper for ARC/non-ARC identification
├── _perform_significance_tests() # Helper for Holm-Bonferroni tests
└── build_rmse_table()            # Main function for RMSE tables

Usage:
    from statflow.utils.table_builders.rmse_table_builder import build_rmse_table
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import streamlit as st
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests

from statflow.utils.data_processing import create_group_label, get_sort_key
from ..table_utils import calculate_scaled_std_for_dataset, calculate_target_distribution_stats


def _calculate_config_stats(dataset_runs: pd.DataFrame, config_order: List[str]) -> Dict:
    """Calculate statistics for each configuration."""
    config_stats = {}
    for config in config_order:
        config_runs = dataset_runs[dataset_runs["config_group"] == config]
        if not config_runs.empty:
            rmse_values = config_runs["metrics.best_test_fitness"]
            config_stats[config] = {
                "median": rmse_values.median(),
                "std": rmse_values.std(),
                "values": rmse_values.tolist(),
            }
    return config_stats


def _identify_arc_configs(config_stats: Dict) -> Tuple[List[str], List[str]]:
    """Identify ARC and non-ARC configurations."""
    arc_configs = [c for c in config_stats.keys() if c.startswith("ARC")]
    non_arc_configs = [c for c in config_stats.keys() if not c.startswith("ARC")]
    return arc_configs, non_arc_configs


def _perform_arc_superiority_tests(config_stats: Dict, arc_configs: List[str], non_arc_configs: List[str]) -> Dict[str, bool]:
    """Perform Holm-Bonferroni corrected tests for each ARC config vs all non-ARC."""
    arc_superior = {}
    for arc_config in arc_configs:
        arc_values = config_stats[arc_config]["values"]

        if not non_arc_configs or len(arc_values) <= 1:
            arc_superior[arc_config] = False
            continue

        p_values_list = []
        configs_tested = []

        for non_arc_config in non_arc_configs:
            non_arc_values = config_stats[non_arc_config]["values"]
            if len(non_arc_values) > 1:
                _, p_value = stats.mannwhitneyu(
                    arc_values, non_arc_values, alternative="less"
                )
                p_values_list.append(p_value)
                configs_tested.append(non_arc_config)

        if p_values_list and len(p_values_list) == len(non_arc_configs):
            reject, _, _, _ = multipletests(p_values_list, alpha=0.05, method="holm")
            arc_superior[arc_config] = all(reject)
        else:
            arc_superior[arc_config] = False

    return arc_superior


@st.cache_data(show_spinner=False)
def build_rmse_table(all_runs_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Build RMSE comparison table across datasets with Holm-Bonferroni corrected significance.

    Uses one-tailed Mann-Whitney U test (alternative='less') to determine if the best
    method is significantly better than each competitor. Applies Holm-Bonferroni correction
    for multiple comparisons.

    Args:
        all_runs_df: DataFrame containing runs from all datasets.

    Returns:
        Tuple of (DataFrame with datasets as rows and configs as columns,
                  dict with significance info per dataset)
    """
    if all_runs_df.empty:
        return pd.DataFrame(), {}

    # Add grouping labels and sort keys
    all_runs_df = all_runs_df.copy()
    all_runs_df["config_group"] = all_runs_df.apply(create_group_label, axis=1)
    all_runs_df["sort_key"] = all_runs_df.apply(get_sort_key, axis=1)
    all_runs_df["variant"] = all_runs_df["params.variant"]

    # Get unique configurations sorted
    config_order = (
        all_runs_df.groupby("config_group")["sort_key"]
        .first()
        .sort_values()
        .index.tolist()
    )

    # Build table data and statistical significance
    table_data = []
    significance_info = {}

    available_datasets = st.session_state['available_datasets'] if 'available_datasets' in st.session_state else []
    for dataset in available_datasets:
        dataset_runs = all_runs_df[all_runs_df["dataset_name"] == dataset]

        if dataset_runs.empty:
            continue

        row = {"Dataset": dataset}

        # Get dataset info
        from statflow.utils.data_processing import get_dataset_info

        datasets_path = Path(__file__).parent.parent.parent / "datasets"
        num_samples, num_features = get_dataset_info(str(datasets_path), dataset)
        row["Samples"] = num_samples if num_samples != -1 else "—"
        row["Features"] = num_features if num_features != -1 else "—"

        # Calculate scaled standard deviations for this dataset
        train_std_scaled, test_std_scaled = calculate_scaled_std_for_dataset(dataset)
        row["Train Std (Scaled)"] = (
            f"{train_std_scaled:.2f}" if not np.isnan(train_std_scaled) else "—"
        )
        row["Test Std (Scaled)"] = (
            f"{test_std_scaled:.2f}" if not np.isnan(test_std_scaled) else "—"
        )

        # Calculate target distribution statistics
        target_mean, target_median, target_std = calculate_target_distribution_stats(
            dataset
        )
        row["Target Mean"] = f"{target_mean:.2f}" if not np.isnan(target_mean) else "—"
        row["Target Median"] = (
            f"{target_median:.2f}" if not np.isnan(target_median) else "—"
        )
        row["Target Std"] = f"{target_std:.2f}" if not np.isnan(target_std) else "—"

        dataset_significance = {}

        # Calculate statistics for each configuration
        config_stats = _calculate_config_stats(dataset_runs, config_order)

        # Identify ARC and non-ARC configs
        arc_configs, non_arc_configs = _identify_arc_configs(config_stats)

        # Perform superiority tests
        arc_superior = _perform_arc_superiority_tests(config_stats, arc_configs, non_arc_configs)

        for config in config_order:
            if config in config_stats:
                median = config_stats[config]["median"]
                std = config_stats[config]["std"]
                value_str = f"{median:.2f} ± {std:.2f}"

                # Add † if this ARC config is significantly better than all non-ARC configs
                if arc_superior.get(config, False):
                    value_str += " †"

                row[config] = value_str
                dataset_significance[config] = arc_superior.get(config, False)
            else:
                row[config] = "—"

        table_data.append(row)
        significance_info[dataset] = dataset_significance

    if not table_data:
        return pd.DataFrame(), {}

    # Create DataFrame
    table_df = pd.DataFrame(table_data)

    # Reorder columns
    col_order = [
        "Dataset",
        "Samples",
        "Features",
        "Train Std (Scaled)",
        "Test Std (Scaled)",
        "Target Mean",
        "Target Median",
        "Target Std",
    ] + [c for c in config_order if c in table_df.columns]

    table_df = table_df[col_order]

    return table_df, significance_info