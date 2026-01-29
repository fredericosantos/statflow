"""
Statistical table builder utilities.

This module provides functions for constructing statistical significance tables.

statistical_table_builder.py
└── build_statistical_table()    # Main function for statistical tables

Usage:
    from statflow.utils.table_builders.statistical_table_builder import build_statistical_table
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from typing import Dict
import streamlit as st
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests

from statflow.utils.data_processing import create_group_label, get_sort_key, get_variant_from_config
from statflow.utils.data_processing import get_dataset_info
from ..table_utils import calculate_target_distribution_stats


def _prepare_data(all_runs_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare data for statistical table building."""
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
    return all_runs_df, config_order


def _find_best_configs(config_medians: Dict, config_order: List[str]) -> Tuple[str, str, str, str]:
    """Find best and second best configs per variant."""
    variant_best_configs = {}
    for config, median in config_medians.items():
        variant = get_variant_from_config(config)
        if (
            variant not in variant_best_configs
            or median < config_medians[variant_best_configs[variant]]
        ):
            variant_best_configs[variant] = config

    # Sort variants by their best config's median fitness (ascending)
    sorted_variants = sorted(
        variant_best_configs.items(), key=lambda x: config_medians[x[1]]
    )
    best_variant, best_config = sorted_variants[0]
    second_best_variant, second_best_config = (
        sorted_variants[1] if len(sorted_variants) > 1 else (None, None)
    )

    best_variant_type = best_variant
    second_best_variant_type = second_best_variant
    return best_config, second_best_config, best_variant_type, second_best_variant_type


def _calculate_p_values(config_data: Dict, config_order: List[str], best_config: str) -> Tuple[Dict, Dict]:
    """Calculate raw p-values and apply Holm-Bonferroni correction."""
    raw_p_values = {}
    best_values = config_data[best_config]

    for config in config_order:
        if config not in config_data:
            raw_p_values[config] = None
        elif config == best_config:
            raw_p_values[config] = None
        else:
            _, p_value = stats.mannwhitneyu(
                best_values, config_data[config], alternative="less"
            )
            raw_p_values[config] = p_value

    # Apply Holm-Bonferroni correction
    valid_configs = [c for c in config_order if raw_p_values.get(c) is not None]
    valid_p_values = [raw_p_values[c] for c in valid_configs]

    hb_corrected = {}
    if valid_p_values:
        reject, corrected_pvals, _, _ = multipletests(
            valid_p_values, alpha=0.05, method="holm"
        )
        for i, config in enumerate(valid_configs):
            hb_corrected[config] = {
                "reject": reject[i],
                "corrected_p": corrected_pvals[i],
            }

    return raw_p_values, hb_corrected


def _format_table(table_data: List[Dict], config_order: List[str]) -> pd.DataFrame:
    """Format the table data into DataFrame."""
    table_df = pd.DataFrame(table_data)

    # Add "H-B Sig." column
    def is_hb_significant(row):
        hb_corrected = row["_hb_corrected"]
        if not hb_corrected:
            return False
        return all(v["reject"] for v in hb_corrected.values())

    # Add "Sig. Superior" column
    def is_sig_superior(row):
        hb_corrected = row["_hb_corrected"]
        if not hb_corrected:
            return False
        best_config = None
        for config in config_order:
            if row.get(config) == "—":
                best_config = config
                break
        if not best_config:
            return False

        best_variant = get_variant_from_config(best_config)
        other_variant_configs = [
            c for c in hb_corrected.keys()
            if get_variant_from_config(c) != best_variant
        ]
        if not other_variant_configs:
            return False
        return all(hb_corrected[c]["reject"] for c in other_variant_configs)

    table_df["H-B Sig."] = table_df.apply(is_hb_significant, axis=1)
    table_df["Sig. Superior"] = table_df.apply(is_sig_superior, axis=1)

    # Drop temporary columns
    table_df = table_df.drop(columns=["_raw_p_values", "_hb_corrected"])

    # Set column order
    col_order = [
        "Dataset",
        "Samples",
        "Features",
        "Target Mean",
        "Target Median",
        "Target Std",
        "Best",
        "Best Config",
        "2nd Best",
        "Best Tree Size",
        "2nd Best Tree Size",
        "H-B Sig.",
        "Sig. Superior",
    ] + config_order
    col_order = [c for c in col_order if c in table_df.columns]
    table_df = table_df[col_order]

    return table_df


@st.cache_data(show_spinner=False)
def build_statistical_table(
    all_runs_df: pd.DataFrame, datasets_path: str = "datasets"
) -> pd.DataFrame:
    """Build statistical significance table comparing variant types to the best performer per dataset.

    Uses one-tailed Mann-Whitney U test (alternative='less') to determine if the best
    method is significantly better than each competitor. Applies Holm-Bonferroni correction
    for multiple comparisons.

    Args:
        all_runs_df: DataFrame containing runs from all datasets.

    Returns:
        DataFrame with datasets as rows and config columns,
        showing p-values comparing each to the best performer.
    """
    if all_runs_df.empty:
        return pd.DataFrame()

    all_runs_df, config_order = _prepare_data(all_runs_df)

    # Build table data
    table_data = []

    available_datasets = st.session_state['available_datasets'] if 'available_datasets' in st.session_state else []
    for dataset in available_datasets:
        dataset_runs = all_runs_df[all_runs_df["dataset_name"] == dataset]

        if dataset_runs.empty:
            continue

        # Get dataset info
        num_samples, num_features = get_dataset_info(datasets_path, dataset)

        stat_row = {
            "Dataset": dataset,
            "Samples": num_samples,
            "Features": num_features,
        }

        # Calculate target distribution statistics
        target_mean, target_median, target_std = calculate_target_distribution_stats(
            dataset
        )
        stat_row["Target Mean"] = (
            f"{target_mean:.2f}" if not np.isnan(target_mean) else "—"
        )
        stat_row["Target Median"] = (
            f"{target_median:.2f}" if not np.isnan(target_median) else "—"
        )
        stat_row["Target Std"] = (
            f"{target_std:.2f}" if not np.isnan(target_std) else "—"
        )

        # Collect data for all configs
        config_data = {}
        config_medians = {}
        config_tree_sizes = {}

        for config in config_order:
            config_runs = dataset_runs[dataset_runs["config_group"] == config]

            if not config_runs.empty:
                rmse_values = config_runs["metrics.best_test_fitness"].dropna()
                tree_values = config_runs["metrics.best_n_nodes"].dropna()
                if len(rmse_values) > 0 and len(tree_values) > 0:
                    median_rmse = rmse_values.median()
                    config_data[config] = rmse_values.values
                    config_medians[config] = median_rmse
                    config_tree_sizes[config] = tree_values.values

        # Skip dataset if no valid data
        if not config_medians:
            continue

        best_config, second_best_config, best_variant_type, second_best_variant_type = _find_best_configs(config_medians, config_order)

        # Tree size stats for best config
        if best_config in config_tree_sizes:
            tree_sizes = config_tree_sizes[best_config]
            tree_median = pd.Series(tree_sizes).median()
            tree_std = pd.Series(tree_sizes).std()
            best_tree_size = f"{tree_median:.1f} ± {tree_std:.1f}"
        else:
            best_tree_size = "N/A"

        # Tree size stats for second best config
        if second_best_config and second_best_config in config_tree_sizes:
            tree_sizes = config_tree_sizes[second_best_config]
            tree_median = pd.Series(tree_sizes).median()
            tree_std = pd.Series(tree_sizes).std()
            second_best_tree_size = f"{tree_median:.1f} ± {tree_std:.1f}"
        else:
            second_best_tree_size = "N/A"

        stat_row.update({
            "Best": best_variant_type,
            "Best Config": best_config,
            "2nd Best": second_best_variant_type if second_best_variant_type else "N/A",
            "Best Tree Size": best_tree_size,
            "2nd Best Tree Size": second_best_tree_size,
        })

        raw_p_values, hb_corrected = _calculate_p_values(config_data, config_order, best_config)

        # Populate stat_row with p-values
        for config in config_order:
            if config not in config_data:
                stat_row[config] = "N/A"
            elif config == best_config:
                stat_row[config] = "—"
            else:
                p_val = raw_p_values.get(config)
                if p_val is not None:
                    stat_row[config] = round(p_val, 4)
                else:
                    stat_row[config] = "N/A"

        # Store data for later columns
        stat_row["_raw_p_values"] = raw_p_values
        stat_row["_hb_corrected"] = hb_corrected

        table_data.append(stat_row)

    if not table_data:
        return pd.DataFrame()

    return _format_table(table_data, config_order)