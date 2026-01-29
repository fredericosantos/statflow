"""
Table building utilities for comparison tables and statistics.

This module provides functions for constructing various comparison tables
from experiment data, including dominance counts, detailed tables,
and data aggregation and formatting.

table_builders.py
├── Dominance count table (build_dominance_count_table)
├── Per-dataset detailed table (build_per_dataset_detailed_table)
├── All datasets detailed table (build_all_datasets_detailed_table)
└── Table data aggregation and formatting

Usage:
    from statflow.utils.table_builders import (
        build_dominance_count_table, build_per_dataset_detailed_table, build_all_datasets_detailed_table
    )
"""

import pandas as pd
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests
from typing import Tuple, Dict, Any
import streamlit as st
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from statflow.config import (
    create_group_label,
    get_sort_key,
    get_dataset_info,
    get_variant_from_config,
    get_all_datasets,
)


@st.cache_data(show_spinner=False)
def calculate_scaled_std_for_dataset(dataset_name: str) -> Tuple[float, float]:
    """Calculate the standard deviation of train and test sets after StandardScaler.

    Uses 70% train / 30% test split with seeds 1-30, averages the std across seeds.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple of (train_std, test_std) averaged across seeds 1-30
    """
    datasets_dir = Path(__file__).parent.parent.parent / "datasets"
    dataset_path = datasets_dir / f"{dataset_name}.csv"

    if not dataset_path.exists():
        return float("nan"), float("nan")

    # Load dataset
    dataset_df = pd.read_csv(dataset_path, sep=",", header=0)

    train_stds = []
    test_stds = []

    # For each seed 1-30
    for seed in range(1, 31):
        # Split data (70% train, 30% test)
        train_set, test_set = train_test_split(
            dataset_df, test_size=0.3, random_state=seed, shuffle=True
        )

        # Extract features
        X_train = train_set.iloc[:, :-1].to_numpy()
        X_test = test_set.iloc[:, :-1].to_numpy()

        # Apply StandardScaler (fit on train, transform both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Calculate std of all values in the scaled arrays
        train_std = np.std(X_train_scaled)
        test_std = np.std(X_test_scaled)

        train_stds.append(train_std)
        test_stds.append(test_std)

    # Average across seeds
    avg_train_std = np.mean(train_stds)
    avg_test_std = np.mean(test_stds)

    return avg_train_std, avg_test_std


@st.cache_data(show_spinner=False)
def calculate_target_distribution_stats(
    dataset_name: str,
) -> Tuple[float, float, float]:
    """Calculate statistics about the original target distribution to assess how close it is to 0.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple of (mean, median, std) of the original target variable
    """
    datasets_dir = Path(__file__).parent.parent.parent / "datasets"
    dataset_path = datasets_dir / f"{dataset_name}.csv"

    if not dataset_path.exists():
        return float("nan"), float("nan"), float("nan")

    # Load dataset
    dataset_df = pd.read_csv(dataset_path, sep=",", header=0)

    # Extract original target (last column)
    original_target = dataset_df.iloc[:, -1].to_numpy()

    # Calculate statistics
    target_mean = np.mean(original_target)
    target_median = np.median(original_target)
    target_std = np.std(original_target)





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

    for dataset in st.session_state.get('available_datasets', []):
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

        # Identify ARC and non-ARC configs
        arc_configs = [c for c in config_stats.keys() if c.startswith("ARC")]
        non_arc_configs = [c for c in config_stats.keys() if not c.startswith("ARC")]

        # For each ARC config, test if it's significantly better than ALL non-ARC configs
        arc_superior = {}  # arc_config -> True if significantly better than all non-ARC

        for arc_config in arc_configs:
            arc_values = config_stats[arc_config]["values"]

            if not non_arc_configs or len(arc_values) <= 1:
                arc_superior[arc_config] = False
                continue

            # Collect p-values for this ARC config vs all non-ARC configs
            p_values_list = []
            configs_tested = []

            for non_arc_config in non_arc_configs:
                non_arc_values = config_stats[non_arc_config]["values"]
                if len(non_arc_values) > 1:
                    # One-tailed test: ARC < non-ARC (lower is better)
                    _, p_value = stats.mannwhitneyu(
                        arc_values, non_arc_values, alternative="less"
                    )
                    p_values_list.append(p_value)
                    configs_tested.append(non_arc_config)

            # Apply Holm-Bonferroni correction for this ARC config's comparisons
            if p_values_list and len(p_values_list) == len(non_arc_configs):
                reject, _, _, _ = multipletests(p_values_list, alpha=0.05, method="holm")
                # ARC is superior if it significantly beats ALL non-ARC configs
                arc_superior[arc_config] = all(reject)
            else:
                arc_superior[arc_config] = False

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

        significance_info[dataset] = dataset_significance

        table_data.append(row)

    if not table_data:
        return pd.DataFrame(), {}

    # Create DataFrame
    table_df = pd.DataFrame(table_data)

    # Reorder columns: Dataset, samples/features, scaled std columns, target stats, then configs in order
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




    if all_runs_df.empty:
        return pd.DataFrame()

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

    # Build table data
    table_data = []

    for dataset in st.session_state.get('available_datasets', []):
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

        # Find best config per variant
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

        # Get variant types (already have them)
        best_variant_type = best_variant
        second_best_variant_type = second_best_variant

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

        # Collect raw p-values for Holm-Bonferroni correction
        raw_p_values = {}
        best_values = config_data[best_config]

        for config in config_order:
            if config not in config_data:
                raw_p_values[config] = None
            elif config == best_config:
                raw_p_values[config] = None
            else:
                # Perform one-tailed Mann-Whitney U test (best < other)
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

    # Create DataFrame
    table_df = pd.DataFrame(table_data)

    # Add "H-B Sig." column: True if best config is significantly better after H-B correction
    def is_hb_significant(row):
        hb_corrected = row["_hb_corrected"]
        if not hb_corrected:
            return False
        # Check if ALL comparisons are significant after H-B correction
        return all(v["reject"] for v in hb_corrected.values())

    # Add "Sig. Superior" column: True if best is significantly better than ALL others
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
        # Check configs from OTHER variants
        other_variant_configs = [
            c for c in hb_corrected.keys()
            if get_variant_from_config(c) != best_variant
        ]
        if not other_variant_configs:
            return False
        return all(hb_corrected[c]["reject"] for c in other_variant_configs)

    table_df["H-B Sig."] = table_df.apply(is_hb_significant, axis=1)
    table_df["Sig. Superior"] = table_df.apply(is_sig_superior, axis=1)

    # Drop the temporary columns
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
def build_dominance_count_table(stat_df: pd.DataFrame) -> pd.DataFrame:
    """Build table counting datasets where each variant is statistically superior.

    Args:
        stat_df: DataFrame from build_statistical_table with Sig. Superior column.

    Returns:
        DataFrame with variants as rows and count of datasets where each variant
        is the best performer and statistically significantly better than ALL other variants
        (after Holm-Bonferroni correction).
    """
    if stat_df.empty or "Sig. Superior" not in stat_df.columns:
        return pd.DataFrame()

    # Get all variants that appear as best at least once
    all_best_variants = stat_df["Best"].dropna().unique()

    # Count Sig. Superior datasets for each variant
    superiority_data = []
    for variant in all_best_variants:
        # Count how many datasets this variant is statistically superior
        count = stat_df[(stat_df["Best"] == variant) & stat_df["Sig. Superior"]].shape[0]
        superiority_data.append({"Variant": variant, "Sig. Superior Datasets": count})

    if not superiority_data:
        return pd.DataFrame()

    superiority_df = pd.DataFrame(superiority_data)
    superiority_df = superiority_df.sort_values("Sig. Superior Datasets", ascending=False)

    return superiority_df


@st.cache_data(show_spinner=False)
def build_per_dataset_detailed_table(
    all_runs_df: pd.DataFrame, dataset_name: str
) -> pd.DataFrame:
    """Build detailed statistics table for a single dataset.

    Shows comprehensive statistics for each configuration including fitness and tree size metrics.

    Args:
        all_runs_df: DataFrame containing runs from all datasets.
        dataset_name: Name of the dataset to analyze.

    Returns:
        DataFrame with configurations as rows and detailed statistics as columns.
    """
    if all_runs_df.empty:
        return pd.DataFrame()

    # Filter to selected dataset
    dataset_runs = all_runs_df[all_runs_df["dataset_name"] == dataset_name].copy()

    if dataset_runs.empty:
        return pd.DataFrame()

    # Add grouping labels and sort keys
    dataset_runs["config_group"] = dataset_runs.apply(create_group_label, axis=1)
    dataset_runs["sort_key"] = dataset_runs.apply(get_sort_key, axis=1)

    # Get unique configurations sorted
    config_order = (
        dataset_runs.groupby("config_group")["sort_key"]
        .first()
        .sort_values()
        .index.tolist()
    )

    # Build table data
    table_data = []
    config_stats = {}

    # Calculate statistics for each configuration
    for config in config_order:
        config_runs = dataset_runs[dataset_runs["config_group"] == config]

        if config_runs.empty:
            continue

        fitness_values = config_runs["metrics.best_test_fitness"]
        tree_size_values = config_runs["metrics.best_n_nodes"]

        config_stats[config] = {
            "median_fitness": fitness_values.median(),
            "mean_fitness": fitness_values.mean(),
            "std_fitness": fitness_values.std(),
            "median_tree_size": tree_size_values.median(),
            "mean_tree_size": tree_size_values.mean(),
            "std_tree_size": tree_size_values.std(),
            "fitness_values": fitness_values.tolist(),
        }

    # Find best configuration (lowest median fitness)
    if not config_stats:
        return pd.DataFrame()

    best_config = min(
        config_stats.keys(), key=lambda x: config_stats[x]["median_fitness"]
    )
    best_fitness_values = config_stats[best_config]["fitness_values"]

    # Build rows with statistics and p-values
    for config in config_order:
        if config not in config_stats:
            continue

        config_stat = config_stats[config]

        row = {
            "Method": config,
            "Median Fitness": f"{config_stat['median_fitness']:.4f}",
            "Mean Fitness": f"{config_stat['mean_fitness']:.4f}",
            "Std Fitness": f"{config_stat['std_fitness']:.4f}",
            "Median Tree Size": f"{config_stat['median_tree_size']:.1f}",
            "Mean Tree Size": f"{config_stat['mean_tree_size']:.1f}",
            "Std Tree Size": f"{config_stat['std_tree_size']:.1f}",
        }

        # Calculate p-value compared to best
        if config == best_config:
            row["P-value vs Best"] = "—"
        else:
            config_fitness_values = config_stat["fitness_values"]
            if len(best_fitness_values) > 1 and len(config_fitness_values) > 1:
                from scipy import stats as scipy_stats

                _, p_value = scipy_stats.mannwhitneyu(
                    best_fitness_values,
                    config_fitness_values,
                    alternative="two-sided",
                )
                row["P-value vs Best"] = f"{p_value:.4f}"
            else:
                row["P-value vs Best"] = "N/A"

        table_data.append(row)

    # Create DataFrame
    table_df = pd.DataFrame(table_data)

    return table_df


@st.cache_data(show_spinner=False)
def build_all_datasets_detailed_table(all_runs_df: pd.DataFrame) -> pd.DataFrame:
    """Build detailed statistics table for all datasets with multi-index.

    Shows comprehensive statistics for each configuration across all datasets
    with a multi-index (Dataset, Method).

    Args:
        all_runs_df: DataFrame containing runs from all datasets.

    Returns:
        DataFrame with multi-index (Dataset, Method) and detailed statistics as columns.
    """
    if all_runs_df.empty:
        return pd.DataFrame()

    # Get list of all unique datasets
    all_datasets = sorted(all_runs_df["dataset_name"].unique().tolist())

    # Collect data for all datasets
    all_table_data = []

    for dataset_name in all_datasets:
        # Get the single-dataset table
        dataset_df = build_per_dataset_detailed_table(all_runs_df, dataset_name)

        if not dataset_df.empty:
            # Add dataset column
            dataset_df.insert(0, "Dataset", dataset_name)
            all_table_data.append(dataset_df)

    if not all_table_data:
        return pd.DataFrame()

    # Combine all datasets
    combined_df = pd.concat(all_table_data, ignore_index=True)

    # Set multi-index (Dataset, Method)
    combined_df = combined_df.set_index(["Dataset", "Method"])

    return combined_df
