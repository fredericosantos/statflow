"""
Results page for the Statflow application.

This page displays experiment results organized by groups and datasets.

results.py
├── main()                          # Main page entry point
├── build_results_table()           # Build aggregated results table
├── get_display_name()              # Extract display name from renames
└── render_results_by_dataset()     # Render results grouped by dataset
"""

import streamlit as st
import polars as pl

from statflow.config import SessionState
from statflow.shared.server_status import ServerStatusManager
from statflow.functional.dataframes.data_processing import fetch_experiment_data


st.set_page_config(
    page_title=f"Results - {st.session_state['app_name']}",
    page_icon=":material/table_chart:",
    layout="wide",
)


def get_display_name(rename_value: str | dict | None, original: str) -> str:
    """Extract display name from rename value (handles dict or string)."""
    if rename_value is None:
        return original
    if isinstance(rename_value, dict):
        return rename_value.get("display_name", original)
    return rename_value


def build_results_table(
    metric_df: pl.DataFrame,
    param_df: pl.DataFrame,
    selected_metrics: list[str],
    group_col: str,
) -> pl.DataFrame:
    """Build aggregated results table with mean/median/std per group."""
    # Join metrics with group labels
    if "run_id" in metric_df.columns and "run_id" in param_df.columns:
        combined = metric_df.join(
            param_df.select(["run_id", group_col]), on="run_id", how="left"
        )
    else:
        if "dataset_name" in metric_df.columns and "dataset_name" in param_df.columns:
            combined = metric_df.join(
                param_df.select(["dataset_name", group_col]).unique(),
                on="dataset_name",
                how="left",
            )
        else:
            combined = metric_df.with_columns(pl.lit("Default").alias(group_col))

    if combined.is_empty():
        return pl.DataFrame()

    # Aggregate by dataset and group
    agg_exprs = []
    for metric in selected_metrics:
        if metric in combined.columns:
            agg_exprs.extend([
                pl.col(metric).mean().alias(f"{metric}_mean"),
                pl.col(metric).median().alias(f"{metric}_median"),
                pl.col(metric).std().alias(f"{metric}_std"),
                pl.col(metric).count().alias(f"{metric}_n"),
            ])

    if not agg_exprs:
        return pl.DataFrame()

    group_cols = ["dataset_name", group_col] if "dataset_name" in combined.columns else [group_col]
    results = combined.group_by(group_cols).agg(agg_exprs)

    return results.sort(group_cols)


def render_results_by_dataset(
    results_df: pl.DataFrame, group_col: str, datasets_to_show: list[str]
) -> None:
    """Render results tables grouped by dataset."""
    if results_df.is_empty():
        st.warning("No results to display.")
        return

    group_renames = st.session_state.get("group_renames", {})
    dataset_renames = st.session_state.get("dataset_renames", {})

    if "dataset_name" in results_df.columns:
        datasets = results_df.get_column("dataset_name").unique().sort().to_list()
        # Filter to only show requested datasets
        datasets = [d for d in datasets if d in datasets_to_show]

        for dataset in datasets:
            rename_value = dataset_renames.get(dataset)
            dataset_display = get_display_name(rename_value, dataset)
            st.subheader(f":material/folder: {dataset_display}")

            dataset_df = results_df.filter(pl.col("dataset_name") == dataset)
            display_df = dataset_df.drop("dataset_name")

            # Apply group renames
            if group_col in display_df.columns:
                display_df = display_df.with_columns(
                    pl.col(group_col).replace(group_renames).alias(group_col)
                )

            st.dataframe(display_df, width='content', hide_index=True)
    else:
        display_df = results_df
        if group_col in display_df.columns:
            display_df = display_df.with_columns(
                pl.col(group_col).replace(group_renames).alias(group_col)
            )
        st.dataframe(display_df, width='content', hide_index=True)


def main():
    SessionState.initialize()

    status_manager = ServerStatusManager()
    status_manager.display_sidebar()

    st.title(":material/table_chart: Results")
    st.markdown("View aggregated experiment results by group and dataset.")

    # Check prerequisites
    if not st.session_state["selected_experiments"]:
        st.warning("Please select experiments first in Get Started.")
        return

    if not st.session_state["selected_datasets"]:
        st.warning("Please select datasets first in Get Started.")
        return

    selected_metrics = st.session_state.get("selected_metrics", [])
    if not selected_metrics:
        st.warning("Please select metrics first in Metrics page.")
        return

    selected_params = st.session_state.get("selected_params", [])
    if not selected_params:
        st.warning("Please select parameters first in Parameters page.")
        return

    # Fetch data
    with st.spinner("Loading experiment data..."):
        metric_df = fetch_experiment_data("metrics.")
        param_df = fetch_experiment_data("params.")

    if metric_df.is_empty():
        st.error("No metric data found.")
        return

    if param_df.is_empty():
        st.error("No parameter data found.")
        return

    # Build group labels from selected params
    exprs = []
    for i, p in enumerate(selected_params):
        if p not in param_df.columns:
            continue
        if i > 0:
            exprs.append(pl.lit(", "))
        exprs.append(pl.lit(f"{p}="))
        exprs.append(pl.col(p).cast(pl.Utf8))

    if exprs:
        param_df = param_df.with_columns(pl.concat_str(exprs).alias("group_label"))
    else:
        param_df = param_df.with_columns(pl.lit("Default").alias("group_label"))

    # Filter to selected groups if any
    selected_groups = st.session_state.get("selected_groups", [])
    if selected_groups and "group_label" in param_df.columns:
        param_df = param_df.filter(pl.col("group_label").is_in(selected_groups))

    # Build results table
    results_df = build_results_table(
        metric_df, param_df, selected_metrics, "group_label"
    )

    if results_df.is_empty():
        st.warning("No results to display. Check your selections.")
        return

    # Dataset filter using pills
    all_datasets = st.session_state["selected_datasets"]
    datasets_to_show = st.pills(
        "Filter Datasets",
        options=all_datasets,
        default=all_datasets,
        selection_mode="multi",
        key="results_dataset_filter",
    )

    if not datasets_to_show:
        st.info("Select at least one dataset to display.")
        return

    # Display options - horizontal row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_mean = st.checkbox(
            "Show Mean",
            value=st.session_state.get("show_mean", True),
            key="show_mean",
        )
    with col2:
        show_median = st.checkbox(
            "Show Median",
            value=st.session_state.get("show_median", False),
            key="show_median",
        )
    with col3:
        show_std = st.checkbox(
            "Show Std Dev",
            value=st.session_state.get("show_std", True),
            key="show_std",
        )
    with col4:
        show_n = st.checkbox(
            "Show Count",
            value=st.session_state.get("show_count", False),
            key="show_count",
        )

    # Filter columns based on options
    cols_to_show = ["dataset_name", "group_label"] if "dataset_name" in results_df.columns else ["group_label"]
    for metric in selected_metrics:
        if show_mean and f"{metric}_mean" in results_df.columns:
            cols_to_show.append(f"{metric}_mean")
        if show_median and f"{metric}_median" in results_df.columns:
            cols_to_show.append(f"{metric}_median")
        if show_std and f"{metric}_std" in results_df.columns:
            cols_to_show.append(f"{metric}_std")
        if show_n and f"{metric}_n" in results_df.columns:
            cols_to_show.append(f"{metric}_n")

    # Filter to existing columns
    cols_to_show = [c for c in cols_to_show if c in results_df.columns]
    display_df = results_df.select(cols_to_show)

    # Render results
    st.divider()
    render_results_by_dataset(display_df, "group_label", datasets_to_show)


if __name__ == "__main__":
    main()
