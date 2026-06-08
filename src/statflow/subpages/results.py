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
import plotly.express as px

from statflow.config import SessionState
from statflow.pages_modules.module_get_started.server_status import ServerStatusManager
from statflow.functional.dataframes.data_processing import (
    fetch_experiment_data,
    apply_metric_filters,
)
from statflow.managers.naming import NamingManager
from statflow.components.filters import render_group_filter


def get_combined_data(
    metric_df: pl.DataFrame,
    param_df: pl.DataFrame,
    group_col: str,
) -> pl.DataFrame:
    """Join metrics with parameter groups."""
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
    return combined


def build_results_table(
    combined_df: pl.DataFrame,
    selected_metrics: list[str],
    group_col: str,
) -> pl.DataFrame:
    """Build aggregated results table with mean/median/std per group."""
    if combined_df.is_empty():
        return pl.DataFrame()

    # Aggregate by dataset and group
    agg_exprs = []
    for metric in selected_metrics:
        if metric in combined_df.columns:
            # Use expressions that explicitly handle NaNs and nulls for accurate stats
            m_col = pl.col(metric).drop_nans()
            agg_exprs.extend([
                m_col.mean().alias(f"{metric}_mean"),
                m_col.median().alias(f"{metric}_median"),
                m_col.std().alias(f"{metric}_std"),
                m_col.drop_nulls().count().alias(f"{metric}_n"),
            ])

    if not agg_exprs:
        return pl.DataFrame()

    group_cols = (
        ["dataset_name", group_col]
        if "dataset_name" in combined_df.columns
        else [group_col]
    )
    results = combined_df.group_by(group_cols).agg(agg_exprs)

    return results.sort(group_cols)


def render_boxplot(
    df: pl.DataFrame,
    metric: str,
    group_col: str,
    group_renames: dict,
    plot_height: int = 400,
    selected_groups: list[str] | None = None,
    points_display: str | bool = "outliers",
) -> None:
    """Render a boxplot for a specific metric."""
    if metric not in df.columns or group_col not in df.columns:
        return

    # Prepare data for plotting (convert to pandas mostly for Plotly Express compatibility check)
    # Plotly Express handles Polars directly in recent versions, but safety first:
    plot_df = df.select([group_col, metric]).to_pandas()

    # Apply renames to group column for display using map logic
    display_map = {
        g: NamingManager.get_group_name(g) for g in plot_df[group_col].unique()
    }
    plot_df[group_col] = plot_df[group_col].map(display_map)

    # Use selected_groups for display order if provided
    category_orders = None
    if selected_groups:
        display_ordered = [NamingManager.get_group_name(g) for g in selected_groups]
        category_orders = {group_col: display_ordered}

    fig = px.box(
        plot_df,
        x=group_col,
        y=metric,
        color=group_col,
        title=f"Distribution of {NamingManager.get_metric_name(metric)}",
        points=points_display,
        category_orders=category_orders,
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title=NamingManager.get_metric_name(metric),
        margin=dict(l=20, r=20, t=40, b=20),
        height=plot_height,
    )
    st.plotly_chart(fig, width='stretch')


def render_dataset_boxplots(
    raw_df: pl.DataFrame,
    dataset: str,
    metrics_to_plot: list[str],
    group_col: str,
    group_renames: dict,
    dataset_renames: dict,
    plot_height: int = 400,
    selected_groups: list[str] | None = None,
    points_display: str | bool = "outliers",
) -> None:
    """Render boxplots for a single dataset."""
    dataset_display = NamingManager.get_dataset_name(dataset)
    st.subheader(f":material/dataset: {dataset_display}")

    if raw_df is not None and not raw_df.is_empty() and metrics_to_plot:
        dataset_raw = raw_df.filter(pl.col("dataset_name") == dataset)
        for metric in metrics_to_plot:
            render_boxplot(
                dataset_raw,
                metric,
                group_col,
                group_renames,
                plot_height=plot_height,
                selected_groups=selected_groups,
                points_display=points_display,
            )


def render_dataset_table(
    results_df: pl.DataFrame,
    dataset: str,
    group_col: str,
    group_renames: dict,
) -> None:
    """Render results table for a single dataset."""
    # Prepare flat renames for Polars replace in table
    # Prepare flat renames for Polars replace in table
    flat_group_renames = {
        k: NamingManager.get_group_name(k) for k in group_renames.keys()
    }

    dataset_agg = results_df.filter(pl.col("dataset_name") == dataset)
    display_df = dataset_agg.drop("dataset_name")

    # Apply group renames
    if group_col in display_df.columns:
        display_df = display_df.with_columns(
            pl.col(group_col).replace(flat_group_renames).alias(group_col)
        )

    st.dataframe(display_df, width="content", hide_index=True)


def main():
    SessionState.initialize()

    status_manager = ServerStatusManager()
    status_manager.display_sidebar()

    with st.sidebar:
        filtered_group_labels = render_group_filter()

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

    # Apply metric filters (NaNs and ranges) silently as per Metrics settings
    metric_df = apply_metric_filters(metric_df)

    if metric_df.is_empty():
        st.warning("No data remains after applying metric filters. Check your filter settings.")
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


    # Filter to selected groups (from Parameters) + applied group filter (transient)
    if filtered_group_labels and "group_label" in param_df.columns:
        param_df = param_df.filter(pl.col("group_label").is_in(filtered_group_labels))

    # Combine data (Raw)
    raw_df = get_combined_data(metric_df, param_df, "group_label")

    # Filter out unidentified groups (null or "None")
    if "group_label" in raw_df.columns:
        raw_df = raw_df.filter(
            pl.col("group_label").is_not_null() & (pl.col("group_label") != "None")
        )

    # Build results table (Aggregated)
    results_df = build_results_table(raw_df, selected_metrics, "group_label")

    if results_df.is_empty():
        st.warning("No results to display. Check your selections.")
        return

    # Sort results_df by filtered_group_labels order if available
    if filtered_group_labels:
        order_map = {g: i for i, g in enumerate(filtered_group_labels)}
        results_df = results_df.with_columns(
            order=pl.col("group_label").replace_strict(
                list(order_map.keys()), list(order_map.values()), default=999
            )
        ).sort(["dataset_name", "order"]).drop("order")

    # 1. Metric to plot selector (Multi-select) - Moved to Sidebar
    with st.sidebar:
        st.divider()

        metrics_to_plot = st.pills(
            "Metric Charts",
            options=selected_metrics,
            default=selected_metrics[:1] if selected_metrics else None,
            selection_mode="multi",
            key="results_metrics_to_plot",
            format_func=lambda m: f":material/candlestick_chart: {NamingManager.get_metric_name(m)}",
        )

        # 2. Dataset Selector (Single Select - Acts as Tab)
        all_datasets = st.session_state["selected_datasets"]

        # Ensure default is valid
        default_dataset = st.session_state.get("results_selected_dataset")
        if default_dataset not in all_datasets:
            default_dataset = all_datasets[0] if all_datasets else None

        st.divider()
        selected_dataset = st.pills(
            "Select Dataset",
            options=all_datasets,
            default=default_dataset,
            selection_mode="single",
            key="results_selected_dataset",
            format_func=lambda d: f":material/dataset: {NamingManager.get_dataset_name(d)}",
        )

        st.divider()
        with st.expander("Visualization Settings", expanded=True, icon=":material/settings:"):
            plot_height = st.slider(
                "Chart Height",
                min_value=200,
                max_value=1200,
                value=st.session_state.get("plot_height", 400),
                step=50,
                key="plot_height",
                help="Adjust the height of the boxplots.",
                on_change=lambda: SessionState.save_key_to_config("plot_height"),
            )

            # Map for user-friendly display options
            points_options = {
                "Outliers Only": "outliers",
                "All Points": "all",
                "Suspected Outliers": "suspectedoutliers",
                "None": False,
            }
            
            # Find current label or default
            current_val = st.session_state.get("points_display", "outliers")
            current_label = next((k for k, v in points_options.items() if v == current_val), "Outliers Only")

            selected_label = st.selectbox(
                "Boxplots Points Display",
                options=list(points_options.keys()),
                index=list(points_options.keys()).index(current_label),
                key="points_display_label",
                help="Choose which data points to show on boxplots.",
            )
            
            # Save actual value if changed
            new_val = points_options[selected_label]
            if new_val != current_val:
                st.session_state.points_display = new_val
                SessionState.save_to_config()
            
            # Also get points_display for internal use
            points_display = st.session_state.points_display
    
    st.title(":material/insights: Results")
    st.markdown("View aggregated experiment results by group and dataset.")

    if not selected_dataset:
        st.info("Select a dataset to display.")
        return

    # 3. Boxplots
    group_renames = st.session_state.get("group_renames", {})
    dataset_renames = st.session_state.get("dataset_renames", {})

    render_dataset_boxplots(
        raw_df,
        selected_dataset,
        metrics_to_plot,
        "group_label",
        group_renames,
        dataset_renames,
        plot_height=plot_height,
        selected_groups=filtered_group_labels,
        points_display=points_display,
    )

    # 4. Checkboxes (Display options)
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_mean = st.checkbox(
            "Show Mean",
            value=st.session_state.get("show_mean", True),
            key="show_mean",
            on_change=lambda: SessionState.save_to_config(),
        )
    with col2:
        show_median = st.checkbox(
            "Show Median",
            value=st.session_state.get("show_median", False),
            key="show_median",
            on_change=lambda: SessionState.save_to_config(),
        )
    with col3:
        show_std = st.checkbox(
            "Show Std Dev",
            value=st.session_state.get("show_std", True),
            key="show_std",
            on_change=lambda: SessionState.save_to_config(),
        )
    with col4:
        show_n = st.checkbox(
            "Show Count",
            value=st.session_state.get("show_count", False),
            key="show_count",
            on_change=lambda: SessionState.save_to_config(),
        )

    # Filter columns based on options
    cols_to_show = (
        ["dataset_name", "group_label"]
        if "dataset_name" in results_df.columns
        else ["group_label"]
    )
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
    agg_display_df = results_df.select(cols_to_show)

    # 5. Dataframe
    render_dataset_table(agg_display_df, selected_dataset, "group_label", group_renames)


if __name__ == "__main__":
    main()
