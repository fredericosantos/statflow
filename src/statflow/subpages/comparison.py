"""
Comparison page for the Statflow application.

Compares 'our' methods vs 'their' methods with statistical significance testing.
Pure statistical logic lives in ``functional/statistics.py``; this module owns
the Streamlit UI only.

comparison.py
├── main()                          # Main page entry point
├── render_comparison_boxplots()    # Boxplot subplots with trophy annotations
├── format_cell()                   # Format value with std and significance
└── _render_direction_control()     # Per-metric better-is-lower/higher control
"""

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

from statflow.components.filters import render_group_filter
from statflow.config import SessionState
from statflow.functional.dataframes.data_processing import (
    apply_metric_filters,
    fetch_experiment_data,
)
from statflow.functional.statistics import (
    build_comparison_table,
    check_significance,
)
from statflow.managers.naming import NamingManager
from statflow.shared.server_status import ServerStatusManager


def render_comparison_boxplots(
    raw_data: pl.DataFrame,
    datasets: list[str],
    visible_methods: list[str],
    all_methods: list[str],
    metric: str,
    winners_per_dataset: dict[str, list[str]],
    plot_height: int = 400,
) -> None:
    """Render boxplot subplots with trophy annotations."""
    if not datasets:
        return

    n_datasets = len(datasets)
    cols = 2
    rows = (n_datasets + 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[NamingManager.get_dataset_name(d) for d in datasets],
        # vertical_spacing=0.1,
    )

    # Use standard Plotly qualitative palette
    colors = plotly.colors.qualitative.Plotly
    # Use all_methods for stable color mapping across focused method switches
    method_to_color = {m: colors[i % len(colors)] for i, m in enumerate(all_methods)}

    for i, dataset in enumerate(datasets):
        row = (i // cols) + 1
        col = (i % cols) + 1

        dataset_data = raw_data.filter(pl.col("dataset_name") == dataset)
        winners = winners_per_dataset.get(dataset, [])

        for method in visible_methods:
            method_data = (
                dataset_data.filter(pl.col("group_label") == method)
                .get_column(metric)
                .drop_nulls()
                .to_numpy()
            )
            display_name = NamingManager.get_group_name(method)

            if len(method_data) > 0:
                fig.add_trace(
                    go.Box(
                        y=method_data,
                        name=display_name,
                        boxpoints="outliers",
                        marker_color=method_to_color[method],
                        showlegend=(i == 0),  # Show legend only for the first subplot
                    ),
                    row=row,
                    col=col,
                )

                # Add trophy annotation if winner
                if method in winners:
                    # Position trophy above the max value
                    y_pos = np.max(method_data)
                    fig.add_annotation(
                        x=display_name,
                        y=y_pos,
                        text="🥇",
                        showarrow=False,
                        yshift=15,
                        font=dict(size=20),
                        row=row,
                        col=col,
                    )

    fig.update_layout(
        height=plot_height * rows,
        title_text=f"Boxplots - {NamingManager.get_metric_name(metric)}",
        showlegend=True,
        # template="plotly_white",
        margin=dict(t=80, b=50, l=50, r=50),
    )

    st.plotly_chart(fig, use_container_width=True)


def format_cell(
    value: float,
    spread: float,
    decimals: int = 4,
    is_significant: bool = False,
) -> str:
    """Format cell value with spread (std or IQR) and significance marker."""
    if value is None:
        return "-"
    formatted = f"{value:.{decimals}f} ± {spread:.{decimals}f}"
    if is_significant:
        formatted += " 🥇"
    return formatted


def _render_direction_control(metric: str | None) -> bool:
    """Per-metric "better is lower/higher" control; returns whether to maximize.

    Remembers each metric's direction in the persisted `metric_directions` map,
    so error-like metrics stay "Lower" and score-like metrics stay "Higher"
    across sessions without re-picking.
    """
    if not metric:
        return False

    directions = st.session_state["metric_directions"]
    default = directions.get(metric, "Lower")
    choice = st.pills(
        "Better is",
        options=["Lower", "Higher"],
        default=default,
        selection_mode="single",
        key=f"comparison_direction_{metric}",
    )
    choice = choice or default

    if directions.get(metric) != choice:
        directions[metric] = choice
        st.session_state["metric_directions"] = directions
        SessionState.save_key_to_config("metric_directions")

    return choice == "Higher"


def main():
    SessionState.initialize()

    status_manager = ServerStatusManager()
    status_manager.display_sidebar()

    with st.sidebar:
        filtered_group_labels = render_group_filter()

        # Dataset filter
        all_datasets = st.session_state["selected_datasets"]
        saved_filter = st.session_state.get("comparison_dataset_filter")
        if saved_filter is None or not saved_filter:
            default_datasets = all_datasets
        else:
            default_datasets = [d for d in saved_filter if d in all_datasets] or all_datasets

        with st.expander("Dataset Filter", expanded=False, icon=":material/dataset:"):
            datasets_to_show = st.pills(
                "Filter Datasets",
                options=all_datasets,
                default=default_datasets,
                selection_mode="multi",
                key="comparison_dataset_filter",
                label_visibility="collapsed",
                format_func=NamingManager.get_dataset_name,
            )

        # Ensure we have a valid selection for the analytical pipeline
        # If the widget returns None or empty (e.g. on load), fallback to default
        if not datasets_to_show and default_datasets:
            datasets_to_show = default_datasets

    st.title(":material/trophy: Comparison")
    st.markdown("Compare your methods against baseline methods with statistical testing.")

    # Check prerequisites
    if not st.session_state["selected_experiments"]:
        st.warning("Please select experiments first in Get Started.")
        return

    if not st.session_state["selected_datasets"]:
        st.warning("Please select datasets first in Get Started.")
        return

    selected_groups = (
        filtered_group_labels if filtered_group_labels else st.session_state["selected_groups"]
    )
    if not selected_groups:
        st.warning("Please select and configure groups in Parameters page.")
        return

    selected_metrics = st.session_state["selected_metrics"]
    if not selected_metrics:
        st.warning("Please select metrics in Metrics page.")
        return

    if not datasets_to_show:
        st.info("Select at least one dataset in the sidebar.")
        return

    st.divider()

    # Single pills: selected = Ours, not selected = Theirs
    saved_ours = st.session_state.get("comparison_our_groups")
    if saved_ours is None or not saved_ours:
        default_ours = selected_groups
    else:
        default_ours = [g for g in saved_ours if g in selected_groups] or selected_groups

    st.markdown("##### Our Methods :violet[(selected)] vs Baseline (not selected)")

    def format_group(g):
        return NamingManager.get_group_name(g)

    our_groups = st.pills(
        "Select your methods",
        options=selected_groups,
        default=default_ours,
        selection_mode="multi",
        key="comparison_our_groups",
        label_visibility="collapsed",
        format_func=format_group,
    )

    their_groups = [g for g in selected_groups if g not in our_groups]

    if not our_groups:
        st.info("Select at least one of your methods.")
        return

    if not their_groups:
        st.info("Deselect at least one method to use as baseline.")
        return

    st.divider()

    # Metric, aggregation, direction, decimals
    cols = st.columns([4, 2, 2, 1], vertical_alignment="center")

    def format_metric(m):
        return NamingManager.get_metric_name(m)

    with cols[0]:
        comparison_metric = st.pills(
            "Comparison Metric",
            options=selected_metrics,
            default=selected_metrics[0] if selected_metrics else None,
            selection_mode="single",
            key="comparison_metric",
            format_func=format_metric,
        )

    with cols[1]:
        agg_type = st.pills(
            "Aggregation",
            options=["Mean ± Std", "Median ± IQR"],
            default="Mean ± Std",
            selection_mode="single",
            key="comparison_agg_type",
        )

    with cols[2]:
        maximize = _render_direction_control(comparison_metric)

    with cols[3]:
        decimals = st.number_input(
            "Decimals",
            min_value=0,
            max_value=10,
            value=st.session_state.get("comparison_decimals", 4),
            step=1,
            key="comparison_decimals",
        )

    if not comparison_metric:
        st.info("Select a metric to compare.")
        return

    # agg_type from st.pills(single) is str | None; default if somehow None.
    agg_type_str: str = agg_type or "Mean ± Std"

    # Fetch data
    with st.spinner("Loading data..."):
        metric_df = fetch_experiment_data("metrics.")
        param_df = fetch_experiment_data("params.")

    if metric_df.is_empty() or param_df.is_empty():
        st.error("No data found.")
        return

    # Apply metric filters (NaNs and ranges) silently as per Metrics settings
    metric_df = apply_metric_filters(metric_df)

    if metric_df.is_empty():
        st.warning("No data remains after applying metric filters.")
        return

    # Build group labels
    selected_params = st.session_state["selected_params"]
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

    # Build comparison table
    agg_df, stats_results, combined_data = build_comparison_table(
        metric_df,
        param_df,
        comparison_metric,
        agg_type_str,
        our_groups,
        their_groups,
        maximize=maximize,
    )

    if agg_df.is_empty():
        st.warning("No results to display.")
        return

    # Filter to datasets to show
    agg_df = agg_df.filter(pl.col("dataset_name").is_in(datasets_to_show))

    # Display legend
    better = "higher" if maximize else "lower"
    st.caption(
        f"🥇 = Significantly better than ALL baselines — {better} "
        f"{NamingManager.get_metric_name(comparison_metric)} "
        "(Wilcoxon rank-sum, Holm-Bonferroni, α=0.05)"
    )

    # Build display table

    # Use user's dataset order from Get Started (datasets_to_show preserves order)
    datasets = [d for d in datasets_to_show if d in agg_df.get_column("dataset_name").to_list()]
    all_methods = our_groups + their_groups

    # Create display data with significance tracking
    display_data = []
    sig_cells = []  # Track (row_idx, col_name) for significant cells

    # Track trophies per our methods
    trophy_counts = {NamingManager.get_group_name(m): 0 for m in our_groups}

    for row_idx, dataset in enumerate(datasets):
        row_data = {"Dataset": NamingManager.get_dataset_name(dataset)}

        for method in all_methods:
            is_ours = method in our_groups
            display_name = NamingManager.get_group_name(method)

            row = agg_df.filter(
                (pl.col("dataset_name") == dataset) & (pl.col("group_label") == method)
            )

            if not row.is_empty():
                value = row.get_column("value")[0]
                spread = row.get_column("spread")[0] or 0

                is_sig = False
                if is_ours:
                    our_stats = stats_results.get(f"{dataset}_{method}", {})
                    if our_stats:
                        all_sig = all(
                            r.get("is_significant", False) and r.get("our_is_better", False)
                            for r in our_stats.values()
                        )
                        if all_sig:
                            is_sig = True
                            sig_cells.append((row_idx, display_name))
                            trophy_counts[display_name] += 1

                if value is not None:
                    row_data[display_name] = format_cell(value, spread, decimals, is_sig)
                else:
                    row_data[display_name] = "-"
            else:
                row_data[display_name] = "-"

        display_data.append(row_data)

    if not display_data:
        st.warning("No data to display.")
        return

    # 5. Achievement Summary
    st.markdown("#### Ranking")
    achievement_data = [
        {"Method": method, "Count 🥇": count} for method, count in trophy_counts.items()
    ]
    # Sort by wins descending
    achievement_df = pl.DataFrame(achievement_data).sort("Count 🥇", descending=True)

    st.dataframe(achievement_df, width="stretch", hide_index=True, height="content")

    # 6. Detailed Results
    st.markdown("#### Per Method Results")
    our_method = st.pills(
        "Per Method results",
        options=our_groups,
        key="comparison_method",
        selection_mode="single",
        format_func=format_group,
        label_visibility="collapsed",
    )

    if our_method:
        # Re-calculate display data for the focused view to include mutual trophies
        focused_display_data = []
        visible_methods = [our_method] + their_groups

        # Track total trophies for the "Total" row
        column_trophies = {NamingManager.get_group_name(m): 0 for m in visible_methods}
        winners_per_dataset = {}  # {dataset_id: [method_id, ...]}

        # We need the raw data for significance checks
        with st.spinner("Calculating mutual trophies..."):
            metric_df_raw = combined_data.select(["dataset_name", "group_label", comparison_metric])

            for dataset in datasets:
                row_data = {"Dataset": NamingManager.get_dataset_name(dataset)}
                dataset_data_raw = metric_df_raw.filter(pl.col("dataset_name") == dataset)
                winners_per_dataset[dataset] = []

                for method in visible_methods:
                    display_name = NamingManager.get_group_name(method)

                    row = agg_df.filter(
                        (pl.col("dataset_name") == dataset) & (pl.col("group_label") == method)
                    )

                    if not row.is_empty():
                        value = row.get_column("value")[0]
                        spread = row.get_column("spread")[0] or 0

                        # Check significance vs other visible methods
                        competitors = [m for m in visible_methods if m != method]
                        is_sig = check_significance(
                            dataset_data_raw,
                            method,
                            competitors,
                            comparison_metric,
                            maximize=maximize,
                        )

                        if value is not None:
                            row_data[display_name] = format_cell(value, spread, decimals, is_sig)
                            if is_sig:
                                column_trophies[display_name] += 1
                                winners_per_dataset[dataset].append(method)
                        else:
                            row_data[display_name] = "-"
                    else:
                        row_data[display_name] = "-"

                focused_display_data.append(row_data)

            # Add Total row
            total_row = {"Dataset": "Total"}
            for name, count in column_trophies.items():
                total_row[name] = f"{count} 🥇"
            focused_display_data.append(total_row)

        display_df = pl.DataFrame(focused_display_data)

        # # Pivot if requested (transpose: methods as rows, datasets as columns)
        # if pivot_table:
        #     # Melt and pivot
        #     method_cols = [NamingManager.get_group_name(m) for m in all_methods]
        #     melted = display_df.unpivot(
        #         index="Dataset",
        #         on=method_cols,
        #         variable_name="Method",
        #         value_name="Value",
        #     )
        #     display_df = melted.pivot(
        #         on="Dataset",
        #         index="Method",
        #         values="Value",
        #     )

        # Display with styling
        st.dataframe(display_df, width="stretch", hide_index=True, height="content")

        # 7. Visual Comparison
        st.divider()
        plot_height = st.session_state.get("plot_height", 400)
        render_comparison_boxplots(
            raw_data=combined_data,
            datasets=datasets,
            visible_methods=visible_methods,
            all_methods=our_groups + their_groups,
            metric=comparison_metric,
            winners_per_dataset=winners_per_dataset,
            plot_height=plot_height,
        )


if __name__ == "__main__":
    main()
