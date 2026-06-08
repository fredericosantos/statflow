"""
Comparison page for the Statflow application.

Compares 'our' methods vs 'their' methods with statistical significance testing.

comparison.py
├── main()                          # Main page entry point
├── perform_statistical_tests()     # Wilcoxon rank-sum with Holm-Bonferroni
├── build_comparison_table()        # Build comparison table with stats
├── format_cell()                   # Format value with std and significance
└── style_dataframe()               # Apply styling to significant cells
"""

import streamlit as st
import polars as pl
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.colors
from plotly.subplots import make_subplots

from statflow.config import SessionState
from statflow.shared.server_status import ServerStatusManager
from statflow.functional.dataframes.data_processing import (
    fetch_experiment_data,
    apply_metric_filters,
)
from statflow.managers.naming import NamingManager
from statflow.components.filters import render_group_filter


def holm_bonferroni_correction(
    p_values: list[float], alpha: float = 0.05
) -> list[bool]:
    """Apply Holm-Bonferroni correction to p-values."""
    n = len(p_values)
    if n == 0:
        return []

    indexed_pvals = list(enumerate(p_values))
    indexed_pvals.sort(key=lambda x: x[1])

    rejected = [False] * n
    for i, (orig_idx, pval) in enumerate(indexed_pvals):
        corrected_alpha = alpha / (n - i)
        if pval <= corrected_alpha:
            rejected[orig_idx] = True
        else:
            break

    return rejected


def perform_statistical_tests(
    raw_data: pl.DataFrame,
    our_group: str,
    their_groups: list[str],
    metric: str,
    group_col: str = "group_label",
) -> dict[str, dict]:
    """Perform Wilcoxon rank-sum tests (one vs all)."""
    results = {}

    our_data = (
        raw_data.filter(pl.col(group_col) == our_group)
        .get_column(metric)
        .drop_nulls()
        .to_numpy()
    )

    if len(our_data) == 0:
        return results

    p_values = []
    group_names = []

    for their_group in their_groups:
        their_data = (
            raw_data.filter(pl.col(group_col) == their_group)
            .get_column(metric)
            .drop_nulls()
            .to_numpy()
        )

        if len(their_data) == 0:
            continue

        try:
            stat, pval = stats.mannwhitneyu(our_data, their_data, alternative="less")
            our_median = np.median(our_data)
            their_median = np.median(their_data)
            our_is_better = our_median < their_median

            p_values.append(pval)
            group_names.append(their_group)
            results[their_group] = {
                "p_value": pval,
                "our_median": our_median,
                "their_median": their_median,
                "our_is_better": our_is_better,
            }
        except Exception:
            continue

    if p_values:
        rejected = holm_bonferroni_correction(p_values)
        for i, group_name in enumerate(group_names):
            results[group_name]["is_significant"] = rejected[i]

    return results


def check_significance(
    raw_data: pl.DataFrame,
    focused_group: str,
    competitor_groups: list[str],
    metric: str,
) -> bool:
    """Check if focused_group is significantly better than ALL competitor_groups."""
    if not competitor_groups:
        return False
    res = perform_statistical_tests(raw_data, focused_group, competitor_groups, metric)
    if not res:
        return False
    return all(
        r.get("is_significant", False) and r.get("our_is_better", False)
        for r in res.values()
    )


def build_comparison_table(
    metric_df: pl.DataFrame,
    param_df: pl.DataFrame,
    metric: str,
    agg_type: str,
    our_groups: list[str],
    their_groups: list[str],
) -> tuple[pl.DataFrame, dict]:
    """Build comparison table with aggregated results."""
    if "run_id" in metric_df.columns and "run_id" in param_df.columns:
        combined = metric_df.join(
            param_df.select(["run_id", "group_label", "dataset_name"]),
            on="run_id",
            how="left",
        )
    else:
        combined = metric_df.join(
            param_df.select(["dataset_name", "group_label"]).unique(),
            on="dataset_name",
            how="left",
        )

    if combined.is_empty():
        return pl.DataFrame(), {}

    all_groups = our_groups + their_groups
    combined = combined.filter(pl.col("group_label").is_in(all_groups))

    m_col = pl.col(metric).drop_nans()
    if agg_type == "Mean ± Std":
        agg_df = combined.group_by(["dataset_name", "group_label"]).agg([
            m_col.mean().alias("value"),
            m_col.std().alias("spread"),
        ])
    else:  # Median ± IQR
        agg_df = combined.group_by(["dataset_name", "group_label"]).agg([
            m_col.median().alias("value"),
            (m_col.quantile(0.75) - m_col.quantile(0.25)).alias("spread"),
        ])

    stats_results = {}
    if our_groups and their_groups:
        datasets = combined.get_column("dataset_name").unique().to_list()
        for dataset in datasets:
            dataset_data = combined.filter(pl.col("dataset_name") == dataset)
            for our_group in our_groups:
                key = f"{dataset}_{our_group}"
                stats_results[key] = perform_statistical_tests(
                    dataset_data, our_group, their_groups, metric
                )

    return agg_df, stats_results, combined


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
            method_data = dataset_data.filter(pl.col("group_label") == method).get_column(metric).drop_nulls().to_numpy()
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
            default_datasets = [
                d for d in saved_filter if d in all_datasets
            ] or all_datasets

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
    st.markdown(
        "Compare your methods against baseline methods with statistical testing."
    )

    # Check prerequisites
    if not st.session_state["selected_experiments"]:
        st.warning("Please select experiments first in Get Started.")
        return

    if not st.session_state["selected_datasets"]:
        st.warning("Please select datasets first in Get Started.")
        return

    selected_groups = (
        filtered_group_labels
        if filtered_group_labels
        else st.session_state.get("selected_groups", [])
    )
    if not selected_groups:
        st.warning("Please select and configure groups in Parameters page.")
        return

    selected_metrics = st.session_state.get("selected_metrics", [])
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
        default_ours = [
            g for g in saved_ours if g in selected_groups
        ] or selected_groups

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

    # Metric, aggregation, decimals, pivot
    cols = st.columns([5, 2, 1, 1], vertical_alignment="center")

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
        decimals = st.number_input(
            "Decimals",
            min_value=0,
            max_value=10,
            value=st.session_state.get("comparison_decimals", 4),
            step=1,
            key="comparison_decimals",
        )

    # with cols[3]:
    #     pivot_table = st.toggle("Pivot", value=False, key="comparison_pivot")

    if not comparison_metric:
        st.info("Select a metric to compare.")
        return

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
    selected_params = st.session_state.get("selected_params", [])
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
        metric_df, param_df, comparison_metric, agg_type, our_groups, their_groups
    )

    if agg_df.is_empty():
        st.warning("No results to display.")
        return

    # Filter to datasets to show
    agg_df = agg_df.filter(pl.col("dataset_name").is_in(datasets_to_show))

    # Display legend
    st.caption(
        "🥇 = Significantly better than ALL baselines (Wilcoxon rank-sum, Holm-Bonferroni, α=0.05)"
    )

    # Build display table

    # Use user's dataset order from Get Started (datasets_to_show preserves order)
    datasets = [
        d for d in datasets_to_show if d in agg_df.get_column("dataset_name").to_list()
    ]
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
                            r.get("is_significant", False)
                            and r.get("our_is_better", False)
                            for r in our_stats.values()
                        )
                        if all_sig:
                            is_sig = True
                            sig_cells.append((row_idx, display_name))
                            trophy_counts[display_name] += 1

                if value is not None:
                    row_data[display_name] = format_cell(
                        value, spread, decimals, is_sig
                    )
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

    st.dataframe(
        achievement_df,
        width="stretch",
        hide_index=True,
        height="content"
    )

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
    full_display_df = pl.DataFrame(display_data)
    
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
                        is_sig = check_significance(dataset_data_raw, method, competitors, comparison_metric)
                        
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
