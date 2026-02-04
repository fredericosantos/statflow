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

from statflow.config import SessionState
from statflow.shared.server_status import ServerStatusManager
from statflow.functional.dataframes.data_processing import fetch_experiment_data


st.set_page_config(
    page_title=f"Comparison - {st.session_state['app_name']}",
    page_icon=":material/compare:",
    layout="wide",
)

MERMAID_DIAGRAM = """
```mermaid
flowchart LR
    subgraph Ours["Our Method(s)"]
        O1["Method A"]
    end
    subgraph Theirs["Baseline Methods"]
        T1["Baseline 1"]
        T2["Baseline 2"]
        T3["Baseline N"]
    end
    O1 -->|"Wilcoxon test"| T1
    O1 -->|"Wilcoxon test"| T2
    O1 -->|"Wilcoxon test"| T3
    T1 & T2 & T3 -->|"Holm-Bonferroni"| C["Corrected p-values"]
    C -->|"p < 0.05"| S["● Significant"]
```
"""


def get_display_name(rename_value: str | dict | None, original: str) -> str:
    """Extract display name from rename value."""
    if rename_value is None:
        return original
    if isinstance(rename_value, dict):
        return rename_value.get("display_name", original)
    return rename_value


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
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

    our_data = raw_data.filter(pl.col(group_col) == our_group).get_column(metric).drop_nulls().to_numpy()

    if len(our_data) == 0:
        return results

    p_values = []
    group_names = []

    for their_group in their_groups:
        their_data = raw_data.filter(pl.col(group_col) == their_group).get_column(metric).drop_nulls().to_numpy()

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

    # Use appropriate spread measure: Std for mean, IQR for median
    if agg_type == "Mean ± Std":
        agg_df = combined.group_by(["dataset_name", "group_label"]).agg([
            pl.col(metric).mean().alias("value"),
            pl.col(metric).std().alias("spread"),
        ])
    else:  # Median ± IQR
        agg_df = combined.group_by(["dataset_name", "group_label"]).agg([
            pl.col(metric).median().alias("value"),
            (pl.col(metric).quantile(0.75) - pl.col(metric).quantile(0.25)).alias("spread"),
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

    return agg_df, stats_results


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

    # Title with mermaid diagram
    col_title, col_diagram = st.columns([2, 1])
    with col_title:
        st.title(":material/compare: Comparison")
        st.markdown("Compare your methods against baseline methods with statistical testing.")
    with col_diagram:
        with st.expander("How it works", expanded=False, icon=":material/help:"):
            st.markdown(MERMAID_DIAGRAM)

    # Check prerequisites
    if not st.session_state["selected_experiments"]:
        st.warning("Please select experiments first in Get Started.")
        return

    if not st.session_state["selected_datasets"]:
        st.warning("Please select datasets first in Get Started.")
        return

    selected_groups = st.session_state.get("selected_groups", [])
    if not selected_groups:
        st.warning("Please select and configure groups in Parameters page.")
        return

    selected_metrics = st.session_state.get("selected_metrics", [])
    if not selected_metrics:
        st.warning("Please select metrics in Metrics page.")
        return

    # Dataset filter - use session state value directly
    all_datasets = st.session_state["selected_datasets"]
    saved_filter = st.session_state.get("comparison_dataset_filter")
    if saved_filter is None or not saved_filter:
        default_datasets = all_datasets
    else:
        default_datasets = [d for d in saved_filter if d in all_datasets] or all_datasets

    datasets_to_show = st.pills(
        ":blue[Filter Datasets]",
        options=all_datasets,
        default=default_datasets,
        selection_mode="multi",
        key="comparison_dataset_filter",
    )

    # On first load, pills returns the default, but session_state may not be set yet
    # So we check the actual widget value, not session_state
    if datasets_to_show is None or len(datasets_to_show) == 0:
        st.info("Select at least one dataset.")
        return

    st.divider()

    # Single pills: selected = Ours, not selected = Theirs
    saved_ours = st.session_state.get("comparison_our_groups")
    if saved_ours is None or not saved_ours:
        default_ours = selected_groups
    else:
        default_ours = [g for g in saved_ours if g in selected_groups] or selected_groups

    st.markdown("##### Our Methods :red[(selected)] vs Baseline (not selected)")
    our_groups = st.pills(
        "Select your methods",
        options=selected_groups,
        default=default_ours,
        selection_mode="multi",
        key="comparison_our_groups",
        label_visibility="collapsed",
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
    cols = st.columns([5, 1, 1, 1])
    with cols[0]:
        comparison_metric = st.pills(
            ":orange[Comparison Metric]",
            options=selected_metrics,
            default=selected_metrics[0] if selected_metrics else None,
            selection_mode="single",
            key="comparison_metric",
        )

    with cols[1]:
        agg_type = st.pills(
            ":violet[Aggregation]",
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

    with cols[3]:
        pivot_table = st.toggle("Pivot", value=False, key="comparison_pivot")

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
    agg_df, stats_results = build_comparison_table(
        metric_df, param_df, comparison_metric, agg_type, our_groups, their_groups
    )

    if agg_df.is_empty():
        st.warning("No results to display.")
        return

    # Filter to datasets to show
    agg_df = agg_df.filter(pl.col("dataset_name").is_in(datasets_to_show))

    # Display legend
    st.caption("● = Significantly better than ALL baselines (Wilcoxon rank-sum, Holm-Bonferroni, α=0.05)")

    # Build display table
    dataset_renames = st.session_state.get("dataset_renames", {})
    group_renames = st.session_state.get("group_renames", {})

    # Use user's dataset order from Get Started (datasets_to_show preserves order)
    datasets = [d for d in datasets_to_show if d in agg_df.get_column("dataset_name").to_list()]
    all_methods = our_groups + their_groups

    # Create display data with significance tracking
    display_data = []
    sig_cells = []  # Track (row_idx, col_name) for significant cells

    for row_idx, dataset in enumerate(datasets):
        row_data = {"Dataset": get_display_name(dataset_renames.get(dataset), dataset)}

        for method in all_methods:
            is_ours = method in our_groups
            display_name = get_display_name(group_renames.get(method), method)

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

    display_df = pl.DataFrame(display_data)

    # Pivot if requested (transpose: methods as rows, datasets as columns)
    if pivot_table:
        # Melt and pivot
        method_cols = [get_display_name(group_renames.get(m), m) for m in all_methods]
        melted = display_df.unpivot(
            index="Dataset",
            on=method_cols,
            variable_name="Method",
            value_name="Value",
        )
        display_df = melted.pivot(
            on="Dataset",
            index="Method",
            values="Value",
        )

    # Display with styling
    st.dataframe(display_df, width='content', hide_index=True)


if __name__ == "__main__":
    main()
