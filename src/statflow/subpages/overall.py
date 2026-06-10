"""
Overall page — cross-dataset statistical significance, effect sizes & LaTeX export.

Shows whether "our" method is better *overall* (across all datasets) using
Wilcoxon signed-rank (2 groups) or Friedman + Holm post-hoc (≥ 3 groups).

overall.py
├── main()                    # Main page entry point
├── _render_block_matrix()    # Aggregated dataset × group table with winner highlighting
├── _render_test_verdict()    # Test outcome: method, statistic, p, n, warnings
└── _render_a12_summary()     # Median A12 across datasets for ours vs each group
"""

import polars as pl
import streamlit as st

from statflow.config import SessionState
from statflow.functional.dataframes.data_processing import (
    apply_metric_filters,
    fetch_experiment_data,
)
from statflow.functional.statistics import (
    AGGREGATIONS,
    CrossDatasetResult,
    a12,
    a12_magnitude,
    aggregate_per_dataset,
    cross_dataset_test,
    cross_dataset_to_latex,
)
from statflow.managers.naming import NamingManager
from statflow.shared.server_status import ServerStatusManager

st.set_page_config(
    page_title="Statflow — Overall",
    page_icon=":material/leaderboard:",
    layout="wide",
)

_AGG_OPTIONS = list(AGGREGATIONS.keys())  # ["median", "mean", "min", "max", "iqm"]


def _render_block_matrix(
    block: pl.DataFrame,
    *,
    dataset_col: str,
    group_cols: list[str],
    maximize: bool,
) -> None:
    """Display the aggregated block matrix with winner highlighted per row."""
    st.markdown("**Aggregated block matrix** (rows = datasets, cols = groups)")

    # Build a display DataFrame with winner bolded via markdown
    display_rows = []
    for row in block.iter_rows(named=True):
        vals = {g: row[g] for g in group_cols if row[g] is not None}
        winner = None
        if vals:
            winner = (
                max(vals, key=lambda g: vals[g]) if maximize else min(vals, key=lambda g: vals[g])
            )

        display_row: dict[str, str] = {dataset_col: str(row[dataset_col])}
        for g in group_cols:
            v = row[g]
            if v is None:
                display_row[NamingManager.get_group_name(g)] = "—"
            else:
                cell = f"{float(v):.4g}"
                if g == winner:
                    cell = f"**{cell}**"
                display_row[NamingManager.get_group_name(g)] = cell
        display_rows.append(display_row)

    st.dataframe(pl.DataFrame(display_rows), use_container_width=True, hide_index=True)


def _render_test_verdict(result: CrossDatasetResult, *, alpha: float = 0.05) -> None:
    """Render the cross-dataset test outcome with context and warnings."""
    if result.method == "wilcoxon_signed_rank":
        method_label = "Wilcoxon signed-rank (2 groups)"
    else:
        method_label = f"Friedman ({result.n_datasets} groups)"

    sig_icon = ":material/check_circle:" if result.p_value <= alpha else ":material/cancel:"
    sig_text = "significant" if result.p_value <= alpha else "not significant"

    st.markdown(f"**Test:** {method_label}")

    cols = st.columns(4)
    cols[0].metric("Statistic", f"{result.statistic:.3f}")
    cols[1].metric("p-value", f"{result.p_value:.4f}" if result.p_value >= 0.001 else "<0.001")
    cols[2].metric("n datasets", result.n_datasets)
    cols[3].metric("Result", f"{sig_icon} {sig_text}")

    if result.note:
        st.info(f":material/info: {result.note}")

    if result.low_power:
        st.warning(
            ":material/warning: Low power: fewer than 5 complete datasets. "
            "Interpret results with caution."
        )

    if result.dropped_datasets:
        st.warning(
            f":material/warning: {len(result.dropped_datasets)} dataset(s) dropped due to "
            f"missing group values: {', '.join(result.dropped_datasets)}"
        )

    # Friedman extras
    if result.method == "friedman" and result.mean_ranks:
        st.markdown("**Mean ranks** (rank 1 = best):")
        ranks_df = pl.DataFrame(
            {
                "Group": [NamingManager.get_group_name(g) for g in result.mean_ranks],
                "Mean rank": list(result.mean_ranks.values()),
            }
        ).sort("Mean rank")
        st.dataframe(ranks_df, use_container_width=True, hide_index=True)

        if result.posthoc is not None and not result.posthoc.is_empty():
            st.markdown("**Post-hoc pairwise comparisons** (Holm-corrected one-sided Wilcoxon):")
            ph_display = result.posthoc.with_columns(
                pl.col("group").map_elements(NamingManager.get_group_name, return_dtype=pl.Utf8)
            )
            st.dataframe(ph_display, use_container_width=True, hide_index=True)
        elif result.p_value <= alpha:
            st.info("Post-hoc: Friedman significant but no post-hoc results available.")
        else:
            st.info("Friedman omnibus not significant (p > 0.05) — no post-hoc comparisons run.")


def _render_a12_summary(
    combined_data: pl.DataFrame,
    *,
    metric: str,
    our_groups: list[str],
    their_groups: list[str],
    datasets: list[str],
    maximize: bool,
) -> None:
    """Show median A12 across datasets for ours vs each competitor."""
    if not our_groups or not their_groups:
        return

    st.markdown("**Median A12 (effect size) across datasets** — ours vs each competitor")

    rows = []
    for our in our_groups:
        for comp in their_groups:
            a12_vals = []
            for ds in datasets:
                ds_data = combined_data.filter(pl.col("dataset_name") == ds)
                ours_vals = (
                    ds_data.filter(pl.col("group_label") == our)
                    .get_column(metric)
                    .drop_nulls()
                    .to_numpy()
                )
                comp_vals = (
                    ds_data.filter(pl.col("group_label") == comp)
                    .get_column(metric)
                    .drop_nulls()
                    .to_numpy()
                )
                if len(ours_vals) > 0 and len(comp_vals) > 0:
                    a12_vals.append(a12(ours_vals, comp_vals, maximize=maximize))

            if a12_vals:
                import numpy as np

                median_a12 = float(np.median(a12_vals))
                mag = a12_magnitude(median_a12)
            else:
                median_a12 = float("nan")
                mag = "—"

            rows.append(
                {
                    "Our group": NamingManager.get_group_name(our),
                    "vs": NamingManager.get_group_name(comp),
                    "Median A12": f"{median_a12:.3f}" if not (median_a12 != median_a12) else "—",
                    "Magnitude": mag,
                    "n datasets": len(a12_vals),
                }
            )

    if rows:
        st.dataframe(pl.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    """Overall page entry point."""
    SessionState.initialize()

    status_manager = ServerStatusManager()
    status_manager.display_sidebar()

    st.title(":material/leaderboard: Overall")
    st.markdown(
        "Cross-dataset significance test: is our method better *overall*? "
        "2 groups → Wilcoxon signed-rank; ≥ 3 groups → Friedman + Holm post-hoc."
    )

    # --- Guards (same as Comparison) ---
    if not st.session_state["selected_experiments"]:
        st.warning("Please select experiments first in Get Started.")
        return

    if not st.session_state["selected_datasets"]:
        st.warning("Please select datasets first in Get Started.")
        return

    selected_groups = st.session_state["selected_groups"]
    if not selected_groups:
        st.warning("Please select and configure groups in Parameters page.")
        return

    selected_metrics = st.session_state["selected_metrics"]
    if not selected_metrics:
        st.warning("Please select metrics in Metrics page.")
        return

    # Reuse the same our_groups selection from Comparison (comparison_our_groups)
    saved_ours = st.session_state.get("comparison_our_groups") or []
    our_groups: list[str] = [g for g in saved_ours if g in selected_groups] or selected_groups
    their_groups: list[str] = [g for g in selected_groups if g not in our_groups]

    if not our_groups:
        st.info("Go to Comparison and select your methods first.")
        return
    if not their_groups:
        st.info("Go to Comparison and deselect at least one method as a baseline.")
        return

    st.divider()

    # --- Aggregation picker ---
    saved_agg = st.session_state.get("cross_dataset_agg", "median")
    if saved_agg not in _AGG_OPTIONS:
        saved_agg = "median"

    agg_choice = st.pills(
        "Aggregation",
        options=_AGG_OPTIONS,
        default=saved_agg,
        selection_mode="single",
        key="cross_dataset_agg_widget",
    )
    agg = agg_choice or saved_agg

    if st.session_state.get("cross_dataset_agg") != agg:
        st.session_state["cross_dataset_agg"] = agg
        SessionState.save_key_to_config("cross_dataset_agg")

    st.divider()

    # --- Fetch data ---
    with st.spinner("Loading data..."):
        metric_df = fetch_experiment_data("metrics.")
        param_df = fetch_experiment_data("params.")

    if metric_df.is_empty() or param_df.is_empty():
        st.error("No data found.")
        return

    metric_df = apply_metric_filters(metric_df)
    if metric_df.is_empty():
        st.warning("No data remains after applying metric filters.")
        return

    # Build group labels (same approach as Comparison)
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

    # Join metric + params
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

    all_groups = our_groups + their_groups
    combined = combined.filter(pl.col("group_label").is_in(all_groups))
    datasets = st.session_state["selected_datasets"]

    if combined.is_empty():
        st.warning("No matching data after joining params and metrics.")
        return

    metric_directions: dict[str, str] = st.session_state.get("metric_directions", {})

    # --- Per-metric section ---
    for metric in selected_metrics:
        direction = metric_directions.get(metric, "Lower")
        maximize = direction == "Higher"

        with st.container():
            st.markdown(f"### {NamingManager.get_metric_name(metric)}")
            better_str = "higher" if maximize else "lower"
            st.caption(f"Direction: {better_str} is better")

            # Filter to this metric
            if metric not in combined.columns:
                st.info(f"Metric column '{metric}' not found in data.")
                continue

            metric_data = combined.select(["dataset_name", "group_label", metric]).drop_nulls()

            if metric_data.is_empty():
                st.info("No data for this metric.")
                continue

            # Build block matrix
            block = aggregate_per_dataset(
                metric_data,
                metric_col=metric,
                group_col="group_label",
                dataset_col="dataset_name",
                agg=agg,
            )

            dataset_col = block.columns[0]
            group_cols = [c for c in block.columns if c != dataset_col]

            if not block.is_empty():
                _render_block_matrix(
                    block,
                    dataset_col=dataset_col,
                    group_cols=group_cols,
                    maximize=maximize,
                )

            # Run cross-dataset test
            try:
                result: CrossDatasetResult = cross_dataset_test(
                    block, ours=our_groups[0], maximize=maximize
                )
                _render_test_verdict(result)
            except ValueError as exc:
                st.error(f":material/error: {exc}")
                continue

            # A12 summary
            _render_a12_summary(
                combined,
                metric=metric,
                our_groups=our_groups,
                their_groups=their_groups,
                datasets=datasets,
                maximize=maximize,
            )

            # LaTeX export expander
            with st.expander("LaTeX export", icon=":material/code:"):
                caption = (
                    f"Cross-dataset results — {NamingManager.get_metric_name(metric)} "
                    f"({agg} aggregation)"
                )
                label = f"tab:overall-{metric}"
                latex = cross_dataset_to_latex(result, block, caption=caption, label=label)
                st.code(latex, language="latex")

            st.divider()


if __name__ == "__main__":
    main()
