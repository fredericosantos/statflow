"""
Plots page — aggregated trend lines over a numeric parameter.

Providers fetch *summary* metrics only (one scalar per run); this page shows
aggregated trends over a numeric parameter (e.g. pop_size), not training curves.

plots.py
├── main()                      # Main page entry point
├── _is_numeric_param()         # Check if ≥ 90% of non-null values cast to float
├── _group_label_expr()         # `param=value, ...` concat expr (Parameters-page format)
├── _build_plot_group_label()   # full label (run selection) + reduced label (per line)
└── _render_line_plot()         # Plotly go.Scatter lines with optional IQR band
"""

from __future__ import annotations

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from statflow.config import SessionState
from statflow.functional.dataframes.data_processing import (
    apply_metric_filters,
    fetch_experiment_data,
    grouping_params,
)
from statflow.functional.statistics import AGGREGATIONS, aggregate_for_plot
from statflow.managers.naming import NamingManager
from statflow.shared.server_status import ServerStatusManager

st.set_page_config(
    page_title="Statflow — Plots",
    page_icon=":material/show_chart:",
    layout="wide",
)

_AGG_OPTIONS = list(AGGREGATIONS.keys())


def _is_numeric_param(series: pl.Series, threshold: float = 0.9) -> bool:
    """Return True if ≥ threshold of non-null values in `series` cast cleanly to float."""
    non_null = series.drop_nulls()
    if len(non_null) == 0:
        return False
    try:
        casted = non_null.cast(pl.Float64, strict=False)
    except Exception:
        return False
    clean = casted.drop_nulls()
    return len(clean) / len(non_null) >= threshold


def _group_label_expr(params: list[str]) -> pl.Expr:
    """`param=value, param2=value2` over `params` (matches the Parameters page)."""
    exprs: list[pl.Expr] = []
    for i, p in enumerate(params):
        if i > 0:
            exprs.append(pl.lit(", "))
        exprs.append(pl.lit(f"{p}="))
        exprs.append(pl.col(p).cast(pl.Utf8))
    return pl.concat_str(exprs) if exprs else pl.lit("Default")


def _build_plot_group_label(
    param_df: pl.DataFrame,
    selected_params: list[str],
    x_param: str,
) -> pl.DataFrame:
    """Add two label columns.

    - ``_full_group_label``: every selected param, in selection order — identical to
      the Parameters page's ``group_label``, so it can be matched against
      ``selected_groups`` to keep only the runs the user picked.
    - ``group_label``: the same minus ``x_param`` — the per-line grouping key, since
      ``x_param`` varies *along* each line rather than between lines.
    """
    present = [p for p in selected_params if p in param_df.columns]
    reduced = [p for p in present if p != x_param]
    return param_df.with_columns(
        _group_label_expr(present).alias("_full_group_label"),
        _group_label_expr(reduced).alias("group_label"),
    )


def _render_line_plot(
    plot_data: pl.DataFrame,
    *,
    x_col: str,
    y_metric: str,
    group_col: str,
    show_band: bool,
    log_x: bool,
    log_y: bool,
    axis_limits: dict,
) -> None:
    """Render aggregated line plot using Plotly go.Scatter."""
    if plot_data.is_empty():
        st.info("No data to plot.")
        return

    groups = plot_data[group_col].unique().sort().to_list()
    colors = plotly.colors.qualitative.Plotly

    fig = go.Figure()

    for i, group in enumerate(groups):
        gdf = plot_data.filter(pl.col(group_col) == group).sort(x_col)
        if gdf.is_empty():
            continue

        x_vals = gdf[x_col].to_numpy()
        y_vals = gdf["y"].to_numpy()
        display_name = NamingManager.get_group_name(group)
        color = colors[i % len(colors)]

        # Band (IQR) trace — drawn first so it sits behind the line
        if show_band and "y_q1" in gdf.columns and "y_q3" in gdf.columns:
            y_q1 = gdf["y_q1"].to_numpy()
            y_q3 = gdf["y_q3"].to_numpy()
            # Upper bound then reversed lower bound (fills the band)
            x_fill = np.concatenate([x_vals, x_vals[::-1]])
            y_fill = np.concatenate([y_q3, y_q1[::-1]])
            fig.add_trace(
                go.Scatter(
                    x=x_fill,
                    y=y_fill,
                    fill="toself",
                    fillcolor=color,
                    line=dict(width=0),
                    opacity=0.15,
                    hoverinfo="skip",
                    showlegend=False,
                    name=display_name,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=display_name,
                line=dict(color=color),
                marker=dict(color=color),
            )
        )

    # Axis labels
    x_label = (
        NamingManager.get_metric_name(x_col) if hasattr(NamingManager, "get_metric_name") else x_col
    )
    y_label = NamingManager.get_metric_name(y_metric)

    layout_kwargs: dict = dict(
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title="Group",
        margin=dict(t=60, b=50, l=60, r=30),
        hovermode="x unified",
    )

    # Log scale
    if log_x:
        layout_kwargs["xaxis_type"] = "log"
    if log_y:
        layout_kwargs["yaxis_type"] = "log"

    # Axis limits
    x_min = axis_limits.get("x_min")
    x_max = axis_limits.get("x_max")
    y_min = axis_limits.get("y_min")
    y_max = axis_limits.get("y_max")
    if x_min is not None or x_max is not None:
        layout_kwargs["xaxis_range"] = [x_min, x_max]
    if y_min is not None or y_max is not None:
        layout_kwargs["yaxis_range"] = [y_min, y_max]

    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Plots page entry point."""
    SessionState.initialize()

    status_manager = ServerStatusManager()
    status_manager.display_sidebar()

    st.title(":material/show_chart: Plots")
    st.markdown(
        "Aggregated trend lines over a numeric parameter. "
        "One line per group; X axis = numeric parameter; Y axis = selected metric."
    )

    # --- Guards ---
    if not st.session_state["selected_experiments"]:
        st.warning("Please select experiments first in Get Started.")
        return

    if not st.session_state["selected_datasets"]:
        st.warning("Please select datasets first in Get Started.")
        return

    selected_groups: list[str] = st.session_state.get("selected_groups", [])
    if not selected_groups:
        st.warning("Please select and configure groups in Parameters page.")
        return

    selected_metrics: list[str] = st.session_state.get("selected_metrics", [])
    available_metrics: list[str] = st.session_state.get("available_metrics", [])
    metric_options = selected_metrics or available_metrics
    if not metric_options:
        st.warning("Please select metrics in Metrics page.")
        return

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

    # --- Identify numeric params ---
    # Group dimensions = selected params + selected tags. Tags are true/false, so
    # they never qualify as the numeric X axis but do split the lines.
    group_params = grouping_params()
    numeric_params = [
        p for p in group_params if p in param_df.columns and _is_numeric_param(param_df[p])
    ]

    if not numeric_params:
        st.info(
            "No numeric parameters detected (need ≥ 90% of non-null values to cast to float). "
            "Adjust parameter selection in the Parameters page."
        )
        return

    st.divider()

    # --- Controls ---
    ctrl_cols = st.columns([3, 3, 2])
    with ctrl_cols[0]:
        x_param = st.selectbox(
            "X axis (numeric parameter)",
            options=numeric_params,
            key="plot_x_param",
        )

    with ctrl_cols[1]:
        y_metric = st.selectbox(
            "Y axis (metric)",
            options=metric_options,
            format_func=NamingManager.get_metric_name,
            key="plot_y_metric",
        )

    with ctrl_cols[2]:
        saved_agg = st.session_state.get("plot_agg", "median")
        if saved_agg not in _AGG_OPTIONS:
            saved_agg = "median"
        agg_choice = st.pills(
            "Aggregation",
            options=_AGG_OPTIONS,
            default=saved_agg,
            selection_mode="single",
            key="plot_agg_widget",
        )
        agg = agg_choice or saved_agg
        if st.session_state.get("plot_agg") != agg:
            st.session_state["plot_agg"] = agg
            SessionState.save_key_to_config("plot_agg")

    # Spread toggle
    show_band = st.toggle("Show spread (IQR band)", value=False, key="plot_show_band")

    # Dataset scope
    all_datasets: list[str] = st.session_state["selected_datasets"]
    scope_options = ["Aggregate across datasets"] + [
        NamingManager.get_dataset_name(d) for d in all_datasets
    ]
    saved_scope = st.session_state.get("plot_dataset_scope", "Aggregate across datasets")
    if saved_scope not in scope_options:
        saved_scope = "Aggregate across datasets"

    scope = st.pills(
        "Dataset scope",
        options=scope_options,
        default=saved_scope,
        selection_mode="single",
        key="plot_scope_widget",
    )
    scope = scope or saved_scope
    if st.session_state.get("plot_dataset_scope") != scope:
        st.session_state["plot_dataset_scope"] = scope
        SessionState.save_key_to_config("plot_dataset_scope")

    # Axis settings expander
    saved_limits: dict = st.session_state.get("plot_axis_limits", {})
    with st.expander("Axis settings", icon=":material/settings:"):
        ax_cols = st.columns(4)
        with ax_cols[0]:
            x_min_v = ax_cols[0].number_input(
                "X min",
                value=saved_limits.get("x_min"),
                key="plot_x_min",
                placeholder="auto",
            )
        with ax_cols[1]:
            x_max_v = ax_cols[1].number_input(
                "X max",
                value=saved_limits.get("x_max"),
                key="plot_x_max",
                placeholder="auto",
            )
        with ax_cols[2]:
            y_min_v = ax_cols[2].number_input(
                "Y min",
                value=saved_limits.get("y_min"),
                key="plot_y_min",
                placeholder="auto",
            )
        with ax_cols[3]:
            y_max_v = ax_cols[3].number_input(
                "Y max",
                value=saved_limits.get("y_max"),
                key="plot_y_max",
                placeholder="auto",
            )

        new_limits = {
            "x_min": x_min_v,
            "x_max": x_max_v,
            "y_min": y_min_v,
            "y_max": y_max_v,
        }
        if new_limits != saved_limits:
            st.session_state["plot_axis_limits"] = new_limits
            SessionState.save_key_to_config("plot_axis_limits")
        axis_limits = new_limits

        log_col1, log_col2 = st.columns(2)
        log_x = log_col1.toggle(
            "Log scale X", value=st.session_state.get("plot_log_x", False), key="plot_log_x_toggle"
        )
        log_y = log_col2.toggle(
            "Log scale Y", value=st.session_state.get("plot_log_y", False), key="plot_log_y_toggle"
        )
        if st.session_state.get("plot_log_x") != log_x:
            st.session_state["plot_log_x"] = log_x
            SessionState.save_key_to_config("plot_log_x")
        if st.session_state.get("plot_log_y") != log_y:
            st.session_state["plot_log_y"] = log_y
            SessionState.save_key_to_config("plot_log_y")

    if not x_param or not y_metric:
        st.info("Select an X and Y parameter above.")
        return

    # --- Build group labels (excluding x param) ---
    param_df_labeled = _build_plot_group_label(
        param_df,
        selected_params=group_params,
        x_param=x_param,
    )

    # Join metric + params
    if "run_id" in metric_df.columns and "run_id" in param_df_labeled.columns:
        combined = metric_df.join(
            param_df_labeled.select(
                [
                    c
                    for c in ["run_id", "_full_group_label", "group_label", "dataset_name", x_param]
                    if c in param_df_labeled.columns
                ]
            ),
            on="run_id",
            how="left",
        )
    else:
        combined = metric_df.join(
            param_df_labeled.select(
                [
                    c
                    for c in ["dataset_name", "_full_group_label", "group_label", x_param]
                    if c in param_df_labeled.columns
                ]
            ).unique(),
            on="dataset_name",
            how="left",
        )

    if y_metric not in combined.columns:
        st.error(f"Metric '{y_metric}' not found in data.")
        return
    if x_param not in combined.columns:
        st.error(f"Parameter '{x_param}' not found in joined data.")
        return

    # Keep only the runs the user picked on the Parameters page. Match on the FULL
    # label (all params); the reduced `group_label` is only the per-line key.
    combined = combined.filter(pl.col("_full_group_label").is_in(selected_groups))

    # Dataset scope filter
    if scope != "Aggregate across datasets":
        # Reverse-lookup dataset id from display name
        dataset_id = next(
            (d for d in all_datasets if NamingManager.get_dataset_name(d) == scope),
            None,
        )
        if dataset_id and "dataset_name" in combined.columns:
            combined = combined.filter(pl.col("dataset_name") == dataset_id)

    # Cast x param to float
    combined = combined.with_columns(pl.col(x_param).cast(pl.Float64, strict=False))
    combined = combined.drop_nulls(subset=[x_param, y_metric, "group_label"])

    if combined.is_empty():
        st.warning("No data available for the selected combination.")
        return

    # Aggregate for plot
    plot_data = aggregate_for_plot(
        combined,
        x_col=x_param,
        y_col=y_metric,
        group_col="group_label",
        agg=agg,
        band=show_band,
    )

    if plot_data.is_empty():
        st.warning("Aggregation produced no results.")
        return

    # Render chart
    _render_line_plot(
        plot_data,
        x_col=x_param,
        y_metric=y_metric,
        group_col="group_label",
        show_band=show_band,
        log_x=log_x,
        log_y=log_y,
        axis_limits=axis_limits,
    )


if __name__ == "__main__":
    main()
