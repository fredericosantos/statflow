"""
Reusable graph components for visualization.

This module provides Streamlit components for rendering various plots
including boxplots, scatterplot and statistical visualizations.

graphs.py
├── Boxplot rendering component
├── Scatter plot component
├── Statistics table component
├── Raw data table component
└── Graph layout and configuration

Usage:
    from statflow.components.graphs import (
        render_boxplot, render_pareto_front
    )
"""

import pandas as pd
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Optional, Dict, Any, List

from statflow.utils.data_processing import (
    create_group_label, get_sort_key, calculate_pareto_front, get_pareto_group_label
)
from statflow.utils.visualization import (
    get_color_for_config, get_pareto_color, get_marker_symbol, get_default_color_mapping
)
from statflow.utils.styling import handle_empty_data


def _create_distribution_plot(
    df: pl.DataFrame,
    column: str,
    plot_type: str,
    color_column: Optional[str] = None,
    title_prefix: str = "",
    interactive: bool = True
) -> go.Figure:
    """Create histogram or box plot for a column with interactive controls.

    Args:
        df: DataFrame with data.
        column: Column to plot.
        plot_type: "histogram" or "box".
        color_column: Optional column to color by.
        title_prefix: Prefix for the plot title.
        interactive: Whether to add interactive controls (zoom, pan, etc.).

    Returns:
        Plotly figure.
    """
    if plot_type == "histogram":
        fig = px.histogram(
            df,
            x=column,
            title=f"{title_prefix}Distribution of {column}",
            color=color_column
        )
    elif plot_type == "box":
        fig = px.box(
            df,
            x=color_column,
            y=column,
            title=f"{title_prefix}Box Plot of {column}"
        )
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    # Add interactive controls if requested
    if interactive:
        fig.update_layout(
            hovermode='closest',
            # Add mode bar buttons for additional interactivity
            modebar_add=['zoom', 'pan', 'select', 'lasso', 'zoomIn', 'zoomOut', 'autoScale', 'resetScale'],
            # Enable dragmode for selection
            dragmode='zoom'
        )
        # Add range slider for histograms
        if plot_type == "histogram":
            fig.update_xaxes(rangeslider_visible=True)
    else:
        fig.update_layout(width=800, height=400)

    return fig


def render_boxplot(
    runs_df: pd.DataFrame,
    show_mean: bool = False,
    points_display: Any = "outliers",
    use_custom_colors: bool = True,
    custom_colors: Optional[Dict] = None,
    width: int = 800,
    height: int = 600
) -> None:
    """Render boxplot of fitness distributions.

    Args:
        runs_df: DataFrame with experiment runs.
        show_mean: Whether to show mean instead of median.
        points_display: How to display points ("outliers", "all", or False).
        use_custom_colors: Whether to use custom colors.
        custom_colors: Custom color mappings.
        width: Graph width in pixels.
        height: Graph height in pixels.
    """
    if runs_df.empty:
        st.warning("No data available for boxplot.")
        return

    # Add grouping labels and sort keys
    plot_df = runs_df.copy()
    plot_df["config_group"] = plot_df.apply(create_group_label, axis=1)
    plot_df["sort_key"] = plot_df.apply(get_sort_key, axis=1)

    # Sort configurations
    config_order = (
        plot_df.groupby("config_group")["sort_key"]
        .first()
        .sort_values()
        .index.tolist()
    )

    # Create color mapping
    if use_custom_colors and custom_colors:
        color_map = {}
        for config in config_order:
            # Get a representative row for this config
            config_row = plot_df[plot_df["config_group"] == config].iloc[0]
            color_map[config] = get_color_for_config(config_row, custom_colors)
    else:
        color_map = get_default_color_mapping(config_order)

    # Create boxplot
    fig = go.Figure()

    for config in config_order:
        config_data = plot_df[plot_df["config_group"] == config]

        if not config_data.empty:
            fig.add_trace(go.Box(
                y=config_data["metrics.best_test_fitness"],
                name=config,
                marker_color=color_map.get(config, "#808080"),
                boxpoints=points_display,
                jitter=0.3,
                pointpos=0,
                showlegend=True,
            ))

    # Update layout
    fig.update_layout(
        title="RMSE Distribution by Configuration",
        xaxis_title="Configuration",
        yaxis_title="RMSE (Test Fitness)",
        width=width,
        height=height,
        showlegend=False,  # Box plots have names on x-axis
    )

    # Update y-axis to log scale for better visualization
    fig.update_yaxes(type="log")

    st.plotly_chart(fig, width="content")


def render_pareto_front(
    runs_df: pd.DataFrame,
    show_error_bars: bool = True,
    use_custom_colors: bool = True,
    custom_colors: Optional[Dict] = None,
    custom_symbols: Optional[Dict] = None,
    width: int = 800,
    height: int = 600
) -> None:
    """Render scatter plot.

    Args:
        runs_df: DataFrame with experiment runs.
        show_error_bars: Whether to show error bars.
        use_custom_colors: Whether to use custom colors.
        custom_colors: Custom color mappings.
        custom_symbols: Custom symbol mappings.
        width: Graph width in pixels.
        height: Graph height in pixels.
    """
    if runs_df.empty:
        st.warning("No data available.")
        return

    pareto_data = []

    # Group by configuration
    plot_df = runs_df.copy()
    plot_df["config_group"] = plot_df.apply(create_group_label, axis=1)

    for config in plot_df["config_group"].unique():
        config_data = plot_df[plot_df["config_group"] == config]

        if len(config_data) > 1:  # Need at least 2 points for Pareto
            pareto_points = calculate_pareto_front(
                config_data,
                x_col="metrics.best_n_nodes",
                y_col="metrics.best_test_fitness"
            )

            if not pareto_points.empty:
                # Calculate averages
                avg_nodes = pareto_points["metrics.best_n_nodes"].mean()
                avg_fitness = pareto_points["metrics.best_test_fitness"].mean()
                std_nodes = pareto_points["metrics.best_n_nodes"].std()
                std_fitness = pareto_points["metrics.best_test_fitness"].std()

                pareto_data.append({
                    "config": config,
                    "avg_nodes": avg_nodes,
                    "avg_fitness": avg_fitness,
                    "std_nodes": std_nodes,
                    "std_fitness": std_fitness,
                    "pareto_group": get_pareto_group_label(pareto_points.iloc[0]),
                })

    if not pareto_data:
        st.warning("No data available.")
        return

    pareto_df = pd.DataFrame(pareto_data)

    # Create scatter plot
    fig = go.Figure()

    for _, row in pareto_df.iterrows():
        color = get_pareto_color(row["pareto_group"], custom_colors)
        symbol = get_marker_symbol(row["pareto_group"], custom_symbols)

        error_x = dict(type="data", array=[row["std_nodes"]]) if show_error_bars else None
        error_y = dict(type="data", array=[row["std_fitness"]]) if show_error_bars else None

        fig.add_trace(go.Scatter(
            x=[row["avg_nodes"]],
            y=[row["avg_fitness"]],
            mode="markers",
            name=row["config"],
            marker=dict(
                color=color,
                symbol=symbol,
                size=10,
            ),
            error_x=error_x,
            error_y=error_y,
            showlegend=True,
        ))

    # Update layout
    fig.update_layout(
        title="Tree Size vs RMSE",
        xaxis_title="Average Tree Size (nodes)",
        yaxis_title="Average RMSE",
        width=width,
        height=height,
    )

    # Update axes to log scale
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    st.plotly_chart(fig, width="content")


def render_3d_scatter_plot(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: Optional[str] = None,
    title: str = "3D Scatter Plot",
    width: int = 800,
    height: int = 600
) -> None:
    """Render interactive 3D scatter plot for three-parameter correlation analysis.

    Args:
        df: DataFrame with data.
        x_col: Column for x-axis.
        y_col: Column for y-axis.
        z_col: Column for z-axis.
        color_col: Optional column to color points by.
        title: Plot title.
        width: Graph width in pixels.
        height: Graph height in pixels.
    """
    if handle_empty_data(df, "No data available for 3D visualization."):
        return

    # Check if required columns exist
    required_cols = [x_col, y_col, z_col]
    if color_col:
        required_cols.append(color_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing required columns: {missing_cols}")
        return

    # Create 3D scatter plot
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        title=title,
        labels={
            x_col: x_col.replace('_', ' ').title(),
            y_col: y_col.replace('_', ' ').title(),
            z_col: z_col.replace('_', ' ').title()
        }
    )

    # Update layout for better interactivity
    fig.update_layout(
        width=width,
        height=height,
        scene=dict(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            zaxis_title=z_col.replace('_', ' ').title(),
        ),
        hovermode='closest'
    )

    # Add interactive controls
    fig.update_layout(
        modebar_add=['zoom', 'pan', 'orbit', 'turntable', 'resetCameraDefault3d', 'resetCameraLastSave3d']
    )

    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(
    df: pl.DataFrame,
    value_cols: List[str],
    category_col: Optional[str] = None,
    title: str = "Radar Chart",
    width: int = 800,
    height: int = 600
) -> None:
    """Render radar chart for multi-metric comparisons.

    Args:
        df: DataFrame with data.
        value_cols: List of columns to plot on radar.
        category_col: Optional column to group by (creates multiple radar lines).
        title: Plot title.
        width: Graph width in pixels.
        height: Graph height in pixels.
    """
    if handle_empty_data(df, "No data available for radar chart."):
        return

    # Check if required columns exist
    missing_cols = [col for col in value_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing required columns: {missing_cols}")
        return

    if category_col and category_col not in df.columns:
        st.warning(f"Category column '{category_col}' not found.")
        return

    # Prepare data for radar chart
    if category_col:
        # Multiple categories - create separate traces
        fig = go.Figure()
        categories = df[category_col].unique().to_list()

        for cat in categories:
            cat_data = df.filter(pl.col(category_col) == cat)
            if not cat_data.is_empty():
                # Calculate means for each value column
                values = []
                for col in value_cols:
                    mean_val = cat_data[col].mean()
                    values.append(float(mean_val))

                # Close the radar by repeating first value
                values.append(values[0])
                categories_closed = value_cols + [value_cols[0]]

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories_closed,
                    fill='toself',
                    name=str(cat)
                ))
    else:
        # Single radar chart - use means of all data
        values = []
        for col in value_cols:
            mean_val = df[col].mean()
            values.append(float(mean_val))

        # Close the radar
        values.append(values[0])
        categories_closed = value_cols + [value_cols[0]]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill='toself'
        ))

    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, None]  # Auto-scale
            )
        ),
        showlegend=category_col is not None
    )

    st.plotly_chart(fig, use_container_width=True)


def render_statistics_table(runs_df: pd.DataFrame, show_mean: bool = False, return_df: bool = False) -> Optional[pd.DataFrame]:
    """Render detailed statistics table.

    Args:
        runs_df: DataFrame with experiment runs.
        show_mean: Whether to show mean instead of median.
        return_df: Whether to return the dataframe instead of displaying.

    Returns:
        DataFrame if return_df is True, None otherwise.
    """
    if runs_df.empty:
        if return_df:
            return None
        st.warning("No data available for statistics table.")
        return None

    # Add grouping labels
    stats_df = runs_df.copy()
    stats_df["config_group"] = stats_df.apply(create_group_label, axis=1)
    stats_df["sort_key"] = stats_df.apply(get_sort_key, axis=1)

    # Group by configuration and calculate statistics
    grouped = stats_df.groupby("config_group")

    stats_data = []
    for config in sorted(stats_df["config_group"].unique(),
                        key=lambda x: stats_df[stats_df["config_group"] == x]["sort_key"].iloc[0]):
        config_data = grouped.get_group(config)

        rmse_values = config_data["metrics.best_test_fitness"]
        nodes_values = config_data["metrics.best_n_nodes"]

        if show_mean:
            rmse_center = rmse_values.mean()
            nodes_center = nodes_values.mean()
            rmse_spread = rmse_values.std()
            nodes_spread = nodes_values.std()
            center_label = "Mean"
        else:
            rmse_center = rmse_values.median()
            nodes_center = nodes_values.median()
            rmse_spread = rmse_values.std()
            nodes_spread = nodes_values.std()
            center_label = "Median"

        stats_data.append({
            "Configuration": config,
            f"RMSE {center_label}": f"{rmse_center:.6f}",
            "RMSE Std": f"{rmse_spread:.6f}",
            f"Tree Size {center_label}": f"{nodes_center:.1f}",
            "Tree Size Std": f"{nodes_spread:.1f}",
            "Sample Size": len(config_data),
        })

    stats_table = pd.DataFrame(stats_data)
    
    if return_df:
        return stats_table
    
    st.dataframe(stats_table, width="content")
    return None


def render_raw_data_table(runs_df: pd.DataFrame, selected_params: Optional[List[str]] = None) -> None:
    """Render raw fitness values table.

    Args:
        runs_df: DataFrame with experiment runs.
        selected_params: List of selected parameter names (without 'params.' prefix) to include.
    """
    if runs_df.empty:
        st.warning("No raw data available.")
        return

    # Select relevant columns for display
    base_cols = [
        "run_id",
        "metrics.best_test_fitness",
        "metrics.best_n_nodes",
    ]
    
    # Add parameter columns
    param_cols = []
    if selected_params:
        param_cols = [f"params.{p}" for p in selected_params if f"params.{p}" in runs_df.columns]
    else:
        # Default parameter columns
        param_cols = [
            "params.variant",
            "params.arc_beta",
            "params.mutation_pool_factor",
            "params.use_oms",
        ]
    
    display_cols = base_cols + param_cols

    # Filter to only existing columns
    available_cols = [col for col in display_cols if col in runs_df.columns]

    if available_cols:
        display_df = runs_df[available_cols].copy()

        # Rename columns for better display
        column_names = {
            "run_id": "Run ID",
            "metrics.best_test_fitness": "RMSE",
            "metrics.best_n_nodes": "Tree Size",
            "params.variant": "Variant",
            "params.arc_beta": "Beta",
            "params.mutation_pool_factor": "MPF",
            "params.use_oms": "Use OMS",
        }

        display_df = display_df.rename(columns=column_names)

        # Add configuration group
        display_df["Configuration"] = runs_df.apply(create_group_label, axis=1)

        # Reorder columns
        col_order = ["Configuration"] + [column_names.get(col, col) for col in available_cols]
        display_df = display_df[col_order]

        st.dataframe(display_df, width="content")
    else:
        st.warning("No relevant columns found in the data.")


def render_parameter_distributions(param_df: pl.DataFrame) -> None:
    """Render distributions for parameters.

    Args:
        param_df: DataFrame with parameter data.
    """
    if handle_empty_data(param_df, "No parameter data available for visualization."):
        return

    # Plotly supports Polars DataFrames natively

    # Exclude dataset_name
    param_cols = [col for col in param_df.columns if col != 'dataset_name']

    if not param_cols:
        st.info("No parameters found for visualization.")
        return

    # Create tabs for different parameter types
    tab_names = ["Histograms", "Box Plots", "Value Counts", "Correlation", "Advanced 3D"]
    tabs = st.tabs(tab_names)

    with tabs[0]:  # Histograms
        st.subheader("Parameter Histograms")
        for param in param_cols[:6]:  # Limit to first 6 to avoid overcrowding
            if param_df[param].dtype in [pl.Int64, pl.Float64]:
                color_col = 'dataset_name' if 'dataset_name' in param_df.columns else None
                fig = _create_distribution_plot(
                    param_df, param, "histogram", color_col, "Parameter "
                )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:  # Box Plots
        st.subheader("Parameter Box Plots")
        for param in param_cols[:6]:
            if param_df[param].dtype in [pl.Int64, pl.Float64]:
                color_col = 'dataset_name' if 'dataset_name' in param_df.columns else None
                fig = _create_distribution_plot(
                    param_df, param, "box", color_col, "Parameter "
                )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:  # Value Counts
        st.subheader("Parameter Value Counts")
        for param in param_cols[:6]:
            value_counts = param_df[param].value_counts().head(20)  # Top 20 values
            if not value_counts.empty:
                fig = px.bar(
                    x=value_counts.index.astype(str),
                    y=value_counts.values,
                    title=f"Top Values for {param}",
                    labels={'x': param, 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:  # Correlation
        st.subheader("Parameter Correlation")
        numeric_cols = [col for col in param_cols if param_df[col].dtype in ['int64', 'float64']]
        if len(numeric_cols) > 1:
            corr_matrix = param_df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title="Parameter Correlation Matrix",
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric parameters for correlation analysis.")

    with tabs[4]:  # Advanced 3D
        st.subheader("Advanced Visualizations")
        numeric_cols = [col for col in param_cols if param_df[col].dtype in [pl.Int64, pl.Float64]]

        if len(numeric_cols) >= 3:
            # 3D Scatter Plot
            st.markdown("### 3D Parameter Scatter Plot")
            col1, col2, col3 = st.columns(3)
            with col1:
                x_param = st.selectbox("X-axis", numeric_cols, key="x_param_3d")
            with col2:
                y_param = st.selectbox("Y-axis", numeric_cols, key="y_param_3d")
            with col3:
                z_param = st.selectbox("Z-axis", numeric_cols, key="z_param_3d")

            color_param = st.selectbox("Color by", [None] + param_cols, key="color_param_3d")

            if x_param and y_param and z_param:
                render_3d_scatter_plot(
                    param_df, x_param, y_param, z_param, color_param,
                    title=f"3D Scatter: {x_param} vs {y_param} vs {z_param}"
                )

            # Radar Chart
            st.markdown("### Parameter Radar Chart")
            selected_params = st.multiselect(
                "Select parameters for radar chart",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))],
                key="radar_params"
            )

            if len(selected_params) >= 3:
                category_col = st.selectbox(
                    "Group by (optional)",
                    [None] + [col for col in param_df.columns if col != 'dataset_name'],
                    key="radar_category"
                )
                render_radar_chart(
                    param_df, selected_params, category_col,
                    title="Parameter Comparison Radar Chart"
                )
            else:
                st.info("Select at least 3 parameters for radar chart.")
        else:
            st.info("Need at least 3 numeric parameters for advanced 3D visualizations.")
    """Render distributions for metrics.

    Args:
        metrics_df: DataFrame with metrics data.
    """
    if handle_empty_data(metrics_df, "No metrics data available for visualization."):
        return

    # Plotly supports Polars DataFrames natively

    # Exclude dataset_name
    metrics_cols = [col for col in metrics_df.columns if col != 'dataset_name']

    if not metrics_cols:
        st.info("No metrics found for visualization.")
        return

    # Create tabs for different visualizations
    tab_names = ["Histograms", "Box Plots", "Correlation", "Experiment Comparison", "Advanced 3D"]
    tabs = st.tabs(tab_names)

    with tabs[0]:  # Histograms
        st.subheader("Metrics Histograms")
        for metric in metrics_cols[:6]:  # Limit to first 6
            if metrics_df[metric].dtype in [pl.Int64, pl.Float64]:
                color_col = 'dataset_name' if 'dataset_name' in metrics_df.columns else None
                fig = _create_distribution_plot(
                    metrics_df, metric, "histogram", color_col, "Metrics "
                )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:  # Box Plots
        st.subheader("Metrics Box Plots")
        for metric in metrics_cols[:6]:
            if metrics_df[metric].dtype in ['int64', 'float64']:
                color_col = 'dataset_name' if 'dataset_name' in metrics_df.columns else None
                fig = _create_distribution_plot(
                    metrics_df, metric, "box", color_col, "Metrics "
                )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:  # Correlation
        st.subheader("Metrics Correlation")
        numeric_cols = [col for col in metrics_cols if metrics_df[col].dtype in ['int64', 'float64']]
        if len(numeric_cols) > 1:
            corr_matrix = metrics_df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                title="Metrics Correlation Matrix",
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric metrics for correlation analysis.")

    with tabs[3]:  # Experiment Comparison
        st.subheader("Metrics by Experiment")
        for metric in metrics_cols[:6]:
            if metrics_df[metric].dtype in [pl.Int64, pl.Float64] and 'experiment_name' in metrics_df.columns:
                fig = px.box(
                    metrics_df,
                    x='experiment_name',
                    y=metric,
                    title=f"{metric} by Experiment"
                )
                st.plotly_chart(fig, use_container_width=True)
            elif 'experiment_name' not in metrics_df.columns:
                st.info("Experiment information not available for comparison.")
                break

    with tabs[4]:  # Advanced 3D
        st.subheader("Advanced Visualizations")
        numeric_cols = [col for col in metrics_cols if metrics_df[col].dtype in [pl.Int64, pl.Float64]]

        if len(numeric_cols) >= 3:
            # 3D Scatter Plot
            st.markdown("### 3D Metrics Scatter Plot")
            col1, col2, col3 = st.columns(3)
            with col1:
                x_metric = st.selectbox("X-axis", numeric_cols, key="x_metric_3d")
            with col2:
                y_metric = st.selectbox("Y-axis", numeric_cols, key="y_metric_3d")
            with col3:
                z_metric = st.selectbox("Z-axis", numeric_cols, key="z_metric_3d")

            color_metric = st.selectbox("Color by", [None] + metrics_cols, key="color_metric_3d")

            if x_metric and y_metric and z_metric:
                render_3d_scatter_plot(
                    metrics_df, x_metric, y_metric, z_metric, color_metric,
                    title=f"3D Scatter: {x_metric} vs {y_metric} vs {z_metric}"
                )

            # Radar Chart
            st.markdown("### Metrics Radar Chart")
            selected_metrics = st.multiselect(
                "Select metrics for radar chart",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))],
                key="radar_metrics"
            )

            if len(selected_metrics) >= 3:
                category_col = st.selectbox(
                    "Group by (optional)",
                    [None] + [col for col in metrics_df.columns if col != 'dataset_name'],
                    key="radar_metrics_category"
                )
                render_radar_chart(
                    metrics_df, selected_metrics, category_col,
                    title="Metrics Comparison Radar Chart"
                )
            else:
                st.info("Select at least 3 metrics for radar chart.")
        else:
            st.info("Need at least 3 numeric metrics for advanced 3D visualizations.")


def add_chart_export_buttons(fig: go.Figure, filename_prefix: str = "chart") -> None:
    """Add export buttons for a Plotly chart.

    Args:
        fig: Plotly figure to add export options for.
        filename_prefix: Prefix for exported file names.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download PNG", key=f"png_{filename_prefix}"):
            # Note: In a real implementation, you'd use fig.write_image()
            # But this requires kaleido package
            st.info("PNG export would be implemented with fig.write_image()")

    with col2:
        if st.button("📄 Download PDF", key=f"pdf_{filename_prefix}"):
            st.info("PDF export would be implemented with fig.write_image()")

    with col3:
        if st.button("📊 Download HTML", key=f"html_{filename_prefix}"):
            # HTML export works without additional dependencies
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                fig.write_html(f.name)
                with open(f.name, 'rb') as file:
                    st.download_button(
                        label="Download HTML",
                        data=file,
                        file_name=f"{filename_prefix}.html",
                        mime="text/html"
                    )
            os.unlink(f.name)