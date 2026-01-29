"""
Settings page.

This page allows users to customize visualization settings, color palettes,
and graph configurations that persist across sessions.

3_⚙️_Settings.py
├── Color palette customization
├── Symbol selection
├── Graph configuration
├── Filter preferences
├── YAML configuration persistence
└── Settings preview

Usage:
    Accessed via navigation from Home.py
"""

import streamlit as st
import pandas as pd
import os

from statflow.config import (
    DEFAULT_PARETO_COLORS, DEFAULT_PARETO_SYMBOLS, DEFAULT_GRAPH_CONFIG,
    initialize_session_state, save_session_state_to_config, save_user_config, setup_sidebar
)


st.set_page_config(
    page_title=f"Settings - {st.session_state.get('app_name', 'Experiment Analysis')}",
    page_icon="⚙️",
    layout="wide",
)

# Initialize session state
initialize_session_state()

# Setup sidebar
setup_sidebar()


def main():
    st.title("⚙️ Settings")

    app_name = st.text_input(
        "🏷️ Application Name",
        value=st.session_state.get('app_name', 'Experiment Analysis'),
        max_chars=50,
        help="This name will be displayed in page titles and headers throughout the application."
    )

    if app_name != st.session_state.get('app_name', 'Experiment Analysis'):
        st.session_state.app_name = app_name
        st.success(f"Application name updated to: **{app_name}**")
        st.rerun()

    datasets_path = st.text_input(
        "📁 Datasets Path",
        value=st.session_state.get('datasets_path', 'datasets'),
        help="Path to the directory containing dataset CSV files."
    )

    if datasets_path != st.session_state.get('datasets_path', 'datasets'):
        st.session_state.datasets_path = datasets_path
        st.success(f"Datasets path updated to: **{datasets_path}**")
        st.rerun()

    # Configuration tabs
    tab_colors, tab_symbols = st.tabs([
        "🎨 Colors", "🔷 Symbols"
    ])

    config_changed = False

    with tab_colors:
        st.subheader("Color Palette Customization")

        st.markdown("Customize colors for different configuration variants in Pareto front plots.")

        variant_cols = st.columns(2)

        with variant_cols[0]:
            # GSGP colors
            st.markdown("#### GSGP Variants")
            cols = st.columns(4)

            with cols[0]:
                gsgp_std_color = st.color_picker(
                    "GSGP Standard",
                    value=st.session_state.get('custom_colors', {}).get('GSGP-std', DEFAULT_PARETO_COLORS['GSGP-std']),
                    key="gsgp_std_color"
                )

            with cols[1]:
                gsgp_oms_color = st.color_picker(
                    "GSGP-OMS",
                    value=st.session_state.get('custom_colors', {}).get('GSGP-OMS', DEFAULT_PARETO_COLORS['GSGP-OMS']),
                    key="gsgp_oms_color"
                )

        with variant_cols[1]:
            # SLIM color
            st.markdown("#### SLIM-GSGP")
            slim_color = st.color_picker(
                "SLIM",
                value=st.session_state.get('custom_colors', {}).get('SLIM', DEFAULT_PARETO_COLORS['SLIM']),
                key="slim_color"
            )

        # ARC beta colors
        st.markdown("#### ARC Beta Values")
        st.markdown("Assign colors to different beta values in ARC configurations:")

        arc_beta_colors = {}
        beta_values = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "1.0"]

        cols = st.columns(8)
        for i, beta in enumerate(beta_values):
            with cols[i % 8]:
                default_color = DEFAULT_PARETO_COLORS['arc_beta'].get(beta, "#808080")
                current_color = st.session_state.get('custom_colors', {}).get('arc_beta', {}).get(beta, default_color)
                arc_beta_colors[beta] = st.color_picker(
                    f"β = {beta}",
                    value=current_color,
                    key=f"arc_beta_{beta}_color"
                )
        
        st.space()

        cols_ = st.columns(2)
        with cols_[0]:
            # Update custom colors
            if st.button("Apply Color Changes", key="apply_colors"):
                custom_colors = {
                    'GSGP-std': gsgp_std_color,
                    'GSGP-OMS': gsgp_oms_color,
                    'SLIM': slim_color,
                    'arc_beta': arc_beta_colors
                }
                st.session_state.custom_colors = custom_colors
                config_changed = True
                st.success("Color palette updated!")

        with cols_[1]:
            # Reset to defaults
            if st.button("Reset to Defaults", key="reset_colors"):
                st.session_state.custom_colors = DEFAULT_PARETO_COLORS.copy()
                config_changed = True
                st.success("Colors reset to defaults!")

    with tab_symbols:
        st.subheader("Symbol Customization")

        st.markdown("Customize marker symbols for different configuration variants in Pareto front plots.")

        # Available Plotly symbols
        symbol_options = [
            "circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down",
            "pentagon", "hexagon", "star", "hexagram"
        ]

        # GSGP symbols
        st.markdown("#### GSGP Variants")
        col1, col2 = st.columns(2)

        with col1:
            gsgp_std_symbol = st.selectbox(
                "GSGP Standard Symbol",
                options=symbol_options,
                index=symbol_options.index(st.session_state.get('custom_symbols', {}).get('gsgp_std', DEFAULT_PARETO_SYMBOLS['gsgp_std'])),
                key="gsgp_std_symbol"
            )

        with col2:
            gsgp_oms_symbol = st.selectbox(
                "GSGP-OMS Symbol",
                options=symbol_options,
                index=symbol_options.index(st.session_state.get('custom_symbols', {}).get('gsgp_oms', DEFAULT_PARETO_SYMBOLS['gsgp_oms'])),
                key="gsgp_oms_symbol"
            )

        # SLIM symbol
        st.markdown("#### SLIM-GSGP")
        slim_symbol = st.selectbox(
            "SLIM Symbol",
            options=symbol_options,
            index=symbol_options.index(st.session_state.get('custom_symbols', {}).get('slim', DEFAULT_PARETO_SYMBOLS['slim'])),
            key="slim_symbol"
        )

        # ARC beta symbols
        st.markdown("#### ARC Beta Values")
        st.markdown("Assign symbols to different beta values in ARC configurations:")

        arc_beta_symbols = {}
        beta_values = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

        cols = st.columns(4)
        for i, beta in enumerate(beta_values):
            with cols[i % 4]:
                default_symbol = DEFAULT_PARETO_SYMBOLS['arc_beta'].get(beta, "circle")
                current_symbol = st.session_state.get('custom_symbols', {}).get('arc_beta', {}).get(beta, default_symbol)
                arc_beta_symbols[beta] = st.selectbox(
                    f"β = {beta}",
                    options=symbol_options,
                    index=symbol_options.index(current_symbol),
                    key=f"arc_beta_{beta}_symbol"
                )

        # Update custom symbols
        if st.button("Apply Symbol Changes", key="apply_symbols"):
            custom_symbols = {
                'gsgp_std': gsgp_std_symbol,
                'gsgp_oms': gsgp_oms_symbol,
                'slim': slim_symbol,
                'arc_beta': arc_beta_symbols
            }
            st.session_state.custom_symbols = custom_symbols
            config_changed = True
            st.success("Symbol settings updated!")

        # Reset to defaults
        if st.button("Reset to Defaults", key="reset_symbols"):
            st.session_state.custom_symbols = DEFAULT_PARETO_SYMBOLS.copy()
            config_changed = True
            st.success("Symbols reset to defaults!")


    # Configuration management
    st.markdown("---")
    st.subheader("💾 Configuration Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Configuration", key="save_config"):
            save_session_state_to_config()
            st.success("Configuration saved to disk!")

    with col2:
        if st.button("🔄 Reload Configuration", key="reload_config"):
            # Re-initialize session state from saved config
            initialize_session_state()
            st.success("Configuration reloaded from disk!")
            st.rerun()

    # Current configuration preview
    with st.expander("🔍 Current Configuration Preview", expanded=False):
        st.json({
            "use_custom_colors": st.session_state.get('use_custom_colors'),
            "show_mean": st.session_state.get('show_mean'),
            "show_error_bars": st.session_state.get('show_error_bars'),
            "graph_width": st.session_state.get('graph_width'),
            "graph_height": st.session_state.get('graph_height'),
            "points_display": st.session_state.get('points_display'),
            "custom_colors": st.session_state.get('custom_colors'),
            "custom_symbols": st.session_state.get('custom_symbols'),
        })

    # Mark config as changed if any updates were made
    if config_changed:
        st.session_state.config_changed = True
        save_session_state_to_config()


if __name__ == "__main__":
    main()