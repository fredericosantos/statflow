"""
Visualization utilities for colors, symbols, and plotting configuration.

This module provides functions for assigning colors and symbols to different
configuration variants, as well as managing visualization settings.

visualization.py
├── Color assignment (get_color_for_config, get_pareto_color)
├── Symbol assignment (get_marker_symbol)
├── Default color mapping (get_default_color_mapping)
└── Visualization configuration helpers

Usage:
    from statflow.utils.visualization import (
        get_color_for_config, get_pareto_color, get_marker_symbol
    )
"""

import pandas as pd
from typing import Dict, Optional


def get_pareto_color(group_label: str, custom_colors: Optional[Dict] = None) -> str:
    """Get color for Pareto front group.

    Args:
        group_label: Group label string.
        custom_colors: Optional dictionary with custom color mappings.

    Returns:
        Color hex string.
    """
    # Use custom colors if provided
    if custom_colors:
        if "GSGP-OMS" == group_label:
            return custom_colors.get("GSGP-OMS", "#5A6C7D")
        elif "GSGP-std" == group_label:
            return custom_colors.get("GSGP-std", "#B8C5D0")
        elif "SLIM" == group_label:
            return custom_colors.get("SLIM", "#D4A574")
        # ARC colors by beta value
        arc_beta_colors = custom_colors.get("arc_beta", {})
        for beta_val, color in arc_beta_colors.items():
            if f"$\\beta={beta_val}$" in group_label:
                return color

    # Fallback to default colors
    if "GSGP-OMS" == group_label:
        return "#5A6C7D"  # Dark slate
    elif "GSGP-std" == group_label:
        return "#B8C5D0"  # Light slate
    elif "SLIM" == group_label:
        return "#D4A574"  # Muted yellow/gold
    # ARC colors by beta value (distinct muted colors across spectrum)
    elif "$\\beta=0.0$" in group_label:
        return "#6B8CAE"  # Steel blue
    elif "$\\beta=0.1$" in group_label:
        return "#5B9AA8"  # Muted turquoise
    elif "$\\beta=0.2$" in group_label:
        return "#4D9B82"  # Seafoam green
    elif "$\\beta=0.3$" in group_label:
        return "#6AA56D"  # Sage green
    elif "$\\beta=0.4$" in group_label:
        return "#8FAE6C"  # Olive green
    elif "$\\beta=0.5$" in group_label:
        return "#B8A05C"  # Muted gold
    elif "$\\beta=0.6$" in group_label:
        return "#C4976C"  # Tan/beige
    elif "$\\beta=0.7$" in group_label:
        return "#B8846C"  # Muted brown
    elif "$\\beta=0.8$" in group_label:
        return "#9B7FA0"  # Muted purple/mauve
    elif "$\\beta=0.9$" in group_label:
        return "#8B6B8F"  # Darker purple
    elif "$\\beta=0.95$" in group_label:
        return "#C85C5C"  # Red (as requested)
    elif "$\\beta=1.0$" in group_label:
        return "#704545"  # Deep maroon
    elif group_label.startswith("ARC"):
        return "#9B8590"  # Muted gray-purple for any other ARC
    else:
        return "#808080"


def get_marker_symbol(group_label: str, custom_symbols: Optional[Dict] = None) -> str:
    """Get marker symbol for Pareto front group.

    Args:
        group_label: Group label string.
        custom_symbols: Optional dict with keys 'gsgp_std', 'gsgp_oms', 'slim', 'arc_beta' mapping to symbol names.

    Returns:
        Marker symbol string for Plotly.
    """
    # Use custom symbols if provided
    if custom_symbols:
        if "GSGP-OMS" == group_label:
            return custom_symbols.get("gsgp_oms", "circle")
        elif "GSGP-std" == group_label:
            return custom_symbols.get("gsgp_std", "square")
        elif "SLIM" == group_label:
            return custom_symbols.get("slim", "diamond")
        arc_beta_symbols = custom_symbols.get("arc_beta", {})
        for beta_val, sym in arc_beta_symbols.items():
            if f"$\\beta={beta_val}$" in group_label:
                return sym

    # Fallback defaults
    if "GSGP-OMS" == group_label:
        return "circle"
    elif "GSGP-std" == group_label:
        return "square"
    elif "SLIM" == group_label:
        return "diamond"
    # ARC symbols by beta value
    elif "$\\beta=0.0$" in group_label:
        return "circle"
    elif "$\\beta=0.1$" in group_label:
        return "square"
    elif "$\\beta=0.2$" in group_label:
        return "diamond"
    elif "$\\beta=0.3$" in group_label:
        return "cross"
    elif "$\\beta=0.4$" in group_label:
        return "x"
    elif "$\\beta=0.5$" in group_label:
        return "triangle-up"
    elif "$\\beta=0.6$" in group_label:
        return "triangle-down"
    elif "$\\beta=0.7$" in group_label:
        return "pentagon"
    elif "$\\beta=0.8$" in group_label:
        return "hexagon"
    elif "$\\beta=0.9$" in group_label:
        return "star"
    elif "$\\beta=1.0$" in group_label:
        return "hexagram"
    elif group_label.startswith("ARC"):
        return "circle"  # Default for any other ARC
    else:
        return "circle"


def get_default_color_mapping(groups: list[str]) -> Dict[str, str]:
    """Map pareto groups to Plotly default colors in a specific order.

    Plotly default color sequence:
    0: #636EFA (blue)
    1: #EF553B (red)
    2: #00CC96 (green)
    3: #AB63FA (purple)
    4: #FFA15A (orange)
    5: #19D3F3 (cyan)
    6: #FF6692 (pink)
    7: #B6E880 (light green)
    8: #FF97FF (light purple)
    9: #FECB52 (yellow)

    Args:
        groups: List of group labels.

    Returns:
        Dictionary mapping group labels to colors.
    """
    # Plotly default colors
    plotly_colors = [
        "#636EFA",  # blue
        "#EF553B",  # red
        "#00CC96",  # green
        "#AB63FA",  # purple
        "#FFA15A",  # orange
        "#19D3F3",  # cyan
        "#FF6692",  # pink
        "#B6E880",  # light green
        "#FF97FF",  # light purple
        "#FECB52",  # yellow
    ]

    # Sort groups to assign colors in desired order
    # Priority: β=0.0 (blue), β=0.95 (red), then others
    sorted_groups = []

    # First add GSGP variants
    for g in sorted(groups):
        if "GSGP" in g:
            sorted_groups.append(g)

    # Then add beta values in specific order
    beta_order = [
        "0.0",
        "0.95",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "1.0",
    ]
    for beta in beta_order:
        for g in groups:
            if f"$\\beta={beta}$" in g and g not in sorted_groups:
                sorted_groups.append(g)

    # Add SLIM
    for g in groups:
        if "SLIM" in g and g not in sorted_groups:
            sorted_groups.append(g)

    # Add any remaining groups
    for g in groups:
        if g not in sorted_groups:
            sorted_groups.append(g)

    # Create mapping
    color_map = {}
    for i, group in enumerate(sorted_groups):
        color_map[group] = plotly_colors[i % len(plotly_colors)]

    return color_map


def get_color_for_config(row: pd.Series, custom_colors: Optional[Dict] = None) -> str:
    """Assign a color to a configuration based on variant and parameters.

    Args:
        row: DataFrame row containing parameter values.
        custom_colors: Optional dictionary with custom color mappings.

    Returns:
        Color string in hex format.
    """
    variant = row.get("params.variant", "N/A")

    # Use custom colors if provided
    if custom_colors:
        if variant == "gsgp":
            use_oms = row.get("params.use_oms", "False")
            if use_oms == "True":
                return custom_colors.get("GSGP-OMS", "#5A6C7D")
            else:
                return custom_colors.get("GSGP-std", "#B8C5D0")
        elif variant == "slim_gsgp":
            return custom_colors.get("SLIM", "#D4A574")
        elif variant == "arc":
            beta_str = row.get("params.arc_beta", "0.0")
            arc_beta_colors = custom_colors.get("arc_beta", {})
            return arc_beta_colors.get(beta_str, "#9B8590")

    # Fallback to original logic with MPF-based colors
    if variant == "arc":
        # Get MPF and beta values
        mpf_str = row.get("params.mutation_pool_factor", "10")
        beta_str = row.get("params.arc_beta", "0.0")

        mpf = int(mpf_str)
        beta = float(beta_str)

        # Normalize beta to [0, 1] for intensity (1.0 = strongest, 0.0 = weakest)
        # Interpolate between light and dark shades
        intensity = 0.4 + (beta * 0.6)  # Range from 0.4 to 1.0

        # Modern muted base colors for different MPF values
        if mpf == 10:
            r, g, b = 106, 168, 156  # Muted teal
        elif mpf == 5:
            r, g, b = 156, 136, 140  # Muted rose
        elif mpf == 2:
            r, g, b = 140, 156, 136  # Muted sage
        elif mpf == 1:
            r, g, b = 168, 140, 156  # Muted lavender
        else:
            r, g, b = 128, 128, 128  # Gray fallback

        # Apply intensity
        r = int(r * intensity + 245 * (1 - intensity))
        g = int(g * intensity + 245 * (1 - intensity))
        b = int(b * intensity + 245 * (1 - intensity))

        return f"#{r:02x}{g:02x}{b:02x}"

    elif variant == "gsgp":
        use_oms = row.get("params.use_oms", "False")
        if use_oms == "True":
            return "#5A6C7D"  # Dark slate
        else:
            return "#B8C5D0"  # Light slate

    elif variant == "slim_gsgp":
        p_inflate_str = row.get("params.arc_beta", "0.5")

        p_inflate = float(p_inflate_str)

        # Normalize p_inflate: 0.1 = strongest (darkest), 0.9 = weakest (lightest)
        # Intensity decreases as p_inflate increases
        intensity = 1.0 - ((p_inflate - 0.1) / 0.8)  # Maps 0.1->1.0, 0.9->0.0
        intensity = max(0.4, min(1.0, intensity))  # Clamp to [0.4, 1.0]

        # Modern muted green base
        r = int(106 * intensity + 230 * (1 - intensity))
        g = int(176 * intensity + 240 * (1 - intensity))
        b = int(124 * intensity + 230 * (1 - intensity))

        return f"#{r:02x}{g:02x}{b:02x}"

    else:
        return "#808080"  # Gray for unknown variants