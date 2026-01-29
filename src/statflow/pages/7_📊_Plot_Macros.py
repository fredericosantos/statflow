"""
Plot Macros page for LaTeX code generation.

This page generates LaTeX code compatible with plotmacros.sty for creating
publication-quality plots from experiment data.

5_📊_Plot_Macros.py
├── Dataset selection from existing data
├── Plot type configuration (Boxplot, Line, Histogram)
├── Color customization
├── LaTeX code generation
└── Integration with plotmacros.sty

Usage:
    Accessed via navigation from Home.py
"""

import streamlit as st
import pandas as pd
import math
import re
import io
import zipfile

from statflow.config import (
    initialize_session_state, save_session_state_to_config, setup_sidebar
)
from statflow.utils.mlflow_client import fetch_all_datasets_parallel
from statflow.components.filters import render_dataset_selector, render_global_filters
from statflow.pages_modules.module_7_Plot_Macros.processor import (
    fetch_plot_data
)


# --- DEFAULT COLORS FROM plotmacros.sty ---
DEFAULT_COLORS = {
    "col11": "#E41A1C",
    "col12": "#377EB8",
    "col13": "#4DAF4A",
    "col14": "#984EA3",
    "col15": "#FF7F00",
    "col16": "#FFFF33",
    "col17": "#A65628",
    "col18": "#F781BF",
    "col19": "#8DD3C7",
    "col21": "#66C2A5",
    "col22": "#FC8D62",
    "col23": "#8DA0CB",
    "col24": "#E78AC3",
    "col25": "#A6D854",
    "col31": "#8DD3C7",
    "col32": "#DC267F",
    "col33": "#FFFFB3",
}


st.set_page_config(
    page_title=f"Plot Macros - {st.session_state.get('app_name', 'Experiment Analysis')}",
    page_icon="📊",
    layout="wide",
)

# Initialize session state
initialize_session_state()

# Setup sidebar
setup_sidebar()


# Initialize custom colors in session state
if "custom_colors_plotmacros" not in st.session_state:
    st.session_state["custom_colors_plotmacros"] = {}


def has_issues(name):
    """Check if a name has spaces or special characters that need cleaning."""
    return ' ' in name or any(not (c.isalnum() or c in '_-') for c in name)


def config_to_column_name(config: str) -> str:
    """Convert a configuration name to a LaTeX-safe column name.
    
    Examples:
        "ARC (β = 0.0)" -> "arc_beta000"
        "ARC (β = 0.95)" -> "arc_beta095"
        "ARC (β = 1.0)" -> "arc_beta100"
        "SLIM (p_inflate = 0.1)" -> "slim_p01"
        "GSGP-OMS" -> "gsgp_oms"
        "GSGP-std" -> "gsgp_std"
    """
    import re
    
    # ARC with beta value
    arc_match = re.match(r'ARC\s*\(β\s*=\s*([\d.]+)\)', config)
    if arc_match:
        beta = arc_match.group(1)
        # Convert to format like "000", "025", "095", "100"
        beta_float = float(beta)
        beta_int = int(beta_float * 100)
        return f"arc_beta{beta_int:03d}"
    
    # SLIM with p_inflate value
    slim_match = re.match(r'SLIM\s*\(p_inflate\s*=\s*([\d.]+)\)', config)
    if slim_match:
        p_inflate = slim_match.group(1)
        # Convert to format like "p01", "p03", "p05"
        p_float = float(p_inflate)
        p_int = int(p_float * 10)
        return f"slim_p{p_int:02d}"
    
    # GSGP variants
    if config == "GSGP-OMS":
        return "gsgp_oms"
    if config == "GSGP-std":
        return "gsgp_std"
    
    # Fallback: clean the name
    clean = re.sub(r'[^a-zA-Z0-9]', '_', config.lower())
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean


def create_config_mappings(config_names: list) -> tuple:
    """Create mappings from config names to LaTeX-safe column names.
    
    Args:
        config_names: List of configuration names
        
    Returns:
        tuple: (cleaned_configs, mappings) where mappings maps original to column names
    """
    mappings = {}
    cleaned_configs = []
    
    for config in config_names:
        col_name = config_to_column_name(config)
        mappings[config] = col_name
        cleaned_configs.append(col_name)
    
    return cleaned_configs, mappings

def calculate_grid_dims(num_files: int, max_cols: int = 5) -> tuple:
    """Calculate grid dimensions (cols, rows) from number of files."""
    cols = min(max_cols, num_files)
    rows = math.ceil(num_files / cols)
    return cols, rows


def hex_to_rgb(hex_code: str) -> tuple:
    """Convert hex color to RGB tuple."""
    h = hex_code.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def sanitize_dataset_name(name: str) -> str:
    """Convert dataset name to LaTeX-safe format using session state renames."""
    from statflow.config import DEFAULT_DATASET_RENAMES
    
    # Get renamed version from session state
    renames = st.session_state.get('dataset_renames', DEFAULT_DATASET_RENAMES)
    display_name = renames.get(name, name)
    
    # If no custom rename, fall back to basic sanitization
    if display_name == name:
        # Remove underscores and special characters
        display_name = name.replace('_', ' ')
        # Remove 'blackbox' prefix and numbers
        display_name = re.sub(r'blackbox\s+\d+\s+', '', display_name)
    
    return display_name


def get_config_columns(df: pd.DataFrame) -> list:
    """Extract configuration columns from dataframe."""
    # Look for columns that represent different configurations
    # Typically these would be columns like "ARC (β = 0.0)", "GSGP-OMS", etc.
    exclude_cols = ['run_id', 'dataset_name', 'params.variant', 'params.arc_beta', 
                    'params.mutation_pool_factor', 'params.use_oms', 'metrics.best_test_fitness',
                    'metrics.best_n_nodes']
    return [col for col in df.columns if col not in exclude_cols]


def generate_csv_data_for_plot_type(all_runs_df, plot_type, selected_datasets, selected_configs, config_mappings):
    """Generate CSV data for each dataset based on plot type.
    
    Args:
        all_runs_df: DataFrame with all experiment runs
        plot_type: "Boxplot", "Line (with Error Bands)", or "Histogram"
        selected_datasets: List of selected dataset names
        selected_configs: List of selected configuration names
        config_mappings: Dict mapping original config names to VAR names
        
    Returns:
        Dict mapping dataset names to CSV content strings
    """
    csv_files = {}
    
    for dataset_name in selected_datasets:
        # Filter data for this dataset
        dataset_df = all_runs_df[all_runs_df["dataset_name"] == dataset_name].copy()
        
        if plot_type == "Boxplot":
            # For boxplots, create a CSV with one column per configuration
            # Each column contains all fitness values for that configuration
            
            # Collect all fitness values first
            data_dict = {}
            for config in selected_configs:
                config_data = dataset_df[dataset_df["config_group"] == config]
                if not config_data.empty:
                    fitness_values = config_data["metrics.best_test_fitness"].tolist()
                else:
                    fitness_values = []
                data_dict[config_mappings.get(config, config)] = fitness_values
            
            # Find max length and pad all lists to same length
            max_len = max(len(values) for values in data_dict.values()) if data_dict else 0
            for key in data_dict:
                if len(data_dict[key]) < max_len:
                    data_dict[key] += [float('nan')] * (max_len - len(data_dict[key]))
            
            # Create DataFrame
            csv_df = pd.DataFrame(data_dict)
            
            csv_files[f"{dataset_name}.csv"] = csv_df.to_csv(index=False, sep=' ')
            
        elif plot_type == "Line (with Error Bands)":
            # For line plots, we need generation, fitness, and error columns
            # This is more complex - we'd need convergence data
            # For now, create a placeholder structure
            csv_df = pd.DataFrame({
                "generation": list(range(10)),  # Placeholder generations
            })
            
            for config in selected_configs:
                config_data = dataset_df[dataset_df["config_group"] == config]
                if not config_data.empty:
                    # Use mean fitness as placeholder (would need actual convergence data)
                    fitness_values = [config_data["metrics.best_test_fitness"].mean()] * 10
                    error_values = [config_data["metrics.best_test_fitness"].std()] * 10
                    csv_df[config_mappings.get(config, config)] = fitness_values
                    csv_df[f"{config_mappings.get(config, config)}_err"] = error_values
                else:
                    csv_df[config_mappings.get(config, config)] = [float('nan')] * 10
                    csv_df[f"{config_mappings.get(config, config)}_err"] = [float('nan')] * 10
            
            csv_files[f"{dataset_name}.csv"] = csv_df.to_csv(index=False, sep=' ')
            
        else:  # Histogram
            # For histograms, create a CSV with one column per configuration
            # Each column contains all fitness values for that configuration
            
            # Collect all fitness values first
            data_dict = {}
            for config in selected_configs:
                config_data = dataset_df[dataset_df["config_group"] == config]
                if not config_data.empty:
                    fitness_values = config_data["metrics.best_test_fitness"].tolist()
                else:
                    fitness_values = []
                data_dict[config_mappings.get(config, config)] = fitness_values
            
            # Find max length and pad all lists to same length
            max_len = max(len(values) for values in data_dict.values()) if data_dict else 0
            for key in data_dict:
                if len(data_dict[key]) < max_len:
                    data_dict[key] += [float('nan')] * (max_len - len(data_dict[key]))
            
            # Create DataFrame
            csv_df = pd.DataFrame(data_dict)
            
            csv_files[f"{dataset_name}.csv"] = csv_df.to_csv(index=False, sep=' ')
    
    return csv_files

def format_method_name_latex(method: str) -> str:
    """Format method name for LaTeX output."""
    if method.startswith("ARC (β = "):
        beta_val = method.replace("ARC (β = ", "").replace(")", "")
        return f"ARC $(\\beta = {beta_val})$"
    elif method == "GSGP-OMS":
        return "GSGP-OMS"
    elif method == "GSGP-std":
        return "GSGP-std"
    elif method == "SLIM":
        return "SLIM"
    return method


def generate_boxplot_groupplot_latex(
    datasets: list,
    configs: list,
    options: dict,
    params: dict,
    config_mappings: dict = None,
    folder_path: str = "data/boxplots"
) -> str:
    """Generate LaTeX code for boxplot groupplot."""
    num_datasets = len(datasets)
    cols, rows = calculate_grid_dims(num_datasets, max_cols=5)
    
    linewidth = params.get("linewidth", 1.25)
    
    code = f"% Use: \\usepackage[linewidth={linewidth}]{{plotmacros}}\n"
    code += "\\begin{figure*}[!ht]\n"
    code += "    \\centering\n\n"
    
    # Define dataset variables
    import string
    letters = list(string.ascii_uppercase)
    for idx, dataset_name in enumerate(datasets, start=1):
        if idx <= 26:
            letter = letters[idx-1]
        else:
            letter = letters[(idx-1)//26 - 1] + letters[(idx-1)%26]
        
        transformed_title = sanitize_dataset_name(dataset_name)
        code += f"    \\def\\dataset{letter}Title{{\\text{{{transformed_title}}}}}\n"
        code += f"    \\def\\dataset{letter}Path{{{folder_path}{dataset_name}.csv}}\n"
    
    code += "\n    \\begin{tikzpicture}\n"
    code += "        \\begin{groupplot}[\n"
    code += "            boxplot,\n"
    code += "            boxplot/draw direction=y,\n"
    code += "            width=5cm,\n"
    code += "            height=4.5cm,\n"
    code += "            group style={\n"
    code += f"                group size={cols} by {rows},\n"
    code += "                horizontal sep=2mm,\n"
    code += "                vertical sep=6mm,\n"
    code += "                yticklabels at=edge left,\n"
    code += "            },\n"
    code += "            ygridded,\n"
    code += "            noinnerticks,\n"
    code += "            xtick={9},\n"
    code += "        ]\n\n"
    
    # Generate subplots
    for idx, dataset_name in enumerate(datasets, start=1):
        if idx <= 26:
            letter = letters[idx-1]
        else:
            letter = letters[(idx-1)//26 - 1] + letters[(idx-1)%26]
        
        is_left_edge = ((idx - 1) % cols) == 0
        
        code += "            \\nextgroupplot[\n"
        code += f"                title={{\\dataset{letter}Title}},\n"
        
        if is_left_edge:
            code += f"                ylabel={{{params.get('ylabel', 'Fitness')}}},\n"
        
        code += "            ]\n\n"
        
        # Get gap parameters
        gap_after = params.get("gap_after")
        gap_size = params.get("gap_size", 0.5)  # In boxplot units, not mm
        num_configs = len(configs)
        
        # Add boxplots for each configuration
        for box_idx, config in enumerate(configs, start=1):
            opts = options["boxplot"][config]
            color_name = opts["color"]
            # Use mapped name if available, otherwise use original
            latex_config_name = config_mappings.get(config, config) if config_mappings else config
            
            # Calculate draw position with gap
            if gap_after:
                # Base position for this boxplot
                base_pos = box_idx
                # Add gap offset for boxplots after the gap position
                if box_idx > gap_after:
                    draw_pos = base_pos + gap_size
                else:
                    draw_pos = base_pos
                code += f"            \\addplot[boxplot, draw={color_name}, fill={color_name}, fill opacity=0.3, boxplot/draw position={draw_pos}] table[y={{{latex_config_name}}}] {{\\dataset{letter}Path}};\n"
            else:
                code += f"            \\boxplot[bpcolor={color_name}]{{\\dataset{letter}Path}}{{{latex_config_name}}}\n"
        code += "\n"
    
    code += "        \\end{groupplot}\n"
    code += "    \\end{tikzpicture}\n\n"
    
    # Add legend
    mid = len(configs) // 2
    first_half = configs[:mid]
    second_half = configs[mid:]
    
    for config in first_half:
        color_name = options["boxplot"][config]["color"]
        code += f"    \\addlegendimageintext{{boxplot legend image={{{color_name}}}{{solid}}}} {config}\n"
    code += "    \\\\\n"
    for config in second_half:
        color_name = options["boxplot"][config]["color"]
        code += f"    \\addlegendimageintext{{boxplot legend image={{{color_name}}}{{solid}}}} {config}\n"
    
    code += f"    \\caption{{Boxplot comparison across {len(datasets)} datasets.}}\n"
    code += "    \\label{fig:boxplot_grid}\n"
    code += "\\end{figure*}\n"
    
    return code


def generate_line_groupplot_latex(
    datasets: list,
    configs: list,
    options: dict,
    params: dict,
    config_mappings: dict = None,
    folder_path: str = "data/lines"
) -> str:
    """Generate LaTeX code for line plot groupplot."""
    num_datasets = len(datasets)
    cols, rows = calculate_grid_dims(num_datasets, max_cols=5)
    
    linewidth = params.get("linewidth", 1.25)
    
    code = f"% Use: \\usepackage[linewidth={linewidth}]{{plotmacros}}\n"
    code += "\\begin{figure*}[!ht]\n"
    code += "    \\centering\n\n"
    
    # Define dataset variables
    import string
    letters = list(string.ascii_uppercase)
    for idx, dataset_name in enumerate(datasets, start=1):
        if idx <= 26:
            letter = letters[idx-1]
        else:
            letter = letters[(idx-1)//26 - 1] + letters[(idx-1)%26]
        
        transformed_title = sanitize_dataset_name(dataset_name)
        code += f"    \\def\\dataset{letter}Title{{\\text{{{transformed_title}}}}}\n"
        code += f"    \\def\\dataset{letter}Path{{{folder_path}{dataset_name}.csv}}\n"
    
    code += "\n    \\begin{tikzpicture}\n"
    code += "        \\begin{groupplot}[\n"
    code += "            width=5cm,\n"
    code += "            height=4.5cm,\n"
    code += "            group style={\n"
    code += f"                group size={cols} by {rows},\n"
    code += "                horizontal sep=2mm,\n"
    code += "                vertical sep=6mm,\n"
    code += "                yticklabels at=edge left,\n"
    code += "            },\n"
    code += "            ygridded,\n"
    code += "            legend pos=north east,\n"
    code += "        ]\n\n"
    
    # Generate subplots
    for idx, dataset_name in enumerate(datasets, start=1):
        if idx <= 26:
            letter = letters[idx-1]
        else:
            letter = letters[(idx-1)//26 - 1] + letters[(idx-1)%26]
        
        is_left_edge = ((idx - 1) % cols) == 0
        
        code += "            \\nextgroupplot[\n"
        code += f"                title={{\\dataset{letter}Title}},\n"
        code += f"                xlabel={{{params.get('xlabel', 'Generation')}}},\n"
        
        if is_left_edge:
            code += f"                ylabel={{{params.get('ylabel', 'Fitness')}}},\n"
        
        code += "            ]\n\n"
        
        # Add line plots for each configuration
        for config in configs:
            opts = options["line"][config]
            color_name = opts["color"]
            ltype = opts.get("ltype", "solid")
            # Use mapped names for column references
            latex_config_name = config_mappings.get(config, config) if config_mappings else config
            latex_err_name = f"{latex_config_name}_err"
            code += f"            \\lineerr[lcolor={color_name}, ltype={ltype}]{{\\dataset{letter}Path}}{{}}{{generation}}{{{latex_config_name}}}{{{latex_err_name}}}\n"
        code += "\n"
    
    code += "        \\end{groupplot}\n"
    code += "    \\end{tikzpicture}\n\n"
    
    # Add legend
    for config in configs:
        color_name = options["line"][config]["color"]
        ltype = options["line"][config].get("ltype", "solid")
        code += f"    \\addlegendimageintext{{line legend image={{{color_name}}}{{{ltype}}}}} {config}\n"
    
    code += f"    \\caption{{Line plots with error bands across {len(datasets)} datasets.}}\n"
    code += "    \\label{fig:line_grid}\n"
    code += "\\end{figure*}\n"
    
    return code


def generate_histogram_groupplot_latex(
    datasets: list,
    configs: list,
    options: dict,
    params: dict,
    config_mappings: dict = None,
    folder_path: str = "data/histograms"
) -> str:
    """Generate LaTeX code for histogram groupplot."""
    num_datasets = len(datasets)
    cols, rows = calculate_grid_dims(num_datasets, max_cols=5)
    bins = params.get("bins", 20)
    linewidth = params.get("linewidth", 1.25)
    
    code = f"% Use: \\usepackage[linewidth={linewidth}]{{plotmacros}}\n"
    code += "\\begin{figure*}[!ht]\n"
    code += "    \\centering\n\n"
    
    # Define dataset variables
    import string
    letters = list(string.ascii_uppercase)
    for idx, dataset_name in enumerate(datasets, start=1):
        if idx <= 26:
            letter = letters[idx-1]
        else:
            letter = letters[(idx-1)//26 - 1] + letters[(idx-1)%26]
        
        transformed_title = sanitize_dataset_name(dataset_name)
        code += f"    \\def\\dataset{letter}Title{{\\text{{{transformed_title}}}}}\n"
        code += f"    \\def\\dataset{letter}Path{{{folder_path}{dataset_name}.csv}}\n"
    
    code += "\n    \\begin{tikzpicture}\n"
    code += "        \\begin{groupplot}[\n"
    code += "            width=5cm,\n"
    code += "            height=4.5cm,\n"
    code += "            group style={\n"
    code += f"                group size={cols} by {rows},\n"
    code += "                horizontal sep=2mm,\n"
    code += "                vertical sep=6mm,\n"
    code += "                yticklabels at=edge left,\n"
    code += "            },\n"
    code += "            ygridded,\n"
    code += "            ybar,\n"
    code += "            bar width=2pt,\n"
    code += "        ]\n\n"
    
    # Generate subplots
    for idx, dataset_name in enumerate(datasets, start=1):
        if idx <= 26:
            letter = letters[idx-1]
        else:
            letter = letters[(idx-1)//26 - 1] + letters[(idx-1)%26]
        
        is_left_edge = ((idx - 1) % cols) == 0
        
        code += "            \\nextgroupplot[\n"
        code += f"                title={{\\dataset{letter}Title}},\n"
        code += f"                xlabel={{{params.get('xlabel', 'Value')}}},\n"
        
        if is_left_edge:
            code += f"                ylabel={{{params.get('ylabel', 'Frequency')}}},\n"
        
        code += "            ]\n\n"
        
        # Add histograms for each configuration
        for config in configs:
            opts = options["hist"][config]
            color_name = opts["color"]
            opacity = opts.get("opacity", 0.5)
            # Use mapped name if available
            latex_config_name = config_mappings.get(config, config) if config_mappings else config
            code += f"            \\hist[bcolor={color_name}, fillopacity={opacity}]{{\\dataset{letter}Path}}{{{latex_config_name}}}{{0}}{{100}}{{{bins}}}\n"
        code += "\n"
    
    code += "        \\end{groupplot}\n"
    code += "    \\end{tikzpicture}\n\n"
    
    # Add legend
    for config in configs:
        color_name = options["hist"][config]["color"]
        code += f"    \\addlegendimageintext{{hist legend image={{{color_name}}}{{solid}}}} {config}\n"
    
    code += f"    \\caption{{Histogram comparison across {len(datasets)} datasets.}}\n"
    code += "    \\label{fig:hist_grid}\n"
    code += "\\end{figure*}\n"
    
    return code


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    
    # Title and navigation
    st.title("📊 Plot Macros - LaTeX Code Generator")
    st.markdown("Generate LaTeX code compatible with `plotmacros.sty` for publication-quality plots.")
    
    if st.button("← Back to Home", key="back_home"):
        st.switch_page("Home.py")
    
    st.markdown("---")
    
    # Sidebar: Dataset and filter selection
    with st.sidebar:
        st.markdown("### Plot Type")
        plot_type = st.selectbox(
            "Select Plot Type:",
            options=["Boxplot", "Line (with Error Bands)", "Histogram"],
            help="Choose the type of plot to generate"
        )
        
        st.markdown("---")
        st.markdown("### Dataset Selection")
        selected_datasets = render_dataset_selector(selection_mode="multi")
        
        # Dataset names customization (shared across app)
        from statflow.components.filters import render_dataset_names_expander
        render_dataset_names_expander()
        
        st.markdown("---")
        
        # Global filters
        available_mpf = ["1", "2", "5", "10"]
        available_beta = ["0.0", "0.25", "0.5", "0.7", "0.9", "0.95", "1.0"]
        available_pinflate = ["0.1", "0.3", "0.5", "0.7", "0.9"]
        
        _, selected_mpf_values, selected_beta_values, selected_pinflate_values = render_global_filters(
            available_mpf, available_beta, available_pinflate
        )
        
        st.markdown("---")
        
        # Color palette section
        st.markdown("### Color Palette")
        st.markdown("**Default colors from `plotmacros.sty`**")
        
        # Display default colors in compact grid
        color_cols = st.columns(3)
        for i, (color_name, hex_color) in enumerate(DEFAULT_COLORS.items()):
            with color_cols[i % 3]:
                st.markdown(
                    f"<div style='text-align: center; font-size: 10px;'>"
                    f"<strong>{color_name}</strong><br>"
                    f"<span style='color:{hex_color}; font-size:30px;'>●</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    
    # Convert to tuples for caching
    selected_mpf_tuple = tuple(selected_mpf_values) if selected_mpf_values else None
    selected_beta_tuple = tuple(selected_beta_values) if selected_beta_values else None
    selected_pinflate_tuple = tuple(selected_pinflate_values) if selected_pinflate_values else None
    
    # Check if datasets are selected
    if not selected_datasets:
        st.info("👈 Please select at least one dataset from the sidebar to begin.")
        
        # Show example
        with st.expander("📋 Example Output", expanded=True):
            st.markdown("### What you'll get:")
            st.markdown("""
            This tool generates LaTeX code for creating professional plots using `plotmacros.sty`:
            
            **Features:**
            - 📦 **Boxplots**: Compare distributions across configurations
            - 📈 **Line plots**: Show convergence with error bands
            - 📊 **Histograms**: Display frequency distributions
            - 🎨 **Customizable colors**: Match your publication style
            - 🔧 **Grid layouts**: Automatically arrange multiple datasets
            
            **Workflow:**
            1. Select datasets from the sidebar
            2. Choose plot type and configurations
            3. Customize colors and styling
            4. Copy the generated LaTeX code
            5. Use in your Overleaf document with `plotmacros.sty`
            """)
        return
    
    # Fetch data
    with st.spinner(f"Fetching data for {len(selected_datasets)} dataset(s)..."):
        all_runs_df = fetch_plot_data(
            selected_mpf_tuple, selected_beta_tuple, selected_pinflate_tuple, selected_datasets
        )
        
        if all_runs_df is None:
            st.error("No data found with the selected filters.")
            return
    
    st.success(f"✓ Loaded data for {len(selected_datasets)} dataset(s)")
    
    # Add config_group column for grouping configurations
    from statflow.utils.data_processing import create_group_label
    all_runs_df_copy = all_runs_df.copy()
    all_runs_df_copy["config_group"] = all_runs_df_copy.apply(create_group_label, axis=1)
    
    # Main content area: Two columns
    col_config, col_output = st.columns([1, 1])
    
    with col_config:
        st.subheader("⚙️ Plot Configuration")
        
        # Get unique configurations from data
        available_configs = sorted(all_runs_df_copy["config_group"].unique().tolist())
        
        # Create column name mappings for LaTeX/CSV compatibility
        cleaned_configs, config_mappings = create_config_mappings(available_configs)
        
        # Show mappings
        if config_mappings:
            st.info("💡 Configuration names mapped to LaTeX-safe column names:")
            mapping_df = pd.DataFrame({
                "Display Name": list(config_mappings.keys()),
                "Column Name": list(config_mappings.values())
            })
            st.dataframe(mapping_df, use_container_width=True)
            st.markdown("---")
        
        st.markdown("#### Configuration Colors")
        st.caption(f"Customize colors for {len(available_configs)} available configurations")
        
        # Configuration color selection (all configs are included since filtering is done in sidebar)
        selected_configs = available_configs.copy()
        config_options = {}
        
        available_colors = {**DEFAULT_COLORS, **st.session_state["custom_colors_plotmacros"]}
        default_colors_list = list(DEFAULT_COLORS.keys())
        
        for idx, config in enumerate(available_configs):
            # Use cleaned name for display if it exists, otherwise use original
            display_name = config_mappings.get(config, config)
            
            # Layout: config name | color picker popover | color circle
            col_name, col_picker, col_circle = st.columns([2, 1, 0.5])
            
            with col_name:
                if config in config_mappings:
                    st.markdown(f"**{display_name}** *(was: {config})*")
                else:
                    st.markdown(f"**{display_name}**")
            
            with col_picker:
                default_color = default_colors_list[idx % len(default_colors_list)]
                selected_color = st.session_state.get(f"color_{config}", default_color)
                if selected_color not in available_colors:
                    selected_color = list(available_colors.keys())[0]
                
                # Color picker using pills
                color_key = f"pills_color_{config}"
                if color_key not in st.session_state:
                    st.session_state[color_key] = selected_color
                
                with st.popover("🎨"):
                    selected = st.pills(
                        "Color:",
                        options=list(available_colors.keys()),
                        default=st.session_state[color_key],
                        selection_mode='single',
                        key=f"pills_select_{config}",
                        label_visibility="collapsed"
                    )
                    if selected:
                        st.session_state[color_key] = selected
                        st.session_state[f"color_{config}"] = selected
            
            with col_circle:
                # Show selected color circle
                final_color = st.session_state[color_key]
                hex_color = available_colors[final_color]
                st.markdown(
                    f"<div style='text-align: center; padding-top: 8px;'>"
                    f"<span style='color:{hex_color}; font-size:20px;'>●</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            config_options[config] = {"color": final_color, "opacity": 0.5}
        
        st.markdown("---")
        st.markdown("#### Styling Options")
        
        # Line width option
        linewidth = st.slider(
            "Line Width",
            min_value=0.5,
            max_value=3.0,
            value=1.25,
            step=0.25,
            help="Line width for plot elements (default 1.25 in plotmacros.sty)"
        )
        st.session_state["plot_linewidth"] = linewidth
        
        # Boxplot group spacing
        if plot_type == "Boxplot":
            st.markdown("**Group Spacing**")
            st.caption("Add a gap between groups of boxplots (e.g., ARC | Others)")
            
            use_group_spacing = st.checkbox(
                "Enable group spacing",
                value=False,
                help="Add a gap between boxplot groups"
            )
            
            if use_group_spacing:
                col_gap_after, col_gap_size = st.columns(2)
                with col_gap_after:
                    gap_after_n = st.number_input(
                        "Gap after boxplot #",
                        min_value=1,
                        max_value=20,
                        value=3,
                        help="Insert gap after this boxplot number"
                    )
                with col_gap_size:
                    gap_size = st.slider(
                        "Gap size",
                        min_value=0.2,
                        max_value=2.0,
                        value=0.5,
                        step=0.1,
                        help="Size of the gap (in boxplot position units)"
                    )
                st.session_state["boxplot_gap_after"] = gap_after_n
                st.session_state["boxplot_gap_size"] = gap_size
            else:
                st.session_state["boxplot_gap_after"] = None
                st.session_state["boxplot_gap_size"] = None
        
        if plot_type == "Histogram":
            bins = st.number_input(
                "Number of Bins",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="Number of bins for histogram"
            )
            st.session_state["hist_bins"] = bins
        
        xlabel = st.text_input("X-axis Label", "Configuration")
        ylabel = st.text_input("Y-axis Label", "Test Fitness (RMSE)")
        folder_path = st.text_input(
            "Data Folder Path",
            f"data/{plot_type.lower().replace(' ', '_')}",
            help="Path where CSV files will be located in Overleaf"
        )
        
        params = {
            "xlabel": xlabel,
            "ylabel": ylabel,
            "linewidth": linewidth,
            "gap_after": st.session_state.get("boxplot_gap_after"),
            "gap_size": st.session_state.get("boxplot_gap_size"),
        }
    
    with col_output:
        st.subheader("📄 Generated LaTeX Code")
        
        if not selected_configs:
            st.info("👈 Select configurations to generate LaTeX code")
        else:
            # Prepare options dict based on plot type
            if plot_type == "Boxplot":
                options_dict = {
                    "boxplot": {config: config_options[config] for config in selected_configs}
                }
                latex_code = generate_boxplot_groupplot_latex(
                    selected_datasets,
                    selected_configs,
                    options_dict,
                    params,
                    config_mappings,
                    folder_path
                )
            elif plot_type == "Line (with Error Bands)":
                options_dict = {
                    "line": {config: config_options[config] for config in selected_configs}
                }
                latex_code = generate_line_groupplot_latex(
                    selected_datasets,
                    selected_configs,
                    options_dict,
                    params,
                    config_mappings,
                    folder_path
                )
            else:  # Histogram
                # Add bins parameter for histograms
                bins = st.session_state.get("hist_bins", 20)
                params["bins"] = bins
                
                options_dict = {
                    "hist": {config: config_options[config] for config in selected_configs}
                }
                latex_code = generate_histogram_groupplot_latex(
                    selected_datasets,
                    selected_configs,
                    options_dict,
                    params,
                    config_mappings,
                    folder_path
                )
            
            # Show custom color definitions if any
            custom_colors = st.session_state["custom_colors_plotmacros"]
            if custom_colors:
                st.info("💡 Add these custom color definitions to your LaTeX document")
                color_code = ""
                for name, hex_color in custom_colors.items():
                    rgb = hex_to_rgb(hex_color)
                    color_code += f"\\definecolor{{{name}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}\n"
                st.code(color_code, language="latex")
                st.markdown("---")
            
            # Show generated code
            st.code(latex_code, language="latex", line_numbers=False, height="stretch")
            
            # Download button
            st.download_button(
                label="📥 Download LaTeX Code",
                data=latex_code,
                file_name=f"{plot_type.lower().replace(' ', '_')}_plot.tex",
                mime="text/plain"
            )
            
            # Generate and provide CSV download
            st.markdown("---")
            st.subheader("📊 CSV Data Files")
            
            with st.spinner("Generating CSV files..."):
                csv_files = generate_csv_data_for_plot_type(
                    all_runs_df_copy, plot_type, selected_datasets, selected_configs, config_mappings
                )
            
            if csv_files:
                st.info(f"💾 Generated {len(csv_files)} CSV files with semantic column names (arc_beta000, slim_p01, gsgp_oms, etc.)")
                
                # Create zip file with all CSVs
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, csv_content in csv_files.items():
                        zip_file.writestr(filename, csv_content)
                
                zip_buffer.seek(0)
                zip_data = zip_buffer.getvalue()
                
                st.download_button(
                    label="📦 Download All CSV Files (ZIP)",
                    data=zip_data,
                    file_name=f"{plot_type.lower().replace(' ', '_')}_data.zip",
                    mime="application/zip",
                    help="Download CSV files with semantic column names for use with the LaTeX code"
                )
                
                # Show preview of first CSV
                if csv_files:
                    first_filename = list(csv_files.keys())[0]
                    st.markdown(f"**Preview of {first_filename}:**")
                    # Show first few lines of CSV
                    csv_preview = csv_files[first_filename].split('\n')[:6]  # First 6 lines
                    st.code('\n'.join(csv_preview), language="csv")
            else:
                st.warning("No CSV files could be generated.")
            
            # Usage instructions
            with st.expander("📖 How to use this code"):
                st.markdown("""
                ### Steps to use in Overleaf:
                
                1. **Download the CSV files** using the ZIP download button above
                2. **Upload the CSV files** to the specified folder path in Overleaf
                3. **Include plotmacros.sty** in your project
                4. **Copy the LaTeX code** above
                5. **Paste it** into your document where you want the plot
                6. **Compile** your document
                
                ### CSV Format:
                The downloaded CSV files are automatically formatted with the correct column names (VAR names where applicable) for LaTeX compatibility.
                Each file corresponds to one dataset and contains the data in the format expected by the plotmacros.sty commands.
                
                **Column Name Mapping:**
                If any configuration names contain special characters (spaces, etc.), they are automatically mapped to VAR names (VAR0, VAR1, etc.) in both the LaTeX code and CSV files.
                
                ### Requirements:
                ```latex
                \\usepackage{plotmacros}
                \\usepackage{pgfplots}
                \\usepackage{tikz}
                ```
                """)
    
    # Save session state
    save_session_state_to_config()


if __name__ == "__main__":
    main()
