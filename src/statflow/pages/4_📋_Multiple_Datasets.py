"""
Multiple Datasets Comparison page.

This page provides comparison tables across multiple datasets including
RMSE comparisons, statistical significance tests, and tree size analysis.

2_📋_Cross_Dataset.py
├── RMSE comparison table
├── Statistical significance analysis
├── Tree size comparison table
├── Bulk download options
└── Filter controls

Usage:
    Accessed via navigation from Home.py
"""

import streamlit as st
import pandas as pd

from statflow.config import (
    initialize_session_state, save_session_state_to_config, setup_sidebar
)
from statflow.utils.table_builders.rmse_table_builder import build_rmse_table
from statflow.utils.table_builders.statistical_table_builder import build_statistical_table
from statflow.utils.table_builders.nodes_table_builder import build_nodes_table
from statflow.utils.table_builders import (
    build_dominance_count_table, build_per_dataset_detailed_table, build_all_datasets_detailed_table
)
from statflow.utils.styling import style_rmse_table
from statflow.components.filters import render_dataset_selector, render_global_filters
from statflow.components.tables import render_table_with_downloads
from statflow.pages_modules.module_4_Multiple_Datasets.processor import (
    fetch_and_process_multiple_datasets, prepare_comparison_tables
)


st.set_page_config(
    page_title=f"Multiple Datasets Comparison - {st.session_state.get('app_name', 'Experiment Analysis')}",
    page_icon="📋",
    layout="wide",
)

# Initialize session state
initialize_session_state()

# Setup sidebar
setup_sidebar()


def main():
    # Title and summary statistics in columns
    col_title, col_datasets, col_runs, col_configs = st.columns([3, 1, 1, 1])

    with col_title:
        st.title("📋 Multiple Datasets Comparison")
        st.markdown("Compare configurations across multiple datasets")

    # Navigation
    if st.button("← Back to Home", key="back_home"):
        st.switch_page("Home.py")

    st.markdown("---")

    # Sidebar for global filters
    with st.sidebar:
        # Dataset selection
        st.markdown("### Dataset Selection")
        selected_datasets = render_dataset_selector(selection_mode="multi")
        
        # Dataset names customization
        from statflow.components.filters import render_dataset_names_expander
        render_dataset_names_expander()
        
        # Get selected metrics from session state
        selected_metrics = st.session_state.get('selected_metrics', [])
        
        st.markdown("---")

        # Global filters
        available_mpf = ["1", "2", "5", "10"]
        available_beta = ["0.0", "0.25", "0.5", "0.7", "0.9", "0.95", "1.0"]
        available_pinflate = ["0.1", "0.3", "0.5", "0.7", "0.9"]
        
        _, selected_mpf_values, selected_beta_values, selected_pinflate_values = render_global_filters(
            available_mpf, available_beta, available_pinflate
        )

    # Convert to tuples for caching
    selected_mpf_tuple = tuple(selected_mpf_values) if selected_mpf_values else None
    selected_beta_tuple = tuple(selected_beta_values) if selected_beta_values else None
    selected_pinflate_tuple = tuple(selected_pinflate_values) if selected_pinflate_values else None

    # Update session state
    st.session_state.selected_mpf_values = selected_mpf_tuple
    st.session_state.selected_beta_values = selected_beta_tuple
    st.session_state.selected_pinflate_values = selected_pinflate_tuple
    st.session_state.selected_datasets = selected_datasets


    # Check if datasets are selected
    if not selected_datasets:
        st.info("Please select at least one dataset to compare.")
        return

    # Fetch selected datasets with filters
    with st.spinner(f"Fetching data for {len(selected_datasets)} dataset{'s' if len(selected_datasets) > 1 else ''}..."):
        all_runs_df = fetch_and_process_multiple_datasets(

            selected_mpf_tuple, selected_beta_tuple, selected_pinflate_tuple, selected_datasets, selected_metrics

        )

        if all_runs_df is None:
            st.error("No data found with the selected filters.")
            return
    with col_datasets:
        total_datasets = len(all_runs_df["dataset_name"].unique())
        st.metric("Datasets", total_datasets)

    with col_runs:
        total_runs = len(all_runs_df)
        st.metric("Total Runs", f"{total_runs:,}")

    with col_configs:
        total_configs = len(all_runs_df.groupby(["params.variant", "params.arc_beta", "params.mutation_pool_factor", "params.use_oms"]).size())
        st.metric("Configurations", total_configs)

    # Create tabs for different comparison views
    tab_rmse, tab_significance, tab_nodes, tab_detailed, tab_names = st.tabs([
        "📊 RMSE Comparison",
        "🎯 Statistical Significance",
        "🌳 Tree Size Comparison",
        "📈 Detailed Per-Dataset Analysis",
        "⚙️ Dataset Names"
    ])

    with tab_rmse:
        with st.spinner("Building RMSE comparison table..."):
            rmse_df, significance_info = build_rmse_table(all_runs_df)

            if not rmse_df.empty:
                # Apply styling
                styled_rmse_df = style_rmse_table(rmse_df, significance_info)
                description_rmse = """Median ± Standard Deviation for each configuration. 
                **Samples/Features** columns show dataset dimensions. 
                **Train/Test Std (Scaled)** columns show the average standard deviation after StandardScaler (70% train/30% test split, seeds 1-30). 
                **Target Mean/Median/Std** columns show the original target distribution statistics (before scaling) to assess how centered around 0 the targets are. 
                **Legend:** Bold = Best performer, † = ARC config significantly better than all selected non-ARC configs (one-tailed Wilcoxon rank-sum with Holm-Bonferroni correction, α=0.05)"""
                
                render_table_with_downloads(
                    styled_rmse_df,
                    "RMSE Comparison Across Multiple Datasets",
                    "rmse_comparison",
                    description=description_rmse
                )
            else:
                st.warning("No RMSE data available for comparison.")

    with tab_significance:
        with st.spinner("Building statistical significance table..."):
            datasets_path = st.session_state.get('datasets_path', 'datasets')
            stat_df = build_statistical_table(all_runs_df, datasets_path)

            dominance_df = build_dominance_count_table(stat_df)

            if not dominance_df.empty:
                render_table_with_downloads(
                    dominance_df,
                    "Variant Superiority Count",
                    "superiority_count",
                    description="""
                    Count of datasets where each variant is the best performer and 
                    statistically significantly better than ALL other variants (one-tailed Wilcoxon rank-sum
                    with Holm-Bonferroni correction, α=0.05).
                    
                    This shows how many times each variant achieves statistical superiority.
                    """
                )
            else:
                st.warning("No superiority data available for analysis.")

            st.markdown("---")
            if not stat_df.empty:
                # Add toggles to filter by significance
                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    show_only_hb_sig = st.checkbox(
                        "Show only H-B Significant datasets",
                        value=False,
                        help="Filter to show only datasets where the best configuration is statistically significantly better than ALL configurations (after Holm-Bonferroni correction)"
                    )
                with col_filter2:
                    show_only_superior = st.checkbox(
                        "Show only Sig. Superior datasets",
                        value=False,
                        help="Filter to show only datasets where the best configuration is statistically significantly better than all configurations of OTHER variants (after Holm-Bonferroni correction)"
                    )
                
                # Filter table based on toggles
                display_df = stat_df.copy()
                if show_only_hb_sig:
                    display_df = display_df[display_df["H-B Sig."]]
                if show_only_superior:
                    display_df = display_df[display_df["Sig. Superior"]]
                
                # Display table without styling (remove colors)
                render_table_with_downloads(
                    display_df,  # Use filtered or full table
                    "Statistical Significance Analysis",
                    "statistical_significance",
                    description="""
                    P-values comparing the best method to each configuration (one-tailed Wilcoxon rank-sum, α=0.05).
                    
                    **Legend:**
                    - Samples : Number of data points in the dataset
                    - Features : Number of input features (excluding target)
                    - Best : Variant type of the best performing configuration
                    - Best Config : Full name of the best performing configuration
                    - 2nd Best : Variant type of the second best configuration
                    - Best Tree Size : Median ± std tree size for best config
                    - 2nd Best Tree Size : Median ± std tree size for 2nd best config
                    - H-B Sig. : True if best is significantly better than ALL configs (Holm-Bonferroni corrected)
                    - Sig. Superior : True if best is significantly better than all OTHER variants (H-B corrected)
                    - P-values are raw (before correction) - lower is more significant
                    - N/A : Not available (insufficient data)
                    - — : Best performing configuration
                    """
                )
            else:
                st.warning("No statistical data available for analysis.")

        # Add dominance count table

    with tab_nodes:
        with st.spinner("Building tree size comparison table..."):
            nodes_df = build_nodes_table(all_runs_df)

            if not nodes_df.empty:
                render_table_with_downloads(
                    nodes_df,
                    "Tree Size Comparison Across Multiple Datasets",
                    "tree_size_comparison",
                    description="Median ± Standard Deviation for tree sizes"
                )
            else:
                st.warning("No tree size data available for comparison.")

    with tab_detailed:
        st.markdown("### Detailed Statistics by Dataset")
        st.markdown("Select a dataset to view comprehensive statistics for all configurations.")

        # LaTeX customization expander
        with st.expander("⚙️ LaTeX Export Customization", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                rotation_angle = st.slider(
                    "Dataset Name Rotation Angle",
                    min_value=-90,
                    max_value=90,
                    value=90,
                    step=5,
                    help="Angle for rotating dataset names in multi-dataset tables (0 = horizontal, 90 = vertical)"
                )
            with col2:
                font_size = st.selectbox(
                    "Font Size",
                    options=["tiny", "scriptsize", "footnotesize", "small", "normalsize", "large", "Large", "LARGE", "huge", "Huge"],
                    index=4,  # normalsize
                    help="LaTeX font size for the table content"
                )
            
            # Store in session state
            st.session_state.latex_rotation_angle = rotation_angle
            st.session_state.latex_font_size = font_size

        # Get list of available datasets from the data
        available_datasets = sorted(all_runs_df["dataset_name"].unique().tolist())

        if not available_datasets:
            st.warning("No datasets available.")
        else:
            # Separate datasets into categories
            blackbox_datasets = sorted([d for d in available_datasets if d.startswith("blackbox_")])
            real_life_datasets = sorted([d for d in available_datasets if not d.startswith("blackbox_")])
            
            # Use st.pills for dataset selection with category options
            pill_options = ["🌐 All", "🔬 Real-Life", "⬛ Blackbox"] + available_datasets
            selected_dataset_pill = st.pills(
                "Select Dataset",
                options=pill_options,
                selection_mode="single",
                label_visibility="collapsed"
            )

            if selected_dataset_pill:
                if selected_dataset_pill in ["🌐 All", "🔬 Real-Life", "⬛ Blackbox"]:
                    # Determine which datasets to include
                    if selected_dataset_pill == "🌐 All":
                        datasets_to_include = available_datasets
                        title_suffix = "All Datasets"
                        filename_suffix = "all_datasets"
                    elif selected_dataset_pill == "🔬 Real-Life":
                        datasets_to_include = real_life_datasets
                        title_suffix = "Real-Life Datasets"
                        filename_suffix = "real_life_datasets"
                    else:  # ⬛ Blackbox
                        datasets_to_include = blackbox_datasets
                        title_suffix = "Blackbox Datasets"
                        filename_suffix = "blackbox_datasets"
                    
                    # Filter dataframe to selected datasets
                    filtered_df = all_runs_df[all_runs_df["dataset_name"].isin(datasets_to_include)]
                    
                    # Build combined table with multi-index
                    with st.spinner(f"Building detailed table for {title_suffix.lower()}..."):
                        detailed_df = build_all_datasets_detailed_table(filtered_df)

                        if not detailed_df.empty:
                            render_table_with_downloads(
                                detailed_df,
                                f"Detailed Statistics: {title_suffix}",
                                f"detailed_{filename_suffix}",
                                description="""
                                Comprehensive statistics for each configuration across selected datasets.
                                
                                **Index:**
                                - **Dataset**: Name of the dataset
                                - **Method**: Configuration name (ARC-GSGP parameters are treated as separate methods)
                                
                                **Columns:**
                                - **Median Fitness**: Median best fitness at the last generation
                                - **Mean Fitness**: Average best fitness at the last generation
                                - **Std Fitness**: Standard deviation of best fitness at the last generation
                                - **Median Tree Size**: Median size of the best individual at the last generation
                                - **Mean Tree Size**: Average size of the best individual at the last generation
                                - **Std Tree Size**: Standard deviation of tree size at the last generation
                                - **P-value vs Best**: Statistical test (Mann-Whitney U) comparing this method to the best performer by median fitness (per dataset)
                                
                                **Legend:** — = Best performing configuration (no self-comparison)
                                """,
                                latex_rotation_angle=st.session_state.get('latex_rotation_angle', 90),
                                latex_font_size=st.session_state.get('latex_font_size', 'normalsize')
                            )
                        else:
                            st.warning(f"No data available for {title_suffix.lower()}.")
                else:
                    # Build table for single dataset
                    with st.spinner(f"Building detailed table for {selected_dataset_pill}..."):
                        detailed_df = build_per_dataset_detailed_table(all_runs_df, selected_dataset_pill)

                        if not detailed_df.empty:
                            render_table_with_downloads(
                                detailed_df,
                                f"Detailed Statistics: {selected_dataset_pill}",
                                f"detailed_{selected_dataset_pill}",
                                description="""
                                Comprehensive statistics for each configuration on the selected dataset.
                                
                                **Columns:**
                                - **Method**: Configuration name (ARC-GSGP parameters are treated as separate methods)
                                - **Median Fitness**: Median best fitness at the last generation
                                - **Mean Fitness**: Average best fitness at the last generation
                                - **Std Fitness**: Standard deviation of best fitness at the last generation
                                - **Median Tree Size**: Median size of the best individual at the last generation
                                - **Mean Tree Size**: Average size of the best individual at the last generation
                                - **Std Tree Size**: Standard deviation of tree size at the last generation
                                - **P-value vs Best**: Statistical test (Mann-Whitney U) comparing this method to the best performer by median fitness
                                
                                **Legend:** — = Best performing configuration (no self-comparison)
                                """,
                                latex_rotation_angle=st.session_state.get('latex_rotation_angle', 90),
                                latex_font_size=st.session_state.get('latex_font_size', 'normalsize')
                            )
                        else:
                            st.warning(f"No data available for {selected_dataset_pill}.")
            else:
                st.info("Please select a dataset from the pills above to view detailed statistics.")

    with tab_names:
        st.markdown("### LaTeX Dataset Command")
        st.markdown("""
        Generate a `dataset_names.tex` file with a `\\dataset{}` command for consistent naming 
        throughout your paper. Use the sidebar to customize dataset names, then export here.
        """)
        
        from statflow.config import DEFAULT_DATASET_RENAMES
        
        # Get current renames from session state
        current_renames = st.session_state.get('dataset_renames', DEFAULT_DATASET_RENAMES.copy())
        
        # Generate LaTeX code for dataset newcommands
        def generate_dataset_latex(renames: dict[str, str]) -> str:
            lines = [
                "% Dataset naming commands - one command per dataset",
                "% Usage: \\datasetEcho, \\datasetLowbwt, etc.",
                "% Include in preamble: \\input{dataset_names}",
                "",
            ]
            
            # Number to Roman numeral mapping for LaTeX-safe names
            num_to_roman = {'0': 'Zero', '1': 'I', '2': 'II', '3': 'III', '4': 'IV', 
                           '5': 'V', '6': 'VI', '7': 'VII', '8': 'VIII', '9': 'IX', '10': 'X'}
            
            # Generate \newcommand for each dataset using the display name
            seen_names = set()
            for original, display in renames.items():
                import re
                
                # Replace numbers with Roman numerals to make LaTeX-safe
                safe_name = display
                # Replace multi-digit numbers first (like 10)
                for num, roman in sorted(num_to_roman.items(), key=lambda x: -len(x[0])):
                    safe_name = safe_name.replace(num, roman)
                
                # Remove any remaining non-letter characters
                safe_name = re.sub(r'[^a-zA-Z]', '', safe_name)
                
                if not safe_name:
                    # Fallback: try using last part of original name
                    safe_name = re.sub(r'[^a-zA-Z]', '', original.split('_')[-1])
                
                # Capitalize first letter
                safe_name = safe_name[0].upper() + safe_name[1:] if safe_name else "Unknown"
                
                # Handle duplicates by adding suffix
                base_name = safe_name
                counter = 2
                while safe_name in seen_names:
                    safe_name = f"{base_name}{chr(ord('A') + counter - 1)}"
                    counter += 1
                seen_names.add(safe_name)
                
                # Escape special chars for LaTeX display
                display_escaped = display.replace("_", r"\_").replace("-", r"{-}")
                
                lines.append(f"\\newcommand{{\\dataset{safe_name}}}{{{display_escaped}}}")
            
            lines.append("")
            lines.append("% Example usage:")
            lines.append("% \\datasetEcho -> Echo")
            lines.append("% \\datasetLowbwt -> Lowbwt")
            
            return "\n".join(lines)
        
        latex_code = generate_dataset_latex(current_renames)
        
        # Download and copy buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download dataset_names.tex",
                icon=":material/download:",
                data=latex_code,
                file_name="dataset_names.tex",
                mime="text/plain",
                key="download_dataset_names_tex"
            )
        with col2:
            with st.popover("Copy LaTeX Code", icon=":material/content_copy:"):
                st.code(latex_code, language="latex")

    # Save configuration changes
    save_session_state_to_config()


if __name__ == "__main__":
    main()