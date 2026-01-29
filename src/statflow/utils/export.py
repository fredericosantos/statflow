"""
Export utilities for downloading data and results.

This module provides functions for creating downloadable files including
ZIP archives of raw data, CSV exports, and other formats.

export.py
├── ZIP file creation (create_fitness_zip)
├── CSV export utilities
├── LaTeX table export
├── Markdown table export
└── File generation helpers

Usage:
    from statflow.utils.export import (
        create_fitness_zip
    )
"""

import io
import zipfile
import pandas as pd
from typing import Optional, Any, Tuple
import streamlit as st

from statflow.config import ALL_DATASETS
from statflow.utils.mlflow_client import fetch_all_datasets_parallel
from statflow.utils.data_processing import create_group_label, get_sort_key


def create_fitness_zip(
    selected_mpf_values: Optional[Tuple[str, ...]] = None,
    selected_beta_values: Optional[Tuple[str, ...]] = None,
    selected_pinflate_values: Optional[Tuple[str, ...]] = None,
    progress_bar: Optional[Any] = None,
) -> Optional[bytes]:
    """Create a ZIP file containing raw fitness CSV for all datasets.

    Args:
        selected_mpf_values: Selected MPF values to filter.
        selected_beta_values: Selected beta values to filter.
        selected_pinflate_values: Selected p_inflate values to filter.
        progress_bar: Optional Streamlit progress bar for updates.

    Returns:
        ZIP file as bytes, or None if no data.
    """
    # Fetch all datasets with filters
    all_runs_df = fetch_all_datasets_parallel(
        selected_mpf_values, selected_beta_values, selected_pinflate_values
    )

    if progress_bar:
        progress_bar.progress(0.1, text="Fetched data, processing datasets...")

    if all_runs_df.empty:
        return None

    # Add grouping labels and sort keys for global config order
    all_runs_df = all_runs_df.copy()
    all_runs_df["config_group"] = all_runs_df.apply(create_group_label, axis=1)
    all_runs_df["sort_key"] = all_runs_df.apply(get_sort_key, axis=1)

    # Get global config order
    config_order = (
        all_runs_df.groupby("config_group")["sort_key"]
        .first()
        .sort_values()
        .index.tolist()
    )

    # Create zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, dataset in enumerate(ALL_DATASETS):
            dataset_runs = all_runs_df[all_runs_df["dataset_name"] == dataset]

            if not dataset_runs.empty:
                # Create CSV content for this dataset
                csv_content = create_dataset_csv(dataset_runs, config_order)
                zip_file.writestr(f"{dataset}_fitness.csv", csv_content)

            if progress_bar:
                progress = 0.1 + (0.9 * (i + 1) / len(ALL_DATASETS))
                progress_bar.progress(progress, text=f"Processing {dataset}...")

    zip_buffer.seek(0)
    if progress_bar:
        progress_bar.progress(1.0, text="ZIP file ready!")
    return zip_buffer.getvalue()


def create_dataset_csv(dataset_runs: pd.DataFrame, config_order: list[str]) -> str:
    """Create CSV content for a single dataset.

    Args:
        dataset_runs: DataFrame with runs for this dataset.
        config_order: Ordered list of configuration names.

    Returns:
        CSV content as string.
    """
    # Prepare data for CSV
    csv_data = []

    for config in config_order:
        config_runs = dataset_runs[dataset_runs["config_group"] == config]

        if not config_runs.empty:
            for _, run in config_runs.iterrows():
                row = {
                    "config": config,
                    "run_id": run.get("run_id", ""),
                    "rmse": run.get("metrics.best_test_fitness", ""),
                    "tree_size": run.get("metrics.best_n_nodes", ""),
                    "variant": run.get("params.variant", ""),
                    "arc_beta": run.get("params.arc_beta", ""),
                    "mutation_pool_factor": run.get("params.mutation_pool_factor", ""),
                    "use_oms": run.get("params.use_oms", ""),
                    "new_oms": run.get("params.new_oms", ""),
                    "pool_refresh_interval": run.get("params.pool_refresh_interval", ""),
                }
                csv_data.append(row)

    if not csv_data:
        return "No data available"

    # Create DataFrame and convert to CSV
    csv_df = pd.DataFrame(csv_data)
    return csv_df.to_csv(index=False)


def export_table_to_csv(df: pd.DataFrame, filename: str = "table.csv") -> str:
    """Export a styled DataFrame to CSV format.

    Args:
        df: DataFrame to export.
        filename: Suggested filename.

    Returns:
        CSV content as string.
    """
    return df.to_csv(index=True)


def export_table_to_latex(
    df: pd.DataFrame, 
    caption: str = "", 
    label: str = "",
    rotation_angle: int = 90,
    font_size: str = "normalsize",
    dataset_renames: dict[str, str] | None = None
) -> str:
    """Export DataFrame to LaTeX table format.

    Automatically detects table type and applies appropriate formatting:
    - Detailed per-dataset tables (multi-method statistics)
    - RMSE comparison tables (datasets × configs)
    - Statistical significance tables (with H-B correction)
    - Generic tables (fallback)

    Args:
        df: DataFrame to export.
        caption: Table caption.
        label: Table label.
        rotation_angle: Rotation angle for dataset names (default 90).
        font_size: LaTeX font size command (default "normalsize").
        dataset_renames: Optional dict mapping original dataset names to display names.

    Returns:
        LaTeX table code as string.
    """
    if dataset_renames is None:
        dataset_renames = {}
    
    # Detect table type based on columns
    is_detailed_table = (
        "Median Fitness" in df.columns and 
        "P-value vs Best" in df.columns
    )
    
    is_rmse_table = (
        "Dataset" in df.columns and
        "Samples" in df.columns and
        "Target Mean" in df.columns and
        "Median Fitness" not in df.columns
    )
    
    is_stat_significance_table = (
        "Dataset" in df.columns and
        "Best" in df.columns and
        "Best Config" in df.columns and
        "Sig. Superior" in df.columns
    )
    
    # Check if multi-index (All datasets view)
    is_multi_index = isinstance(df.index, pd.MultiIndex)
    
    if is_detailed_table:
        return _export_detailed_table_latex(df, caption, label, rotation_angle, font_size, is_multi_index, dataset_renames)
    elif is_rmse_table:
        return _export_rmse_table_latex(df, caption, label, rotation_angle, font_size, dataset_renames)
    elif is_stat_significance_table:
        return _export_stat_significance_latex(df, caption, label, rotation_angle, font_size, dataset_renames)
    else:
        return _export_generic_table_latex(df, caption, label)


def _get_display_name(dataset: str, renames: dict[str, str]) -> str:
    """Get display name for a dataset, applying renames and escaping for LaTeX."""
    display_name = renames.get(dataset, dataset)
    # Escape underscores for LaTeX
    return display_name.replace("_", r"\_")


def _export_rmse_table_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    rotation_angle: int,
    font_size: str,
    dataset_renames: dict[str, str]
) -> str:
    """Export RMSE comparison table to LaTeX format."""
    # Get config columns (everything after Target Std)
    meta_cols = ["Dataset", "Samples", "Features", "Train Std (Scaled)", 
                 "Test Std (Scaled)", "Target Mean", "Target Median", "Target Std"]
    config_cols = [c for c in df.columns if c not in meta_cols]
    
    latex_lines = [
        "\\begin{table*}[t]",
        f"\\{font_size}",
        "\\centering",
    ]
    
    # Scientific caption with formatting explanation
    full_caption = caption
    if caption:
        full_caption += ". "
    full_caption += "Values shown as median $\\pm$ standard deviation across 30 independent runs. "
    full_caption += "\\textbf{Bold} indicates the best-performing configuration per dataset. "
    full_caption += "$\\dagger$ denotes statistically significant superiority over all other configurations "
    full_caption += "(one-tailed Wilcoxon rank-sum test with Holm-Bonferroni correction, $\\alpha = 0.05$)."
    
    latex_lines.append(f"\\caption{{{full_caption}}}")
    if label:
        latex_lines.append(f"\\label{{{label}}}")
    
    # Column spec: Dataset rotated + config columns
    num_config_cols = len(config_cols)
    col_spec = "c" + "c" * num_config_cols
    
    latex_lines.extend([
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ])
    
    # Header: Dataset + config names
    header_parts = ["{Dataset}"]
    for config in config_cols:
        config_formatted = _format_config_header_latex(config)
        header_parts.append(f"{{{config_formatted}}}")
    latex_lines.append(" & ".join(header_parts) + " \\\\")
    latex_lines.append("\\midrule")
    
    # Find best per row (lowest median)
    for _, row in df.iterrows():
        original_dataset = str(row["Dataset"])
        dataset = _get_display_name(original_dataset, dataset_renames)
        
        # Parse values to find best
        config_values = {}
        for config in config_cols:
            val = str(row[config])
            if val != "—":
                # Extract median from "X.XX ± Y.YY" or "X.XX ± Y.YY †"
                median_str = val.split("±")[0].strip()
                if "†" in median_str:
                    median_str = median_str.replace("†", "").strip()
                config_values[config] = float(median_str)
        
        best_config = min(config_values.keys(), key=lambda x: config_values[x]) if config_values else None
        
        row_parts = [dataset]
        for config in config_cols:
            val = str(row[config])
            if val == "—":
                row_parts.append("{---}")
            elif config == best_config:
                # Bold the best value
                row_parts.append(f"\\textbf{{{val}}}")
            else:
                row_parts.append(val)
        
        latex_lines.append(" & ".join(row_parts) + " \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    
    return "\n".join(latex_lines)


def _export_stat_significance_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    rotation_angle: int,
    font_size: str,
    dataset_renames: dict[str, str]
) -> str:
    """Export Statistical Significance table to LaTeX format."""
    # Key columns to include
    key_cols = ["Dataset", "Samples", "Features", "Best", "Best Config", 
                "2nd Best", "Best Tree Size", "2nd Best Tree Size", "H-B Sig.", "Sig. Superior"]
    
    # Get config columns (p-values)
    config_cols = [c for c in df.columns if c not in key_cols]
    
    latex_lines = [
        "\\begin{table*}[t]",
        f"\\{font_size}",
        "\\centering",
    ]
    
    # Scientific caption with formatting explanation
    full_caption = caption
    if caption:
        full_caption += ". "
    full_caption += "P-values from one-tailed Wilcoxon rank-sum tests comparing the best configuration to each alternative. "
    full_caption += "\\textbf{Bold} p-values indicate statistical significance ($p < 0.05$). "
    full_caption += "H-B: $\\checkmark$ if the best configuration is significantly better than \\emph{all} configurations after Holm-Bonferroni correction. "
    full_caption += "Sig.~Sup.: $\\checkmark$ if the best configuration is significantly better than all configurations of \\emph{other} variants ($\\alpha = 0.05$)."
    
    latex_lines.append(f"\\caption{{{full_caption}}}")
    if label:
        latex_lines.append(f"\\label{{{label}}}")
    
    # Simplified column spec: Dataset, Best, Best Config, H-B Sig, Sig. Superior, then p-values
    display_cols = ["Dataset", "Best", "Best Config", "H-B Sig.", "Sig. Superior"] + config_cols
    col_spec = "l l l c c" + " c" * len(config_cols)
    
    latex_lines.extend([
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ])
    
    # Header
    header_parts = ["{Dataset}", "{Best}", "{Best Config}", "{H-B}", "{Sig. Sup.}"]
    for config in config_cols:
        config_formatted = _format_config_header_latex(config)
        header_parts.append(f"{{{config_formatted}}}")
    latex_lines.append(" & ".join(header_parts) + " \\\\")
    latex_lines.append("\\midrule")
    
    # Data rows
    for _, row in df.iterrows():
        original_dataset = str(row["Dataset"])
        dataset = _get_display_name(original_dataset, dataset_renames)
        best = str(row["Best"])
        best_config = _format_config_header_latex(str(row["Best Config"]))
        hb_sig = "\\checkmark" if row["H-B Sig."] else ""
        sig_sup = "\\checkmark" if row["Sig. Superior"] else ""
        
        row_parts = [
            dataset,
            best,
            best_config,
            hb_sig,
            sig_sup
        ]
        
        for config in config_cols:
            val = str(row[config])
            if val == "—":
                row_parts.append("{---}")
            elif val == "N/A":
                row_parts.append("N/A")
            else:
                # Format p-value, bold if < 0.05
                p_val = float(val)
                if p_val < 0.05:
                    row_parts.append(f"\\textbf{{{val}}}")
                else:
                    row_parts.append(val)
        
        latex_lines.append(" & ".join(row_parts) + " \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    
    return "\n".join(latex_lines)


def _format_config_header_latex(config: str) -> str:
    """Format configuration name for LaTeX header."""
    # Handle ARC with beta parameter
    if "ARC (β = " in config or "ARC (beta = " in config.lower():
        beta_val = config.split("=")[1].strip().rstrip(")")
        return f"ARC $(\\beta = {beta_val})$"
    
    # Handle underscores and special chars
    config = config.replace("_", r"\_")
    
    return config


def _export_detailed_table_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    rotation_angle: int,
    font_size: str,
    is_multi_index: bool,
    dataset_renames: dict[str, str]
) -> str:
    """Export detailed per-dataset table to LaTeX format."""
    latex_lines = [
        "\\begin{table*}[t]",
        f"\\{font_size}",
        "\\centering",
    ]
    
    if caption:
        latex_lines.append(f"\\caption{{{caption}}}")
    if label:
        latex_lines.append(f"\\label{{{label}}}")
    
    # Column specification for detailed table
    if is_multi_index:
        col_spec = "c l S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=4.1] S[table-format=4.1] S[table-format=4.1] S[table-format=1.4]"
    else:
        col_spec = "l S[table-format=1.4] S[table-format=1.4] S[table-format=1.4] S[table-format=4.1] S[table-format=4.1] S[table-format=4.1] S[table-format=1.4]"
    
    latex_lines.extend([
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ])
    
    # Header
    if is_multi_index:
        header = "{Dataset} & {Method} & {Med Fit} & {Mean Fit} & {Std Fit} & {Med Size} & {Mean Size} & {Std Size} & {P-val} \\\\"
    else:
        header = "{Method} & {Med Fit} & {Mean Fit} & {Std Fit} & {Med Size} & {Mean Size} & {Std Size} & {P-val} \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Data rows
    if is_multi_index:
        grouped = df.groupby(level=0)
        first_group = True
        
        for dataset, group in grouped:
            if not first_group:
                latex_lines.append("\\midrule")
            first_group = False
            
            num_rows = len(group)
            
            for i, (idx, row) in enumerate(group.iterrows()):
                method = idx[1]
                method_formatted = format_method_name_latex(method)
                
                is_best = str(row["P-value vs Best"]) == "—"
                
                values = []
                for col in df.columns:
                    val = str(row[col])
                    if val == "—":
                        values.append("{---}")
                    elif is_best:
                        values.append(f"\\textbf{{{val}}}")
                    else:
                        values.append(val)
                
                if is_best:
                    method_formatted = f"\\textbf{{{method_formatted}}}"
                
                if i == 0:
                    dataset_display = _get_display_name(str(dataset), dataset_renames)
                    dataset_cell = f"\\multirow{{{num_rows}}}{{*}}{{\\rotatebox[origin=c]{{{rotation_angle}}}{{{dataset_display}}}}}"
                    row_str = f"{dataset_cell} & {method_formatted} & {' & '.join(values)} \\\\"
                else:
                    row_str = f"& {method_formatted} & {' & '.join(values)} \\\\"
                
                latex_lines.append(row_str)
    else:
        for idx, row in df.iterrows():
            method = str(idx)
            method_formatted = format_method_name_latex(method)
            
            is_best = str(row["P-value vs Best"]) == "—"
            
            values = []
            for col in df.columns:
                val = str(row[col])
                if val == "—":
                    values.append("{---}")
                elif is_best:
                    values.append(f"\\textbf{{{val}}}")
                else:
                    values.append(val)
            
            if is_best:
                method_formatted = f"\\textbf{{{method_formatted}}}"
            
            row_str = f"{method_formatted} & {' & '.join(values)} \\\\"
            latex_lines.append(row_str)
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    
    return "\n".join(latex_lines)


def _export_generic_table_latex(
    df: pd.DataFrame,
    caption: str,
    label: str
) -> str:
    """Export generic table to LaTeX format (fallback)."""
    index_names = df.index.names if isinstance(df.index, pd.MultiIndex) else [df.index.name]
    num_index_cols = len([n for n in index_names if n is not None]) if any(n is not None for n in index_names) else 1
    num_cols = num_index_cols + len(df.columns)
    col_spec = "l" * num_cols

    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{|{col_spec}|}}",
        "\\hline",
    ]

    # Header with index names
    if isinstance(df.index, pd.MultiIndex):
        index_headers = [str(name) if name else "" for name in df.index.names]
    else:
        index_headers = [str(df.index.name) if df.index.name else ""]
    header = " & ".join(index_headers + list(df.columns))
    latex_lines.append(f"{header} \\\\")
    latex_lines.append("\\hline")

    # Data rows with index values
    for idx, row in df.iterrows():
        if isinstance(idx, tuple):
            index_vals = [str(v) for v in idx]
        else:
            index_vals = [str(idx)]
        row_str = " & ".join(index_vals + [str(val) for val in row])
        latex_lines.append(f"{row_str} \\\\")

    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
    ])

    if caption:
        latex_lines.append(f"\\caption{{{caption}}}")

    if label:
        latex_lines.append(f"\\label{{{label}}}")

    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)


def format_method_name_latex(method: str) -> str:
    """Format method name for LaTeX output.
    
    Args:
        method: Method name string.
        
    Returns:
        LaTeX-formatted method name.
    """
    # Handle ARC with beta parameter
    if method.startswith("ARC (β = "):
        beta_val = method.replace("ARC (β = ", "").replace(")", "")
        return f"ARC $(\\beta = {beta_val})$"
    
    # Handle other specific cases
    if method == "GSGP-OMS":
        return "GSGP-OMS"
    elif method == "GSGP-std":
        return "GSGP-std"
    elif method == "SLIM":
        return "SLIM"
    
    # Default: return as-is
    return method


def export_table_to_markdown(df: pd.DataFrame) -> str:
    """Export DataFrame to Markdown table format.

    Args:
        df: DataFrame to export.

    Returns:
        Markdown table as string.
    """
    # Create header with index names
    if isinstance(df.index, pd.MultiIndex):
        index_headers = [str(name) if name else "" for name in df.index.names]
    else:
        index_headers = [str(df.index.name) if df.index.name else ""]
    
    all_columns = index_headers + list(df.columns)
    header = "| " + " | ".join(all_columns) + " |"
    separator = "|" + "|".join(["---"] * len(all_columns)) + "|"

    # Create rows with index values
    rows = []
    for idx, row in df.iterrows():
        if isinstance(idx, tuple):
            index_vals = [str(v) for v in idx]
        else:
            index_vals = [str(idx)]
        row_str = "| " + " | ".join(index_vals + [str(val) for val in row]) + " |"
        rows.append(row_str)

    return "\n".join([header, separator] + rows)