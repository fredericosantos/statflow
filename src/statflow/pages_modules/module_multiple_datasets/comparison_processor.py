"""
Comparison processing logic for multiple datasets analysis.

This module handles comparison calculations and statistical analysis across datasets.

comparison_processor.py
├── prepare_comparison_tables()  # Prepares tables for comparison display.
└── Comparison calculations and statistical analysis
"""

def prepare_comparison_tables(all_runs_df: pl.DataFrame) -> tuple[pl.DataFrame, dict, pl.DataFrame]:
    """Prepare tables for comparison display.

    Args:
        all_runs_df: DataFrame with all runs.

    Returns:
        Tuple of (rmse_df, significance_info, nodes_df).
    """
    from statflow.functional.table_builders.rmse_table_builder import build_rmse_table
    from statflow.functional.table_builders.nodes_table_builder import build_nodes_table

    rmse_df, significance_info = build_rmse_table(all_runs_df)
    nodes_df = build_nodes_table(all_runs_df)

    return rmse_df, significance_info, nodes_df