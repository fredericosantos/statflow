"""
Filter processing logic for single dataset analysis.

This module handles extraction and processing of filter values from runs data.

filter_processor.py
├── extract_filter_values()  # Extracts available values for filters from runs data.
└── Filter value extraction and processing logic
"""

import polars as pl


def extract_filter_values(runs_df: pl.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Extract available values for filters from runs data.

    Args:
        runs_df: DataFrame with runs data.

    Returns:
        Tuple of (available_mpf, available_beta, available_pinflate).
    """
    available_mpf = sorted(runs_df.select(pl.col("params.mutation_pool_factor").drop_nulls().unique().cast(pl.Utf8)).to_series().to_list())
    available_beta = sorted(runs_df.select(pl.col("params.arc_beta").drop_nulls().unique().cast(pl.Utf8)).to_series().to_list())
    available_pinflate = sorted(runs_df.filter(pl.col("params.variant") == "slim_gsgp").select(pl.col("params.arc_beta").drop_nulls().unique().cast(pl.Utf8)).to_series().to_list())

    return available_mpf, available_beta, available_pinflate