"""
Table builders package.

This package contains utilities for building various comparison tables.

table_builders/
├── __init__.py          # Exposes table builder functions.
├── comparison_table_builder.py  # Cross-dataset comparison tables.
├── rmse_table_builder.py        # RMSE tables with significance.
├── statistical_table_builder.py # Statistical significance tables.
└── nodes_table_builder.py       # Tree size comparison tables.
"""

from .comparison_table_builder import build_comparison_table
from .rmse_table_builder import build_rmse_table
from .statistical_table_builder import build_statistical_table
from .nodes_table_builder import build_nodes_table

__all__ = [
    "build_comparison_table",
    "build_rmse_table",
    "build_statistical_table",
    "build_nodes_table",
]