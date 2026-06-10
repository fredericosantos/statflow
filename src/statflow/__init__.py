"""
Statflow: Multi-page Streamlit application for MLflow experiment analysis and visualization.

This package provides a modular, multi-page Streamlit application for analyzing
experiment results from MLflow. The application is organized into logical pages
for different analysis workflows, with a focus on parameter exploration, metrics
comparison, and dataset analysis.

statflow/
├── __init__.py          # Package initialization and main structure
├── app.py               # Main navigation launcher with page routing
├── config.py            # Constants, configuration management, and YAML persistence
├── .streamlit/          # Streamlit configuration (fonts, themes)
├── subpages/            # Individual analysis pages
│   ├── get_started.py   # Main entry point with experiment/dataset setup
│   ├── parameters.py    # Parameter exploration and filtering
│   ├── metrics.py       # Metrics overview and selection
│   ├── single_dataset.py # Single dataset analysis (boxplots, Pareto)
│   ├── multiple_datasets.py # Multiple datasets comparison tables
│   ├── export_data.py   # Bulk export functionality
│   ├── plot_macros.py   # Advanced plotting and visualization
│   └── settings.py      # Advanced filtering and customization
├── components/         # Reusable UI components
│   ├── __init__.py      # UI components package
│   ├── downloads.py     # Download button and ZIP creation
│   ├── filters.py       # Sidebar filter widgets
│   ├── graphs.py        # Graph rendering and visualization
│   └── tables.py        # Table display with downloads
├── functional/          # Shared functional utilities
│   ├── __init__.py      # Functional package
│   ├── mlflow/          # MLflow client utilities
│   ├── dataframes/      # Data processing (Polars operations)
│   ├── export/          # Data export functions
│   ├── table_builders/  # Table builder classes
│   ├── visualization/   # Visualization helpers
│   └── table_utils/     # Table utility functions
├── pages_modules/       # Page-specific logic modules
│   ├── visualization.py # Color and symbol assignment for plots
│   ├── styling.py       # Table styling and UI utilities
│   └── export.py        # Export functionality (ZIP, CSV, etc.)
└── components/          # Reusable UI components
    ├── __init__.py      # Components package initialization
    ├── filters.py       # Sidebar filter widgets
    ├── graphs.py        # Graph rendering components
    ├── tables.py        # Table display components
    └── downloads.py     # Download button components

Usage:
    uv run streamlit run statflow/app.py
"""
