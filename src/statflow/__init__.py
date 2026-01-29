"""
Statflow: Multi-page Streamlit application for MLflow experiment analysis and visualization.

This package provides a modular, multi-page Streamlit application for analyzing
experiment results from MLflow. The application is organized into logical pages
for different analysis workflows, with a focus on parameter exploration, metrics
comparison, and dataset analysis.

statflow/
├── __init__.py          # Package initialization and main structure
├── config.py            # Constants, configuration management, and YAML persistence
├── Home.py              # Main entry point with navigation overview
├── pages/               # Individual analysis pages
│   ├── 1_🔧_Parameters.py     # Parameter exploration and filtering
│   ├── 2_📊_Metrics.py        # Metrics overview and selection
│   ├── 3_🔬_Single_Dataset.py  # Single dataset analysis (boxplots, Pareto)
│   ├── 4_📋_Multiple_Datasets.py # Multiple datasets comparison tables
│   ├── 5_💾_Export.py          # Bulk export functionality
│   └── 6_⚙️_Settings.py        # Advanced filtering and customization
├── pages_modules/       # Business logic modules for each page
│   ├── __init__.py      # Modules package initialization
│   ├── module_1_Parameters/    # Parameter processing logic
│   ├── module_2_Metrics/       # Metrics processing logic
│   ├── module_3_Single_Dataset/ # Single dataset processing
│   ├── module_4_Multiple_Datasets/ # Multiple datasets processing
│   ├── module_5_Export/        # Export processing
│   └── module_6_Settings/      # Settings processing
├── utils/               # Utility modules for data processing and analysis
│   ├── __init__.py      # Utils package initialization
│   ├── mlflow_client.py # MLflow data fetching and client management
│   ├── data_processing.py # Data transformation and labeling functions
│   ├── table_builders/  # Table construction modules
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
    uv run streamlit run statflow/Home.py
"""