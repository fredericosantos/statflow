# Statflow - MLflow Experiment Analysis Tool

A comprehensive, modular Streamlit application for analyzing and visualizing MLflow experiment results with advanced parameter exploration, metrics comparison, and dataset analysis capabilities.

## Features

- **🏠 Home**: Overview and navigation to analysis pages
- **🔧 Parameters**: Explore and filter experiment parameters with correlation analysis
- **📊 Metrics**: Analyze metrics distributions and experiment comparisons
- **🔬 Single Dataset Analysis**: Deep dive into individual datasets with boxplots and Pareto fronts
- **📋 Multiple Datasets Comparison**: Compare configurations across datasets with statistical tests
- **💾 Export**: Bulk export tools for raw data, tables, and statistical analysis
- **⚙️ Settings**: Customize colors, symbols, and graph settings (persisted to YAML)

## Quick Start

```bash
# Navigate to the project directory
cd /path/to/statflow

# Run the multi-page application
uv run streamlit run src/statflow/Home.py
```

## Project Structure

```
statflow/
├── __init__.py              # Package initialization
├── config.py                # Configuration management and YAML persistence
├── Home.py                  # Main entry point with navigation
├── pages/                   # Individual analysis pages
│   ├── 1_🔧_Parameters.py       # Parameter exploration and filtering
│   ├── 2_📊_Metrics.py          # Metrics analysis and comparison
│   ├── 3_🔬_Single_Dataset.py   # Single dataset analysis
│   ├── 4_📋_Multiple_Datasets.py # Multiple datasets comparison
│   ├── 5_💾_Export.py           # Data export tools
│   └── 6_⚙️_Settings.py         # Settings and customization
├── pages_modules/           # Business logic modules
│   ├── module_1_Parameters/     # Parameter processing
│   ├── module_2_Metrics/        # Metrics processing
│   ├── module_3_Single_Dataset/ # Single dataset processing
│   ├── module_4_Multiple_Datasets/ # Multiple datasets processing
│   ├── module_5_Export/         # Export processing
│   └── module_6_Settings/       # Settings processing
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── mlflow_client.py     # MLflow data fetching
│   ├── data_processing.py   # Data transformation and labeling
│   ├── table_builders/      # Table construction modules
│   ├── table_utils.py       # Shared table utilities
│   ├── visualization.py     # Colors and symbols
│   ├── styling.py           # Table styling and UI utilities
│   └── export.py            # Export utilities
└── components/              # Reusable UI components
    ├── __init__.py
    ├── filters.py           # Filter widgets
    ├── graphs.py            # Graph rendering
    ├── tables.py            # Table display components
    └── downloads.py         # Download buttons
```

## Configuration

User preferences (colors, symbols, graph settings) are automatically saved to `.statflow_config.yaml`.

### Default Settings

- **Colors**: Custom palette for different experiment variants
- **Symbols**: Distinct markers for different configurations
- **Graph Size**: 800x600 pixels
- **Display**: Median statistics, error bars enabled

## Data Sources

- **MLflow Tracking URI**: Configurable (default: http://0.0.0.0:5000)
- **Datasets**: Dynamic loading from MLflow experiments
- **Parameters**: Automatic parameter extraction and filtering
- **Metrics**: Comprehensive metrics analysis and comparison

## Export Formats

- **CSV**: Raw data and comparison tables
- **LaTeX**: Publication-ready tables with customizable formatting
- **Markdown**: Documentation-friendly tables
- **ZIP**: Bulk raw data archives

## Requirements

- Python 3.13+
- Streamlit >=1.53.0
- MLflow >=3.8.1
- Polars >=1.37.1
- Plotly >=6.5.2
- NumPy, SciPy, Statsmodels

## Usage Notes

1. **Navigation**: Use the Home page to navigate between analysis modes
2. **Parameter Exploration**: Start with Parameters page to understand experiment configurations
3. **Metrics Analysis**: Use Metrics page to compare performance across experiments
4. **Dataset Analysis**: Dive deep into individual or multiple datasets
5. **Caching**: Data fetching is cached for performance - use refresh buttons if needed
6. **Filters**: Apply filters in sidebars to focus analysis on specific configurations
7. **Configuration**: Customize appearance in Settings - preferences persist between sessions
8. **Export**: Use the Export page for bulk downloads in multiple formats

## Development

The application follows a modular architecture with:
- Separation of concerns (data, visualization, UI)
- Reusable components
- Comprehensive caching for performance
- YAML-based configuration persistence
- Type hints and documentation throughout
- Polars DataFrames for efficient data processing