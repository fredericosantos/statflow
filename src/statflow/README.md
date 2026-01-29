# GSGP-ARC Multi-Page Streamlit Application

A comprehensive, modular Streamlit application for analyzing and visualizing GSGP-ARC experiment results from MLflow.

## Features

- **🏠 Home**: Overview and navigation to analysis pages
- **🔬 Single Dataset Analysis**: Deep dive into individual datasets with boxplots and Pareto fronts
- **📋 Cross-Dataset Comparison**: Compare configurations across all datasets with statistical tests
- **⚙️ Configuration Explorer**: Customize colors, symbols, and graph settings (persisted to YAML)
- **💾 Data Export**: Bulk export tools for raw data, tables, and statistical analysis

## Quick Start

```bash
# Navigate to the project directory
cd /path/to/gsgp-arc

# Run the multi-page application
uv run streamlit run src/statflow/Home.py
```

Or run directly with:

```bash
uv run streamlit run src/statflow/Home.py
```

## Project Structure

```
statflow/
├── __init__.py              # Package initialization
├── config.py                # Configuration management and YAML persistence
├── Home.py                  # Main entry point with navigation
├── pages/                   # Individual analysis pages
│   ├── 1_🔬_Single_Dataset.py    # Single dataset analysis
│   ├── 2_📋_Cross_Dataset.py     # Cross-dataset comparisons
│   ├── 3_⚙️_Settings.py     # Settings and customization
│   └── 4_💾_Export.py            # Data export tools
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── mlflow_client.py     # MLflow data fetching
│   ├── data_processing.py   # Data transformation and labeling
│   ├── table_builders.py    # Comparison table construction
│   ├── visualization.py     # Colors and symbols
│   ├── styling.py           # Table styling
│   └── export.py            # Export utilities
└── components/              # Reusable UI components
    ├── __init__.py
    ├── filters.py           # Filter widgets
    ├── graphs.py            # Graph rendering
    └── downloads.py         # Download buttons
```

## Configuration

User preferences (colors, symbols, graph settings) are automatically saved to `.statflow_config.yaml`.

### Default Settings

- **Colors**: Custom palette for GSGP, SLIM, and ARC variants
- **Symbols**: Distinct markers for different beta values
- **Graph Size**: 800x600 pixels
- **Display**: Median statistics, error bars enabled

## Data Sources

- **MLflow Tracking URI**: `http://0.0.0.0:5000`
- **Datasets**: 20 total (8 real-life + 12 blackbox)
- **Variants**: GSGP (standard/OMS), SLIM-GSGP, ARC (multiple beta values)

## Export Formats

- **CSV**: Raw data and comparison tables
- **LaTeX**: Publication-ready tables
- **Markdown**: Documentation-friendly tables
- **ZIP**: Bulk raw fitness data archives

## Requirements

- Python 3.13+
- Streamlit
- MLflow
- Pandas, NumPy, SciPy
- Plotly

## Usage Notes

1. **Navigation**: Use the Home page to navigate between analysis modes
2. **Caching**: Data fetching is cached for performance - use the refresh buttons if needed
3. **Filters**: Apply filters in sidebars to focus analysis on specific configurations
4. **Configuration**: Customize appearance in the Configuration Explorer - settings persist between sessions
5. **Export**: Use the Data Export page for bulk downloads and different formats

## Original Application

The original monolithic `Home.py` remains unchanged and can still be used with:

```bash
uv run streamlit run src/statflow/Home.py
```

## Development

The application follows a modular architecture with:
- Separation of concerns (data, visualization, UI)
- Reusable components
- Comprehensive caching for performance
- YAML-based configuration persistence
- Type hints and documentation throughout