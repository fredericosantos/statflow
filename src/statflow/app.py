"""
Launcher for the Statflow multi-page Streamlit application.

This is the main entry point that provides navigation to all application pages.

app.py
├── Page navigation setup with organized sections
├── Multi-page application launcher
├── Global configuration and styling
└── Font customization (SF Mono, JetBrains Mono)

Usage:
    uv run streamlit run src/statflow/app.py --server.address 0.0.0.0
"""

import streamlit as st

# Create page objects for navigation
get_started_page = st.Page("subpages/get_started.py", title="Get Started", icon=":material/home:")
parameters_page = st.Page("subpages/parameters.py", title="Parameters", icon=":material/tune:")
metrics_page = st.Page("subpages/metrics.py", title="Metrics", icon=":material/bar_chart:")
results_page = st.Page("subpages/results.py", title="Results", icon=":material/insights:")
comparison_page = st.Page("subpages/comparison.py", title="Comparison", icon=":material/trophy:")
# single_dataset_page = st.Page("subpages/single_dataset.py", title="Single Dataset", icon=":material/science:")
# multiple_datasets_page = st.Page("subpages/multiple_datasets.py", title="Multiple Datasets", icon=":material/list:")
# export_data_page = st.Page("subpages/export_data.py", title="Data Export", icon=":material/save:")
# plot_macros_page = st.Page("subpages/plot_macros.py", title="Plot Macros", icon=":material/bar_chart:")
# settings_page = st.Page("subpages/settings.py", title="Settings", icon=":material/settings:")

# Create organized navigation with sections
pg = st.navigation(
    {
        "Setup": [get_started_page, parameters_page, metrics_page],
        "Analysis": [results_page, comparison_page],
        # "Exporting": [plot_macros_page, export_data_page],
        # "Settings": [settings_page],
    },
    # position="top"
)

# Set global page config
st.set_page_config(
    page_title="Statflow - MLflow Experiment Analysis",
    page_icon=":material/analytics:",
    layout="wide",
)

# Run the navigation
pg.run()
