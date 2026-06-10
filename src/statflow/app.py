"""
Launcher for the Statflow multi-page Streamlit application.

Entry point: `uv run streamlit run src/statflow/app.py`

app.py
├── Page navigation (Setup: Get Started, Parameters, Metrics)
└── Page navigation (Analysis: Results, Comparison, Overall, Plots)

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
overall_page = st.Page("subpages/overall.py", title="Overall", icon=":material/leaderboard:")
plots_page = st.Page("subpages/plots.py", title="Plots", icon=":material/show_chart:")

# Create organized navigation with sections
pg = st.navigation(
    {
        "Setup": [get_started_page, parameters_page, metrics_page],
        "Analysis": [results_page, comparison_page, overall_page, plots_page],
    },
)

# Set global page config
st.set_page_config(
    page_title="Statflow - MLflow Experiment Analysis",
    page_icon=":material/analytics:",
    layout="wide",
)

# Run the navigation
pg.run()
