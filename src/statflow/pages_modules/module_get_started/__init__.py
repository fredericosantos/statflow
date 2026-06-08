"""
Logic and UI modules for the Get Started page.

This package contains the page-specific logic and UI for selecting the data
source, experiments, and datasets to analyze.

module_get_started/
├── __init__.py              # Package initialization and exports
├── constants.py             # Dataset-param mode enums
├── provider_config.py       # Data-source provider picker + connection config
├── server_status.py         # Provider-unreachable handling + db viewer (MLflow)
├── experiment_selector.py   # Experiment/dataset selection from the provider
├── dataset_mode.py          # Dataset definition mode selection (UI)
└── dataset_config.py        # Dataset configuration logic
"""

# Re-export for callers that import the module from the package namespace.
from . import dataset_mode as dataset_mode
