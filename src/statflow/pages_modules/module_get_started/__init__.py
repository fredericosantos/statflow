"""
Logic and UI modules for get started page.

This package contains logic and UI components for the get started page,
organized by functionality for better maintainability.

module_get_started/
├── __init__.py              # Package initialization and exports
├── experiment_selector.py   # Experiment selection logic
├── dataset_config.py        # Dataset configuration logic
├── parameter_config.py      # Parameter selection and linking logic
├── dataset_mode.py          # Dataset definition mode selection (UI)
├── experiment_selection.py  # Experiment/dataset selection logic (UI)
└── parameter_selection.py   # Parameter selection and linking UI
"""

# Import and re-export UI modules for backward compatibility
from . import dataset_mode, parameter_selection