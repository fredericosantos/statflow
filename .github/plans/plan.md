# Statflow Library Structure Refactoring

## Objective

Refactor Statflow into installable library with proper package structure, remove MLflow server management, implement modular architecture separating UI from business logic, and improve code quality by removing improper error handling and default fallbacks.

## Tasks

- [x] Fix imports and basic structure
  - [x] `src/statflow/__init__.py`
  - [x] `src/statflow/config.py`
- [x] Refactor Home.py with experiment/dataset selection
  - [x] Remove MLflow server start/run code
  - [x] Add experiment selection with multi-select
  - [x] Add dynamic datasets based on parameter
  - [x] Add parameter selection and linking UI
- [x] Update config file path to `.statflow_config.yaml`
  - [x] `src/statflow/config.py`
- [x] Clean up try/except and defaults across codebase
  - [x] Remove `.get()` fallbacks
  - [x] Remove masking try/except blocks
- [x] Implement pages_modules structure
  - [x] Create `pages_modules/` directory
  - [x] Separate UI from business logic
- [x] Create AGENTS.md documentation