# Plan.md - Statflow Refactoring Plan

## Overview
Refactor the Statflow repository to be an installable library with proper structure, remove MLflow server management, and abstract the app to work with any MLflow experiment.

## Key Objectives
1. **Library Structure**: Move files properly inside `src/statflow` for installable package.
2. **MLflow Abstraction**: Remove server running functionality, keep only URI connection.
3. **Home Page Refactor**: Simplify to experiment/dataset selection with proper parameter handling.
4. **Modular Architecture**: Separate UI (pages) from business logic (pages_modules).
5. **Code Quality**: Remove try/except blocks, deprecated code, and duplicates.
6. **Configuration**: Fix config file path and MLflow URI management.

## Detailed Changes

### 1. Library Structure
- **Move Files**: Ensure all code is under `src/statflow/`.
- **Package Structure**:
  ```
  src/statflow/
  ├── __init__.py
  ├── config.py
  ├── Home.py
  ├── components/
  ├── pages/
  ├── pages_modules/
  ├── utils/
  └── ...
  ```

### 2. MLflow Server Management
- **Remove**: Any code that starts/runs MLflow server.
- **Keep**: Only URI connection and status checking.
- **Centralize**: Single `mlflow.set_tracking_uri()` in Home.py.

### 3. Home Page Refactor
- **Remove**: Everything below "🗄️ MLflow Server" section.
- **Add**: Experiment selection with spinner and multi-select pills.
- **Dynamic Datasets**: Make ALL_DATASETS a function based on selected parameter.
- **Parameter Selection**: 
  - Ask user for dataset parameter (default to MLflow "Dataset" or "dataset_name").
  - Warning about filtered runs.
- **Dataset Ordering**: Use `sort_items` below pills, auto-updates.
- **Parameter Linking**: Expander with checkboxes and linking dropdowns.
- **Session State**: Store params/metrics on experiment selection, use throughout app.

### 4. Code Quality Improvements
- **Remove try/except**: Replace with proper error handling or let errors surface.
- **Clean Defaults**: One ground truth, no `.get()` with fallbacks.
- **Config Path**: Change from `~/.statflow/config.yaml` to `.statflow_config.yaml` in repo root.
- **Imports**: Fix incorrect imports (e.g., `streamlit_comparison` → `statflow`).

### 5. Modular Architecture
- **Pages**: Focus on UI elements only.
- **Pages Modules**: Business logic in `pages_modules/module_X/processor.py`.
- **General Modules**: Shared logic in `pages_modules/general/`.

### 6. Documentation
- **AGENTS.md**: Technical notes for future development.
- **Plan.md**: This document.

## Implementation Order
1. Fix imports and basic structure. 
2. Refactor Home.py as specified.
3. Update config.py for proper paths.
4. Clean up try/except and defaults across codebase.
5. Implement pages_modules structure.
6. Create AGENTS.md.
7. Test and validate.

## Validation
- All modules import successfully.
- App runs without MLflow server management.
- Dynamic dataset/parameter handling works.
- No try/except blocks in UI code.