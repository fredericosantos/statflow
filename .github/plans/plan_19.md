# Architecture Restructure - Functional Separation

## Objective

Separate UI components from functional code by creating `functional/` folder with subfolders by functionality, keeping `components/` UI-only.

## Tasks

- [x] Create `functional/` directory structure
  - [x] `functional/mlflow/`
  - [x] `functional/dataframes/`
  - [x] `functional/export/`
  - [x] `functional/table_builders/`
  - [x] `functional/visualization/`
  - [x] `functional/table_utils/`
  - [x] Create `__init__.py` for all subfolders
- [x] Move functional files from `components/` to `functional/`
  - [x] `mlflow_client.py` to `functional/mlflow/`
  - [x] `data_processing.py` to `functional/dataframes/`
  - [x] `export.py` to `functional/export/`
  - [x] `table_builders/` to `functional/table_builders/`
  - [x] `visualization.py` to `functional/visualization/`
  - [x] `table_utils.py` to `functional/table_utils/`
- [x] Update `components/` to UI-only
  - [x] Keep only `downloads.py`, `filters.py`, `graphs.py`, `tables.py`
  - [x] Update `components/__init__.py`
- [x] Update all imports
  - [x] `components/` files
  - [x] `functional/` subfolders
  - [x] `pages_modules/` modules
  - [x] `subpages/`
  - [x] `utils/`
- [x] Update documentation
  - [x] AGENTS.md
  - [x] `statflow/__init__.py`
  - [x] All `__init__.py` files