# Dynamic Configuration and Parameter Management

## Objective

Abstract `config.py` for cross-project usage by making dataset and parameter configurations dynamic, move parameter setup to Parameters page, and add dataset renaming functionality with persistence.

## Tasks

- [x] Update `config.py` for dynamic configuration
  - [x] Remove hardcoded DATASETS_REAL_LIFE and DATASETS_BLACKBOX
  - [x] Remove hardcoded parameter defaults
  - [x] Keep `dataset_renames` functionality but make dynamic
- [x] Move parameter setup to Parameters page
  - [x] Move `parameter_selection.render_parameter_selection` from `get_started.py` to `parameters.py`
  - [x] Update `parameters.py` to include parameter setup at top
  - [x] Remove parameter setup from `get_started.py`
  - [x] Update imports and dependencies
- [x] Add dataset renaming functionality
  - [x] Add dataset renaming UI in `parameters.py`
  - [x] Create function to edit `dataset_renames` dict
  - [x] Save changes to config on update
  - [x] Ensure renames used throughout app
- [x] Update dependencies
  - [x] Update code relying on hardcoded config
  - [x] Test parameter selection in new location
  - [x] Test dataset renaming persistence
- [x] Remove remaining `.get()` fallbacks
  - [x] Replace all `st.session_state.get()` calls in `subpages/` with direct access
  - [x] Syntax checks for modified files</content>
<parameter name="filePath">/home/fsx/repos/statflow/.github/plans/plan_20.md