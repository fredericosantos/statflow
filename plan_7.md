# Plan_7.md - Remaining Import Updates and Final Validation

## 1) Incomplete Tasks from plan_6.md

### Update Imports
- **Import Paths**: Imports in pages, modules, and docstrings still reference old paths (e.g., `from statflow.utils.rmse_table_builder import build_rmse_table` instead of `from statflow.utils.table_builders.rmse_table_builder import build_rmse_table`).
- **Docstrings**: Usage examples in builder files have outdated import paths.

### Validation
- **User Testing**: Plan specified not to run Streamlit server, but testing was performed. Ensure no regressions from import updates.

## 2) New Plan to be Implemented

### Key Objectives
1. **Update Import Paths**: Change all imports to reflect the new `utils/table_builders/` subfolder structure.
2. **Update Docstrings**: Correct usage examples in builder files.
3. **Final Validation**: Test imports and basic functionality without running the full server.

### Detailed Changes

#### 1. Update Import Paths
- In all pages (e.g., `pages/2_📋_Multiple_Datasets.py`):
  - Change `from statflow.utils.rmse_table_builder import build_rmse_table` to `from statflow.utils.table_builders.rmse_table_builder import build_rmse_table`
  - Similarly for `statistical_table_builder` and `nodes_table_builder`
- In modules (e.g., `pages_modules/module_2_Multiple_Datasets/processor.py`):
  - Update the same imports.
- In builder files themselves:
  - Update any self-referential imports if present (though they should import from config/utils directly).

#### 2. Update Docstrings
- In each builder file (e.g., `statistical_table_builder.py`):
  - Change usage example from `from statflow.utils.statistical_table_builder import build_statistical_table` to `from statflow.utils.table_builders.statistical_table_builder import build_statistical_table`

#### 3. Final Validation
- Run import tests for all modules.
- Perform manual testing of key features without launching the Streamlit server.
- Ensure no functionality regressions due to path changes.

### Implementation Notes
- Use absolute paths for clarity.
- Verify that the `utils/table_builders/__init__.py` correctly exposes functions if needed, but prioritize direct imports as per plan.
- After updates, test that `uv run python -c "import statflow; from statflow.pages import *"` works.</content>
<parameter name="filePath">/home/fsx/repos/statflow/plan_7.md