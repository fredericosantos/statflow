# Plan_6.md - Final Remaining Statflow Refactoring Tasks

## 1) Incomplete Tasks from plan_5.md

### Code Quality Issues
- **.get() Fallbacks**: Fixed in comparison_table_builder.py, but no further issues.
- **Try/Except Blocks**: All handled.
- **Large Functions**: Breakdown completed for build_rmse_table, but not for build_statistical_table and build_nodes_table (though build_nodes_table is simple).
- **Atomic Files**: Splitting completed, but functions still present in table_builders.py; not fully removed.
- **File Organization**: utils/ folder not organized; all files are flat, including new builder files.

### Documentation
- **AGENTS.md Update**: Completed.

### Validation
- **User Testing**: App starts, but full manual testing not performed.

## 2) New Plan to be Implemented

### Key Objectives
1. **Organize utils Folder**: Create subfolders for logical groupings (e.g., `utils/table_builders/` for table-related builders).
2. **Fully Remove Unused Functions**: Delete remaining functions from `table_builders.py` that have been moved to new files.
3. **Break Down Remaining Large Functions**: Add helpers to `statistical_table_builder.py` for maintainability.
4. **Update Imports**: Adjust import paths after reorganization.
5. **Validation**: Full user manual testing (do NOT run Streamlit server after implementation).

### Detailed Changes

#### 1. Organize utils Folder
- Create `utils/table_builders/` subfolder.
- Move `comparison_table_builder.py`, `rmse_table_builder.py`, `statistical_table_builder.py`, `nodes_table_builder.py` into `utils/table_builders/`.
- Update `__init__.py` in `utils/table_builders/` to expose functions.
- Keep `table_builders.py` in `utils/` if it has shared utilities, else move remaining functions.

#### 2. Fully Remove Unused Functions from table_builders.py
- Delete `build_rmse_table`, `build_statistical_table`, `build_nodes_table` from `table_builders.py`.
- Retain shared functions like `calculate_scaled_std_for_dataset`, `calculate_target_distribution_stats` if used elsewhere.

#### 3. Break Down Remaining Large Functions
- In `statistical_table_builder.py`:
  - Split `build_statistical_table` into helpers: `_prepare_stat_data`, `_find_best_configs`, `_calculate_p_values`, `_apply_holm_bonferroni`, `_format_table`.
- Ensure each helper has single responsibility.

#### 4. Update Imports
- Change imports in affected files to use new paths (e.g., `from statflow.utils.table_builders.comparison_table_builder import build_comparison_table`).
- Update all pages and modules accordingly.

#### 5. Validation
- User to manually test all features without running the server post-implementation.
- Ensure no import errors or functionality regressions.

### Implementation Order
1. Organize utils folder structure.
2. Fully remove unused functions from table_builders.py.
3. Break down large functions in statistical_table_builder.py.
4. Update all imports across codebase.
5. User manual testing (no server run).

### Validation
- utils/ folder organized.
- Unused functions removed.
- Functions broken down.
- Imports updated.
- App fully functional.</content>
<parameter name="filePath">/home/fsx/repos/statflow/plan_6.md