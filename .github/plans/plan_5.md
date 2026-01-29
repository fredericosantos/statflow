# Plan_5.md - Final Remaining Statflow Refactoring Tasks

## 1) Incomplete Tasks from plan_4.md

### Code Quality Issues
- **.get() Fallbacks**: Still present in newly created `comparison_table_builder.py` (uses `st.session_state.get('available_datasets', [])`).
- **Try/Except Blocks**: All non-IO try/except removed, but no further issues.
- **Large Functions**: Breakdown in `table_builders.py` only partial (helpers added for `build_comparison_table`, but not for `build_rmse_table`, `build_statistical_table`, `build_nodes_table`).
- **Atomic Files**: Splitting started but incomplete; `table_builders.py` still contains all functions, and only `comparison_table_builder.py` created. Imports not updated across codebase.

### Documentation
- **AGENTS.md Update**: Completed successfully.

### Validation
- **User Testing**: App starts without errors, but full manual testing not performed (data loading, persistence, UI functionality).

## 2) New Plan to be Implemented

### Key Objectives
1. **Fix Remaining .get() Fallbacks**: Replace in `comparison_table_builder.py` with direct access.
2. **Complete Function Breakdown**: Break down remaining large functions in `table_builders.py` into atomic sub-functions.
3. **Complete File Splitting**: Create remaining atomic files (`rmse_table_builder.py`, `statistical_table_builder.py`, `nodes_table_builder.py`), move functions, remove from `table_builders.py`, update all imports.
4. **Update Imports**: Change all import statements in `pages/`, `pages_modules/`, and other files to use new atomic modules.
5. **Validation**: Full user manual testing.

### Detailed Changes

#### 1. Fix .get() in comparison_table_builder.py
- Replace `st.session_state.get('available_datasets', [])` with direct access: `st.session_state['available_datasets']` if present, else empty list or raise.

#### 2. Complete Function Breakdown
- In `table_builders.py`:
  - `build_rmse_table`: Split into sub-functions for data preparation, statistical testing, Holm-Bonferroni correction.
  - `build_statistical_table`: Separate p-value calculation, multiple testing, table formatting.
  - `build_nodes_table`: Break into data aggregation, comparison logic.
- Ensure all functions are atomic with single responsibilities.

#### 3. Complete File Splitting
- Create `rmse_table_builder.py` with `build_rmse_table` and helpers.
- Create `statistical_table_builder.py` with `build_statistical_table` and helpers.
- Create `nodes_table_builder.py` with `build_nodes_table` and helpers.
- Remove all functions from `table_builders.py`, leaving only shared utilities if any.

#### 4. Update Imports
- In `pages/2_📋_Multiple_Datasets.py`: Change imports to new files.
- In `pages/3_💾_Export.py`: Update imports.
- In `pages_modules/module_2_Multiple_Datasets/processor.py`: Update.
- Any other files importing from `table_builders`.

#### 5. Validation
- User to manually test all features:
  - App launch and session persistence.
  - Single and multiple dataset loading.
  - Table generation (comparison, RMSE, statistical, nodes).
  - Export and settings.
  - Plot macros.
- Ensure no regressions from removed try/except or .get().

### Implementation Order
1. Fix .get() in comparison_table_builder.py.
2. Complete breakdown of remaining functions in table_builders.py.
3. Create remaining atomic files and move functions.
4. Remove functions from table_builders.py.
5. Update all imports across codebase.
6. User manual testing.

### Validation
- All .get() fallbacks eliminated.
- Functions fully atomic.
- Files split and imports updated.
- App fully functional.</content>
<parameter name="filePath">/home/fsx/repos/statflow/plan_5.md