# Plan_4.md - Final Remaining Statflow Refactoring Tasks

## 1) Incomplete Tasks from plan_3.md

### Code Quality Issues
- **.get() Fallbacks**: Many `.get(key, default)` calls remain in `config.py` (e.g., in `initialize_session_state`, `save_session_state_to_config`, `get_all_datasets`). These should be replaced with direct access or single truth sources.
- **Try/Except Blocks**: Non-IO try/except blocks still present in `utils/` files (e.g., parsing in `visualization.py`, table building in `table_builders.py`). Remove those masking bugs, keep IO-related ones (file reads, network calls).
- **Large Functions**: `table_builders.py` (876 lines) contains large functions like `build_comparison_table`, `build_rmse_table`, etc. Break down into smaller atomic functions.
- **Atomic Files**: `table_builders.py` is monolithic; split into smaller files (e.g., `comparison_table_builder.py`, `rmse_table_builder.py`).

### Documentation
- **AGENTS.md Update**: Attempted but failed; still has old content. Add sections on modular architecture, session persistence, dynamic datasets, code quality practices.

### Validation
- **User Testing**: Not performed; app functionality needs manual testing for relaunches, data loading, persistence, UI.

## 2) New Plan to be Implemented

### Key Objectives
1. **Eliminate .get() Fallbacks**: Replace with direct access or define constants/single sources.
2. **Remove Non-IO Try/Except**: Scan and remove try/except that mask bugs in utils.
3. **Break Down Large Functions**: Split large functions in `table_builders.py` into atomic ones.
4. **Split Monolithic Files**: Break `table_builders.py` into multiple atomic files.
5. **Update Documentation**: Successfully update AGENTS.md with new technical notes.
6. **Validation**: User manual testing to ensure no regressions.

### Detailed Changes

#### 1. Eliminate .get() Fallbacks
- In `config.py`:
  - `get_all_datasets()`: Use direct access to `st.session_state['available_datasets']` if exists, else raise or handle explicitly.
  - `initialize_session_state()`: Replace `.get()` with explicit checks or defaults defined as constants.
  - `save_session_state_to_config()`: Same, use direct access.
- Define constants for defaults instead of inline fallbacks.

#### 2. Remove Non-IO Try/Except
- Scan `utils/data_processing.py`, `visualization.py`, `table_builders.py` for try/except not related to IO (file/network).
- Remove those that catch broad exceptions masking bugs (e.g., ValueError/TypeError in parsing).
- Keep try/except for `pd.read_csv`, `requests.get`, `yaml.safe_load`, etc.

#### 3. Break Down Large Functions
- In `table_builders.py`:
  - `build_comparison_table`: Split into sub-functions for data aggregation, formatting, etc.
  - `build_rmse_table`: Break into significance calculation, table construction.
  - `build_statistical_table`: Separate p-value computation, multiple testing correction.
  - `build_nodes_table`: Split tree size extraction and comparison.
- Ensure each function has single responsibility.

#### 4. Split Monolithic Files
- Create new files in `utils/`:
  - `comparison_table_builder.py`: For cross-dataset comparisons.
  - `rmse_table_builder.py`: For RMSE with significance.
  - `statistical_table_builder.py`: For statistical tests.
  - `nodes_table_builder.py`: For tree size comparisons.
- Update imports in affected modules.

#### 5. Update Documentation
- Add to AGENTS.md:
  - Section on Session State Persistence (.statflow_config.yaml).
  - Dynamic Datasets (get_all_datasets based on session).
  - Atomic Files (single responsibility, small size).
  - Code Quality (no .get() fallbacks, no masking try/except).

#### 6. Validation
- User to manually test:
  - App launch and session persistence on relaunch.
  - Data loading for single/multiple datasets.
  - Export functionality.
  - Settings updates.
  - Plot macros generation.
- Ensure no errors surface due to removed try/except.

### Implementation Order
1. Eliminate .get() fallbacks in config.py.
2. Remove non-IO try/except in utils.
3. Break down large functions in table_builders.py.
4. Split table_builders.py into atomic files.
5. Update AGENTS.md.
6. User manual testing.

### Validation
- No .get() fallbacks or masking try/except.
- Functions are atomic and small.
- AGENTS.md updated.
- App fully functional per user testing.</content>
<parameter name="filePath">/home/fsx/repos/statflow/plan_4.md