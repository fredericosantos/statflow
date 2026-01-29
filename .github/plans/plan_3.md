# Plan_3.md - Final Statflow Refactoring Tasks

## 1) Failed Tasks from plan_2.md

### Incomplete or Not Implemented
- **Modular Architecture**: Only implemented for 2 pages (Single Dataset, Multiple Datasets); remaining pages (Export, Settings, Plot Macros) still have mixed UI/business logic; `pages_modules/general/` not created.
- **Code Quality**: Try/except blocks remain in `utils/` and `config.py` (non-IO); .get() with fallbacks not eliminated; large functions not fully broken down into atomic files.
- **Documentation**: AGENTS.md not updated with technical notes on modular structure, session persistence, etc.
- **Validation**: Not performed (user will manually test).

### Hasty Implementations
- Modular structure: Partial implementation; business logic not fully separated in all pages.
- Code quality: Defaults and try/except cleanup incomplete; atomic files not ensured across all modules.

## 2) New Plan to be Implemented

### Key Objectives
1. **Complete Modular Architecture**: Finish pages_modules/ for all pages; move all business logic, ensure atomic files.
2. **Full Code Quality**: Remove all non-IO try/except, eliminate .get() fallbacks, break down large functions.
3. **Documentation**: Update AGENTS.md with comprehensive technical notes.
4. **Validation**: User manual testing to ensure functionality.

### Detailed Changes

#### 1. Complete Modular Architecture
- Create `pages_modules/module_3_Export/`, `module_4_Settings/`, `module_5_Plot_Macros/` with `processor.py`.
- Move data export, settings management, plot macro logic to respective processors.
- Create `pages_modules/general/` for shared utilities (e.g., data fetchers).
- Ensure pages only handle UI; all processing in modules.
- Update all `__init__.py` with detailed descriptions per env.instructions.md.
- Make all files atomic: single responsibility, small size.

#### 2. Full Code Quality
- Scan and remove try/except in `utils/data_processing.py`, `utils/visualization.py`, `utils/mlflow_client.py`, `config.py` (keep IO ones like requests, yaml load).
- Replace `.get(key, default)` with direct access or single truth sources (e.g., define constants).
- Break down large functions in `table_builders.py`, `config.py` into smaller atomic functions.
- Ensure PEP 8, type hints, f-strings, pathlib throughout.

#### 3. Documentation
- Update AGENTS.md with sections on modular architecture, session persistence, dynamic datasets, code quality practices.

#### 4. Validation
- User to manually test app launch, data loading, persistence, UI functionality.

### Implementation Order
1. Complete modular architecture for remaining pages.
2. Full code quality cleanup.
3. Update AGENTS.md.
4. User manual testing.

### Validation
- All pages use modular processors.
- No masking try/except or .get() fallbacks.
- AGENTS.md updated.
- App fully functional per user testing.</content>
<parameter name="filePath">plan_3.md