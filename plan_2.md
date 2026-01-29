# Plan_2.md - Remaining Statflow Refactoring Tasks

## 1) Failed Tasks from plan.md

### Incomplete or Not Implemented
- **Dynamic Datasets**: ALL_DATASETS not made a function based on selected parameter (used session state instead).
- **Clean Defaults**: No removal of .get() with fallbacks; multiple ground truths still exist.
- **Code Quality Improvements**: Try/except blocks removed only in pages/; remaining in utils/ and config.py (except IO/network).
- **Modular Architecture**: pages_modules/ structure not implemented; business logic still mixed in pages/.
- **Documentation**: AGENTS.md exists but may need updates for new technical notes.
- **Validation**: Not performed (user will manually test).

### Hasty Implementations
- ALL_DATASETS handling: Changed to session state access instead of proper function as specified.
- Config saving: .statflow_config.yaml created but not integrated to save/load full session state for relaunch defaults.
- Try/except cleanup: Partial, only UI pages done; utils/config still have masking blocks.

## 2) New Plan to be Implemented

### Key Objectives
1. **Modular Architecture**: Implement pages_modules/ with atomic files; separate UI from business logic.
2. **Session State Persistence**: Save/load full session state to .statflow_config.yaml for relaunch defaults.
3. **Code Quality**: Complete try/except removal (non-IO), clean defaults, ensure atomic files per env.instructions.md.
4. **Dynamic Datasets**: Implement ALL_DATASETS as function using selected parameter.
5. **Documentation**: Update AGENTS.md with current technical notes.

### Detailed Changes

#### 1. Modular Architecture
- Create `pages_modules/` directory.
- For each page (1_🔬_Single_Dataset.py, etc.), create `pages_modules/module_X/` with `processor.py` for business logic.
- Move data fetching, processing, and complex logic to processor.py.
- Pages focus only on UI rendering and user interaction.
- Shared logic in `pages_modules/general/`.
- Ensure atomic files: each file small, focused on single responsibility.
- Update __init__.py files with descriptions as per env.instructions.md.

#### 2. Session State Persistence
- Modify config.py to save/load full session state (selected_experiments, available_params, etc.) to .statflow_config.yaml.
- On app launch, load defaults from config to session state.
- Ensure config saves on changes or exit.

#### 3. Code Quality
- Remove remaining try/except blocks in utils/ and config.py (keep for IO/network).
- Eliminate .get() with fallbacks; use single ground truth values.
- Ensure atomic files: break down large functions/classes into smaller ones.
- Follow PEP 8, type hints, f-strings, pathlib, etc. per env.instructions.md.

#### 4. Dynamic Datasets
- In config.py, define `get_all_datasets()` function that returns datasets based on session state or selected parameter.
- Update all usages to call this function instead of session state direct access.

#### 5. Documentation
- Update AGENTS.md with notes on modular structure, session persistence, etc.

### Implementation Order
1. Implement modular architecture: Create pages_modules/, move logic, update __init__.py.
2. Add session state persistence to config.py.
3. Complete code quality: Remove try/except, clean defaults, ensure atomic files.
4. Implement get_all_datasets() function.
5. Update AGENTS.md.
6. User manual testing.

### Validation
- Modular structure enforced.
- Session state persists across relaunches.
- No masking try/except, clean defaults.
- Dynamic datasets via function.
- Atomic files with proper descriptions.</content>
<parameter name="filePath">plan_2.md