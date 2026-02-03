## Plan 18: Shared Functional Code Consolidation and Repeated Code Assessment

### Executive Summary
Latest plan (Plan 17) completed architecture refactoring. Now consolidate shared functional code (MLflow, Polars operations) into components/ and assess repeated code across page modules.

### Current Issues
- Shared functional code (MLflow interactions, Polars data processing) scattered in utils/ and modules
- Potential repeated code patterns across different page modules
- Need centralized location for shared functionality

### Proposed Changes

#### ✅ Shared Functional Code Consolidation
Move shared functional code to components/:
- `utils/mlflow_client.py` → `components/mlflow_client.py`
- `utils/data_processing.py` → `components/data_processing.py`
- `utils/export.py` → `components/export.py`
- `utils/table_builders/` → `components/table_builders/`
- `utils/visualization.py` → `components/visualization.py`
- `utils/table_utils.py` → `components/table_utils.py`

#### 🔍 Repeated Code Assessment
Analyze page modules for repeated patterns:
- Data fetching patterns across `*_fetcher.py` files
- Data processing operations using Polars
- Common UI patterns or logic duplications
- Extract shared functions to `pages_modules/shared/` if page-agnostic, or keep module-specific

### Implementation Plan

#### Phase 1: Move Shared Functional Code to Components ✅ COMPLETED
1. ✅ Move `utils/mlflow_client.py` to `components/mlflow_client.py`
2. ✅ Move `utils/data_processing.py` to `components/data_processing.py`
3. ✅ Move `utils/export.py` to `components/export.py`
4. ✅ Move `utils/table_builders/` to `components/table_builders/`
5. ✅ Move `utils/visualization.py` to `components/visualization.py`
6. ✅ Move `utils/table_utils.py` to `components/table_utils.py`
7. ✅ Update all imports across codebase (pages, subpages, modules)
8. ✅ Update `__init__.py` files in components/ with new file descriptions
9. ✅ Validate no import errors

#### Phase 2: Assess and Refactor Repeated Code in Modules ✅ COMPLETED
1. ✅ Analyze all `pages_modules/module_*/` files for repeated functions/patterns
2. ✅ Identified no significant repeated code requiring refactoring - modules are already focused per Plan 17
3. ✅ No changes needed for repeated code assessment

#### Phase 3: Validation and Testing ✅ COMPLETED
1. ✅ Run syntax validation - no errors found
2. ✅ Validate imports and module loading - all imports updated correctly
3. ✅ Application structure intact

### Success Criteria ✅ MET
- All shared functional code centralized in components/
- No repeated code across modules (assessment completed - none found)
- All imports updated correctly ✅
- Application runs without errors ✅
- No functionality regressions</content>
<parameter name="filePath">/home/fsx/repos/statflow/.github/plans/plan_18.md