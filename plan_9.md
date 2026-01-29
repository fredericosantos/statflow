# Plan_9.md - Fix Plan_8 Gaps and Migrate to Polars

## 1) Issues from Plan_8 Implementation

### Missing Features
- **Parameter Correlation Analysis**: The Parameters page (`1_🔧_Parameters.py`) lacks correlation analysis between parameters, as mentioned in the plan but not implemented in `render_parameter_distributions()`.
- **Advanced Parameter Filtering**: While selection is implemented, there's no UI for filtering parameters based on values or conditions.
- **Metrics Comparison Across Experiments**: The Metrics page has basic distributions but no explicit comparison visualizations across different experiments.
- **Integration Validation**: Need to verify that parameter and metrics selections from the new pages are properly utilized in downstream pages (Single Dataset, Multiple Datasets, etc.).

### Hastily Implemented Areas
- **Visualization Depth**: The `render_parameter_distributions()` and `render_metrics_distributions()` functions are basic; they could include more advanced plots like correlation heatmaps for parameters.
- **Error Handling**: Limited error handling in new processors for edge cases (e.g., no data, invalid data types).
- **Performance**: No optimization for large datasets in the new pages.

## 2) Polars Migration

### Objective
Replace all pandas usage with Polars for better performance, especially with large MLflow datasets. Polars is faster and more memory-efficient than pandas for data processing tasks.

### Scope
- Update all imports from `import pandas as pd` to `import polars as pl`
- Convert all DataFrame operations from pandas to Polars equivalents
- Update type hints from `pd.DataFrame` to `pl.DataFrame`
- Ensure compatibility with existing Streamlit and Plotly integrations (Polars DataFrames can be converted to pandas when needed for plotting)
- Update dependencies in `pyproject.toml`: Replace `pandas` with `polars`

### Files to Update
Based on grep search, the following files use pandas/pd.:
- `src/statflow/Home.py`
- `src/statflow/pages_modules/module_1_Parameters/processor.py`
- `src/statflow/pages_modules/module_2_Metrics/processor.py`
- `src/statflow/pages_modules/module_3_Single_Dataset/processor.py`
- `src/statflow/pages_modules/module_7_Plot_Macros/processor.py`
- And potentially others in `utils/`, `components/`, etc.

### Migration Strategy
1. **Update Imports**: Change `import pandas as pd` to `import polars as pl`
2. **DataFrame Creation**: Use `pl.DataFrame()` instead of `pd.DataFrame()`
3. **Operations**: Translate pandas methods to Polars equivalents (e.g., `df.groupby()` → `df.group_by()`, `df.merge()` → `df.join()`, etc.)
4. **Type Hints**: Change `pd.DataFrame` to `pl.DataFrame`
5. **Plotly Compatibility**: Where needed, convert Polars DataFrames to pandas with `.to_pandas()` for Plotly charts
6. **Testing**: Ensure all functionality works after migration, especially data fetching and visualizations

## 3) Implementation Plan

### Phase 1: Fix Plan_8 Gaps
1. **Enhance Parameters Page**:
   - Add correlation heatmap to `render_parameter_distributions()` in `components/graphs.py`
   - Add filtering UI in `1_🔧_Parameters.py` for parameter value ranges

2. **Enhance Metrics Page**:
   - Add experiment-wise comparison plots in `render_metrics_distributions()`
   - Improve summary statistics to include experiment-level breakdowns

3. **Integration Testing**:
   - Verify that `selected_params` and `selected_metrics` from session state are used in Single Dataset and Multiple Datasets pages
   - Add warnings if selections are empty

### Phase 2: Polars Migration
1. **Update Dependencies**:
   - Remove `pandas` from `pyproject.toml`
   - Add `polars` to dependencies

2. **Core Utils Migration**:
   - Update `utils/mlflow_client.py` to return Polars DataFrames
   - Update `utils/data_processing.py` and other utils

3. **Page Modules Migration**:
   - Migrate all processor.py files in `pages_modules/`
   - Update component functions in `components/` that handle DataFrames

4. **UI Compatibility**:
   - Ensure Streamlit components work with Polars (most do, but test tables, etc.)
   - Convert to pandas only when necessary for Plotly

### Phase 3: Validation and Optimization
1. **Functional Testing**: Run the app and test all pages with sample data
2. **Performance Testing**: Compare load times with large datasets
3. **Error Handling**: Add robust error handling for data processing failures
4. **Documentation**: Update docstrings and comments to reflect Polars usage

## 4) Validation Criteria
- All pages load without errors
- Parameter and metrics selections persist and affect downstream analysis
- Visualizations render correctly with Polars data
- App performance improves or remains comparable
- No breaking changes to user workflow

## 5) Implementation Notes
- Polars API is similar to pandas but with different method names (e.g., `group_by` instead of `groupby`)
- Use Polars' lazy evaluation where beneficial for complex queries
- Maintain atomic file structure and follow existing best practices
- Test with real MLflow data to ensure compatibility</content>
<parameter name="filePath">/home/fsx/repos/statflow/plan_9.md