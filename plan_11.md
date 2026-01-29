# Plan_11.md - Complete Remaining Enhancements

## 1) Outstanding Issues from Plan_10.md

### Experiment Comparison for Metrics
- **Problem**: `render_metrics_distributions()` lacks experiment-wise comparisons because DataFrames don't include experiment labels
- **Solution**: Modify `fetch_all_datasets_parallel()` to add `experiment_name` column to returned DataFrame
- **Implementation**:
  - Update function to accept `selected_experiments` parameter
  - Map experiment names to IDs
  - Add experiment column when concatenating runs

### Integration Validation
- **Verify Parameter Usage**: Ensure Single Dataset and Multiple Datasets pages filter options based on `selected_params`
- **Verify Metrics Usage**: Ensure pages respect `selected_metrics` for available metrics
- **Add Warnings**: Show warnings when selections are empty but required

### Enhanced Error Handling
- **Data Validation**: Robust checks for empty DataFrames, invalid types
- **User Feedback**: Clear messages for no data, filtering results, etc.

## 2) Implementation Plan

### Phase 1: Experiment Metadata Addition
1. **Update `fetch_all_datasets_parallel()`**:
   - Add `selected_experiments` parameter
   - Use `get_metadata_from_experiments()` to get experiment IDs
   - Modify search_runs to filter by experiment_ids
   - Add `experiment_name` column to each run's DataFrame

2. **Update Processors**:
   - Modify `fetch_metrics_data()` and `fetch_parameter_data()` to pass `selected_experiments`
   - Ensure DataFrames include experiment information

3. **Update Visualizations**:
   - Modify `render_metrics_distributions()` to add "Experiment Comparison" tab
   - Use experiment column for grouping in plots

### Phase 2: Integration Fixes
1. **Review Single Dataset Page**:
   - Check if `selected_params` filters available parameters
   - Add logic to restrict parameter options based on selection

2. **Review Multiple Datasets Page**:
   - Ensure `selected_metrics` limits available metrics
   - Add validation for empty selections

3. **Add User Feedback**:
   - Warnings for missing selections
   - Info messages for filtering results

### Phase 3: Polish and Testing
1. **Functional Testing**: Test all features with real MLflow data
2. **UI Consistency**: Ensure new elements match existing styling
3. **Performance**: Verify Polars improvements
4. **Documentation**: Update docstrings for new features

## 3) Validation Criteria
- Experiment comparisons render correctly with proper grouping
- Parameter and metrics selections restrict downstream options appropriately
- Clear user feedback for all states (no data, filtered, etc.)
- App maintains performance with Polars
- No breaking changes to existing workflows

## 4) Implementation Notes
- Maintain backward compatibility with existing function signatures where possible
- Use Polars expressions for efficient filtering and grouping
- Follow established patterns for error handling and UI components
- Test with multiple experiments to ensure comparison features work</content>
<parameter name="filePath">/home/fsx/repos/statflow/plan_11.md