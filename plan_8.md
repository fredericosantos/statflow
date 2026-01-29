# Plan_8.md - Add Parameters and Metrics Subpages

## 1) New Feature Objective

### App Navigation Enhancement
- **Current Structure**: Home → Single Dataset → Multiple Datasets → Export → Settings → Plot Macros
- **Proposed Structure**: Home → Parameters → Metrics → Single Dataset → Multiple Datasets → Export → Settings → Plot Macros
- **Rationale**: Provide dedicated pages for exploring and configuring parameters and metrics before diving into dataset-specific analysis, improving user workflow and discoverability.

## 2) Implementation Plan

### Key Objectives
1. **Create Parameters Page**: A new page for parameter exploration, configuration, and visualization.
2. **Create Metrics Page**: A new page for metrics overview, selection, and analysis.
3. **Update Navigation**: Modify the app's page structure to include these subpages in the correct order.
4. **Integrate with Session State**: Ensure parameters and metrics selections persist and affect downstream pages.

### Detailed Changes

#### 1. Create Parameters Page (`pages/1_🔧_Parameters.py`)
- **Purpose**: Allow users to explore, filter, and configure experiment parameters.
- **Features**:
  - Parameter distribution visualization
  - Parameter correlation analysis
  - Parameter filtering and selection
  - Integration with session state for parameter choices
- **UI Components**: Use existing components like filters, tables, and graphs.
- **Business Logic**: Create `pages_modules/module_1_Parameters/processor.py` for data processing.

#### 2. Create Metrics Page (`pages/2_📊_Metrics.py`)
- **Purpose**: Provide an overview of available metrics, their distributions, and selection.
- **Features**:
  - Metrics summary statistics
  - Metrics comparison across experiments
  - Metrics selection for analysis
  - Visualizations of metric distributions
- **UI Components**: Leverage existing tables and graphs components.
- **Business Logic**: Create `pages_modules/module_2_Metrics/processor.py` for metrics processing.

#### 3. Update Page Navigation
- **Rename Existing Pages**: Adjust page numbers and emojis to accommodate new pages.
  - Current `1_🔬_Single_Dataset.py` → `3_🔬_Single_Dataset.py`
  - Current `2_📋_Multiple_Datasets.py` → `4_📋_Multiple_Datasets.py`
  - And so on for subsequent pages.
- **Update Imports**: Ensure all cross-page references use correct page numbers.
- **Session State Integration**: Ensure parameters and metrics selections from these pages are available in session state for use in later pages.

#### 4. Update Home Page
- **Modify Home.py**: Update navigation links and descriptions to reflect new page structure.
- **Add Transitions**: Provide smooth flow from Home to Parameters to Metrics to dataset analysis.

#### 5. Validation
- Test navigation flow between pages.
- Ensure session state persistence for parameter and metric selections.
- Verify that downstream pages (Single Dataset, etc.) correctly use selections from Parameters and Metrics pages.

### Implementation Notes
- Follow existing patterns for page structure, module separation, and component reuse.
- Use atomic files and proper imports as established in previous plans.
- Ensure UI consistency with existing pages (icons, styling, etc.).
- Add appropriate documentation and comments.</content>
<parameter name="filePath">/home/fsx/repos/statflow/plan_8.md