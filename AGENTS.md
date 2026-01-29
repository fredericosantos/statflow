# AGENTS.md

## Technical Notes

- **MLflow Tracking URI**: Always use `st.session_state['mlflow_server_url']` and call `mlflow.set_tracking_uri()` before any MLflow operations.
- **Session State**: 
    - `available_params`, `available_param_values`, and `available_metrics` are populated in `Home.py` upon experiment selection.
    - These should be used instead of querying MLflow for schema/metadata.
- **Modular Structure**:
    - `pages/` should only contain Streamlit UI code.
    - Business logic/complex processing should reside in `pages_modules/module_X/` (where X is the page name).
- **Python Version**: Using Python 3.13. Use `Type | None` instead of `Optional[Type]`.
- **Imports**: Ensure `__init__.py` files exist in all directories and contain package descriptions as per `env.instructions.md`.
- **Error Handling**: Follow the project philosophy of removing `try/except` blocks that mask bugs, except for external network/IO operations.
- **Tools**: Use `uv run` for executing scripts in the terminal.

## Session State Persistence
- User preferences and session data are saved to `.statflow_config.yaml` in the project root.
- On app launch, session state is loaded from this file, allowing persistence across relaunches.
- Includes settings like selected datasets, graph configurations, and custom colors/symbols.

## Dynamic Datasets
- `get_all_datasets()` function retrieves available datasets based on `st.session_state['available_datasets']`.
- Falls back to `DEFAULT_DATASETS` if not set, enabling dynamic dataset handling without hardcoding.

## Atomic Files
- Code is organized into small, single-responsibility files and functions.
- Each module focuses on one aspect, improving maintainability and readability.

## Code Quality Practices
- No `.get()` fallbacks with defaults; use direct access or explicit checks.
- No try/except blocks that mask bugs; exceptions are allowed to surface for debugging.
- Functions are broken down into atomic units with clear responsibilities.

## UI Guidelines
- **Filtering Warning**: ⚠️ Runs without a value in the dataset parameter field will be filtered out.
- **Icons**: Emojis, when possible, should ALWAYS be `:material/name_of_icon`.
