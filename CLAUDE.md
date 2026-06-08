# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Statflow is a Streamlit app for analyzing and comparing MLflow experiment results
(parameter exploration, metrics distributions, statistical comparison of method groups).
The intended use is symbolic-regression / GP research runs, but nothing is hardcoded to that.

> **Stale docs warning.** `README.md` and several `__init__.py` / module docstrings describe an
> older layout (`Home.py`, `pages/`, `utils/`, `module_1_Parameters`, `module_metrics`,
> `module_single_dataset`, etc.). Most of that no longer exists — the last commit was a rewrite
> ("rewrite w/o legacy"). Trust the actual file tree and the sections below, **not** the README
> or the ASCII trees inside docstrings, for structure. The docstring trees are still accurate
> for the *contents of the file they live in*.

## Commands

```bash
uv sync                                              # install deps
uv run streamlit run src/statflow/app.py             # run the app (entry point is app.py)
uv run streamlit run src/statflow/app.py --server.address 0.0.0.0   # bind all interfaces
uv run ruff check . / uv run ruff format .           # lint / format
uv run ty check                                      # type check (Astral ty)
```

There are **no tests** and **no ruff/ty config** in `pyproject.toml` yet; the commands above work
on Astral defaults. Don't claim tests pass — there are none to run.

The app requires a reachable **MLflow tracking server**. Default URI is `http://0.0.0.0:5000`
(`MLFLOW_TRACKING_URI` in `config.py`); it's overridable at runtime via the `mlflow_server_url`
session-state key (set in the UI). `ServerStatusManager` hits `<uri>/health` and the Get Started
page blocks until the server is up.

## Architecture

Streamlit multipage app. `app.py` declares the pages with `st.Page(...)` and wires them into
`st.navigation({...})`. Page script paths in `app.py` are **relative to `app.py`** (e.g.
`"subpages/parameters.py"`), which is why the app must be launched via `app.py`.

Data and UI flow:

1. **Get Started** (`subpages/get_started.py`) — pick experiments / datasets. Selecting them calls
   `RunsCache.load_runs(...)`, which fetches MLflow runs into a single **Polars DataFrame** stored
   in `st.session_state`.
2. **RunsCache** (`loggers/mlflow/runs_cache.py`) — the central data store. All-classmethod
   "singleton" backed by session state. Runs are one wide DataFrame where columns are prefixed
   `params.<name>` and `metrics.<name>` (plus `run_id`, `start_time`, etc.). On load it derives
   `available_params` / `available_metrics` by stripping those prefixes. Supports timestamp-based
   incremental pagination (`load_more_runs`) and dedups by `run_id`.
3. **Subpages** (`subpages/`) filter that cached DataFrame via
   `functional/dataframes/data_processing.py::fetch_experiment_data(prefix)` and
   `RunsCache.filter_by_datasets(...)`, then render. Parameters/Comparison build a **`group_label`**
   column by concatenating `param=value, param2=value2`; method groups are compared on that label.
   Comparison runs Wilcoxon rank-sum with Holm–Bonferroni correction (`subpages/comparison.py`).

The intended pipeline is Get Started → Parameters → Metrics → Results / Comparison. Several pages
referenced in `app.py` (Single Dataset, Export, Settings, etc.) are commented out — only the five
above currently exist.

### Module layout

| Path | Role |
|---|---|
| `app.py` | Navigation + page registration. Entry point. |
| `subpages/` | **UI only**, one file per page. Each has a `main()` called under `if __name__ == "__main__"` and a module-level `st.set_page_config(...)`. |
| `config.py` | `SessionState` (session-state manager), `DEFAULT_STATE` (state schema), `PERSISTABLE_KEYS`, YAML load/save. |
| `loggers/mlflow/` | `mlflow_client.py` (experiment discovery, `@st.cache_data`) and `runs_cache.py` (the run-data store). |
| `functional/dataframes/` | Shared Polars logic (`fetch_experiment_data`, `apply_metric_filters`, `calculate_pareto_front`). |
| `pages_modules/module_get_started/` | Page-specific logic for Get Started (experiment/dataset selection, dataset modes, server status). |
| `components/` | Reusable widgets: `selection_ui.py` (`SelectionManager` — unified select/order/rename), `filters.py`. |
| `managers/naming.py` | `NamingManager` — resolves user display-name overrides for datasets/metrics/groups. |
| `shared/server_status.py` | `ServerStatusManager` — MLflow health check + sidebar UI. |

### State & persistence

`st.session_state` is the single source of truth. `SessionState.initialize()` (idempotent, called at
the top of each page's `main()`) seeds every key in `DEFAULT_STATE`, preferring values loaded from
`.statflow_config.yaml` (in the **current working directory**, gitignored). Only keys in
`PERSISTABLE_KEYS` are written back via `save_to_config()` / `save_key_to_config()` — transient UI
state stays out of that list. When adding a new persisted preference, add it to **both** `DEFAULT_STATE`
and `PERSISTABLE_KEYS`.

## Conventions (project-specific)

- **MLflow access**: always `mlflow.set_tracking_uri(st.session_state["mlflow_server_url"])` before
  any MLflow call. Don't re-query MLflow for schema/metadata — read `available_params`,
  `available_param_values`, `available_metrics` from session state (populated by `RunsCache`).
- **Icons**: never raw emoji in UI. Always `:material/<name>:` in `icon=` args and titles.
- **Error handling**: no `try/except` that hides bugs and no `.get()` fallback defaults that paper
  over missing state — use direct `st.session_state["key"]` access so failures surface. Exceptions:
  external network/IO (MLflow fetches, HTTP health checks, file reads) may catch.
- **Atomic files**: small single-responsibility files; each file/module starts with a docstring
  containing a mini ASCII tree of its own contents. Keep that docstring in sync when you change a file.
- **Python 3.13**, `X | None` not `Optional[X]`. **Polars**, not pandas (MLflow returns pandas;
  convert at the boundary with `pl.from_pandas`).
- **Adding a page**: create `subpages/<name>.py` with `main()` + the `__main__` guard, then register
  it in the `st.navigation({...})` dict in `app.py`. Page-specific logic goes in
  `pages_modules/module_<name>/`; anything reusable across pages goes in `functional/` or `components/`.
