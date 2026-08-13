# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Statflow is a Streamlit app for exploring and statistically comparing experiment-tracking runs
from **either MLflow or Weights & Biases** (parameter exploration, metrics distributions,
ours-vs-theirs significance testing). The intended use is symbolic-regression / GP research runs,
but nothing is hardcoded to that — metric direction (lower/higher-is-better) is configurable.

## Commands

```bash
uv sync                                              # install deps
uv run streamlit run src/statflow/app.py             # run the app (entry point is app.py)
uv run streamlit run src/statflow/app.py --server.address 0.0.0.0   # bind all interfaces
uv run pytest                                        # tests
uv run ruff check . / uv run ruff format .           # lint / format
uv run ty check                                      # type check (Astral ty)

uv tool install .                                    # install the `statflow` command
statflow --server.port 8513                          # run from any directory
```

`pyproject.toml` configures pytest, ruff (line-length 100, py313) and ty. All four checks run in
CI on every push and PR (`.github/workflows/ci.yml`) — run them locally before opening a PR.

The app requires a reachable **data source**, chosen at runtime in the Get Started sidebar and
persisted in the `provider` session-state key (`"mlflow"` or `"wandb"`):

- **MLflow** — a tracking server. Default URI `http://0.0.0.0:5000` (`MLFLOW_TRACKING_URI` in
  `config.py`), overridable via the `mlflow_server_url` key. Status = `<uri>/health`.
- **Weights & Biases** — the W&B website's GraphQL API, authenticated by the `api.wandb.ai` key in
  `~/.netrc` (or `WANDB_API_KEY`). Entity via the `wandb_entity` key (blank = default). Status =
  an auth probe.

`ServerStatusManager.check_status()` delegates to the active provider; Get Started blocks until it
returns true.

## Architecture

Streamlit multipage app. `app.py` declares the pages with `st.Page(...)` and wires them into
`st.navigation({...})`. Page script paths in `app.py` are **relative to `app.py`** (e.g.
`"subpages/parameters.py"`), which is why the app must be launched via `app.py`.

Data and UI flow:

1. **Providers** (`loggers/`) — the data-source seam. `loggers/base.py` defines the `RunProvider`
   ABC and the **canonical DataFrame schema contract** (`run_id`, `start_time`, `params.<name>`,
   `metrics.<name>`). `loggers/registry.py` resolves the active provider by name. `mlflow/provider.py`
   and `wandb/provider.py` each translate their backend into that schema — so everything downstream
   is backend-agnostic. **All `import mlflow` lives in `mlflow/provider.py`; W&B talks to the GraphQL
   API via `requests` + `~/.netrc`.**
2. **Get Started** (`subpages/get_started.py`) — pick the provider, then experiments / datasets.
   Selecting them calls `RunsCache.load_runs(...)`.
3. **RunsCache** (`loggers/runs_cache.py`) — the central, **provider-agnostic** data store.
   All-classmethod "singleton" backed by session state. It delegates fetching to the active provider
   and keeps only merge/dedup/derivation: runs are one wide DataFrame, and it derives
   `available_params` / `available_metrics` by stripping the prefixes. Supports cursor-based
   incremental pagination (`load_more_runs`, opaque per-provider cursors) and dedups by `run_id`.
4. **Subpages** (`subpages/`) filter that cached DataFrame via
   `functional/dataframes/data_processing.py::fetch_experiment_data(prefix)` and
   `RunsCache.filter_by_datasets(...)`, then render. Parameters/Comparison build a **`group_label`**
   column by concatenating `param=value, param2=value2`; method groups are compared on that label.
   Comparison runs one-sided Wilcoxon rank-sum (Mann–Whitney U) with Holm–Bonferroni correction and
   a **per-metric minimize/maximize** direction (`subpages/comparison.py`).

The intended pipeline is Get Started → Parameters → Metrics → Results / Comparison. Several pages
referenced in `app.py` (Single Dataset, Export, Settings, etc.) are commented out — only the five
above currently exist.

### Module layout

| Path | Role |
|---|---|
| `app.py` | Navigation + page registration. Entry point. |
| `subpages/` | **UI only**, one file per page. Each has a `main()` called under `if __name__ == "__main__"` and a module-level `st.set_page_config(...)`. |
| `config.py` | `SessionState` (session-state manager), `DEFAULT_STATE` (state schema), `PERSISTABLE_KEYS`, YAML load/save. |
| `loggers/base.py` | `RunProvider` ABC + canonical DataFrame schema contract. |
| `loggers/registry.py` | Provider registration (`@register_provider`) + lazy lookup (`get_provider`). |
| `loggers/runs_cache.py` | `RunsCache` — provider-agnostic fetch/merge/dedup/cache. |
| `loggers/mlflow/provider.py` | `MLflowProvider` — all `mlflow.*` access lives here. |
| `loggers/wandb/provider.py` | `WandbProvider` — W&B public GraphQL API via `requests` + netrc. |
| `functional/dataframes/` | Shared Polars logic (`fetch_experiment_data`, `apply_metric_filters`, `calculate_pareto_front`). |
| `pages_modules/module_get_started/` | Get Started logic: `provider_config` (data-source picker), experiment/dataset selection, dataset modes, server status. |
| `components/` | Reusable widgets: `selection_ui.py` (`SelectionManager` — unified select/order/rename), `filters.py`. |
| `managers/naming.py` | `NamingManager` — resolves user display-name overrides for datasets/metrics/groups. |
| `shared/server_status.py` | `ServerStatusManager` — delegates status to the active provider + sidebar UI. |

### State & persistence

`st.session_state` is the single source of truth. `SessionState.initialize()` (idempotent, called at
the top of each page's `main()`) seeds every key in `DEFAULT_STATE`, preferring values loaded from
`~/.statflow/config.yaml`. On first run, the legacy `<cwd>/.statflow_config.yaml` (if present) is
**copied** to the new path automatically — the original is left untouched. Only keys in
`PERSISTABLE_KEYS` are written back via `save_to_config()` / `save_key_to_config()` — transient UI
state stays out of that list. When adding a new persisted preference, add it to **both** `DEFAULT_STATE`
and `PERSISTABLE_KEYS`.

## Conventions (project-specific)

- **Provider access**: backend calls live **only** in `loggers/<name>/provider.py`. Subpages and
  page modules go through `RunsCache` / `get_provider(...)`, never a backend SDK directly. Don't
  re-query the backend for schema/metadata — read `available_params`, `available_param_values`,
  `available_metrics` from session state (populated by `RunsCache`). A new backend = a new
  `RunProvider` that emits the canonical schema in `loggers/base.py`; nothing downstream changes.
- **Icons**: never raw emoji in UI. Always `:material/<name>:` in `icon=` args and titles.
- **Error handling**: no `try/except` that hides bugs and no `.get()` fallback defaults that paper
  over missing state — use direct `st.session_state["key"]` access so failures surface. Exceptions:
  external network/IO (MLflow fetches, HTTP health checks, file reads) may catch.
- **Atomic files**: small single-responsibility files; each file/module starts with a docstring
  containing a mini ASCII tree of its own contents. Keep that docstring in sync when you change a file.
- **Python 3.13**, `X | None` not `Optional[X]`. **Polars**, not pandas — convert at the provider
  boundary (MLflow returns pandas → `pl.from_pandas`; W&B returns JSON → `pl.from_dicts`).
- **Adding a page**: create `subpages/<name>.py` with `main()` + the `__main__` guard, then register
  it in the `st.navigation({...})` dict in `app.py`. Page-specific logic goes in
  `pages_modules/module_<name>/`; anything reusable across pages goes in `functional/` or `components/`.
