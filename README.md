# Statflow

Statflow is a [Streamlit](https://streamlit.io/) app for exploring and **statistically
comparing** experiment-tracking runs. It connects to either an **MLflow** tracking server or
**Weights & Biases** (the W&B website, via your existing API key), pulls runs into one wide table,
and helps you answer the question that matters: *is my method significantly better than the
baselines?*

It grew out of symbolic-regression / genetic-programming research, but nothing is hardcoded to
that — any params/metrics work, and whether a metric is better lower or higher is configurable.

## Features

- **Two data sources, one interface** — MLflow and W&B, selectable at runtime. Every page behaves
  identically regardless of source.
- **Parameter exploration** — pick parameters, filter values, and build method groups from
  `param=value` combinations.
- **Metric distributions & filters** — range/NaN filters applied consistently across pages.
- **Ours-vs-theirs significance** — Wilcoxon rank-sum (Mann–Whitney U) with Holm–Bonferroni
  correction, per-metric minimize/maximize direction, 🥇 markers and boxplots.
- **Persistent preferences** — selections and settings saved to `~/.statflow/config.yaml`.

## Requirements

- Python 3.13+ and [uv](https://docs.astral.sh/uv/).
- A data source:
  - **MLflow** — a reachable tracking server (default `http://0.0.0.0:5000`).
  - **Weights & Biases** — an API key in `~/.netrc` (machine `api.wandb.ai`) or the
    `WANDB_API_KEY` environment variable. Set `WANDB_BASE_URL` for a self-hosted instance.

## Quick start

**Install it as a command** (runs from any directory):

```bash
uv tool install git+https://github.com/fredericosantos/statflow   # or: uv tool install .
statflow                                                          # run the app
statflow --server.port 8502 --server.headless true                # args go to Streamlit
uv tool upgrade statflow                                          # update later
```

**Or run it from a clone** (for development):

```bash
uv sync                                                              # install deps
uv run streamlit run src/statflow/app.py                            # run the app
uv run streamlit run src/statflow/app.py --server.address 0.0.0.0   # bind all interfaces
```

The app entry point is **`app.py`** — the `st.Page(...)` paths in `st.navigation(...)` are relative
to it, so it must be launched via `app.py`. The `statflow` command (`cli.py`) does this for you,
resolving the packaged `app.py` regardless of your current directory.

## Data sources (providers)

Choose the data source in the **Get Started → Data Source** sidebar panel:

| | MLflow | Weights & Biases |
|---|---|---|
| Connection | Tracking URI | Entity (blank = your default) |
| "Experiments" are | MLflow experiments | W&B projects |
| Params come from | `params.*` | run `config` (unwrapped `.value`) |
| Metrics come from | `metrics.*` | numeric run `summaryMetrics` |
| Auth | server `/health` | `~/.netrc` key (never stored in the app) |

Both providers produce the **same canonical wide table** — `run_id`, `start_time`,
`params.<name>`, `metrics.<name>` — so the rest of the app is backend-agnostic.

> **W&B note.** Statflow talks to the W&B public **GraphQL API directly** (`requests` + the
> `~/.netrc` key), not through the `wandb` Python client. This avoids the client's local
> service-auth layer and keeps the dependency surface small.

**Adding a provider:** implement `RunProvider` in `loggers/<name>/provider.py`, register it with
`@register_provider`, and add it to the registry's module map. Nothing downstream changes.

## Workflow

**Get Started → Parameters → Metrics → Results / Comparison.**

1. **Get Started** — pick the provider and the experiments/datasets to analyze. Selecting them
   loads runs into a cached Polars DataFrame.
2. **Parameters** — choose which parameters to compare and build the `group_label` (e.g.
   `method=ours, pop=500`).
3. **Metrics** — choose metrics and apply range/NaN filters.
4. **Comparison** — mark your methods as "ours" vs the rest as baselines, pick a metric and its
   direction, and read significance.

## Statistical comparison

For each dataset and each "our" method, Statflow runs a **one-sided Mann–Whitney U test**
(Wilcoxon rank-sum) of that method against every baseline on the chosen metric, applies
**Holm–Bonferroni** correction across the baselines, and awards 🥇 when the method is significantly
better than **all** baselines at α = 0.05. "Better" follows the per-metric **direction** you set
(*Lower* for error/loss, *Higher* for accuracy/score), which is remembered per metric.

## Project layout

| Path | Role |
|---|---|
| `app.py` | Navigation + page registration. Entry point. |
| `cli.py` | `statflow` console script — runs the packaged `app.py` via Streamlit. |
| `config.py` | `SessionState`, `DEFAULT_STATE`, `PERSISTABLE_KEYS`, YAML persistence. |
| `loggers/base.py` | `RunProvider` interface + canonical DataFrame schema contract. |
| `loggers/registry.py` | Provider registration and lazy lookup by name. |
| `loggers/runs_cache.py` | `RunsCache` — provider-agnostic fetch/merge/dedup/cache. |
| `loggers/mlflow/provider.py` | MLflow provider. |
| `loggers/wandb/provider.py` | Weights & Biases provider (GraphQL). |
| `functional/dataframes/` | Shared Polars logic (filtering, Pareto fronts). |
| `pages_modules/module_get_started/` | Get Started logic (provider/experiment/dataset selection). |
| `components/` | Reusable widgets (`selection_ui`, `filters`). |
| `managers/naming.py` | Display-name overrides for datasets/metrics/groups. |
| `shared/server_status.py` | Provider status check + sidebar UI. |
| `subpages/` | One file per page (UI only). |

## Configuration

User preferences persist to **`~/.statflow/config.yaml`**. On first run, if the legacy
`.statflow_config.yaml` exists in the current working directory, it is copied to the new
location (the original is left untouched). On launch, `SessionState.initialize()` seeds every
key from `DEFAULT_STATE`, preferring saved values; only keys in `PERSISTABLE_KEYS` are written
back.

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ty check          # type check
```

All four run in CI on every push and pull request (`.github/workflows/ci.yml`).
