"""
Run-data providers and the backend-agnostic cache.

This package owns how statflow reaches experiment-tracking backends and turns
their runs into the single canonical wide Polars DataFrame the rest of the app
consumes. Adding a backend means adding a `RunProvider` under a subpackage and
registering it — nothing downstream of `RunsCache` changes.

loggers/
├── base.py        # RunProvider interface + canonical DataFrame schema contract
├── registry.py    # provider registration + lazy lookup by name
├── runs_cache.py  # RunsCache: provider-agnostic fetch/merge/dedup/cache
├── mlflow/        # MLflow provider
└── wandb/         # Weights & Biases provider
"""
