"""
Provider abstraction for experiment-tracking backends.

A ``RunProvider`` turns a tracking backend (MLflow, Weights & Biases, ...) into
the single wide Polars DataFrame the rest of statflow consumes. Everything
downstream of ``RunsCache`` is backend-agnostic because every provider conforms
to the same column contract.

Canonical run-DataFrame schema (one row per run):
  - ``run_id``      : str        unique run identifier (dedup key)
  - ``start_time``  : datetime   run creation time (pagination cursor source)
  - ``params.<k>``  : str        run parameters / config (stringified)
  - ``metrics.<k>`` : float      run metrics / summary values
Extra columns are allowed but ignored downstream; the four reserved shapes above
are the contract providers MUST honor.

base.py
├── RUN_ID_COL / START_TIME_COL / PARAM_PREFIX / METRIC_PREFIX  # schema contract
└── RunProvider                # abstract backend interface
    ├── name                   # short identifier ("mlflow", "wandb")
    ├── label                  # human-facing name for the UI
    ├── check_status()         # is the backend reachable / authenticated?
    ├── list_experiments()     # selectable top-level containers
    └── fetch_runs()           # (experiments, max_results, cursors) -> (df, cursors)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

# --- Canonical run-DataFrame schema contract -------------------------------
RUN_ID_COL = "run_id"
START_TIME_COL = "start_time"
PARAM_PREFIX = "params."
METRIC_PREFIX = "metrics."


class RunProvider(ABC):
    """Abstract experiment-tracking backend.

    Implementations translate a backend's runs into the canonical wide Polars
    DataFrame documented in this module. Providers are stateless with respect to
    caching — ``RunsCache`` owns merging, dedup, and session storage. A provider
    only knows how to *reach* its backend and *shape* the result.
    """

    #: Short machine identifier, used as the registry key and the persisted
    #: ``provider`` session-state value (e.g. ``"mlflow"``, ``"wandb"``).
    name: str = ""

    #: Human-facing label for selectors and status UI.
    label: str = ""

    @abstractmethod
    def check_status(self) -> bool:
        """Return True if the backend is reachable and usable.

        For server-backed providers this is a health probe; for API/token
        providers it is an auth check. Network/IO errors must be caught and
        reported as ``False`` rather than raised.
        """

    @abstractmethod
    def list_experiments(self) -> list[str]:
        """Return the selectable top-level containers.

        MLflow experiments, W&B projects, etc. — whatever the user picks on the
        Get Started page to scope which runs are fetched.
        """

    @abstractmethod
    def fetch_runs(
        self,
        experiments: list[str],
        max_results: int,
        cursors: dict[str, Any] | None = None,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Fetch one page of runs for the given experiments.

        Args:
            experiments: container names from :meth:`list_experiments`.
            max_results: maximum runs to fetch per experiment for this page.
            cursors: per-experiment pagination cursors returned by a prior call.
                The shape is opaque to the caller — each provider defines and
                consumes its own cursor (MLflow uses a ``start_time`` watermark,
                W&B a run offset, ...). ``None`` means start from the newest run.

        Returns:
            ``(runs_df, updated_cursors)`` where ``runs_df`` follows the
            canonical schema documented in this module and ``updated_cursors``
            advances pagination for the next call. ``runs_df`` may be empty when
            a page yields nothing new.
        """
