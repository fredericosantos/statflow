"""
MLflow implementation of the RunProvider interface.

Owns every direct MLflow API call in statflow: experiment discovery, run
fetching (with start_time-watermark pagination), and the server health probe.
The tracking URI is read from session state (`mlflow_server_url`) and set on the
MLflow client inside each call so nothing outside this module touches MLflow.

provider.py
├── _experiment_names()   # cached experiment listing for a tracking URI
└── MLflowProvider        # RunProvider for an MLflow tracking server
    ├── check_status()    # GET <uri>/health
    ├── list_experiments()
    └── fetch_runs()      # search_runs per experiment -> canonical wide df
"""

from __future__ import annotations

import datetime
from typing import Any

import mlflow
import pandas as pd
import polars as pl
import requests
import streamlit as st

from statflow.loggers.base import START_TIME_COL, RunProvider
from statflow.loggers.registry import register_provider


@st.cache_data(ttl=600, show_spinner=False)
def _experiment_names(tracking_uri: str) -> list[str]:
    """Active experiment names for a tracking URI (cached per URI)."""
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments()
    return [exp.name for exp in experiments if exp.lifecycle_stage == "active"]


@register_provider
class MLflowProvider(RunProvider):
    """RunProvider backed by an MLflow tracking server."""

    name = "mlflow"
    label = "MLflow"

    def _tracking_uri(self) -> str:
        return st.session_state["mlflow_server_url"]

    def check_status(self) -> bool:
        tracking_uri = self._tracking_uri()
        if not tracking_uri.startswith("http"):
            # file:// or other non-HTTP store — assume usable, no health endpoint.
            return True
        try:
            health_url = tracking_uri.rstrip("/") + "/health"
            response = requests.get(health_url, timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_experiments(self) -> list[str]:
        return _experiment_names(self._tracking_uri())

    def fetch_runs(
        self,
        experiments: list[str],
        max_results: int,
        cursors: dict[str, Any] | None = None,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        cursors = dict(cursors or {})
        mlflow.set_tracking_uri(self._tracking_uri())

        all_new_runs: list[pl.DataFrame] = []
        for exp_name in experiments:
            # Cursor is the oldest start_time (ms) seen so far for this experiment;
            # paginate by fetching strictly older runs.
            filter_string = ""
            if exp_name in cursors:
                filter_string = f"attributes.start_time < {cursors[exp_name]}"

            try:
                exp = mlflow.get_experiment_by_name(exp_name)
                if not exp:
                    continue

                runs_pdf = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=filter_string,
                    max_results=max_results,
                    order_by=["attributes.start_time DESC"],
                )
            except Exception:
                # External IO: skip an experiment that fails rather than abort the batch.
                continue

            # mlflow.search_runs returns DataFrame | list[Run] depending on the
            # output_format kwarg (default "pandas").  Guard with isinstance so ty
            # knows we're operating on a pandas DataFrame before pl.from_pandas.
            if not isinstance(runs_pdf, pd.DataFrame) or runs_pdf.empty:
                continue

            runs_df = pl.from_pandas(runs_pdf)
            all_new_runs.append(runs_df)

            if START_TIME_COL in runs_df.columns:
                min_val = runs_df.get_column(START_TIME_COL).min()
                # Polars .min() returns a wide union; guard that it's a datetime
                # before calling .timestamp() to narrow the type for ty.
                if isinstance(min_val, datetime.datetime):
                    cursors[exp_name] = int(min_val.timestamp() * 1000)

        if not all_new_runs:
            return pl.DataFrame(), cursors

        # `align` reconciles differing param/metric columns across experiments.
        new_df = pl.concat(all_new_runs, how="align")
        return new_df, cursors
