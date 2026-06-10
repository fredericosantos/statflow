"""
Tests for provider schema contracts.

Monkeypatches the backend boundary for each provider and asserts that
fetch_runs() output conforms to the canonical schema defined in
statflow/loggers/base.py:
  - run_id   : str column present
  - start_time : datetime column present
  - params.* columns with str dtype
  - metrics.* columns with float dtype
  - cursors are updated after a call
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl

from statflow.loggers.base import (
    METRIC_PREFIX,
    PARAM_PREFIX,
    RUN_ID_COL,
    START_TIME_COL,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _FakeSessionState(dict):
    """Minimal session-state stand-in that supports attribute and key access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


# ---------------------------------------------------------------------------
# MLflow provider
# ---------------------------------------------------------------------------


_FAKE_MLFLOW_RUNS = pd.DataFrame(
    {
        "run_id": ["run_abc", "run_def"],
        "start_time": [
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-02 11:00:00"),
        ],
        "params.lr": ["0.01", "0.001"],
        "params.n_layers": ["3", "5"],
        "metrics.rmse": [0.12, 0.09],
        "metrics.r2": [0.88, 0.91],
    }
)


class _FakeExp:
    experiment_id = "1"
    name = "my_exp"


def test_mlflow_provider_schema(monkeypatch):
    """MLflowProvider.fetch_runs returns canonical schema columns."""
    import streamlit as st

    fake_state = _FakeSessionState(mlflow_server_url="http://fake:5000", provider="mlflow")
    monkeypatch.setattr(st, "session_state", fake_state)

    import mlflow  # noqa: PLC0415 — must be after monkeypatch of session_state

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", lambda name: _FakeExp())
    monkeypatch.setattr(mlflow, "search_runs", lambda **kw: _FAKE_MLFLOW_RUNS.copy())

    from statflow.loggers.mlflow.provider import MLflowProvider

    provider = MLflowProvider()
    df, cursors = provider.fetch_runs(["my_exp"], max_results=100)

    assert not df.is_empty(), "DataFrame should not be empty"
    assert RUN_ID_COL in df.columns, "run_id column missing"
    assert START_TIME_COL in df.columns, "start_time column missing"

    param_cols = [c for c in df.columns if c.startswith(PARAM_PREFIX)]
    assert len(param_cols) > 0, "No params.* columns found"

    metric_cols = [c for c in df.columns if c.startswith(METRIC_PREFIX)]
    assert len(metric_cols) > 0, "No metrics.* columns found"

    # Metrics should be float-compatible
    for col in metric_cols:
        assert df[col].dtype in (pl.Float32, pl.Float64), (
            f"{col} should be float, got {df[col].dtype}"
        )


def test_mlflow_provider_cursors_updated(monkeypatch):
    """Cursors dict advances after a successful fetch."""
    import streamlit as st

    fake_state = _FakeSessionState(mlflow_server_url="http://fake:5000", provider="mlflow")
    monkeypatch.setattr(st, "session_state", fake_state)

    import mlflow

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", lambda name: _FakeExp())
    monkeypatch.setattr(mlflow, "search_runs", lambda **kw: _FAKE_MLFLOW_RUNS.copy())

    from statflow.loggers.mlflow.provider import MLflowProvider

    provider = MLflowProvider()
    _df, cursors = provider.fetch_runs(["my_exp"], max_results=100)
    assert "my_exp" in cursors, "Cursor for experiment should be set after fetch"
    assert isinstance(cursors["my_exp"], int), "MLflow cursor should be an int (ms timestamp)"


def test_mlflow_provider_empty_experiment(monkeypatch):
    """fetch_runs with an empty experiment returns (empty df, cursors)."""
    import streamlit as st

    fake_state = _FakeSessionState(mlflow_server_url="http://fake:5000", provider="mlflow")
    monkeypatch.setattr(st, "session_state", fake_state)

    import mlflow

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", lambda name: None)
    monkeypatch.setattr(mlflow, "search_runs", lambda **kw: pd.DataFrame())

    from statflow.loggers.mlflow.provider import MLflowProvider

    provider = MLflowProvider()
    df, cursors = provider.fetch_runs(["missing_exp"], max_results=100)
    assert df.is_empty()


# ---------------------------------------------------------------------------
# W&B provider
# ---------------------------------------------------------------------------


def _make_wandb_viewer_data() -> dict:
    return {"viewer": {"username": "test_user", "entity": "test_entity"}}


def _make_wandb_runs_data() -> dict:
    return {
        "project": {
            "runs": {
                "edges": [
                    {
                        "cursor": "cursor_1",
                        "node": {
                            "name": "run_xyz",
                            "displayName": "my-run",
                            "createdAt": "2024-03-01T10:00:00Z",
                            "state": "finished",
                            "group": "grp1",
                            "config": '{"lr": {"value": 0.01, "desc": null}, "epochs": {"value": 10, "desc": null}}',
                            "summaryMetrics": '{"rmse": 0.05, "r2": 0.95, "_runtime": 120}',
                        },
                    }
                ],
                "pageInfo": {"hasNextPage": False, "endCursor": "cursor_1"},
            }
        }
    }


def _wandb_graphql_side_effect(url, json, auth, timeout):
    """Fake requests.post that returns viewer or runs data based on the query."""
    query = json["query"]

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self) -> dict:
            if "viewer" in query and "projects" not in query and "runs" not in query:
                return {"data": _make_wandb_viewer_data()}
            return {"data": _make_wandb_runs_data()}

    return FakeResponse()


def test_wandb_provider_schema(monkeypatch):
    """WandbProvider.fetch_runs returns canonical schema columns."""
    import requests
    import streamlit as st

    fake_state = _FakeSessionState(wandb_entity="test_entity", provider="wandb")
    monkeypatch.setattr(st, "session_state", fake_state)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(requests, "post", _wandb_graphql_side_effect)

    from statflow.loggers.wandb.provider import WandbProvider

    provider = WandbProvider()
    df, cursors = provider.fetch_runs(["test_project"], max_results=10)

    assert not df.is_empty(), "W&B DataFrame should not be empty"
    assert RUN_ID_COL in df.columns, "run_id column missing"
    assert START_TIME_COL in df.columns, "start_time column missing"

    param_cols = [c for c in df.columns if c.startswith(PARAM_PREFIX)]
    assert len(param_cols) > 0, "No params.* columns found"

    metric_cols = [c for c in df.columns if c.startswith(METRIC_PREFIX)]
    assert len(metric_cols) > 0, "No metrics.* columns found"

    # Metrics should be float-compatible
    for col in metric_cols:
        assert df[col].dtype in (pl.Float32, pl.Float64), (
            f"{col} should be float, got {df[col].dtype}"
        )


def test_wandb_provider_no_system_metrics(monkeypatch):
    """W&B provider strips _-prefixed system metrics from summaryMetrics."""
    import requests
    import streamlit as st

    fake_state = _FakeSessionState(wandb_entity="test_entity", provider="wandb")
    monkeypatch.setattr(st, "session_state", fake_state)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(requests, "post", _wandb_graphql_side_effect)

    from statflow.loggers.wandb.provider import WandbProvider

    provider = WandbProvider()
    df, _ = provider.fetch_runs(["test_project"], max_results=10)

    metric_cols = [c for c in df.columns if c.startswith(METRIC_PREFIX)]
    # _runtime should not appear
    assert "metrics._runtime" not in metric_cols, "_runtime should be excluded"


def test_wandb_provider_cursors_updated(monkeypatch):
    """Cursor advances after a successful W&B fetch."""
    import requests
    import streamlit as st

    fake_state = _FakeSessionState(wandb_entity="test_entity", provider="wandb")
    monkeypatch.setattr(st, "session_state", fake_state)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setattr(requests, "post", _wandb_graphql_side_effect)

    from statflow.loggers.wandb.provider import WandbProvider

    provider = WandbProvider()
    _df, cursors = provider.fetch_runs(["test_project"], max_results=10)
    assert "test_project" in cursors
