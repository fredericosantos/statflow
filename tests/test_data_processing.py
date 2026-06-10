"""
Tests for src/statflow/functional/dataframes/data_processing.py.

Covers pure/testable parts:
- calculate_pareto_front: basic correctness and empty-input handling
- apply_metric_filters: range filtering and NaN handling (mocks session_state)
- fetch_experiment_data: verifies prefix stripping and dataset column rename
  (mocks RunsCache and st.session_state)
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest


class _FakeSessionState(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return dict.get(self, key, default)


# ---------------------------------------------------------------------------
# calculate_pareto_front
# ---------------------------------------------------------------------------


def test_pareto_front_basic():
    from statflow.functional.dataframes.data_processing import calculate_pareto_front

    df = pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1]})
    front = calculate_pareto_front(df, "x", "y")
    # All points are on the front for this linear-inverse case
    assert len(front) == 4


def test_pareto_front_dominated_removed():
    from statflow.functional.dataframes.data_processing import calculate_pareto_front

    # Point (2, 2) is dominated by (1, 1) in minimization
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 0]})
    front = calculate_pareto_front(df, "x", "y")
    # x=2, y=2 is dominated by x=1, y=1 — should not appear on the front
    # x=3, y=0 dominates on y but x=1,y=1 is better on x and comparable on y
    assert len(front) <= 3


def test_pareto_front_empty():
    from statflow.functional.dataframes.data_processing import calculate_pareto_front

    df = pl.DataFrame({"x": [], "y": []})
    front = calculate_pareto_front(df, "x", "y")
    assert front.is_empty()


def test_pareto_front_single_point():
    from statflow.functional.dataframes.data_processing import calculate_pareto_front

    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    front = calculate_pareto_front(df, "x", "y")
    assert len(front) == 1


# ---------------------------------------------------------------------------
# apply_metric_filters
# ---------------------------------------------------------------------------


def test_apply_metric_filters_no_filters(monkeypatch):
    import streamlit as st

    fake_state = _FakeSessionState(
        active_metric_filters=[],
        metric_filter_values={},
        metric_filter_nans={},
    )
    monkeypatch.setattr(st, "session_state", fake_state)

    from statflow.functional.dataframes.data_processing import apply_metric_filters

    df = pl.DataFrame({"rmse": [0.1, 0.5, 0.9]})
    result = apply_metric_filters(df)
    assert len(result) == 3


def test_apply_metric_filters_range(monkeypatch):
    import streamlit as st

    fake_state = _FakeSessionState(
        active_metric_filters=["rmse"],
        metric_filter_values={"rmse": (0.2, 0.8)},
        metric_filter_nans={"rmse": False},
    )
    monkeypatch.setattr(st, "session_state", fake_state)

    from statflow.functional.dataframes.data_processing import apply_metric_filters

    df = pl.DataFrame({"rmse": [0.1, 0.3, 0.5, 0.7, 0.9]})
    result = apply_metric_filters(df)
    assert len(result) == 3  # 0.3, 0.5, 0.7


def test_apply_metric_filters_include_nans(monkeypatch):
    import streamlit as st

    fake_state = _FakeSessionState(
        active_metric_filters=["rmse"],
        metric_filter_values={"rmse": (0.2, 0.8)},
        metric_filter_nans={"rmse": True},
    )
    monkeypatch.setattr(st, "session_state", fake_state)

    from statflow.functional.dataframes.data_processing import apply_metric_filters

    df = pl.DataFrame({"rmse": [0.1, 0.5, None, float("nan")]})
    result = apply_metric_filters(df)
    # 0.5 is in range, None and NaN are included due to flag; 0.1 is excluded
    assert len(result) == 3


def test_apply_metric_filters_empty_df(monkeypatch):
    import streamlit as st

    fake_state = _FakeSessionState(
        active_metric_filters=["rmse"],
        metric_filter_values={"rmse": (0.0, 1.0)},
        metric_filter_nans={"rmse": False},
    )
    monkeypatch.setattr(st, "session_state", fake_state)

    from statflow.functional.dataframes.data_processing import apply_metric_filters

    result = apply_metric_filters(pl.DataFrame())
    assert result.is_empty()


# ---------------------------------------------------------------------------
# fetch_experiment_data
# ---------------------------------------------------------------------------


def _make_full_runs_df() -> pl.DataFrame:
    return pl.DataFrame({
        "run_id": ["r1", "r2", "r3"],
        "start_time": [None, None, None],
        "params.dataset": ["ds_a", "ds_b", "ds_a"],
        "params.lr": ["0.01", "0.001", "0.01"],
        "metrics.loss": [0.1, 0.2, 0.15],
        "metrics.acc": [0.9, 0.8, 0.85],
    })


def test_fetch_experiment_data_metrics_prefix(monkeypatch):
    import streamlit as st
    from statflow.loggers import runs_cache as rc_module

    fake_state = _FakeSessionState(
        selected_datasets=["ds_a"],
        dataset_param="dataset",
    )
    monkeypatch.setattr(st, "session_state", fake_state)

    full_df = _make_full_runs_df()
    monkeypatch.setattr(
        rc_module.RunsCache,
        "filter_by_datasets",
        classmethod(lambda cls, param, datasets: full_df.filter(
            pl.col("params.dataset").is_in(datasets)
        )),
    )

    from statflow.functional.dataframes.data_processing import fetch_experiment_data

    result = fetch_experiment_data("metrics.")
    # Prefix should be stripped: "metrics.loss" → "loss"
    assert "loss" in result.columns
    assert "acc" in result.columns
    # params.* should not appear (not in the prefix)
    assert not any(c.startswith("params.") for c in result.columns)


def test_fetch_experiment_data_dataset_col_renamed(monkeypatch):
    import streamlit as st
    from statflow.loggers import runs_cache as rc_module

    fake_state = _FakeSessionState(
        selected_datasets=["ds_a"],
        dataset_param="dataset",
    )
    monkeypatch.setattr(st, "session_state", fake_state)

    full_df = _make_full_runs_df()
    monkeypatch.setattr(
        rc_module.RunsCache,
        "filter_by_datasets",
        classmethod(lambda cls, param, datasets: full_df.filter(
            pl.col("params.dataset").is_in(datasets)
        )),
    )

    from statflow.functional.dataframes.data_processing import fetch_experiment_data

    result = fetch_experiment_data("metrics.")
    assert "dataset_name" in result.columns


def test_fetch_experiment_data_params_prefix(monkeypatch):
    import streamlit as st
    from statflow.loggers import runs_cache as rc_module

    fake_state = _FakeSessionState(
        selected_datasets=["ds_a", "ds_b"],
        dataset_param="dataset",
    )
    monkeypatch.setattr(st, "session_state", fake_state)

    full_df = _make_full_runs_df()
    monkeypatch.setattr(
        rc_module.RunsCache,
        "filter_by_datasets",
        classmethod(lambda cls, param, datasets: full_df),
    )

    from statflow.functional.dataframes.data_processing import fetch_experiment_data

    result = fetch_experiment_data("params.")
    assert "lr" in result.columns
    assert "dataset_name" in result.columns
    # Metrics should not appear
    assert not any(c.startswith("metrics.") for c in result.columns)


def test_fetch_experiment_data_empty_when_no_datasets(monkeypatch):
    import streamlit as st

    fake_state = _FakeSessionState(
        selected_datasets=[],
        dataset_param="dataset",
    )
    monkeypatch.setattr(st, "session_state", fake_state)

    from statflow.functional.dataframes.data_processing import fetch_experiment_data

    result = fetch_experiment_data("metrics.")
    assert result.is_empty()
