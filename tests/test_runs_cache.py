"""
Tests for src/statflow/loggers/runs_cache.py.

Covers:
- merge/dedup by run_id
- derived params/metrics refresh after merge
- RunsCache.get_runs() returning empty DataFrame when nothing cached
- clear_cache() removing all cache keys
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from statflow.loggers.base import RUN_ID_COL


class _FakeSessionState(dict):
    """Minimal stand-in for st.session_state (supports both key and attr access)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        return dict.get(self, key, default)


def _make_runs(run_ids: list[str], param_val: str = "a", metric_val: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": run_ids,
            "start_time": [None] * len(run_ids),
            "params.lr": [param_val] * len(run_ids),
            "metrics.loss": [metric_val] * len(run_ids),
        }
    )


class _FakeProvider:
    name = "fake"

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def fetch_runs(self, experiments, max_results, cursors=None):
        cursors = dict(cursors or {})
        for exp in experiments:
            cursors[exp] = "cursor_1"
        return self._df, cursors


@pytest.fixture()
def patched_cache(monkeypatch):
    """Patch st.session_state and get_provider so RunsCache uses our fakes."""
    import streamlit as st

    fake_state = _FakeSessionState(provider="fake")
    monkeypatch.setattr(st, "session_state", fake_state)

    # Return fixture so tests can swap provider df
    return fake_state


def _install_fake_provider(monkeypatch, df: pl.DataFrame) -> None:
    from statflow.loggers import runs_cache as rc_module

    monkeypatch.setattr(rc_module, "get_provider", lambda name: _FakeProvider(df))


def test_get_runs_returns_empty_when_nothing_cached(patched_cache, monkeypatch):
    _install_fake_provider(monkeypatch, pl.DataFrame())
    from statflow.loggers.runs_cache import RunsCache

    assert RunsCache.get_runs().is_empty()


def test_load_runs_populates_cache(patched_cache, monkeypatch):
    df = _make_runs(["r1", "r2"])
    _install_fake_provider(monkeypatch, df)
    from statflow.loggers.runs_cache import RunsCache

    result = RunsCache.load_runs(["exp1"])
    assert not result.is_empty()
    assert len(result) == 2


def test_load_runs_dedup_by_run_id(patched_cache, monkeypatch):
    """Loading the same runs twice should not duplicate rows."""
    df = _make_runs(["r1", "r2"])
    _install_fake_provider(monkeypatch, df)
    from statflow.loggers.runs_cache import RunsCache

    RunsCache.load_runs(["exp1"])
    # Force a second fetch by clearing the experiment set so load_more_runs can fire
    # Simulate a "load more" call that returns the same run_ids
    RunsCache.load_more_runs(["exp1"])
    cached = RunsCache.get_runs()
    # Dedup: still just 2 unique run_ids
    assert cached.get_column(RUN_ID_COL).n_unique() == 2
    assert len(cached) == 2


def test_load_runs_merges_new_rows(patched_cache, monkeypatch):
    """load_more_runs with new run_ids appends and deduplicates."""
    from statflow.loggers import runs_cache as rc_module
    from statflow.loggers.runs_cache import RunsCache

    call_count = 0
    batches = [
        _make_runs(["r1", "r2"]),
        _make_runs(["r3", "r4"]),
    ]

    class _SequentialProvider:
        name = "sequential"

        def fetch_runs(self, experiments, max_results, cursors=None):
            nonlocal call_count
            df = batches[min(call_count, len(batches) - 1)]
            call_count += 1
            return df, {"exp1": f"cursor_{call_count}"}

    monkeypatch.setattr(rc_module, "get_provider", lambda name: _SequentialProvider())

    RunsCache.load_runs(["exp1"])
    RunsCache.load_more_runs(["exp1"])
    cached = RunsCache.get_runs()
    assert len(cached) == 4
    assert set(cached.get_column(RUN_ID_COL).to_list()) == {"r1", "r2", "r3", "r4"}


def _sequential_provider(monkeypatch, batches: list[pl.DataFrame]) -> None:
    """Install a provider that returns `batches` in order, one per fetch."""
    from statflow.loggers import runs_cache as rc_module

    call_count = 0

    class _SequentialProvider:
        name = "sequential"

        def fetch_runs(self, experiments, max_results, cursors=None):
            nonlocal call_count
            df = batches[min(call_count, len(batches) - 1)]
            call_count += 1
            return df, {"exp1": f"cursor_{call_count}"}

    monkeypatch.setattr(rc_module, "get_provider", lambda name: _SequentialProvider())


def test_load_more_runs_with_all_null_param_column(patched_cache, monkeypatch):
    """A page where a param is null for every run must not break the merge.

    Regression: merging used `pl.concat(how="align")`, which full-outer-joins on
    the common columns. An all-null param column is Null dtype and cannot be a
    join key against a String one, so this raised
    "datatypes of join keys don't match - `params.optimizer_lr`: str ... null".
    """
    from statflow.loggers.runs_cache import RunsCache

    page1 = pl.DataFrame({"run_id": ["r1"], "params.optimizer_lr": ["0.01"]})
    page2 = pl.DataFrame({"run_id": ["r2"], "params.optimizer_lr": [None]})
    assert page2.schema["params.optimizer_lr"] == pl.Null, "fixture must reproduce Null dtype"

    _sequential_provider(monkeypatch, [page1, page2])

    RunsCache.load_runs(["exp1"])
    RunsCache.load_more_runs(["exp1"])

    cached = RunsCache.get_runs()
    assert set(cached.get_column(RUN_ID_COL).to_list()) == {"r1", "r2"}
    assert cached.schema["params.optimizer_lr"] == pl.String
    lookup = dict(zip(cached["run_id"], cached["params.optimizer_lr"], strict=True))
    assert lookup == {"r1": "0.01", "r2": None}


def test_load_more_runs_with_disjoint_columns(patched_cache, monkeypatch):
    """Columns present in only one page survive the merge, filled with null."""
    from statflow.loggers.runs_cache import RunsCache

    page1 = pl.DataFrame({"run_id": ["r1"], "params.a": ["x"], "metrics.loss": [1.0]})
    page2 = pl.DataFrame({"run_id": ["r2"], "params.b": ["y"], "metrics.loss": [2.0]})
    _sequential_provider(monkeypatch, [page1, page2])

    RunsCache.load_runs(["exp1"])
    RunsCache.load_more_runs(["exp1"])

    cached = RunsCache.get_runs().sort("run_id")
    assert len(cached) == 2
    assert {"params.a", "params.b", "metrics.loss"} <= set(cached.columns)
    assert cached["params.a"].to_list() == ["x", None]
    assert cached["params.b"].to_list() == [None, "y"]
    assert cached["metrics.loss"].to_list() == [1.0, 2.0]


def test_clear_cache(patched_cache, monkeypatch):
    df = _make_runs(["r1"])
    _install_fake_provider(monkeypatch, df)
    from statflow.loggers.runs_cache import RunsCache

    RunsCache.load_runs(["exp1"])
    assert not RunsCache.get_runs().is_empty()
    RunsCache.clear_cache()
    assert RunsCache.get_runs().is_empty()


def test_derived_params_refreshed_after_load(patched_cache, monkeypatch):
    """available_params and available_metrics are derived from the loaded DataFrame."""
    df = _make_runs(["r1", "r2"])
    _install_fake_provider(monkeypatch, df)
    from statflow.loggers.runs_cache import RunsCache

    RunsCache.load_runs(["exp1"])
    assert "lr" in patched_cache.get("available_params", [])
    assert "loss" in patched_cache.get("available_metrics", [])


def test_filter_by_datasets(patched_cache, monkeypatch):
    """filter_by_datasets filters rows by a params.* column."""
    df = pl.DataFrame(
        {
            "run_id": ["r1", "r2", "r3"],
            "start_time": [None, None, None],
            "params.dataset": ["ds_a", "ds_b", "ds_a"],
            "metrics.loss": [0.1, 0.2, 0.3],
        }
    )
    _install_fake_provider(monkeypatch, df)
    from statflow.loggers.runs_cache import RunsCache

    RunsCache.load_runs(["exp1"])
    filtered = RunsCache.filter_by_datasets("dataset", ["ds_a"])
    assert len(filtered) == 2
    assert all(v == "ds_a" for v in filtered.get_column("params.dataset").to_list())


def test_load_runs_empty_experiments(patched_cache, monkeypatch):
    """load_runs with no experiments returns empty DataFrame."""
    _install_fake_provider(monkeypatch, pl.DataFrame())
    from statflow.loggers.runs_cache import RunsCache

    result = RunsCache.load_runs([])
    assert result.is_empty()
