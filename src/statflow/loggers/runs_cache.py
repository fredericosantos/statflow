"""
Centralized, provider-agnostic store for run data with Polars caching.

A single point of access for fetching, caching, and querying runs. Fetching is
delegated to the active RunProvider (MLflow, W&B, ...) selected by the
`provider` session-state key; RunsCache owns merging, dedup, pagination cursors,
and the derived `available_params` / `available_metrics`. It never touches a
backend API directly — every provider returns the same canonical wide DataFrame
(see `loggers/base.py`).

runs_cache.py
├── RunsCache                     # Main cache manager class
│   ├── load_runs()              # Initial load of runs
│   ├── load_more_runs()         # Incremental fetch using pagination cursors
│   ├── get_runs()               # Get cached DataFrame
│   ├── get_available_params()   # Extract param column names
│   ├── get_available_metrics()  # Extract metric column names
│   ├── get_param_values()       # Get unique values for a param
│   ├── filter_by_datasets()     # Filter runs by dataset param values
│   └── clear_cache()            # Clear the cache
"""

import polars as pl
import streamlit as st

from statflow.loggers.base import METRIC_PREFIX, PARAM_PREFIX, RUN_ID_COL
from statflow.loggers.registry import get_provider


class RunsCache:
    """Provider-agnostic store for run data with Polars caching."""

    CACHE_KEY = "_runs_cache_df"
    EXPERIMENTS_KEY = "_runs_cache_experiments"
    CURSORS_KEY = "_runs_cache_cursors"  # Per-experiment pagination cursors

    @classmethod
    def load_runs(
        cls,
        experiments: list[str],
        max_results: int = 2000,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """Load initial runs for selected experiments into session state cache.

        Args:
            experiments: List of experiment names to fetch runs for.
            max_results: Maximum number of runs to fetch per experiment.
            force_refresh: If True, clear cache and reload.

        Returns:
            Polars DataFrame containing all runs.
        """
        if not experiments:
            cls.clear_cache()
            return pl.DataFrame()

        cached_experiments = st.session_state.get(cls.EXPERIMENTS_KEY, [])

        # If experiments changed or forcing refresh, reset cache
        if set(cached_experiments) != set(experiments) or force_refresh:
            cls.clear_cache()
            st.session_state[cls.EXPERIMENTS_KEY] = list(experiments)
            st.session_state[cls.CURSORS_KEY] = {}

        # If already have data, return it (re-deriving params/metrics in case the
        # session lost them, e.g. after a provider switch or a fresh page load).
        if cls.CACHE_KEY in st.session_state and not st.session_state[cls.CACHE_KEY].is_empty():
            df = st.session_state[cls.CACHE_KEY]
            cls._refresh_derived(df)
            return df

        # Initial fetch
        return cls._fetch_runs(experiments, max_results)

    @classmethod
    def load_more_runs(
        cls,
        experiments: list[str],
        max_results: int = 2000,
    ) -> int:
        """Load more runs using timestamp-based pagination.

        Args:
            experiments: List of experiment names to fetch runs for.
            max_results: Maximum number of runs to fetch per experiment.

        Returns:
            Number of new runs added.
        """
        if not experiments:
            return 0

        initial_count = len(cls.get_runs()) if not cls.get_runs().is_empty() else 0
        cls._fetch_runs(experiments, max_results)
        new_count = len(cls.get_runs()) - initial_count
        return new_count

    @classmethod
    def _fetch_runs(
        cls,
        experiments: list[str],
        max_results: int,
    ) -> pl.DataFrame:
        """Internal: fetch a page of runs via the active provider, then merge."""
        provider = get_provider(st.session_state["provider"])
        cursors = st.session_state.get(cls.CURSORS_KEY, {})

        new_df, cursors = provider.fetch_runs(experiments, max_results, cursors)
        st.session_state[cls.CURSORS_KEY] = cursors

        if new_df.is_empty():
            if cls.CACHE_KEY not in st.session_state:
                st.session_state[cls.CACHE_KEY] = pl.DataFrame()
            return st.session_state[cls.CACHE_KEY]

        # Merge with existing cache
        existing_df = st.session_state.get(cls.CACHE_KEY, pl.DataFrame())
        if existing_df.is_empty():
            combined_df = new_df
        else:
            combined_df = pl.concat([existing_df, new_df], how="align")

        # Deduplicate by run_id to avoid duplicates
        if RUN_ID_COL in combined_df.columns:
            combined_df = combined_df.unique(subset=[RUN_ID_COL], keep="first")

        st.session_state[cls.CACHE_KEY] = combined_df
        cls._refresh_derived(combined_df)

        return combined_df

    @classmethod
    def _refresh_derived(cls, df: pl.DataFrame) -> None:
        """Recompute the session's available_params / available_metrics from `df`."""
        st.session_state["available_params"] = cls._extract_params(df)
        st.session_state["available_metrics"] = cls._extract_metrics(df)

    @classmethod
    def get_runs(cls) -> pl.DataFrame:
        """Get the cached runs DataFrame.

        Returns:
            Cached Polars DataFrame, or empty DataFrame if not cached.
        """
        return st.session_state.get(cls.CACHE_KEY, pl.DataFrame())

    @classmethod
    def get_available_params(cls) -> list[str]:
        """Get available parameter names from cached runs.

        Returns:
            List of parameter names (without 'params.' prefix).
        """
        return st.session_state.get("available_params", [])

    @classmethod
    def get_available_metrics(cls) -> list[str]:
        """Get available metric names from cached runs.

        Returns:
            List of metric names (without 'metrics.' prefix).
        """
        return st.session_state.get("available_metrics", [])

    @classmethod
    def get_param_values(cls, param: str) -> list:
        """Get unique values for a parameter from cached runs.

        Args:
            param: Parameter name (without 'params.' prefix).

        Returns:
            List of unique values for the parameter.
        """
        df = cls.get_runs()
        if df.is_empty():
            return []

        col_name = f"{PARAM_PREFIX}{param}"
        if col_name not in df.columns:
            return []

        values = df.get_column(col_name).drop_nulls().unique().to_list()
        return [v for v in values if v and str(v).strip()]

    @classmethod
    def filter_by_datasets(cls, dataset_param: str, datasets: list[str]) -> pl.DataFrame:
        """Filter cached runs by dataset parameter values.

        Args:
            dataset_param: Parameter name that defines datasets.
            datasets: List of dataset values to filter for.

        Returns:
            Filtered Polars DataFrame.
        """
        df = cls.get_runs()
        if df.is_empty() or not datasets:
            return df

        col_name = f"{PARAM_PREFIX}{dataset_param}"
        if col_name not in df.columns:
            return df

        return df.filter(pl.col(col_name).is_in(datasets))

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the runs cache from session state."""
        for key in [cls.CACHE_KEY, cls.EXPERIMENTS_KEY, cls.CURSORS_KEY]:
            if key in st.session_state:
                del st.session_state[key]

    @classmethod
    def _extract_params(cls, df: pl.DataFrame) -> list[str]:
        """Extract parameter names from DataFrame columns."""
        return sorted(
            [col[len(PARAM_PREFIX) :] for col in df.columns if col.startswith(PARAM_PREFIX)]
        )

    @classmethod
    def _extract_metrics(cls, df: pl.DataFrame) -> list[str]:
        """Extract metric names from DataFrame columns."""
        return sorted(
            [col[len(METRIC_PREFIX) :] for col in df.columns if col.startswith(METRIC_PREFIX)]
        )

    @classmethod
    def get_run_count(cls) -> int:
        """Get total number of cached runs."""
        df = cls.get_runs()
        return len(df) if not df.is_empty() else 0
