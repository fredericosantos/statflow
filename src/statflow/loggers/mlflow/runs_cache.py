"""
Centralized manager for MLflow runs data with Polars caching.

This module provides a single point of access for fetching, caching,
and querying MLflow run data. Supports incremental loading with
timestamp-based pagination.

runs_cache.py
├── RunsCache                     # Main cache manager class
│   ├── load_runs()              # Initial load of runs
│   ├── load_more_runs()         # Incremental fetch using timestamp pagination
│   ├── get_runs()               # Get cached DataFrame
│   ├── get_available_params()   # Extract param column names
│   ├── get_available_metrics()  # Extract metric column names
│   ├── get_param_values()       # Get unique values for a param
│   ├── filter_by_datasets()     # Filter runs by dataset param values
│   └── clear_cache()            # Clear the cache
"""

import polars as pl
import streamlit as st
import mlflow


class RunsCache:
    """Centralized manager for MLflow runs data with Polars caching."""

    CACHE_KEY = "_runs_cache_df"
    EXPERIMENTS_KEY = "_runs_cache_experiments"
    TIMESTAMPS_KEY = "_runs_cache_timestamps"  # Per-experiment last timestamp

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
            st.session_state[cls.TIMESTAMPS_KEY] = {}

        # If already have data, return it
        if (
            cls.CACHE_KEY in st.session_state
            and not st.session_state[cls.CACHE_KEY].is_empty()
        ):
            return st.session_state[cls.CACHE_KEY]

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
        """Internal: Fetch runs from MLflow with pagination."""
        mlflow.set_tracking_uri(
            st.session_state.get("mlflow_server_url", "http://0.0.0.0:5000")
        )

        timestamps = st.session_state.get(cls.TIMESTAMPS_KEY, {})
        all_new_runs = []

        for exp_name in experiments:
            filter_string = ""
            if exp_name in timestamps:
                last_time = timestamps[exp_name]
                filter_string = f"attributes.start_time < {last_time}"

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

                if runs_pdf is not None and not runs_pdf.empty:
                    runs_df = pl.from_pandas(runs_pdf)
                    all_new_runs.append(runs_df)

                    # Update timestamp cursor for this experiment
                    if "start_time" in runs_df.columns:
                        min_val = runs_df.get_column("start_time").min()
                        if min_val is not None:
                            timestamps[exp_name] = int(min_val.timestamp() * 1000)

            except Exception:
                continue

        st.session_state[cls.TIMESTAMPS_KEY] = timestamps

        if not all_new_runs:
            if cls.CACHE_KEY not in st.session_state:
                st.session_state[cls.CACHE_KEY] = pl.DataFrame()
            return st.session_state[cls.CACHE_KEY]

        # Combine new runs - use align for schema differences
        new_df = pl.concat(all_new_runs, how="align")

        # Merge with existing cache
        existing_df = st.session_state.get(cls.CACHE_KEY, pl.DataFrame())
        if existing_df.is_empty():
            combined_df = new_df
        else:
            combined_df = pl.concat([existing_df, new_df], how="align")

        # Deduplicate by run_id to avoid duplicates
        if "run_id" in combined_df.columns:
            combined_df = combined_df.unique(subset=["run_id"], keep="first")

        st.session_state[cls.CACHE_KEY] = combined_df

        # Update derived state
        st.session_state["available_params"] = cls._extract_params(combined_df)
        st.session_state["available_metrics"] = cls._extract_metrics(combined_df)

        return combined_df

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

        col_name = f"params.{param}"
        if col_name not in df.columns:
            return []

        values = df.get_column(col_name).drop_nulls().unique().to_list()
        return [v for v in values if v and str(v).strip()]

    @classmethod
    def filter_by_datasets(
        cls, dataset_param: str, datasets: list[str]
    ) -> pl.DataFrame:
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

        col_name = f"params.{dataset_param}"
        if col_name not in df.columns:
            return df

        return df.filter(pl.col(col_name).is_in(datasets))

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the runs cache from session state."""
        for key in [cls.CACHE_KEY, cls.EXPERIMENTS_KEY, cls.TIMESTAMPS_KEY]:
            if key in st.session_state:
                del st.session_state[key]

    @classmethod
    def _extract_params(cls, df: pl.DataFrame) -> list[str]:
        """Extract parameter names from DataFrame columns."""
        prefix = "params."
        return sorted([
            col[len(prefix) :] for col in df.columns if col.startswith(prefix)
        ])

    @classmethod
    def _extract_metrics(cls, df: pl.DataFrame) -> list[str]:
        """Extract metric names from DataFrame columns."""
        prefix = "metrics."
        return sorted([
            col[len(prefix) :] for col in df.columns if col.startswith(prefix)
        ])

    @classmethod
    def get_run_count(cls) -> int:
        """Get total number of cached runs."""
        df = cls.get_runs()
        return len(df) if not df.is_empty() else 0
