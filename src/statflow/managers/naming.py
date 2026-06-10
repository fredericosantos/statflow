"""
Centralized manager for resolving display names for application entities.

This module provides a unified way to retrieve user-defined display names
for datasets, metrics, and parameter groups from the session state.

naming.py
├── NamingManager
│   ├── get_dataset_name()
│   ├── get_metric_name()
│   ├── get_group_name()
│   └── get_name()
"""

from statflow.config import DEFAULT_STATE, SessionState


class NamingManager:
    """Manager for resolving display names."""

    DATASET_RENAMES_KEY = "dataset_renames"
    METRIC_RENAMES_KEY = "metric_renames"
    GROUP_RENAMES_KEY = "group_renames"

    @classmethod
    def get_name(cls, original: str, rename_key: str) -> str:
        """Generic method to get display name from a session state key."""
        # Use default from config as fallback
        default_renames = DEFAULT_STATE.get(rename_key, {})
        renames = SessionState.get(rename_key, default_renames)
        entry = renames.get(original)

        if entry is None:
            return original

        if isinstance(entry, dict):
            return entry.get("display_name", original)
        elif isinstance(entry, str):
            return entry

        return original

    @classmethod
    def get_dataset_name(cls, original: str) -> str:
        """Get display name for a dataset."""
        return cls.get_name(original, cls.DATASET_RENAMES_KEY)

    @classmethod
    def get_metric_name(cls, original: str) -> str:
        """Get display name for a metric."""
        return cls.get_name(original, cls.METRIC_RENAMES_KEY)

    @classmethod
    def get_group_name(cls, original: str) -> str:
        """Get display name for a parameter group."""
        return cls.get_name(original, cls.GROUP_RENAMES_KEY)
