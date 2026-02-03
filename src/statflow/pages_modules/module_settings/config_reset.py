"""
Configuration reset logic for settings management.

This module handles resetting configuration to default values.

config_reset.py
├── reset_to_defaults()  # Resets config to default values.
└── Configuration reset logic
"""

from statflow.config import save_config, get_default_config


def reset_to_defaults() -> None:
    """Reset configuration to defaults."""
    defaults = get_default_config()
    save_config(defaults)