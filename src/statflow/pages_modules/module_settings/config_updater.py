"""
Configuration update logic for settings management.

This module handles updating and persisting configuration settings.

config_updater.py
├── update_config()  # Updates configuration settings.
└── Configuration update and persistence logic
"""

def update_config(updates: dict[str, Any]) -> None:
    """Update configuration with new values.

    Args:
        updates: Dict of config updates.
    """
    # Load current config, update, save
    config = load_user_config()
    config.update(updates)
    save_user_config(config)