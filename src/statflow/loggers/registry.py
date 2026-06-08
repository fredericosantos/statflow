"""
Registry / factory for run providers.

Providers register themselves with the ``@register_provider`` decorator at
import time; ``get_provider`` resolves the active one by name. Known provider
modules are imported lazily on first lookup so the registry is populated without
forcing every backend's heavy dependencies to load up front.

registry.py
├── register_provider()    # decorator: register a RunProvider subclass
├── available_providers()  # list registered provider names
└── get_provider()         # resolve the active provider instance by name
"""

from __future__ import annotations

import importlib

from statflow.loggers.base import RunProvider

# Provider name -> module that defines (and self-registers) it. Imported lazily
# so e.g. selecting MLflow never imports wandb and vice versa.
_PROVIDER_MODULES: dict[str, str] = {
    "mlflow": "statflow.loggers.mlflow.provider",
    "wandb": "statflow.loggers.wandb.provider",
}

_REGISTRY: dict[str, type[RunProvider]] = {}


def register_provider(cls: type[RunProvider]) -> type[RunProvider]:
    """Class decorator that registers a ``RunProvider`` under its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`")
    _REGISTRY[cls.name] = cls
    return cls


def available_providers() -> list[str]:
    """Names of providers known to the registry (registered or importable)."""
    return sorted(set(_REGISTRY) | set(_PROVIDER_MODULES))


def get_provider(name: str) -> RunProvider:
    """Return a fresh instance of the named provider.

    Imports the provider's module on demand to trigger registration.
    """
    if name not in _REGISTRY:
        module = _PROVIDER_MODULES.get(name)
        if module is None:
            raise KeyError(
                f"Unknown provider {name!r}; available: {available_providers()}"
            )
        importlib.import_module(module)

    if name not in _REGISTRY:
        raise KeyError(f"Provider {name!r} did not register itself on import")

    return _REGISTRY[name]()
