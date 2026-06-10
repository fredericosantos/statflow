"""
Weights & Biases implementation of the RunProvider interface.

Queries the W&B public GraphQL API (https://api.wandb.ai/graphql) directly with
the API key from ``~/.netrc`` — no dependency on the ``wandb`` client, so it is
unaffected by its local-service auth layer. W&B's data model maps onto the
canonical schema as:

  - project            -> experiment (selectable container)
  - run.name           -> run_id
  - run.createdAt      -> start_time
  - run.config[k].value-> params.<k>   (config entries are {"value", "desc"})
  - numeric summary[k] -> metrics.<k>  (skips "_"-prefixed system keys)

A few useful W&B run attributes (display name, group, state) are also surfaced
as params so they can be used as grouping handles; real config keys win on any
name clash.

provider.py
├── _api_key() / _graphql_url()   # netrc/env-based auth + endpoint
├── _config_value / _is_number / _stringify / _parse_dt   # field mappers
└── WandbProvider                 # RunProvider over the W&B GraphQL API
    ├── check_status()            # viewer auth probe
    ├── list_experiments()        # projects under the active entity
    └── fetch_runs()              # cursor-paginated runs -> canonical wide df
"""

from __future__ import annotations

import json
import netrc
import os
from datetime import datetime
from typing import Any

import polars as pl
import requests
import streamlit as st

from statflow.loggers.base import RunProvider
from statflow.loggers.registry import register_provider

_WANDB_HOST = "api.wandb.ai"
_DEFAULT_URL = f"https://{_WANDB_HOST}/graphql"
_PAGE_CAP = 500  # W&B caps `first` per page; loop pages to reach max_results.

_VIEWER_Q = "query { viewer { username entity } }"

_PROJECTS_Q = """
query Projects($entity: String!, $first: Int!) {
  projects(entityName: $entity, first: $first) {
    edges { node { name } }
  }
}
"""

_RUNS_Q = """
query Runs($entity: String!, $project: String!, $first: Int!, $after: String) {
  project(name: $project, entityName: $entity) {
    runs(first: $first, after: $after, order: "-createdAt") {
      edges {
        cursor
        node { name displayName createdAt state group config summaryMetrics }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

_DONE = "__DONE__"  # cursor sentinel: project fully paginated.


def _api_key() -> str:
    """W&B API key from WANDB_API_KEY or ~/.netrc (machine api.wandb.ai)."""
    env = os.environ.get("WANDB_API_KEY")
    if env:
        return env
    try:
        auth = netrc.netrc().authenticators(_WANDB_HOST)
    except FileNotFoundError:
        auth = None
    if not auth or not auth[2]:
        raise RuntimeError(
            f"No W&B API key: set WANDB_API_KEY or add a '{_WANDB_HOST}' entry to ~/.netrc"
        )
    return auth[2]


def _graphql_url() -> str:
    """GraphQL endpoint; honors WANDB_BASE_URL for self-hosted instances."""
    base = os.environ.get("WANDB_BASE_URL")
    if base:
        return base.rstrip("/") + "/graphql"
    return _DEFAULT_URL


def _config_value(v: Any) -> Any:
    """Unwrap a W&B config entry ({"value": ..., "desc": ...}) to its value."""
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _stringify(v: Any) -> str | None:
    """Render a param value as a string (config values are heterogeneous)."""
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, default=str)
    return str(v)


def _parse_dt(s: str | None) -> datetime | None:
    """Parse W&B createdAt (ISO-8601, trailing Z) to naive UTC datetime."""
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


@st.cache_data(ttl=600, show_spinner=False)
def _project_names(entity: str) -> list[str]:
    """Project names for an entity (cached per entity)."""
    data = _execute(_PROJECTS_Q, {"entity": entity, "first": 500})
    edges = data["projects"]["edges"]
    return sorted(e["node"]["name"] for e in edges)


def _execute(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """POST a GraphQL query and return its `data`, raising on errors."""
    response = requests.post(
        _graphql_url(),
        json={"query": query, "variables": variables},
        auth=("api", _api_key()),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(f"W&B GraphQL error: {payload['errors']}")
    return payload["data"]


@register_provider
class WandbProvider(RunProvider):
    """RunProvider backed by the Weights & Biases public GraphQL API."""

    name = "wandb"
    label = "Weights & Biases"

    def _entity(self) -> str:
        """Active entity: configured `wandb_entity`, else the viewer's default."""
        configured = st.session_state["wandb_entity"]
        if configured:
            return configured
        viewer = _execute(_VIEWER_Q, {})["viewer"]
        return viewer["entity"] or viewer["username"]

    def check_status(self) -> bool:
        try:
            return bool(_execute(_VIEWER_Q, {})["viewer"])
        except (requests.RequestException, RuntimeError, OSError):
            return False

    def list_experiments(self) -> list[str]:
        return _project_names(self._entity())

    def fetch_runs(
        self,
        experiments: list[str],
        max_results: int,
        cursors: dict[str, Any] | None = None,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        cursors = dict(cursors or {})
        entity = self._entity()

        rows: list[dict[str, Any]] = []
        for project in experiments:
            after = cursors.get(project)
            if after == _DONE:
                continue

            collected = 0
            while collected < max_results:
                page_size = min(_PAGE_CAP, max_results - collected)
                try:
                    data = _execute(
                        _RUNS_Q,
                        {
                            "entity": entity,
                            "project": project,
                            "first": page_size,
                            "after": after,
                        },
                    )
                except (requests.RequestException, RuntimeError):
                    # External IO: skip a project that fails rather than abort.
                    break

                runs = data["project"]["runs"]
                for edge in runs["edges"]:
                    rows.append(self._map_run(edge["node"]))
                collected += len(runs["edges"])

                page_info = runs["pageInfo"]
                after = page_info["endCursor"]
                if not page_info["hasNextPage"]:
                    after = _DONE
                    break

            cursors[project] = after

        if not rows:
            return pl.DataFrame(), cursors

        return pl.from_dicts(rows, infer_schema_length=None), cursors

    @staticmethod
    def _map_run(node: dict[str, Any]) -> dict[str, Any]:
        """Map a W&B run node onto a canonical-schema row."""
        row: dict[str, Any] = {
            "run_id": node["name"],
            "start_time": _parse_dt(node.get("createdAt")),
            # Useful W&B attributes as grouping handles (config keys override).
            "params.run_name": node.get("displayName"),
            "params.group": node.get("group"),
            "params.state": node.get("state"),
        }

        config = json.loads(node["config"]) if node.get("config") else {}
        for key, raw in config.items():
            if key.startswith("_"):
                continue
            row[f"params.{key}"] = _stringify(_config_value(raw))

        summary = json.loads(node["summaryMetrics"]) if node.get("summaryMetrics") else {}
        for key, value in summary.items():
            if key.startswith("_"):
                continue
            if _is_number(value):
                row[f"metrics.{key}"] = float(value)

        return row
