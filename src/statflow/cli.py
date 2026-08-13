"""
Console-script entry point, so Statflow is installable with `uv tool install`.

Streamlit apps are launched by `streamlit run <script>` rather than by importing
a module, so this shim hands off to Streamlit's own CLI with the packaged
``app.py``. Resolving that path next to this file (instead of the caller's cwd)
is what lets the ``statflow`` command run from any directory, and is what keeps
the ``st.Page("subpages/...")`` paths in ``app.py`` — which Streamlit resolves
relative to the main script — working from an installed copy.

Bare ``statflow`` should be correct on its own, so the defaults below replace
Streamlit's (which would bind every interface on port 8501). They are applied as
environment variables rather than injected argv, which keeps Streamlit's own
precedence intact: an explicit CLI flag outranks an env var, and ``setdefault``
leaves an env var the user already set alone. So all three of these work::

    statflow                              # 127.0.0.1:8513
    statflow --server.port 9999           # flag wins
    STREAMLIT_SERVER_PORT=9999 statflow   # pre-set env wins

cli.py
├── DEFAULTS   # env defaults: own port, loopback-only, headless, no telemetry
└── main()     # `statflow` console script -> streamlit run <package>/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP = Path(__file__).parent / "app.py"

# Statflow's own port, distinct from Streamlit's 8501 so it never collides with
# another Streamlit app on the same host. Loopback-only because Statflow has no
# authentication of its own: put a reverse proxy in front rather than binding a
# public interface. Override with --server.address to expose it deliberately.
DEFAULTS = {
    "STREAMLIT_SERVER_PORT": "8513",
    "STREAMLIT_SERVER_ADDRESS": "127.0.0.1",
    "STREAMLIT_SERVER_HEADLESS": "true",
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
}


def main() -> None:
    """Run the Statflow Streamlit app, forwarding extra CLI args to Streamlit."""
    from streamlit.web.cli import main as streamlit_main

    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)

    sys.argv = ["streamlit", "run", str(APP), *sys.argv[1:]]
    streamlit_main()
