"""
Console-script entry point, so Statflow is installable with `uv tool install`.

Streamlit apps are launched by `streamlit run <script>` rather than by importing
a module, so this shim hands off to Streamlit's own CLI with the packaged
``app.py``. Resolving that path next to this file (instead of the caller's cwd)
is what lets the ``statflow`` command run from any directory, and is what keeps
the ``st.Page("subpages/...")`` paths in ``app.py`` — which Streamlit resolves
relative to the main script — working from an installed copy.

Extra arguments are forwarded to Streamlit, e.g.::

    statflow --server.port 8513 --server.address 127.0.0.1 --server.headless true

cli.py
└── main()   # `statflow` console script -> streamlit run <package>/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).parent / "app.py"


def main() -> None:
    """Run the Statflow Streamlit app, forwarding extra CLI args to Streamlit."""
    from streamlit.web.cli import main as streamlit_main

    sys.argv = ["streamlit", "run", str(APP), *sys.argv[1:]]
    streamlit_main()
