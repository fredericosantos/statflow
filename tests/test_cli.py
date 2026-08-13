"""
Tests for the `statflow` console-script entry point.

Guards the two things that make `uv tool install` work: the packaged `app.py`
is resolved next to the package (not the caller's cwd), and the subpages sit
beside it so the relative `st.Page(...)` paths resolve from an installed copy.
"""

from __future__ import annotations

import os
import sys

from statflow.cli import APP, DEFAULTS, main


def test_app_path_is_packaged_next_to_subpages():
    """app.py ships inside the package, with subpages/ beside it."""
    assert APP.name == "app.py"
    assert APP.is_file(), f"{APP} should exist inside the package"
    # Streamlit resolves st.Page paths relative to the main script's directory.
    for page in ("get_started", "parameters", "metrics", "results", "comparison"):
        assert (APP.parent / "subpages" / f"{page}.py").is_file()


def test_main_invokes_streamlit_run_with_packaged_app(monkeypatch):
    """main() delegates to Streamlit's CLI as `streamlit run <package>/app.py`."""
    called: list[list[str]] = []
    monkeypatch.setattr(
        "streamlit.web.cli.main", lambda *a, **kw: called.append(list(sys.argv)), raising=True
    )
    monkeypatch.setattr(sys, "argv", ["statflow"])

    main()

    assert called, "streamlit CLI was not invoked"
    assert called[0][:2] == ["streamlit", "run"]
    assert called[0][2] == str(APP)


def test_main_forwards_extra_args(monkeypatch):
    """Extra CLI args (ports, address, ...) are passed through to Streamlit."""
    called: list[list[str]] = []
    monkeypatch.setattr(
        "streamlit.web.cli.main", lambda *a, **kw: called.append(list(sys.argv)), raising=True
    )
    monkeypatch.setattr(sys, "argv", ["statflow", "--server.port", "9999"])

    main()

    assert called[0][-2:] == ["--server.port", "9999"]


def test_bare_command_defaults_to_own_port_and_loopback(monkeypatch):
    """Bare `statflow` must not inherit Streamlit's 8501/all-interfaces defaults."""
    monkeypatch.setattr("streamlit.web.cli.main", lambda *a, **kw: None, raising=True)
    monkeypatch.setattr(sys, "argv", ["statflow"])
    for key in DEFAULTS:
        monkeypatch.delenv(key, raising=False)

    main()

    assert os.environ["STREAMLIT_SERVER_PORT"] == "8513"
    # No auth in the app: loopback only, never a public/LAN interface by default.
    assert os.environ["STREAMLIT_SERVER_ADDRESS"] == "127.0.0.1"
    assert os.environ["STREAMLIT_SERVER_HEADLESS"] == "true"
    assert os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"


def test_preset_env_is_not_overridden(monkeypatch):
    """A user-set env var wins over our default (setdefault, not assignment)."""
    monkeypatch.setattr("streamlit.web.cli.main", lambda *a, **kw: None, raising=True)
    monkeypatch.setattr(sys, "argv", ["statflow"])
    monkeypatch.setenv("STREAMLIT_SERVER_PORT", "1234")
    monkeypatch.setenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

    main()

    assert os.environ["STREAMLIT_SERVER_PORT"] == "1234"
    assert os.environ["STREAMLIT_SERVER_ADDRESS"] == "0.0.0.0"
