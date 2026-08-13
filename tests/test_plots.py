"""
Tests for the Plots page's group-label construction.

The Plots page must keep two labels in lock-step with the Parameters page:
  - a FULL label (every selected param) used to match `selected_groups`,
  - a REDUCED label (minus the x-axis param) used as the per-line grouping key.
Regression guard for the bug where the reduced label was matched against
`selected_groups`, emptying the data ("No data available for the selected
combination.").
"""

from __future__ import annotations

import polars as pl

from statflow.subpages.plots import _build_plot_group_label, _group_label_expr


def _parmeters_page_label(df: pl.DataFrame, params: list[str]) -> list[str]:
    """Replicate the Parameters page's group_label for the same param set."""
    return df.with_columns(_group_label_expr(params).alias("g"))["g"].to_list()


def test_full_label_matches_parameters_page_and_reduced_excludes_x():
    df = pl.DataFrame(
        {
            "run_id": ["r1", "r2", "r3"],
            "pop_size": ["100", "200", "100"],
            "method": ["A", "A", "B"],
        }
    )
    selected_params = ["pop_size", "method"]

    out = _build_plot_group_label(df, selected_params, x_param="pop_size")

    # Full label == what the Parameters page produces from the same params, so it
    # can be matched against selected_groups.
    assert out["_full_group_label"].to_list() == _parmeters_page_label(df, selected_params)
    assert out["_full_group_label"].to_list() == [
        "pop_size=100, method=A",
        "pop_size=200, method=A",
        "pop_size=100, method=B",
    ]

    # Reduced label drops the x param so runs differing only in x collapse to one line.
    assert out["group_label"].to_list() == ["method=A", "method=A", "method=B"]


def test_full_label_is_in_selected_groups_but_reduced_is_not():
    """The exact failure the fix addresses: filtering must use the full label."""
    df = pl.DataFrame({"run_id": ["r1"], "pop_size": ["100"], "method": ["A"]})
    out = _build_plot_group_label(df, ["pop_size", "method"], x_param="pop_size")

    selected_groups = ["pop_size=100, method=A"]  # as built by the Parameters page
    assert out["_full_group_label"].is_in(selected_groups).all()  # match -> data kept
    assert not out["group_label"].is_in(selected_groups).any()  # old code -> emptied


def test_single_param_chosen_as_x_gives_default_reduced_label():
    df = pl.DataFrame({"run_id": ["r1", "r2"], "pop_size": ["100", "200"]})
    out = _build_plot_group_label(df, ["pop_size"], x_param="pop_size")

    assert out["_full_group_label"].to_list() == ["pop_size=100", "pop_size=200"]
    assert out["group_label"].to_list() == ["Default", "Default"]
