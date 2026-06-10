"""
Tests for src/statflow/functional/statistics.py.

Covers:
- holm_bonferroni_correction: ordering and adjusted threshold correctness
- perform_statistical_tests: direction handling (maximize vs minimize)
- build_comparison_table: normal path and empty-input 3-tuple path
"""

import numpy as np
import polars as pl
import pytest

from statflow.functional.statistics import (
    build_comparison_table,
    check_significance,
    holm_bonferroni_correction,
    perform_statistical_tests,
)


# ---------------------------------------------------------------------------
# holm_bonferroni_correction
# ---------------------------------------------------------------------------


def test_holm_bonferroni_empty():
    assert holm_bonferroni_correction([]) == []


def test_holm_bonferroni_single_rejected():
    # Single p-value below alpha → rejected
    result = holm_bonferroni_correction([0.01], alpha=0.05)
    assert result == [True]


def test_holm_bonferroni_single_not_rejected():
    result = holm_bonferroni_correction([0.06], alpha=0.05)
    assert result == [False]


def test_holm_bonferroni_three_values_hand_computed():
    # Three p-values: 0.01, 0.04, 0.08  (alpha=0.05)
    # Sorted ascending: 0.01, 0.04, 0.08
    # Step 0: threshold = 0.05/3 = 0.0167 → 0.01 <= 0.0167 → reject (orig idx 0)
    # Step 1: threshold = 0.05/2 = 0.025  → 0.04 > 0.025 → stop
    # orig idx 0 → True, orig idx 1 → False, orig idx 2 → False
    result = holm_bonferroni_correction([0.01, 0.04, 0.08], alpha=0.05)
    assert result == [True, False, False]


def test_holm_bonferroni_all_rejected():
    # Very small p-values; all should be rejected
    result = holm_bonferroni_correction([0.001, 0.002, 0.003], alpha=0.05)
    # Sorted: 0.001, 0.002, 0.003
    # threshold[0] = 0.05/3 ≈ 0.0167 → 0.001 ≤ 0.0167 → reject
    # threshold[1] = 0.05/2 = 0.025  → 0.002 ≤ 0.025  → reject
    # threshold[2] = 0.05/1 = 0.05   → 0.003 ≤ 0.05   → reject
    assert result == [True, True, True]


def test_holm_bonferroni_preserves_input_order():
    # Input order [0.03, 0.01, 0.08]; sorted: 0.01, 0.03, 0.08
    # Step 0: 0.05/3 ≈ 0.0167 → 0.01 ≤ 0.0167 → reject orig idx=1
    # Step 1: 0.05/2 = 0.025  → 0.03 > 0.025  → stop
    result = holm_bonferroni_correction([0.03, 0.01, 0.08], alpha=0.05)
    assert result[0] is False
    assert result[1] is True
    assert result[2] is False


# ---------------------------------------------------------------------------
# perform_statistical_tests — direction handling
# ---------------------------------------------------------------------------


def _make_df(our_vals: list[float], their_vals: list[float]) -> pl.DataFrame:
    """Build a minimal DataFrame with group_label and score columns."""
    return pl.DataFrame(
        {
            "group_label": ["ours"] * len(our_vals) + ["theirs"] * len(their_vals),
            "score": our_vals + their_vals,
        }
    )


def test_perform_statistical_tests_minimize_ours_better():
    """minimize=True (maximize=False): ours has lower values → our_is_better=True."""
    df = _make_df(
        our_vals=[0.1, 0.2, 0.15, 0.12, 0.18],
        their_vals=[0.5, 0.6, 0.55, 0.52, 0.58],
    )
    result = perform_statistical_tests(df, "ours", ["theirs"], "score", maximize=False)
    assert "theirs" in result
    assert result["theirs"]["our_is_better"] is True
    assert result["theirs"]["is_significant"] is True


def test_perform_statistical_tests_minimize_ours_worse():
    """minimize=True: ours has higher values → our_is_better=False."""
    df = _make_df(
        our_vals=[0.9, 0.8, 0.85, 0.88, 0.82],
        their_vals=[0.1, 0.2, 0.15, 0.12, 0.18],
    )
    result = perform_statistical_tests(df, "ours", ["theirs"], "score", maximize=False)
    assert "theirs" in result
    assert result["theirs"]["our_is_better"] is False


def test_perform_statistical_tests_maximize_ours_better():
    """maximize=True: ours has higher values → our_is_better=True."""
    df = _make_df(
        our_vals=[0.9, 0.8, 0.85, 0.88, 0.82],
        their_vals=[0.1, 0.2, 0.15, 0.12, 0.18],
    )
    result = perform_statistical_tests(df, "ours", ["theirs"], "score", maximize=True)
    assert "theirs" in result
    assert result["theirs"]["our_is_better"] is True
    assert result["theirs"]["is_significant"] is True


def test_perform_statistical_tests_maximize_ours_worse():
    """maximize=True: ours has lower values → our_is_better=False."""
    df = _make_df(
        our_vals=[0.1, 0.2, 0.15, 0.12, 0.18],
        their_vals=[0.9, 0.8, 0.85, 0.88, 0.82],
    )
    result = perform_statistical_tests(df, "ours", ["theirs"], "score", maximize=True)
    assert "theirs" in result
    assert result["theirs"]["our_is_better"] is False


def test_perform_statistical_tests_empty_our_group():
    """Empty our_group → empty result dict."""
    df = _make_df(our_vals=[], their_vals=[0.5, 0.6])
    result = perform_statistical_tests(df, "ours", ["theirs"], "score")
    assert result == {}


def test_perform_statistical_tests_empty_their_group():
    """Empty competitor group → no entry for that group."""
    df = _make_df(our_vals=[0.1, 0.2], their_vals=[])
    result = perform_statistical_tests(df, "ours", ["theirs"], "score")
    assert result == {}


def test_perform_statistical_tests_multiple_competitors_holm():
    """Multiple competitors: Holm correction is applied across them."""
    rng = np.random.default_rng(42)
    our = rng.normal(0.1, 0.05, 50).tolist()
    b1  = rng.normal(0.9, 0.05, 50).tolist()
    b2  = rng.normal(0.95, 0.05, 50).tolist()

    df = pl.DataFrame({
        "group_label": ["ours"] * 50 + ["b1"] * 50 + ["b2"] * 50,
        "score": our + b1 + b2,
    })
    result = perform_statistical_tests(df, "ours", ["b1", "b2"], "score", maximize=False)
    assert "b1" in result
    assert "b2" in result
    # Both should be rejected given the large separation
    assert result["b1"]["is_significant"] is True
    assert result["b2"]["is_significant"] is True


def test_perform_statistical_tests_result_keys():
    """Result dict has the expected keys for each competitor."""
    df = _make_df(
        our_vals=[0.1, 0.2, 0.15],
        their_vals=[0.5, 0.6, 0.55],
    )
    result = perform_statistical_tests(df, "ours", ["theirs"], "score")
    entry = result["theirs"]
    for k in ("p_value", "our_median", "their_median", "our_is_better", "is_significant"):
        assert k in entry, f"Missing key: {k}"


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------


def _make_comparison_dfs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (metric_df, param_df) suitable for build_comparison_table."""
    param_df = pl.DataFrame({
        "run_id": [f"r{i}" for i in range(12)],
        "dataset_name": ["ds1"] * 6 + ["ds2"] * 6,
        "group_label": ["ours", "ours", "ours", "theirs", "theirs", "theirs"] * 2,
    })
    rng = np.random.default_rng(0)
    metric_df = pl.DataFrame({
        "run_id": [f"r{i}" for i in range(12)],
        "accuracy": rng.uniform(0.1, 0.9, 12).tolist(),
    })
    return metric_df, param_df


def test_build_comparison_table_returns_3_tuple():
    metric_df, param_df = _make_comparison_dfs()
    result = build_comparison_table(
        metric_df, param_df, "accuracy", "Mean ± Std", ["ours"], ["theirs"]
    )
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_build_comparison_table_agg_df_shape():
    metric_df, param_df = _make_comparison_dfs()
    agg_df, stats_results, combined = build_comparison_table(
        metric_df, param_df, "accuracy", "Mean ± Std", ["ours"], ["theirs"]
    )
    assert not agg_df.is_empty()
    assert "dataset_name" in agg_df.columns
    assert "group_label" in agg_df.columns
    assert "value" in agg_df.columns
    assert "spread" in agg_df.columns


def test_build_comparison_table_median_iqr():
    metric_df, param_df = _make_comparison_dfs()
    agg_df, _, _ = build_comparison_table(
        metric_df, param_df, "accuracy", "Median ± IQR", ["ours"], ["theirs"]
    )
    assert not agg_df.is_empty()
    assert "value" in agg_df.columns


def test_build_comparison_table_empty_metric_df():
    """Empty metric_df → (empty, {}, empty) 3-tuple."""
    _, param_df = _make_comparison_dfs()
    agg_df, stats_results, combined = build_comparison_table(
        pl.DataFrame(), param_df, "accuracy", "Mean ± Std", ["ours"], ["theirs"]
    )
    assert agg_df.is_empty()
    assert stats_results == {}
    assert combined.is_empty()


def test_build_comparison_table_empty_param_df():
    """When joined result is empty → returns the empty 3-tuple."""
    metric_df, _ = _make_comparison_dfs()
    agg_df, stats_results, combined = build_comparison_table(
        metric_df, pl.DataFrame(), "accuracy", "Mean ± Std", ["ours"], ["theirs"]
    )
    # join with empty param_df → combined is empty
    assert agg_df.is_empty()
    assert stats_results == {}
    assert combined.is_empty()


def test_build_comparison_table_stats_results_populated():
    """stats_results should contain per-dataset significance data."""
    metric_df, param_df = _make_comparison_dfs()
    _, stats_results, _ = build_comparison_table(
        metric_df, param_df, "accuracy", "Mean ± Std", ["ours"], ["theirs"]
    )
    assert len(stats_results) > 0
    # Each key is "{dataset}_{our_group}"
    for key in stats_results:
        assert "_" in key


def test_check_significance_no_competitors():
    df = _make_df(our_vals=[0.1, 0.2], their_vals=[0.5, 0.6])
    assert check_significance(df, "ours", [], "score") is False
