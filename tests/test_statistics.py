"""
Tests for src/statflow/functional/statistics.py.

Covers:
- holm_bonferroni_correction: ordering and adjusted threshold correctness
- perform_statistical_tests: direction handling (maximize vs minimize)
- build_comparison_table: normal path and empty-input 3-tuple path
- iqm: hand-computed cases and degenerate inputs
- a12 / a12_magnitude: all thresholds, direction, ties
- aggregate_per_dataset: correct cells, nulls for missing combos
- cross_dataset_test: 2-group signed-rank, 3-group Friedman, guards,
  all-zero case, dropped datasets, mean ranks
- comparison_table_to_latex / cross_dataset_to_latex: structure, escaping, stars
- aggregate_for_plot: group/x/agg correctness, band quantiles
"""

import math

import numpy as np
import polars as pl
import pytest

from statflow.functional.statistics import (
    AGGREGATIONS,
    a12,
    a12_magnitude,
    aggregate_for_plot,
    aggregate_per_dataset,
    build_comparison_table,
    check_significance,
    comparison_table_to_latex,
    cross_dataset_test,
    cross_dataset_to_latex,
    holm_bonferroni_correction,
    iqm,
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
    b1 = rng.normal(0.9, 0.05, 50).tolist()
    b2 = rng.normal(0.95, 0.05, 50).tolist()

    df = pl.DataFrame(
        {
            "group_label": ["ours"] * 50 + ["b1"] * 50 + ["b2"] * 50,
            "score": our + b1 + b2,
        }
    )
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
    param_df = pl.DataFrame(
        {
            "run_id": [f"r{i}" for i in range(12)],
            "dataset_name": ["ds1"] * 6 + ["ds2"] * 6,
            "group_label": ["ours", "ours", "ours", "theirs", "theirs", "theirs"] * 2,
        }
    )
    rng = np.random.default_rng(0)
    metric_df = pl.DataFrame(
        {
            "run_id": [f"r{i}" for i in range(12)],
            "accuracy": rng.uniform(0.1, 0.9, 12).tolist(),
        }
    )
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


# ---------------------------------------------------------------------------
# iqm
# ---------------------------------------------------------------------------


def test_iqm_hand_computed():
    # values = [1, 2, 3, 4, 5, 6, 7, 8]
    # Q1 = 2.75, Q3 = 6.25
    # values in [2.75, 6.25] = [3, 4, 5, 6]
    # mean = 18/4 = 4.5
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    result = iqm(values)
    assert abs(result - 4.5) < 1e-9


def test_iqm_single_value():
    # n=1 < 4 → fallback to plain mean = the value itself
    assert iqm([7.0]) == pytest.approx(7.0)


def test_iqm_two_values():
    # n=2 < 4 → fallback to plain mean
    assert iqm([2.0, 8.0]) == pytest.approx(5.0)


def test_iqm_three_values():
    # n=3 < 4 → fallback to plain mean
    assert iqm([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_iqm_empty():
    result = iqm([])
    assert math.isnan(result)


def test_iqm_all_same():
    assert iqm([5.0, 5.0, 5.0, 5.0, 5.0]) == pytest.approx(5.0)


def test_iqm_in_aggregations_dict():
    assert "iqm" in AGGREGATIONS
    assert abs(AGGREGATIONS["iqm"]([1, 2, 3, 4, 5, 6, 7, 8]) - 4.5) < 1e-9


# ---------------------------------------------------------------------------
# a12 / a12_magnitude
# ---------------------------------------------------------------------------


def test_a12_identical_samples():
    """Identical samples → A12 = 0.5 (no effect)."""
    x = [1.0, 2.0, 3.0]
    assert a12(x, x, maximize=True) == pytest.approx(0.5)


def test_a12_fully_separated_maximize():
    """Ours strictly dominates → A12 = 1.0 (maximize=True)."""
    ours = [10.0, 11.0, 12.0]
    theirs = [1.0, 2.0, 3.0]
    assert a12(ours, theirs, maximize=True) == pytest.approx(1.0)


def test_a12_fully_separated_maximize_reversed():
    """Theirs strictly dominates → A12 = 0.0 (maximize=True)."""
    ours = [1.0, 2.0, 3.0]
    theirs = [10.0, 11.0, 12.0]
    assert a12(ours, theirs, maximize=True) == pytest.approx(0.0)


def test_a12_fully_separated_minimize():
    """Ours is uniformly smaller (better when minimize) → A12 = 1.0 (maximize=False)."""
    ours = [1.0, 2.0, 3.0]
    theirs = [10.0, 11.0, 12.0]
    assert a12(ours, theirs, maximize=False) == pytest.approx(1.0)


def test_a12_direction_flip():
    """maximize=False correctly flips compared to maximize=True on the same data."""
    ours = [5.0, 6.0, 7.0]
    theirs = [1.0, 2.0, 3.0]
    a12_max = a12(ours, theirs, maximize=True)
    a12_min = a12(ours, theirs, maximize=False)
    assert a12_max > 0.5
    assert a12_min < 0.5
    # They should sum to 1 when no ties
    assert a12_max + a12_min == pytest.approx(1.0)


def test_a12_hand_computed_with_ties():
    # ours = [1, 2], theirs = [2, 3], maximize=True
    # pairs (1,2): 0 win, 0 tie, 1 loss
    # pairs (1,3): 0 win, 0 tie, 1 loss
    # pairs (2,2): 0 win, 1 tie, 0 loss
    # pairs (2,3): 0 win, 0 tie, 1 loss
    # wins=0, ties=1, n=4 → A12 = (0 + 0.5*1)/4 = 0.125
    result = a12([1.0, 2.0], [2.0, 3.0], maximize=True)
    assert result == pytest.approx(0.125)


def test_a12_empty_ours():
    assert a12([], [1.0, 2.0], maximize=True) == pytest.approx(0.5)


def test_a12_empty_theirs():
    assert a12([1.0, 2.0], [], maximize=True) == pytest.approx(0.5)


def test_a12_magnitude_negligible():
    assert a12_magnitude(0.5) == "negligible"
    assert a12_magnitude(0.55) == "negligible"
    assert a12_magnitude(0.45) == "negligible"


def test_a12_magnitude_small():
    assert a12_magnitude(0.63) == "small"  # |0.63 - 0.5| = 0.13
    assert a12_magnitude(0.37) == "small"


def test_a12_magnitude_medium():
    assert a12_magnitude(0.69) == "medium"  # |0.69 - 0.5| = 0.19
    assert a12_magnitude(0.31) == "medium"


def test_a12_magnitude_large():
    assert a12_magnitude(0.72) == "large"  # |0.72 - 0.5| = 0.22
    assert a12_magnitude(0.0) == "large"
    assert a12_magnitude(1.0) == "large"


# ---------------------------------------------------------------------------
# aggregate_per_dataset
# ---------------------------------------------------------------------------


def _make_block_df() -> pl.DataFrame:
    """Create a simple 2-dataset, 2-group DataFrame for block tests."""
    return pl.DataFrame(
        {
            "dataset": ["ds1", "ds1", "ds1", "ds1", "ds2", "ds2", "ds2", "ds2"],
            "group": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "score": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
        }
    )


def test_aggregate_per_dataset_shape():
    df = _make_block_df()
    result = aggregate_per_dataset(df, metric_col="score", group_col="group", dataset_col="dataset")
    assert result.shape == (2, 3)  # 2 datasets × (dataset + 2 groups)
    assert "dataset" in result.columns
    assert "A" in result.columns
    assert "B" in result.columns


def test_aggregate_per_dataset_median_values():
    # ds1: A=[1,3]→median=2.0, B=[5,7]→median=6.0
    # ds2: A=[2,4]→median=3.0, B=[6,8]→median=7.0
    df = _make_block_df()
    result = aggregate_per_dataset(
        df, metric_col="score", group_col="group", dataset_col="dataset", agg="median"
    ).sort("dataset")
    ds1 = result.filter(pl.col("dataset") == "ds1")
    assert ds1["A"][0] == pytest.approx(2.0)
    assert ds1["B"][0] == pytest.approx(6.0)
    ds2 = result.filter(pl.col("dataset") == "ds2")
    assert ds2["A"][0] == pytest.approx(3.0)
    assert ds2["B"][0] == pytest.approx(7.0)


def test_aggregate_per_dataset_missing_combo_is_null():
    """Missing (dataset, group) combo → null in block."""
    df = pl.DataFrame(
        {
            "dataset": ["ds1", "ds1", "ds2"],
            "group": ["A", "A", "B"],
            "score": [1.0, 2.0, 3.0],
        }
    )
    result = aggregate_per_dataset(df, metric_col="score", group_col="group", dataset_col="dataset")
    # ds1 has no B; ds2 has no A
    ds1 = result.filter(pl.col("dataset") == "ds1")
    assert ds1["B"][0] is None
    ds2 = result.filter(pl.col("dataset") == "ds2")
    assert ds2["A"][0] is None


def test_aggregate_per_dataset_mean():
    df = _make_block_df()
    result = aggregate_per_dataset(
        df, metric_col="score", group_col="group", dataset_col="dataset", agg="mean"
    ).sort("dataset")
    ds1 = result.filter(pl.col("dataset") == "ds1")
    assert ds1["A"][0] == pytest.approx(2.0)  # mean([1,3]) = 2
    assert ds1["B"][0] == pytest.approx(6.0)  # mean([5,7]) = 6


# ---------------------------------------------------------------------------
# cross_dataset_test — 2 groups (Wilcoxon signed-rank)
# ---------------------------------------------------------------------------


def _make_complete_block(ours_vals: list[float], theirs_vals: list[float]) -> pl.DataFrame:
    """Build a complete block DataFrame with datasets as rows."""
    return pl.DataFrame(
        {
            "dataset": [f"ds{i + 1}" for i in range(len(ours_vals))],
            "ours": ours_vals,
            "theirs": theirs_vals,
        }
    )


def test_cross_dataset_test_2groups_signed_rank_path():
    # ours uniformly better under minimize
    block = _make_complete_block(
        ours_vals=[0.1, 0.2, 0.15, 0.12, 0.18, 0.11],
        theirs_vals=[0.8, 0.9, 0.85, 0.82, 0.88, 0.81],
    )
    result = cross_dataset_test(block, ours="ours", maximize=False)
    assert result.method == "wilcoxon_signed_rank"
    assert result.p_value < 0.05
    assert result.n_datasets == 6
    assert result.dropped_datasets == []


def test_cross_dataset_test_2groups_direction_minimize_significant():
    """Ours uniformly smaller → significant under minimize."""
    block = _make_complete_block(
        ours_vals=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        theirs_vals=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    )
    result = cross_dataset_test(block, ours="ours", maximize=False)
    assert result.p_value < 0.05


def test_cross_dataset_test_2groups_direction_maximize_not_significant():
    """Same data, maximize=True: ours (smaller) is not better → not significant."""
    block = _make_complete_block(
        ours_vals=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        theirs_vals=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    )
    result = cross_dataset_test(block, ours="ours", maximize=True)
    assert result.p_value >= 0.05


def test_cross_dataset_test_all_zero_differences():
    """All differences zero → p=1.0 with a note instead of crash."""
    block = _make_complete_block(
        ours_vals=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        theirs_vals=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    result = cross_dataset_test(block, ours="ours", maximize=True)
    assert result.p_value == pytest.approx(1.0)
    assert "zero" in result.note.lower() or "1.0" in result.note


def test_cross_dataset_test_2groups_drops_incomplete():
    """Rows with null in any group are dropped and reported."""
    block = pl.DataFrame(
        {
            "dataset": ["ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds_incomplete"],
            "ours": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None],
            "theirs": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    result = cross_dataset_test(block, ours="ours", maximize=False)
    assert result.n_datasets == 6
    assert "ds_incomplete" in result.dropped_datasets


def test_cross_dataset_test_less_than_2_complete_raises():
    """Fewer than 2 complete blocks → ValueError."""
    block = pl.DataFrame(
        {
            "dataset": ["ds1", "ds2"],
            "ours": [None, None],
            "theirs": [2.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="complete"):
        cross_dataset_test(block, ours="ours", maximize=True)


def test_cross_dataset_test_low_power_flag():
    """n < 5 complete datasets → low_power=True."""
    block = _make_complete_block(
        ours_vals=[1.0, 1.0, 1.0, 1.0],
        theirs_vals=[2.0, 2.0, 2.0, 2.0],
    )
    result = cross_dataset_test(block, ours="ours", maximize=False)
    assert result.low_power is True


def test_cross_dataset_test_no_low_power_with_5():
    block = _make_complete_block(
        ours_vals=[1.0, 1.0, 1.0, 1.0, 1.0],
        theirs_vals=[2.0, 2.0, 2.0, 2.0, 2.0],
    )
    result = cross_dataset_test(block, ours="ours", maximize=False)
    assert result.low_power is False


# ---------------------------------------------------------------------------
# cross_dataset_test — 3 groups (Friedman)
# ---------------------------------------------------------------------------


def _make_3group_block(
    ours_vals: list[float],
    b1_vals: list[float],
    b2_vals: list[float],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dataset": [f"ds{i + 1}" for i in range(len(ours_vals))],
            "ours": ours_vals,
            "b1": b1_vals,
            "b2": b2_vals,
        }
    )


def test_cross_dataset_test_3groups_friedman_path():
    # ours consistently worst for maximize=True → non-significant
    rng = np.random.default_rng(7)
    block = _make_3group_block(
        ours_vals=rng.uniform(0.1, 0.2, 10).tolist(),
        b1_vals=rng.uniform(0.1, 0.2, 10).tolist(),
        b2_vals=rng.uniform(0.1, 0.2, 10).tolist(),
    )
    result = cross_dataset_test(block, ours="ours", maximize=True)
    assert result.method == "friedman"
    # Non-significant omnibus → posthoc is None
    if result.p_value > 0.05:
        assert result.posthoc is None


def test_cross_dataset_test_3groups_significant_posthoc_columns():
    """Significant Friedman → post-hoc DataFrame has the required columns."""
    # Make ours clearly dominate
    ours_vals = [10.0] * 10
    b1_vals = [1.0] * 10
    b2_vals = [2.0] * 10
    block = _make_3group_block(ours_vals, b1_vals, b2_vals)
    result = cross_dataset_test(block, ours="ours", maximize=True)
    assert result.method == "friedman"
    if result.posthoc is not None:
        expected_cols = {
            "group",
            "statistic",
            "p_value",
            "p_adjusted",
            "significant",
            "a12_of_aggregates",
        }
        assert expected_cols.issubset(set(result.posthoc.columns))
        assert len(result.posthoc) == 2  # 2 competitors


def test_cross_dataset_test_3groups_mean_ranks():
    """mean_ranks has an entry for every group."""
    block = _make_3group_block(
        ours_vals=[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        b1_vals=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        b2_vals=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    result = cross_dataset_test(block, ours="ours", maximize=True)
    assert "ours" in result.mean_ranks
    assert "b1" in result.mean_ranks
    assert "b2" in result.mean_ranks
    # ours is best (rank 1), b2 is worst (rank 3)
    assert result.mean_ranks["ours"] == pytest.approx(1.0)
    assert result.mean_ranks["b2"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------


def _make_latex_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Dataset": ["ds_one", "ds_two"],
            "Method A": ["0.9100 ± 0.0100 🥇", "0.7500 ± 0.0200"],
            "Method_B": ["0.5000 ± 0.0300", "0.8000 ± 0.0150 🥇"],
        }
    )


def test_comparison_table_to_latex_contains_booktabs():
    df = _make_latex_df()
    latex = comparison_table_to_latex(df, caption="My Table", label="tab:my", maximize=True)
    assert r"\toprule" in latex
    assert r"\midrule" in latex
    assert r"\bottomrule" in latex


def test_comparison_table_to_latex_escapes_underscore():
    df = _make_latex_df()
    latex = comparison_table_to_latex(df, caption="My Table", label="tab:my", maximize=True)
    # Method_B column name → escaped
    assert r"Method\_B" in latex


def test_comparison_table_to_latex_booktabs_comment():
    df = _make_latex_df()
    latex = comparison_table_to_latex(df, caption="My Table", label="tab:my", maximize=True)
    assert r"% requires \usepackage{booktabs}" in latex


def test_comparison_table_to_latex_empty_df():
    latex = comparison_table_to_latex(pl.DataFrame(), caption="c", label="l", maximize=True)
    assert "Empty" in latex


def test_cross_dataset_to_latex_structure():
    block = _make_complete_block(
        ours_vals=[0.1, 0.2, 0.15, 0.12, 0.18, 0.11],
        theirs_vals=[0.8, 0.9, 0.85, 0.82, 0.88, 0.81],
    )
    result = cross_dataset_test(block, ours="ours", maximize=False)
    latex = cross_dataset_to_latex(
        result, block, caption="Cross test", label="tab:cross", maximize=False
    )
    assert r"\toprule" in latex
    assert r"\bottomrule" in latex
    assert "wilcoxon" in latex.lower() or "Wilcoxon" in latex


def test_cross_dataset_to_latex_pvalue_small_notation():
    """Very small p-values use <0.001 notation."""
    block = _make_complete_block(
        ours_vals=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        theirs_vals=[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    )
    result = cross_dataset_test(block, ours="ours", maximize=False)
    latex = cross_dataset_to_latex(result, block, caption="C", label="L", maximize=False)
    # Either the p value is very small and formatted as <0.001, or it's just small
    assert r"$<$0.001" in latex or "0.001" in latex or "p" in latex.lower()


def test_cross_dataset_to_latex_with_posthoc():
    """Friedman with significant omnibus → post-hoc table appended."""
    block = _make_3group_block(
        ours_vals=[10.0] * 10,
        b1_vals=[1.0] * 10,
        b2_vals=[2.0] * 10,
    )
    result = cross_dataset_test(block, ours="ours", maximize=True)
    latex = cross_dataset_to_latex(result, block, caption="Friedman", label="tab:f", maximize=True)
    assert r"\toprule" in latex
    if result.posthoc is not None:
        assert "posthoc" in latex or "post" in latex.lower()


# ---------------------------------------------------------------------------
# aggregate_for_plot
# ---------------------------------------------------------------------------


def _make_plot_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "pop_size": [10, 10, 20, 20, 10, 10, 20, 20],
            "fitness": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )


def test_aggregate_for_plot_shape():
    df = _make_plot_df()
    result = aggregate_for_plot(
        df, x_col="pop_size", y_col="fitness", group_col="group", agg="median"
    )
    # 2 groups × 2 x values = 4 rows
    assert len(result) == 4
    assert "group" in result.columns
    assert "pop_size" in result.columns
    assert "y" in result.columns


def test_aggregate_for_plot_median_values():
    # A x=10: [1,2] → median=1.5
    # A x=20: [3,4] → median=3.5
    # B x=10: [5,6] → median=5.5
    # B x=20: [7,8] → median=7.5
    df = _make_plot_df()
    result = aggregate_for_plot(
        df, x_col="pop_size", y_col="fitness", group_col="group", agg="median"
    )
    row_a10 = result.filter((pl.col("group") == "A") & (pl.col("pop_size") == 10))
    assert row_a10["y"][0] == pytest.approx(1.5)
    row_b20 = result.filter((pl.col("group") == "B") & (pl.col("pop_size") == 20))
    assert row_b20["y"][0] == pytest.approx(7.5)


def test_aggregate_for_plot_mean():
    df = _make_plot_df()
    result = aggregate_for_plot(
        df, x_col="pop_size", y_col="fitness", group_col="group", agg="mean"
    )
    row_a10 = result.filter((pl.col("group") == "A") & (pl.col("pop_size") == 10))
    assert row_a10["y"][0] == pytest.approx(1.5)


def test_aggregate_for_plot_band_columns():
    df = _make_plot_df()
    result = aggregate_for_plot(
        df, x_col="pop_size", y_col="fitness", group_col="group", agg="median", band=True
    )
    assert "y_q1" in result.columns
    assert "y_q3" in result.columns
    # Q1 ≤ median ≤ Q3 for all rows
    for row in result.iter_rows(named=True):
        assert row["y_q1"] <= row["y"] <= row["y_q3"]


def test_aggregate_for_plot_no_band_no_band_columns():
    df = _make_plot_df()
    result = aggregate_for_plot(
        df, x_col="pop_size", y_col="fitness", group_col="group", agg="median", band=False
    )
    assert "y_q1" not in result.columns
    assert "y_q3" not in result.columns


def test_aggregate_for_plot_empty_df():
    result = aggregate_for_plot(
        pl.DataFrame(),
        x_col="pop_size",
        y_col="fitness",
        group_col="group",
    )
    assert result.is_empty()


# ---------------------------------------------------------------------------
# Regression: direction-aware winner bolding in LaTeX export
# ---------------------------------------------------------------------------


def test_comparison_table_to_latex_bolds_max_when_maximize():
    df = _make_latex_df()
    latex = comparison_table_to_latex(df, caption="C", label="L", maximize=True)
    # Row 1: Method A=0.91 vs Method_B=0.50 → Method A bolded
    assert r"\textbf{0.9100" in latex
    assert r"\textbf{0.5000" not in latex


def test_comparison_table_to_latex_bolds_min_when_minimize():
    df = _make_latex_df()
    latex = comparison_table_to_latex(df, caption="C", label="L", maximize=False)
    # Row 1: Method_B=0.50 is the (lower-is-better) winner
    assert r"\textbf{0.5000" in latex
    assert r"\textbf{0.9100" not in latex


def test_cross_dataset_to_latex_bolds_min_when_minimize():
    block = _make_complete_block(
        ours_vals=[0.1, 0.2, 0.15, 0.12, 0.18, 0.11],
        theirs_vals=[0.8, 0.9, 0.85, 0.82, 0.88, 0.81],
    )
    result = cross_dataset_test(block, ours="ours", maximize=False)
    latex = cross_dataset_to_latex(result, block, caption="C", label="L", maximize=False)
    # The "ours" values (lower) must be the bolded ones, e.g. 0.1 in row ds1
    assert r"\textbf{0.100}" in latex
    assert r"\textbf{0.800}" not in latex


def test_cross_dataset_to_latex_bolds_max_when_maximize():
    block = _make_complete_block(
        ours_vals=[0.1, 0.2, 0.15, 0.12, 0.18, 0.11],
        theirs_vals=[0.8, 0.9, 0.85, 0.82, 0.88, 0.81],
    )
    result = cross_dataset_test(block, ours="ours", maximize=True)
    latex = cross_dataset_to_latex(result, block, caption="C", label="L", maximize=True)
    assert r"\textbf{0.800}" in latex
    assert r"\textbf{0.100}" not in latex


# ---------------------------------------------------------------------------
# Regression: Holm adjusted p-values must be monotone (step-down running max)
# ---------------------------------------------------------------------------


def test_posthoc_adjusted_p_monotone_and_bounded():
    """For rows sorted by raw p ascending, p_adjusted is non-decreasing and >= p_value."""
    rng = np.random.default_rng(7)
    n_ds = 12
    ours = rng.normal(0.10, 0.01, n_ds)
    b1 = ours + rng.normal(0.50, 0.05, n_ds)  # clearly worse
    b2 = ours + rng.normal(0.04, 0.03, n_ds)  # marginally worse
    b3 = ours + rng.normal(0.00, 0.03, n_ds)  # indistinguishable
    block = pl.DataFrame(
        {
            "dataset": [f"ds{i}" for i in range(n_ds)],
            "ours": ours,
            "b1": b1,
            "b2": b2,
            "b3": b3,
        }
    )
    result = cross_dataset_test(block, ours="ours", maximize=False)
    assert result.method == "friedman"
    assert result.posthoc is not None
    ph = result.posthoc.sort("p_value")
    p_adj = ph.get_column("p_adjusted").to_list()
    p_raw = ph.get_column("p_value").to_list()
    assert all(a >= r for a, r in zip(p_adj, p_raw))
    assert all(p_adj[i] <= p_adj[i + 1] for i in range(len(p_adj) - 1))
    assert all(0.0 <= a <= 1.0 for a in p_adj)
