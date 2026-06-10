"""
Pure statistical functions for comparing experiment groups.

Extracted from subpages/comparison.py so they can be tested independently
without triggering Streamlit's page-config side effects.

statistics.py
├── holm_bonferroni_correction()   # Holm-Bonferroni p-value correction
├── perform_statistical_tests()    # Mann-Whitney U one-vs-all, one-sided
├── check_significance()           # All-vs-all significance check helper
└── build_comparison_table()       # Aggregated stats + per-dataset tests + raw rows
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Holm-Bonferroni correction to p-values.

    Sorts p-values ascending, tests each against a shrinking threshold
    (alpha / (n - rank)), and stops rejecting as soon as a p-value exceeds
    its threshold.  Returns a bool list parallel to the *input* order.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed_pvals = list(enumerate(p_values))
    indexed_pvals.sort(key=lambda x: x[1])

    rejected = [False] * n
    for i, (orig_idx, pval) in enumerate(indexed_pvals):
        corrected_alpha = alpha / (n - i)
        if pval <= corrected_alpha:
            rejected[orig_idx] = True
        else:
            break

    return rejected


def perform_statistical_tests(
    raw_data: pl.DataFrame,
    our_group: str,
    their_groups: list[str],
    metric: str,
    group_col: str = "group_label",
    maximize: bool = False,
) -> dict[str, dict[str, Any]]:
    """Mann-Whitney U (Wilcoxon rank-sum) one-vs-all, one-sided by direction.

    ``maximize=False`` (default) treats lower values as better (e.g. error/loss);
    ``maximize=True`` treats higher values as better (e.g. accuracy/R^2). The
    test's one-sided alternative and the median comparison both follow it.

    Returns a mapping from competitor group name to a dict with keys:
      - ``p_value``: raw p-value from the test
      - ``our_median``: median of ``our_group``
      - ``their_median``: median of the competitor group
      - ``our_is_better``: bool, whether our median is directionally better
      - ``is_significant``: bool, after Holm-Bonferroni correction (added in a
        second pass once all p-values are collected)
    """
    results: dict[str, dict[str, Any]] = {}

    alternative = "greater" if maximize else "less"

    our_data = (
        raw_data.filter(pl.col(group_col) == our_group).get_column(metric).drop_nulls().to_numpy()
    )

    if len(our_data) == 0:
        return results

    p_values: list[float] = []
    group_names: list[str] = []

    for their_group in their_groups:
        their_data = (
            raw_data.filter(pl.col(group_col) == their_group)
            .get_column(metric)
            .drop_nulls()
            .to_numpy()
        )

        if len(their_data) == 0:
            continue

        try:
            _stat, pval = stats.mannwhitneyu(our_data, their_data, alternative=alternative)
            our_median = float(np.median(our_data))
            their_median = float(np.median(their_data))
            our_is_better = our_median > their_median if maximize else our_median < their_median

            p_values.append(pval)
            group_names.append(their_group)
            results[their_group] = {
                "p_value": pval,
                "our_median": our_median,
                "their_median": their_median,
                "our_is_better": our_is_better,
            }
        except Exception:
            continue

    if p_values:
        rejected = holm_bonferroni_correction(p_values)
        for i, group_name in enumerate(group_names):
            results[group_name]["is_significant"] = rejected[i]

    return results


def check_significance(
    raw_data: pl.DataFrame,
    focused_group: str,
    competitor_groups: list[str],
    metric: str,
    maximize: bool = False,
) -> bool:
    """Return True iff focused_group is significantly better than ALL competitor_groups."""
    if not competitor_groups:
        return False
    res = perform_statistical_tests(
        raw_data, focused_group, competitor_groups, metric, maximize=maximize
    )
    if not res:
        return False
    return all(
        r.get("is_significant", False) and r.get("our_is_better", False) for r in res.values()
    )


def build_comparison_table(
    metric_df: pl.DataFrame,
    param_df: pl.DataFrame,
    metric: str,
    agg_type: str,
    our_groups: list[str],
    their_groups: list[str],
    maximize: bool = False,
) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame]:
    """Build comparison table: (aggregated stats, per-dataset tests, joined raw rows).

    Returns a 3-tuple:
      - ``agg_df``: aggregated value + spread per (dataset, group)
      - ``stats_results``: ``{"{dataset}_{our_group}": {their_group: {...}}}``
      - ``combined``: raw per-run rows with dataset_name and group_label joined
    """
    if metric_df.is_empty() or param_df.is_empty():
        return pl.DataFrame(), {}, pl.DataFrame()

    if "run_id" in metric_df.columns and "run_id" in param_df.columns:
        combined = metric_df.join(
            param_df.select(["run_id", "group_label", "dataset_name"]),
            on="run_id",
            how="left",
        )
    else:
        combined = metric_df.join(
            param_df.select(["dataset_name", "group_label"]).unique(),
            on="dataset_name",
            how="left",
        )

    if combined.is_empty():
        return pl.DataFrame(), {}, pl.DataFrame()

    all_groups = our_groups + their_groups
    combined = combined.filter(pl.col("group_label").is_in(all_groups))

    m_col = pl.col(metric).drop_nans()
    if agg_type == "Mean ± Std":
        agg_df = combined.group_by(["dataset_name", "group_label"]).agg(
            [
                m_col.mean().alias("value"),
                m_col.std().alias("spread"),
            ]
        )
    else:  # Median ± IQR
        agg_df = combined.group_by(["dataset_name", "group_label"]).agg(
            [
                m_col.median().alias("value"),
                (m_col.quantile(0.75) - m_col.quantile(0.25)).alias("spread"),
            ]
        )

    stats_results: dict[str, Any] = {}
    if our_groups and their_groups:
        datasets = combined.get_column("dataset_name").unique().to_list()
        for dataset in datasets:
            dataset_data = combined.filter(pl.col("dataset_name") == dataset)
            for our_group in our_groups:
                key = f"{dataset}_{our_group}"
                stats_results[key] = perform_statistical_tests(
                    dataset_data, our_group, their_groups, metric, maximize=maximize
                )

    return agg_df, stats_results, combined
