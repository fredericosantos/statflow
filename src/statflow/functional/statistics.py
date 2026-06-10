"""
Pure statistical functions for comparing experiment groups.

Extracted from subpages/comparison.py so they can be tested independently
without triggering Streamlit's page-config side effects.

statistics.py
├── holm_bonferroni_correction()        # Holm-Bonferroni p-value correction
├── perform_statistical_tests()         # Mann-Whitney U one-vs-all, one-sided
├── check_significance()                # All-vs-all significance check helper
├── build_comparison_table()            # Aggregated stats + per-dataset tests + raw rows
│
├── AGGREGATIONS                        # dict of named aggregation callables
├── iqm()                               # Interquartile mean (Q1–Q3 inclusive)
├── aggregate_per_dataset()             # Wide block matrix (dataset × group)
│
├── a12()                               # Vargha–Delaney A12 effect size
├── a12_magnitude()                     # Magnitude label for A12
│
├── CrossDatasetResult                  # Dataclass for cross_dataset_test output
├── cross_dataset_test()                # Wilcoxon signed-rank (2 groups) or Friedman (≥3)
│
├── comparison_table_to_latex()         # LaTeX booktabs table from comparison DataFrame
└── cross_dataset_to_latex()            # LaTeX booktabs table from CrossDatasetResult
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

# ==============================================================================
# Existing M2 functions (unchanged)
# ==============================================================================


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


# ==============================================================================
# Part 1.1 — Aggregations
# ==============================================================================


def iqm(values: Sequence[float] | np.ndarray) -> float:
    """Compute the interquartile mean (mean of values within [Q1, Q3], inclusive).

    For degenerate inputs (n < 4, too few values to split into quartiles),
    falls back to the plain mean of available values (nan for n=0).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan")
    if len(arr) < 4:
        # Not enough data to split into quartiles; use plain mean
        return float(np.mean(arr))
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    mask = (arr >= q1) & (arr <= q3)
    trimmed = arr[mask]
    if len(trimmed) == 0:
        return float(np.mean(arr))
    return float(np.mean(trimmed))


AGGREGATIONS: dict[str, Callable[[Sequence[float] | np.ndarray], float]] = {
    "median": lambda v: float(np.median(np.asarray(v, dtype=float))),
    "mean": lambda v: float(np.mean(np.asarray(v, dtype=float))),
    "min": lambda v: float(np.min(np.asarray(v, dtype=float))),
    "max": lambda v: float(np.max(np.asarray(v, dtype=float))),
    "iqm": iqm,
}


def aggregate_per_dataset(
    df: pl.DataFrame,
    *,
    metric_col: str,
    group_col: str,
    dataset_col: str,
    agg: str = "median",
) -> pl.DataFrame:
    """Build a wide block matrix: one row per dataset, one column per group.

    Each cell = aggregated metric for that (dataset, group)'s runs.
    Missing (dataset, group) combinations → null.

    Args:
        df: Input DataFrame containing at least `metric_col`, `group_col`, `dataset_col`.
        metric_col: Name of the metric column.
        group_col: Name of the group/method column.
        dataset_col: Name of the dataset column.
        agg: Aggregation key from AGGREGATIONS (default "median").

    Returns:
        Wide Polars DataFrame with `dataset_col` as the first column,
        followed by one column per group.
    """
    if df.is_empty():
        return pl.DataFrame()

    agg_fn = AGGREGATIONS[agg]

    datasets = df[dataset_col].unique().sort().to_list()
    groups = df[group_col].unique().sort().to_list()

    rows = []
    for dataset in datasets:
        row: dict[str, Any] = {dataset_col: dataset}
        for group in groups:
            subset = (
                df.filter((pl.col(dataset_col) == dataset) & (pl.col(group_col) == group))
                .get_column(metric_col)
                .drop_nulls()
                .to_numpy()
            )
            if len(subset) == 0:
                row[group] = None
            else:
                row[group] = agg_fn(subset)
        rows.append(row)

    return pl.DataFrame(rows)


# ==============================================================================
# Part 1.2 — Vargha–Delaney A12 effect size
# ==============================================================================


def a12(
    ours: Sequence[float] | np.ndarray,
    theirs: Sequence[float] | np.ndarray,
    maximize: bool = True,
) -> float:
    """Vargha–Delaney A12 effect size.

    Returns P(random 'ours' value beats random 'theirs') + 0.5·P(tie),
    where 'beats' means larger when ``maximize=True``, smaller when ``maximize=False``.

    Vectorized via numpy broadcasting (no O(n·m) Python loops).

    Args:
        ours: Values for our method.
        theirs: Values for the baseline method.
        maximize: If True, higher is better (default). If False, lower is better.

    Returns:
        A12 in [0, 1]. 0.5 = no effect. >0.5 = ours is better.
    """
    x = np.asarray(ours, dtype=float)
    y = np.asarray(theirs, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return 0.5

    # Broadcasting: x[:, None] vs y[None, :]
    diff = x[:, None] - y[None, :]  # shape (n, m)
    if maximize:
        wins = float(np.sum(diff > 0))
        ties = float(np.sum(diff == 0))
    else:
        wins = float(np.sum(diff < 0))
        ties = float(np.sum(diff == 0))

    n = len(x) * len(y)
    return (wins + 0.5 * ties) / n


def a12_magnitude(a: float) -> str:
    """Return a magnitude label for an A12 value (Vargha & Vanneste thresholds).

    Thresholds are on |A12 - 0.5|:
      negligible < 0.06
      small      < 0.14
      medium     < 0.21
      large      >= 0.21
    """
    delta = abs(a - 0.5)
    if delta < 0.06:
        return "negligible"
    elif delta < 0.14:
        return "small"
    elif delta < 0.21:
        return "medium"
    else:
        return "large"


# ==============================================================================
# Part 1.3 — Cross-dataset test
# ==============================================================================


@dataclass
class CrossDatasetResult:
    """Result of a cross-dataset statistical test.

    Attributes:
        method: "wilcoxon_signed_rank" | "friedman"
        statistic: Test statistic value.
        p_value: Raw p-value.
        n_datasets: Number of complete blocks actually used.
        dropped_datasets: Datasets excluded because they had missing groups.
        posthoc: Friedman post-hoc table (None for signed-rank or non-significant Friedman).
        mean_ranks: Average rank per group (Friedman only; {} otherwise).
        low_power: True when n_datasets < 5 (small-n warning).
        note: Optional string for edge-case messages (e.g. all-zero differences).
    """

    method: str
    statistic: float
    p_value: float
    n_datasets: int
    dropped_datasets: list[str] = field(default_factory=list)
    posthoc: pl.DataFrame | None = None
    mean_ranks: dict[str, float] = field(default_factory=dict)
    low_power: bool = False
    note: str = ""


def _compute_mean_ranks(
    block: pl.DataFrame, *, maximize: bool, groups: list[str]
) -> dict[str, float]:
    """Compute mean rank of each group across datasets (rank 1 = best per direction)."""
    group_cols = [c for c in groups if c in block.columns]
    rank_sums: dict[str, float] = {g: 0.0 for g in group_cols}
    n_rows = len(block)

    for row in block.iter_rows(named=True):
        values = {g: row[g] for g in group_cols if row[g] is not None}
        if not values:
            continue
        # Rank: argsort descending (maximize) or ascending (minimize)
        sorted_groups = sorted(values, key=lambda g: values[g], reverse=maximize)
        for rank_idx, g in enumerate(sorted_groups):
            rank_sums[g] = rank_sums.get(g, 0.0) + (rank_idx + 1)

    return {g: rank_sums[g] / n_rows for g in group_cols}


def cross_dataset_test(
    block: pl.DataFrame,
    *,
    ours: str,
    maximize: bool,
    alpha: float = 0.05,
) -> CrossDatasetResult:
    """Run a cross-dataset significance test over the aggregated block matrix.

    Dispatches dynamically:
    - **Exactly 2 groups** → one-sided Wilcoxon signed-rank.
    - **≥ 3 groups** → Friedman omnibus; if significant, Holm-corrected pairwise
      one-sided signed-rank of `ours` vs each other group.

    Args:
        block: Output of ``aggregate_per_dataset``.  Must have a dataset identifier
            column (first column) plus one column per group.
        ours: Name of the 'ours' group column.
        maximize: True if higher metric values are better.
        alpha: Significance level (default 0.05).

    Returns:
        ``CrossDatasetResult`` populated with test outcome.

    Raises:
        ValueError: If fewer than 2 complete blocks remain after dropping incomplete rows.
    """
    if block.is_empty():
        raise ValueError("Block matrix is empty — no data to test.")

    dataset_col = block.columns[0]
    group_cols = [c for c in block.columns if c != dataset_col]

    if len(group_cols) < 2:
        raise ValueError(f"Need at least 2 groups; got {group_cols}.")

    # Drop incomplete rows (any null in any group column)
    complete_mask = pl.all_horizontal(pl.col(c).is_not_null() for c in group_cols)
    complete_block = block.filter(complete_mask)
    dropped_datasets = block.filter(~complete_mask)[dataset_col].to_list()

    n_datasets = len(complete_block)
    if n_datasets < 2:
        raise ValueError(
            f"Only {n_datasets} complete dataset block(s) remain after dropping "
            f"incomplete rows {dropped_datasets}. At least 2 are required."
        )

    low_power = n_datasets < 5

    if len(group_cols) == 2:
        # Two-group path: one-sided Wilcoxon signed-rank
        other = [g for g in group_cols if g != ours][0]
        ours_vals = complete_block[ours].to_numpy().astype(float)
        other_vals = complete_block[other].to_numpy().astype(float)

        alternative = "greater" if maximize else "less"
        note = ""

        # Detect all-zero differences before calling scipy to avoid a RuntimeWarning
        # (older scipy raises ValueError; newer versions return p=1.0 with a warning).
        if np.all(ours_vals == other_vals):
            stat, pval, note = 0.0, 1.0, "All differences are zero; p-value set to 1.0."
        else:
            try:
                stat, pval = stats.wilcoxon(
                    ours_vals, other_vals, alternative=alternative, zero_method="wilcox"
                )
            except ValueError as exc:
                if "all zero" in str(exc).lower():
                    stat, pval, note = 0.0, 1.0, "All differences are zero; p-value set to 1.0."
                else:
                    raise

        return CrossDatasetResult(
            method="wilcoxon_signed_rank",
            statistic=float(stat),
            p_value=float(pval),
            n_datasets=n_datasets,
            dropped_datasets=dropped_datasets,
            low_power=low_power,
            note=note,
        )

    else:
        # Friedman path (≥ 3 groups)
        columns_arrays = [complete_block[g].to_numpy().astype(float) for g in group_cols]
        stat, pval = stats.friedmanchisquare(*columns_arrays)

        mean_ranks = _compute_mean_ranks(complete_block, maximize=maximize, groups=group_cols)

        posthoc: pl.DataFrame | None = None
        if pval <= alpha:
            # Post-hoc: pairwise one-sided Wilcoxon (ours vs each other group), Holm-corrected
            other_groups = [g for g in group_cols if g != ours]
            ours_vals = complete_block[ours].to_numpy().astype(float)
            alternative = "greater" if maximize else "less"

            raw_p: list[float] = []
            raw_stat: list[float] = []
            raw_a12: list[float] = []

            for other in other_groups:
                other_vals = complete_block[other].to_numpy().astype(float)
                try:
                    s, p = stats.wilcoxon(
                        ours_vals, other_vals, alternative=alternative, zero_method="wilcox"
                    )
                except ValueError as exc:
                    if "all zero" in str(exc).lower():
                        s, p = 0.0, 1.0
                    else:
                        raise
                raw_stat.append(float(s))
                raw_p.append(float(p))
                raw_a12.append(a12(ours_vals, other_vals, maximize=maximize))

            significant = holm_bonferroni_correction(raw_p, alpha=alpha)

            # Holm step-down adjusted p-values: enforce monotonicity with a
            # running max over the ascending-p order, so a smaller raw p can
            # never display a smaller adjusted p than a larger raw p.
            n = len(raw_p)
            sorted_idx = sorted(range(n), key=lambda i: raw_p[i])
            p_adjusted = [0.0] * n
            running_max = 0.0
            for rank, orig_i in enumerate(sorted_idx):
                running_max = max(running_max, raw_p[orig_i] * (n - rank))
                p_adjusted[orig_i] = min(running_max, 1.0)

            posthoc = pl.DataFrame(
                {
                    "group": other_groups,
                    "statistic": raw_stat,
                    "p_value": raw_p,
                    "p_adjusted": p_adjusted,
                    "significant": significant,
                    "a12_of_aggregates": raw_a12,
                }
            )

        return CrossDatasetResult(
            method="friedman",
            statistic=float(stat),
            p_value=float(pval),
            n_datasets=n_datasets,
            dropped_datasets=dropped_datasets,
            posthoc=posthoc,
            mean_ranks=mean_ranks,
            low_power=low_power,
        )


# ==============================================================================
# Part 1.4 — LaTeX export
# ==============================================================================

_LATEX_SPECIALS = re.compile(r"([_\%\&\#\$\{\}])")


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in a string."""
    return _LATEX_SPECIALS.sub(r"\\\1", str(text))


def _fmt_number(value: float, sig: int = 3) -> str:
    """Format to `sig` significant digits; use <0.001 notation for tiny p-values."""
    if abs(value) < 0.001 and value != 0.0:
        return r"$<$0.001"
    if value == 0.0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
    decimals = max(0, sig - 1 - magnitude)
    return f"{value:.{decimals}f}"


def _star(p: float) -> str:
    """Return significance star(s) for a p-value."""
    if p < 0.001:
        return r"$^{***}$"
    elif p < 0.01:
        return r"$^{**}$"
    elif p < 0.05:
        return r"$^{*}$"
    return ""


def comparison_table_to_latex(
    df: pl.DataFrame,
    *,
    caption: str,
    label: str,
    maximize: bool,
) -> str:
    """Render a comparison DataFrame to a LaTeX booktabs table.

    Expected columns: ``Dataset``, plus one column per method (formatted strings
    from ``format_cell``).  The function detects the winning cell per row
    (lowest numeric value when ``maximize=False``; highest when ``maximize=True``
    — uses the first numeric token in each cell) and bolds it.

    Args:
        df: Display DataFrame as produced by the Comparison page.
        caption: Table caption.
        label: LaTeX \\label{...} key.
        maximize: True if higher metric values are better (direction of "winning").

    Returns:
        A LaTeX snippet string suitable for st.code(..., language="latex").
    """
    if df.is_empty():
        return "% Empty table"

    cols = df.columns
    n_cols = len(cols)
    col_spec = "l" + "r" * (n_cols - 1)

    header = " & ".join(_escape_latex(c) for c in cols) + r" \\"

    lines = [
        r"% requires \usepackage{booktabs}",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{_escape_latex(caption)}}}",
        rf"\label{{{_escape_latex(label)}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    def _first_num(cell: str) -> float | None:
        """Extract the leading numeric value from a formatted cell like '0.1234 ± 0.01 🥇'."""
        m = re.match(r"^\s*([0-9.eE+\-]+)", cell)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    for row in df.iter_rows(named=True):
        method_cols = cols[1:]  # skip Dataset column
        nums = {c: _first_num(str(row[c])) for c in method_cols}
        valid_nums = {c: v for c, v in nums.items() if v is not None}

        winner_col: str | None = None
        if valid_nums:
            pick = max if maximize else min
            winner_col = pick(valid_nums, key=lambda c: valid_nums[c])

        cells: list[str] = [_escape_latex(str(row[cols[0]]))]
        for c in method_cols:
            cell_str = str(row[c])
            escaped = _escape_latex(cell_str)
            if c == winner_col:
                escaped = rf"\textbf{{{escaped}}}"
            cells.append(escaped)

        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def cross_dataset_to_latex(
    result: CrossDatasetResult,
    block: pl.DataFrame,
    *,
    caption: str,
    label: str,
    maximize: bool,
) -> str:
    """Render a CrossDatasetResult + block matrix to a LaTeX booktabs table.

    The top section shows the aggregated block matrix (datasets × groups) with
    the winning value per row bolded.  Below a \\midrule, the test summary is
    appended as a multi-column row.  If post-hoc results exist, they are
    appended in a second sub-table.

    Args:
        result: Output of ``cross_dataset_test``.
        block: The ``aggregate_per_dataset`` output used for the test.
        caption: Table caption.
        label: LaTeX \\label{...} key.
        maximize: True if higher metric values are better (direction of "winning").

    Returns:
        LaTeX snippet string.
    """
    if block.is_empty():
        return "% Empty block"

    dataset_col = block.columns[0]
    group_cols = [c for c in block.columns if c != dataset_col]
    n_cols = 1 + len(group_cols)
    col_spec = "l" + "r" * len(group_cols)

    header = (
        " & ".join([_escape_latex(dataset_col)] + [_escape_latex(g) for g in group_cols]) + r" \\"
    )

    lines = [
        r"% requires \usepackage{booktabs}",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{_escape_latex(caption)}}}",
        rf"\label{{{_escape_latex(label)}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for row in block.iter_rows(named=True):
        vals = {g: row[g] for g in group_cols if row[g] is not None}
        winner_col: str | None = None
        if vals:
            pick = max if maximize else min
            winner_col = pick(vals, key=lambda g: vals[g])

        cells = [_escape_latex(str(row[dataset_col]))]
        for g in group_cols:
            v = row[g]
            if v is None:
                cells.append("--")
            else:
                formatted = _fmt_number(float(v))
                if g == winner_col:
                    formatted = rf"\textbf{{{formatted}}}"
                cells.append(formatted)
        lines.append(" & ".join(cells) + r" \\")

    # Test summary
    method_display = result.method.replace("_", " ").title()
    p_str = _fmt_number(result.p_value) + _star(result.p_value)
    summary = (
        rf"\multicolumn{{{n_cols}}}{{l}}{{"
        rf"{method_display}: $\chi^2$={_fmt_number(result.statistic)}, "
        rf"$p$={p_str}, $n$={result.n_datasets}"
        rf"}}"
    )
    lines += [r"\midrule", summary + r" \\"]

    if result.note:
        lines.append(
            rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{_escape_latex(result.note)}}}}}" + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    # Post-hoc sub-table
    if result.posthoc is not None and not result.posthoc.is_empty():
        ph = result.posthoc
        ph_header = r"Group & Statistic & $p$ & $p_{\text{adj}}$ & Sig. & A12 \\"
        lines += [
            "",
            r"% Post-hoc pairwise comparisons (Holm-corrected one-sided Wilcoxon)",
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{{_escape_latex(caption)} — post-hoc}}",
            rf"\label{{{_escape_latex(label)}-posthoc}}",
            r"\begin{tabular}{lrrrrl}",
            r"\toprule",
            ph_header,
            r"\midrule",
        ]
        for row in ph.iter_rows(named=True):
            sig_marker = r"\checkmark" if row["significant"] else ""
            p_adj_str = _fmt_number(float(row["p_adjusted"])) + _star(float(row["p_adjusted"]))
            cells = [
                _escape_latex(str(row["group"])),
                _fmt_number(float(row["statistic"])),
                _fmt_number(float(row["p_value"])) + _star(float(row["p_value"])),
                p_adj_str,
                sig_marker,
                _fmt_number(float(row["a12_of_aggregates"])),
            ]
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    return "\n".join(lines)


# ==============================================================================
# Plot aggregation helper (used by Plots page; pure, unit-tested)
# ==============================================================================


def aggregate_for_plot(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: str,
    agg: str = "median",
    band: bool = False,
) -> pl.DataFrame:
    """Aggregate run-level data for a line plot.

    Groups by (group_col, x_col) and computes the aggregated y value.
    Optionally computes Q1/Q3 band columns.

    Args:
        df: Input DataFrame with at least x_col, y_col, group_col.
        x_col: Name of the numeric x-axis parameter column.
        y_col: Name of the metric column.
        group_col: Name of the group/series column.
        agg: Aggregation key from AGGREGATIONS.
        band: If True, also compute ``y_q1`` and ``y_q3`` columns.

    Returns:
        Polars DataFrame with columns: group_col, x_col, y (aggregated),
        and optionally y_q1, y_q3.
    """
    if df.is_empty():
        return pl.DataFrame()

    agg_fn = AGGREGATIONS[agg]

    groups = df[group_col].unique().sort().to_list()
    x_vals = df[x_col].unique().sort().to_list()

    rows = []
    for group in groups:
        for x_val in x_vals:
            subset = (
                df.filter((pl.col(group_col) == group) & (pl.col(x_col) == x_val))
                .get_column(y_col)
                .drop_nulls()
                .to_numpy()
            )
            if len(subset) == 0:
                continue
            row: dict[str, Any] = {
                group_col: group,
                x_col: x_val,
                "y": agg_fn(subset),
            }
            if band:
                row["y_q1"] = float(np.quantile(subset, 0.25))
                row["y_q3"] = float(np.quantile(subset, 0.75))
            rows.append(row)

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows)
