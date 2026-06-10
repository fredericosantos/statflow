# Milestone 3 — Cross-dataset significance, effect sizes, LaTeX export, aggregated line plots

Spec authored by the head engineer (Claude / Fable 5). Implementation agents: follow this
document exactly; where it is silent, follow `CLAUDE.md` conventions in the repo root.
Deviations from this spec must be reported back, not silently improvised.

## Context

Statflow today compares "ours vs theirs" **per dataset** with one-sided Mann–Whitney U +
Holm–Bonferroni (`subpages/comparison.py`). Milestone 2 (branch `m2-quality-tests-ci`)
extracts the pure statistics into `src/statflow/functional/statistics.py` and adds a pytest
suite — **this milestone builds on top of that branch**, not on master, if M2 is not yet merged.

Three features:

1. **Cross-dataset significance** ("is our method better *overall*?") — dynamic: 2 groups →
   Wilcoxon signed-rank; ≥3 groups → Friedman + Holm-corrected post-hoc.
2. **Vargha–Delaney A12 effect size** — per-dataset in the existing Comparison table, and
   summarized on the new Overall page.
3. **LaTeX export** of both tables via `st.code(..., language="latex")` (built-in copy button).

Bonus feature (separate work item): an **aggregated line-plots page**.

Reference: Demšar (2006), "Statistical Comparisons of Classifiers over Multiple Data Sets",
JMLR — this is the methodology reviewers will expect.

## Dependencies

- Add `scipy` as an **explicit** dependency (`uv add scipy`) — it is currently only transitive
  via statsmodels, but we call `scipy.stats` directly.
- Plotly is already a dependency; use `plotly.graph_objects` / `plotly.express` for plots.

## Part 1 — statistics library extensions (`functional/statistics.py`)

All functions are **pure** (no streamlit imports, no session state) and fully unit-tested.
Operate on Polars. Keep the module's docstring tree in sync.

### 1.1 Aggregation

```python
AGGREGATIONS: dict[str, Callable]  # keys: "median", "mean", "min", "max", "iqm"
```

- `iqm` = interquartile mean (mean of values within [Q1, Q3], inclusive). Implement it
  manually with Polars/numpy quantiles; test against a hand-computed case.
- `aggregate_per_dataset(df, *, metric_col, group_col, dataset_col, agg) -> pl.DataFrame`
  Returns a wide block matrix: one row per dataset, one column per group, cell = aggregated
  metric over that (dataset, group)'s runs. Missing (dataset, group) combinations → null.

### 1.2 Effect size

- `a12(ours: Sequence[float], theirs: Sequence[float]) -> float`
  Vargha–Delaney: P(random "ours" value beats random "theirs") + 0.5·P(tie).
  **Direction-aware wrapper**: when the metric is minimized, "beats" means *smaller*. Expose
  `a12(ours, theirs, maximize: bool)` so callers never have to flip it themselves.
  Vectorize via numpy broadcasting (no O(n·m) Python loops).
  Magnitude labels (Vargha & Vanneste thresholds on |A12 − 0.5|): negligible < 0.06,
  small < 0.14, medium < 0.21, large ≥ 0.21. Provide `a12_magnitude(a: float) -> str`.

### 1.3 Cross-dataset test (dynamic dispatch)

```python
@dataclass
class CrossDatasetResult:
    method: str                  # "wilcoxon_signed_rank" | "friedman"
    statistic: float
    p_value: float
    n_datasets: int              # complete blocks actually used
    dropped_datasets: list[str]  # datasets excluded for missing groups
    posthoc: pl.DataFrame | None # Friedman only; None otherwise
    mean_ranks: dict[str, float] # Friedman only; {} otherwise

def cross_dataset_test(block: pl.DataFrame, *, ours: str, maximize: bool,
                       alpha: float = 0.05) -> CrossDatasetResult
```

- `block` is the output of `aggregate_per_dataset`. Drop incomplete rows (any null) first
  and record them in `dropped_datasets` — Friedman and signed-rank both require complete blocks.
- **Exactly 2 groups** → `scipy.stats.wilcoxon(ours_col, theirs_col,
  alternative="greater" if maximize else "less", zero_method="wilcox")`. One-sided, paired.
- **≥3 groups** → `scipy.stats.friedmanchisquare(*columns)` (two-sided omnibus). If
  significant at `alpha`, run post-hoc: pairwise one-sided Wilcoxon signed-rank of `ours`
  vs every other group, Holm-corrected (reuse the existing Holm function from M2's
  statistics module — do not duplicate it). `posthoc` columns:
  `group, statistic, p_value, p_adjusted, significant, a12_of_aggregates`.
  Also compute `mean_ranks` (average rank of each group across datasets, rank 1 = best
  given the direction) — this is the headline number for Friedman.
- **Guards**: < 2 complete datasets → raise `ValueError` with an explanatory message
  (the page catches and shows it); 2–4 datasets → set a `low_power: bool` field on the
  result so the UI can warn that n is small.
- scipy `wilcoxon` raises on all-zero differences — catch that specific case and return
  p=1.0 with a note field rather than crashing.

### 1.4 LaTeX export

```python
def comparison_table_to_latex(df: pl.DataFrame, *, caption: str, label: str) -> str
def cross_dataset_to_latex(result: CrossDatasetResult, block: pl.DataFrame, *, caption: str, label: str) -> str
```

- booktabs style (`\toprule`/`\midrule`/`\bottomrule`), no vertical rules.
- Escape LaTeX specials in names (`_`, `%`, `&`, `#`).
- Bold the winning value per row; significance markers on p-values:
  `*` p<0.05, `**` p<0.01, `***` p<0.001 (after Holm adjustment).
- Numbers: 3 significant digits for metrics, p-values as `<0.001` below that threshold.
- Include a `% requires \usepackage{booktabs}` comment at the top of the snippet.

## Part 2 — UI

### 2.1 Comparison page (existing) — keep it lean

- Add an **A12 column** (with magnitude label, e.g. `0.71 (large)`) to the per-dataset
  comparison table. Computed on run-level values per (dataset, pair), direction-aware.
- Add an expander `LaTeX export` (icon `:material/code:`) at the bottom containing
  `st.code(comparison_table_to_latex(...), language="latex")`.
- Nothing else moves on this page. If, while integrating, the page's `main()` exceeds
  reasonable size, extract logic into `pages_modules/module_comparison/` — UI stays.

### 2.2 New page: Overall (`subpages/overall.py`)

Register in `app.py` navigation after Comparison: title "Overall", icon
`:material/leaderboard:` (page_icon likewise). Pipeline becomes
Get Started → Parameters → Metrics → Results → Comparison → Overall.

Layout, top to bottom:

1. Same guards as Comparison (experiments/datasets/metrics selected, ours-groups chosen —
   reuse the existing "our groups" selection from session state; do not invent a second
   selector).
2. **Aggregation picker**: `st.pills` single-select over `median | mean | min | max | iqm`,
   default `median`. Persist as `cross_dataset_agg` (add to `DEFAULT_STATE` **and**
   `PERSISTABLE_KEYS`).
3. Per selected metric (respect the per-metric minimize/maximize directions already stored
   in `metric_directions`):
   - The aggregated block matrix (datasets × groups) as `st.dataframe`, winner per row
     highlighted.
   - The test verdict: which test ran (signed-rank vs Friedman — say why: "2 groups" /
     "k groups"), statistic, p, n datasets, dropped datasets (warning if any), low-power
     warning when n < 5.
   - Friedman case: mean-ranks bar + the post-hoc table.
   - Median A12 across datasets for ours-vs-each-group (run-level A12 per dataset,
     then median).
   - `LaTeX export` expander with `st.code`.
4. All number-crunching calls into `functional/statistics.py`; the page file contains UI only.

### 2.3 New page: Plots (`subpages/plots.py`) — bonus work item

**Constraint to respect**: providers fetch *summary* metrics only (one scalar per run) —
there is no per-step training history. So these are **aggregated trend plots over a numeric
parameter**, not training curves. Do not attempt history fetching.

Design:

- **X axis**: selectbox over available *numeric* params (cast `params.<p>` to float; a param
  qualifies if ≥ 90% of its non-null values cast cleanly). Typical: `pop_size`.
- **Y axis**: selectbox over selected metrics (fall back to all available metrics if none
  selected).
- **Series (lines)**: one line per group, where the grouping is the existing `group_label`
  *excluding* the x-axis param (otherwise every x value is its own group and lines collapse
  to points). If the x param is not in the current group params, group_label is used as is.
- **Aggregator**: same `AGGREGATIONS` pills as the Overall page (separate session key
  `plot_agg`, persisted).
- **Band**: optional toggle "Show spread" → shaded band between Q1 and Q3 (IQR) around the
  line. Off by default.
- **Dataset handling**: pills single-select — "Aggregate across datasets" (pool all runs) or
  one specific dataset. Persisted as `plot_dataset_scope`.
- **Axis controls**: in an expander "Axis settings": four `st.number_input` fields
  (x min, x max, y min, y max), each blank/None = auto. Persist as `plot_axis_limits`
  (dict). Apply via plotly `update_layout(xaxis_range=..., yaxis_range=...)` only when set.
- **Log scale** toggles for x and y (params like pop_size are log-ish). Persisted.
- Plotly `go.Scatter` lines+markers; legend = group names (respect `NamingManager` renames
  for metrics and groups). One chart per selected metric? No — one chart, with the y metric
  chosen in the selectbox; keep the page simple.
- Data path: `fetch_experiment_data` for params and metrics, joined on `run_id` — check how
  Results/Comparison join them today and reuse that exact approach (likely via
  `RunsCache.filter_by_datasets` + the two prefixed fetches).
- Pure aggregation logic (group → x → agg(y), band quantiles) goes in
  `functional/statistics.py` or `functional/dataframes/` — unit-tested; the page is UI only.
- Register in `app.py` after Overall: title "Plots", icon `:material/show_chart:`.
- New session keys: add to `DEFAULT_STATE` + `PERSISTABLE_KEYS`.

## Part 3 — Tests (pytest, extend M2's suite)

Pure-function tests, no Streamlit imports needed:

- `a12`: identical samples → 0.5; fully separated → 1.0 (and 0.0 reversed); a hand-computed
  small case with ties; `maximize=False` flips correctly; magnitude thresholds.
- `iqm`: hand-computed case; degenerate (n < 4) behaves sanely (document the choice).
- `aggregate_per_dataset`: correct cells, nulls for missing combos.
- `cross_dataset_test`:
  - 2 groups → signed-rank path; direction correctness (construct blocks where ours is
    uniformly better under minimize → significant; same data with maximize → not).
  - 3 groups → Friedman path; non-significant omnibus → `posthoc is None`-or-empty
    (pick one, document it); significant omnibus → post-hoc Holm columns correct.
  - incomplete blocks dropped and reported; <2 complete rows raises; all-zero differences
    p=1.0 path.
  - mean-ranks correctness on a constructed block.
- LaTeX: snapshot-style assertions — contains `\toprule`, escaped `_`, bolded winner,
  `<0.001` formatting, stars match adjusted p.
- Plot aggregation helper: group/x/agg correctness + band quantiles.

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `uv run pytest`
must all pass. Boot smoke test as in M2 (headless run on a spare port, curl 200, no traceback,
kill).

## Part 4 — Process / ghp

- Read `.gh-pm.yml`. Create milestone "Milestone 3 - Cross-dataset stats, effect sizes & plots"
  (`gh milestone create`), issues (assign milestone, add to project board):
  1. statistics lib: aggregations, A12, cross-dataset dispatch (p1 high)
  2. Overall page (p1 high)
  3. A12 + LaTeX export in Comparison (p2 medium)
  4. Plots page (p2 medium)
  5. tests for all of the above (p1 high) — may be folded into 1–4's commits but track it
- One branch `m3-cross-dataset-stats` **based on the M2 branch if M2 is unmerged, else master**.
- Commits reference issue numbers. Open a PR at the end (base = master if M2 merged by then,
  else stack on M2's branch and say so in the PR body). **Do not merge.**
- Never modify `.statflow_config.yaml` (user's real config, gitignored). New persisted keys
  must have safe defaults when absent from an existing YAML.

## Out of scope (do not build)

- Per-step training curves / provider history fetching (future: `fetch_history` on the ABC).
- Nemenyi / critical-difference diagrams (post-hoc vs ours is enough for now).
- Configurable alpha in the UI (keep 0.05 constant where it is today).
