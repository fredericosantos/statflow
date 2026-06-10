"""
Shared UI components for selection, ordering, and renaming.

This module provides reusable Streamlit components for:
- Selecting items from a list
- Ordering selected items
- Renaming items (Display Name / LaTeX Name)

components/selection_ui.py
├── SelectionManager                # Unified selection, ordering, renaming component
├── render_item_selector()          # Generic pill/multiselect for items
├── render_selection_pills()        # Controlled multi-select + Select/Deselect all
├── render_item_ordering()          # Generic item reordering UI
└── render_renaming_ui()            # Generic renaming UI
"""

from collections.abc import Callable
from typing import Literal, cast

import streamlit as st
from streamlit_sortables import sort_items

from statflow.config import SessionState
from statflow.managers.naming import NamingManager


def reorder_label(label: str, from_order: list[str], to_order: list[str]) -> str:
    """Reorder label parts from one param order to another.

    Example: 'a=1, b=2' with from_order=['a','b'] to_order=['b','a'] -> 'b=2, a=1'
    """
    parts = [p.strip() for p in label.split(", ")]
    param_parts = {}
    for part in parts:
        if "=" in part:
            param_name = part.split("=")[0]
            param_parts[param_name] = part

    reordered = [param_parts.get(p, f"{p}=?") for p in to_order if p in param_parts]
    return ", ".join(reordered)


class SelectionManager:
    """Unified selection, ordering, and renaming component.

    Args:
        options: List of available options.
        session_key: Session state key for selected items.
        label: Label for the selection UI.
        enable_ordering: Enable drag-to-reorder UI.
        enable_renaming: Enable renaming UI.
        renames_session_key: Session state key for renames dict.
        use_fragment: Wrap in @st.fragment to prevent scroll-to-top.
        cache_key: Key for caching selections (e.g., sorted param combo).
        cache_param_order: Current param order for label reordering.
    """

    def __init__(
        self,
        options: list[str],
        session_key: str,
        label: str = "Select Items",
        enable_ordering: bool = True,
        enable_renaming: bool = True,
        renames_session_key: str | None = None,
        use_fragment: bool = False,
        cache_key: str | None = None,
        cache_param_order: list[str] | None = None,
        on_change: Callable[[], None] | None = None,
    ):
        self.options = options
        self.session_key = session_key
        self.label = label
        self.enable_ordering = enable_ordering
        self.enable_renaming = enable_renaming
        self.renames_session_key = renames_session_key or f"{session_key}_renames"
        self.use_fragment = use_fragment
        self.cache_key = cache_key
        self.cache_param_order = cache_param_order
        self.on_change = on_change

        self._default = self._compute_default()

    def _compute_default(self) -> list[str]:
        """Compute default selection, using cache if available."""
        if not self.cache_key:
            # No caching - use session state or all options
            current = st.session_state.get(self.session_key)
            if current is not None:
                return [x for x in current if x in self.options] or self.options
            return self.options

        # Use cache
        cache = st.session_state.get("group_selections_cache", {})
        cached_data = cache.get(self.cache_key)

        if cached_data is None:
            return self.options

        cached_groups = cached_data.get("groups", [])
        cached_order = cached_data.get("param_order", self.cache_param_order or [])

        # Reorder if param order changed
        if self.cache_param_order and cached_order != self.cache_param_order:
            reordered = [
                reorder_label(g, cached_order, self.cache_param_order) for g in cached_groups
            ]
        else:
            reordered = cached_groups

        # Validate against available options
        valid = [g for g in reordered if g in self.options]
        if not valid:
            return self.options

        # Update session state with reordered groups
        st.session_state[self.session_key] = valid
        return valid

    def _save_to_cache(self, selected: list[str]) -> None:
        """Save selection to cache if caching is enabled."""
        if not self.cache_key:
            return

        cache = st.session_state.get("group_selections_cache", {})
        cache[self.cache_key] = {
            "groups": list(selected),
            "param_order": self.cache_param_order or [],
        }
        st.session_state["group_selections_cache"] = cache
        SessionState.save_to_config()

    def _get_display_name(self, item: str) -> str:
        """Get display name for item using renames map."""
        return NamingManager.get_name(item, self.renames_session_key)

    def _render_selection(self) -> list[str]:
        """Render the pills selection UI."""
        if not self.options:
            st.info("No items available.")
            return []

        with st.expander(self.label, expanded=True, icon=":material/category:"):
            selected = st.pills(
                "Select items",
                options=self.options,
                default=self._default,
                selection_mode="multi",
                key=f"selector_{self.session_key}",
                label_visibility="collapsed",
                format_func=lambda x: self._get_display_name(x),
            )

            if selected is not None:
                if st.session_state.get(self.session_key) != list(selected):
                    st.session_state[self.session_key] = list(selected)
                    self._save_to_cache(selected)
                    SessionState.save_to_config()
                    if self.on_change:
                        self.on_change()

        return list(selected) if selected else []

    def _render_ordering(self, items: list[str]) -> list[str]:
        """Render the ordering UI."""
        ordered = render_item_ordering(
            items=items,
            session_key=self.session_key,
            label="Order Groups",
            renames_session_key=self.renames_session_key,
        )
        if list(ordered) != list(items):
            self._save_to_cache(ordered)
            if self.on_change:
                self.on_change()
        return ordered

    def _render_renaming(self, items: list[str]) -> None:
        """Render the renaming UI."""
        render_renaming_ui(
            items=items,
            session_key_renames=self.renames_session_key,
            label="Rename Groups",
        )

    def _render_all(self) -> list[str]:
        """Render all components."""
        selected = self._render_selection()

        if selected and self.enable_ordering:
            selected = self._render_ordering(selected)

        if selected and self.enable_renaming:
            self._render_renaming(selected)

        return selected

    def render(self) -> list[str]:
        """Render the complete selection UI.

        Returns:
            List of selected items (in order if ordering enabled).
        """
        if self.use_fragment:

            @st.fragment
            def fragment_wrapper():
                return self._render_all()

            fragment_wrapper()
            return st.session_state.get(self.session_key, [])
        else:
            return self._render_all()


def render_item_selector(
    options: list[str],
    session_key: str,
    label: str = "Select Items",
    default: list[str] | None = None,
    key_suffix: str = "",
    renames_session_key: str | None = None,
) -> list[str]:
    """Render a generic item selector UI.

    Args:
        options: List of available options.
        session_key: Session state key to store selection.
        label: Label for the selector.
        default: Default selected items. If None, uses current session state or all options.
        key_suffix: Suffix for widget keys to avoid conflicts.

    Returns:
        List of selected items.
    """
    if not options:
        st.info("No items available.")
        return []

    # Render section header
    st.markdown(f"#### {label}")

    # Determine default selection
    if default is None:
        current = st.session_state.get(session_key)
        if current is not None:
            # Filter current selection to ensure it only includes valid options
            default = [x for x in current if x in options]
        else:
            default = options

    # Prepare formatter
    def format_option(option: str) -> str:
        if renames_session_key:
            return NamingManager.get_name(option, renames_session_key)
        return option

    # st.pills with selection_mode="multi" returns list[V]; ty can't resolve V=str
    # from the overload, so we cast the result explicitly.
    selected_items: list[str] = cast(
        list[str],
        st.pills(
            label,
            options=options,
            default=default,
            key=f"selector_{session_key}{key_suffix}",
            selection_mode="multi",
            label_visibility="collapsed",
            format_func=format_option,
        )
        or [],
    )

    # Update session state directly
    if st.session_state.get(session_key) != selected_items:
        st.session_state[session_key] = selected_items
        SessionState.save_to_config()

    return selected_items


def render_selection_pills(
    options: list[str],
    session_key: str,
    *,
    label: str,
    format_func: Callable[[str], str] | None = None,
    on_change: Callable[[], None] | None = None,
    label_visibility: Literal["visible", "hidden", "collapsed"] = "visible",
) -> list[str]:
    """Controlled multi-select pills with Select all / Deselect all.

    Selection lives entirely in the widget's own key (``sel_<session_key>``) and
    no ``default=`` is passed, which is the Streamlit-blessed "controlled widget"
    pattern: it avoids the "default value but also set via Session State"
    warning and lets the action buttons set the selection directly (reliably).

    The key is seeded once (all options selected) and clamped to the current
    options each render, so a stale config never crashes the widget. The result
    is mirrored into ``session_key`` and persisted. Returns the selected items.

    Args:
        options: Available options.
        session_key: Session-state key to mirror the selection into.
        label: Pills label.
        format_func: Optional display formatter for each option.
        on_change: Optional callback fired on any selection change (manual or
            via the action buttons).
        label_visibility: Passed through to ``st.pills``.
    """
    if not options:
        return []

    widget_key = f"sel_{session_key}"

    # Seed once (all selected), else clamp the existing selection to the current
    # options. Both happen *before* the widget is instantiated, so no warning.
    if widget_key not in st.session_state:
        saved = st.session_state.get(session_key)
        st.session_state[widget_key] = (
            [x for x in saved if x in options] if saved else list(options)
        )
    else:
        st.session_state[widget_key] = [x for x in st.session_state[widget_key] if x in options]

    def _select_all() -> None:
        st.session_state[widget_key] = list(options)
        if on_change:
            on_change()

    def _deselect_all() -> None:
        st.session_state[widget_key] = []
        if on_change:
            on_change()

    col_sel, col_desel, _ = st.columns([1, 1, 4], vertical_alignment="center")
    col_sel.button(
        "Select all",
        icon=":material/select_all:",
        key=f"select_all_{session_key}",
        on_click=_select_all,
        width="stretch",
    )
    col_desel.button(
        "Deselect all",
        icon=":material/deselect:",
        key=f"deselect_all_{session_key}",
        on_click=_deselect_all,
        width="stretch",
    )

    selected = st.pills(
        label,
        options=options,
        key=widget_key,
        selection_mode="multi",
        format_func=format_func or (lambda x: x),
        label_visibility=label_visibility,
        on_change=on_change,
    )
    selected = list(selected or [])

    # Mirror into session_key when the *set* changes — an order-only difference
    # (e.g. from a downstream "order items" step writing session_key) is left
    # intact rather than clobbered by the pills' ordering.
    if set(st.session_state.get(session_key) or []) != set(selected):
        st.session_state[session_key] = selected
        SessionState.save_to_config()

    return selected


def render_item_ordering(
    items: list[str],
    session_key: str,
    label: str = "Order Items",
    key_suffix: str = "",
    renames_session_key: str | None = None,
) -> list[str]:
    """Render a generic item ordering UI.

    Args:
        items: List of items to order.
        session_key: Session state key to store ordered items.
        label: Label for the section.
        key_suffix: Suffix for widget keys.
        renames_session_key: Session state key for renames dict.

    Returns:
        Ordered list of items.
    """
    if not items:
        return items

    with st.expander(label, expanded=False, icon=":material/swap_vert:"):
        # helper to get display name
        def get_display_name(item: str) -> str:
            if renames_session_key:
                return NamingManager.get_name(item, renames_session_key)
            return item

        # Map display names to original items to reverse lookup later
        # If duplicates exist in display names, we might have an issue, but we'll assume uniqueness for UI purposes
        # or just take the first match. Ideally display names should be unique.
        display_map = {get_display_name(item): item for item in items}
        display_items = [get_display_name(item) for item in items]

        # Generate a unique key based on content to force update when items change
        # Include length and hash of sorted tuple to detect content changes efficiently
        # Use display_items to ensure widget updates when names are edited
        sort_key = (
            f"order_{session_key}{key_suffix}_{len(items)}_{hash(tuple(sorted(display_items)))}"
        )

        ordered_display_items = sort_items(display_items, key=sort_key)

    # Map back to original items
    ordered_items = [display_map[d] for d in ordered_display_items if d in display_map]

    # Check if any items were lost (e.g. if display names changed dynamically), fallback to generic
    if len(ordered_items) != len(items):
        ordered_items = items

    # Update session state
    if st.session_state.get(session_key) != ordered_items:
        st.session_state[session_key] = ordered_items
        SessionState.save_to_config()

    return ordered_items


def render_renaming_ui(
    items: list[str],
    session_key_renames: str,
    label: str = "Rename Items",
    key_suffix: str = "",
) -> None:
    """Render UI for renaming items with display and LaTeX names.

    Args:
        items: List of items to rename.
        session_key_renames: Session state key to store/retrieve renaming dictionary.
        label: Label for the expander.
        key_suffix: Suffix for widget keys.
    """
    # Get current renames
    saved_renames: dict = st.session_state.get(session_key_renames, {})

    # Callback to handle updates immediately before script rerun
    def _update_rename(item_key: str, key_type: str, widget_key: str):
        new_value = st.session_state[widget_key]
        renames = st.session_state.get(session_key_renames, {})

        # Ensure entry exists as dict
        if item_key not in renames:
            renames[item_key] = {"display_name": item_key, "latex_name": item_key}
        elif isinstance(renames[item_key], str):
            renames[item_key] = {"display_name": renames[item_key], "latex_name": renames[item_key]}

        # Update specific field
        if key_type == "display":
            renames[item_key]["display_name"] = new_value
        elif key_type == "latex":
            renames[item_key]["latex_name"] = new_value

        st.session_state[session_key_renames] = renames
        SessionState.save_to_config()

    with st.expander(label, expanded=False, icon=":material/edit:"):
        st.markdown("Customize names for display and LaTeX export:")

        # Header row
        col_sizes = [2, 2, 2, 3]
        cols = st.columns(col_sizes, vertical_alignment="center")
        cols[0].markdown("**Original**", text_alignment="center")
        cols[1].markdown("**Display Name**", text_alignment="center")
        cols[2].markdown("**LaTeX Name**", text_alignment="center")
        cols[3].markdown("**LaTeX Preview**", text_alignment="center")

        for item in items:
            # Get current values from mapping
            entry = saved_renames.get(item)
            if isinstance(entry, dict):
                current_display = entry.get("display_name", item)
                current_latex = entry.get("latex_name", item)
            elif isinstance(entry, str):
                # Backward compatibility/simple string
                current_display = entry
                current_latex = entry
            else:
                current_display = item
                current_latex = item

            cols = st.columns(col_sizes, vertical_alignment="center")

            # Column 1: Original name
            cols[0].code(item)

            # Column 2: Display name input
            display_key = f"rename_display_{session_key_renames}_{item}{key_suffix}"
            cols[1].text_input(
                "Display",
                value=current_display,
                key=display_key,
                label_visibility="collapsed",
                icon=":material/edit:",
                on_change=_update_rename,
                args=(item, "display", display_key),
            )

            # Column 3: LaTeX name input
            latex_key = f"rename_latex_{session_key_renames}_{item}{key_suffix}"
            new_latex = cols[2].text_input(
                "LaTeX",
                value=current_latex,
                key=latex_key,
                label_visibility="collapsed",
                icon=":material/edit:",
                on_change=_update_rename,
                args=(item, "latex", latex_key),
            )

            # Column 4: LaTeX preview
            cols[3].latex(new_latex)
