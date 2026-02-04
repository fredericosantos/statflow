"""
Shared UI components for selection, ordering, and renaming.

This module provides reusable Streamlit components for:
- Selecting items from a list
- Ordering selected items
- Renaming items (Display Name / LaTeX Name)

components/selection_ui.py
├── SelectionManager                # Unified selection, ordering, renaming component
├── render_item_selector()          # Generic pill/multiselect for items
├── render_item_ordering()          # Generic item reordering UI
└── render_renaming_ui()            # Generic renaming UI
"""

import streamlit as st
from streamlit_sortables import sort_items


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
                reorder_label(g, cached_order, self.cache_param_order)
                for g in cached_groups
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
            )
            
            if selected is not None:
                st.session_state[self.session_key] = list(selected)
                self._save_to_cache(selected)
        
        return list(selected) if selected else []
    
    def _render_ordering(self, items: list[str]) -> list[str]:
        """Render the ordering UI."""
        if not items:
            return items
        
        with st.expander("Order Items", expanded=False, icon=":material/swap_vert:"):
            sort_key = f"order_{self.session_key}_{len(items)}_{hash(tuple(sorted(items)))}"
            ordered = sort_items(items, key=sort_key)
            st.session_state[self.session_key] = ordered
        
        return ordered
    
    def _render_renaming(self, items: list[str]) -> None:
        """Render the renaming UI."""
        render_renaming_ui(
            items=items,
            session_key_renames=self.renames_session_key,
            label="Rename Items",
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

    # Determine default selection
    if default is None:
        current = st.session_state.get(session_key)
        if current is not None:
            # Filter current selection to ensure it only includes valid options
            default = [x for x in current if x in options]
        else:
            default = options

    selected_items = st.pills(
        label,
        options=options,
        default=default,
        key=f"selector_{session_key}{key_suffix}",
        selection_mode="multi",
    )
    
    # Update session state directly
    st.session_state[session_key] = selected_items
    return selected_items


def render_item_ordering(
    items: list[str],
    session_key: str,
    label: str = "Order Items",
    key_suffix: str = "",
) -> list[str]:
    """Render a generic item ordering UI.

    Args:
        items: List of items to order.
        session_key: Session state key to store ordered items.
        label: Label for the section.
        key_suffix: Suffix for widget keys.

    Returns:
        Ordered list of items.
    """
    if not items:
        return items

    st.space()
    st.markdown(label)
    
    # Generate a unique key based on content to force update when items change
    # Include length and hash of sorted tuple to detect content changes efficiently
    sort_key = f"order_{session_key}{key_suffix}_{len(items)}_{hash(tuple(sorted(items)))}"
    
    ordered_items = sort_items(items, key=sort_key)
    
    # Update session state
    st.session_state[session_key] = ordered_items
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
    # Get current renames, merging defaults with user customizations
    saved_renames: dict = st.session_state.get(session_key_renames, {})
    current_renames = saved_renames.copy()

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
            entry = current_renames.get(item)
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
            new_display = cols[1].text_input(
                "Display",
                value=current_display,
                key=f"rename_display_{session_key_renames}_{item}{key_suffix}",
                label_visibility="collapsed",
                icon=":material/edit:"
            )

            # Column 3: LaTeX name input
            new_latex = cols[2].text_input(
                "LaTeX",
                value=current_latex,
                key=f"rename_latex_{session_key_renames}_{item}{key_suffix}",
                label_visibility="collapsed",
                icon=":material/edit:"
            )

            # Column 4: LaTeX preview
            cols[3].latex(new_latex)

            # Update mapping if changed
            if new_display != current_display or new_latex != current_latex:
                current_renames[item] = {
                    "display_name": new_display,
                    "latex_name": new_latex
                }

        # Update session state
        st.session_state[session_key_renames] = current_renames
