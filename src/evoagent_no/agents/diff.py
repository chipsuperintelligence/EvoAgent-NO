"""SEARCH/REPLACE edits: how an agent changes part of a generator without rewriting all of it.

An edit in a reply looks like

    <<<<<<< SEARCH
    lines copied exactly from the current EVOLVE-BLOCK
    =======
    the lines that replace them
    >>>>>>> REPLACE

A reply may hold several. Each SEARCH must match exactly one place in the block, so an edit never
lands somewhere the agent did not mean. EVOLVE-BLOCK marker lines copied into an edit are ignored:
the edit applies to what is between them.
"""

from __future__ import annotations

import re

_EDIT = re.compile(r"<{7} SEARCH\n(.*?)\n?={7}\n(.*?)\n?>{7} REPLACE", re.DOTALL)


_MARKER = re.compile(r"^(<{7} SEARCH|={7}|>{7} REPLACE)\s*$", re.MULTILINE)
_BLOCK_MARKER = re.compile(r"^[ \t]*# EVOLVE-BLOCK-(START|END)[ \t]*\n?", re.MULTILINE)


def parse_edits(reply: str) -> list[tuple[str, str]]:
    """(search, replace) pairs, in order. Empty if the reply has none. Raises ValueError when an
    edit's text still holds a marker line, which means the reply's edit format is broken."""
    # A stray ======= right before >>>>>>> REPLACE is a common slip; it is not part of the text.
    edits = [(search, re.sub(r"\n?={7}[ \t]*$", "", replace)) for search, replace in _EDIT.findall(reply)]
    for i, (search, replace) in enumerate(edits, start=1):
        if _MARKER.search(search) or _MARKER.search(replace):
            raise ValueError(f"edit {i} is malformed: a <<<<<<< / ======= / >>>>>>> marker is inside its text")
    return edits


def apply_edits(block: str, edits: list[tuple[str, str]]) -> str:
    """Apply edits one after another; raise ValueError if a SEARCH is missing or ambiguous."""
    for i, (search, replace) in enumerate(edits, start=1):
        search, replace = _BLOCK_MARKER.sub("", search), _BLOCK_MARKER.sub("", replace)
        if not search.strip():
            raise ValueError(f"edit {i} has an empty SEARCH")
        count = block.count(search)
        if count == 0:
            raise ValueError(f"edit {i}: SEARCH text not found in the EVOLVE-BLOCK")
        if count > 1:
            raise ValueError(f"edit {i}: SEARCH text appears {count} times; include more lines to make it unique")
        block = block.replace(search, replace, 1)
    return block
