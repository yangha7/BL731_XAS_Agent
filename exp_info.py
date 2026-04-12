"""
Experiment Info Manager
=======================
Manages user-provided metadata / comments for scan files.

Data is stored as a JSON dictionary in ``exported_data/exp_info.txt``.
Keys are scan numbers (as strings, e.g. "45679") and values are
comment strings.  The dictionary is kept sorted by scan number so
that insertions for earlier scans go to the correct position.

Typical usage from the agent:

    >>> import exp_info
    >>> exp_info.add_comment("45679", "calibration scan for TiO2")
    >>> exp_info.get_comment("45679")
    'calibration scan for TiO2'
    >>> exp_info.search("TiO2")
    [('45679', 'calibration scan for TiO2')]
"""

import os
import json
import re
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# exp_info.txt lives in its own persistent directory, separate from
# exported_data/ (which is temporary and may be deleted).
EXP_INFO_DIR = os.environ.get("EXP_INFO_DIR", "exp_info")
EXP_INFO_FILE = os.path.join(EXP_INFO_DIR, "exp_info.txt")


def _ensure_dir():
    """Create the exp_info/ directory if it doesn't exist."""
    os.makedirs(EXP_INFO_DIR, exist_ok=True)


def _load() -> OrderedDict:
    """Load the exp_info dictionary from disk.

    Returns an OrderedDict sorted by scan number (numerically).
    If the file doesn't exist or is empty, returns an empty OrderedDict.
    """
    if not os.path.isfile(EXP_INFO_FILE):
        return OrderedDict()
    try:
        with open(EXP_INFO_FILE, "r") as f:
            data = json.load(f)
        # Sort by numeric scan number
        sorted_items = sorted(data.items(), key=lambda kv: _scan_sort_key(kv[0]))
        return OrderedDict(sorted_items)
    except (json.JSONDecodeError, ValueError):
        return OrderedDict()


def _save(data: OrderedDict):
    """Save the exp_info dictionary to disk (pretty-printed JSON)."""
    _ensure_dir()
    # Re-sort before saving
    sorted_items = sorted(data.items(), key=lambda kv: _scan_sort_key(kv[0]))
    sorted_data = OrderedDict(sorted_items)
    with open(EXP_INFO_FILE, "w") as f:
        json.dump(sorted_data, f, indent=2)


def _scan_sort_key(scan_key: str):
    """Extract a numeric sort key from a scan identifier.

    Handles keys like "45679", "SigScan45679", "TimeScan3028", etc.
    Falls back to the string itself for non-numeric keys.
    """
    m = re.search(r'(\d+)', scan_key)
    if m:
        return int(m.group(1))
    return scan_key


def _normalize_key(scan_id: str) -> str:
    """Normalize a scan identifier to just the number string.

    "SigScan45679" -> "45679"
    "45679" -> "45679"
    "260401/SigScan45679" -> "45679"
    "TimeScan3028" -> "3028"
    """
    scan_id = scan_id.strip()
    # Remove directory prefix
    if "/" in scan_id:
        scan_id = scan_id.rsplit("/", 1)[1]
    # Remove SigScan/TimeScan prefix
    m = re.search(r'(\d+)', scan_id)
    if m:
        return m.group(1)
    return scan_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_comment(scan_id: str, comment: str) -> str:
    """Add or update a comment for a scan.

    If the scan already has a comment, the new comment is appended
    (separated by ' | ') unless the existing comment already contains
    the new text (duplicate check).

    Parameters
    ----------
    scan_id : str
        Scan identifier (e.g. "45679", "SigScan45679").
    comment : str
        The comment / metadata to store.

    Returns
    -------
    str
        Confirmation message.
    """
    key = _normalize_key(scan_id)
    comment = comment.strip()
    if not comment:
        return f"No comment provided for scan {key}."

    data = _load()
    existing = data.get(key, "")

    if existing:
        # Check if the new comment is already contained in the existing one
        if comment.lower() in existing.lower():
            return f"Scan {key} already has this info: \"{existing}\""
        # Append the new comment
        data[key] = existing + " | " + comment
        _save(data)
        return f"Appended comment for scan {key}. Full entry: \"{data[key]}\""
    else:
        data[key] = comment
        _save(data)
        return f"Added comment for scan {key}: \"{comment}\""


def set_comment(scan_id: str, comment: str) -> str:
    """Set (replace) the comment for a scan, overwriting any existing one.

    Parameters
    ----------
    scan_id : str
        Scan identifier.
    comment : str
        The comment to store (replaces existing).

    Returns
    -------
    str
        Confirmation message.
    """
    key = _normalize_key(scan_id)
    comment = comment.strip()
    data = _load()
    data[key] = comment
    _save(data)
    return f"Set comment for scan {key}: \"{comment}\""


def get_comment(scan_id: str) -> str:
    """Get the comment for a scan.

    Returns the comment string, or empty string if none exists.
    """
    key = _normalize_key(scan_id)
    data = _load()
    return data.get(key, "")


def get_all() -> OrderedDict:
    """Return the full exp_info dictionary (sorted by scan number)."""
    return _load()


def search(query: str) -> list[tuple[str, str]]:
    """Search comments for a keyword or phrase (case-insensitive).

    Parameters
    ----------
    query : str
        Search term (e.g. "TiO2", "calibration", "Fe L-edge").

    Returns
    -------
    list of (scan_number, comment) tuples matching the query,
    sorted by scan number (most recent last).
    """
    data = _load()
    query_lower = query.lower()
    results = []
    for key, comment in data.items():
        if query_lower in comment.lower():
            results.append((key, comment))
    return results


def search_latest(query: str) -> tuple[str, str] | None:
    """Find the most recent (highest scan number) scan matching a query.

    Returns (scan_number, comment) or None if no match.
    """
    matches = search(query)
    if not matches:
        return None
    # Already sorted by scan number; last one is the most recent
    return matches[-1]


def remove_comment(scan_id: str) -> str:
    """Remove the comment for a scan.

    Returns a confirmation message.
    """
    key = _normalize_key(scan_id)
    data = _load()
    if key in data:
        del data[key]
        _save(data)
        return f"Removed comment for scan {key}."
    return f"No comment found for scan {key}."


def bulk_add(entries: dict[str, str]) -> str:
    """Add comments for multiple scans at once.

    Parameters
    ----------
    entries : dict
        Mapping of scan_id -> comment.

    Returns
    -------
    str
        Summary of what was added/updated.
    """
    data = _load()
    added = 0
    updated = 0
    for scan_id, comment in entries.items():
        key = _normalize_key(scan_id)
        comment = comment.strip()
        if not comment:
            continue
        existing = data.get(key, "")
        if existing:
            if comment.lower() not in existing.lower():
                data[key] = existing + " | " + comment
                updated += 1
        else:
            data[key] = comment
            added += 1
    _save(data)
    return f"Bulk update: {added} added, {updated} updated."


def summary() -> str:
    """Return a human-readable summary of all stored comments.

    Format:
        Scan 45264: calibration scan for TiO2
        Scan 45611: Fe L-edge sample A
        ...
    """
    data = _load()
    if not data:
        return "No experiment info recorded yet."
    lines = [f"Scan {k}: {v}" for k, v in data.items()]
    return f"Experiment info ({len(data)} entries):\n" + "\n".join(lines)
