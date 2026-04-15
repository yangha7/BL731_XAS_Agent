"""
Experiment Info Manager
=======================
Manages user-provided metadata / comments for scan files and beamline events.

Data is stored as a JSON dictionary in ``exp_info/exp_info.txt``.
Keys are scan numbers (as strings, e.g. "45679") and values are
comment strings.  The dictionary is kept sorted by scan number so
that insertions for earlier scans go to the correct position.

A special ``_events`` key holds a list of beamline events (shutdowns,
optic replacements, maintenance, etc.) that are not tied to specific
scans but provide temporal context for the experiment history.

Typical usage from the agent:

    >>> import exp_info
    >>> exp_info.add_comment("45679", "calibration scan for TiO2")
    >>> exp_info.get_comment("45679")
    'calibration scan for TiO2'
    >>> exp_info.search("TiO2")
    [('45679', 'calibration scan for TiO2')]
    >>> exp_info.add_event("2026-03", "shutdown", "Major shutdown for maintenance")
    >>> exp_info.search_events("shutdown")
    [{'date': '2026-03', 'type': 'shutdown', 'description': 'Major shutdown for maintenance'}]
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


def _load_raw() -> dict:
    """Load the raw JSON from disk (includes both scan entries and _events)."""
    if not os.path.isfile(EXP_INFO_FILE):
        return {}
    try:
        with open(EXP_INFO_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {}


def _load() -> OrderedDict:
    """Load scan comments from disk (excludes _events).

    Returns an OrderedDict sorted by scan number (numerically).
    If the file doesn't exist or is empty, returns an empty OrderedDict.
    """
    raw = _load_raw()
    # Filter out the _events key — only scan entries
    scan_items = {k: v for k, v in raw.items() if not k.startswith("_")}
    sorted_items = sorted(scan_items.items(), key=lambda kv: _scan_sort_key(kv[0]))
    return OrderedDict(sorted_items)


def _save(data: OrderedDict):
    """Save scan comments to disk, preserving the _events list."""
    _ensure_dir()
    # Load existing events so we don't lose them
    raw = _load_raw()
    events = raw.get("_events", [])
    # Re-sort scan entries before saving
    scan_items = {k: v for k, v in data.items() if not k.startswith("_")}
    sorted_items = sorted(scan_items.items(), key=lambda kv: _scan_sort_key(kv[0]))
    sorted_data = OrderedDict(sorted_items)
    # Append _events at the end
    if events:
        sorted_data["_events"] = events
    with open(EXP_INFO_FILE, "w") as f:
        json.dump(sorted_data, f, indent=2)


# ---------------------------------------------------------------------------
# Date normalization for beamline events
# ---------------------------------------------------------------------------

_MONTH_NAMES = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09",
    "oct": "10", "nov": "11", "dec": "12",
}


def _normalize_date(user_input: str) -> str:
    """Convert flexible date input to ISO-like format for storage and sorting.

    Supported formats:
        "March 2026"       -> "2026-03"
        "3/2026"           -> "2026-03"
        "4/1/2026"         -> "2026-04-01"
        "2026-04-01"       -> "2026-04-01"  (passthrough)
        "2026-03"          -> "2026-03"     (passthrough)
        "January 15, 2026" -> "2026-01-15"
        "1/15/2026"        -> "2026-01-15"
    """
    s = user_input.strip()

    # Already ISO: YYYY-MM-DD or YYYY-MM
    if re.match(r'^\d{4}-\d{2}(-\d{2})?$', s):
        return s

    # "Month YYYY" or "Month DD, YYYY"
    for name, num in _MONTH_NAMES.items():
        pattern = re.compile(
            rf'^{name}\s+(\d{{1,2}}),?\s+(\d{{4}})$', re.IGNORECASE
        )
        m = pattern.match(s)
        if m:
            day = int(m.group(1))
            year = m.group(2)
            return f"{year}-{num}-{day:02d}"

        pattern2 = re.compile(rf'^{name}\s+(\d{{4}})$', re.IGNORECASE)
        m2 = pattern2.match(s)
        if m2:
            year = m2.group(1)
            return f"{year}-{num}"

    # M/D/YYYY
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"{year}-{month:02d}-{day:02d}"

    # M/YYYY (month/year only)
    m = re.match(r'^(\d{1,2})/(\d{4})$', s)
    if m:
        month, year = int(m.group(1)), m.group(2)
        return f"{year}-{month:02d}"

    # Fallback: return as-is
    return s


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
    """Return a human-readable summary of all stored comments and events.

    Format:
        Scan 45264: calibration scan for TiO2
        Scan 45611: Fe L-edge sample A
        ...

        Beamline Events:
        [2026-03] shutdown: Major shutdown for maintenance
        [2026-04-01] optic_replacement: Grating replaced
    """
    data = _load()
    events = _load_events()
    parts = []

    if data:
        lines = [f"Scan {k}: {v}" for k, v in data.items()]
        parts.append(f"Experiment info ({len(data)} scan entries):\n" + "\n".join(lines))

    if events:
        event_lines = [f"[{e['date']}] {e['type']}: {e['description']}" for e in events]
        parts.append(f"Beamline events ({len(events)} entries):\n" + "\n".join(event_lines))

    if not parts:
        return "No experiment info recorded yet."

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Beamline Events API
# ---------------------------------------------------------------------------

def _load_events() -> list[dict]:
    """Load the _events list from the JSON file.

    Returns a list of event dicts sorted by date, each with keys:
    'date' (ISO string), 'type' (str), 'description' (str).
    """
    raw = _load_raw()
    events = raw.get("_events", [])
    # Sort by date string (ISO format sorts correctly)
    return sorted(events, key=lambda e: e.get("date", ""))


def _save_events(events: list[dict]):
    """Save the events list back to disk, preserving scan entries."""
    _ensure_dir()
    raw = _load_raw()
    # Rebuild: scan entries first (sorted), then _events
    scan_items = {k: v for k, v in raw.items() if not k.startswith("_")}
    sorted_items = sorted(scan_items.items(), key=lambda kv: _scan_sort_key(kv[0]))
    sorted_data = OrderedDict(sorted_items)
    # Sort events by date and store
    sorted_events = sorted(events, key=lambda e: e.get("date", ""))
    if sorted_events:
        sorted_data["_events"] = sorted_events
    with open(EXP_INFO_FILE, "w") as f:
        json.dump(sorted_data, f, indent=2)


def add_event(date: str, event_type: str, description: str) -> str:
    """Add a beamline event (shutdown, optic replacement, maintenance, etc.).

    Parameters
    ----------
    date : str
        Date of the event. Flexible format accepted:
        "March 2026", "4/1/2026", "2026-03", "2026-04-01", etc.
    event_type : str
        Type of event: "shutdown", "optic_replacement", "maintenance", "note", etc.
    description : str
        Description of the event.

    Returns
    -------
    str
        Confirmation message.
    """
    normalized_date = _normalize_date(date)
    event_type = event_type.strip().lower()
    description = description.strip()

    if not description:
        return "No description provided for the event."

    events = _load_events()

    # Duplicate check
    for e in events:
        if (e["date"] == normalized_date and
                e["type"] == event_type and
                description.lower() in e["description"].lower()):
            return (f"Event already recorded: [{normalized_date}] {event_type}: "
                    f"\"{e['description']}\"")

    events.append({
        "date": normalized_date,
        "type": event_type,
        "description": description,
    })
    _save_events(events)
    return (f"Added beamline event: [{normalized_date}] {event_type}: "
            f"\"{description}\"")


def list_events() -> list[dict]:
    """Return all beamline events sorted by date."""
    return _load_events()


def search_events(query: str) -> list[dict]:
    """Search beamline events by keyword (case-insensitive).

    Searches across date, type, and description fields.

    Parameters
    ----------
    query : str
        Search term (e.g. "shutdown", "grating", "2026-03").
        Use "all" to return all events.

    Returns
    -------
    list of event dicts matching the query, sorted by date.
    """
    events = _load_events()
    if query.strip().lower() == "all":
        return events

    query_lower = query.lower()
    results = []
    for e in events:
        searchable = f"{e['date']} {e['type']} {e['description']}".lower()
        if query_lower in searchable:
            results.append(e)
    return results


def remove_event(index: int) -> str:
    """Remove a beamline event by its index (0-based) in the sorted list.

    Parameters
    ----------
    index : int
        Index of the event to remove (from list_events() ordering).

    Returns
    -------
    str
        Confirmation message.
    """
    events = _load_events()
    if index < 0 or index >= len(events):
        return f"Invalid event index {index}. There are {len(events)} events (0-{len(events)-1})."
    removed = events.pop(index)
    _save_events(events)
    return (f"Removed event: [{removed['date']}] {removed['type']}: "
            f"\"{removed['description']}\"")


def events_summary() -> str:
    """Return a human-readable summary of all beamline events.

    Format:
        [0] [2026-03] shutdown: Major shutdown for maintenance
        [1] [2026-04-01] optic_replacement: Grating replaced
    """
    events = _load_events()
    if not events:
        return "No beamline events recorded yet."
    lines = [f"[{i}] [{e['date']}] {e['type']}: {e['description']}"
             for i, e in enumerate(events)]
    return f"Beamline events ({len(events)} entries):\n" + "\n".join(lines)
