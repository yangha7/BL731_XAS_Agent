"""
XAS Data Utilities
==================
Parsing, normalization, export, and analysis functions for beamline XAS scan data.
Includes Athena-style pre-edge subtraction / post-edge normalization,
and Savitzky-Golay smoothed 1st & 2nd derivatives.
All file I/O is on-demand — nothing is loaded at import time.
"""

import os
import re
import glob
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import UnivariateSpline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADER_LINES = 15          # Default lines before the column-header row (may vary by scan mode)
ENERGY_COL_CANDIDATES = [
    "Mono eV Calib high E Grating",
    "Mono eV Calib low E Grating",
    "Energy_eV",                      # exported two-column files
]
TEY_COL = "TEY UHV XAS"
TFY_COL = "TFY UHV XAS"
I0_COL = "I0"
MCP_COL = "MCP Np"

SIGNAL_COLUMNS = {
    "TEY": TEY_COL,
    "TFY": TFY_COL,
    "MCP": MCP_COL,
    "I0": I0_COL,
}

EXPORT_DIR = os.environ.get("XAS_EXPORT_DIR", "exported_data")
DATA_DIR = os.environ.get("XAS_DATA_DIR", "731_Data")

# ---------------------------------------------------------------------------
# Parsing (on-demand)
# ---------------------------------------------------------------------------

def _extract_float(line: str) -> float:
    m = re.search(r":\s*([\d.Ee+-]+)", line)
    return float(m.group(1)) if m else 0.0


def _extract_int(line: str) -> int:
    m = re.search(r":\s*(\d+)", line)
    return int(m.group(1)) if m else 0


def parse_header(filepath: str) -> dict:
    """Read only the first 15 lines to extract metadata.

    Handles two scan modes:
      - Simple mode (line 3 = "Start, Stop, Increment"):
            line 4 = scan_type, line 5 = "Start: ...", line 6 = "Stop: ...",
            line 7 = "Increment: ...", line 5 stored as scan_file = ""
      - Region file mode (line 3 = "From File"):
            line 4 = scan_type, line 5 = region file path
    Lines 9-12 are the same in both modes.
    """
    meta = {}
    with open(filepath, "r") as f:
        lines = [next(f).rstrip("\n") for _ in range(HEADER_LINES)]
    meta["date"] = lines[0].replace("Date: ", "").strip() if lines[0].startswith("Date:") else ""
    meta["scan_type"] = lines[3].strip()

    # Detect scan mode from line 3 (index 2)
    scan_mode = lines[2].strip()
    meta["scan_mode"] = scan_mode
    if scan_mode == "From File":
        meta["scan_file"] = lines[4].strip()
    else:
        # Simple mode: line 5 is "Start: ...", not a file path
        meta["scan_file"] = ""
        meta["start"] = _extract_float(lines[4])
        meta["stop"] = _extract_float(lines[5])
        meta["increment"] = _extract_float(lines[6])

    meta["delay_after_move"] = _extract_float(lines[8])
    meta["count_time"] = _extract_float(lines[9])
    meta["scan_number"] = _extract_int(lines[10])
    meta["bidirectional"] = "Yes" in lines[11]
    meta["filename"] = os.path.basename(filepath)
    meta["filepath"] = filepath
    return meta


def _detect_header_lines(filepath: str) -> int:
    """Detect the number of header lines before the column-header row.

    Scans for the line containing 'Time of Day' (always the first column
    in full SigScan files).  Returns the 0-indexed row number of that line,
    so skiprows=N skips everything before it.
    Returns -1 if no 'Time of Day' line is found (i.e. not a SigScan file).
    """
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if line.startswith("Time of Day"):
                return i
            if i > 30:  # safety limit
                break
    return -1  # not a SigScan file


def load_scan(filepath: str) -> pd.DataFrame:
    """Load a scan or data file into a DataFrame.

    Works uniformly on any tab/comma/space-separated data file:
      - Full SigScan files (55+ columns with 15-line header)
      - Calibrated copies (same format, in exported_data/calibrated/)
      - Renamed copies (same format, in exported_data/renamed/)
      - Two-column exported files (e.g. Energy_eV <tab> TEY / I0)
      - Any generic columnar text file
    """
    skip = _detect_header_lines(filepath)
    if skip >= 0:
        # Standard SigScan file — skip header lines before column row
        df = pd.read_csv(filepath, sep="\t", skiprows=skip, header=0, engine="python")
        df.columns = [c.strip() for c in df.columns]
        return df

    # Not a SigScan file — auto-detect format
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find the column-header / first-data line
    header_line_idx = None
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = re.split(r'[\t,\s]+', stripped)
        numeric_count = sum(1 for t in tokens if re.match(r'^[+-]?\d*\.?\d+([eE][+-]?\d+)?$', t))
        if numeric_count >= 2 and numeric_count >= len(tokens) * 0.5:
            data_start = i
            break
        else:
            header_line_idx = i

    # Detect separator
    if data_start < len(lines):
        sample = lines[data_start]
        if "\t" in sample:
            sep = "\t"
        elif "," in sample:
            sep = ","
        else:
            sep = r"\s+"
    else:
        sep = "\t"

    # If we found a text header line just before data, use it as column names
    if header_line_idx is not None and header_line_idx == data_start - 1:
        skip_rows = header_line_idx
        hdr = 0
    elif header_line_idx is not None:
        skip_rows = header_line_idx
        hdr = 0
    else:
        skip_rows = data_start
        hdr = None

    try:
        df = pd.read_csv(filepath, sep=sep, skiprows=skip_rows,
                         header=hdr, engine="python", comment="#")
    except Exception:
        df = pd.read_csv(filepath, sep=sep, skiprows=data_start,
                         header=None, engine="python", comment="#")

    df.columns = [str(c).strip() for c in df.columns]

    # Ensure numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(how="all", inplace=True)

    return df


def resolve_scan_id(raw: str) -> str:
    """Normalize scan id: '45611' -> 'SigScan45611'.

    Also handles subdirectory prefixes: 'subdir/45611' -> 'subdir/SigScan45611'.
    """
    raw = raw.strip()
    # Handle subdirectory prefix (e.g. "subdir/45611")
    if "/" in raw:
        parts = raw.rsplit("/", 1)
        prefix, name = parts[0], parts[1]
        if not name.startswith("SigScan"):
            name = "SigScan" + name
        return f"{prefix}/{name}"
    if not raw.startswith("SigScan"):
        raw = "SigScan" + raw
    return raw


def scan_filepath(scan_id: str, data_dir: str = DATA_DIR) -> str:
    """Return the full path for a scan id, searching recursively through subdirectories.

    Supports both plain IDs ('45616') and subdirectory-prefixed IDs ('subdir/45616').
    Also finds suffixed variants like 'SigScan45264_cal.txt' when searching for '45264'.
    Raises FileNotFoundError if the scan file is not found anywhere under data_dir.
    """
    sid = resolve_scan_id(scan_id)
    # If sid contains a path separator, treat it as a relative path under data_dir
    if "/" in sid:
        fp = os.path.join(data_dir, f"{sid}.txt")
        if os.path.isfile(fp):
            return fp
        # Fall through to recursive search using just the filename part
        basename = sid.rsplit("/", 1)[1]
    else:
        basename = sid
        # Fast path: check top-level directory first
        fp = os.path.join(data_dir, f"{basename}.txt")
        if os.path.isfile(fp):
            return fp
    # Search recursively in subdirectories — exact match first
    target = f"{basename}.txt"
    for root, _dirs, files in os.walk(data_dir):
        if target in files:
            return os.path.join(root, target)
    # Second pass: find suffixed variants (e.g. SigScan45264_cal.txt)
    prefix = basename  # e.g. "SigScan45264"
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.startswith(prefix) and f.endswith(".txt") and f != target:
                return os.path.join(root, f)
    raise FileNotFoundError(f"Scan file not found: {target} (searched {data_dir} recursively)")


def list_scan_files(data_dir: str = DATA_DIR, date_filter: str | None = None) -> list[str]:
    """Return sorted list of scan IDs by recursively listing the data directory.

    Scans in subdirectories are included with their relative path prefix,
    e.g. '260401/SigScan45650' for files in 731_Data/260401/.

    Parameters
    ----------
    data_dir : str
        Root data directory.
    date_filter : str or None
        Optional date filter. Accepts formats like:
        - '260401' or '2026-04-01' or '20260401' — single date
        - '260401-260403' — date range (inclusive)
        - 'today', 'yesterday', 'this_week', 'last_week'
        If None, returns all scans.
    """
    results = []
    # Determine which date directories to include
    allowed_dates = _resolve_date_filter(date_filter) if date_filter else None

    for root, _dirs, files in os.walk(data_dir):
        rel = os.path.relpath(root, data_dir)
        # If we have a date filter, check if this directory matches
        if allowed_dates is not None and rel != ".":
            # The top-level subdirectory name is the date code
            top_dir = rel.split(os.sep)[0]
            if top_dir not in allowed_dates:
                continue

        for f in files:
            if f.startswith("SigScan") and f.endswith(".txt"):
                stem = Path(f).stem
                if rel == ".":
                    # Top-level files: only include if no date filter,
                    # or if date filter is None (show all)
                    if allowed_dates is None:
                        results.append(stem)
                else:
                    results.append(f"{rel}/{stem}")
    return sorted(results)


def list_date_dirs(data_dir: str = DATA_DIR) -> list[str]:
    """Return sorted list of date-coded subdirectory names (e.g. ['260401', '260402'])."""
    dirs = []
    for entry in os.scandir(data_dir):
        if entry.is_dir() and re.match(r'^\d{6}$', entry.name):
            dirs.append(entry.name)
    return sorted(dirs)


def _resolve_date_filter(date_filter: str) -> set[str]:
    """Convert a date filter string into a set of YYMMDD directory names."""
    import datetime as _dt

    date_filter = date_filter.strip().lower()
    today = _dt.date.today()

    def _to_yymmdd(d: _dt.date) -> str:
        return d.strftime("%y%m%d")

    def _date_range(start: _dt.date, end: _dt.date) -> set[str]:
        dates = set()
        current = start
        while current <= end:
            dates.add(_to_yymmdd(current))
            current += _dt.timedelta(days=1)
        return dates

    # Named ranges
    if date_filter == "today":
        return {_to_yymmdd(today)}
    elif date_filter == "yesterday":
        return {_to_yymmdd(today - _dt.timedelta(days=1))}
    elif date_filter in ("this_week", "this week", "past_week", "past week", "last 7 days"):
        return _date_range(today - _dt.timedelta(days=6), today)
    elif date_filter in ("last_week", "last week"):
        start = today - _dt.timedelta(days=today.weekday() + 7)
        end = start + _dt.timedelta(days=6)
        return _date_range(start, end)

    # Range: "260401-260403"
    if "-" in date_filter and not date_filter.startswith("20"):
        parts = date_filter.split("-", 1)
        if len(parts) == 2 and re.match(r'^\d{6}$', parts[0]) and re.match(r'^\d{6}$', parts[1]):
            # Parse YYMMDD range
            start = _dt.datetime.strptime(parts[0], "%y%m%d").date()
            end = _dt.datetime.strptime(parts[1], "%y%m%d").date()
            return _date_range(start, end)

    # ISO date range: "2026-04-01 to 2026-04-03" or "2026-04-01-2026-04-03"
    to_match = re.match(r'(\d{4}-\d{2}-\d{2})\s*(?:to|-)\s*(\d{4}-\d{2}-\d{2})', date_filter)
    if to_match:
        start = _dt.datetime.strptime(to_match.group(1), "%Y-%m-%d").date()
        end = _dt.datetime.strptime(to_match.group(2), "%Y-%m-%d").date()
        return _date_range(start, end)

    # Single YYMMDD
    if re.match(r'^\d{6}$', date_filter):
        return {date_filter}

    # Single ISO date: "2026-04-01" or "20260401"
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            d = _dt.datetime.strptime(date_filter, fmt).date()
            return {_to_yymmdd(d)}
        except ValueError:
            continue

    # Fallback: treat as literal directory name
    return {date_filter}


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _find_energy_col(df: pd.DataFrame) -> str:
    """Find the energy column name dynamically."""
    for col in ENERGY_COL_CANDIDATES:
        if col in df.columns:
            return col
    # Fallback: look for any column containing "Mono" and "eV"
    for col in df.columns:
        if "Mono" in col and "eV" in col:
            return col
    # Fallback: look for any column containing "energy" or "eV" (case-insensitive)
    for col in df.columns:
        cl = col.lower()
        if "energy" in cl or "ev" in cl:
            return col
    # Last resort: use the first column (common for two-column files)
    if len(df.columns) >= 1:
        return df.columns[0]
    raise KeyError(
        f"No energy column found. Tried {ENERGY_COL_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def get_energy(df: pd.DataFrame) -> np.ndarray:
    return df[_find_energy_col(df)].values


def get_signal(df: pd.DataFrame, signal_name: str) -> np.ndarray:
    col = SIGNAL_COLUMNS.get(signal_name.upper())
    # Try exact mapped-column match first (e.g. TEY -> "TEY UHV XAS")
    if col is not None and col in df.columns:
        return df[col].values
    # Try exact column name match (case-insensitive)
    sig_lower = signal_name.lower()
    for c in df.columns:
        if c.lower() == sig_lower:
            return df[c].values
    # Flexible matching: look for column whose name starts with the signal name
    # (avoids "I0" matching "TEY / I0")
    for c in df.columns:
        if c.lower().startswith(sig_lower):
            return df[c].values
    # For two-column files: if only 2 columns, return the second (non-energy) column
    if len(df.columns) == 2:
        return df.iloc[:, 1].values
    if col is None:
        raise ValueError(f"Unknown signal '{signal_name}'. Choose from {list(SIGNAL_COLUMNS.keys())} or available columns: {list(df.columns)}")
    raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")


def normalize_by_i0(df: pd.DataFrame, signal_name: str) -> np.ndarray:
    sig = get_signal(df, signal_name)
    # Check for exact I0 column — only divide if a dedicated I0 column exists
    # (two-column exported files are already normalized, so skip I0 division)
    if I0_COL in df.columns:
        i0 = df[I0_COL].values
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(i0 != 0, sig / i0, 0.0)
    return sig


def list_available_signals(df: pd.DataFrame) -> list[str]:
    return [label for label, col in SIGNAL_COLUMNS.items() if col in df.columns]


# ---------------------------------------------------------------------------
# XAS Analysis — Normalization (Athena-style)
# ---------------------------------------------------------------------------

def find_e0(energy: np.ndarray, mu: np.ndarray) -> float:
    """Find E0 as the energy of the maximum of the 1st derivative of mu.

    Uses a Savitzky-Golay smoothed derivative to avoid noise artifacts.
    """
    # Smooth first, then differentiate
    window = max(5, len(energy) // 20)
    if window % 2 == 0:
        window += 1
    window = min(window, len(energy) - 2)
    if window < 5:
        window = 5
    deriv = savgol_filter(mu, window_length=window, polyorder=3, deriv=1,
                          delta=np.mean(np.diff(energy)))
    return energy[np.argmax(np.abs(deriv))]


def pre_edge_subtraction(
    energy: np.ndarray,
    mu: np.ndarray,
    e0: float | None = None,
    pre_edge_range: tuple[float, float] = (-30, -10),
    post_edge_range: tuple[float, float] = (15, None),
) -> dict:
    """Athena-style pre-edge subtraction and post-edge normalization.

    Parameters
    ----------
    energy : array
        Energy in eV.
    mu : array
        Absorption signal (TEY/I0, MCP/I0, etc.).
    e0 : float or None
        Edge energy. If None, determined automatically.
    pre_edge_range : (float, float)
        Energy range relative to E0 for pre-edge line fit (eV).
        Default: (-30, -10) means E0-30 to E0-10.
    post_edge_range : (float, float)
        Energy range relative to E0 for post-edge fit (eV).
        Default: (15, None) means E0+15 to end of data.

    Returns
    -------
    dict with keys:
        e0, pre_edge_line, post_edge_line, edge_step,
        norm (normalized mu), flat (flattened mu)
    """
    if e0 is None:
        e0 = find_e0(energy, mu)

    # ── Pre-edge line (linear fit) ────────────────────────────────────────
    pre_lo = e0 + pre_edge_range[0]
    pre_hi = e0 + pre_edge_range[1]
    # Clamp to data range
    pre_lo = max(pre_lo, energy.min())
    pre_hi = min(pre_hi, e0 - 1)
    pre_mask = (energy >= pre_lo) & (energy <= pre_hi)
    if pre_mask.sum() < 3:
        # Fallback: use first 10% of data
        n10 = max(3, len(energy) // 10)
        pre_mask = np.zeros(len(energy), dtype=bool)
        pre_mask[:n10] = True
    pre_coeffs = np.polyfit(energy[pre_mask], mu[pre_mask], 1)
    pre_edge_line = np.polyval(pre_coeffs, energy)

    # ── Post-edge line (linear or quadratic fit) ──────────────────────────
    post_lo = e0 + post_edge_range[0]
    post_hi = e0 + post_edge_range[1] if post_edge_range[1] is not None else energy.max()
    post_lo = max(post_lo, e0 + 5)
    post_hi = min(post_hi, energy.max())
    post_mask = (energy >= post_lo) & (energy <= post_hi)
    if post_mask.sum() < 3:
        # Fallback: use last 20% of data
        n20 = max(3, len(energy) // 5)
        post_mask = np.zeros(len(energy), dtype=bool)
        post_mask[-n20:] = True
    post_coeffs = np.polyfit(energy[post_mask], mu[post_mask], 2)
    post_edge_line = np.polyval(post_coeffs, energy)

    # ── Edge step ─────────────────────────────────────────────────────────
    edge_step = np.polyval(post_coeffs, e0) - np.polyval(pre_coeffs, e0)
    if abs(edge_step) < 1e-12:
        edge_step = 1.0  # avoid division by zero

    # ── Normalized mu ─────────────────────────────────────────────────────
    norm = (mu - pre_edge_line) / edge_step

    # ── Flattened (remove post-edge slope from normalized) ────────────────
    # Fit a line to the post-edge region of the normalized spectrum
    norm_post = norm[post_mask]
    if len(norm_post) >= 3:
        flat_coeffs = np.polyfit(energy[post_mask], norm_post, 1)
        # Only flatten above E0
        flat = norm.copy()
        above_e0 = energy >= e0
        flat[above_e0] = norm[above_e0] - (np.polyval(flat_coeffs, energy[above_e0]) - 1.0)
    else:
        flat = norm

    return {
        "e0": e0,
        "pre_edge_line": pre_edge_line,
        "post_edge_line": post_edge_line,
        "edge_step": edge_step,
        "norm": norm,
        "flat": flat,
    }


# ---------------------------------------------------------------------------
# XAS Analysis — Derivatives
# ---------------------------------------------------------------------------

def smooth_derivative(
    energy: np.ndarray,
    mu: np.ndarray,
    order: int = 1,
    window: int | None = None,
    polyorder: int = 3,
) -> np.ndarray:
    """Compute the smoothed nth derivative of mu(E) using Savitzky-Golay filter.

    Parameters
    ----------
    energy : array
        Energy in eV.
    mu : array
        Absorption signal.
    order : int
        Derivative order (1 or 2).
    window : int or None
        Savitzky-Golay window length (must be odd). If None, auto-selected.
    polyorder : int
        Polynomial order for the filter (default 3).

    Returns
    -------
    array : the derivative dⁿmu/dEⁿ
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")

    if window is None:
        # Auto-select: ~5% of data length, minimum 7, must be odd
        window = max(7, len(energy) // 20)
        if window % 2 == 0:
            window += 1
    window = min(window, len(energy) - 2)
    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        polyorder = window - 1

    delta = np.mean(np.diff(energy))
    return savgol_filter(mu, window_length=window, polyorder=polyorder,
                         deriv=order, delta=delta)


# ---------------------------------------------------------------------------
# XAS Edge Energy Database
# ---------------------------------------------------------------------------
# Reference energies (eV) for common XAS edges.
# Sources: X-ray Data Booklet (LBNL), Hephaestus/Athena tables.

XAS_EDGE_DB = {
    # 3d transition metals — K edges
    "Ti": {"K": 4966.0, "L3": 453.8, "L2": 460.2},
    "V":  {"K": 5470.0, "L3": 519.8, "L2": 526.7},
    "Cr": {"K": 5989.0, "L3": 574.1, "L2": 583.8},
    "Mn": {"K": 6539.0, "L3": 638.7, "L2": 649.9},
    "Fe": {"K": 7112.0, "L3": 706.8, "L2": 719.9},
    "Co": {"K": 7709.0, "L3": 778.1, "L2": 793.2},
    "Ni": {"K": 8333.0, "L3": 852.7, "L2": 869.9},
    "Cu": {"K": 8979.0, "L3": 932.7, "L2": 952.3},
    "Zn": {"K": 9659.0, "L3": 1021.8, "L2": 1044.9},
    # Rare earths — M4,5 edges (most relevant for soft X-ray)
    "La": {"M5": 832.0, "M4": 849.0, "L3": 5483.0, "L2": 5891.0},
    "Ce": {"M5": 883.0, "M4": 901.0, "L3": 5723.0, "L2": 6164.0},
    "Pr": {"M5": 931.0, "M4": 951.0, "L3": 5964.0, "L2": 6440.0},
    "Nd": {"M5": 980.0, "M4": 1000.0, "L3": 6208.0, "L2": 6722.0},
    "Pm": {"M5": 1027.0, "M4": 1052.0},
    "Sm": {"M5": 1080.0, "M4": 1107.0, "L3": 6716.0, "L2": 7312.0},
    "Eu": {"M5": 1131.0, "M4": 1161.0, "L3": 6977.0, "L2": 7617.0},
    "Gd": {"M5": 1185.0, "M4": 1218.0, "L3": 7243.0, "L2": 7930.0},
    "Tb": {"M5": 1241.0, "M4": 1277.0, "L3": 7514.0, "L2": 8252.0},
    "Dy": {"M5": 1295.0, "M4": 1333.0, "L3": 7790.0, "L2": 8581.0},
    "Ho": {"M5": 1351.0, "M4": 1392.0, "L3": 8071.0, "L2": 8918.0},
    "Er": {"M5": 1409.0, "M4": 1453.0, "L3": 8358.0, "L2": 9264.0},
    "Tm": {"M5": 1468.0, "M4": 1515.0},
    "Yb": {"M5": 1528.0, "M4": 1576.0, "L3": 8944.0, "L2": 9978.0},
    "Lu": {"M5": 1589.0, "M4": 1640.0},
    # Light elements — K edges
    "C":  {"K": 284.2},
    "N":  {"K": 409.9},
    "O":  {"K": 543.1},
    "F":  {"K": 696.7},
    # Others
    "S":  {"K": 2472.0, "L3": 162.5},
    "P":  {"K": 2145.5},
    "Si": {"K": 1839.0},
    "Al": {"K": 1559.6},
    "Mg": {"K": 1303.0},
    "Ca": {"K": 4038.5, "L3": 346.2, "L2": 349.7},
    "Sc": {"K": 4492.0, "L3": 398.7, "L2": 403.6},
}


def identify_edge(energy_eV: float, tolerance: float = 30.0,
                  hint_element: str | None = None) -> list[dict]:
    """Find candidate element/edge matches for a given energy.

    Parameters
    ----------
    energy_eV : float
        Observed edge or peak energy in eV.
    tolerance : float
        Maximum allowed deviation in eV (default 30).
    hint_element : str or None
        If provided, prioritize matches for this element.

    Returns
    -------
    list of dict: [{"element": str, "edge": str, "ref_energy": float, "delta": float}, ...]
        Sorted by |delta|, with hint_element matches first if given.
    """
    matches = []
    for elem, edges in XAS_EDGE_DB.items():
        for edge_name, ref_e in edges.items():
            delta = energy_eV - ref_e
            if abs(delta) <= tolerance:
                matches.append({
                    "element": elem,
                    "edge": edge_name,
                    "ref_energy": ref_e,
                    "delta": delta,
                })
    # Sort: hint element first, then by |delta|
    def sort_key(m):
        is_hint = 0 if (hint_element and m["element"] == hint_element) else 1
        return (is_hint, abs(m["delta"]))
    matches.sort(key=sort_key)
    return matches


def extract_element_hint(scan_file_path: str) -> str | None:
    """Extract element name from the scan file metadata path.

    E.g. 'C:\\...\\La M45 edge.txt' -> 'La'
    """
    # Match patterns like "La M45", "Ce M45", "Ni L23", etc.
    m = re.search(r'[\\/](\w{1,2})\s+[A-Z]\d', scan_file_path)
    if m:
        return m.group(1)
    # Fallback: try to find element symbol before "edge"
    m = re.search(r'[\\/](\w{1,2})\s+\w+\s+edge', scan_file_path, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Peak Detection
# ---------------------------------------------------------------------------

def detect_peaks(
    energy: np.ndarray,
    mu: np.ndarray,
    sensitivity: str = "normal",
    min_height_frac: float | None = None,
    min_prominence_frac: float | None = None,
) -> dict:
    """Detect peaks in an XAS spectrum with tunable sensitivity.

    Parameters
    ----------
    energy : array
        Energy in eV.
    mu : array
        Absorption signal (typically signal/I0).
    sensitivity : str
        "low" (only major peaks), "normal", "high" (more peaks),
        "very_high" (shoulders and minor features).
    min_height_frac : float or None
        Override: minimum peak height as fraction of data range (0-1).
    min_prominence_frac : float or None
        Override: minimum prominence as fraction of data range (0-1).

    Returns
    -------
    dict with keys:
        peaks: list of {"energy": float, "intensity": float, "type": str}
        sensitivity: str
        n_peaks: int
    """
    data_range = mu.max() - mu.min()
    if data_range == 0:
        return {"peaks": [], "sensitivity": sensitivity, "n_peaks": 0}

    # Sensitivity presets
    presets = {
        "low":       {"height_frac": 0.30, "prominence_frac": 0.15, "distance_pts": 20},
        "normal":    {"height_frac": 0.10, "prominence_frac": 0.05, "distance_pts": 10},
        "high":      {"height_frac": 0.03, "prominence_frac": 0.02, "distance_pts": 5},
        "very_high": {"height_frac": 0.01, "prominence_frac": 0.005, "distance_pts": 3},
    }
    preset = presets.get(sensitivity, presets["normal"])

    height_frac = min_height_frac if min_height_frac is not None else preset["height_frac"]
    prom_frac = min_prominence_frac if min_prominence_frac is not None else preset["prominence_frac"]

    min_height = mu.min() + data_range * height_frac
    min_prominence = data_range * prom_frac
    min_distance = preset["distance_pts"]

    # Direct peak detection on the signal
    peak_indices, properties = find_peaks(
        mu,
        height=min_height,
        prominence=min_prominence,
        distance=min_distance,
    )

    peaks = []
    prominences = properties.get("prominences", np.zeros(len(peak_indices)))
    for i, idx in enumerate(peak_indices):
        peaks.append({
            "energy": float(energy[idx]),
            "intensity": float(mu[idx]),
            "prominence": float(prominences[i]),
            "type": "peak",
        })

    # Filter out spurious weak peaks: remove peaks whose prominence is < 10%
    # of the most prominent peak (avoids false positives like 930.50 eV)
    if peaks:
        max_prom = max(p["prominence"] for p in peaks)
        if max_prom > 0:
            prom_threshold = max_prom * 0.10
            peaks = [p for p in peaks if p["prominence"] >= prom_threshold]

    # Detect shoulders via 2nd derivative for normal and above
    if sensitivity in ("normal", "high", "very_high"):
        shoulder_peaks = _detect_shoulders(energy, mu, sensitivity)
        # Merge, avoiding duplicates (within 1.5 eV of existing peaks)
        for sp in shoulder_peaks:
            if not any(abs(sp["energy"] - p["energy"]) < 1.5 for p in peaks):
                peaks.append(sp)

    # Sort by energy
    peaks.sort(key=lambda p: p["energy"])

    return {"peaks": peaks, "sensitivity": sensitivity, "n_peaks": len(peaks)}


def _detect_shoulders(
    energy: np.ndarray,
    mu: np.ndarray,
    sensitivity: str = "high",
) -> list[dict]:
    """Detect shoulder features using the 2nd derivative.

    Shoulders appear as local minima (dips) in the 2nd derivative
    that don't correspond to full peaks in the original signal.
    """
    # Compute smoothed 2nd derivative
    # Use wider smoothing window for normal sensitivity to avoid noise
    window_divisors = {"normal": 12, "high": 15, "very_high": 30}
    divisor = window_divisors.get(sensitivity, 15)
    window = max(7, len(energy) // divisor)
    if window % 2 == 0:
        window += 1
    d2 = smooth_derivative(energy, mu, order=2, window=window)

    # Find minima in 2nd derivative (= inflection points / shoulders)
    neg_d2 = -d2  # invert to find minima as peaks
    data_range = neg_d2.max() - neg_d2.min()
    if data_range == 0:
        return []

    # Prominence thresholds: stricter for normal, looser for very_high
    prom_fracs = {"normal": 0.10, "high": 0.06, "very_high": 0.03}
    prom_frac = prom_fracs.get(sensitivity, 0.08)
    min_prom = data_range * prom_frac

    min_dist = 5 if sensitivity == "normal" else 3
    indices, _ = find_peaks(neg_d2, prominence=min_prom, distance=min_dist)

    shoulders = []
    for idx in indices:
        shoulders.append({
            "energy": float(energy[idx]),
            "intensity": float(mu[idx]),
            "type": "shoulder",
        })
    return shoulders


# ---------------------------------------------------------------------------
# Export & Rename
# ---------------------------------------------------------------------------

def ensure_export_dir(subdir: str | None = None) -> str:
    path = os.path.join(EXPORT_DIR, subdir) if subdir else EXPORT_DIR
    os.makedirs(path, exist_ok=True)
    return path


def rename_scan(scan_id: str, new_name: str, data_dir: str = DATA_DIR) -> str:
    """Copy a scan file to the export directory with a new user-specified name.

    The original raw data file is never modified. A copy is created in
    the exported_data directory with the new name.

    Parameters
    ----------
    scan_id : str
        Original scan identifier (e.g. '45616' or 'SigScan45616').
    new_name : str
        New descriptive filename (e.g. 'Ce_M45_edge_sample1').
        The .txt extension is added automatically if not present.
    data_dir : str
        Data directory to search for the original file.

    Returns
    -------
    str : Path to the newly created copy.
    """
    import shutil
    src_path = scan_filepath(scan_id, data_dir)
    out_dir = ensure_export_dir("renamed")

    # Sanitize the new name: replace spaces with underscores, remove unsafe chars
    safe_name = re.sub(r'[^\w\-.]', '_', new_name.strip())
    if not safe_name.lower().endswith(".txt"):
        safe_name += ".txt"

    dst_path = os.path.join(out_dir, safe_name)

    # Avoid overwriting existing files
    if os.path.exists(dst_path):
        base, ext = os.path.splitext(safe_name)
        counter = 1
        while os.path.exists(dst_path):
            dst_path = os.path.join(out_dir, f"{base}_{counter}{ext}")
            counter += 1

    shutil.copy2(src_path, dst_path)
    return dst_path


def export_data(
    energy: np.ndarray,
    signal: np.ndarray,
    signal_name: str,
    scan_id: str,
    filename: str | None = None,
    subdir: str | None = None,
) -> str:
    # Recognized data file extensions (no auto-append needed)
    KNOWN_EXTENSIONS = {".txt", ".csv", ".tsv", ".dat"}

    out_dir = ensure_export_dir(subdir)
    if filename is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scan_id}_{signal_name}_{ts}.txt"

    # Only append .txt if the file has no recognized extension
    _, ext = os.path.splitext(filename)
    if ext.lower() not in KNOWN_EXTENSIONS:
        filename += ".txt"
        ext = ".txt"

    out_path = os.path.join(out_dir, filename)

    # Choose delimiter based on extension
    if ext.lower() == ".csv":
        delimiter = ","
        header_line = f"Energy_eV,{signal_name}"
    elif ext.lower() == ".tsv":
        delimiter = "\t"
        header_line = f"Energy_eV\t{signal_name}"
    else:
        delimiter = "\t"
        header_line = f"Energy_eV\t{signal_name}"

    data = np.column_stack([energy, signal])
    np.savetxt(out_path, data, delimiter=delimiter, header=header_line, comments="", fmt="%.10g")
    return out_path
