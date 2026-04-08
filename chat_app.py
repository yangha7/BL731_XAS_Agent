"""
XAS Agent – Interactive Chat Web App
=====================================
A Flask-based chat interface for the XAS AI Agent.
Run with:  python chat_app.py
Then open http://localhost:5050 in your browser.
"""

import os
import sys
import json
import textwrap
import base64
import io
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, jsonify, render_template_string

# ── Ensure local imports work ─────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import xas_utils as xu

# ── Load environment ─────────────────────────────────────────────────────────
load_dotenv()

# ── LLM Provider Configuration ──────────────────────────────────────────────
# Supported providers and their defaults:
#   cborg   → CBORG_API_KEY,  base_url=https://api.cborg.lbl.gov/v1,  model=claude-sonnet
#   openai  → OPENAI_API_KEY, base_url=https://api.openai.com/v1,     model=gpt-4o
#   gemini  → GEMINI_API_KEY, base_url=https://generativelanguage.googleapis.com/v1beta/openai, model=gemini-2.0-flash
#   claude  → ANTHROPIC_API_KEY, base_url=https://api.anthropic.com/v1, model=claude-sonnet-4-20250514
#
# Set LLM_PROVIDER in .env (or it auto-detects from available API keys).
# Override model with LLM_MODEL and base URL with LLM_BASE_URL.

PROVIDER_DEFAULTS = {
    "cborg": {
        "key_env": "CBORG_API_KEY",
        "base_url": "https://api.cborg.lbl.gov/v1",
        "model": "claude-sonnet",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
    },
    "claude": {
        "key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-20250514",
    },
}


def _configure_llm():
    """Detect and configure the LLM provider from environment variables."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if provider and provider in PROVIDER_DEFAULTS:
        # Explicit provider selection
        cfg = PROVIDER_DEFAULTS[provider]
        api_key = os.environ.get(cfg["key_env"], "")
        if not api_key:
            raise ValueError(
                f"LLM_PROVIDER={provider} but {cfg['key_env']} is not set in .env"
            )
    elif provider:
        # Custom provider — require LLM_API_KEY and LLM_BASE_URL
        api_key = os.environ.get("LLM_API_KEY", "")
        base_url = os.environ.get("LLM_BASE_URL", "")
        model = os.environ.get("LLM_MODEL", "")
        if not all([api_key, base_url, model]):
            raise ValueError(
                f"Custom LLM_PROVIDER='{provider}' requires LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in .env"
            )
        return OpenAI(api_key=api_key, base_url=base_url), model, provider
    else:
        # Auto-detect: try each provider in order
        for pname, cfg in PROVIDER_DEFAULTS.items():
            api_key = os.environ.get(cfg["key_env"], "")
            if api_key and api_key != "your-api-key-here":
                provider = pname
                break
        else:
            raise ValueError(
                "No LLM API key found. Set one of: CBORG_API_KEY, OPENAI_API_KEY, "
                "GEMINI_API_KEY, or ANTHROPIC_API_KEY in your .env file.\n"
                "Or set LLM_PROVIDER + LLM_API_KEY + LLM_BASE_URL for a custom provider."
            )
        cfg = PROVIDER_DEFAULTS[provider]

    # Allow overrides
    base_url = os.environ.get("LLM_BASE_URL", cfg["base_url"])
    model = os.environ.get("LLM_MODEL", cfg["model"])

    return OpenAI(api_key=api_key, base_url=base_url), model, provider


client, MODEL, LLM_PROVIDER = _configure_llm()
print(f"✅ LLM Provider: {LLM_PROVIDER} | Model: {MODEL}")

DATA_DIR = os.environ.get("XAS_DATA_DIR", "731_Data")

# ── Matplotlib defaults ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ── State ─────────────────────────────────────────────────────────────────────
_last_plot = {}
_cache = {}
_pending_images = []  # collect base64 plot images during tool calls


def _load(scan_id: str) -> tuple:
    sid = xu.resolve_scan_id(scan_id)
    if sid not in _cache:
        fp = xu.scan_filepath(sid, DATA_DIR)
        _cache[sid] = {"meta": xu.parse_header(fp), "df": xu.load_scan(fp)}
    return sid, _cache[sid]["meta"], _cache[sid]["df"]


def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Tool definitions ─────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_scans",
            "description": "List available XAS scan files. Subdirectories are named by date (YYMMDD format, e.g. '260401' = 2026-04-01). If no date is given, defaults to the past week. Accepts date formats: '260401', '2026-04-01', 'today', 'yesterday', 'this_week', or a range like '260401-260403'. Use 'all' to list everything (may be very large).",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date filter: a specific date ('260401', '2026-04-01'), a range ('260401-260403'), a keyword ('today', 'yesterday', 'this_week', 'last_week'), or 'all' for everything. Default: past week."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_scan",
            "description": "Plot a single XAS scan. Plots the specified signal (TEY, TFY, or MCP) vs Mono Energy in eV. Optionally normalize by I0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_id": {"type": "string", "description": "Scan identifier, e.g. 'SigScan45611' or '45611'."},
                    "signal": {"type": "string", "enum": ["TEY", "TFY", "MCP"], "description": "Signal channel to plot."},
                    "normalize": {"type": "boolean", "description": "If true, divide signal by I0. Default false."},
                },
                "required": ["scan_id", "signal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_scans",
            "description": "Overlay multiple XAS scans on one plot for comparison.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_ids": {"type": "array", "items": {"type": "string"}, "description": "List of scan identifiers."},
                    "signal": {"type": "string", "enum": ["TEY", "TFY", "MCP"], "description": "Signal channel."},
                    "normalize": {"type": "boolean", "description": "If true, divide by I0. Default false."},
                },
                "required": ["scan_ids", "signal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_data",
            "description": "Save the last plotted data to an ASCII text file in exported_data/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Optional filename (e.g. 'my_data.txt')."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_scan_info",
            "description": "Show detailed metadata, energy range, and available signals for one or more scans. Pass a single scan_id string OR a scan_ids array for batch queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_id": {"type": "string", "description": "Single scan identifier (e.g. '45611')."},
                    "scan_ids": {"type": "array", "items": {"type": "string"}, "description": "List of scan identifiers for batch metadata queries."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "normalize_scan",
            "description": "Perform Athena-style XAS normalization on a scan: divide by I0, subtract pre-edge, normalize to edge step = 1. Plots the normalized spectrum and shows E0 and edge step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_id": {"type": "string", "description": "Scan identifier."},
                    "signal": {"type": "string", "enum": ["TEY", "TFY", "MCP"], "description": "Signal channel to normalize."},
                    "e0": {"type": "number", "description": "Optional edge energy in eV. If not given, determined automatically from the maximum of the 1st derivative."},
                    "flatten": {"type": "boolean", "description": "If true, also flatten the post-edge region. Default false."},
                },
                "required": ["scan_id", "signal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "derivative_scan",
            "description": "Compute and plot the 1st or 2nd derivative of a scan signal. Uses Savitzky-Golay smoothing. The signal is first divided by I0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_id": {"type": "string", "description": "Scan identifier."},
                    "signal": {"type": "string", "enum": ["TEY", "TFY", "MCP"], "description": "Signal channel."},
                    "order": {"type": "integer", "enum": [1, 2], "description": "Derivative order: 1 for first derivative, 2 for second derivative."},
                    "smooth_window": {"type": "integer", "description": "Optional Savitzky-Golay window size (odd integer). Auto-selected if not given."},
                },
                "required": ["scan_id", "signal", "order"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_peaks_scan",
            "description": "Detect peaks and shoulders in an XAS scan. Sensitivity controls how many features are found: 'low' = only major peaks, 'normal' = main peaks, 'high' = more peaks + shoulders, 'very_high' = all features including minor shoulders. If the user asks to 'find more peaks', increase sensitivity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_id": {"type": "string", "description": "Scan identifier."},
                    "signal": {"type": "string", "enum": ["TEY", "TFY", "MCP"], "description": "Signal channel."},
                    "sensitivity": {"type": "string", "enum": ["low", "normal", "high", "very_high"], "description": "Peak detection sensitivity. Default 'normal'."},
                },
                "required": ["scan_id", "signal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identify_edge",
            "description": "Identify what element and absorption edge a scan corresponds to, based on the detected peak energies and scan file metadata. Returns candidate element/edge matches ranked by likelihood.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_id": {"type": "string", "description": "Scan identifier."},
                    "signal": {"type": "string", "enum": ["TEY", "TFY", "MCP"], "description": "Signal channel to analyze."},
                },
                "required": ["scan_id", "signal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_scan",
            "description": "Copy a scan file to the exported_data/renamed/ directory with a user-specified descriptive name. The original raw data file is never modified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scan_id": {"type": "string", "description": "Scan identifier (e.g. '45611' or 'SigScan45611')."},
                    "new_name": {"type": "string", "description": "Descriptive name for the copy (e.g. 'Cu_L3_edge_sample_A'). A .txt extension is added automatically if missing."},
                },
                "required": ["scan_id", "new_name"],
            },
        },
    },
]


# ── Tool implementations ─────────────────────────────────────────────────────

def tool_list_scans(date: str = None, **kw) -> str:
    # Default to past week if no date specified
    if date is None or date.strip() == "":
        date_filter = "this_week"
        label = "past week"
    elif date.strip().lower() == "all":
        date_filter = None  # no filter — show everything
        label = "all dates"
    else:
        date_filter = date.strip()
        label = date_filter

    ids = xu.list_scan_files(DATA_DIR, date_filter=date_filter)

    # Also get available date directories for context
    date_dirs = xu.list_date_dirs(DATA_DIR)

    MAX_DISPLAY = 100
    if len(ids) > MAX_DISPLAY and date_filter is None:
        # Too many to list — summarize by directory
        from collections import Counter
        dir_counts = Counter()
        for scan_id in ids:
            if "/" in scan_id:
                d = scan_id.split("/")[0]
            else:
                d = "(top-level)"
            dir_counts[d] += 1
        summary = [f"Found {len(ids)} scans across {len(dir_counts)} directories (too many to list all):"]
        for d, count in sorted(dir_counts.items()):
            summary.append(f"  {d}: {count} scans")
        summary.append(f"\nTip: Specify a date to narrow results, e.g. 'list scans from 260401' or 'list scans from this week'.")
        summary.append(f"Available date directories: {', '.join(date_dirs)}")
        return "\n".join(summary)

    if not ids:
        msg = f"No scans found for {label}."
        if date_dirs:
            msg += f"\nAvailable date directories: {', '.join(date_dirs)}"
        return msg

    header = f"Found {len(ids)} scans ({label}):"
    return header + "\n" + "\n".join(ids)


def tool_plot_scan(scan_id: str, signal: str, normalize: bool = False, **kw) -> str:
    global _last_plot
    try:
        sid, meta, df = _load(scan_id)
    except FileNotFoundError as e:
        return str(e)

    energy = xu.get_energy(df)
    if normalize:
        signal_data = xu.normalize_by_i0(df, signal)
        ylabel = f"{signal} / I0"
    else:
        signal_data = xu.get_signal(df, signal)
        ylabel = signal

    fig, ax = plt.subplots()
    ax.plot(energy, signal_data, "b-", linewidth=1.2)
    ax.set_xlabel("Mono Energy (eV)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{sid}  —  {meta['scan_type']}  ({meta['date']})")
    plt.tight_layout()

    _pending_images.append(_fig_to_base64(fig))
    _last_plot = {"energy": energy, "signal": signal_data, "signal_name": ylabel, "scan_id": sid}
    return f"Plotted {ylabel} for {sid}. Energy: {energy.min():.2f}–{energy.max():.2f} eV, {len(energy)} pts."


def tool_compare_scans(scan_ids: list, signal: str, normalize: bool = False, **kw) -> str:
    global _last_plot
    loaded = []
    for raw in scan_ids:
        try:
            loaded.append(_load(raw))
        except FileNotFoundError as e:
            return str(e)

    fig, ax = plt.subplots()
    for sid, meta, df in loaded:
        energy = xu.get_energy(df)
        if normalize:
            sig_data = xu.normalize_by_i0(df, signal)
            ylabel = f"{signal} / I0"
        else:
            sig_data = xu.get_signal(df, signal)
            ylabel = signal
        ax.plot(energy, sig_data, linewidth=1.0, label=sid)

    ax.set_xlabel("Mono Energy (eV)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Comparison — {ylabel}")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    _pending_images.append(_fig_to_base64(fig))

    first_sid, _, df0 = loaded[0]
    _last_plot = {
        "energy": xu.get_energy(df0),
        "signal": xu.normalize_by_i0(df0, signal) if normalize else xu.get_signal(df0, signal),
        "signal_name": ylabel,
        "scan_id": "_".join(s for s, _, _ in loaded),
    }
    return f"Compared {len(loaded)} scans: {', '.join(s for s, _, _ in loaded)}."


def tool_save_data(filename: str = None, **kw) -> str:
    if not _last_plot:
        return "Error: Nothing to save. Please plot something first."
    path = xu.export_data(
        energy=_last_plot["energy"],
        signal=_last_plot["signal"],
        signal_name=_last_plot["signal_name"],
        scan_id=_last_plot["scan_id"],
        filename=filename,
    )
    return f"Data saved to: {path}"


def _format_scan_info(scan_id: str) -> str:
    """Return formatted metadata for a single scan, or an error string."""
    try:
        sid, meta, df = _load(scan_id)
    except FileNotFoundError as e:
        return f"Scan {scan_id}: {e}"
    except Exception as e:
        return f"Scan {scan_id}: Error – {e}"
    energy = xu.get_energy(df)
    signals = xu.list_available_signals(df)
    lines = [
        f"Scan: {sid}",
        f"Date: {meta['date']}",
        f"Scan Mode: {meta.get('scan_mode', 'unknown')}",
        f"Scan Type: {meta['scan_type']}",
    ]
    if meta.get("scan_file"):
        lines.append(f"Scan File: {meta['scan_file']}")
    if "start" in meta:
        lines.append(f"Start: {meta['start']:.2f} eV, Stop: {meta['stop']:.2f} eV, Step: {meta['increment']:.5f} eV")
    lines.extend([
        f"Count Time: {meta['count_time']}s",
        f"Delay After Move: {meta['delay_after_move']}s",
        f"Energy Range: {energy.min():.2f} – {energy.max():.2f} eV",
        f"Data Points: {len(df)}",
        f"Available Signals: {', '.join(signals)}",
    ])
    return "\n".join(lines)


def tool_show_scan_info(scan_id: str = None, scan_ids: list = None, **kw) -> str:
    """Show metadata for one or more scans."""
    ids = []
    if scan_ids:
        ids = list(scan_ids)
    elif scan_id:
        ids = [scan_id]
    else:
        return "Error: Please provide a scan_id or scan_ids."

    if len(ids) == 1:
        return _format_scan_info(ids[0])

    # Multiple scans — format each with a separator
    results = []
    for sid in ids:
        results.append(_format_scan_info(sid))
    return "\n\n---\n\n".join(results)


def tool_normalize_scan(scan_id: str, signal: str, e0: float = None, flatten: bool = False, **kw) -> str:
    global _last_plot
    try:
        sid, meta, df = _load(scan_id)
    except FileNotFoundError as e:
        return str(e)

    energy = xu.get_energy(df)
    mu = xu.normalize_by_i0(df, signal)

    result = xu.pre_edge_subtraction(energy, mu, e0=e0)
    norm_data = result["flat"] if flatten else result["norm"]
    ylabel = f"Normalized {signal}/I0" + (" (flattened)" if flatten else "")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw mu with pre-edge and post-edge lines
    ax1 = axes[0]
    ax1.plot(energy, mu, "b-", linewidth=1.2, label=f"{signal}/I0")
    ax1.plot(energy, result["pre_edge_line"], "r--", linewidth=0.8, label="Pre-edge line")
    ax1.plot(energy, result["post_edge_line"], "g--", linewidth=0.8, label="Post-edge line")
    ax1.axvline(result["e0"], color="orange", linestyle=":", linewidth=1, label=f"E0 = {result['e0']:.2f} eV")
    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel(f"{signal} / I0")
    ax1.set_title(f"{sid} — Raw with fit lines")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: normalized spectrum
    ax2 = axes[1]
    ax2.plot(energy, norm_data, "b-", linewidth=1.2)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=0.5)
    ax2.axvline(result["e0"], color="orange", linestyle=":", linewidth=1, label=f"E0 = {result['e0']:.2f} eV")
    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel(ylabel)
    ax2.set_title(f"{sid} — Normalized")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _pending_images.append(_fig_to_base64(fig))

    _last_plot = {"energy": energy, "signal": norm_data, "signal_name": ylabel, "scan_id": sid}
    return (
        f"Normalized {signal}/I0 for {sid}.\n"
        f"E0 = {result['e0']:.2f} eV\n"
        f"Edge step = {result['edge_step']:.6f}\n"
        f"Energy range: {energy.min():.2f}–{energy.max():.2f} eV, {len(energy)} pts."
    )


def tool_derivative_scan(scan_id: str, signal: str, order: int = 1, smooth_window: int = None, **kw) -> str:
    global _last_plot
    try:
        sid, meta, df = _load(scan_id)
    except FileNotFoundError as e:
        return str(e)

    energy = xu.get_energy(df)
    mu = xu.normalize_by_i0(df, signal)

    deriv = xu.smooth_derivative(energy, mu, order=order, window=smooth_window)
    ordinal = "1st" if order == 1 else "2nd"
    ylabel = f"d{'²' if order == 2 else ''}(μ)/dE{'²' if order == 2 else ''}"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08})

    # Top: original mu
    ax1.plot(energy, mu, "b-", linewidth=1.2)
    ax1.set_ylabel(f"{signal} / I0")
    ax1.set_title(f"{sid} — {signal}/I0 and {ordinal} Derivative")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)  # hide x labels on top plot

    # Bottom: derivative (aligned energy axis)
    ax2.plot(energy, deriv, "r-", linewidth=1.2)
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.5)

    # Find ALL strong peaks in the derivative for labeling
    from scipy.signal import find_peaks as _find_peaks
    deriv_range = deriv.max() - deriv.min()
    peak_labels = []

    if order == 2:
        # For 2nd derivative: find all strong NEGATIVE peaks (minima)
        # These correspond to absorption features / edges
        neg_deriv = -deriv
        neg_range = neg_deriv.max() - neg_deriv.min()
        if neg_range > 0:
            neg_indices, neg_props = _find_peaks(
                neg_deriv,
                prominence=neg_range * 0.08,
                distance=5,
            )
            colors = ["#E67E22", "#8E44AD", "#2ECC71", "#E74C3C", "#3498DB",
                       "#1ABC9C", "#F39C12", "#9B59B6"]
            for i, idx in enumerate(neg_indices):
                c = colors[i % len(colors)]
                lbl = f"Min at {energy[idx]:.2f} eV"
                ax2.axvline(energy[idx], color=c, linestyle="--", linewidth=1, alpha=0.8,
                            label=lbl)
                ax1.axvline(energy[idx], color=c, linestyle="--", linewidth=0.8, alpha=0.5)
                peak_labels.append(f"{energy[idx]:.2f} eV")
    else:
        # For 1st derivative: mark the single strongest peak (E0 estimate)
        peak_idx = np.argmax(np.abs(deriv))
        ax2.axvline(energy[peak_idx], color="orange", linestyle=":", linewidth=1,
                    label=f"Peak at {energy[peak_idx]:.2f} eV")
        ax1.axvline(energy[peak_idx], color="orange", linestyle=":", linewidth=1, alpha=0.7)
        peak_labels.append(f"{energy[peak_idx]:.2f} eV")

    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel(ylabel)
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _pending_images.append(_fig_to_base64(fig))

    _last_plot = {"energy": energy, "signal": deriv, "signal_name": f"{ordinal}_deriv_{signal}", "scan_id": sid}

    peaks_str = ", ".join(peak_labels) if peak_labels else "none detected"
    return (
        f"Computed {ordinal} derivative of {signal}/I0 for {sid}.\n"
        f"Key features: {peaks_str}\n"
        f"Energy range: {energy.min():.2f}–{energy.max():.2f} eV, {len(energy)} pts."
    )


def tool_find_peaks_scan(scan_id: str, signal: str, sensitivity: str = "normal", **kw) -> str:
    global _last_plot
    try:
        sid, meta, df = _load(scan_id)
    except FileNotFoundError as e:
        return str(e)

    energy = xu.get_energy(df)
    mu = xu.normalize_by_i0(df, signal)

    result = xu.detect_peaks(energy, mu, sensitivity=sensitivity)
    peaks = result["peaks"]

    # Plot: spectrum with peaks marked
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energy, mu, "b-", linewidth=1.2, label=f"{signal}/I0")

    # Mark peaks and shoulders differently
    for p in peaks:
        if p["type"] == "peak":
            ax.axvline(p["energy"], color="red", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.plot(p["energy"], p["intensity"], "rv", markersize=8)
        else:  # shoulder
            ax.axvline(p["energy"], color="green", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.plot(p["energy"], p["intensity"], "g^", markersize=7)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(f"{signal} / I0")
    ax.set_title(f"{sid} — Peak Detection (sensitivity: {sensitivity})")

    # Legend
    import matplotlib.lines as mlines
    peak_marker = mlines.Line2D([], [], color="red", marker="v", linestyle="--",
                                 markersize=8, label="Peak")
    shoulder_marker = mlines.Line2D([], [], color="green", marker="^", linestyle=":",
                                     markersize=7, label="Shoulder")
    ax.legend(handles=[peak_marker, shoulder_marker], fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _pending_images.append(_fig_to_base64(fig))

    _last_plot = {"energy": energy, "signal": mu, "signal_name": f"{signal}_I0", "scan_id": sid}

    # Build text summary
    lines = [f"Found {result['n_peaks']} features in {sid} (sensitivity: {sensitivity}):"]
    for i, p in enumerate(peaks, 1):
        lines.append(f"  {i}. {p['type'].capitalize():8s} at {p['energy']:.2f} eV  (intensity: {p['intensity']:.6f})")

    if sensitivity in ("low", "normal"):
        lines.append(f"\nTip: Use higher sensitivity ('high' or 'very_high') to find more peaks and shoulders.")

    return "\n".join(lines)


def tool_identify_edge(scan_id: str, signal: str, **kw) -> str:
    try:
        sid, meta, df = _load(scan_id)
    except FileNotFoundError as e:
        return str(e)

    energy = xu.get_energy(df)
    mu = xu.normalize_by_i0(df, signal)

    # Find E0
    e0 = xu.find_e0(energy, mu)

    # Get element hint from scan file metadata
    hint = xu.extract_element_hint(meta.get("scan_file", ""))

    # Find candidate edges
    matches = xu.identify_edge(e0, tolerance=30.0, hint_element=hint)

    # Also detect main peaks for additional context
    peak_result = xu.detect_peaks(energy, mu, sensitivity="normal")
    peak_energies = [p["energy"] for p in peak_result["peaks"]]

    # Build response
    lines = [
        f"Scan: {sid}",
        f"Scan file: {meta.get('scan_file', 'N/A')}",
        f"Energy range: {energy.min():.2f} – {energy.max():.2f} eV",
        f"E0 (edge energy): {e0:.2f} eV",
        f"Main peaks at: {', '.join(f'{e:.1f} eV' for e in peak_energies)}",
    ]

    if hint:
        lines.append(f"Element hint from metadata: {hint}")

    lines.append("\nCandidate edge assignments (ranked by likelihood):")
    if not matches:
        lines.append("  No matches found within ±30 eV tolerance.")
    else:
        for i, m in enumerate(matches[:6], 1):
            marker = " ⭐" if (hint and m["element"] == hint) else ""
            lines.append(
                f"  {i}. {m['element']} {m['edge']} edge "
                f"(ref: {m['ref_energy']:.1f} eV, Δ = {m['delta']:+.1f} eV){marker}"
            )

    # Provide a conclusion
    if matches:
        best = matches[0]
        if hint and best["element"] == hint:
            lines.append(f"\n✅ Best match: {best['element']} {best['edge']} edge "
                         f"(confirmed by scan file metadata)")
        else:
            lines.append(f"\n🔍 Best match by energy: {best['element']} {best['edge']} edge")
            if hint:
                hint_matches = [m for m in matches if m["element"] == hint]
                if hint_matches:
                    hm = hint_matches[0]
                    lines.append(f"   But metadata suggests: {hm['element']} {hm['edge']} edge "
                                 f"(Δ = {hm['delta']:+.1f} eV)")

    return "\n".join(lines)


def tool_rename_scan(scan_id: str, new_name: str, **kw) -> str:
    """Copy a scan file to exported_data/renamed/ with a descriptive name."""
    try:
        sid = xas_utils.resolve_scan_id(scan_id)
    except Exception as e:
        return f"Error: {e}"
    try:
        dst = xas_utils.rename_scan(sid, new_name)
        return (f"Scan {sid} copied successfully.\n"
                f"  New file: {dst}\n"
                f"  (Original raw data is unchanged.)")
    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error copying scan: {e}"


TOOL_DISPATCH = {
    "list_scans": tool_list_scans,
    "plot_scan": tool_plot_scan,
    "compare_scans": tool_compare_scans,
    "save_data": tool_save_data,
    "show_scan_info": tool_show_scan_info,
    "normalize_scan": tool_normalize_scan,
    "derivative_scan": tool_derivative_scan,
    "find_peaks_scan": tool_find_peaks_scan,
    "identify_edge": tool_identify_edge,
    "rename_scan": tool_rename_scan,
}


# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent(f"""\
You are an XAS (X-ray Absorption Spectroscopy) data analysis assistant.
You help the user visualize and analyze XAS scan data collected at a synchrotron beamline.

Data directory: {DATA_DIR}
Scan files are named SigScan<number>.txt (e.g. SigScan45611.txt).
Data is organized in date-coded subdirectories (YYMMDD format, e.g. 260401 = 2026-04-01).
There may be hundreds of subdirectories spanning years of data.
Available signal types: TEY (Total Electron Yield), TFY (Total Fluorescence Yield), MCP (MicroChannel Plate, column "MCP Np")
Energy column: Mono Energy in eV

Available tools:
- list_scans: List scan files. Accepts an optional date filter (e.g. '260401', '2026-04-01', 'today', 'this_week', or a range '260401-260403'). Defaults to the past week. Use 'all' to list everything.
- plot_scan: Plot a single scan (raw or divided by I0)
- compare_scans: Overlay multiple scans on one plot
- show_scan_info: Show metadata for a scan
- normalize_scan: Athena-style XAS normalization
- derivative_scan: Compute smoothed 1st or 2nd derivative
- find_peaks_scan: Detect peaks and shoulders with tunable sensitivity
- identify_edge: Identify element and absorption edge from peak energies and metadata
- save_data: Save the last plotted data to a text file
- rename_scan: Copy a scan file with a descriptive name to exported_data/renamed/ (original raw data is never modified)

Rules:
- The user may refer to scans by full ID (SigScan45611) or just the number (45611)
- Scans are found automatically regardless of which subdirectory they are in
- When the user says "plot" or "show", use plot_scan or compare_scans
- When the user says "normalize", use normalize_scan
- When the user says "derivative", "1st derivative", "2nd derivative", or "d/dE", use derivative_scan
- When the user says "find peaks", "detect peaks", or "peak detection", use find_peaks_scan
- When the user says "find more peaks" or "more features", use find_peaks_scan with higher sensitivity
- When the user asks "what element" or "what edge" or "identify", use identify_edge
- When the user says "save" or "export", use save_data
- When the user says "rename", "copy as", or "save as", use rename_scan to create a named copy
- When the user asks to "list" scans, use list_scans with an appropriate date filter
- If the user asks to list scans without specifying a date, default to the past week
- If the user mentions a specific date like "April 1st" or "yesterday", convert it to the appropriate date filter
- When the user asks about a specific scan's details, use show_scan_info
- When the user asks for metadata on multiple scans, use show_scan_info with scan_ids (array) instead of calling it multiple times
- I0 is the incident beam intensity; normalizing by I0 removes beam current variations
- normalize_scan always divides by I0 first, then does pre-edge subtraction and post-edge normalization
- derivative_scan always divides by I0 first, then computes the derivative
- find_peaks_scan sensitivity levels: 'low' (major peaks only), 'normal' (default), 'high' (more peaks + shoulders), 'very_high' (all features)
- If the user asks to "find more peaks" after a previous detection, increase the sensitivity level
- Be helpful and concise
- If the request is ambiguous, make a reasonable assumption and explain what you did
""")

conversation = [{"role": "system", "content": SYSTEM_PROMPT}]


def agent_chat(user_message: str) -> dict:
    """Send a message to the agent and return {text, images, tools_used}."""
    global _pending_images
    _pending_images = []
    tools_used = []

    conversation.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL, messages=conversation, tools=TOOLS, tool_choice="auto",
    )
    msg = response.choices[0].message

    while msg.tool_calls:
        conversation.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            tools_used.append(f"🔧 {fn_name}({json.dumps(fn_args)})")
            fn = TOOL_DISPATCH.get(fn_name)
            try:
                result = fn(**fn_args) if fn else f"Error: Unknown tool '{fn_name}'"
            except Exception as exc:
                result = f"Error in {fn_name}: {exc}"
            conversation.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        response = client.chat.completions.create(
            model=MODEL, messages=conversation, tools=TOOLS, tool_choice="auto",
        )
        msg = response.choices[0].message

    conversation.append(msg)

    return {
        "text": msg.content or "",
        "images": _pending_images[:],
        "tools_used": tools_used,
    }


# ── Flask App ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🤖 XAS Agent Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #1976d2;
    color: white;
    padding: 12px 20px;
    font-size: 18px;
    font-weight: 600;
    flex-shrink: 0;
  }
  header small { font-weight: 400; opacity: 0.8; font-size: 13px; }
  #chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; }
  .msg.user {
    align-self: flex-end;
    background: #e3f2fd;
    border: 1px solid #bbdefb;
    color: #0d47a1;
  }
  .msg.assistant {
    align-self: flex-start;
    background: white;
    border: 1px solid #e0e0e0;
    color: #333;
  }
  .msg.tool {
    align-self: flex-start;
    background: #fff3e0;
    border: 1px solid #ffe0b2;
    color: #e65100;
    font-family: monospace;
    font-size: 13px;
    padding: 6px 10px;
  }
  .msg img {
    max-width: 100%;
    border-radius: 8px;
    margin-top: 8px;
    border: 1px solid #ddd;
  }
  .msg pre {
    background: #f5f5f5;
    padding: 8px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 13px;
    margin: 6px 0;
  }
  .msg code { font-size: 13px; }
  .msg p { margin: 4px 0; }
  #input-area {
    flex-shrink: 0;
    display: flex;
    gap: 8px;
    padding: 12px 20px;
    background: white;
    border-top: 1px solid #ddd;
  }
  #msg-input {
    flex: 1;
    padding: 10px 14px;
    border: 1px solid #ccc;
    border-radius: 8px;
    font-size: 15px;
    outline: none;
  }
  #msg-input:focus { border-color: #1976d2; box-shadow: 0 0 0 2px rgba(25,118,210,0.2); }
  #send-btn {
    padding: 10px 24px;
    background: #1976d2;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 15px;
    cursor: pointer;
    font-weight: 600;
  }
  #send-btn:hover { background: #1565c0; }
  #send-btn:disabled { background: #90caf9; cursor: not-allowed; }
  #clear-btn {
    padding: 10px 16px;
    background: #ff9800;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
  }
  #clear-btn:hover { background: #f57c00; }
  .typing { color: #888; font-style: italic; }
  .welcome {
    align-self: center;
    color: #888;
    font-style: italic;
    text-align: center;
    padding: 20px;
  }
  .welcome b { color: #555; }
</style>
<!-- Markdown rendering -->
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
  <header>
    🤖 XAS Agent Chat
    <small>— Talk to your XAS data in natural language</small>
  </header>

  <div id="chat-area">
    <div class="welcome">
      🤖 XAS Agent ready!<br>
      Try: <b>List all scans</b> · <b>Plot TEY for 45612</b> · <b>Compare TEY of 45611, 45612, 45613</b> · <b>Save the last plot</b>
    </div>
  </div>

  <div id="input-area">
    <input type="text" id="msg-input" placeholder="Type your message…" autocomplete="off" autofocus />
    <button id="send-btn">Send</button>
    <button id="clear-btn">Clear</button>
  </div>

<script>
const chatArea = document.getElementById("chat-area");
const msgInput = document.getElementById("msg-input");
const sendBtn  = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

function addMessage(role, html) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.innerHTML = html;
  chatArea.appendChild(div);
  scrollToBottom();
}

function addImages(images) {
  images.forEach(b64 => {
    const div = document.createElement("div");
    div.className = "msg assistant";
    div.innerHTML = '<img src="data:image/png;base64,' + b64 + '" alt="plot" />';
    chatArea.appendChild(div);
  });
  scrollToBottom();
}

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text) return;

  // Save to input history
  inputHistory.push(text);
  historyIndex = -1;
  savedInput = "";

  addMessage("user", text);
  msgInput.value = "";
  sendBtn.disabled = true;
  msgInput.disabled = true;

  // Show typing indicator
  const typingDiv = document.createElement("div");
  typingDiv.className = "msg assistant typing";
  typingDiv.textContent = "Thinking…";
  chatArea.appendChild(typingDiv);
  scrollToBottom();

  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text}),
    });
    const data = await resp.json();

    // Remove typing indicator
    typingDiv.remove();

    // Show tool calls
    if (data.tools_used && data.tools_used.length > 0) {
      data.tools_used.forEach(t => addMessage("tool", t));
    }

    // Show images
    if (data.images && data.images.length > 0) {
      addImages(data.images);
    }

    // Show text response (render markdown)
    if (data.text) {
      const rendered = typeof marked !== "undefined" ? marked.parse(data.text) : data.text;
      addMessage("assistant", rendered);
    }

    if (data.error) {
      addMessage("tool", "❌ " + data.error);
    }
  } catch (err) {
    typingDiv.remove();
    addMessage("tool", "❌ Network error: " + err.message);
  }

  sendBtn.disabled = false;
  msgInput.disabled = false;
  msgInput.focus();
}

async function clearChat() {
  try {
    await fetch("/clear", {method: "POST"});
  } catch(e) {}
  chatArea.innerHTML = '<div class="welcome">Chat cleared. Start a new conversation!</div>';
}

// ── Input history (Up/Down arrow) ──────────────────────────────────────────
const inputHistory = [];
let historyIndex = -1;
let savedInput = "";

sendBtn.addEventListener("click", sendMessage);
msgInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  } else if (e.key === "ArrowUp") {
    if (inputHistory.length === 0) return;
    if (historyIndex === -1) {
      savedInput = msgInput.value;
      historyIndex = inputHistory.length - 1;
    } else if (historyIndex > 0) {
      historyIndex--;
    }
    msgInput.value = inputHistory[historyIndex];
    e.preventDefault();
  } else if (e.key === "ArrowDown") {
    if (historyIndex === -1) return;
    if (historyIndex < inputHistory.length - 1) {
      historyIndex++;
      msgInput.value = inputHistory[historyIndex];
    } else {
      historyIndex = -1;
      msgInput.value = savedInput;
    }
    e.preventDefault();
  }
});
clearBtn.addEventListener("click", clearChat);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400
    try:
        result = agent_chat(user_msg)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "text": "", "images": [], "tools_used": []}), 500


@app.route("/clear", methods=["POST"])
def clear_endpoint():
    global conversation
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("=" * 60)
    print("  🤖 XAS Agent Chat")
    print("  Open http://localhost:5050 in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5050, debug=False)
