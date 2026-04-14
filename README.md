# XAS AI Agent

An interactive AI-powered assistant for X-ray Absorption Spectroscopy (XAS) data analysis, built with Flask and the CBORG LLM API.

## Overview

This project provides a conversational chat interface where users can explore, visualize, and analyze XAS scan data collected at a synchrotron beamline. The agent understands natural language requests and uses a suite of analysis tools to process soft XAS spectra at Bl731 at ALS. This includes C, N, O K-edge, 3d transition metal L-edge, and the rare earth elements M4,5 edge.

## Changelog

### Branch: pc-windows-config (Windows-specific configuration)
**Date:** April 14, 2026  
**Status:** Active development branch (not merged to main)

**Changes:**
- **File Explorer Refresh Fix**: Disabled automatic file explorer refresh after export operations (`save_data`, `save_image`). File explorer now only refreshes when manually clicking the ⟳ button in the sidebar header.
- **Windows Compatibility**: Added Windows-specific batch file launcher (`XAS_Agent.bat`) on desktop for one-click startup with comprehensive error checking.
- **Environment Configuration**: Configured custom data directories for Windows environment (`XAS_DATA_DIR`, `XAS_EXPORT_DIR`, `EXP_INFO_DIR`).
- **CORS Support**: Added Flask-CORS for cross-origin requests in web interface.
- **Code Cleanup**: Removed user-generated `exp_info.txt` file (should not be in repository).

**Technical Details:**
- Modified `chat_app.py`: Commented out automatic refresh logic in JavaScript (lines ~2790-2792)
- Added comprehensive validation in batch file: Python path check, directory existence, file presence
- Environment variables properly loaded before module imports
- All changes are PC-specific and should not be merged to main branch

## Project Structure

```
731_Agent/
├── run_agent.py         # ⭐ One-click launcher — starts server + opens browser
├── chat_app.py          # Flask web app — chat UI + tool definitions + agent loop
├── xas_utils.py         # Core XAS data utilities (parsing, normalization, derivatives, peaks)
├── exp_info.py          # Experiment info manager — persistent scan metadata/comments
├── XAS_Agent.ipynb      # Jupyter notebook — alternative launcher
├── .env                 # API key (CBORG_API_KEY)
├── 731_Data/            # Scan data files (SigScan*.txt)
│   ├── SigScan45611.txt
│   ├── SigScan45612.txt
│   ├── 260401/          # Date-coded subdirectories (YYMMDD)
│   └── ...
├── exp_info/            # Persistent experiment metadata (long-term memory)
│   └── exp_info.txt     # JSON dictionary of scan comments, sorted by scan number
├── exported_data/       # Saved/exported analysis results (temporary, can be deleted)
│   ├── *.txt / *.csv    # Exported data files (txt, csv, tsv, dat)
│   ├── renamed/         # Renamed scan copies (full raw data)
│   ├── calibrated/      # Energy-calibrated scan copies
│   └── images/          # Saved plot images (PNG)
└── README.md            # This file
```

## Prerequisites

### Python Packages

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install flask numpy pandas matplotlib scipy python-dotenv openai
```

### LLM Provider Setup

The agent supports multiple LLM providers. Set **one** API key in the `.env` file and the agent auto-detects which provider to use:

| Provider | Environment Variable | Default Model | Base URL |
|----------|---------------------|---------------|----------|
| **CBORG** (LBL) | `CBORG_API_KEY` | `claude-sonnet` | `https://api.cborg.lbl.gov/v1` |
| **OpenAI** | `OPENAI_API_KEY` | `gpt-4o` | `https://api.openai.com/v1` |
| **Google Gemini** | `GEMINI_API_KEY` | `gemini-2.0-flash` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| **Anthropic Claude** | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` | `https://api.anthropic.com/v1` |

Example `.env` configurations:

```bash
# CBORG (LBL users)
CBORG_API_KEY=sk-...

# OpenAI
OPENAI_API_KEY=sk-...

# Google Gemini
GEMINI_API_KEY=AIza...

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...
```

**Optional overrides** (in `.env`):

```bash
LLM_PROVIDER=openai          # Force a specific provider
LLM_MODEL=gpt-4o-mini        # Override the default model
LLM_BASE_URL=https://...     # Override the API base URL
```

If multiple API keys are set, the agent picks the first one found in this order: CBORG → OpenAI → Gemini → Claude. Use `LLM_PROVIDER` to override.

### Data Directory

By default, the agent looks for scan data in `731_Data/` relative to the project root. To deploy on a different computer or point to a different data location, set `XAS_DATA_DIR` in `.env`:

```bash
XAS_DATA_DIR=/path/to/your/scan/data    # absolute or relative path
XAS_EXPORT_DIR=/path/to/exported/data   # where saved data goes (default: exported_data)
```

## How to Run

### Option 1: One-Click Launcher (Recommended)

```bash
cd 731_Agent
python run_agent.py
```

This starts the Flask server and **automatically opens your browser** at `http://localhost:5050`. Press `Ctrl+C` to stop the server.

### Option 2: Direct Server Start

```bash
cd 731_Agent
python chat_app.py
```

Then open `http://localhost:5050` in your browser manually.

### Option 3: From the Jupyter Notebook

1. Open `XAS_Agent.ipynb` in VS Code or JupyterLab
2. Run all cells — the last cell launches the Flask server and opens your browser
3. Chat with the agent at `http://localhost:5050`

## Data Directory Structure

The agent recursively scans the `731_Data/` directory for scan files, including all subdirectories. For example:

```
731_Data/
├── SigScan45611.txt          # top-level scans
├── SigScan45616.txt
├── 260401/                   # subdirectory by date
│   ├── SigScan45652.txt
│   ├── SigScan45666-0001.txt # multi-segment scans
│   └── ...
├── 260402/
│   └── ...
└── 260406/
    └── ...
```

You can refer to scans by just their number (e.g., *"plot scan 45652"*) — the agent will find them automatically regardless of which subdirectory they're in. The `list_scans` tool shows subdirectory prefixes so you know where each scan lives.

## Available Tools

The agent has access to the following analysis tools, which it calls automatically based on your natural language requests:

| Tool | Description | Example Prompt |
|------|-------------|----------------|
| **list_scans** | List available scan files (with date filtering) | *"What scans do we have?"*, *"List scans from 260401"* |
| **list_exports** | List all files in exported_data/ (renamed, calibrated, saved) | *"What files are in exported_data?"* |
| **plot_scan** | Plot a single scan (raw or signal/I0) | *"Plot scan 45616 MCP"* |
| **compare_scans** | Overlay multiple scans on one plot | *"Compare scans 45611 and 45612 TEY"* |
| **compare_files** | Overlay multiple exported data files on one plot | *"Plot RuCl3 and RuO2 TEY together"* |
| **plot_file** | Plot a generic two-column data file | *"Plot exported_data/RuCl3_Powder_TEY_I0.txt"* |
| **show_scan_info** | Show scan metadata (date, file path, columns, user notes) | *"Show info for scan 45616"* |
| **normalize_scan** | Athena-style pre-edge subtraction and post-edge normalization | *"Normalize scan 45616 MCP"* |
| **derivative_scan** | Compute smoothed 1st or 2nd derivative (Savitzky-Golay) | *"Show the 2nd derivative of scan 45616 MCP"* |
| **find_peaks_scan** | Detect peaks and shoulders with tunable sensitivity | *"Find peaks in scan 45616 MCP"* |
| **identify_edge** | Identify element and absorption edge from peak energies + metadata | *"What element is scan 45616?"* |
| **save_data** | Export the last plotted data to a file (txt, csv, tsv, dat) | *"Save that data as results.csv"* |
| **save_image** | Export the last plot as a PNG image | *"Save the plot as PNG"* |
| **rename_scan** | Copy a scan with a descriptive name (original unchanged) | *"Rename scan 45616 to Cu_L3_edge_sample_A"* |
| **calibrate_scans** | Apply energy calibration shift to scan files in batch | *"Calibrate all scans from 260401"* |
| **update_exp_info** | Add/update user comments for a scan (persistent metadata) | *"45679 is a calibration scan for TiO2"* |
| **search_exp_info** | Search experiment info by keyword or list all entries | *"Which scans are TiO2?"*, *"Show all notes"* |

### Peak Detection Sensitivity

The `find_peaks_scan` tool supports four sensitivity levels:

| Level | Behavior |
|-------|----------|
| `low` | Only major peaks |
| `normal` | Peaks + prominent shoulders (default) |
| `high` | More peaks + subtle shoulders |
| `very_high` | All features including minor inflections |

Example: *"Find peaks in scan 45616 MCP with high sensitivity"*

## Chat Interface Features

- **Markdown rendering** — agent responses are rendered with formatted text, tables, and code blocks
- **Inline plots** — all plots appear directly in the chat as images
- **Input history** — press ↑/↓ arrow keys to recall previous messages
- **Clear chat** — click the 🗑️ button to reset the conversation
- **File explorer sidebar** — browse scan files and exported data with a collapsible tree view
- **Energy calibration panel** — set reference/measured energy values and apply calibration shifts in batch
- **Clean exported data** — one-click button to clear all exported files (with confirmation dialog)
- **Quick-link file insertion** — click any file in the sidebar to insert its name into the chat input
- **Confirmation popup** — when modifying existing scan comments, a modal dialog with Yes/No buttons appears for user confirmation (prevents accidental overwrites)

### Plot Styling

The agent supports extensive plot customization through natural language:

- **Line styles**: color, linestyle (`-`, `--`, `-.`, `:`), linewidth
- **Per-curve styles**: pass a `styles` array in `compare_scans` for individual curve styling
- **Axis styling**: font family, title/label/tick sizes, legend size
- **Per-axis colors**: independent colors for X-axis, left Y-axis, and right Y-axis labels and ticks
- **Custom labels and titles**: set legend labels and plot titles via natural language

Example: *"Use Arial font, make the title 16pt, color the left axis blue and right axis red"*

### Auto-Scale Modes

When comparing multiple spectra with very different intensity ranges, use `auto_scale`:

| Mode | Behavior | Example Prompt |
|------|----------|----------------|
| `overlay` | Normalize each spectrum to [0,1] and overlay at same baseline | *"Auto scale these spectra"* |
| `offset` | Normalize each spectrum to [0,1] and stack with vertical gaps | *"Auto scale with offset"* |

The default mode is `overlay` — all curves are normalized and overlaid for direct shape comparison.

### Dual-Axis Plotting

Plot two different signals on left and right Y-axes:

```
"Compare scans 45611 and 45612 with TEY on left and MCP on right"
```

Each axis has independent color styling via `y_label_color` / `y_right_label_color` and `y_tick_color` / `y_right_tick_color`.

### Batch Calibration

Apply energy calibration to multiple scans at once:

1. Enable the calibration checkbox in the sidebar
2. Set reference and measured energy values
3. Ask: *"Calibrate all scans from 260401"* or *"Calibrate scans 45611, 45612, 45613"*

Calibrated copies are saved to `exported_data/calibrated/` with `_cal` suffix.

### Working with Exported Data

- Scans in `exported_data/` are automatically searchable by scan ID (e.g., `plot_scan`, `compare_scans`)
- Two-column data files can be plotted with `plot_file` or compared with `compare_files`
- Use `list_exports` to discover what files have been exported
- All file tools support `e_min`/`e_max` for energy range filtering
- Data can be exported as `.txt` (tab-separated), `.csv` (comma-separated), `.tsv`, or `.dat` — the format is determined by the file extension you specify

### Experiment Info (Persistent Scan Metadata)

The agent maintains a persistent experiment info file (`exp_info/exp_info.txt`) that stores user-provided comments about scans. This serves as a **long-term memory** that persists even if `exported_data/` is deleted.

**How it works:**

- When you tell the agent about a scan (e.g., *"45679 is a calibration scan for TiO2"*), it automatically records the comment with a timestamp from the scan file header
- Comments are stored as a JSON dictionary sorted by scan number, with entries like:
  ```json
  {
    "45679": "[4/2/2026 13:48:26] calibration scan for TiO2",
    "45680": "[4/2/2026 14:15:03] Fe L-edge sample A at 300K"
  }
  ```
- You can search by keyword: *"Which scans are TiO2?"*, *"Plot the most recent TiO2 calibration scan"*
- You can correct mistakes: *"Actually 45679 is RuO2, not TiO2"* — a confirmation popup appears showing old vs. new comment before replacing
- When you ask for scan info (`show_scan_info`), user notes are automatically included
- Time-based queries work too: *"Show me the first TiO2 scan from April"*

**Confirmation popup for comment changes:**

- **First comment** for a scan is saved directly — no confirmation needed
- **Any subsequent change** (append or replace) triggers a modal popup showing the current comment and the proposed new comment
- The user must click **"Yes, Replace"** or **"No, Keep Original"** — this prevents accidental overwrites
- Clicking outside the popup or pressing the background is treated as "No"
- The confirmation is enforced at the backend level, so it works reliably regardless of LLM behavior

**Key design decisions:**

- `exp_info/` is separate from `exported_data/` — it's meant to be kept long-term
- Only the user's exact words are stored (plus the auto-prepended timestamp)
- Duplicate comments are detected and skipped; new details are appended with ` | ` separator
- The scan number in filenames determines chronological order (higher = more recent)

### Numbered Options

When the agent presents multiple choices, they are shown as a numbered list. You can simply type the number (e.g., `1`) to select an option instead of typing the full request:

```
Agent: Here are some options for scan 45679:
  1. Plot the TEY signal
  2. Show the scan metadata
  3. Compare with the previous calibration scan
  Or type your own request if none of the above applies.

You: 1
→ Executes "Plot the TEY signal"
```

## Data Format

Scan files (`SigScan*.txt`) are tab-separated text files with a 15-line header containing metadata, followed by column headers and data rows.

### Key Columns

| Column | Description |
|--------|-------------|
| `Mono eV Calib high E Grating` | Photon energy (eV) |
| `I0` | Incident beam intensity |
| `TEY UHV XAS` | Total Electron Yield signal |
| `TFY UHV XAS` | Total Fluorescence Yield signal |
| `MCP Np` | MicroChannel Plate signal |

### Header Metadata (Lines 1–15)

| Line | Content |
|------|---------|
| 1 | Date |
| 2 | Blank |
| 3 | "From File" |
| 4 | Grating info |
| 5 | Source scan file path (contains element/edge hint, e.g., `Ce M45 edge.txt`) |
| 6–8 | Calibration values |
| 9 | Delay after move (s) |
| 10 | Count time (s) — see Known Issues |
| 11 | Scan number |
| 12 | Bi-directional flag |
| 13 | Stay at end |
| 14 | Description length |
| 15 | Blank |

## Analysis Methods

### Normalization (Athena-style)

Implements the standard XAFS normalization procedure:
1. **E0 detection** — maximum of the smoothed 1st derivative
2. **Pre-edge line** — linear fit to the pre-edge region (default: E0 − 150 to E0 − 30 eV)
3. **Post-edge line** — quadratic fit to the post-edge region (default: E0 + 50 to E0 + 300 eV)
4. **Edge step** — difference between post-edge and pre-edge lines at E0
5. **Normalized μ(E)** — `(μ − pre_edge_line) / edge_step`

### Derivatives

Uses Savitzky-Golay filtering for smooth, noise-resistant derivatives:
- **1st derivative** — useful for E0 determination (peak = edge energy)
- **2nd derivative** — useful for identifying spectral features; all strong negative peaks (minima) are labeled with colored dashed lines on the plot

### Peak Detection

Combines two approaches:
1. **Direct peak finding** — `scipy.signal.find_peaks` on the signal with height and prominence thresholds
2. **Shoulder detection** — finds local minima in the 2nd derivative that correspond to inflection points / shoulders in the original spectrum
3. **Relative prominence filter** — removes spurious weak peaks whose prominence is < 10% of the strongest peak

### Edge Identification

Uses a built-in XAS edge energy database covering:
- 3d transition metals (K, L2,3 edges)
- Rare earth elements (M4,5, L2,3 edges)
- Light elements (C, N, O, F K edges)
- Other common elements (S, P, Si, Al, Mg, Ca, Sc)

Matches observed peak energies against reference values within ±30 eV tolerance, prioritizing element hints extracted from the scan file metadata path.

## Known Issues

### Count Time in Header Metadata

The `Count Time (s)` field in the scan file header (line 10) may report an incorrect value (e.g., `1.0 s`) when scans are run using a region file. In region-file scan mode, the actual dwell time per point is determined by the region file settings and can differ significantly from the header value (e.g., actual ~11 s per point vs. reported 1 s). **This is a scan software issue**, not an agent bug. The actual time per data point can be verified from the `Time (s)` column in the data.

## Architecture

```
User ↔ Browser (HTML/CSS/JS)
       ↕ AJAX POST /chat
     Flask Server (chat_app.py)
       ↕ OpenAI-compatible API
     CBORG LLM (claude-sonnet)
       ↕ Tool calls
     xas_utils.py (data processing)    exp_info.py (scan metadata)
       ↕ File I/O                        ↕ File I/O
     731_Data/*.txt (scan files)       exp_info/exp_info.txt (persistent)
     exported_data/ (temporary)
```

The agent uses an iterative tool-calling loop: the LLM receives the user message, decides which tool(s) to call, the server executes them and returns results (including base64-encoded plot images), and the LLM formulates a final response.

## License

Internal use — LBNL / ALS beamline data analysis.
