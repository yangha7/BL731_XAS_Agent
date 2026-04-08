# XAS AI Agent

An interactive AI-powered assistant for X-ray Absorption Spectroscopy (XAS) data analysis, built with Flask and the CBORG LLM API.

## Overview

This project provides a conversational chat interface where users can explore, visualize, and analyze XAS scan data collected at a synchrotron beamline. The agent understands natural language requests and uses a suite of analysis tools to process rare-earth M4,5-edge (and other) XAS spectra.

## Project Structure

```
731_Agent/
├── chat_app.py          # Flask web app — chat UI + tool definitions + agent loop
├── xas_utils.py         # Core XAS data utilities (parsing, normalization, derivatives, peaks)
├── XAS_Agent.ipynb      # Jupyter notebook — launches the Flask app
├── .env                 # API key (CBORG_API_KEY)
├── 731_Data/            # Scan data files (SigScan*.txt)
│   ├── SigScan45611.txt
│   ├── SigScan45612.txt
│   └── ...
├── exported_data/       # Saved/exported analysis results
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

### Option 1: From the Jupyter Notebook

1. Open `XAS_Agent.ipynb` in VS Code or JupyterLab
2. Run all cells — the last cell launches the Flask server and opens your browser
3. Chat with the agent at `http://localhost:5050`

### Option 2: Directly from the Terminal

```bash
cd 731_Agent
python chat_app.py
```

Then open `http://localhost:5050` in your browser.

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
| **list_scans** | List all available scan files | *"What scans do we have?"* |
| **plot_scan** | Plot a single scan (raw or signal/I0) | *"Plot scan 45616 MCP"* |
| **compare_scans** | Overlay multiple scans on one plot | *"Compare scans 45611 and 45612 TEY"* |
| **show_scan_info** | Show scan metadata (date, file path, columns, etc.) | *"Show info for scan 45616"* |
| **normalize_scan** | Athena-style pre-edge subtraction and post-edge normalization | *"Normalize scan 45616 MCP"* |
| **derivative_scan** | Compute smoothed 1st or 2nd derivative (Savitzky-Golay) | *"Show the 2nd derivative of scan 45616 MCP"* |
| **find_peaks_scan** | Detect peaks and shoulders with tunable sensitivity | *"Find peaks in scan 45616 MCP"* |
| **identify_edge** | Identify element and absorption edge from peak energies + metadata | *"What element is scan 45616?"* |
| **save_data** | Export the last plotted data to a text file | *"Save that data"* |

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
     xas_utils.py (data processing)
       ↕ File I/O
     731_Data/*.txt (scan files)
```

The agent uses an iterative tool-calling loop: the LLM receives the user message, decides which tool(s) to call, the server executes them and returns results (including base64-encoded plot images), and the LLM formulates a final response.

## License

Internal use — LBNL / ALS beamline data analysis.
