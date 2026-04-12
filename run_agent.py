#!/usr/bin/env python3
"""
XAS Agent Launcher
==================
Run this script to start the XAS AI Agent chat interface.
The browser will open automatically at http://localhost:5050.

Usage:
    python run_agent.py
"""

import subprocess
import sys
import time
import socket
import webbrowser

PORT = 5050


def _port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def main():
    if _port_in_use(PORT):
        print(f"⚠️  Port {PORT} is already in use.")
        print(f"   The agent may already be running at http://localhost:{PORT}")
        print(f"   Opening browser...")
        webbrowser.open(f"http://localhost:{PORT}")
        return

    print("🚀 Starting XAS Agent Chat server...")

    # Start the Flask server as a subprocess
    proc = subprocess.Popen(
        [sys.executable, "chat_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait until the server is actually listening
    for _ in range(60):  # up to 30 seconds
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    else:
        print("❌ Server did not start in time. Check chat_app.py for errors.")
        proc.terminate()
        return

    print(f"✅ Server running at http://localhost:{PORT}")
    print("   Opening browser...")
    webbrowser.open(f"http://localhost:{PORT}")
    print("   Press Ctrl+C to stop the server.\n")

    # Stream server output until interrupted
    try:
        for line in proc.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        proc.terminate()
        proc.wait()
        print("✅ Server stopped.")


if __name__ == "__main__":
    main()
