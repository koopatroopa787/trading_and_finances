#!/usr/bin/env python3
"""
Convenience script to run the Streamlit dashboard.

This script ensures proper Python path setup and runs the dashboard
from the correct directory.

Usage:
    python run_dashboard.py
    python run_dashboard.py --port 8502
"""
import sys
import os
from pathlib import Path
import subprocess

# Get project root
project_root = Path(__file__).parent.absolute()

# Add to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change to project root
os.chdir(project_root)

# Get port from command line args if provided
port = "8501"
if len(sys.argv) > 1:
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = sys.argv[i + 1]
            break

# Run streamlit
dashboard_path = project_root / "dashboard" / "app.py"

print("=" * 60)
print("Starting Streamlit Dashboard")
print("=" * 60)
print(f"Project root: {project_root}")
print(f"Dashboard: {dashboard_path}")
print(f"Port: {port}")
print(f"URL: http://localhost:{port}")
print("=" * 60)
print("\nPress Ctrl+C to stop the server\n")

try:
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", port,
        "--server.address", "localhost"
    ])
except KeyboardInterrupt:
    print("\n\nStopping dashboard...")
except Exception as e:
    print(f"\nError running dashboard: {e}")
    print("\nTry installing streamlit:")
    print("  pip install streamlit>=1.25.0")
    sys.exit(1)
