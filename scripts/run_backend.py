"""
VinBigData_XAI - FastAPI Backend Startup Script

Starts the production inference API on http://127.0.0.1:8765
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    print("=" * 80)
    print("Starting VinBigData_XAI FastAPI Backend")
    print("=" * 80)
    print()
    print("API will be available at: http://127.0.0.1:8765")
    print("API Docs:                http://127.0.0.1:8765/docs")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--reload",
    ]

    subprocess.run(command, cwd=str(PROJECT_ROOT))
