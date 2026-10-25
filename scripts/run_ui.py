
from pathlib import Path
import subprocess
import sys
import os


PRODUCT_ROOT = Path(__file__).resolve().parent.parent

# Change to project root so imports work correctly
os.chdir(str(PRODUCT_ROOT))

STREAMLIT_APP = (
    PRODUCT_ROOT
    / "streamlit_app.py"
)


if not STREAMLIT_APP.exists():

    raise FileNotFoundError(
        f"Streamlit app wrapper not found: {STREAMLIT_APP}"
    )


command = [

    sys.executable,

    "-m",

    "streamlit",

    "run",

    str(STREAMLIT_APP),
]


if __name__ == "__main__":

    subprocess.run(command)

    "--server.address",

    "127.0.0.1",

    "--server.port",

    "8501",




subprocess.run(
    command,
    cwd=str(
        PRODUCT_ROOT
    ),
    check=True,
)
