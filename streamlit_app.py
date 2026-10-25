"""
VinBigData_XAI - Streamlit Entry Point

Run from project root: streamlit run streamlit_app.py
This wrapper ensures proper imports by setting sys.path.
"""

import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent

# Add to sys.path so app.ui.* imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load and execute app.py with proper context
app_file = PROJECT_ROOT / "app" / "ui" / "app.py"
with open(app_file) as f:
    code = f.read()

# Execute in a namespace with __file__ set correctly
namespace = {
    "__file__": str(app_file),
    "__name__": "__main__",
}
exec(code, namespace)  # noqa: S102
