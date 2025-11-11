"""LIFT entry-point module.

Purpose:
- Starts the Streamlit frontend for the LIFT application when invoked via
  `python -m lift` or the console script entry point `lift`.

Relationships:
- Launches `lift/frontend/app.py` in a Streamlit process.
- Ensures the working directory is the package root so Streamlit can load
  configuration and static resources.

Key Logic:
- Resolve package root, chdir to it, compute path to `frontend/app.py`,
  and execute `python -m streamlit run <app.py>` using the current interpreter.
"""

import os
from pathlib import Path
import sys
import subprocess


def main():
    # get package root directory
    package_root = Path(__file__).parent
    # change working directory to package root; required for streamlit to find config.toml resources
    os.chdir(package_root)
    # get the absolute path to the frontend.py file in the 'lift' package
    app_path = package_root / "frontend" / "app.py"

    # required to avoid manual call of streamlit run path/to/lift/lift/frontend.py
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
