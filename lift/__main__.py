import os
from pathlib import Path
import sys
import subprocess


# change working directory to package root; required for streamlit to find config.toml resources
package_root = Path(__file__).parent
os.chdir(package_root)

def main():
    # get the absolute path to the frontend.py file in the 'lift' package
    app_path = Path(__file__).resolve().parent / 'frontend' / 'frontend.py'

    # required to avoid manual call of streamlit run path/to/lift/lift/frontend.py
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()