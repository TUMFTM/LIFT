from pathlib import Path
import sys
import subprocess



def main():
    # get the absolute path to the app.py file in the 'lift' package
    app_path = Path(__file__).resolve().parent / 'frontend.py'

    # required to avoid manual call of streamlit run path/to/lift/lift/app.py
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()