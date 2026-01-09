import io
import os
from functools import wraps
from pathlib import Path

import pandas as pd
import streamlit as st


# store flag in current process and delete environment variable
# flag is only set by frontend.py and therefore only exists when a streamlit app is running
_use_streamlit_cache = os.environ.pop("LIFT_USE_STREAMLIT_CACHE", "0") == "1"


def safe_cache_data(*dargs, **dkwargs):
    """
    Safe replacement for st.cache_data:
    - Detects Streamlit context by checking the `LIFT_USE_STREAMLIT_CACHE` environment variable (set by frontend).
    - In Streamlit app: uses st.cache_data
    - Outside Streamlit: no-op
    Supports both @safe_cache_data and @safe_cache_data(...)
    """

    def decorator(func):
        if _use_streamlit_cache:
            return st.cache_data(**dkwargs)(func)
        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

    # Used without arguments -> @safe_cache_data
    if dargs and callable(dargs[0]):
        return decorator(dargs[0])

    # Used with arguments -> @safe_cache_data(...)
    return decorator


def read_csv(input_filename: Path | str, transpose: bool = False) -> pd.DataFrame:
    """Reads a CSV file into a pandas DataFrame, with optional transposition.

    This function loads a CSV file into a pandas DataFrame using a two-level
    column header and the first column as the index. If `transpose` is True,
    the CSV is first read without headers, transposed, and then re-read from
    an in-memory buffer to ensure pandas correctly infers and applies column
    dtypes after transposition. This is useful for CSV files which formatted
    in a human-readable way.

    Args:
        input_filename (Path | str): Path to the CSV file to read.
        transpose (bool, optional): Whether to transpose the CSV contents
            before loading them into the DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: The resulting DataFrame with a two-level column header
        and the first column set as the index.
    """

    input_file_buffer = input_filename

    if transpose:
        # pd.read_csv() already applies dtypes to columns when reading.
        # Therefore, we need to first read the CSV, transpose it, and then read it again from a buffer to correctly apply dtypes.
        input_file_buffer = io.StringIO()
        pd.read_csv(input_filename, header=None, index_col=0).transpose().to_csv(
            input_file_buffer, header=True, index=False
        )
        input_file_buffer.seek(0)

    return pd.read_csv(input_file_buffer, index_col=0, header=[0, 1])
