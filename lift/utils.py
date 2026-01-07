import io
from pathlib import Path

import pandas as pd


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
