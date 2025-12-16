from pathlib import Path
import sys

import pandas as pd


def transpose_csv(
    input_filename: Path | str,
    save: bool = False,
):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_filename, header=None)

    # Transpose the DataFrame
    df_transposed = df.transpose()

    # Generate the output filename by appending '_transposed' before the extension
    output_filename = input_filename.replace(".csv", "_transposed.csv")

    # Save the transposed DataFrame to a new CSV file
    if save:
        df_transposed.to_csv(output_filename, header=False, index=False)
        print(f"Transposed file saved as: {output_filename}")
    return df_transposed


if __name__ == "__main__":
    # Ensure the user provides a filename as an argument
    if len(sys.argv) != 2:
        print("Usage: python transpose_csv.py <input_filename>")
    else:
        filename = sys.argv[1]
        transpose_csv(filename, save=True)
