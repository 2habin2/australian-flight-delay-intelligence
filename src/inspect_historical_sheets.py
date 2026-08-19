"""
Inspect the structure of all BITRE historical worksheets.

This script compares worksheet shapes and column names before
the historical data is combined into one dataset.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the location of the original BITRE workbook.
DATA_PATH = Path(
    "data/raw/OTP_Time_Series_Master_Current_5.xlsx"
)


def main() -> None:
    """Inspect the structure of every worksheet in the workbook."""

    # Confirm that the raw workbook exists.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"BITRE dataset was not found at: {DATA_PATH}"
        )

    # Open the workbook without loading all worksheets at once.
    excel_file = pd.ExcelFile(DATA_PATH)

    print("Historical worksheet inspection")
    print("===============================\n")

    # Store the first worksheet's columns as the reference schema.
    reference_columns = None

    # Inspect every worksheet individually.
    for sheet_name in excel_file.sheet_names:

        # Load the current worksheet.
        df = pd.read_excel(
            DATA_PATH,
            sheet_name=sheet_name
        )

        # Set the first worksheet as the reference schema.
        if reference_columns is None:
            reference_columns = list(df.columns)

        # Compare the current worksheet's columns
        # with the reference worksheet.
        schema_matches = (
            list(df.columns) == reference_columns
        )

        print(f"Worksheet: {sheet_name}")
        print(f"Shape: {df.shape}")
        print(f"Schema matches reference: {schema_matches}")

        # Display the column names if the schema differs.
        if not schema_matches:
            print("Columns:")
            for column in df.columns:
                print(f"  - {column}")

        # Display the available date range when Month exists.
        if "Month" in df.columns:
            month_values = pd.to_datetime(
                df["Month"],
                errors="coerce"
            )

            print(
                f"Date range: "
                f"{month_values.min()} to {month_values.max()}"
            )

        print("-" * 50)


# Run the main function only when this script is executed directly.
if __name__ == "__main__":
    main()