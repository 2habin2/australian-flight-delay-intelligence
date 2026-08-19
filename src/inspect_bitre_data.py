"""
Inspect the BITRE airline on-time performance dataset.

This script loads the most recent BITRE worksheet and performs
basic structural checks before data cleaning begins.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the location of the original BITRE Excel workbook.
DATA_PATH = Path(
    "data/raw/OTP_Time_Series_Master_Current_5.xlsx"
)

# Select the worksheet containing the most recent data.
SHEET_NAME = "2023-26 OTP"


def main() -> None:
    """Inspect the BITRE airline on-time performance dataset."""

    # Check that the raw dataset exists before attempting to load it.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"BITRE dataset was not found at: {DATA_PATH}"
        )

    # Load the worksheet using the first row as column names.
    df = pd.read_excel(
        DATA_PATH,
        sheet_name=SHEET_NAME
    )

    # ---------------------------------------------------------
    # Dataset dimensions
    # ---------------------------------------------------------

    print("Dataset shape:")
    print("--------------")
    print(df.shape)

    # ---------------------------------------------------------
    # Column names
    # ---------------------------------------------------------

    print("\nColumns:")
    print("--------")

    for column in df.columns:
        print(column)

    # ---------------------------------------------------------
    # Data types
    # ---------------------------------------------------------

    print("\nData types:")
    print("-----------")
    print(df.dtypes)

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    print("\nMissing values:")
    print("---------------")
    print(df.isna().sum())

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print("\nDuplicate rows:")
    print("---------------")
    print(duplicate_count)

    # ---------------------------------------------------------
    # Date range
    # ---------------------------------------------------------

    # Convert Month to datetime to support date-based analysis.
    df["Month"] = pd.to_datetime(
        df["Month"],
        errors="coerce"
    )

    print("\nDate range:")
    print("-----------")
    print(f"Earliest month: {df['Month'].min()}")
    print(f"Latest month:   {df['Month'].max()}")

    # ---------------------------------------------------------
    # Airlines
    # ---------------------------------------------------------

    print("\nAirlines:")
    print("---------")

    for airline in sorted(df["Airline"].dropna().unique()):
        print(airline)

    # ---------------------------------------------------------
    # Airports
    # ---------------------------------------------------------

    departing_airports = df["Departing Port"].nunique()
    arriving_airports = df["Arriving Port"].nunique()

    print("\nAirport counts:")
    print("---------------")
    print(f"Unique departing airports: {departing_airports}")
    print(f"Unique arriving airports:  {arriving_airports}")

    # ---------------------------------------------------------
    # Preview
    # ---------------------------------------------------------

    print("\nFirst five observations:")
    print("------------------------")
    print(df.head().to_string(index=False))


# Run the main function only when this script is executed directly.
if __name__ == "__main__":
    main()

