"""
Clean the BITRE airline on-time performance dataset.

This script standardises column names and airline names,
creates data-quality flags, and produces cleaned datasets
for analysis and modelling.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the location of the original BITRE Excel workbook.
RAW_DATA_PATH = Path(
    "data/raw/OTP_Time_Series_Master_Current_5.xlsx"
)

# Define the directory for generated processed datasets.
PROCESSED_DATA_DIR = Path("data/processed")

# Select the worksheet containing the most recent data.
SHEET_NAME = "2023-26 OTP"

# Define output file locations.
CLEAN_DATA_PATH = (
    PROCESSED_DATA_DIR / "bitre_otp_clean.csv"
)

ROUTE_DATA_PATH = (
    PROCESSED_DATA_DIR / "bitre_otp_route_level.csv"
)


def main() -> None:
    """Clean and prepare the BITRE dataset."""

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    # Confirm that the original dataset exists.
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"BITRE dataset was not found at: {RAW_DATA_PATH}"
        )

    # Create the processed-data directory if required.
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Load raw data
    # ---------------------------------------------------------

    # Load the selected BITRE worksheet.
    df = pd.read_excel(
        RAW_DATA_PATH,
        sheet_name=SHEET_NAME
    )

    print("Raw dataset shape:")
    print("------------------")
    print(df.shape)

    # ---------------------------------------------------------
    # Standardise column names
    # ---------------------------------------------------------

    # Replace source column names with consistent snake_case names.
    column_mapping = {
        "Route": "route",
        "Departing Port": "departing_port",
        "Arriving Port": "arriving_port",
        "Airline": "airline",
        "Month": "month",
        "Sectors Scheduled": "sectors_scheduled",
        "Sectors Flown": "sectors_flown",
        "Cancellations": "cancellations",
        "Departures On Time": "departures_on_time",
        "Arrivals On Time": "arrivals_on_time",
        "Departures Delayed": "departures_delayed",
        "Arrivals Delayed": "arrivals_delayed",
        "OnTime Departures \n(%)": "on_time_departures_pct",
        "OnTime Arrivals \n(%)": "on_time_arrivals_pct",
        "Cancellations \n\n(%)": "cancellations_pct",
    }

    df = df.rename(columns=column_mapping)

    # ---------------------------------------------------------
    # Standardise text values
    # ---------------------------------------------------------

    # Remove leading or trailing whitespace from text columns.
    text_columns = [
        "route",
        "departing_port",
        "arriving_port",
        "airline",
    ]

    for column in text_columns:
        df[column] = df[column].str.strip()

    # Correct the inconsistent capitalisation identified
    # during the validation stage.
    df["airline"] = df["airline"].replace(
        {
            "virgin Australia": "Virgin Australia",
        }
    )

    # ---------------------------------------------------------
    # Standardise date values
    # ---------------------------------------------------------

    # Ensure the month column uses pandas datetime values.
    df["month"] = pd.to_datetime(
        df["month"],
        errors="raise"
    )

    # ---------------------------------------------------------
    # Create aggregate flags
    # ---------------------------------------------------------

    # Flag observations containing results across all airlines.
    df["is_all_airlines"] = (
        df["airline"] == "All Airlines"
    )

    # Flag observations containing network-wide airport totals.
    df["is_all_ports"] = (
        df["route"] == "All Ports-All Ports"
    )

    # Identify observations representing an actual
    # airline-route-month combination.
    df["is_route_level"] = (
        ~df["is_all_airlines"]
        & ~df["is_all_ports"]
    )

    # ---------------------------------------------------------
    # Create data-quality flags
    # ---------------------------------------------------------

    # Check that scheduled sectors equal flown sectors
    # plus cancellations.
    df["scheduled_sector_valid"] = (
        df["sectors_scheduled"]
        == (
            df["sectors_flown"]
            + df["cancellations"]
        )
    )

    # Check that flown sectors equal the sum of
    # on-time and delayed departures.
    df["departure_classification_valid"] = (
        df["sectors_flown"]
        == (
            df["departures_on_time"]
            + df["departures_delayed"]
        )
    )

    # Check that flown sectors equal the sum of
    # on-time and delayed arrivals.
    df["arrival_classification_valid"] = (
        df["sectors_flown"]
        == (
            df["arrivals_on_time"]
            + df["arrivals_delayed"]
        )
    )

    # Flag rows where an on-time percentage cannot be
    # calculated because no sectors were flown.
    df["no_flown_sectors"] = (
        df["sectors_flown"] == 0
    )

    # ---------------------------------------------------------
    # Validate cleaned data
    # ---------------------------------------------------------

    # Confirm that no unexpected duplicate business keys exist.
    duplicate_keys = df.duplicated(
        subset=[
            "route",
            "airline",
            "month",
        ]
    ).sum()

    if duplicate_keys > 0:
        raise ValueError(
            f"Found {duplicate_keys} duplicate business keys."
        )

    # Confirm that all scheduled-sector relationships are valid.
    invalid_scheduled_rows = (
        ~df["scheduled_sector_valid"]
    ).sum()

    if invalid_scheduled_rows > 0:
        raise ValueError(
            "Invalid scheduled-sector relationships detected."
        )

    # ---------------------------------------------------------
    # Create route-level modelling dataset
    # ---------------------------------------------------------

    # Keep only individual airline and actual route observations.
    route_df = df[
        df["is_route_level"]
    ].copy()

    # ---------------------------------------------------------
    # Save processed datasets
    # ---------------------------------------------------------

    # Save the full cleaned dataset, including aggregate rows.
    df.to_csv(
        CLEAN_DATA_PATH,
        index=False
    )

    # Save the route-level dataset for later modelling.
    route_df.to_csv(
        ROUTE_DATA_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Processing summary
    # ---------------------------------------------------------

    print("\nCleaning complete")
    print("-----------------")

    print(f"Clean dataset rows: {len(df)}")
    print(f"Route-level rows:   {len(route_df)}")

    print(
        f"All Airlines rows:  "
        f"{df['is_all_airlines'].sum()}"
    )

    print(
        f"All Ports rows:     "
        f"{df['is_all_ports'].sum()}"
    )

    print(
        f"Zero-flight rows:   "
        f"{df['no_flown_sectors'].sum()}"
    )

    print(
        f"Invalid departure classifications: "
        f"{(~df['departure_classification_valid']).sum()}"
    )

    print(
        f"Invalid arrival classifications:   "
        f"{(~df['arrival_classification_valid']).sum()}"
    )

    print("\nOutput files:")
    print("-------------")
    print(CLEAN_DATA_PATH)
    print(ROUTE_DATA_PATH)


# Run the main function only when this script is executed directly.
if __name__ == "__main__":
    main()