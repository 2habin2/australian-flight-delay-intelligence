"""
Clean the complete BITRE historical on-time performance dataset.

This script combines BITRE worksheets from 2010 to 2026,
removes non-data worksheet rows, standardises fields, creates
data-quality flags, and produces cleaned historical datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the location of the original BITRE workbook.
RAW_DATA_PATH = Path(
    "data/raw/OTP_Time_Series_Master_Current_5.xlsx"
)

# Define the directory for generated processed datasets.
PROCESSED_DATA_DIR = Path("data/processed")

# Define the historical worksheets to combine.
SHEET_NAMES = [
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023-26 OTP",
]

# Define the output file locations.
HISTORICAL_CLEAN_PATH = (
    PROCESSED_DATA_DIR
    / "bitre_otp_historical_clean.csv"
)

HISTORICAL_ROUTE_PATH = (
    PROCESSED_DATA_DIR
    / "bitre_otp_historical_route_level.csv"
)


def load_historical_data() -> pd.DataFrame:
    """Load and combine all BITRE historical worksheets."""

    # Store each worksheet before concatenation.
    dataframes = []

    # Load every historical worksheet.
    for sheet_name in SHEET_NAMES:
        sheet_df = pd.read_excel(
            RAW_DATA_PATH,
            sheet_name=sheet_name,
        )

        # Record the source worksheet for data lineage.
        sheet_df["Source Sheet"] = sheet_name

        dataframes.append(sheet_df)

    # Combine every worksheet into one DataFrame.
    historical_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    return historical_df


def main() -> None:
    """Clean and prepare the complete historical BITRE dataset."""

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    # Confirm that the original workbook exists.
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"BITRE dataset was not found at: {RAW_DATA_PATH}"
        )

    # Create the processed-data directory if required.
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load historical data
    # ---------------------------------------------------------

    df = load_historical_data()

    print("Historical workbook shape:")
    print("--------------------------")
    print(df.shape)

    # ---------------------------------------------------------
    # Identify and remove non-data rows
    # ---------------------------------------------------------

    # Convert Month values to datetime.
    # Worksheet notes and invalid values become NaT.
    df["Month"] = pd.to_datetime(
        df["Month"],
        errors="coerce",
    )

    # Identify rows that do not represent
    # airline performance observations.
    non_data_rows = (
        df["Route"].isna()
        | df["Airline"].isna()
        | df["Month"].isna()
    )

    non_data_count = non_data_rows.sum()

    # Remove worksheet notes and blank rows.
    df = df.loc[
        ~non_data_rows
    ].copy()

    print("\nNon-data rows removed:")
    print("----------------------")
    print(non_data_count)

    # ---------------------------------------------------------
    # Standardise column names
    # ---------------------------------------------------------

    # Replace source column names with consistent
    # snake_case column names.
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
        "Source Sheet": "source_sheet",
    }

    df = df.rename(
        columns=column_mapping
    )

    # ---------------------------------------------------------
    # Standardise numeric data types
    # ---------------------------------------------------------

    # Define count columns that must contain numeric values.
    count_columns = [
        "sectors_scheduled",
        "sectors_flown",
        "cancellations",
        "departures_on_time",
        "arrivals_on_time",
        "departures_delayed",
        "arrivals_delayed",
    ]

    # Convert count columns to numeric values.
    # Unexpected text values will raise an error.
    for column in count_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    # Define percentage columns that may contain explicit
    # missing-value markers such as "na" in the source workbook.
    percentage_columns = [
        "on_time_departures_pct",
        "on_time_arrivals_pct",
        "cancellations_pct",
    ]

    # Convert known text-based missing-value markers
    # to proper pandas missing values.
    missing_markers = [
        "na",
        "NA",
        "n/a",
        "N/A",
        "",
    ]

    for column in percentage_columns:
        df[column] = df[column].replace(
            missing_markers,
            pd.NA,
        )

        # Convert the remaining percentage values to numeric.
        # Unexpected non-numeric values will still raise an error.
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    # Store the worksheet identifier explicitly as text.
    df["source_sheet"] = df["source_sheet"].astype("string")
    
    # ---------------------------------------------------------
    # Standardise text values
    # ---------------------------------------------------------

    # Remove leading and trailing whitespace
    # from key text columns.
    text_columns = [
        "route",
        "departing_port",
        "arriving_port",
        "airline",
        "source_sheet",
    ]

    for column in text_columns:
        df[column] = df[column].str.strip()

    # Correct the known Virgin Australia
    # capitalisation inconsistency.
    df["airline"] = df["airline"].replace(
        {
            "virgin Australia": "Virgin Australia",
        }
    )

    # Historical airline names such as Regional Express,
    # Rex Airlines and Tigerair Australia are preserved
    # because they represent historical source terminology.

    # ---------------------------------------------------------
    # Standardise date values
    # ---------------------------------------------------------

    # Confirm that the month column uses datetime values.
    df["month"] = pd.to_datetime(
        df["month"],
        errors="raise",
    )

    # ---------------------------------------------------------
    # Create aggregate flags
    # ---------------------------------------------------------

    # Flag observations containing results
    # across all airlines.
    df["is_all_airlines"] = (
        df["airline"] == "All Airlines"
    )

    # Flag network-wide airport aggregate observations.
    df["is_all_ports"] = (
        df["route"] == "All Ports-All Ports"
    )

    # Identify observations representing an individual
    # airline operating on an actual directional route.
    df["is_route_level"] = (
        ~df["is_all_airlines"]
        & ~df["is_all_ports"]
    )

    # ---------------------------------------------------------
    # Create data-quality flags
    # ---------------------------------------------------------

    # Validate that scheduled sectors equal
    # flown sectors plus cancellations.
    df["scheduled_sector_valid"] = np.isclose(
        df["sectors_scheduled"],
        (
            df["sectors_flown"]
            + df["cancellations"]
        ),
        equal_nan=False,
    )

    # Validate departure classifications while allowing
    # for floating-point precision in historical records.
    df["departure_classification_valid"] = np.isclose(
        df["sectors_flown"],
        (
            df["departures_on_time"]
            + df["departures_delayed"]
        ),
        equal_nan=False,
    )

    # Validate arrival classifications while allowing
    # for floating-point precision in historical records.
    df["arrival_classification_valid"] = np.isclose(
        df["sectors_flown"],
        (
            df["arrivals_on_time"]
            + df["arrivals_delayed"]
        ),
        equal_nan=False,
    )

    # Flag observations where no sectors were flown.
    df["no_flown_sectors"] = (
        df["sectors_flown"] == 0
    )

    # Flag observations containing a missing
    # departure or arrival on-time percentage.
    df["missing_on_time_percentage"] = (
        df["on_time_departures_pct"].isna()
        | df["on_time_arrivals_pct"].isna()
    )

    # ---------------------------------------------------------
    # Cleaned-data validation
    # ---------------------------------------------------------

    # Confirm that route, airline and month remain unique
    # after airline-name standardisation.
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

    # Confirm that all scheduled-sector relationships
    # are valid after removing non-data rows.
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

    # Keep only actual airline-route observations.
    # Aggregate airline and network records are excluded.
    route_df = df.loc[
        df["is_route_level"]
    ].copy()

    # ---------------------------------------------------------
    # Validate route-level dataset
    # ---------------------------------------------------------

    # Confirm that aggregate airline records were removed.
    route_all_airlines = (
        route_df["is_all_airlines"]
    ).sum()

    # Confirm that network aggregate records were removed.
    route_all_ports = (
        route_df["is_all_ports"]
    ).sum()

    if route_all_airlines > 0:
        raise ValueError(
            "All Airlines records remain in the route-level dataset."
        )

    if route_all_ports > 0:
        raise ValueError(
            "All Ports records remain in the route-level dataset."
        )

    # Confirm that known aggregate classification anomalies
    # are not present in the route-level dataset.
    route_invalid_departures = (
        ~route_df["departure_classification_valid"]
    ).sum()

    route_invalid_arrivals = (
        ~route_df["arrival_classification_valid"]
    ).sum()

    if route_invalid_departures > 0:
        raise ValueError(
            "Invalid departure classifications remain "
            "in the route-level dataset."
        )

    if route_invalid_arrivals > 0:
        raise ValueError(
            "Invalid arrival classifications remain "
            "in the route-level dataset."
        )

    # ---------------------------------------------------------
    # Sort datasets
    # ---------------------------------------------------------

    # Sort the full historical dataset for
    # consistent downstream processing.
    df = df.sort_values(
        [
            "month",
            "route",
            "airline",
        ]
    ).reset_index(drop=True)

    # Sort the route-level dataset using the same keys.
    route_df = route_df.sort_values(
        [
            "month",
            "route",
            "airline",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Save processed datasets
    # ---------------------------------------------------------

    # Save the complete cleaned historical dataset,
    # including aggregate observations.
    df.to_csv(
        HISTORICAL_CLEAN_PATH,
        index=False,
    )

    # Save the historical route-level dataset
    # for later analysis and modelling.
    route_df.to_csv(
        HISTORICAL_ROUTE_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Processing summary
    # ---------------------------------------------------------

    print("\nHistorical cleaning complete")
    print("----------------------------")

    print(
        f"Workbook rows:              "
        f"{len(df) + non_data_count}"
    )

    print(
        f"Non-data rows removed:      "
        f"{non_data_count}"
    )

    print(
        f"Clean historical rows:      "
        f"{len(df)}"
    )

    print(
        f"Route-level rows:           "
        f"{len(route_df)}"
    )

    print(
        f"All Airlines rows:          "
        f"{df['is_all_airlines'].sum()}"
    )

    print(
        f"All Ports rows:             "
        f"{df['is_all_ports'].sum()}"
    )

    print(
        f"Zero-flight rows:           "
        f"{df['no_flown_sectors'].sum()}"
    )

    print(
        f"Missing percentage rows:    "
        f"{df['missing_on_time_percentage'].sum()}"
    )

    print(
        f"Invalid scheduled rows:     "
        f"{(~df['scheduled_sector_valid']).sum()}"
    )

    print(
        f"Invalid departure rows:     "
        f"{(~df['departure_classification_valid']).sum()}"
    )

    print(
        f"Invalid arrival rows:       "
        f"{(~df['arrival_classification_valid']).sum()}"
    )

    print(
        f"Route invalid departures:   "
        f"{(~route_df['departure_classification_valid']).sum()}"
    )

    print(
        f"Route invalid arrivals:     "
        f"{(~route_df['arrival_classification_valid']).sum()}"
    )

    print(
        f"Historical date range:      "
        f"{df['month'].min().date()} to "
        f"{df['month'].max().date()}"
    )

    print("\nOutput files:")
    print("-------------")
    print(HISTORICAL_CLEAN_PATH)
    print(HISTORICAL_ROUTE_PATH)


# Run the main function only when this script is executed directly.
if __name__ == "__main__":
    main()