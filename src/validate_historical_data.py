"""
Validate the complete BITRE historical on-time performance dataset.

This script combines the worksheets from 2010 to 2026 in memory,
identifies non-data worksheet rows, and checks data quality before
historical cleaning is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the location of the original BITRE workbook.
DATA_PATH = Path(
    "data/raw/OTP_Time_Series_Master_Current_5.xlsx"
)

# Define the worksheets containing historical observations.
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


def load_historical_data() -> pd.DataFrame:
    """Load and combine all historical BITRE worksheets."""

    # Store each worksheet before concatenation.
    dataframes = []

    # Load each historical worksheet.
    for sheet_name in SHEET_NAMES:
        sheet_df = pd.read_excel(
            DATA_PATH,
            sheet_name=sheet_name,
        )

        # Record the source worksheet for data lineage.
        sheet_df["Source Sheet"] = sheet_name

        dataframes.append(sheet_df)

    # Combine all worksheets into one DataFrame.
    historical_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    return historical_df


def main() -> None:
    """Run validation checks on the complete historical dataset."""

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    # Confirm that the original workbook exists.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"BITRE dataset was not found at: {DATA_PATH}"
        )

    # ---------------------------------------------------------
    # Load historical data
    # ---------------------------------------------------------

    df = load_historical_data()

    print("Historical workbook shape:")
    print("--------------------------")
    print(df.shape)

    # ---------------------------------------------------------
    # Standardise date values
    # ---------------------------------------------------------

    # Convert valid Month values to pandas datetime values.
    # Invalid values, including worksheet notes, become NaT.
    df["Month"] = pd.to_datetime(
        df["Month"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Identify non-data rows
    # ---------------------------------------------------------

    # Identify worksheet rows that do not represent
    # airline performance observations.
    non_data_rows = (
        df["Route"].isna()
        | df["Airline"].isna()
        | df["Month"].isna()
    )

    print("\nNon-data rows:")
    print("--------------")
    print(
        f"Total non-data rows: "
        f"{non_data_rows.sum()}"
    )

    # Display rows identified as worksheet notes or blanks.
    if non_data_rows.any():
        print("\nNon-data row details:")

        print(
            df.loc[
                non_data_rows,
                [
                    "Route",
                    "Airline",
                    "Month",
                    "Source Sheet",
                ],
            ].to_string(index=False)
        )

    # Keep only actual flight-performance observations
    # for the remaining validation checks.
    data_df = df.loc[
        ~non_data_rows
    ].copy()

    print("\nHistorical observation shape:")
    print("-----------------------------")
    print(data_df.shape)

    # ---------------------------------------------------------
    # Date coverage
    # ---------------------------------------------------------

    earliest_month = data_df["Month"].min()
    latest_month = data_df["Month"].max()
    unique_months = data_df["Month"].nunique()

    print("\nHistorical date range:")
    print("----------------------")
    print(f"Earliest month: {earliest_month}")
    print(f"Latest month:   {latest_month}")
    print(f"Unique months:  {unique_months}")

    # ---------------------------------------------------------
    # Calendar continuity
    # ---------------------------------------------------------

    # Create the complete expected monthly range.
    expected_months = pd.date_range(
        start=earliest_month,
        end=latest_month,
        freq="MS",
    )

    # Identify calendar months that are completely absent.
    missing_months = expected_months.difference(
        data_df["Month"].drop_duplicates()
    )

    print("\nCalendar continuity:")
    print("--------------------")
    print(
        f"Missing calendar months: "
        f"{len(missing_months)}"
    )

    if len(missing_months) > 0:
        print(missing_months)

    # ---------------------------------------------------------
    # Exact duplicate validation
    # ---------------------------------------------------------

    # Exclude the source worksheet column when checking
    # whether complete data observations are duplicated.
    source_columns = [
        column
        for column in data_df.columns
        if column != "Source Sheet"
    ]

    exact_duplicates = data_df.duplicated(
        subset=source_columns,
        keep=False,
    )

    print("\nExact duplicate validation:")
    print("---------------------------")
    print(
        f"Rows involved in exact duplicates: "
        f"{exact_duplicates.sum()}"
    )

    # ---------------------------------------------------------
    # Business-key validation
    # ---------------------------------------------------------

    # Check whether route, airline and month uniquely
    # identify each historical observation.
    key_columns = [
        "Route",
        "Airline",
        "Month",
    ]

    duplicate_keys = data_df.duplicated(
        subset=key_columns,
        keep=False,
    )

    print("\nHistorical business-key validation:")
    print("-----------------------------------")
    print(
        f"Rows involved in duplicate keys: "
        f"{duplicate_keys.sum()}"
    )

    # Display duplicate business keys if any are found.
    if duplicate_keys.any():
        print("\nDuplicate business-key details:")

        print(
            data_df.loc[
                duplicate_keys,
                [
                    "Route",
                    "Airline",
                    "Month",
                    "Source Sheet",
                ],
            ]
            .sort_values(
                [
                    "Month",
                    "Route",
                    "Airline",
                ]
            )
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # Scheduled-sector validation
    # ---------------------------------------------------------

    # Check that scheduled sectors equal flown sectors
    # plus cancellations while allowing for floating-point
    # precision in historical records.
    scheduled_valid = np.isclose(
        data_df["Sectors Scheduled"],
        (
            data_df["Sectors Flown"]
            + data_df["Cancellations"]
        ),
        equal_nan=False,
    )

    print("\nScheduled-sector validation:")
    print("----------------------------")
    print(
        f"Invalid rows: "
        f"{(~scheduled_valid).sum()}"
    )

    # ---------------------------------------------------------
    # Departure validation
    # ---------------------------------------------------------

    # Check that flown sectors equal the sum of
    # on-time and delayed departures.
    departure_valid = np.isclose(
        data_df["Sectors Flown"],
        (
            data_df["Departures On Time"]
            + data_df["Departures Delayed"]
        ),
        equal_nan=False,
    )

    print("\nDeparture validation:")
    print("---------------------")
    print(
        f"Invalid rows: "
        f"{(~departure_valid).sum()}"
    )

    # ---------------------------------------------------------
    # Arrival validation
    # ---------------------------------------------------------

    # Check that flown sectors equal the sum of
    # on-time and delayed arrivals.
    arrival_valid = np.isclose(
        data_df["Sectors Flown"],
        (
            data_df["Arrivals On Time"]
            + data_df["Arrivals Delayed"]
        ),
        equal_nan=False,
    )

    print("\nArrival validation:")
    print("-------------------")
    print(
        f"Invalid rows: "
        f"{(~arrival_valid).sum()}"
    )

    # ---------------------------------------------------------
    # Invalid classification records
    # ---------------------------------------------------------

    # Identify observations that fail either the departure
    # or arrival classification relationship.
    invalid_classification_rows = data_df[
        (~departure_valid)
        | (~arrival_valid)
    ]

    print("\nInvalid flight-classification records:")
    print("--------------------------------------")
    print(
        f"Total invalid records: "
        f"{len(invalid_classification_rows)}"
    )

    if not invalid_classification_rows.empty:
        print("\nInvalid record details:")

        print(
            invalid_classification_rows[
                [
                    "Route",
                    "Airline",
                    "Month",
                    "Sectors Scheduled",
                    "Sectors Flown",
                    "Cancellations",
                    "Departures On Time",
                    "Departures Delayed",
                    "Arrivals On Time",
                    "Arrivals Delayed",
                    "Source Sheet",
                ]
            ].to_string(index=False)
        )

    # ---------------------------------------------------------
    # Route validation
    # ---------------------------------------------------------

    # Reconstruct the expected directional route using
    # the departure and arrival airport names.
    expected_route = (
        data_df["Departing Port"]
        + "-"
        + data_df["Arriving Port"]
    )

    route_valid = (
        data_df["Route"] == expected_route
    )

    print("\nRoute validation:")
    print("-----------------")
    print(
        f"Route mismatches: "
        f"{(~route_valid).sum()}"
    )

    # Display route mismatches if any exist.
    if (~route_valid).any():
        print("\nRoute mismatch details:")

        print(
            data_df.loc[
                ~route_valid,
                [
                    "Route",
                    "Departing Port",
                    "Arriving Port",
                    "Airline",
                    "Month",
                    "Source Sheet",
                ],
            ].to_string(index=False)
        )

    # ---------------------------------------------------------
    # Missing percentage validation
    # ---------------------------------------------------------

    # Convert known text-based missing-value markers
    # in percentage columns to proper missing values.
    percentage_columns = [
        "OnTime Departures \n(%)",
        "OnTime Arrivals \n(%)",
        "Cancellations \n\n(%)",
    ]

    missing_markers = [
        "na",
        "NA",
        "n/a",
        "N/A",
        "",
    ]

    for column in percentage_columns:
        data_df[column] = data_df[column].replace(
            missing_markers,
            pd.NA,
        )

        # Convert remaining percentage values to numeric.
        data_df[column] = pd.to_numeric(
            data_df[column],
            errors="raise",
        )

    # Identify observations where either on-time
    # performance percentage is missing.
    missing_percentage_rows = data_df[
        data_df["OnTime Departures \n(%)"].isna()
        | data_df["OnTime Arrivals \n(%)"].isna()
    ]

    print("\nMissing percentage validation:")
    print("------------------------------")
    print(
        f"Rows with missing percentages: "
        f"{len(missing_percentage_rows)}"
    )

    # Check whether every missing percentage occurs
    # when no sectors were flown.
    missing_due_to_zero_flown = (
        missing_percentage_rows["Sectors Flown"] == 0
    ).all()

    print(
        f"All caused by zero flown sectors: "
        f"{missing_due_to_zero_flown}"
    )

    # Identify missing percentages associated with
    # observations where flights were actually operated.
    unexpected_missing = missing_percentage_rows[
        missing_percentage_rows["Sectors Flown"] > 0
    ]

    print(
        f"Missing percentages with flights flown: "
        f"{len(unexpected_missing)}"
    )

    if not unexpected_missing.empty:
        print("\nUnexpected missing percentage details:")

        print(
            unexpected_missing[
                [
                    "Route",
                    "Airline",
                    "Month",
                    "Sectors Flown",
                    "OnTime Departures \n(%)",
                    "OnTime Arrivals \n(%)",
                    "Source Sheet",
                ]
            ].to_string(index=False)
        )

    # ---------------------------------------------------------
    # Aggregate observations
    # ---------------------------------------------------------

    # Count observations containing results across all airlines.
    all_airlines_count = (
        data_df["Airline"] == "All Airlines"
    ).sum()

    # Count network-wide airport aggregate observations.
    all_ports_count = (
        data_df["Route"] == "All Ports-All Ports"
    ).sum()

    print("\nAggregate observations:")
    print("-----------------------")
    print(
        f"All Airlines rows: "
        f"{all_airlines_count}"
    )
    print(
        f"All Ports rows:    "
        f"{all_ports_count}"
    )

    # ---------------------------------------------------------
    # Airline naming consistency
    # ---------------------------------------------------------

    # Summarise airline names across the complete history
    # to identify spelling, naming and capitalisation changes.
    airline_summary = (
        data_df.groupby("Airline")
        .agg(
            row_count=("Airline", "size"),
            earliest_month=("Month", "min"),
            latest_month=("Month", "max"),
        )
        .sort_index()
    )

    print("\nHistorical airline names:")
    print("-------------------------")
    print(airline_summary.to_string())

    # ---------------------------------------------------------
    # Virgin Australia naming check
    # ---------------------------------------------------------

    # Identify records containing the Virgin Australia
    # capitalisation variants found in the source workbook.
    virgin_rows = data_df[
        data_df["Airline"].str.lower()
        == "virgin australia"
    ]

    print("\nVirgin Australia naming variants:")
    print("---------------------------------")

    print(
        virgin_rows.groupby("Airline")
        .agg(
            row_count=("Airline", "size"),
            earliest_month=("Month", "min"),
            latest_month=("Month", "max"),
        )
        .to_string()
    )

    # ---------------------------------------------------------
    # Worksheet row counts
    # ---------------------------------------------------------

    # Summarise valid observations contributed
    # by each source worksheet.
    sheet_summary = (
        data_df.groupby("Source Sheet")
        .size()
        .rename("row_count")
    )

    print("\nValid observations by source worksheet:")
    print("---------------------------------------")
    print(sheet_summary.to_string())

    # ---------------------------------------------------------
    # Validation summary
    # ---------------------------------------------------------

    print("\nHistorical validation summary:")
    print("------------------------------")
    print(f"Workbook rows:       {len(df)}")
    print(f"Non-data rows:       {non_data_rows.sum()}")
    print(f"Valid observations:  {len(data_df)}")
    print(
        f"Exact duplicate rows: "
        f"{exact_duplicates.sum()}"
    )
    print(
        f"Duplicate key rows:   "
        f"{duplicate_keys.sum()}"
    )
    print(
        f"Invalid scheduled rows: "
        f"{(~scheduled_valid).sum()}"
    )
    print(
        f"Invalid departure rows: "
        f"{(~departure_valid).sum()}"
    )
    print(
        f"Invalid arrival rows:   "
        f"{(~arrival_valid).sum()}"
    )
    print(
        f"Route mismatches:       "
        f"{(~route_valid).sum()}"
    )
    print(
        f"Missing percentage rows: "
        f"{len(missing_percentage_rows)}"
    )


# Run the main function only when this script is executed directly.
if __name__ == "__main__":
    main()