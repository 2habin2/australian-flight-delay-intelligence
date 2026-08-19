"""
Validate the BITRE airline on-time performance dataset.

This script checks important data-quality rules before
cleaning or transforming the dataset.
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
    """Run data-quality checks on the BITRE dataset."""

    # ---------------------------------------------------------
    # File validation
    # ---------------------------------------------------------

    # Confirm that the raw dataset exists before loading it.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"BITRE dataset was not found at: {DATA_PATH}"
        )

    # Load the selected worksheet into a pandas DataFrame.
    df = pd.read_excel(
        DATA_PATH,
        sheet_name=SHEET_NAME
    )

    # ---------------------------------------------------------
    # Missing percentage values
    # ---------------------------------------------------------

    # Identify rows where either on-time percentage is missing.
    missing_percentage_rows = df[
        df["OnTime Departures \n(%)"].isna()
        | df["OnTime Arrivals \n(%)"].isna()
    ]

    print("Rows with missing on-time percentages:")
    print("--------------------------------------")

    print(
        missing_percentage_rows[
            [
                "Route",
                "Airline",
                "Month",
                "Sectors Scheduled",
                "Sectors Flown",
                "Cancellations",
                "OnTime Departures \n(%)",
                "OnTime Arrivals \n(%)",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nTotal rows with missing percentages: "
        f"{len(missing_percentage_rows)}"
    )

    # ---------------------------------------------------------
    # Scheduled-sector validation
    # ---------------------------------------------------------

    # Check that every scheduled sector was either
    # flown or cancelled.
    scheduled_valid = (
        df["Sectors Scheduled"]
        == df["Sectors Flown"] + df["Cancellations"]
    )

    print("\nScheduled-sector validation:")
    print("----------------------------")
    print(f"Invalid rows: {(~scheduled_valid).sum()}")

    # ---------------------------------------------------------
    # Departure validation
    # ---------------------------------------------------------

    # Check that every flown sector is classified as either
    # an on-time departure or a delayed departure.
    departure_valid = (
        df["Sectors Flown"]
        == (
            df["Departures On Time"]
            + df["Departures Delayed"]
        )
    )

    print("\nDeparture validation:")
    print("---------------------")
    print(f"Invalid rows: {(~departure_valid).sum()}")

    # ---------------------------------------------------------
    # Arrival validation
    # ---------------------------------------------------------

    # Check that every flown sector is classified as either
    # an on-time arrival or a delayed arrival.
    arrival_valid = (
        df["Sectors Flown"]
        == (
            df["Arrivals On Time"]
            + df["Arrivals Delayed"]
        )
    )

    print("\nArrival validation:")
    print("-------------------")
    print(f"Invalid rows: {(~arrival_valid).sum()}")

    # ---------------------------------------------------------
    # Invalid flight-classification records
    # ---------------------------------------------------------

    # Identify observations that fail either the departure
    # or arrival classification rule.
    invalid_classification_rows = df[
        (~departure_valid) | (~arrival_valid)
    ]

    print("\nInvalid flight-classification records:")
    print("--------------------------------------")

    print(
        invalid_classification_rows[
            [
                "Route",
                "Airline",
                "Month",
                "Sectors Flown",
                "Departures On Time",
                "Departures Delayed",
                "Arrivals On Time",
                "Arrivals Delayed",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Route validation
    # ---------------------------------------------------------

    # Reconstruct the expected route from the departure
    # and arrival airport names.
    expected_route = (
        df["Departing Port"]
        + "-"
        + df["Arriving Port"]
    )

    # Compare the reconstructed route with the source route.
    route_valid = df["Route"] == expected_route

    print("\nRoute validation:")
    print("-----------------")
    print(f"Route mismatches: {(~route_valid).sum()}")

    # ---------------------------------------------------------
    # Business-key validation
    # ---------------------------------------------------------

    # Define the columns expected to uniquely identify
    # each monthly airline-route observation.
    key_columns = [
        "Route",
        "Airline",
        "Month",
    ]

    # Identify rows sharing the same business key.
    duplicate_keys = df.duplicated(
        subset=key_columns,
        keep=False
    )

    print("\nDuplicate route-airline-month combinations:")
    print("-------------------------------------------")
    print(f"Rows involved: {duplicate_keys.sum()}")

    # ---------------------------------------------------------
    # Aggregate airline rows
    # ---------------------------------------------------------

    # Count rows containing aggregated results across all airlines.
    all_airlines_count = (
        df["Airline"] == "All Airlines"
    ).sum()

    print("\nAll Airlines aggregate rows:")
    print("----------------------------")
    print(all_airlines_count)

    # ---------------------------------------------------------
    # Airline naming consistency
    # ---------------------------------------------------------

    # Display unique airline names to identify spelling
    # or capitalisation inconsistencies.
    print("\nAirline names:")
    print("--------------")

    for airline in sorted(df["Airline"].dropna().unique()):
        print(airline)

    # ---------------------------------------------------------
    # Virgin Australia naming check
    # ---------------------------------------------------------

    # Identify records containing different capitalisations
    # of the Virgin Australia airline name.
    virgin_rows = df[
        df["Airline"].str.lower() == "virgin australia"
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
    )


# Run the main function only when this script is executed directly.
if __name__ == "__main__":
    main()