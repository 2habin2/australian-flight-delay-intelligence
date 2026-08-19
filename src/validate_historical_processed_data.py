"""
Validate the processed historical BITRE datasets.

This script checks that the historical cleaning pipeline produced
consistent full and route-level datasets from 2010 to 2026.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the locations of the processed historical datasets.
HISTORICAL_CLEAN_PATH = Path(
    "data/processed/bitre_otp_historical_clean.csv"
)

HISTORICAL_ROUTE_PATH = Path(
    "data/processed/bitre_otp_historical_route_level.csv"
)

# Define expected data types when loading processed CSV files.
CSV_DTYPES = {
    "on_time_departures_pct": "float64",
    "on_time_arrivals_pct": "float64",
    "cancellations_pct": "float64",
    "source_sheet": "string",
}


def main() -> None:
    """Validate the processed historical BITRE datasets."""

    # ---------------------------------------------------------
    # File validation
    # ---------------------------------------------------------

    # Confirm that the full historical dataset exists.
    if not HISTORICAL_CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Historical clean dataset was not found at: "
            f"{HISTORICAL_CLEAN_PATH}"
        )

    # Confirm that the route-level historical dataset exists.
    if not HISTORICAL_ROUTE_PATH.exists():
        raise FileNotFoundError(
            f"Historical route dataset was not found at: "
            f"{HISTORICAL_ROUTE_PATH}"
        )

    # ---------------------------------------------------------
    # Load processed datasets
    # ---------------------------------------------------------

    # Load the complete cleaned historical dataset.
    clean_df = pd.read_csv(
        HISTORICAL_CLEAN_PATH,
        parse_dates=["month"],
        dtype=CSV_DTYPES,
        low_memory=False,
    )

    # Load the historical route-level dataset.
    route_df = pd.read_csv(
        HISTORICAL_ROUTE_PATH,
        parse_dates=["month"],
        dtype=CSV_DTYPES,
        low_memory=False,
    )

    print("Processed historical dataset shapes:")
    print("------------------------------------")
    print(f"Clean historical dataset: {clean_df.shape}")
    print(f"Route-level dataset:      {route_df.shape}")

    # ---------------------------------------------------------
    # Expected row counts
    # ---------------------------------------------------------

    # Check the row counts produced by the cleaning pipeline.
    clean_row_count_valid = (
        len(clean_df) == 94245
    )

    route_row_count_valid = (
        len(route_df) == 69188
    )

    print("\nRow-count validation:")
    print("---------------------")
    print(
        f"Clean historical rows correct: "
        f"{clean_row_count_valid}"
    )
    print(
        f"Route-level rows correct:      "
        f"{route_row_count_valid}"
    )

    # ---------------------------------------------------------
    # Date coverage
    # ---------------------------------------------------------

    earliest_month = route_df["month"].min()
    latest_month = route_df["month"].max()
    unique_months = route_df["month"].nunique()

    print("\nHistorical date coverage:")
    print("-------------------------")
    print(f"Earliest month: {earliest_month}")
    print(f"Latest month:   {latest_month}")
    print(f"Unique months:  {unique_months}")

    # Create the complete expected monthly range.
    expected_months = pd.date_range(
        start="2010-01-01",
        end="2026-06-01",
        freq="MS",
    )

    missing_months = expected_months.difference(
        route_df["month"].drop_duplicates()
    )

    print(
        f"Missing calendar months: "
        f"{len(missing_months)}"
    )

    # ---------------------------------------------------------
    # Airline naming validation
    # ---------------------------------------------------------

    # Confirm that the known lowercase capitalisation
    # inconsistency was removed.
    lowercase_virgin_count = (
        clean_df["airline"] == "virgin Australia"
    ).sum()

    print("\nAirline naming validation:")
    print("--------------------------")
    print(
        f"Lowercase Virgin Australia rows: "
        f"{lowercase_virgin_count}"
    )

    # ---------------------------------------------------------
    # Aggregate validation
    # ---------------------------------------------------------

    # Confirm that the route-level dataset does not contain
    # records representing all airlines.
    route_all_airlines = (
        route_df["airline"] == "All Airlines"
    ).sum()

    # Confirm that network-wide route aggregates are absent.
    route_all_ports = (
        route_df["route"] == "All Ports-All Ports"
    ).sum()

    print("\nRoute-level aggregate validation:")
    print("---------------------------------")
    print(f"All Airlines rows: {route_all_airlines}")
    print(f"All Ports rows:    {route_all_ports}")

    # ---------------------------------------------------------
    # Duplicate validation
    # ---------------------------------------------------------

    # Check whether route, airline and month uniquely
    # identify route-level observations.
    duplicate_keys = route_df.duplicated(
        subset=[
            "route",
            "airline",
            "month",
        ]
    ).sum()

    print("\nRoute-level duplicate validation:")
    print("---------------------------------")
    print(
        f"Duplicate business keys: "
        f"{duplicate_keys}"
    )

    # ---------------------------------------------------------
    # Scheduled-sector validation
    # ---------------------------------------------------------

    # Count invalid scheduled-sector relationships.
    invalid_scheduled = (
        ~route_df["scheduled_sector_valid"]
    ).sum()

    print("\nScheduled-sector validation:")
    print("----------------------------")
    print(
        f"Invalid scheduled rows: "
        f"{invalid_scheduled}"
    )

    # ---------------------------------------------------------
    # Flight-classification validation
    # ---------------------------------------------------------

    # Count invalid departure classifications.
    invalid_departures = (
        ~route_df["departure_classification_valid"]
    ).sum()

    # Count invalid arrival classifications.
    invalid_arrivals = (
        ~route_df["arrival_classification_valid"]
    ).sum()

    print("\nRoute-level classification validation:")
    print("--------------------------------------")
    print(
        f"Invalid departure classifications: "
        f"{invalid_departures}"
    )
    print(
        f"Invalid arrival classifications:   "
        f"{invalid_arrivals}"
    )

    # ---------------------------------------------------------
    # Missing percentage validation
    # ---------------------------------------------------------

    # Identify observations containing missing
    # on-time performance percentages.
    missing_percentage_rows = route_df[
        route_df["on_time_departures_pct"].isna()
        | route_df["on_time_arrivals_pct"].isna()
    ]

    # Confirm that missing percentages only occur when
    # no flights were operated.
    missing_percentage_valid = (
        missing_percentage_rows["sectors_flown"] == 0
    ).all()

    print("\nMissing percentage validation:")
    print("------------------------------")
    print(
        f"Rows with missing percentages: "
        f"{len(missing_percentage_rows)}"
    )
    print(
        f"All associated with zero flown sectors: "
        f"{missing_percentage_valid}"
    )

    # ---------------------------------------------------------
    # Full historical anomaly validation
    # ---------------------------------------------------------

    # Confirm that the known source-level classification
    # anomalies remain preserved in the complete dataset.
    full_invalid_departures = (
        ~clean_df["departure_classification_valid"]
    ).sum()

    full_invalid_arrivals = (
        ~clean_df["arrival_classification_valid"]
    ).sum()

    print("\nFull historical anomaly validation:")
    print("-----------------------------------")
    print(
        f"Invalid departure classifications: "
        f"{full_invalid_departures}"
    )
    print(
        f"Invalid arrival classifications:   "
        f"{full_invalid_arrivals}"
    )

    # ---------------------------------------------------------
    # Final validation result
    # ---------------------------------------------------------

    checks_passed = (
        clean_row_count_valid
        and route_row_count_valid
        and earliest_month == pd.Timestamp("2010-01-01")
        and latest_month == pd.Timestamp("2026-06-01")
        and len(missing_months) == 0
        and lowercase_virgin_count == 0
        and route_all_airlines == 0
        and route_all_ports == 0
        and duplicate_keys == 0
        and invalid_scheduled == 0
        and invalid_departures == 0
        and invalid_arrivals == 0
        and missing_percentage_valid
        and full_invalid_departures == 2
        and full_invalid_arrivals == 2
    )

    print("\nFinal validation:")
    print("-----------------")

    if checks_passed:
        print(
            "All historical processed-data "
            "validation checks passed."
        )
    else:
        print(
            "One or more historical processed-data "
            "validation checks failed."
        )


# Run the main function only when this script is executed directly.
if __name__ == "__main__":
    main()