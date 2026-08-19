"""
Validate the cleaned BITRE datasets.

This script checks that the cleaning pipeline produced
consistent full and route-level datasets.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the locations of the processed datasets.
CLEAN_DATA_PATH = Path(
    "data/processed/bitre_otp_clean.csv"
)

ROUTE_DATA_PATH = Path(
    "data/processed/bitre_otp_route_level.csv"
)


def main() -> None:
    """Validate the processed BITRE datasets."""

    # ---------------------------------------------------------
    # File validation
    # ---------------------------------------------------------

    # Confirm that both processed datasets exist.
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset was not found at: {CLEAN_DATA_PATH}"
        )

    if not ROUTE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Route-level dataset was not found at: {ROUTE_DATA_PATH}"
        )

    # ---------------------------------------------------------
    # Load processed data
    # ---------------------------------------------------------

    # Load both processed CSV files.
    clean_df = pd.read_csv(
        CLEAN_DATA_PATH,
        parse_dates=["month"]
    )

    route_df = pd.read_csv(
        ROUTE_DATA_PATH,
        parse_dates=["month"]
    )

    print("Processed dataset shapes:")
    print("-------------------------")
    print(f"Clean dataset:       {clean_df.shape}")
    print(f"Route-level dataset: {route_df.shape}")

    # ---------------------------------------------------------
    # Airline naming validation
    # ---------------------------------------------------------

    # Confirm that the inconsistent lowercase airline name
    # was removed during cleaning.
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
    # Aggregate-row validation
    # ---------------------------------------------------------

    # Confirm that route-level data excludes aggregate airlines.
    route_all_airlines = (
        route_df["airline"] == "All Airlines"
    ).sum()

    # Confirm that route-level data excludes network totals.
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

    # Check the expected business key in the route-level dataset.
    duplicate_keys = route_df.duplicated(
        subset=[
            "route",
            "airline",
            "month",
        ]
    ).sum()

    print("\nRoute-level duplicate validation:")
    print("---------------------------------")
    print(f"Duplicate business keys: {duplicate_keys}")

    # ---------------------------------------------------------
    # Missing-value validation
    # ---------------------------------------------------------

    # Identify route-level observations with missing
    # departure or arrival on-time percentages.
    missing_percentage_rows = route_df[
        route_df["on_time_departures_pct"].isna()
        | route_df["on_time_arrivals_pct"].isna()
    ]

    # Check whether every missing percentage occurs when
    # zero sectors were flown.
    missing_values_valid = (
        missing_percentage_rows["sectors_flown"] == 0
    ).all()

    print("\nMissing percentage validation:")
    print("------------------------------")
    print(
        f"Rows with missing percentages: "
        f"{len(missing_percentage_rows)}"
    )
    print(
        f"All caused by zero flown sectors: "
        f"{missing_values_valid}"
    )

    # ---------------------------------------------------------
    # Classification validation
    # ---------------------------------------------------------

    # Count invalid departure classifications remaining
    # in the route-level dataset.
    invalid_departures = (
        ~route_df["departure_classification_valid"]
    ).sum()

    # Count invalid arrival classifications remaining
    # in the route-level dataset.
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
    # Date-range validation
    # ---------------------------------------------------------

    print("\nRoute-level date range:")
    print("-----------------------")
    print(f"Earliest month: {route_df['month'].min()}")
    print(f"Latest month:   {route_df['month'].max()}")

    # ---------------------------------------------------------
    # Final validation result
    # ---------------------------------------------------------

    checks_passed = (
        lowercase_virgin_count == 0
        and route_all_airlines == 0
        and route_all_ports == 0
        and duplicate_keys == 0
        and missing_values_valid
        and invalid_departures == 0
        and invalid_arrivals == 0
    )

    print("\nFinal validation:")
    print("-----------------")

    if checks_passed:
        print("All processed-data validation checks passed.")
    else:
        print("One or more processed-data checks failed.")


# Run the main function only when this script is executed directly.
if __name__ == "__main__":
    main()