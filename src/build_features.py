"""
Build modelling features for Australian flight delay forecasting.

This script transforms the cleaned historical BITRE route-level dataset
into a modelling-ready dataset using only information available before
each target month.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the cleaned historical route-level input dataset.
INPUT_PATH = Path(
    "data/processed/bitre_otp_historical_route_level.csv"
)

# Define the output path for the modelling-ready dataset.
OUTPUT_PATH = Path(
    "data/processed/bitre_otp_model_features.csv"
)

# Define the minimum number of flown sectors required
# for a stable monthly modelling target.
MIN_SECTORS_FLOWN = 25


def load_data() -> pd.DataFrame:
    """Load the cleaned historical route-level BITRE dataset."""

    # Confirm that the processed input dataset exists.
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Historical route-level dataset was not found at: "
            f"{INPUT_PATH}"
        )

    # Load the dataset and parse the month column as dates.
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["month"],
        dtype={
            "source_sheet": "string",
        },
        low_memory=False,
    )

    return df

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the departure delay target and calendar features."""

    # Keep observations where at least one flight operated.
    model_df = df[
        df["sectors_flown"] > 0
    ].copy()

    # Calculate the monthly route-airline departure delay rate.
    model_df["departure_delay_pct"] = (
        model_df["departures_delayed"]
        / model_df["sectors_flown"]
        * 100
    )

    # Identify observations meeting the selected
    # minimum flight-volume threshold.
    model_df["target_volume_valid"] = (
        model_df["sectors_flown"]
        >= MIN_SECTORS_FLOWN
    )

    # Sort observations chronologically within
    # each airline and directional route.
    model_df = model_df.sort_values(
        [
            "airline",
            "route",
            "month",
        ]
    ).reset_index(drop=True)

    # Create calendar features available before
    # the target month is completed.
    model_df["year"] = model_df["month"].dt.year
    model_df["month_of_year"] = (
        model_df["month"].dt.month
    )

    return model_df

def add_route_airline_lag_features(
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add historical route-airline performance features."""

    # Keep the original historical columns required
    # for constructing exact calendar-month lags.
    lag_source = model_df[
        [
            "airline",
            "route",
            "month",
            "departure_delay_pct",
            "cancellations_pct",
            "departures_delayed",
            "sectors_flown",
        ]
    ].copy()

    # Create exact one-, two-, and three-month lag features.
    for lag in [1, 2, 3]:

        lag_df = lag_source.copy()

        # Shift each historical observation forward so that
        # it aligns with the future month using that history.
        lag_df["month"] = (
            lag_df["month"]
            + pd.DateOffset(months=lag)
        )

        # Rename columns to identify their historical period.
        lag_df = lag_df.rename(
            columns={
                "departure_delay_pct": f"delay_pct_lag_{lag}",
                "cancellations_pct": f"cancellation_pct_lag_{lag}",
                "departures_delayed": f"departures_delayed_lag_{lag}",
                "sectors_flown": f"sectors_flown_lag_{lag}",
            }
        )

        # Attach the historical values to the target month.
        model_df = model_df.merge(
            lag_df,
            on=[
                "airline",
                "route",
                "month",
            ],
            how="left",
        )

    # Calculate delayed departures across the previous
    # three available calendar months.
    model_df["departures_delayed_previous_3m"] = (
        model_df[
            [
                "departures_delayed_lag_1",
                "departures_delayed_lag_2",
                "departures_delayed_lag_3",
            ]
        ]
        .sum(
            axis=1,
            min_count=1,
        )
    )

    # Calculate flown sectors across the previous
    # three available calendar months.
    model_df["sectors_flown_previous_3m"] = (
        model_df[
            [
                "sectors_flown_lag_1",
                "sectors_flown_lag_2",
                "sectors_flown_lag_3",
            ]
        ]
        .sum(
            axis=1,
            min_count=1,
        )
    )

    # Calculate a flight-volume-weighted recent delay rate.
    model_df["delay_pct_previous_3m"] = (
        model_df["departures_delayed_previous_3m"]
        / model_df["sectors_flown_previous_3m"]
        * 100
    )

    return model_df

def add_context_lag_features(
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add previous-month airline-level and route-level performance."""

    # ---------------------------------------------------------
    # Airline-level previous-month performance
    # ---------------------------------------------------------

    # Aggregate monthly performance across all routes
    # operated by each airline.
    airline_monthly = (
        model_df
        .groupby(
            [
                "airline",
                "month",
            ],
            as_index=False,
        )
        .agg(
            airline_departures_delayed=(
                "departures_delayed",
                "sum",
            ),
            airline_sectors_flown=(
                "sectors_flown",
                "sum",
            ),
        )
    )

    # Calculate the airline-wide monthly departure delay rate.
    airline_monthly["airline_delay_pct"] = (
        airline_monthly["airline_departures_delayed"]
        / airline_monthly["airline_sectors_flown"]
        * 100
    )

    # Shift airline performance forward by one month so that
    # only historical information is attached to the target month.
    airline_monthly["month"] = (
        airline_monthly["month"]
        + pd.DateOffset(months=1)
    )

    # Rename the feature to identify its historical timing.
    airline_monthly = airline_monthly.rename(
        columns={
            "airline_delay_pct": "airline_delay_pct_lag_1",
        }
    )

    # Attach previous-month airline performance.
    model_df = model_df.merge(
        airline_monthly[
            [
                "airline",
                "month",
                "airline_delay_pct_lag_1",
            ]
        ],
        on=[
            "airline",
            "month",
        ],
        how="left",
    )

    # ---------------------------------------------------------
    # Route-level previous-month performance
    # ---------------------------------------------------------

    # Aggregate monthly performance across all airlines
    # operating on each directional route.
    route_monthly = (
        model_df
        .groupby(
            [
                "route",
                "month",
            ],
            as_index=False,
        )
        .agg(
            route_departures_delayed=(
                "departures_delayed",
                "sum",
            ),
            route_sectors_flown=(
                "sectors_flown",
                "sum",
            ),
        )
    )

    # Calculate the directional route's monthly delay rate.
    route_monthly["route_delay_pct"] = (
        route_monthly["route_departures_delayed"]
        / route_monthly["route_sectors_flown"]
        * 100
    )

    # Shift route performance forward by one month so that
    # only

    # Shift route performance forward by one month so that
    # only historical information is attached to the target month.
    route_monthly["month"] = (
        route_monthly["month"]
        + pd.DateOffset(months=1)
    )

    # Rename the feature to identify its historical timing.
    route_monthly = route_monthly.rename(
        columns={
            "route_delay_pct": "route_delay_pct_lag_1",
        }
    )

    # Attach previous-month route performance.
    model_df = model_df.merge(
        route_monthly[
            [
                "route",
                "month",
                "route_delay_pct_lag_1",
            ]
        ],
        on=[
            "route",
            "month",
        ],
        how="left",
    )

    return model_df

def create_final_model_dataset(
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create the final modelling-ready dataset."""

    # Define historical features required by the initial model.
    required_history_features = [
        "delay_pct_lag_1",
        "delay_pct_previous_3m",
        "cancellation_pct_lag_1",
        "sectors_flown_lag_1",
        "airline_delay_pct_lag_1",
        "route_delay_pct_lag_1",
    ]

    # Keep observations meeting the minimum target-volume rule.
    final_model_df = model_df[
        model_df["target_volume_valid"]
    ].copy()

    # Remove observations without the historical information
    # required by the initial forecasting model.
    final_model_df = final_model_df.dropna(
        subset=required_history_features
    ).copy()

    # Sort the final dataset chronologically.
    final_model_df = final_model_df.sort_values(
        "month"
    ).reset_index(drop=True)

    return final_model_df

def main() -> None:
    """Build and save the modelling-ready feature dataset."""

    # ---------------------------------------------------------
    # Load cleaned historical data
    # ---------------------------------------------------------

    df = load_data()

    print("Feature engineering pipeline")
    print("----------------------------")
    print(
        f"Input route-level rows: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------
    # Create target and calendar features
    # ---------------------------------------------------------

    model_df = create_target(
        df
    )

    print(
        f"Rows with operated flights: "
        f"{len(model_df):,}"
    )

    print(
        f"Rows meeting {MIN_SECTORS_FLOWN}-flight threshold: "
        f"{model_df['target_volume_valid'].sum():,}"
    )

    # ---------------------------------------------------------
    # Create historical features
    # ---------------------------------------------------------

    model_df = add_route_airline_lag_features(
        model_df
    )

    model_df = add_context_lag_features(
        model_df
    )

    # ---------------------------------------------------------
    # Create final modelling dataset
    # ---------------------------------------------------------

    final_model_df = create_final_model_dataset(
        model_df
    )

    # Confirm that the final dataset has no missing
    # values in the features required by the initial model.
    required_features = [
        "delay_pct_lag_1",
        "delay_pct_previous_3m",
        "cancellation_pct_lag_1",
        "sectors_flown_lag_1",
        "airline_delay_pct_lag_1",
        "route_delay_pct_lag_1",
    ]

    remaining_missing = (
        final_model_df[
            required_features
        ]
        .isna()
        .sum()
        .sum()
    )

    if remaining_missing != 0:
        raise ValueError(
            "Final modelling dataset contains missing "
            "required historical features."
        )

    # ---------------------------------------------------------
    # Save processed feature dataset
    # ---------------------------------------------------------

    # Ensure the processed-data directory exists.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the modelling-ready dataset.
    final_model_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nFeature engineering complete")
    print("----------------------------")
    print(
        f"Final modelling rows: "
        f"{len(final_model_df):,}"
    )
    print(
        f"Remaining required-feature missing values: "
        f"{remaining_missing}"
    )
    print(
        f"Date range: "
        f"{final_model_df['month'].min().date()} to "
        f"{final_model_df['month'].max().date()}"
    )

    print("\nOutput file:")
    print("------------")
    print(OUTPUT_PATH)


# Run the feature engineering pipeline only when
# this script is executed directly.
if __name__ == "__main__":
    main()