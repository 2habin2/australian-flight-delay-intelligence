"""
Generate departure-delay predictions using the trained flight delay model.

This module loads the modelling dataset and trained model artifact
required to produce future monthly route-airline predictions.
"""

from pathlib import Path

import joblib
import pandas as pd


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the cleaned historical route-level dataset.
HISTORICAL_DATA_PATH = Path(
    "data/processed/bitre_otp_historical_route_level.csv"
)

# Define the trained model artifact.
MODEL_PATH = Path(
    "models/flight_delay_model.joblib"
)


# ---------------------------------------------------------
# Model feature configuration
# ---------------------------------------------------------

# Define categorical predictors expected by the trained model.
CATEGORICAL_FEATURES = [
    "airline",
    "route",
]

# Define numerical predictors expected by the trained model.
NUMERIC_FEATURES = [
    "year",
    "month_of_year",
    "delay_pct_lag_1",
    "delay_pct_previous_3m",
    "cancellation_pct_lag_1",
    "sectors_flown_lag_1",
    "airline_delay_pct_lag_1",
    "route_delay_pct_lag_1",
]

# Combine all predictors in the same order used during training.
MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the cleaned historical route-level dataset."""

    # Confirm that the historical dataset exists.
    if not HISTORICAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Historical dataset was not found at: "
            f"{HISTORICAL_DATA_PATH}"
        )

    # Load the complete cleaned route-level history.
    feature_df = pd.read_csv(
        HISTORICAL_DATA_PATH,
        parse_dates=["month"],
        low_memory=False,
    )

    # Define the historical columns required to construct
    # the future model features.
    required_columns = [
        "airline",
        "route",
        "month",
        "sectors_scheduled",
        "sectors_flown",
        "cancellations",
        "departures_delayed",
    ]

    # Identify any required historical columns that are missing.
    missing_columns = [
        column
        for column in required_columns
        if column not in feature_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Historical dataset is missing required columns: "
            f"{missing_columns}"
        )

    return feature_df


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

def load_model():
    """Load the trained flight delay prediction pipeline."""

    # Confirm that the trained model artifact exists.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model was not found at: "
            f"{MODEL_PATH}"
        )

    # Load the fitted preprocessing and model pipeline.
    model_pipeline = joblib.load(
        MODEL_PATH
    )

    return model_pipeline

# ---------------------------------------------------------
# Prediction input validation
# ---------------------------------------------------------

def validate_prediction_request(
    feature_df: pd.DataFrame,
    airline: str,
    route: str,
    forecast_month: pd.Timestamp,
) -> None:
    """Validate the airline, route and forecast month."""

    # Confirm that the requested airline exists in the
    # historical modelling dataset.
    available_airlines = set(
        feature_df["airline"].dropna().unique()
    )

    if airline not in available_airlines:
        raise ValueError(
            f"Airline was not found in historical data: "
            f"{airline}"
        )

    # Confirm that the requested directional route exists.
    available_routes = set(
        feature_df["route"].dropna().unique()
    )

    if route not in available_routes:
        raise ValueError(
            f"Route was not found in historical data: "
            f"{route}"
        )

    # Confirm that the selected airline has historical
    # observations for the requested directional route.
    airline_route_exists = (
        (
            feature_df["airline"] == airline
        )
        & (
            feature_df["route"] == route
        )
    ).any()

    if not airline_route_exists:
        raise ValueError(
            f"No historical observations were found for "
            f"{airline} on {route}."
        )

    # Convert the forecast date to the first day of its month
    # so that monthly observations use a consistent format.
    forecast_month = pd.Timestamp(
        forecast_month
    ).to_period("M").to_timestamp()
    
    # Identify the next forecastable calendar month.
    latest_month = feature_df["month"].max()
    next_month = (
        latest_month.to_period("M") + 1
    ).to_timestamp()

    # Require the forecast to use the month immediately after
    # the latest available historical observation.
    if forecast_month != next_month:
        raise ValueError(
            "Forecast month must be the next available month "
            f"after the historical data ({next_month.date()})."
        )
    
# ---------------------------------------------------------
# Prediction feature construction
# ---------------------------------------------------------

def build_prediction_features(
    feature_df: pd.DataFrame,
    airline: str,
    route: str,
    forecast_month: pd.Timestamp,
) -> pd.DataFrame:
    """Construct model features for the requested future month."""

    # Standardise the forecast month to the first day
    # of the requested calendar month.
    forecast_month = pd.Timestamp(
        forecast_month
    ).to_period("M").to_timestamp()

    # Validate the requested airline, route and forecast month.
    validate_prediction_request(
        feature_df,
        airline,
        route,
        forecast_month,
    )

    # Identify the previous three calendar months.
    previous_month = (
        forecast_month.to_period("M") - 1
    ).to_timestamp()

    previous_2_month = (
        forecast_month.to_period("M") - 2
    ).to_timestamp()

    previous_3_month = (
        forecast_month.to_period("M") - 3
    ).to_timestamp()

    # Select historical observations for the requested
    # airline and directional route.
    airline_route_df = feature_df[
        (feature_df["airline"] == airline)
        & (feature_df["route"] == route)
    ].copy()

    # Retrieve the immediately preceding month.
    previous_row = airline_route_df[
        airline_route_df["month"] == previous_month
    ]

    if previous_row.empty:
        raise ValueError(
            f"No {previous_month.date()} observation exists "
            f"for {airline} on {route}."
        )

    previous_row = previous_row.iloc[0]

    # Calculate the previous month's route-airline delay rate.
    delay_pct_lag_1 = (
        previous_row["departures_delayed"]
        / previous_row["sectors_flown"]
        * 100
    )

    # Calculate the previous month's cancellation rate.
    cancellation_pct_lag_1 = (
        previous_row["cancellations"]
        / previous_row["sectors_scheduled"]
        * 100
    )

    # Record previous-month flight volume.
    sectors_flown_lag_1 = previous_row[
        "sectors_flown"
    ]

    # Select the previous three calendar months for the
    # requested airline-route combination.
    recent_3m_df = airline_route_df[
        airline_route_df["month"].isin(
            [
                previous_month,
                previous_2_month,
                previous_3_month,
            ]
        )
    ]

    # Require at least one operated sector across the
    # recent three-month history.
    recent_sectors_flown = recent_3m_df[
        "sectors_flown"
    ].sum()

    if recent_sectors_flown <= 0:
        raise ValueError(
            "No operated sectors were found in the previous "
            "three months for this airline and route."
        )

    # Calculate the flight-volume-weighted three-month
    # departure delay rate.
    delay_pct_previous_3m = (
        recent_3m_df["departures_delayed"].sum()
        / recent_sectors_flown
        * 100
    )

    # Select all routes operated by the requested airline
    # during the previous calendar month.
    airline_previous_month_df = feature_df[
        (feature_df["airline"] == airline)
        & (feature_df["month"] == previous_month)
    ]

    airline_sectors_flown = airline_previous_month_df[
        "sectors_flown"
    ].sum()

    if airline_sectors_flown <= 0:
        raise ValueError(
            "No previous-month airline flight history "
            "is available."
        )

    # Calculate airline-wide previous-month delay performance.
    airline_delay_pct_lag_1 = (
        airline_previous_month_df[
            "departures_delayed"
        ].sum()
        / airline_sectors_flown
        * 100
    )

    # Select all airlines operating the requested directional
    # route during the previous calendar month.
    route_previous_month_df = feature_df[
        (feature_df["route"] == route)
        & (feature_df["month"] == previous_month)
    ]

    route_sectors_flown = route_previous_month_df[
        "sectors_flown"
    ].sum()

    if route_sectors_flown <= 0:
        raise ValueError(
            "No previous-month route flight history "
            "is available."
        )

    # Calculate route-wide previous-month delay performance.
    route_delay_pct_lag_1 = (
        route_previous_month_df[
            "departures_delayed"
        ].sum()
        / route_sectors_flown
        * 100
    )

    # Construct one observation using the exact predictors
    # expected by the trained model.
    prediction_features = pd.DataFrame(
        [
            {
                "airline": airline,
                "route": route,
                "year": forecast_month.year,
                "month_of_year": forecast_month.month,
                "delay_pct_lag_1": delay_pct_lag_1,
                "delay_pct_previous_3m": delay_pct_previous_3m,
                "cancellation_pct_lag_1": cancellation_pct_lag_1,
                "sectors_flown_lag_1": sectors_flown_lag_1,
                "airline_delay_pct_lag_1": airline_delay_pct_lag_1,
                "route_delay_pct_lag_1": route_delay_pct_lag_1,
            }
        ],
        columns=MODEL_FEATURES,
    )

    return prediction_features

# ---------------------------------------------------------
# Delay prediction
# ---------------------------------------------------------

def predict_delay_rate(
    feature_df: pd.DataFrame,
    model_pipeline,
    airline: str,
    route: str,
    forecast_month: pd.Timestamp,
) -> float:
    """Predict the monthly departure delay rate."""

    # Construct the historical and calendar features required
    # for the requested future month.
    prediction_features = build_prediction_features(
        feature_df,
        airline,
        route,
        forecast_month,
    )

    # Generate one prediction using the trained model pipeline.
    predictions = model_pipeline.predict(
        prediction_features
    )

    # Confirm that exactly one prediction was returned.
    if len(predictions) != 1:
        raise ValueError(
            "Expected exactly one model prediction."
        )

    # Convert the model output to a standard Python float.
    predicted_delay_rate = float(
        predictions[0]
    )

    # Confirm that the model returned a valid numerical value.
    if pd.isna(predicted_delay_rate):
        raise ValueError(
            "Model returned a missing prediction."
        )

    return predicted_delay_rate

# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

def main() -> None:
    """Run an interactive future departure-delay prediction."""

    # Load the historical data and trained model pipeline.
    feature_df = load_data()
    model_pipeline = load_model()

    # Determine the next month supported by the available data.
    latest_month = feature_df["month"].max()
    forecast_month = (
        latest_month.to_period("M") + 1
    ).to_timestamp()

    print("Australian Flight Delay Intelligence")
    print("------------------------------------")

    print(
        f"Latest historical month: "
        f"{latest_month.strftime('%B %Y')}"
    )

    print(
        f"Forecast month: "
        f"{forecast_month.strftime('%B %Y')}"
    )

    # Request the airline and directional route.
    airline = input(
        "\nEnter airline: "
    ).strip()

    route = input(
        "Enter directional route "
        "(example: Melbourne-Sydney): "
    ).strip()

    try:
        # Generate the future monthly departure-delay forecast.
        predicted_delay_rate = predict_delay_rate(
            feature_df,
            model_pipeline,
            airline,
            route,
            forecast_month,
        )

    except ValueError as error:
        print(
            f"\nPrediction could not be generated: "
            f"{error}"
        )
        return

    # Display the prediction in a user-readable format.
    print("\nForecast result")
    print("---------------")
    print(f"Airline: {airline}")
    print(f"Route: {route}")
    print(
        f"Forecast month: "
        f"{forecast_month.strftime('%B %Y')}"
    )
    print(
        f"Predicted departure delay rate: "
        f"{predicted_delay_rate:.2f}%"
    )


# Run the prediction interface only when this script
# is executed directly.
if __name__ == "__main__":
    main()