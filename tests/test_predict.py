"""
Tests for the reusable flight delay prediction pipeline.
"""

import pandas as pd
import pytest

from src.predict import (
    build_prediction_features,
    load_data,
)


def test_load_data_returns_historical_dataset():
    """Check that the historical prediction dataset loads correctly."""

    # Load the cleaned route-level historical dataset.
    historical_df = load_data()

    # Confirm that the dataset contains observations.
    assert not historical_df.empty

    # Define columns required by the prediction pipeline.
    required_columns = {
        "airline",
        "route",
        "month",
        "sectors_scheduled",
        "sectors_flown",
        "cancellations",
        "departures_delayed",
    }

    # Confirm that every required column is available.
    assert required_columns.issubset(
        historical_df.columns
    )

    # Confirm that the month column was parsed as datetime.
    assert pd.api.types.is_datetime64_any_dtype(
        historical_df["month"]
    )

def test_build_prediction_features_returns_expected_values():
    """Check feature construction for a known future forecast."""

    # Load the cleaned historical route-level dataset.
    historical_df = load_data()

    # Construct the July 2026 prediction features for
    # Qantas operating from Melbourne to Sydney.
    prediction_features = build_prediction_features(
        historical_df,
        "Qantas",
        "Melbourne-Sydney",
        pd.Timestamp("2026-07-01"),
    )

    # Confirm that exactly one prediction row was created.
    assert len(prediction_features) == 1

    # Select the generated observation.
    prediction_row = prediction_features.iloc[0]

    # Confirm categorical and calendar features.
    assert prediction_row["airline"] == "Qantas"
    assert prediction_row["route"] == "Melbourne-Sydney"
    assert prediction_row["year"] == 2026
    assert prediction_row["month_of_year"] == 7

    # Confirm historical lag features using approximate
    # comparisons for floating-point values.
    assert prediction_row["delay_pct_lag_1"] == pytest.approx(
        11.873351,
        abs=0.000001,
    )

    assert prediction_row["delay_pct_previous_3m"] == pytest.approx(
        11.370514,
        abs=0.000001,
    )

    assert prediction_row["cancellation_pct_lag_1"] == pytest.approx(
        2.944942,
        abs=0.000001,
    )

    assert prediction_row["sectors_flown_lag_1"] == pytest.approx(
        758.0
    )

    assert prediction_row["airline_delay_pct_lag_1"] == pytest.approx(
        11.980479,
        abs=0.000001,
    )

    assert prediction_row["route_delay_pct_lag_1"] == pytest.approx(
        17.270789,
        abs=0.000001,
    )

def test_build_prediction_features_rejects_unknown_airline():
    """Check that an unknown airline produces a clear error."""

    # Load the cleaned historical route-level dataset.
    historical_df = load_data()

    # Confirm that an airline absent from the historical data
    # cannot be used to construct prediction features.
    with pytest.raises(
        ValueError,
        match="Airline was not found in historical data",
    ):
        build_prediction_features(
            historical_df,
            "Fake Airline",
            "Melbourne-Sydney",
            pd.Timestamp("2026-07-01"),
        )


def test_build_prediction_features_rejects_unknown_route():
    """Check that an unknown route produces a clear error."""

    # Load the cleaned historical route-level dataset.
    historical_df = load_data()

    # Confirm that a route absent from the historical data
    # cannot be used to construct prediction features.
    with pytest.raises(
        ValueError,
        match="Route was not found in historical data",
    ):
        build_prediction_features(
            historical_df,
            "Qantas",
            "Fake-Route",
            pd.Timestamp("2026-07-01"),
        )