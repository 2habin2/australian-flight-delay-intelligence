"""
Evaluate the final Australian flight delay forecasting model.

This script loads the trained model pipeline and evaluates it using
the reserved January 2025 to June 2026 test period.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the modelling-ready feature dataset.
FEATURE_DATA_PATH = Path(
    "data/processed/bitre_otp_model_features.csv"
)

# Define the trained model artifact.
MODEL_PATH = Path(
    "models/flight_delay_model.joblib"
)


# ---------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------

# Define the start of the reserved test period.
TEST_START_DATE = pd.Timestamp(
    "2025-01-01"
)

# Define categorical predictors.
CATEGORICAL_FEATURES = [
    "airline",
    "route",
]

# Define numerical predictors.
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

# Combine all predictors used by the trained model.
MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)

# Define the regression target.
TARGET_COLUMN = "departure_delay_pct"

def load_test_data() -> pd.DataFrame:
    """Load and isolate the reserved model test dataset."""

    # Confirm that the modelling-ready feature dataset exists.
    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset was not found at: "
            f"{FEATURE_DATA_PATH}"
        )

    # Load the complete modelling-ready dataset.
    df = pd.read_csv(
        FEATURE_DATA_PATH,
        parse_dates=["month"],
        low_memory=False,
    )

    # Select observations belonging to the reserved
    # January 2025 to June 2026 test period.
    test_df = df[
        df["month"] >= TEST_START_DATE
    ].copy()

    # Confirm that test observations are available.
    if test_df.empty:
        raise ValueError(
            "Reserved test dataset contains no observations."
        )

    return test_df


def load_model():
    """Load the trained flight delay model pipeline."""

    # Confirm that the trained model artifact exists.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model was not found at: "
            f"{MODEL_PATH}"
        )

    # Load the fitted preprocessing and regression pipeline.
    model_pipeline = joblib.load(
        MODEL_PATH
    )

    return model_pipeline


def evaluate_model(
    test_df: pd.DataFrame,
    model_pipeline,
) -> tuple[float, float, float]:
    """Evaluate the trained model on the reserved test period."""

    # Select predictors for the reserved test observations.
    X_test = test_df[
        MODEL_FEATURES
    ].copy()

    # Select the observed departure delay rates.
    y_test = test_df[
        TARGET_COLUMN
    ].copy()

    # Confirm that test predictors contain no missing values.
    missing_feature_values = (
        X_test
        .isna()
        .sum()
        .sum()
    )

    if missing_feature_values != 0:
        raise ValueError(
            "Test predictors contain missing values."
        )

    # Generate predictions using the trained pipeline.
    predictions = model_pipeline.predict(
        X_test
    )

    # Calculate mean absolute error in percentage points.
    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    # Calculate root mean squared error.
    rmse = (
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )

    # Calculate the proportion of target variance explained.
    r2 = r2_score(
        y_test,
        predictions,
    )

    return mae, rmse, r2

def main() -> None:
    """Run final evaluation on the reserved test period."""

    # ---------------------------------------------------------
    # Load test data and trained model
    # ---------------------------------------------------------

    test_df = load_test_data()
    model_pipeline = load_model()

    print("Final flight delay model evaluation")
    print("-----------------------------------")

    print(
        f"Test observations: "
        f"{len(test_df):,}"
    )

    print(
        f"Test period: "
        f"{test_df['month'].min().date()} to "
        f"{test_df['month'].max().date()}"
    )

    # ---------------------------------------------------------
    # Evaluate final model
    # ---------------------------------------------------------

    mae, rmse, r2 = evaluate_model(
        test_df,
        model_pipeline,
    )

    print("\nFinal model test performance:")
    print("-----------------------------")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")


# Run evaluation only when this script is executed directly.
if __name__ == "__main__":
    main()


