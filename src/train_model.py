"""
Train the final Australian flight delay forecasting model.

This script loads the modelling-ready feature dataset, trains the
selected Gradient Boosting model using observations through December
2024, and saves the fitted pipeline for later evaluation and prediction.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# ---------------------------------------------------------
# File configuration
# ---------------------------------------------------------

# Define the modelling-ready feature dataset.
FEATURE_DATA_PATH = Path(
    "data/processed/bitre_otp_model_features.csv"
)

# Define the directory used to store trained model artifacts.
MODEL_DIR = Path(
    "models"
)

# Define the output path for the trained model pipeline.
MODEL_PATH = MODEL_DIR / "flight_delay_model.joblib"


# ---------------------------------------------------------
# Modelling configuration
# ---------------------------------------------------------

# Define categorical predictors describing the airline
# and directional route.
CATEGORICAL_FEATURES = [
    "airline",
    "route",
]

# Define numerical predictors available before the
# target month's delay performance is known.
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

# Combine categorical and numerical predictors.
MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)

# Define the regression target.
TARGET_COLUMN = "departure_delay_pct"

# Define the start of the untouched test period.
TEST_START_DATE = pd.Timestamp(
    "2025-01-01"
)

def load_data() -> pd.DataFrame:
    """Load the modelling-ready flight delay feature dataset."""

    # Confirm that the feature dataset exists.
    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset was not found at: "
            f"{FEATURE_DATA_PATH}"
        )

    # Load the feature dataset and parse the month column as dates.
    df = pd.read_csv(
        FEATURE_DATA_PATH,
        parse_dates=["month"],
        low_memory=False,
    )

    return df


def create_time_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological development and test datasets."""

    # Use observations before 2025 for model development.
    development_df = df[
        df["month"] < TEST_START_DATE
    ].copy()

    # Reserve observations from 2025 onward for final testing.
    test_df = df[
        df["month"] >= TEST_START_DATE
    ].copy()

    # Confirm that both chronological datasets contain observations.
    if development_df.empty:
        raise ValueError(
            "Development dataset contains no observations."
        )

    if test_df.empty:
        raise ValueError(
            "Test dataset contains no observations."
        )

    # Confirm that the development period ends before
    # the untouched test period begins.
    if (
        development_df["month"].max()
        >= test_df["month"].min()
    ):
        raise ValueError(
            "Development and test periods overlap."
        )

    return development_df, test_df

def create_model() -> Pipeline:
    """Create the final Gradient Boosting modelling pipeline."""

    # Encode categorical predictors while leaving
    # numerical predictors unchanged.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    # Create the final Gradient Boosting model using
    # the configuration selected during validation.
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        categorical_features=[0, 1],
        random_state=42,
    )

    # Combine preprocessing and model training
    # into one reusable pipeline.
    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return model_pipeline

def train_model(
    development_df: pd.DataFrame,
) -> Pipeline:
    """Train the final model using development-period observations."""

    # Select model predictors from the development dataset.
    X_development = development_df[
        MODEL_FEATURES
    ].copy()

    # Select the departure delay target.
    y_development = development_df[
        TARGET_COLUMN
    ].copy()

    # Confirm that the training predictors contain no missing values.
    missing_feature_values = (
        X_development
        .isna()
        .sum()
        .sum()
    )

    if missing_feature_values != 0:
        raise ValueError(
            "Development predictors contain missing values."
        )

    # Create the selected modelling pipeline.
    model_pipeline = create_model()

    # Fit the complete preprocessing and regression pipeline.
    model_pipeline.fit(
        X_development,
        y_development,
    )

    return model_pipeline

def main() -> None:
    """Train and save the final flight delay forecasting model."""

    # ---------------------------------------------------------
    # Load modelling data
    # ---------------------------------------------------------

    df = load_data()

    print("Flight delay model training")
    print("---------------------------")
    print(
        f"Total modelling rows: "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------
    # Create chronological split
    # ---------------------------------------------------------

    development_df, test_df = create_time_split(
        df
    )

    print(
        f"Development rows: "
        f"{len(development_df):,}"
    )

    print(
        f"Reserved test rows: "
        f"{len(test_df):,}"
    )

    print(
        f"Development period: "
        f"{development_df['month'].min().date()} to "
        f"{development_df['month'].max().date()}"
    )

    print(
        f"Reserved test period: "
        f"{test_df['month'].min().date()} to "
        f"{test_df['month'].max().date()}"
    )

    # ---------------------------------------------------------
    # Train final model
    # ---------------------------------------------------------

    model_pipeline = train_model(
        development_df
    )

    print("\nModel training complete.")

    # ---------------------------------------------------------
    # Save trained model
    # ---------------------------------------------------------

    # Create the model artifact directory if required.
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the complete fitted preprocessing and model pipeline.
    joblib.dump(
        model_pipeline,
        MODEL_PATH,
    )

    print("\nModel artifact saved:")
    print("---------------------")
    print(MODEL_PATH)


# Run model training only when this script is executed directly.
if __name__ == "__main__":
    main()

