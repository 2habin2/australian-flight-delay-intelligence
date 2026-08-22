"""
API for the Australian Flight Delay Intelligence project.

This module exposes the flight delay prediction pipeline
through HTTP endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import (
    load_data,
    load_model,
    predict_delay_rate,
)

# Create the FastAPI application.
app = FastAPI(
    title="Australian Flight Delay Intelligence API",
    version="1.0.0",
)

# ---------------------------------------------------------
# API request models
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    """Define the information required for a delay prediction."""

    airline: str
    route: str


@app.get("/")
def read_root() -> dict:
    """Return basic information about the API."""

    # Return a simple response confirming that the API is running.
    return {
        "message": "Australian Flight Delay Intelligence API",
        "status": "running",
    }

# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict_departure_delay(
    request: PredictionRequest,
) -> dict:
    """Generate the next-month departure delay forecast."""

    try:
        # Load the historical route-level data.
        historical_df = load_data()

        # Load the trained model pipeline.
        model_pipeline = load_model()

        # Identify the next forecastable calendar month.
        latest_month = historical_df["month"].max()

        forecast_month = (
            latest_month.to_period("M") + 1
        ).to_timestamp()

        # Generate the route-level departure delay forecast.
        predicted_delay_rate = predict_delay_rate(
            historical_df,
            model_pipeline,
            request.airline,
            request.route,
            forecast_month,
        )

    except ValueError as error:
        # Return a client error for invalid airline or route input.
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        # Return a server error when required project files are missing.
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    # Return the prediction as a JSON response.
    return {
        "airline": request.airline,
        "route": request.route,
        "forecast_month": forecast_month.strftime(
            "%B %Y"
        ),
        "predicted_departure_delay_rate": round(
            predicted_delay_rate,
            2,
        ),
    }