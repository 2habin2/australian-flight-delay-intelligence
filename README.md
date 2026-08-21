# Australian Flight Delay Intelligence

An end-to-end data science and engineering project that analyses historical Australian domestic airline performance and forecasts monthly departure delay rates using official Bureau of Infrastructure and Transport Research Economics (BITRE) data.

## Project Overview

Australian domestic airline performance varies substantially across airlines, routes, seasons and periods of disruption. This project builds a reproducible data pipeline to investigate these patterns and forecast future departure delay rates.

The project uses monthly BITRE on-time performance data covering January 2010 to June 2026. Each route-level observation represents an airline operating on a directional route during a particular month.

The workflow includes:

- historical data validation and cleaning
- exploratory data analysis
- time-aware feature engineering
- chronological model validation
- Gradient Boosting regression
- evaluation on an unseen future test period

The final model is evaluated on data from January 2025 to June 2026, which was kept separate from model development.

## Model Performance

| Metric | Test Result |
| --- | ---: |
| MAE | 6.594 percentage points |
| RMSE | 8.479 percentage points |
| R² | 0.447 |

On the reserved January 2025 to June 2026 test period, the final model achieved an MAE of 6.594 percentage points, meaning its predicted monthly departure delay rate differed from the observed rate by approximately 6.6 percentage points on average.

## Data Source

The project uses official Australian domestic airline on-time performance data published by the Bureau of Infrastructure and Transport Research Economics (BITRE).

Source: [BITRE Airline On Time Performance](https://www.bitre.gov.au/statistics/aviation/otp_month)

The historical workbook contains monthly airline performance data from January 2010 to June 2026 across multiple worksheets.

At the route level, each observation represents:

**airline × directional route × month**

For example, Melbourne–Sydney and Sydney–Melbourne are treated as separate directional routes.

The source data includes measures such as:

- sectors scheduled
- sectors flown
- cancellations
- departures on time
- departures delayed
- arrivals on time
- arrivals delayed
- on-time departure and arrival percentages
- cancellation percentage

Aggregate observations such as `All Airlines` and `All Ports-All Ports` are retained in the cleaned historical dataset for broader analysis but excluded from the route-level modelling dataset.

Raw and generated processed datasets are excluded from version control. The repository contains the Python pipelines required to validate, clean and transform the original BITRE data reproducibly.

## Data Pipeline

The project separates data validation, cleaning, feature engineering, model training and evaluation into reproducible Python scripts.

```text
BITRE historical Excel workbook
        ↓
Historical data validation
        ↓
Historical data cleaning
        ↓
Route-level dataset
        ↓
Feature engineering
        ↓
Modelling-ready dataset
        ↓
Model training
        ↓
Future-period evaluation
```

### Pipeline Scripts

| Script | Purpose |
| --- | --- |
| `src/inspect_historical_sheets.py` | Inspects worksheet structure and historical coverage in the BITRE workbook. |
| `src/validate_historical_data.py` | Checks historical data quality, duplicates and business-rule consistency. |
| `src/clean_historical_data.py` | Combines and cleans historical worksheets and creates the route-level dataset. |
| `src/validate_historical_processed_data.py` | Validates the cleaned historical datasets before analysis and modelling. |
| `src/build_features.py` | Creates the modelling target, calendar variables and historical lag features. |
| `src/train_model.py` | Trains the selected Gradient Boosting model using data through December 2024. |
| `src/evaluate_model.py` | Evaluates the trained model on the reserved January 2025 to June 2026 test period. |

Generated processed datasets and trained model artifacts are excluded from version control because they can be reproduced from the project pipeline.

## Exploratory Data Analysis

Exploratory analysis was performed on the cleaned route-level data covering January 2010 to June 2026.

Key findings include:

- Australian domestic departure performance was relatively stable before 2020, followed by substantial disruption during the COVID-19 period.
- Domestic flight volume fell sharply during 2020 and 2021 before recovering toward pre-pandemic levels.
- Departure delays increased substantially during 2022, with July 2022 recording the highest monthly delay rate in the historical period.
- Performance improved gradually after 2022, although delay rates remained more volatile than in many pre-pandemic years.
- Among the four major airlines with complete 2010–2026 coverage, Qantas recorded the lowest historical departure delay rate, while Jetstar recorded the highest.
- Delay performance varies across directional routes, supporting the inclusion of route information as a model feature.
- Delay rates also show seasonality, with July having the highest median departure delay rate across complete calendar years.
- Route-airline-month observations with very low flight volumes have substantially more volatile delay percentages.

Based on the flight-volume analysis, observations with at least 25 flown sectors were retained for modelling. This retains approximately 84% of eligible observations while reducing instability in the regression target.

## Feature Engineering

The modelling dataset uses only information that would have been available before the target month's departure performance was known. This helps prevent data leakage from future observations.

The regression target is:

`departure_delay_pct`

which represents the percentage of flown sectors that departed late for a particular airline, directional route and month.

The final model uses the following predictors:

| Feature | Description |
| --- | --- |
| `airline` | Airline operating the route. |
| `route` | Directional origin-destination route. |
| `year` | Calendar year of the target observation. |
| `month_of_year` | Calendar month used to represent seasonality. |
| `delay_pct_lag_1` | Departure delay rate for the same airline-route combination in the previous month. |
| `delay_pct_previous_3m` | Flight-volume-weighted departure delay rate from the previous three available months. |
| `cancellation_pct_lag_1` | Previous-month cancellation rate for the same airline-route combination. |
| `sectors_flown_lag_1` | Number of sectors flown by the same airline-route combination in the previous month. |
| `airline_delay_pct_lag_1` | Airline-wide departure delay rate from the previous month. |
| `route_delay_pct_lag_1` | Route-wide departure delay rate from the previous month. |

Historical features are shifted before being joined to the target month so that current-month performance is not used to predict itself.

After applying the minimum flight-volume requirement and removing observations without the required historical features, the final modelling dataset contains **57,612 observations** covering February 2010 to June 2026.

## Modelling Approach

Because the dataset is time-dependent, model development uses chronological splits rather than a random train-test split. This prevents future observations from being used to predict earlier periods.

The modelling process uses three stages:

- **Training period:** February 2010 to December 2023
- **Validation period:** January 2024 to December 2024
- **Final test period:** January 2025 to June 2026

The 2025–2026 test period remained untouched during model selection and hyperparameter comparison.

### Models Compared

Three approaches were evaluated during model development:

1. **Persistence baseline**  
   Uses the previous month's departure delay rate as the prediction for the current month.

2. **Ridge Regression**  
   Provides a regularised linear benchmark using categorical, seasonal and historical performance features.

3. **Gradient Boosting Regression**  
   Captures nonlinear relationships between airline, route, seasonality and historical operational performance.

Validation performance showed that Gradient Boosting provided the strongest predictive performance.

| Model | Validation MAE | Validation RMSE | Validation R² |
| --- | ---: | ---: | ---: |
| Persistence baseline | 8.006 | 10.654 | 0.223 |
| Ridge Regression | 7.063 | 9.221 | 0.418 |
| Gradient Boosting | 6.943 | 8.980 | 0.448 |

A small controlled hyperparameter comparison was then performed using the 2024 validation period. The selected, more regularized Gradient Boosting configuration achieved a validation MAE of **6.931**, RMSE of **8.958** and R² of **0.451**.

The selected configuration was then retrained using all development data through December 2024 before being evaluated once on the reserved future test set.

## Final Model Results

After model selection, the final Gradient Boosting pipeline was retrained using observations from February 2010 through December 2024 and evaluated on the previously unseen January 2025 to June 2026 test period.

The reserved test set contained **5,092 observations**.

| Metric | Final Test Result |
| --- | ---: |
| MAE | 6.594 percentage points |
| RMSE | 8.479 percentage points |
| R² | 0.447 |

The final MAE means that predicted monthly departure delay rates differ from the observed rates by approximately **6.6 percentage points on average**.

The positive R² indicates that the model explains approximately **44.7% of the variation** in departure delay rates in the future test period.

### Model Interpretation

Permutation feature importance showed that the strongest predictive signals were:

1. recent three-month airline-route delay performance
2. previous-month airline-route delay performance
3. calendar month
4. directional route

This suggests that recent operational performance, seasonality and route characteristics are important signals for forecasting monthly departure delay rates.

Prediction-error analysis also showed that the model performs better within the more common range of delay rates and tends to produce less extreme predictions when actual delay rates are unusually high or low. This is an important limitation during major disruption periods.

## Project Structure

```text
australian-flight-delay-intelligence/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_historical_eda.ipynb
│   └── 02_feature_engineering.ipynb
├── src/
│   ├── inspect_bitre_data.py
│   ├── inspect_historical_sheets.py
│   ├── validate_bitre_data.py
│   ├── validate_historical_data.py
│   ├── clean_bitre_data.py
│   ├── clean_historical_data.py
│   ├── validate_processed_data.py
│   ├── validate_historical_processed_data.py
│   ├── build_features.py
│   ├── train_model.py
│   └── evaluate_model.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

The `data/raw`, `data/processed` and generated model files are excluded from version control. The repository instead provides the code required to recreate the processed data, modelling features and trained model.

## Reproducing the Pipeline

Create and activate a Python virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the BITRE historical on-time performance workbook and place it at:

`data/raw/OTP_Time_Series_Master_Current_5.xlsx`

The raw workbook is excluded from version control and must be downloaded separately before running the pipeline.

Run the main pipeline stages from the project root:

```bash
python src/clean_historical_data.py
python src/validate_historical_processed_data.py
python src/build_features.py
python src/train_model.py
python src/evaluate_model.py
```

The final command evaluates the trained model against the reserved January 2025 to June 2026 test period.

## Technologies

The project uses:

- Python
- pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook
- joblib
- Git and GitHub

## Limitations

The BITRE dataset is aggregated at the monthly airline-route level, so the model predicts monthly departure delay rates rather than delays for individual flights.

The current model is based primarily on historical operational performance, airline, route and seasonality. It does not yet incorporate external factors such as weather conditions, airport congestion, public holidays or major operational disruptions.

The model also tends to produce less extreme predictions when actual delay rates are unusually high or low, which may reduce accuracy during major disruption periods.

## Future Development

Future extensions could include integrating weather data, adding airport-level contextual features and developing automated data ingestion and model retraining workflows.

The modelling pipeline could also be exposed through an API or interactive dashboard to provide route-level delay intelligence and future delay-risk forecasts.


