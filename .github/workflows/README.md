# GitHub Actions Workflows

This directory contains the automated workflows that power the AQI Predictor CI/CD pipeline. 

## Workflows Overview

### 1. Feature Pipeline (`feature_pipeline.yml`)
- **Schedule**: Runs every hour at minute 0 (`0 * * * *`).
- **Purpose**: Fetches the latest raw AQI and weather data, computes derived and temporal features, and writes the latest feature row to the Hopsworks Feature Store.
- **Manual Trigger**: Can be run manually for testing.

### 2. Training Pipeline (`training_pipeline.yml`)
- **Schedule**: Runs daily at 3:00 AM UTC (`0 3 * * *`).
- **Purpose**: Orchestrates the entire training lifecycle:
  1. Data Preparation (Chronological Splitting)
  2. Baseline Model Training
  3. Deep Learning (LSTM) Training
  4. Model Comparison (Model selection against Persistence Baseline)
  5. Model Registry Upload
- **Safeguard**: Includes a 30-minute timeout to prevent runaway LSTM training costs.
- **Manual Trigger**: Can be run manually for testing.

## Required Secrets

For these workflows to authenticate with external APIs (AQICN, OpenWeather, Hopsworks), you must configure the following repository secrets. 

Navigate to **Settings > Secrets and variables > Actions > Repository secrets** and add:

- `AQICN_API_TOKEN`
- `OPENWEATHER_API_KEY`
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT_NAME`
- `CITY_NAME`
- `CITY_LAT`
- `CITY_LON`

## How to Trigger Manually

1. Go to the **Actions** tab in your GitHub repository.
2. Under "All workflows" on the left sidebar, select either **Feature Pipeline** or **Training Pipeline**.
3. Click the **Run workflow** dropdown button on the right side.
4. Click the green **Run workflow** button.
