# AQI Predictor

**AQI Predictor** is a machine learning data pipeline designed to ingest, engineer, and store Air Quality Index (AQI) data and meteorological features for predicting air quality levels. The project connects to real-time weather and AQI services, processes incoming data streams into ML-ready features, and manages feature data within the Hopsworks Feature Store.

---

## 📁 Project Structure

```
aqi-predictor/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_fetch.py
│   ├── feature_engineering.py
│   ├── feature_store.py
│   └── run_pipeline.py
├── tests/
│   ├── __init__.py
│   ├── test_data_fetch.py
│   ├── test_feature_engineering.py
│   ├── test_feature_store.py
│   └── test_run_pipeline.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

### 1. Clone & Navigate to Project Directory
```bash
git clone <repository-url>
cd aqi-predictor
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (`.env`)

Copy the template file `.env.example` to `.env`:

```bash
cp .env.example .env
```

Open `.env` and fill in your configuration values:

```env
# API Credentials
AQICN_API_TOKEN=your_aqicn_api_token_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
HOPSWORKS_API_KEY=your_hopsworks_api_key_here

# Hopsworks Feature Store Configuration
HOPSWORKS_PROJECT_NAME=your_hopsworks_project_name_here

# Location Settings
CITY_NAME=London
CITY_LAT=51.5074
CITY_LON=-0.1278
```

> **Note**: `src/config.py` validates required environment variables at import time and will raise a `ValueError` if any variable is missing.

---

## 🚀 Running the Pipeline

### 1. Execute a Dry-Run Pipeline

To run the pipeline without writing to Hopsworks Feature Store, use the `--dry-run` flag:

```bash
python -m src.run_pipeline --dry-run
```

You can also override the configured city using `--city`:

```bash
python -m src.run_pipeline --city Paris --dry-run
```

#### Example Successful Dry-Run Output:

```
2026-07-31 17:58:06,392 - __main__ - INFO - Starting Feature Pipeline for city: 'London' (Dry-run: True)
2026-07-31 17:58:06,392 - src.data_fetch - INFO - Fetching AQICN data for city 'London' (Attempt 1/3)
2026-07-31 17:58:06,392 - src.data_fetch - INFO - Fetching OpenWeather data for (lat: 51.5074, lon: -0.1278) (Attempt 1/3)
2026-07-31 17:58:06,392 - __main__ - INFO - Computing temporal & derived features...
2026-07-31 17:58:06,392 - __main__ - INFO - Dry-run mode: Feature row generated successfully.

--- Generated Feature Row ---
{
  "fetched_at": "2026-07-31T12:57:56.701192+00:00",
  "city": "London",
  "aqi": 45,
  "pm25": 12.0,
  "pm10": 24.0,
  "o3": 10.5,
  "no2": 14.2,
  "so2": 3.1,
  "co": 0.8,
  "temperature": 19.5,
  "humidity": 65.0,
  "wind_speed": 3.6,
  "pressure": 1013.0,
  "hour": 12,
  "day_of_week": 4,
  "day_of_month": 31,
  "month": 7,
  "is_weekend": 0,
  "season": "summer",
  "aqi_change_rate_1h": 0.05,
  "aqi_change_rate_24h": -0.10,
  "aqi_rolling_mean_6h": 44.5,
  "aqi_rolling_mean_24h": 46.2,
  "pm25_pm10_ratio": 0.5,
  "target_aqi_3d": null
}

✅ Pipeline run complete for London at 2026-07-31T12:57:56.701192+00:00
```

### 2. Execute Full Pipeline (Store to Hopsworks)

Once your `.env` contains valid Hopsworks credentials, run:

```bash
python -m src.run_pipeline
```

---

## 🧪 Running Tests

To run the complete unit test suite:

```bash
python -m unittest discover -s tests
```

Or using `pytest`:

```

## Deployment

### Streamlit Community Cloud

This repository is configured to be deployed automatically on [Streamlit Community Cloud](https://streamlit.io/cloud).

**Steps to deploy:**
1. Connect your GitHub repository to Streamlit Community Cloud.
2. Set the **Main file path** to `app/streamlit_app.py`.
3. In the Streamlit app dashboard, go to **Settings > Secrets**.
4. Paste the required API keys using TOML format. It should exactly match this snippet:

```toml
# Streamlit Cloud Secrets (.streamlit/secrets.toml format)
AQICN_API_TOKEN = "your-aqicn-token"
OPENWEATHER_API_KEY = "your-openweather-key"
HOPSWORKS_API_KEY = "your-hopsworks-key"
HOPSWORKS_PROJECT_NAME = "your-hopsworks-project-name"
CITY_NAME = "Lahore"
CITY_LAT = "31.558"
CITY_LON = "74.3507"
```

Once the secrets are saved, Streamlit will automatically install dependencies from `requirements.txt` and launch the interactive Air Quality dashboard.
