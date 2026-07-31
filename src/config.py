import os
from pathlib import Path
from dotenv import load_dotenv

# Path to project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file in project root
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# Required environment variables list
REQUIRED_ENV_VARS = [
    "AQICN_API_TOKEN",
    "OPENWEATHER_API_KEY",
    "HOPSWORKS_API_KEY",
    "HOPSWORKS_PROJECT_NAME",
    "CITY_NAME",
    "CITY_LAT",
    "CITY_LON",
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

if missing_vars:
    raise ValueError(
        f"Missing required environment variable(s): {', '.join(missing_vars)}. "
        "Please set them in your .env file or environment."
    )

AQICN_API_TOKEN: str = os.getenv("AQICN_API_TOKEN")
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY: str = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME: str = os.getenv("HOPSWORKS_PROJECT_NAME")
CITY_NAME: str = os.getenv("CITY_NAME")
CITY_LAT: float = float(os.getenv("CITY_LAT"))
CITY_LON: float = float(os.getenv("CITY_LON"))
