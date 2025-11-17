# Aussie Rain ML API

This project exposes a machine learning model as a REST API using **FastAPI** and **Docker**.

## Problem

We use the **WeatherAUS** dataset from the Australian Bureau of Meteorology to predict  
whether it will rain tomorrow (`RainTomorrow = Yes/No`) based on daily weather observations.

## Data

- Source: public "WeatherAUS" dataset (10+ years of daily observations)
- Target: `RainTomorrow` (Yes/No)
- Features:
  - Numeric: temperatures, rainfall, evaporation, sunshine, humidity, pressure, wind speed, cloud cover
  - Categorical: `Location`, wind direction (`WindGustDir`, `WindDir9am`, `WindDir3pm`), `RainToday`

## Tech Stack

- **Python**, **scikit-learn** — training and preprocessing
- **FastAPI** — REST API for model inference
- **Uvicorn** — ASGI server
- **Docker** — containerization and deployment
- (Optionally) **Render** — public hosting of the container

## API

### `POST /predict`

Request body (JSON):

```json
{
  "MinTemp": 10.0,
  "MaxTemp": 20.0,
  "Rainfall": 0.0,
  "Evaporation": 5.0,
  "Sunshine": 7.0,
  "WindGustSpeed": 40.0,
  "WindSpeed9am": 10.0,
  "WindSpeed3pm": 15.0,
  "Humidity9am": 70.0,
  "Humidity3pm": 50.0,
  "Pressure9am": 1015.0,
  "Pressure3pm": 1012.0,
  "Cloud9am": 4.0,
  "Cloud3pm": 4.0,
  "Temp9am": 16.0,
  "Temp3pm": 18.0,
  "Location": "Sydney",
  "WindGustDir": "N",
  "WindDir9am": "N",
  "WindDir3pm": "N",
  "RainToday": "No"
}

