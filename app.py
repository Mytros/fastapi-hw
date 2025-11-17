from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Load model bundle at startup
# ---------------------------------------------------------
BUNDLE_PATH = "models/aussie_rain.joblib"

bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
encoder = bundle["encoder"]
NUM = list(bundle["numeric_cols"])
CAT = list(bundle["categorical_cols"])

app = FastAPI(
    title="Aussie Rain ML API",
    description="Predicts whether it will rain tomorrow in Australia.",
    version="1.0.0",
)


# ---------------------------------------------------------
# Pydantic model for request body
# ---------------------------------------------------------
class WeatherRequest(BaseModel):
    # numeric features
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float
    Sunshine: float
    WindGustSpeed: float
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float
    Cloud3pm: float
    Temp9am: float
    Temp3pm: float

    # categorical features
    Location: str
    WindGustDir: str
    WindDir9am: str
    WindDir3pm: str
    RainToday: str


class PredictionResponse(BaseModel):
    RainTomorrow: str
    probability: float


# ---------------------------------------------------------
# Preprocessing function (same logic as Streamlit app)
# ---------------------------------------------------------
def preprocess_row(data: dict) -> np.ndarray:
    """Apply imputer -> scaler -> OHE exactly as during training."""
    df_in = pd.DataFrame([data])

    df_num = df_in[NUM]
    df_cat = df_in[CAT]

    # Impute numeric columns
    df_num_imputed = pd.DataFrame(
        imputer.transform(df_num),
        columns=NUM,
        index=df_in.index,
    )

    # Scale numeric columns
    df_num_scaled = pd.DataFrame(
        scaler.transform(df_num_imputed),
        columns=NUM,
        index=df_in.index,
    )

    # One-hot encode categorical columns
    X_cat = encoder.transform(df_cat)
    if hasattr(X_cat, "toarray"):
        X_cat = X_cat.toarray()

    # Final feature matrix
    X = np.hstack([df_num_scaled.values, X_cat])
    return X


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Aussie Rain ML API is running. Go to /docs for Swagger UI."}


@app.post("/predict", response_model=PredictionResponse)
def predict(weather: WeatherRequest):
    # Convert input to dict
    data = weather.dict()

    # Preprocess
    X = preprocess_row(data)

    # Predict probability of rain tomorrow (class 1)
    prob = float(model.predict_proba(X)[0, 1])
    label = "Yes" if prob >= 0.5 else "No"

    return PredictionResponse(RainTomorrow=label, probability=prob)
