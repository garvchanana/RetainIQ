import joblib
import pandas as pd
import numpy as np


# ==============================
# LOAD MODEL + FEATURES
# ==============================

def load_model_and_features(model_path: str, features_path: str):
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features


# ==============================
# PREPARE INPUT FOR MODEL
# ==============================

def prepare_model_input(df: pd.DataFrame, features: list) -> pd.DataFrame:
    data = df.copy()

    # Ensure all required features exist
    for col in features:
        if col not in data.columns:
            data[col] = 0

    # Keep only required features
    data = data[features]

    # Handle missing values
    data = data.fillna(0)

    return data


# ==============================
# PREDICT CHURN PROBABILITY
# ==============================

def predict_churn(df: pd.DataFrame, model, features: list, threshold: float = 0.45):
    data = df.copy()

    X = prepare_model_input(data, features)

    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]

    data["churn_probability"] = probs

    # Convert to risk label
    data["churn_risk"] = np.where(probs >= threshold, "High", "Low")

    return data