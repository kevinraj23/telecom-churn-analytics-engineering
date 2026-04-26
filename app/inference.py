import os
import joblib
import numpy as np
import pandas as pd
from app.features import MODEL_FEATURES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'churn_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_churn(customer_data: dict):
    """
    customer_data: dict with keys matching MODEL_FEATURES
    Returns: (churn_probability, risk_label)
    """
    model, scaler = load_model()

    df = pd.DataFrame([customer_data])[MODEL_FEATURES]
    scaled = scaler.transform(df)
    prob = model.predict_proba(scaled)[0][1]

    if prob >= 0.7:
        risk = "🔴 High Risk"
    elif prob >= 0.4:
        risk = "🟡 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    return round(prob * 100, 2), risk
