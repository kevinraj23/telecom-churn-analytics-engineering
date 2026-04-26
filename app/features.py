import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    Used in both training (notebook) and inference (app).
    """
    df = df.copy()

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])

    # Binary flags
    df['is_fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)

    # Tenure bucket
    df['tenure_bucket'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 72],
        labels=['0-12 months', '12-24 months', '24+ months']
    )

    return df


# Features used by the model — must match training exactly
MODEL_FEATURES = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'is_fiber',
    'is_month_to_month',
    'SeniorCitizen'
]
