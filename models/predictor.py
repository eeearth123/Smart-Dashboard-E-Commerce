# ============================================================
# models/predictor.py — โหลดโมเดล + ทำนาย
# ============================================================
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from config import MODEL_FILE, FEATURES_FILE


@st.cache_resource
def load_model() -> tuple:
    """
    โหลด model และ feature list จากไฟล์ .pkl
    Returns: (model | None, feature_names | None, error | None)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # ขึ้นไป 1 ระดับ (จาก models/ → root)
    root_dir = os.path.dirname(base_dir)

    try:
        model    = joblib.load(os.path.join(root_dir, MODEL_FILE))
        features = joblib.load(os.path.join(root_dir, FEATURES_FILE))
        return model, features, None
    except Exception as e:
        return None, None, str(e)


def predict_churn(df: pd.DataFrame, model, feature_names: list, threshold: float) -> tuple:
    """
    ทำนาย churn probability และ binary prediction
    Returns: (proba_array, pred_array)
    """
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        X[col] = df[col] if col in df.columns else 0
    X = X.fillna(X.median())

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X).astype(float)

    return proba, (proba >= threshold).astype(int)
