# ============================================================
# models/predictor.py — โหลดโมเดล V5 + ทำนาย 3-Class
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
    โหลด model และ feature list จากไฟล์ .pkl (V5 Hybrid Model)
    Returns: (model | None, feature_names | None, error | None)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)

    try:
        model_path    = os.path.join(root_dir, MODEL_FILE)
        features_path = os.path.join(root_dir, FEATURES_FILE)

        # Fallback to desktop root if not found in root_dir
        if not os.path.exists(model_path):
            desktop_dir = r"d:\Desktop\Churn interi Dashboard"
            model_path = os.path.join(desktop_dir, MODEL_FILE)
            features_path = os.path.join(desktop_dir, FEATURES_FILE)

        model    = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features, None
    except Exception as e:
        return None, None, str(e)


def predict_churn(df: pd.DataFrame, model, feature_names: list, threshold: float) -> tuple:
    """
    ทำนาย churn probability และ binary prediction สำหรับ V5 (3-Class Model)
    Returns: (proba_array, pred_array)
    """
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        X[col] = df[col] if col in df.columns else 0
    X = X.fillna(X.median())

    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(X)
        if proba_all.shape[1] == 3:
            # V5 3-Class Model: Class 2 = Churn, Class 0 = Stay
            proba = proba_all[:, 2]
        elif proba_all.shape[1] == 2:
            proba = proba_all[:, 1]
        else:
            proba = 1 - proba_all[:, 0]
    else:
        proba = model.predict(X).astype(float)

    return proba, (proba >= threshold).astype(int)
