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
    ทำนายความน่าจะเป็นทั้ง 3 Class สำหรับ V5 (3-Class Model)
    Returns: (prob_stay, prob_delay, prob_churn, pred_class)
    """
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        X[col] = df[col] if col in df.columns else 0
    X = X.fillna(X.median())

    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(X)
        if proba_all.shape[1] == 3:
            # V5 3-Class Model: Class 0 = Stay, Class 1 = Delay, Class 2 = Churn
            prob_stay  = proba_all[:, 0]
            prob_delay = proba_all[:, 1]
            prob_churn = proba_all[:, 2]
            pred_class = np.argmax(proba_all, axis=1)
        elif proba_all.shape[1] == 2:
            prob_stay  = proba_all[:, 0]
            prob_delay = np.zeros(len(X))
            prob_churn = proba_all[:, 1]
            pred_class = np.where(prob_churn >= threshold, 2, 0)
        else:
            prob_stay  = proba_all[:, 0]
            prob_delay = np.zeros(len(X))
            prob_churn = 1 - proba_all[:, 0]
            pred_class = np.where(prob_churn >= threshold, 2, 0)
    else:
        prob_churn = model.predict(X).astype(float)
        prob_stay  = 1 - prob_churn
        prob_delay = np.zeros(len(X))
        pred_class = np.where(prob_churn >= threshold, 2, 0)

    return prob_stay, prob_delay, prob_churn, pred_class
