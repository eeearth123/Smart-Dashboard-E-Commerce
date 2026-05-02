# ============================================================
# app.py — Entry point
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(
    page_title="Olist Executive Cockpit",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

from i18n import t, render_lang_toggle
from config import BEST_THRESHOLD
from data.loader import load_bq_data
from data.features import process_features, assign_status
from models.predictor import load_model, predict_churn
import pages.p1_business  as p1
import pages.p2_churn     as p2
import pages.p3_action    as p3
import pages.p4_cycle     as p4
import pages.p5_logistics as p5
import pages.p6_seller    as p6
import pages.p7_customer  as p7

# ── Sidebar: ภาษา + refresh ──────────────────────────────────
render_lang_toggle()
with st.sidebar:
    if st.button(t("refresh_data")):
        st.cache_data.clear()
        st.rerun()

# ── Load + process ────────────────────────────────────────────
df_raw, bq_error = load_bq_data()
model, feature_names, model_error = load_model()

if bq_error:
    st.error(f"⚠️ BigQuery Error: {bq_error}")
    st.stop()
if model_error:
    st.warning(f"⚠️ Model: {model_error}")

df = process_features(df_raw)

if model is not None and feature_names:
    proba, pred = predict_churn(df, model, feature_names, BEST_THRESHOLD)
    df["churn_probability"] = proba
    df["churn_prediction"]  = pred
    if "product_category_name" in df.columns:
        cat_risk_map = df.groupby("product_category_name")["churn_probability"].mean()
        df["cat_churn_risk"] = df["product_category_name"].map(cat_risk_map)
else:
    df["churn_probability"] = 0.5
    df["churn_prediction"]  = 1

df["is_churn"] = df["churn_prediction"]
df["status"]   = df.apply(assign_status, axis=1)

# ── Sidebar: info + navigation ────────────────────────────────
PAGE_LABELS = [
    t("page_business"),
    t("page_churn"),
    t("page_action"),
    t("page_cycle"),
    t("page_logistics"),
    t("page_seller"),
    t("page_customer"),
]

with st.sidebar:
    st.title(t("app_title"))
    st.success(t("data_loaded", n=f"{len(df):,}"))
    st.info(t("model_threshold", v=f"{BEST_THRESHOLD:.2f}"))
    st.markdown(t("business_rules"))
    for rule_key in ["rule_lost","rule_high","rule_warning","rule_medium","rule_active"]:
        st.markdown(f"- {t(rule_key)}")

    # ใช้ index เพื่อ avoid key mismatch เมื่อเปลี่ยนภาษา
    page_idx = st.radio(
        t("navigation"),
        options=range(len(PAGE_LABELS)),
        format_func=lambda i: PAGE_LABELS[i],
        key="nav_radio",
    )
    st.markdown("---")

# ── Router ────────────────────────────────────────────────────
PAGE_RENDERS = [
    lambda: p1.render(df),
    lambda: p2.render(df, t, BEST_THRESHOLD),
    lambda: p3.render(df, model, feature_names),
    lambda: p4.render(df),
    lambda: p5.render(df),
    lambda: p6.render(df),
    lambda: p7.render(df),
]

PAGE_RENDERS[page_idx]()
