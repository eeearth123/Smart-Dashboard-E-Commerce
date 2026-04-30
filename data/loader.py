# ============================================================
# data/loader.py — โหลดข้อมูลจาก BigQuery
#
# ✅ ไม่ต้องแก้ไฟล์นี้เลยเมื่อ deploy บน Streamlit Cloud
#    แค่ไปตั้ง Secrets ใน Streamlit Cloud UI ครั้งเดียว
#    (ดูวิธีใน DEPLOY.md)
# ============================================================
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery
from config import BQ_TABLE, BQ_LOCATION, CACHE_TTL


@st.cache_data(ttl=CACHE_TTL)
def load_bq_data() -> tuple:
    """
    โหลดข้อมูลจาก BigQuery
    Returns: (DataFrame | None, error_message | None)
    """
    try:
        creds_info = st.secrets["connections"]["bigquery"]["service_account_info"]

        scopes = [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/bigquery",
        ]
        credentials = service_account.Credentials.from_service_account_info(
            creds_info, scopes=scopes
        )
        client = bigquery.Client(
            credentials=credentials,
            project=creds_info["project_id"],
            location=BQ_LOCATION,
        )
        df = client.query(f"SELECT * FROM `{BQ_TABLE}`").to_dataframe()
        return df, None

    except Exception as e:
        return None, str(e)
