# ============================================================
# config.py — Central settings (แก้ที่นี่ที่เดียวพอ)
# ============================================================

# Model
BEST_THRESHOLD = 0.12

# BigQuery
BQ_TABLE   = "academic-moon-483615-t2.analytics_olist.mart_churn_features"
BQ_LOCATION = "asia-southeast1"

# Business rules — lateness thresholds
LATE_LOST    = 3.0
LATE_WARNING = 1.5

# Churn probability thresholds
PROB_HIGH   = 0.15
PROB_MEDIUM = 0.10

# Model filenames (ต้องอยู่ในโฟลเดอร์เดียวกับ app.py)
MODEL_FILE    = "modelV5.pkl"
FEATURES_FILE = "model_features_V5.pkl"

# Cache TTL (seconds)
CACHE_TTL = 600
