# ============================================================
# config.py — Central settings (แก้ที่นี่ที่เดียวพอ)
# ============================================================

# Model
BEST_THRESHOLD = 0.55

# BigQuery
BQ_TABLE   = "academic-moon-483615-t2.Dashboard.input"
BQ_LOCATION = "asia-southeast1"

# Business rules — lateness thresholds
LATE_LOST    = 3.0
LATE_WARNING = 1.5

# Churn probability thresholds
PROB_HIGH   = 0.75
PROB_MEDIUM = 0.40

# Model filenames (ต้องอยู่ในโฟลเดอร์เดียวกับ app.py)
MODEL_FILE    = "olist_churn_model_final (1).pkl"
FEATURES_FILE = "model_features_final (1).pkl"

# Cache TTL (seconds)
CACHE_TTL = 600
