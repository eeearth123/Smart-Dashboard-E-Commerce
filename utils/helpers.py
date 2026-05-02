# ============================================================
# utils/helpers.py — ฟังก์ชันใช้ร่วมกันทุกหน้า
# ============================================================
import pandas as pd
import numpy as np

# ── Status values ที่ใช้จริงใน df['status'] ─────────────────
ALL_STATUSES = [
    "High Risk",
    "Warning (Late > 1.5x)",
    "Medium Risk",
    "Lost (Late > 3x)",
    "Active",
]

# Map: internal status → translation key ใน i18n
STATUS_TO_I18N_KEY = {
    "High Risk":              "status_high",
    "Warning (Late > 1.5x)": "status_warning",
    "Medium Risk":            "status_medium",
    "Lost (Late > 3x)":      "status_lost",
    "Active":                 "status_active",
}

# ── 2×2 Matrix Groups (AI × Rule-based) ──────────────────────
# ใช้ร่วมกันใน p2_churn.py และ p3_action.py
MATRIX_GROUPS = {
    "urgent":  "🚨 ด่วน (AI+Rule เห็นตรง)",
    "early":   "🔍 Early Warning (AI เห็นก่อน)",
    "monitor": "⚠️ Monitor (Rule เห็น, AI ยังให้โอกาส)",
    "active":  "✅ Active",
}
MATRIX_GROUP_LIST = list(MATRIX_GROUPS.values())


# ── ฟังก์ชันทั่วไป ─────────────────────────────────────────

def safe_cats(df: pd.DataFrame, col: str = "product_category_name") -> list:
    """คืน list หมวดสินค้าที่ไม่ null เรียงตามตัวอักษร"""
    if col not in df.columns:
        return []
    return sorted([x for x in df[col].unique() if pd.notna(x)])


def status_display_options(t_func) -> tuple[list, dict, dict]:
    """
    สร้าง label ที่แปลแล้ว และ dict สําหรับ map กลับ
    Returns:
        display_list  — list label ที่ใช้ใน multiselect
        to_internal   — { display_label: internal_status }
        to_display    — { internal_status: display_label }
    """
    to_display   = {s: t_func(STATUS_TO_I18N_KEY[s]) for s in ALL_STATUSES}
    to_internal  = {v: k for k, v in to_display.items()}
    display_list = list(to_display.values())
    return display_list, to_internal, to_display


def assign_matrix_group(df: pd.DataFrame, threshold: float = 0.55) -> pd.DataFrame:
    """
    เพิ่มคอลัมน์ matrix_group ให้ df

    Logic (2×2 AI × Rule):
        ai_churn  = churn_probability >= threshold
        rule_late = lateness_score >= 1.5

        ai  & rule  → urgent   (ทั้งสองเห็นตรง → ด่วนที่สุด)
        ai  & ~rule → early    (AI จับก่อน Rule ยังไม่เห็น)
        ~ai & rule  → monitor  (Rule เห็นแต่ AI ยังให้โอกาส)
        ~ai & ~rule → active   (ปกติดี)
    """
    ai_churn  = df["churn_probability"] >= threshold
    rule_late = df["lateness_score"] >= 1.5

    conditions = [
        ai_churn  &  rule_late,
        ai_churn  & ~rule_late,
        ~ai_churn &  rule_late,
        ~ai_churn & ~rule_late,
    ]
    choices = list(MATRIX_GROUPS.values())
    df["matrix_group"] = np.select(conditions, choices, default=choices[-1])
    return df
