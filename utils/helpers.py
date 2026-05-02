# ============================================================
# utils/helpers.py — ฟังก์ชันใช้ร่วมกันทุกหน้า
# ============================================================
import pandas as pd
import numpy as np


# ── Matrix group labels (ใช้ร่วมกันทุกหน้า) ─────────────────
MATRIX_GROUPS = {
    "urgent":  "🚨 ด่วน (AI+Rule เห็นตรง)",
    "early":   "🔍 Early Warning (AI เห็นก่อน)",
    "monitor": "⚠️ Monitor (Rule เห็น, AI ยังให้โอกาส)",
    "active":  "✅ Active",
}


# Status values ที่ใช้จริงใน df['status'] (ภาษาอังกฤษเสมอ)
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


def safe_cats(df: pd.DataFrame, col: str = "product_category_name") -> list:
    """คืน list หมวดสินค้าที่ไม่ null เรียงตามตัวอักษร"""
    if col not in df.columns:
        return []
    return sorted([x for x in df[col].unique() if pd.notna(x)])


def assign_matrix_group(df: pd.DataFrame, threshold: float = 0.55) -> pd.DataFrame:
    """
    เพิ่ม column 'matrix_group' โดยแบ่งตาม AI × Rule-based
      - urgent  : AI churn AND lateness >= 1.5  (ทั้งคู่เห็นตรงกัน)
      - early   : AI churn AND lateness < 1.5   (AI เห็นก่อน rule)
      - monitor : AI ไม่ churn AND lateness >= 1.5 (rule เห็น, AI ยังให้โอกาส)
      - active  : ทั้งคู่ไม่เห็น
    """
    ai_on   = df["churn_probability"] >= threshold
    rule_on = df["lateness_score"]    >= 1.5

    conditions = [
        ai_on  &  rule_on,
        ai_on  & ~rule_on,
        ~ai_on &  rule_on,
    ]
    choices = [
        MATRIX_GROUPS["urgent"],
        MATRIX_GROUPS["early"],
        MATRIX_GROUPS["monitor"],
    ]
    df["matrix_group"] = np.select(conditions, choices, default=MATRIX_GROUPS["active"])
    return df


def status_display_options(t_func) -> tuple[list, dict, dict]:
    """
    สร้าง label ที่แปลแล้ว และ dict สำหรับ map กลับ
    Returns:
        display_list  — list label ที่ใช้ใน multiselect
        to_internal   — { display_label: internal_status }
        to_display    — { internal_status: display_label }
    """
    to_display  = {s: t_func(STATUS_TO_I18N_KEY[s]) for s in ALL_STATUSES}
    to_internal = {v: k for k, v in to_display.items()}
    display_list = list(to_display.values())
    return display_list, to_internal, to_display

MATRIX_GROUPS    # dict ชื่อ 4 กลุ่ม
MATRIX_GROUP_LIST  # list สำหรับ multiselect
assign_matrix_group(df, threshold)  # ฟังก์ชันสร้างคอลัมน์ matrix_group
