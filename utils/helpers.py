# ============================================================
# utils/helpers.py — ฟังก์ชันใช้ร่วมกันทุกหน้า
# ============================================================
import pandas as pd


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
