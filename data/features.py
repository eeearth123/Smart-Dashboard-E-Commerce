# ============================================================
# data/features.py — Feature Engineering ทั้งหมด
# ============================================================
import numpy as np
import pandas as pd
from config import LATE_LOST, LATE_WARNING, PROB_HIGH, PROB_MEDIUM


def process_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """แปลงข้อมูลดิบเป็น features พร้อม predict"""
    df = df_raw.copy()

    # Convert potential decimal columns to float
    for col in ["price", "freight_value", "payment_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # ── Parse timestamps ─────────────────────────────────────
    for col in [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "order_purchase_timestamp" in df.columns:
        df = df.sort_values(
            ["customer_unique_id", "order_purchase_timestamp"]
        ).reset_index(drop=True)

    # ── Logistics features ────────────────────────────────────
    if "order_delivered_customer_date" in df.columns and "order_purchase_timestamp" in df.columns:
        df["delivery_days"] = (
            df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
        ).dt.days.clip(lower=0)
    else:
        df["delivery_days"] = np.nan

    if "order_estimated_delivery_date" in df.columns and "order_purchase_timestamp" in df.columns:
        df["estimated_days"] = (
            df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
        ).dt.days
    else:
        df["estimated_days"] = np.nan

    df["delivery_vs_estimated"] = df["estimated_days"] - df["delivery_days"]

    # ── Price & Freight ───────────────────────────────────────
    if "freight_value" in df.columns and "price" in df.columns:
        df["freight_ratio"] = np.where(df["price"] > 0, df["freight_value"] / df["price"], 0)
        df["payment_value"] = df["price"] + df["freight_value"]
    else:
        df["freight_ratio"] = 0
        df["payment_value"] = df.get("price", 0)

    # ── Payment features ──────────────────────────────────────
    df["uses_multiple_payments"] = (
        (df["payment_sequential"].fillna(1) > 1).astype(int)
        if "payment_sequential" in df.columns
        else 0
    )
    df["uses_voucher"] = (
        (df["payment_type"].fillna("") == "voucher").astype(int)
        if "payment_type" in df.columns
        else 0
    )

    # ── Review score ──────────────────────────────────────────
    if "review_score" in df.columns:
        df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
        df["is_low_score"]  = (df["review_score"].fillna(3) <= 2).astype(int)
        df["is_high_score"] = (df["review_score"].fillna(3) == 5).astype(int)
    else:
        df["review_score"]  = 3.0
        df["is_low_score"]  = 0
        df["is_high_score"] = 0

    # ── Purchase count & repeat buyer ────────────────────────
    df["purchase_count"]    = df.groupby("customer_unique_id").cumcount() + 1
    df["is_first_purchase"] = (df["purchase_count"] == 1).astype(int)
    df["is_repeat_buyer"]   = (df["purchase_count"] >= 2).astype(int)

    # ── Gap features ──────────────────────────────────────────
    if "order_purchase_timestamp" in df.columns:
        df["prev_purchase_date"] = df.groupby("customer_unique_id")[
            "order_purchase_timestamp"
        ].shift(1)
        df["days_since_last_purchase"] = (
            df["order_purchase_timestamp"] - df["prev_purchase_date"]
        ).dt.days

        median_gap = df.loc[df["is_repeat_buyer"] == 1, "days_since_last_purchase"].median()
        if pd.isna(median_gap):
            median_gap = 90.0

        df["avg_purchase_gap"] = df.groupby("customer_unique_id")[
            "days_since_last_purchase"
        ].transform("mean")
        global_avg = df["avg_purchase_gap"].median()
        df["avg_purchase_gap"]         = df["avg_purchase_gap"].fillna(global_avg)
        df["gap_vs_avg"]               = df["avg_purchase_gap"] - df["days_since_last_purchase"]
        df["days_since_last_purchase"] = df["days_since_last_purchase"].fillna(median_gap)
        df["gap_vs_avg"]               = df["gap_vs_avg"].fillna(0)
        df["gap_real"]                 = np.where(df["is_repeat_buyer"] == 1, df["days_since_last_purchase"], 0)
        df["gap_vs_avg_real"]          = np.where(df["is_repeat_buyer"] == 1, df["gap_vs_avg"], 0)
    else:
        for c in ["days_since_last_purchase", "avg_purchase_gap", "gap_vs_avg", "gap_real", "gap_vs_avg_real"]:
            df[c] = 0

    # ── Category churn risk placeholder ──────────────────────
    if "cat_churn_risk" not in df.columns:
        df["cat_churn_risk"] = 0.80

    # ── Lateness score ────────────────────────────────────────
    if "order_purchase_timestamp" in df.columns:
        ref_date   = df["order_purchase_timestamp"].max()
        last_order = df.groupby("customer_unique_id")["order_purchase_timestamp"].transform("max")
        df["days_since_purchase"] = (ref_date - last_order).dt.days

        tmp = df.sort_values(["customer_unique_id", "product_category_name", "order_purchase_timestamp"])
        tmp["prev_ts"] = tmp.groupby(["customer_unique_id", "product_category_name"])[
            "order_purchase_timestamp"
        ].shift(1)
        tmp["order_gap"] = (tmp["order_purchase_timestamp"] - tmp["prev_ts"]).dt.days
        valid_gaps = tmp[(tmp["order_gap"] >= 7) & (tmp["order_gap"] <= 730)]

        if len(valid_gaps) > 10:
            cat_med = valid_gaps.groupby("product_category_name")["order_gap"].median().rename("cat_median_days")
            df = df.merge(cat_med, on="product_category_name", how="left")
        else:
            df["cat_median_days"] = 180

        df["cat_median_days"] = df["cat_median_days"].fillna(180).clip(lower=7)
        df["lateness_score"]  = (df["days_since_purchase"] / df["cat_median_days"]).clip(lower=0)
    else:
        df["days_since_purchase"] = 90
        df["cat_median_days"]     = 180
        df["lateness_score"]      = 0.5

    # ── Delay days ────────────────────────────────────────────
    if "order_delivered_customer_date" in df.columns and "order_estimated_delivery_date" in df.columns:
        df["delay_days"] = (
            df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
        ).dt.days.fillna(0)
    else:
        df["delay_days"] = 0

    # ── Bad experience score (V2) ─────────────────────────────
    df["first_purchase_late"] = np.where((df["is_first_purchase"] == 1) & (df["delivery_vs_estimated"] < 0), 1, 0)
    df["first_purchase_bad_review"] = np.where((df["is_first_purchase"] == 1) & (df["review_score"] <= 2), 1, 0)
    df["is_extremely_late"] = (df["delay_days"] > 7).astype(int)

    df["bad_experience_score"] = (
        df["is_low_score"] +
        df["first_purchase_bad_review"] +
        df["first_purchase_late"] +
        df["is_extremely_late"]
    )

    return df


def assign_status(row: pd.Series) -> str:
    """กำหนดสถานะลูกค้าตาม 3-Class probabilities และ business rules"""
    prob_churn = row.get("prob_churn", 0)
    prob_delay = row.get("prob_delay", 0)
    pred_class = row.get("churn_prediction", 0)
    late = row.get("lateness_score", 0)

    # 1. Rules based on explicit late thresholds
    if late > LATE_LOST:
        return "Lost (Late > 3x)"
    
    # 2. Model predictions for Churn
    if pred_class == 2 or prob_churn > PROB_HIGH:
        return "High Risk"
    
    # 3. Model predictions for Delay/Warning
    if pred_class == 1 or prob_delay > PROB_HIGH or late > LATE_WARNING:
        return "Warning (Late > 1.5x)"
    
    # 4. Borderline Churn
    if prob_churn >= PROB_MEDIUM:
        return "Medium Risk"
    
    return "Active"
