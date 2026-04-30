# ============================================================
# pages/p6_seller.py — Seller Audit
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
from i18n import t
from utils.helpers import safe_cats, status_display_options


def render(df: pd.DataFrame) -> None:
    st.title(t("page_seller"))

    if "seller_id" not in df.columns:
        st.error(t("p6_no_seller"))
        return

    # ── Filters ───────────────────────────────────────────────
    display_list, to_internal, _ = status_display_options(t)
    c1, c2 = st.columns(2)
    with c1:
        sel_c = st.multiselect(t("cat_label"), safe_cats(df), key="p6_cat")
    with c2:
        sel_s_display = st.multiselect(t("status_label"), display_list, key="p6_status")
    sel_s = [to_internal[s] for s in sel_s_display if s in to_internal]

    dfs = df.copy()
    if sel_c: dfs = dfs[dfs["product_category_name"].isin(sel_c)]
    if sel_s: dfs = dfs[dfs["status"].isin(sel_s)]

    # ── Aggregate by seller ───────────────────────────────────
    agg = {
        "order_purchase_timestamp": "count",
        "payment_value":            "sum",
        "churn_probability":        "mean",
        "delivery_days":            "mean",
    }
    if "review_score" in dfs.columns:
        agg["review_score"] = "mean"

    ss = dfs.groupby("seller_id").agg(agg).reset_index()
    ss = ss.rename(columns={"order_purchase_timestamp": "orders"})
    if "review_score" not in ss.columns:
        ss["review_score"] = np.nan
    ss = ss[ss["orders"] >= 3]

    # ── KPIs ──────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("p6_metric_shops"),  f"{len(ss):,}")
    k2.metric(t("p6_metric_rev"),    f"R$ {ss['payment_value'].sum():,.0f}")
    k3.metric(t("p6_metric_review"),
              f"{ss['review_score'].mean():.2f}" if ss["review_score"].notna().any() else t("na"))
    k4.metric(t("p6_metric_del"),    f"{ss['delivery_days'].mean():.1f}{t('days_unit')}")

    st.markdown("---")

    # ── Sort + Table ──────────────────────────────────────────
    sort_opts = [t("p6_sort_risk"), t("p6_sort_late"), t("p6_sort_score"), t("p6_sort_rev"), t("p6_sort_vol")]
    cs_, cd_ = st.columns([1, 3])
    with cs_:
        sort_m = st.radio(t("p6_sort"), sort_opts)
    with cd_:
        sort_map = {
            t("p6_sort_risk"):  ("churn_probability", False),
            t("p6_sort_late"):  ("delivery_days",     False),
            t("p6_sort_score"): ("review_score",      True),
            t("p6_sort_rev"):   ("payment_value",     False),
            t("p6_sort_vol"):   ("orders",            False),
        }
        col, asc = sort_map.get(sort_m, ("churn_probability", False))
        sdf = ss.sort_values(col, ascending=asc)

        st.dataframe(
            sdf,
            column_config={
                "orders":            st.column_config.NumberColumn("Orders"),
                "payment_value":     st.column_config.NumberColumn("Revenue", format="R$%.0f"),
                "delivery_days":     st.column_config.NumberColumn(t("p6_col_del"), format="%.1f"),
                "review_score":      st.column_config.NumberColumn("Review", format="%.1f⭐"),
                "churn_probability": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
            },
            hide_index=True,
            use_container_width=True,
            height=600,
        )
