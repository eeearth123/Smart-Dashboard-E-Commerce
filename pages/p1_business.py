# ============================================================
# pages/p1_business.py — Business Overview
# ============================================================
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from i18n import t
from utils.helpers import safe_cats


def render(df: pd.DataFrame) -> None:
    st.title(t("page_business"))
    st.caption(t("p1_caption"))

    with st.expander(t("filter_expand"), expanded=False):
        sel_cats = st.multiselect(t("cat_label"), safe_cats(df), key="p1_cat")

    dfd = df[df["product_category_name"].isin(sel_cats)].copy() if sel_cats else df.copy()
    st.markdown("---")

    # ── KPI row: Customer counts + AOV ────────────────────────
    n_total  = dfd["customer_unique_id"].nunique() if "customer_unique_id" in dfd.columns else 0
    n_repeat = 0
    if "customer_unique_id" in dfd.columns and "purchase_count" in dfd.columns:
        n_repeat = dfd[dfd["purchase_count"] >= 2]["customer_unique_id"].nunique()
    elif "customer_unique_id" in dfd.columns:
        counts   = dfd.groupby("customer_unique_id").size()
        n_repeat = (counts >= 2).sum()

    avg_order = dfd["payment_value"].mean() if "payment_value" in dfd.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 " + t("p1_total_cust"),   f"{n_total:,} " + t("p1_unit_ppl"))
    c2.metric("🔄 " + t("p1_repeat_cust"),  f"{n_repeat:,} " + t("p1_unit_ppl"))
    c3.metric("🛒 " + t("p1_onetime_cust"), f"{n_total - n_repeat:,} " + t("p1_unit_ppl"))
    c4.metric("💰 Avg Order Value",          f"R$ {avg_order:,.0f}")

    st.markdown("---")

    # ── Monthly Customer Trend (2 stacked charts) ─────────────
    st.subheader(t("p1_trend"))
    _render_customer_trend(dfd)
    st.markdown("---")

    # ── Top categories (Original Revenue & Churn Risk) ────────
    st.subheader(t("p1_top_cat"))
    _render_top_categories(dfd)


def _render_customer_trend(dfd: pd.DataFrame) -> None:
    """Two separate charts stacked vertically:
    1. Repeat Buyers Monthly Trend
    2. New / One-Time Buyers Monthly Trend
    Excludes incomplete trailing months (Sep & Oct 2018)."""
    if "order_purchase_timestamp" not in dfd.columns or dfd.empty:
        st.info(t("no_data"))
        return

    tmp = dfd.copy()
    tmp["_month"] = tmp["order_purchase_timestamp"].dt.to_period("M")

    # Identify repeat buyers
    if "purchase_count" in tmp.columns:
        tmp["_is_repeat"] = (tmp["purchase_count"] >= 2).astype(int)
    else:
        cust_counts = tmp.groupby("customer_unique_id")["_month"].transform("count")
        tmp["_is_repeat"] = (cust_counts >= 2).astype(int)

    monthly_repeat = (
        tmp[tmp["_is_repeat"] == 1]
        .groupby("_month")["customer_unique_id"]
        .nunique()
        .rename("repeat_customers")
    )
    monthly_onetime = (
        tmp[tmp["_is_repeat"] == 0]
        .groupby("_month")["customer_unique_id"]
        .nunique()
        .rename("onetime_customers")
    )

    all_months = pd.period_range(
        start=tmp["_month"].min(), end=tmp["_month"].max(), freq="M"
    )
    trend = pd.DataFrame({"month": all_months})
    trend = trend.merge(
        monthly_repeat.reset_index().rename(columns={"_month": "month"}),
        on="month", how="left",
    )
    trend = trend.merge(
        monthly_onetime.reset_index().rename(columns={"_month": "month"}),
        on="month", how="left",
    )
    trend = trend.fillna(0)

    # Filter out incomplete months after 2018-08 (Sep 2018 has 16 orders, Oct 2018 has 4 orders)
    trend = trend[trend["month"] <= "2018-08"].copy()

    # Convert period to timestamp for Altair
    trend["month_ts"] = trend["month"].apply(lambda p: p.to_timestamp())

    # Calculate MoM % change
    trend["repeat_mom_pct"] = (
        trend["repeat_customers"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan) * 100
    )
    trend["onetime_mom_pct"] = (
        trend["onetime_customers"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan) * 100
    )

    # ── Chart 1 (Top): Repeat Buyers Trend ──
    st.markdown("#### " + t("p1_trend_repeat"))
    base1 = alt.Chart(trend).encode(
        x=alt.X("month_ts:T", axis=alt.Axis(format="%b %Y", labelAngle=-45, title=""))
    )
    bars1 = base1.mark_bar(
        color="#FF7043", opacity=0.7, cornerRadiusTopLeft=3, cornerRadiusTopRight=3
    ).encode(
        y=alt.Y("repeat_customers:Q", title=t("p1_repeat_axis")),
        tooltip=[
            alt.Tooltip("month_ts:T", format="%B %Y", title=t("p1_tt_month")),
            alt.Tooltip("repeat_customers:Q", format=",.0f", title=t("p1_repeat_cust")),
            alt.Tooltip("repeat_mom_pct:Q", format="+.1f", title=t("p1_tt_mom")),
        ],
    )
    line1 = base1.mark_line(
        color="#E53935", strokeWidth=2.5,
        point=alt.OverlayMarkDef(color="#E53935", size=45),
    ).encode(
        y=alt.Y("repeat_customers:Q"),
        tooltip=[
            alt.Tooltip("month_ts:T", format="%B %Y", title=t("p1_tt_month")),
            alt.Tooltip("repeat_customers:Q", format=",.0f", title=t("p1_repeat_cust")),
            alt.Tooltip("repeat_mom_pct:Q", format="+.1f", title=t("p1_tt_mom")),
        ],
    )
    st.altair_chart(
        alt.layer(bars1, line1).properties(height=300),
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart 2 (Bottom): New / One-Time Buyers Trend ──
    st.markdown("#### " + t("p1_trend_onetime"))
    base2 = alt.Chart(trend).encode(
        x=alt.X("month_ts:T", axis=alt.Axis(format="%b %Y", labelAngle=-45, title=""))
    )
    bars2 = base2.mark_bar(
        color="#1E88E5", opacity=0.75, cornerRadiusTopLeft=3, cornerRadiusTopRight=3
    ).encode(
        y=alt.Y("onetime_customers:Q", title=t("p1_onetime_axis")),
        tooltip=[
            alt.Tooltip("month_ts:T", format="%B %Y", title=t("p1_tt_month")),
            alt.Tooltip("onetime_customers:Q", format=",.0f", title=t("p1_onetime_cust")),
            alt.Tooltip("onetime_mom_pct:Q", format="+.1f", title=t("p1_tt_mom")),
        ],
    )
    line2 = base2.mark_line(
        color="#1565C0", strokeWidth=2.5,
        point=alt.OverlayMarkDef(color="#1565C0", size=45),
    ).encode(
        y=alt.Y("onetime_customers:Q"),
        tooltip=[
            alt.Tooltip("month_ts:T", format="%B %Y", title=t("p1_tt_month")),
            alt.Tooltip("onetime_customers:Q", format=",.0f", title=t("p1_onetime_cust")),
            alt.Tooltip("onetime_mom_pct:Q", format="+.1f", title=t("p1_tt_mom")),
        ],
    )
    st.altair_chart(
        alt.layer(bars2, line2).properties(height=300),
        use_container_width=True,
    )


def _render_top_categories(dfd: pd.DataFrame) -> None:
    """Original Top Categories implementation: Revenue (R$), Orders, Avg Order, Churn Risk."""
    if "product_category_name" not in dfd.columns or dfd.empty:
        return

    cat_sales = (
        dfd.groupby("product_category_name")
        .agg(
            revenue=("payment_value", "sum"),
            orders=("payment_value", "count"),
            avg_order=("payment_value", "mean"),
            churn_risk=("churn_probability", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    col_chart, col_table = st.columns([1.5, 2])

    with col_chart:
        top20 = cat_sales.head(20)
        chart = (
            alt.Chart(top20).mark_bar()
            .encode(
                x=alt.X("revenue:Q", title="Revenue (R$)"),
                y=alt.Y("product_category_name:N", sort="-x", title=None),
                color=alt.Color(
                    "churn_risk:Q",
                    scale=alt.Scale(domain=[0.3, 0.9], range=["#2ecc71", "#e74c3c"]),
                    title=t("p1_col_churn"),
                ),
                tooltip=[
                    alt.Tooltip("product_category_name", title=t("p1_col_cat")),
                    alt.Tooltip("revenue", format=",.0f", title="Revenue (R$)"),
                    alt.Tooltip("orders", format=","),
                    alt.Tooltip("churn_risk", format=".1%", title=t("p1_col_churn")),
                ],
            )
            .properties(height=500, title=t("p1_chart_title"))
        )
        st.altair_chart(chart, use_container_width=True)

    with col_table:
        st.markdown(t("p1_table_hdr"))
        st.dataframe(
            cat_sales.rename(columns={
                "product_category_name": t("p1_col_cat"),
                "revenue":               t("p1_col_rev"),
                "orders":                t("p1_col_orders"),
                "avg_order":             t("p1_col_avg"),
                "churn_risk":            t("p1_col_churn"),
            }),
            column_config={
                t("p1_col_rev"):    st.column_config.NumberColumn(format="R$ %.0f"),
                t("p1_col_orders"): st.column_config.NumberColumn(format="%,d"),
                t("p1_col_avg"):    st.column_config.NumberColumn(format="R$ %.0f"),
                t("p1_col_churn"):  st.column_config.ProgressColumn(
                    format="%.2f", min_value=0, max_value=1
                ),
            },
            use_container_width=True,
            hide_index=True,
            height=500,
        )
