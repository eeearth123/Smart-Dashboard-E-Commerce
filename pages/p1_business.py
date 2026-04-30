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

    # ── KPI metrics ───────────────────────────────────────────
    total_rev   = dfd["payment_value"].sum() if "payment_value" in dfd.columns else 0
    avg_order   = dfd["payment_value"].mean() if "payment_value" in dfd.columns else 0
    n_customers = dfd["customer_unique_id"].nunique() if "customer_unique_id" in dfd.columns else 0

    mom_growth = _calc_mom_growth(dfd)

    k1, k2, k3 = st.columns(3)
    k1.metric(t("p1_rev"), f"R$ {total_rev:,.0f}")
    k2.metric(
        t("p1_mom"),
        f"{mom_growth:+.1f}%" if mom_growth is not None else t("na"),
        delta=f"{mom_growth:+.1f}%" if mom_growth is not None else None,
    )
    k3.metric(t("p1_aov"), f"R$ {avg_order:,.0f}")
    st.markdown("---")

    # ── Revenue trend ─────────────────────────────────────────
    st.subheader(t("p1_trend"))
    _render_revenue_trend(dfd)
    st.markdown("---")

    # ── Top categories ────────────────────────────────────────
    st.subheader(t("p1_top_cat"))
    _render_top_categories(dfd)


# ── Private helpers ───────────────────────────────────────────

def _calc_mom_growth(dfd: pd.DataFrame):
    if "order_purchase_timestamp" not in dfd.columns or dfd.empty:
        return None
    dfd = dfd.copy()
    dfd["_month"]   = dfd["order_purchase_timestamp"].dt.to_period("M")
    all_months      = pd.period_range(start=dfd["_month"].min(), end=dfd["_month"].max(), freq="M")
    monthly_rev     = dfd.groupby("_month")["payment_value"].sum().reindex(all_months, fill_value=0)
    if len(monthly_rev) >= 3:
        last_m, prev_m = monthly_rev.iloc[-2], monthly_rev.iloc[-3]
        return (last_m - prev_m) / prev_m * 100 if prev_m > 0 else None
    if len(monthly_rev) == 2:
        last_m, first_m = monthly_rev.iloc[-1], monthly_rev.iloc[-2]
        return (last_m - first_m) / first_m * 100 if first_m > 0 else None
    return None


def _render_revenue_trend(dfd: pd.DataFrame) -> None:
    if "order_purchase_timestamp" not in dfd.columns or dfd.empty:
        st.info(t("no_data"))
        return

    rev_trend = (
        dfd.set_index("order_purchase_timestamp")["payment_value"]
        .resample("MS").sum().fillna(0).reset_index()
    )
    rev_trend.columns = ["Month", "Revenue"]
    rev_trend["Growth"] = rev_trend["Revenue"].pct_change().replace([np.inf, -np.inf], np.nan) * 100
    plot_df = rev_trend.iloc[:-1] if len(rev_trend) > 1 else rev_trend

    base = alt.Chart(plot_df).encode(
        x=alt.X("Month:T", axis=alt.Axis(format="%b %Y", labelAngle=-45, title=""))
    )
    bars = base.mark_bar(color="#1E88E5", opacity=0.7).encode(
        y=alt.Y("Revenue:Q", title="Revenue (R$)", axis=alt.Axis(grid=False)),
        tooltip=[alt.Tooltip("Month:T", format="%B %Y"), alt.Tooltip("Revenue:Q", format=",.0f")],
    )
    line = base.mark_line(color="#E53935", strokeWidth=3, point=alt.OverlayMarkDef(color="#E53935")).encode(
        y=alt.Y("Growth:Q", title="Growth (%)", axis=alt.Axis(titleColor="#E53935", orient="right")),
        tooltip=[alt.Tooltip("Month:T", format="%B %Y"), alt.Tooltip("Growth:Q", format=".1f", title="Growth %")],
    )
    st.altair_chart(
        alt.layer(bars, line).resolve_scale(y="independent").properties(height=350),
        use_container_width=True,
    )


def _render_top_categories(dfd: pd.DataFrame) -> None:
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
            alt.Chart(top20)
            .mark_bar()
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
                t("p1_col_churn"):  st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            },
            use_container_width=True,
            hide_index=True,
            height=500,
        )
