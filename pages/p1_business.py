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

    # ── KPI row: Customer counts only ─────────────────────────
    n_total  = dfd["customer_unique_id"].nunique() if "customer_unique_id" in dfd.columns else 0
    n_repeat = 0
    if "customer_unique_id" in dfd.columns and "purchase_count" in dfd.columns:
        n_repeat = dfd[dfd["purchase_count"] >= 2]["customer_unique_id"].nunique()
    elif "customer_unique_id" in dfd.columns:
        counts   = dfd.groupby("customer_unique_id").size()
        n_repeat = (counts >= 2).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("👥 " + t("p1_total_cust"),  f"{n_total:,} " + t("p1_unit_ppl"))
    c2.metric("🔄 " + t("p1_repeat_cust"), f"{n_repeat:,} " + t("p1_unit_ppl"))
    c3.metric("🛒 " + t("p1_onetime_cust"), f"{n_total - n_repeat:,} " + t("p1_unit_ppl"))

    st.markdown("---")

    # ── Monthly Repeat vs One-time Customer Trend ─────────────
    st.subheader(t("p1_trend"))
    _render_customer_trend(dfd)
    st.markdown("---")

    # ── Top categories by customer count (repeat vs first) ────
    st.subheader(t("p1_top_cat"))
    _render_top_categories(dfd)


def _render_customer_trend(dfd):
    """Bar chart for one-time buyers + line chart for repeat buyers,
    with MoM % change in tooltip, dual Y-axis."""
    if "order_purchase_timestamp" not in dfd.columns or dfd.empty:
        st.info(t("no_data"))
        return

    tmp = dfd.copy()
    tmp["_month"] = tmp["order_purchase_timestamp"].dt.to_period("M")

    # Identify repeat buyers: customers with purchase_count >= 2
    if "purchase_count" in tmp.columns:
        tmp["_is_repeat"] = (tmp["purchase_count"] >= 2).astype(int)
    else:
        cust_counts = tmp.groupby("customer_unique_id")["_month"].transform("count")
        tmp["_is_repeat"] = (cust_counts >= 2).astype(int)

    # Count unique customers per month per type
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

    # Convert period to timestamp for Altair
    trend["month_ts"] = trend["month"].apply(lambda p: p.to_timestamp())

    # Remove last month if incomplete
    if len(trend) > 1:
        trend = trend.iloc[:-1]

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

    # ── Build dual-axis chart ──
    base = alt.Chart(trend).encode(
        x=alt.X("month_ts:T", axis=alt.Axis(format="%b %Y", labelAngle=-45, title=""))
    )

    # Bars: One-time buyers (left axis)
    bars = base.mark_bar(
        color="#B0BEC5", opacity=0.7, cornerRadiusTopLeft=3, cornerRadiusTopRight=3
    ).encode(
        y=alt.Y(
            "onetime_customers:Q",
            title=t("p1_onetime_axis"),
            axis=alt.Axis(grid=False),
        ),
        tooltip=[
            alt.Tooltip("month_ts:T", format="%B %Y", title=t("p1_tt_month")),
            alt.Tooltip("onetime_customers:Q", format=",.0f", title=t("p1_onetime_cust")),
            alt.Tooltip("onetime_mom_pct:Q", format="+.1f", title=t("p1_tt_mom")),
        ],
    )

    # Line: Repeat buyers (right axis)
    line = base.mark_line(
        color="#FF7043", strokeWidth=3,
        point=alt.OverlayMarkDef(color="#FF7043", size=60),
    ).encode(
        y=alt.Y(
            "repeat_customers:Q",
            title=t("p1_repeat_axis"),
            axis=alt.Axis(titleColor="#FF7043", orient="right"),
        ),
        tooltip=[
            alt.Tooltip("month_ts:T", format="%B %Y", title=t("p1_tt_month")),
            alt.Tooltip("repeat_customers:Q", format=",.0f", title=t("p1_repeat_cust")),
            alt.Tooltip("repeat_mom_pct:Q", format="+.1f", title=t("p1_tt_mom")),
        ],
    )

    chart = (
        alt.layer(bars, line)
        .resolve_scale(y="independent")
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)

    # ── Legend caption ──
    st.caption(
        "🟧 " + t("p1_legend_bar") + "　　"
        "🟠 " + t("p1_legend_line")
    )


def _render_top_categories(dfd):
    """Horizontal stacked bar chart: customer count by category,
    split into Repeat (color A) vs First-time (color B)."""
    if "product_category_name" not in dfd.columns or dfd.empty:
        return

    tmp = dfd.copy()

    # Classify each customer-category pair as repeat or first-time
    if "purchase_count" in tmp.columns:
        tmp["_buyer_type"] = np.where(
            tmp["purchase_count"] >= 2,
            t("p1_repeat_cust"),
            t("p1_onetime_cust"),
        )
    else:
        cust_counts = tmp.groupby("customer_unique_id").size()
        repeat_ids  = set(cust_counts[cust_counts >= 2].index)
        tmp["_buyer_type"] = np.where(
            tmp["customer_unique_id"].isin(repeat_ids),
            t("p1_repeat_cust"),
            t("p1_onetime_cust"),
        )

    # Aggregate: unique customers per category per buyer type
    cat_cust = (
        tmp.groupby(["product_category_name", "_buyer_type"])["customer_unique_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_unique_id": "customer_count"})
    )

    # Top 20 categories by total customer count
    top_cats = (
        cat_cust.groupby("product_category_name")["customer_count"]
        .sum()
        .nlargest(20)
        .index
        .tolist()
    )
    cat_cust = cat_cust[cat_cust["product_category_name"].isin(top_cats)]

    col_chart, col_table = st.columns([1.5, 2])

    with col_chart:
        chart = (
            alt.Chart(cat_cust)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("customer_count:Q", title=t("p1_cust_count_axis")),
                y=alt.Y(
                    "product_category_name:N",
                    sort=alt.EncodingSortField(
                        field="customer_count", op="sum", order="descending"
                    ),
                    title=None,
                ),
                color=alt.Color(
                    "_buyer_type:N",
                    scale=alt.Scale(
                        domain=[t("p1_repeat_cust"), t("p1_onetime_cust")],
                        range=["#FF7043", "#78909C"],
                    ),
                    title=t("p1_legend_type"),
                ),
                tooltip=[
                    alt.Tooltip("product_category_name:N", title=t("p1_col_cat")),
                    alt.Tooltip("_buyer_type:N", title=t("p1_legend_type")),
                    alt.Tooltip("customer_count:Q", format=",", title=t("p1_cust_count_axis")),
                ],
            )
            .properties(height=550, title=t("p1_chart_title"))
        )
        st.altair_chart(chart, use_container_width=True)

    with col_table:
        # Pivot for table display
        pivot = cat_cust.pivot_table(
            index="product_category_name",
            columns="_buyer_type",
            values="customer_count",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()

        # Ensure both columns exist
        repeat_label  = t("p1_repeat_cust")
        onetime_label = t("p1_onetime_cust")
        if repeat_label not in pivot.columns:
            pivot[repeat_label] = 0
        if onetime_label not in pivot.columns:
            pivot[onetime_label] = 0

        pivot["total"] = pivot[repeat_label] + pivot[onetime_label]
        pivot["repeat_pct"] = np.where(
            pivot["total"] > 0,
            pivot[repeat_label] / pivot["total"] * 100,
            0,
        )
        pivot = pivot.sort_values("total", ascending=False)
        pivot = pivot.rename(columns={
            "product_category_name": t("p1_col_cat"),
            repeat_label:  "🔄 " + repeat_label,
            onetime_label: "🛒 " + onetime_label,
            "total":       "👥 " + t("p1_total_cust"),
            "repeat_pct":  "🔄 %",
        })

        st.markdown(t("p1_table_hdr"))
        st.dataframe(
            pivot,
            column_config={
                "🔄 %": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100
                ),
            },
            use_container_width=True,
            hide_index=True,
            height=550,
        )
