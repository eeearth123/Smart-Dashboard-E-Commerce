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

    # ── Monthly Customer Trend (2 separate charts) ────────────
    st.subheader(t("p1_trend"))
    _render_customer_trend(dfd)
    st.markdown("---")

    # ── Top categories by customer count (repeat vs first) ────
    st.subheader(t("p1_top_cat"))
    _render_top_categories(dfd)


def _render_customer_trend(dfd):
    """Two separate charts: bar for one-time buyers, bar+line for repeat buyers,
    with MoM % change in tooltip. Last incomplete month annotated."""
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

    # ── Detect last incomplete month ──
    last_month_period = trend["month"].iloc[-1]
    last_month_start  = last_month_period.to_timestamp()
    last_month_end    = last_month_period.to_timestamp(how="end")
    max_date          = tmp["order_purchase_timestamp"].max()

    # Days elapsed in the last month
    days_elapsed = (max_date - last_month_start).days + 1
    total_days   = (last_month_end - last_month_start).days + 1
    is_incomplete = days_elapsed < total_days

    # Mark the last month
    trend["is_incomplete"] = False
    if is_incomplete and len(trend) > 0:
        trend.loc[trend.index[-1], "is_incomplete"] = True

    # Separate complete vs incomplete for display
    trend_complete   = trend[~trend["is_incomplete"]]
    trend_incomplete = trend[trend["is_incomplete"]]

    # Calculate MoM % change (on full data for tooltip)
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

    # ── Incomplete month annotation ──
    # (silently excluded from charts)

    # Remove incomplete month for charting
    plot_df = trend[~trend["is_incomplete"]].copy()

    # Re-calculate MoM on clean data
    plot_df["repeat_mom_pct"] = (
        plot_df["repeat_customers"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan) * 100
    )
    plot_df["onetime_mom_pct"] = (
        plot_df["onetime_customers"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan) * 100
    )

    # ── Chart 1: One-time Buyers (Bar chart) ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🛒 " + t("p1_onetime_cust"))
        base1 = alt.Chart(plot_df).encode(
            x=alt.X("month_ts:T", axis=alt.Axis(format="%b %Y", labelAngle=-45, title=""))
        )
        bars1 = base1.mark_bar(
            color="#78909C", opacity=0.85,
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
        ).encode(
            y=alt.Y("onetime_customers:Q", title=t("p1_onetime_axis")),
            tooltip=[
                alt.Tooltip("month_ts:T", format="%B %Y", title=t("p1_tt_month")),
                alt.Tooltip("onetime_customers:Q", format=",.0f", title=t("p1_onetime_cust")),
                alt.Tooltip("onetime_mom_pct:Q", format="+.1f", title=t("p1_tt_mom")),
            ],
        )
        st.altair_chart(bars1.properties(height=320), use_container_width=True)

    # ── Chart 2: Repeat Buyers (Bar + Line) ──
    with col2:
        st.markdown("##### 🔄 " + t("p1_repeat_cust"))
        base2 = alt.Chart(plot_df).encode(
            x=alt.X("month_ts:T", axis=alt.Axis(format="%b %Y", labelAngle=-45, title=""))
        )
        bars2 = base2.mark_bar(
            color="#FF7043", opacity=0.6,
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
        ).encode(
            y=alt.Y("repeat_customers:Q", title=t("p1_repeat_axis")),
            tooltip=[
                alt.Tooltip("month_ts:T", format="%B %Y", title=t("p1_tt_month")),
                alt.Tooltip("repeat_customers:Q", format=",.0f", title=t("p1_repeat_cust")),
                alt.Tooltip("repeat_mom_pct:Q", format="+.1f", title=t("p1_tt_mom")),
            ],
        )
        line2 = base2.mark_line(
            color="#E53935", strokeWidth=2,
            point=alt.OverlayMarkDef(color="#E53935", size=40),
        ).encode(
            y=alt.Y("repeat_customers:Q"),
            tooltip=[
                alt.Tooltip("month_ts:T", format="%B %Y", title=t("p1_tt_month")),
                alt.Tooltip("repeat_customers:Q", format=",.0f", title=t("p1_repeat_cust")),
                alt.Tooltip("repeat_mom_pct:Q", format="+.1f", title=t("p1_tt_mom")),
            ],
        )
        st.altair_chart(
            alt.layer(bars2, line2).properties(height=320),
            use_container_width=True,
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
