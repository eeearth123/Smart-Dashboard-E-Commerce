# ============================================================
# pages/p2_churn.py — Churn Overview
# ============================================================
import altair as alt
import pandas as pd
import streamlit as st
from i18n import t
from utils.helpers import safe_cats
from config import BEST_THRESHOLD


def render(df: pd.DataFrame) -> None:
    st.title(t("page_churn"))

    # ── Segmentation guide ────────────────────────────────────
    with st.expander(t("p2_seg_expander"), expanded=True):
        st.markdown(f"""
| {t('p2_seg_status')} | {t('p2_seg_cond')} |
|---|---|
| 🔴 Lost | {t('p2_lost_cond')} |
| 🟥 High Risk | {t('p2_high_cond')} |
| 🟧 Warning | {t('p2_warn_cond')} |
| 🟨 **Medium Risk** | **{t('p2_med_cond')}** |
| 🟩 Active | {t('p2_act_cond')} |
        """)

    with st.expander(t("filter_expand"), expanded=False):
        sel_cats = st.multiselect(t("cat_label"), safe_cats(df), key="p2_cat")

    dfd = df[df["product_category_name"].isin(sel_cats)].copy() if sel_cats else df.copy()
    st.markdown("---")

    # ── KPI metrics ───────────────────────────────────────────
    total   = len(dfd)
    risk_df = dfd[dfd["status"].isin(["High Risk", "Warning (Late > 1.5x)"])]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(t("p2_at_risk"),   f"{len(risk_df)/total*100:.1f}%" if total else "0%")
    k2.metric(t("p2_ai_pred"),   f"{(dfd['churn_probability'] >= BEST_THRESHOLD).mean()*100:.1f}%")
    k3.metric(t("p2_rev_risk"),  f"R$ {risk_df['payment_value'].sum():,.0f}")
    k4.metric(t("p2_ratio"),     f"{len(risk_df):,} / {total:,}")
    k5.metric(
        t("p2_avg_cycle"),
        f"{dfd['cat_median_days'].mean():.0f}{t('days_unit')}"
        if "cat_median_days" in dfd.columns else t("na"),
    )
    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(t("p2_trend"))
        _render_trend(dfd)
    with c2:
        st.subheader(t("p2_rev_chart"))
        _render_donut(dfd)


# ── Private helpers ───────────────────────────────────────────

def _render_trend(dfd: pd.DataFrame) -> None:
    if "order_purchase_timestamp" not in dfd.columns or dfd.empty:
        st.info(t("p2_no_trend"))
        return

    dfd = dfd.copy()
    dfd["month_year"] = dfd["order_purchase_timestamp"].dt.to_period("M")
    rule_lbl = t("p2_rule_lbl")
    ai_lbl   = t("p2_ai_lbl")

    rows = []
    for name, grp in dfd.groupby("month_year"):
        tot = len(grp)
        if not tot:
            continue
        rule = len(grp[grp["status"].isin(["High Risk", "Warning (Late > 1.5x)"])])
        ai   = (grp["churn_probability"] >= BEST_THRESHOLD).sum()
        rows.append({"Date": str(name), rule_lbl: rule/tot*100, ai_lbl: ai/tot*100})

    tdf = pd.DataFrame(rows)
    if len(tdf) <= 1:
        st.info(t("p2_no_trend"))
        return

    tdf        = tdf.iloc[:-1]
    tdf["Date"] = pd.to_datetime(tdf["Date"])
    melted      = tdf.melt("Date", var_name="Type", value_name="Rate (%)")

    chart = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date", axis=alt.Axis(format="%b %Y", title=t("p2_timeline"))),
            y=alt.Y("Rate (%)", title=t("p2_churn_rate")),
            color=alt.Color(
                "Type",
                scale=alt.Scale(domain=[rule_lbl, ai_lbl], range=["#e67e22", "#8e44ad"]),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=["Date", "Type", alt.Tooltip("Rate (%)", format=".1f")],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_donut(dfd: pd.DataFrame) -> None:
    if dfd.empty:
        return
    stats   = dfd.groupby("status").agg(Count=("customer_unique_id", "count"), Revenue=("payment_value", "sum")).reset_index()
    domain  = ["Active", "Medium Risk", "Warning (Late > 1.5x)", "High Risk", "Lost (Late > 3x)"]
    palette = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#95a5a6"]
    donut = (
        alt.Chart(stats)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta("Count", type="quantitative"),
            color=alt.Color("status", scale=alt.Scale(domain=domain, range=palette), legend=dict(orient="bottom")),
            tooltip=["status", alt.Tooltip("Count", format=","), alt.Tooltip("Revenue", format=",.0f")],
        )
        .properties(height=350)
    )
    st.altair_chart(donut, use_container_width=True)
