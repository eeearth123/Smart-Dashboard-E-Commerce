# ============================================================
# pages/p4_cycle.py — Buying Cycle Analysis
# ============================================================
import altair as alt
import pandas as pd
import streamlit as st
from i18n import t
from utils.helpers import safe_cats


def render(df: pd.DataFrame) -> None:
    st.title(t("page_cycle"))

    sel_c = st.multiselect(t("cat_label"), safe_cats(df), key="p4_cat")
    df_cy = df[df["product_category_name"].isin(sel_c)].copy() if sel_c else df.copy()

    # ── KPI ───────────────────────────────────────────────────
    g_avg  = df["cat_median_days"].mean()
    c_avg  = df_cy["cat_median_days"].mean()
    c_late = df_cy["lateness_score"].mean() if "lateness_score" in df_cy.columns else 0
    fast   = (df_cy["cat_median_days"] <= 30).sum()

    m1, m2, m3 = st.columns(3)
    m1.metric(t("p4_avg_cycle"),
              f"{c_avg:.0f}{t('days_unit')}",
              f"{c_avg - g_avg:+.0f}{t('p4_cycle_delta')}",
              delta_color="inverse")
    m2.metric(t("p4_lateness"), f"{c_late:.2f}x")
    m3.metric(t("p4_fast"),     f"{fast:,}{t('people_unit')}")

    st.markdown("---")
    st.subheader(t("p4_trend"))
    _render_trend(df_cy)

    st.markdown("---")
    st.subheader(t("p4_detail"))
    _render_detail_table(df_cy)

    st.markdown("---")
    st.subheader(t("p4_heatmap"))
    _render_heatmap(df_cy)


def _render_trend(df_cy: pd.DataFrame) -> None:
    if "order_purchase_timestamp" not in df_cy.columns:
        st.info(t("p4_no_trend"))
        return

    tmp = df_cy.sort_values(["customer_unique_id", "order_purchase_timestamp"]).copy()
    tmp["prev_t"] = tmp.groupby("customer_unique_id")["order_purchase_timestamp"].shift(1)
    tmp["gap"]    = (tmp["order_purchase_timestamp"] - tmp["prev_t"]).dt.days
    rep = tmp[tmp["gap"].notna() & (tmp["gap"] > 0)].copy()

    if rep.empty:
        st.info(t("p4_no_repeat"))
        return

    rep["month_year"] = rep["order_purchase_timestamp"].dt.to_period("M")
    tgap = rep.groupby("month_year")["gap"].mean().reset_index()
    if len(tgap) <= 1:
        st.info(t("p4_no_trend"))
        return

    tgap        = tgap.iloc[:-1]
    tgap["Date"] = pd.to_datetime(tgap["month_year"].astype(str))
    st.altair_chart(
        alt.Chart(tgap)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Date", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("gap", title=t("p4_gap_y"), scale=alt.Scale(zero=False)),
            color=alt.value("#e67e22"),
            tooltip=["Date", alt.Tooltip("gap", format=".1f")],
        )
        .properties(height=350),
        use_container_width=True,
    )


def _render_detail_table(df_cy: pd.DataFrame) -> None:
    summ = (
        df_cy.groupby("product_category_name")
        .agg(
            Customers=("customer_unique_id", "count"),
            Cycle_Days=("cat_median_days", "mean"),
            Late_Score=("lateness_score", "mean"),
            Churn_Risk=("churn_probability", "mean"),
        )
        .reset_index()
        .sort_values("Cycle_Days")
    )
    st.dataframe(
        summ,
        column_config={
            "Customers":  st.column_config.NumberColumn(t("p4_col_cust"),  format=f"%d{t('people_unit')}"),
            "Cycle_Days": st.column_config.NumberColumn(t("p4_col_cycle"), format=f"%.0f{t('days_unit')}"),
            "Late_Score": st.column_config.NumberColumn(t("p4_col_late"),  format="%.2fx"),
            "Churn_Risk": st.column_config.ProgressColumn("Risk",          format="%.2f", min_value=0, max_value=1),
        },
        hide_index=True,
        use_container_width=True,
    )


def _render_heatmap(df_cy: pd.DataFrame) -> None:
    if "order_purchase_timestamp" not in df_cy.columns:
        return
    sea = df_cy.copy()
    sea["month_num"]  = sea["order_purchase_timestamp"].dt.month
    sea["month_name"] = sea["order_purchase_timestamp"].dt.strftime("%b")
    hm = sea.groupby(["product_category_name", "month_num", "month_name"]).size().reset_index(name="vol")
    top_c = sea["product_category_name"].value_counts().head(15).index.tolist()
    hm    = hm[hm["product_category_name"].isin(top_c)]
    if hm.empty:
        return
    st.altair_chart(
        alt.Chart(hm)
        .mark_rect()
        .encode(
            x=alt.X("month_name", sort=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]),
            y=alt.Y("product_category_name", title=None),
            color=alt.Color("vol", scale=alt.Scale(scheme="orangered"), title="Vol"),
            tooltip=["product_category_name", "month_name", alt.Tooltip("vol", format=",")],
        )
        .properties(height=500),
        use_container_width=True,
    )
    st.info(t("p4_heat_tip"))
