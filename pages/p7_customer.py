# ============================================================
# pages/p7_customer.py — Customer Deep Dive
# ============================================================
import altair as alt
import pandas as pd
import streamlit as st
from i18n import t
from utils.helpers import safe_cats, status_display_options


def render(df: pd.DataFrame) -> None:
    st.title(t("page_customer"))

    display_list, to_internal, to_display = status_display_options(t)
    default_display = [to_display["High Risk"], to_display["Warning (Late > 1.5x)"]]

    with st.expander(t("p7_filters"), expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_status_display = st.multiselect(t("status_label"), display_list, default=default_display)
            sel_status = [to_internal[s] for s in sel_status_display if s in to_internal]
        with f2:
            sel_cats = st.multiselect(t("cat_label"), safe_cats(df))
        with f3:
            search_id = st.text_input(t("p7_search"), "")

    # ── Filter ────────────────────────────────────────────────
    mask = df["status"].isin(sel_status)
    if sel_cats:  mask = mask & df["product_category_name"].isin(sel_cats)
    if search_id: mask = mask & df["customer_unique_id"].str.contains(search_id, case=False, na=False)
    filtered = df[mask]

    # ── Category risk chart ───────────────────────────────────
    if "product_category_name" in df.columns and not filtered.empty:
        cat_ov   = df.groupby("product_category_name").agg(Total=("customer_unique_id","count"), Cycle=("cat_median_days","mean")).reset_index()
        cat_risk = filtered.groupby("product_category_name").agg(Risk=("customer_unique_id","count")).reset_index()
        cat_s    = cat_risk.merge(cat_ov, on="product_category_name", how="left")
        cat_s["Risk_Pct"] = cat_s["Risk"] / cat_s["Total"]
        cat_s = cat_s.sort_values("Risk", ascending=False)

        cc, ct = st.columns([1.5, 2.5])
        with cc:
            st.subheader(t("p7_top10"))
            base   = alt.Chart(cat_s.head(10)).encode(y=alt.Y("product_category_name", sort="-x", title=None))
            b_tot  = base.mark_bar(color="#f0f2f6").encode(x="Total")
            b_risk = base.mark_bar(color="#e74c3c").encode(x="Risk")
            st.altair_chart(b_tot + b_risk, use_container_width=True)
        with ct:
            st.subheader(t("p7_detail"))
            st.dataframe(cat_s, use_container_width=True, hide_index=True)

    # ── Customer list ─────────────────────────────────────────
    st.markdown("---")
    st.subheader(t("p7_list", n=f"{len(filtered):,}"))
    show_cols = [c for c in [
        "customer_unique_id", "status", "churn_probability",
        "lateness_score", "cat_median_days", "payment_value", "product_category_name",
    ] if c in df.columns]

    st.dataframe(
        filtered[show_cols].sort_values("churn_probability", ascending=False),
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
            "lateness_score":    st.column_config.NumberColumn(t("p7_col_late"), format="%.1fx"),
        },
        use_container_width=True,
    )
