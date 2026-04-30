# ============================================================
# pages/p5_logistics.py — Logistics Insights
# ============================================================
import pydeck as pdk
import pandas as pd
import streamlit as st
from i18n import t
from utils.helpers import safe_cats, status_display_options

#좌표 รัฐบราซิล
BRAZIL_COORDS: dict[str, list[float]] = {
    "AC":[-9.02,-70.81],"AL":[-9.57,-36.78],"AM":[-3.41,-65.85],
    "AP":[0.90,-52.00], "BA":[-12.58,-41.70],"CE":[-5.49,-39.32],
    "DF":[-15.79,-47.88],"ES":[-19.18,-40.30],"GO":[-15.82,-49.84],
    "MA":[-5.19,-45.16],"MG":[-19.81,-43.95],"MS":[-20.77,-54.78],
    "MT":[-12.96,-56.92],"PA":[-6.31,-52.46],"PB":[-7.24,-36.78],
    "PE":[-8.81,-36.95],"PI":[-7.71,-42.72],"PR":[-25.25,-52.02],
    "RJ":[-22.90,-43.17],"RN":[-5.40,-36.95],"RO":[-11.50,-63.58],
    "RR":[2.73,-62.07], "RS":[-30.03,-51.22],"SC":[-27.24,-50.21],
    "SE":[-10.57,-37.38],"SP":[-23.55,-46.63],"TO":[-10.17,-48.33],
}


def render(df: pd.DataFrame) -> None:
    st.title(t("page_logistics"))

    if "customer_state" not in df.columns:
        st.error(t("p5_no_state"))
        return

    # ── Filters ───────────────────────────────────────────────
    display_list, to_internal, _ = status_display_options(t)
    c1, c2 = st.columns(2)
    with c1:
        sel_c = st.multiselect(t("cat_label"), safe_cats(df), key="p5_cat")
    with c2:
        sel_s_display = st.multiselect(t("status_label"), display_list, key="p5_status")
    sel_s = [to_internal[s] for s in sel_s_display if s in to_internal]

    df_log = df.copy()
    if sel_c: df_log = df_log[df_log["product_category_name"].isin(sel_c)]
    if sel_s: df_log = df_log[df_log["status"].isin(sel_s)]

    # ── Aggregate by state ────────────────────────────────────
    sm = (
        df_log.groupby("customer_state")
        .agg(
            payment_value=("payment_value", "sum"),
            delivery_days=("delivery_days", "mean"),
            delay_count=("delay_days", lambda x: (x > 0).sum()),
            churn_probability=("churn_probability", "mean"),
            total_orders=("order_purchase_timestamp", "count"),
        )
        .reset_index()
    )
    sm["lat"] = sm["customer_state"].map(lambda x: BRAZIL_COORDS.get(x, [0, 0])[0])
    sm["lon"] = sm["customer_state"].map(lambda x: BRAZIL_COORDS.get(x, [0, 0])[1])

    # ── Focus + KPIs ──────────────────────────────────────────
    st.markdown("---")
    cs, k1, k2, k3 = st.columns([1.5, 1, 1, 1])
    with cs:
        zoom = st.selectbox(t("p5_focus"), ["All"] + sorted(sm["customer_state"].unique()))

    disp     = sm if zoom == "All" else sm[sm["customer_state"] == zoom]
    view_lat = disp["lat"].mean() if zoom != "All" else -14.24
    view_lon = disp["lon"].mean() if zoom != "All" else -51.93
    view_z   = 6 if zoom != "All" else 3.5

    k1.metric(t("p5_metric_rev"),  f"R$ {disp['payment_value'].sum():,.0f}")
    k2.metric(t("p5_metric_del"),  f"{disp['delivery_days'].mean():.1f}{t('days_unit')}")
    k3.metric(t("p5_metric_late"), f"{int(disp['delay_count'].sum()):,}{t('times_unit')}")

    # ── Map + Table ───────────────────────────────────────────
    cm_, ct_ = st.columns([2, 1])
    with cm_:
        st.subheader(t("p5_map_title", z=zoom))
        _render_map(sm, view_lat, view_lon, view_z)

    with ct_:
        st.subheader(t("p5_issues"))
        sort_opts = [t("p5_sort_late"), t("p5_sort_risk")]
        sort_m    = st.radio(t("p5_sort"), sort_opts, horizontal=True, key="p5_sort")
        sort_col  = "delay_count" if sort_m == t("p5_sort_late") else "churn_probability"
        top_i     = sm.sort_values(sort_col, ascending=False).head(10)
        st.dataframe(
            top_i[["customer_state", "payment_value", "delivery_days", "delay_count", "churn_probability"]],
            column_config={
                "payment_value":     st.column_config.NumberColumn(t("p5_col_money"), format="R$%.0f"),
                "delivery_days":     st.column_config.NumberColumn(t("p5_col_days"),  format="%.1f"),
                "churn_probability": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
            },
            hide_index=True,
            use_container_width=True,
        )

    # ── City drilldown ────────────────────────────────────────
    st.markdown("---")
    st.subheader(t("p5_city"))
    if "customer_city" in df_log.columns:
        city_m = (
            df_log.groupby(["customer_state", "customer_city"])
            .agg(n=("customer_unique_id","count"), revenue=("payment_value","sum"),
                 del_days=("delivery_days","mean"), late=("delay_days", lambda x: (x>0).sum()),
                 risk=("churn_probability","mean"))
            .reset_index()
        )
        city_m = city_m[city_m["n"] >= 2]
        disp_c = city_m[city_m["customer_state"] == zoom] if zoom != "All" else city_m
        st.info(t("p5_city_state", s=zoom) if zoom != "All" else t("p5_city_all"))
        st.dataframe(
            disp_c.sort_values("late", ascending=False).head(50),
            column_config={
                "n":       st.column_config.NumberColumn(t("p5_col_cust")),
                "revenue": st.column_config.NumberColumn(t("p5_metric_rev"), format="R$%.0f"),
                "del_days":st.column_config.NumberColumn(t("p5_col_days"),   format="%.1f"),
                "late":    st.column_config.NumberColumn(t("p5_col_late2")),
                "risk":    st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
            },
            hide_index=True,
            use_container_width=True,
        )


def _render_map(sm: pd.DataFrame, lat: float, lon: float, zoom: float) -> None:
    sm = sm.copy()
    sm["color"] = sm["churn_probability"].apply(
        lambda x: [231,76,60,200] if x > 0.8 else ([241,196,15,200] if x > 0.5 else [46,204,113,200])
    )
    mx = sm["payment_value"].max()
    sm["radius"] = (sm["payment_value"] / mx * 400_000) if mx > 0 else 10_000

    layer = pdk.Layer(
        "ScatterplotLayer", sm,
        get_position="[lon,lat]", get_color="color", get_radius="radius",
        pickable=True, opacity=0.8, stroked=True, filled=True,
        radius_min_pixels=5, radius_max_pixels=60,
    )
    tooltip = {
        "html": "<b>{customer_state}</b><br/>💰 R$ {payment_value}<br/>🚚 {delivery_days} d<br/>⚠️ {delay_count}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=20),
        tooltip=tooltip, map_provider="carto", map_style="light",
    ))
