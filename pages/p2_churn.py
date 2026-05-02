# ============================================================
# pages/p2_churn.py — Churn Overview (v4)
# Layout: Filter → Matrix+Category → พฤติกรรม(+KPI) → Lost → CTA
# ============================================================
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from utils.helpers import (
    safe_cats, assign_matrix_group, MATRIX_GROUPS,
)

# ── Colour palettes ───────────────────────────────────────────
STATUS_COLORS = {
    "Active":                 "#2ecc71",
    "Medium Risk":            "#f1c40f",
    "Warning (Late > 1.5x)":  "#e67e22",
    "High Risk":              "#e74c3c",
    "Lost (Late > 3x)":       "#95a5a6",
}
MATRIX_COLORS = {
    MATRIX_GROUPS["urgent"]:  "#e74c3c",
    MATRIX_GROUPS["early"]:   "#2980b9",
    MATRIX_GROUPS["monitor"]: "#e67e22",
    MATRIX_GROUPS["active"]:  "#2ecc71",
}
CELL_STYLE = [
    ("🔴 ด่วนที่สุด",    "#fde8e8", "#a32d2d", "ทำแคมเปญทันที"),
    ("🔵 Early Warning", "#e6f1fb", "#185fa5", "ส่ง engagement ก่อนสาย"),
    ("🟡 ติดตาม",        "#faeeda", "#854f0b", "monitor อีก 2 สัปดาห์"),
    ("🟢 ดีอยู่",         "#eaf3de", "#3b6d11", "รักษาประสบการณ์ดีต่อไป"),
]

# 8 กลุ่ม filter label → internal logic
# แบ่งเป็น status-based และ matrix-based
FILTER_GROUPS = {
    "🟩 Active":           ("status",       "Active"),
    "🟨 Medium Risk":      ("status",       "Medium Risk"),
    "🟧 Warning":          ("status",       "Warning (Late > 1.5x)"),
    "🟥 High Risk":        ("status",       "High Risk"),
    "⬛ Lost":             ("status",       "Lost (Late > 3x)"),
    "🚨 Urgent":           ("matrix_group", MATRIX_GROUPS["urgent"]),
    "🔍 Early Warning":    ("matrix_group", MATRIX_GROUPS["early"]),
    "⚠️ Monitor":          ("matrix_group", MATRIX_GROUPS["monitor"]),
}


def render(df: pd.DataFrame, t, threshold: float = 0.55):
    st.title("📊 Churn Overview")
    st.caption("ภาพรวม → เลือกกลุ่มที่อยากโฟกัส → วางแผนแคมเปญที่หน้าถัดไป")

    required = ["churn_probability", "lateness_score", "status", "payment_value"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ ไม่พบ columns: {missing}")
        return

    df = assign_matrix_group(df.copy(), threshold=threshold)

    # ============================================================
    # SECTION 1 — FILTER
    # ============================================================
    st.markdown("---")
    with st.expander("🌪️ กรองข้อมูล", expanded=False):
        fa, fb = st.columns(2)
        with fa:
            sel_cats = st.multiselect(
                "หมวดสินค้า (ว่าง = ทั้งหมด):",
                safe_cats(df), key="p2_cat",
            )
        with fb:
            sel_groups = st.multiselect(
                "กลุ่ม (ว่าง = ทั้งหมด):",
                list(FILTER_GROUPS.keys()),
                key="p2_group",
            )

        st.markdown("""
**📖 คำอธิบายกลุ่ม:**

| กลุ่ม | เงื่อนไข | ความหมาย |
|---|---|---|
| 🟩 Active | AI < 40% และ Late ≤ 1.5x | ลูกค้าปกติ ยังซื้ออยู่ |
| 🟨 Medium Risk | AI 40–75% | AI เริ่มเห็นสัญญาณ ยังไม่ฉุกเฉิน |
| 🟧 Warning | Late > 1.5x | ช้ากว่ารอบปกติ rule เริ่มเตือน |
| 🟥 High Risk | AI > 75% | AI มั่นใจสูงว่าจะหาย |
| ⬛ Lost | Late > 3.0x | หายไปนานมากแล้ว rule ถือว่าสูญ |
| 🚨 Urgent | AI > threshold **และ** Late > 1.5x | ทั้งคู่เห็นตรงกัน — ด่วนที่สุด |
| 🔍 Early Warning | AI > threshold **แต่** Late ≤ 1.5x | AI เห็นก่อน rule — ยังมีเวลา |
| ⚠️ Monitor | AI ≤ threshold **แต่** Late > 1.5x | rule เห็น AI ยังให้โอกาส |
        """)

    # apply filter
    df_d = df.copy()
    if sel_cats:
        df_d = df_d[df_d["product_category_name"].isin(sel_cats)]
    if sel_groups:
        masks = []
        for g in sel_groups:
            col, val = FILTER_GROUPS[g]
            masks.append(df_d[col] == val)
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        df_d = df_d[combined]

    total = len(df_d)
    if total == 0:
        st.warning("ไม่มีข้อมูลตาม filter ที่เลือก")
        return

    # ============================================================
    # SECTION 2 — MATRIX + CATEGORY
    # ============================================================
    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🔲 AI × Rule-based — เห็นตรงกันไหม?")
        st.caption(
            "กลุ่ม **Early Warning** คือ value ของโมเดล — "
            "จับได้ก่อน rule ถ้าไม่มี AI จะมองไม่เห็นกลุ่มนี้เลย"
        )
        ai_on   = df_d["churn_probability"] >= threshold
        rule_on = df_d["lateness_score"]    >= 1.5
        group_masks = [
            ( ai_on &  rule_on),
            ( ai_on & ~rule_on),
            (~ai_on &  rule_on),
            (~ai_on & ~rule_on),
        ]
        cells = []
        for mask in group_masks:
            n   = int(mask.sum())
            rev = float(df_d.loc[mask, "payment_value"].sum())
            cells.append({"n": n, "pct": n / total * 100, "rev": rev})

        row1 = st.columns(2)
        row2 = st.columns(2)
        grid_cols    = [row1[0], row1[1], row2[0], row2[1]]
        group_labels = list(MATRIX_GROUPS.values())

        for col, data, (tag, bg, color, action), label in zip(
            grid_cols, cells, CELL_STYLE, group_labels
        ):
            with col:
                st.markdown(
                    f"""<div style="background:{bg};border-radius:10px;
                        padding:12px;margin:4px 0">
                      <div style="font-size:11px;font-weight:600;color:{color}">{tag}</div>
                      <div style="font-size:22px;font-weight:700;color:{color}">{data['n']:,}</div>
                      <div style="font-size:11px;color:{color}">{data['pct']:.1f}%
                        &nbsp;·&nbsp; R$ {data['rev']:,.0f}</div>
                      <div style="font-size:10px;color:{color};opacity:.8;margin-top:2px">{label}</div>
                      <div style="font-size:11px;color:{color};margin-top:4px;font-weight:600">→ {action}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Lost → AI breakdown
        lost_df = df_d[df_d["status"] == "Lost (Late > 3x)"]
        if len(lost_df) > 0:
            st.markdown("---")
            st.caption(
                f"**⚰️ Lost group ({len(lost_df):,} คน)** — "
                "Rule ถือว่าสูญไปแล้ว แต่ AI มองว่า:"
            )
            vc = lost_df["matrix_group"].value_counts().reset_index()
            vc.columns = ["matrix_group", "count"]
            all_g   = list(MATRIX_GROUPS.values())
            missing = [g for g in all_g if g not in vc["matrix_group"].values]
            if missing:
                vc = pd.concat([
                    vc,
                    pd.DataFrame({"matrix_group": missing, "count": 0}),
                ], ignore_index=True)
            vc["pct"] = vc["count"] / len(lost_df) * 100

            lost_bar = (
                alt.Chart(vc)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("count:Q", title="จำนวนคน"),
                    y=alt.Y("matrix_group:N", sort="-x", title=None),
                    color=alt.Color(
                        "matrix_group:N",
                        scale=alt.Scale(
                            domain=list(MATRIX_COLORS.keys()),
                            range=list(MATRIX_COLORS.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("matrix_group", title="กลุ่ม"),
                        alt.Tooltip("count:Q", format=",", title="จำนวน"),
                        alt.Tooltip("pct:Q",   format=".1f", title="%"),
                    ],
                )
                .properties(height=130, title="Lost customers → AI จัดอยู่กลุ่มไหน?")
            )
            st.altair_chart(lost_bar, use_container_width=True)
            st.caption("💡 Lost + Early Warning → ยังพอดึงกลับได้ ลองแคมเปญ re-engagement")

    with col_right:
        st.subheader("📦 สัดส่วนความเสี่ยงต่อหมวดสินค้า")
        st.caption("หมวดไหนมีสีแดง/ส้มเยอะ = ลูกค้าหนีสูง → เลือก filter หมวดนั้นก่อนไปหน้า Action Plan")
        if "product_category_name" in df_d.columns:
            _render_category_chart(df_d)
        else:
            st.info("ไม่มีข้อมูล product_category_name")

    # ============================================================
    # SECTION 3 — พฤติกรรมแต่ละกลุ่ม (มี KPI อยู่ข้างใน)
    # ============================================================
    st.markdown("---")
    st.subheader("🔍 พฤติกรรมแต่ละกลุ่ม")

    group_tabs = st.tabs([
        "🚨 Urgent", "🔍 Early Warning", "⚠️ Monitor", "✅ Active", "⚰️ Lost"
    ])
    group_dfs = {
        "urgent":  df_d[df_d["matrix_group"] == MATRIX_GROUPS["urgent"]],
        "early":   df_d[df_d["matrix_group"] == MATRIX_GROUPS["early"]],
        "monitor": df_d[df_d["matrix_group"] == MATRIX_GROUPS["monitor"]],
        "active":  df_d[df_d["matrix_group"] == MATRIX_GROUPS["active"]],
        "lost":    df_d[df_d["status"] == "Lost (Late > 3x)"],
    }
    for tab, key in zip(group_tabs, ["urgent", "early", "monitor", "active", "lost"]):
        with tab:
            _render_behaviour(group_dfs[key], total)

    # ============================================================
    # SECTION 4 — CTA
    # ============================================================
    st.markdown("---")
    urgent_n = (df_d["matrix_group"] == MATRIX_GROUPS["urgent"]).sum()
    early_n  = (df_d["matrix_group"] == MATRIX_GROUPS["early"]).sum()
    cta1, cta2 = st.columns([3, 1])
    with cta1:
        st.info(
            f"**พร้อมแล้ว? ไปวางแผนแคมเปญที่หน้าถัดไป** 🎯\n\n"
            f"แนะนำเริ่มจาก **Urgent ({urgent_n:,} คน)** — ทั้ง AI และ Rule เห็นตรงกัน\n\n"
            f"หรือ **Early Warning ({early_n:,} คน)** ถ้าอยากทำ retention ก่อนสาย"
        )
    with cta2:
        if st.button("🎯 ไปหน้า Action Plan →", use_container_width=True, type="primary"):
            st.session_state["p3_prefilter_matrix"] = MATRIX_GROUPS["urgent"]
            st.session_state["active_page"] = "3. 🎯 Action Plan"
            st.rerun()


# ============================================================
# HELPERS
# ============================================================

def _render_category_chart(df_d: pd.DataFrame) -> None:
    """Stacked bar ทุกหมวด — เลื่อนดูได้"""
    all_cats = df_d["product_category_name"].dropna().unique().tolist()
    if not all_cats:
        st.info("ไม่มีข้อมูลหมวดสินค้า")
        return

    # เรียงตาม High Risk % สูงสุด
    all_statuses = list(STATUS_COLORS.keys())
    idx = pd.MultiIndex.from_product(
        [all_cats, all_statuses],
        names=["product_category_name", "status"],
    )
    cat_total = df_d.groupby("product_category_name").size().rename("total")
    cat_stat  = (
        df_d.groupby(["product_category_name", "status"])
        .size()
        .reindex(idx, fill_value=0)
        .reset_index(name="count")
    )
    cat_stat = cat_stat.merge(cat_total, on="product_category_name", how="left")
    cat_stat["pct"] = cat_stat["count"] / cat_stat["total"] * 100

    sort_order = (
        cat_stat[cat_stat["status"] == "High Risk"]
        .sort_values("pct", ascending=False)["product_category_name"].tolist()
    )
    y_order = sort_order + [c for c in all_cats if c not in sort_order]

    # ความสูง dynamic ตามจำนวนหมวด (min 400, ~22px/หมวด)
    chart_h = max(400, len(all_cats) * 22)

    chart = (
        alt.Chart(cat_stat).mark_bar()
        .encode(
            x=alt.X("pct:Q", stack="normalize",
                    axis=alt.Axis(format="%", title="สัดส่วน")),
            y=alt.Y("product_category_name:N", sort=y_order, title=None),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=list(STATUS_COLORS.keys()),
                    range=list(STATUS_COLORS.values()),
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            order=alt.Order("status:N"),
            tooltip=[
                alt.Tooltip("product_category_name", title="หมวด"),
                alt.Tooltip("status",  title="สถานะ"),
                alt.Tooltip("count:Q", format=",",   title="จำนวน"),
                alt.Tooltip("pct:Q",   format=".1f", title="%"),
            ],
        )
        .properties(height=chart_h)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(f"💡 แสดงทุกหมวด ({len(all_cats)} หมวด) เรียงจาก High Risk มากสุด → น้อยสุด")


def _render_behaviour(grp: pd.DataFrame, total_all: int) -> None:
    """KPI + พฤติกรรม: ราคา ค่าส่ง ส่งช้า รอบซื้อ + หมวด+ราคา + วิธีจ่าย"""
    if grp.empty:
        st.info("ไม่มีข้อมูลในกลุ่มนี้")
        return

    n = len(grp)

    # ── KPI row ───────────────────────────────────────────────
    st.caption("📊 **KPI กลุ่มนี้**")
    k1, k2, k3, k4 = st.columns(4)

    rev       = grp["payment_value"].sum() if "payment_value" in grp.columns else 0
    avg_cycle = grp["cat_median_days"].mean() if "cat_median_days" in grp.columns else None

    k1.metric("💰 Revenue",        f"R$ {rev:,.0f}")
    k2.metric("🔄 Avg Cycle",      f"{avg_cycle:.0f} วัน" if avg_cycle else "N/A")
    k3.metric("👥 จำนวน",          f"{n:,} คน")
    k4.metric("📊 % ของทั้งหมด",   f"{n/total_all*100:.1f}%")

    st.markdown("")

    # ── พฤติกรรม row ──────────────────────────────────────────
    st.caption("🛒 **พฤติกรรมการซื้อ**")
    b1, b2, b3, b4 = st.columns(4)

    avg_price   = grp["price"].mean()         if "price"          in grp.columns else None
    avg_freight = grp["freight_value"].mean() if "freight_value"  in grp.columns else None

    # ส่งช้า = ส่งช้ากว่า estimated จริง (delay_days > 0)
    late_pct = (grp["delay_days"] > 0).mean() * 100 if "delay_days" in grp.columns else None

    avg_prob = grp["churn_probability"].mean() * 100 if "churn_probability" in grp.columns else None

    b1.metric("🛒 ราคาสินค้าเฉลี่ย",
              f"R$ {avg_price:,.0f}"   if avg_price   is not None else "N/A")
    b2.metric("🚚 ค่าส่งเฉลี่ย",
              f"R$ {avg_freight:,.0f}" if avg_freight is not None else "N/A")
    b3.metric("⏰ ส่งช้ากว่ากำหนด",
              f"{late_pct:.1f}%"       if late_pct   is not None else "N/A",
              delta_color="inverse")
    b4.metric("🤖 Avg Churn Prob",
              f"{avg_prob:.1f}%"       if avg_prob   is not None else "N/A",
              delta_color="inverse")

    st.markdown("")
    c_cat, c_pay = st.columns(2)

    # ── ทุกหมวดสินค้า + ราคาเฉลี่ย เลื่อนดูได้ ──────────────
    with c_cat:
        if "product_category_name" in grp.columns and "price" in grp.columns:
            cat_price = (
                grp.groupby("product_category_name")
                .agg(
                    orders=("price", "count"),
                    avg_price=("price", "mean"),
                )
                .reset_index()
                .sort_values("orders", ascending=False)
                .rename(columns={
                    "product_category_name": "หมวด",
                    "orders":    "ออเดอร์",
                    "avg_price": "ราคาเฉลี่ย (R$)",
                })
            )
            n_cats   = len(cat_price)
            chart_h  = max(200, n_cats * 20)

            bar = (
                alt.Chart(cat_price)
                .mark_bar(color="#3498db",
                          cornerRadiusTopRight=4,
                          cornerRadiusBottomRight=4)
                .encode(
                    x=alt.X("ออเดอร์:Q", title="จำนวนออเดอร์"),
                    y=alt.Y("หมวด:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("หมวด"),
                        alt.Tooltip("ออเดอร์:Q",          format=","),
                        alt.Tooltip("ราคาเฉลี่ย (R$):Q",  format=",.0f"),
                    ],
                )
                .properties(
                    height=chart_h,
                    title=f"ทุกหมวดสินค้า ({n_cats} หมวด) + ราคาเฉลี่ย",
                )
            )

            # text label ราคาเฉลี่ย
            text = (
                alt.Chart(cat_price)
                .mark_text(align="left", dx=4, fontSize=10, color="#555")
                .encode(
                    x=alt.X("ออเดอร์:Q"),
                    y=alt.Y("หมวด:N", sort="-x"),
                    text=alt.Text("ราคาเฉลี่ย (R$):Q", format=",.0f"),
                )
            )
            st.altair_chart(bar + text, use_container_width=True)
        elif "product_category_name" in grp.columns:
            # fallback ถ้าไม่มี price
            top_cat = (
                grp["product_category_name"].value_counts()
                .reset_index()
            )
            top_cat.columns = ["หมวด", "ออเดอร์"]
            st.altair_chart(
                alt.Chart(top_cat)
                .mark_bar(color="#3498db")
                .encode(
                    x=alt.X("ออเดอร์:Q"),
                    y=alt.Y("หมวด:N", sort="-x", title=None),
                    tooltip=["หมวด", alt.Tooltip("ออเดอร์:Q", format=",")],
                )
                .properties(height=max(200, len(top_cat)*20), title="ทุกหมวดสินค้า"),
                use_container_width=True,
            )

    # ── วิธีจ่ายเงิน ──────────────────────────────────────────
    with c_pay:
        if "payment_type" in grp.columns:
            pay_dist = (
                grp["payment_type"].value_counts(normalize=True)
                .mul(100).reset_index()
            )
            pay_dist.columns = ["วิธีจ่าย", "สัดส่วน (%)"]
            pie = (
                alt.Chart(pay_dist)
                .mark_arc(innerRadius=40)
                .encode(
                    theta=alt.Theta("สัดส่วน (%):Q"),
                    color=alt.Color(
                        "วิธีจ่าย:N",
                        scale=alt.Scale(scheme="tableau10"),
                        legend=alt.Legend(orient="right"),
                    ),
                    tooltip=[
                        alt.Tooltip("วิธีจ่าย"),
                        alt.Tooltip("สัดส่วน (%):Q", format=".1f"),
                    ],
                )
                .properties(height=220, title="วิธีชำระเงิน")
            )
            st.altair_chart(pie, use_container_width=True)
