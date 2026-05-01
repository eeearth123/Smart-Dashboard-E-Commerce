# ============================================================
# pages/p2_churn.py — Churn Overview (v3)
# Layout: Filter → Matrix+Category → KPI → Behaviour → Lost → CTA
# ============================================================
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from utils.helpers import (
    safe_cats, status_display_options,
    assign_matrix_group, MATRIX_GROUPS,
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


def render(df: pd.DataFrame, t, threshold: float = 0.55):
    st.title("📊 Churn Overview")
    st.caption("ภาพรวม → เลือกกลุ่มที่อยากโฟกัส → วางแผนแคมเปญที่หน้าถัดไป")

    # ── guard ─────────────────────────────────────────────────
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
            display_list, to_internal, to_display = status_display_options(t)
            sel_status_display = st.multiselect(
                "สถานะ (ว่าง = ทั้งหมด):", display_list, key="p2_status",
            )

        st.markdown("""
**📖 คำอธิบายกลุ่ม:**

| สถานะ | เงื่อนไข | ความหมาย |
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

    df_d = df.copy()
    if sel_cats:
        df_d = df_d[df_d["product_category_name"].isin(sel_cats)]
    if sel_status_display:
        sel_int = [to_internal[s] for s in sel_status_display if s in to_internal]
        df_d = df_d[df_d["status"].isin(sel_int)]

    total = len(df_d)
    if total == 0:
        st.warning("ไม่มีข้อมูลตาม filter ที่เลือก")
        return

    # ============================================================
    # SECTION 2 — MATRIX + CATEGORY (บนสุด)
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
            rev = float(df_d.loc[mask, "payment_value"].sum()) if "payment_value" in df_d.columns else 0
            cells.append({"n": n, "pct": n / total * 100, "rev": rev})

        row1 = st.columns(2)
        row2 = st.columns(2)
        grid_cols = [row1[0], row1[1], row2[0], row2[1]]
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

        # ── Lost → AI matrix ──────────────────────────────────
        lost_df = df_d[df_d["status"] == "Lost (Late > 3x)"]
        if len(lost_df) > 0:
            st.markdown("---")
            st.caption(
                f"**⚰️ Lost group ({len(lost_df):,} คน)** — "
                "Rule ถือว่าสูญไปแล้ว แต่ AI มองว่า:"
            )
            vc = lost_df["matrix_group"].value_counts().reset_index()
            vc.columns = ["matrix_group", "count"]
            # เติม group ที่หายไปให้ครบ 4 กลุ่ม
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
        st.caption(
            "หมวดไหนมีสีแดง/ส้มเยอะ = ลูกค้าหนีสูง "
            "→ เลือก filter หมวดนั้นก่อนไปหน้า Action Plan"
        )
        if "product_category_name" in df_d.columns:
            _render_category_chart(df_d)
        else:
            st.info("ไม่มีข้อมูล product_category_name")

    # ============================================================
    # SECTION 3 — KPI
    # ============================================================
    st.markdown("---")
    st.subheader("📊 KPI ตามกลุ่มที่เลือก")

    at_risk_mask = df_d["status"].isin(["High Risk", "Warning (Late > 1.5x)"])
    rev_total    = df_d["payment_value"].sum()
    rev_at_risk  = df_d.loc[at_risk_mask, "payment_value"].sum()
    avg_cycle    = df_d["cat_median_days"].mean() if "cat_median_days" in df_d.columns else None

    st.caption("💰 **รายได้และภาพรวม**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Revenue (กลุ่มที่เลือก)", f"R$ {rev_total:,.0f}")
    k2.metric("💸 Revenue at Risk",
              f"R$ {rev_at_risk:,.0f}",
              f"{rev_at_risk/rev_total*100:.1f}% ของทั้งหมด" if rev_total else "")
    k3.metric("🔄 Avg Repurchase Cycle",
              f"{avg_cycle:.0f} วัน" if avg_cycle else "N/A")
    k4.metric("👥 จำนวนทั้งหมด", f"{total:,} คน")

    st.markdown("")
    st.caption("🔢 **สัดส่วน AI × Rule กลุ่ม**")
    urgent_n  = (df_d["matrix_group"] == MATRIX_GROUPS["urgent"]).sum()
    early_n   = (df_d["matrix_group"] == MATRIX_GROUPS["early"]).sum()
    monitor_n = (df_d["matrix_group"] == MATRIX_GROUPS["monitor"]).sum()
    active_n  = (df_d["matrix_group"] == MATRIX_GROUPS["active"]).sum()

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("🚨 Urgent",        f"{urgent_n:,} คน",  f"{urgent_n/total*100:.1f}%")
    g2.metric("🔍 Early Warning", f"{early_n:,} คน",   f"{early_n/total*100:.1f}%")
    g3.metric("⚠️ Monitor",       f"{monitor_n:,} คน", f"{monitor_n/total*100:.1f}%")
    g4.metric("✅ Active",         f"{active_n:,} คน",  f"{active_n/total*100:.1f}%")

    # ============================================================
    # SECTION 4 — พฤติกรรมแต่ละกลุ่ม
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
            _render_behaviour(group_dfs[key])

    # ============================================================
    # SECTION 5 — CTA
    # ============================================================
    st.markdown("---")
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
    """Stacked bar — สัดส่วน status ต่อหมวดสินค้า Top 12"""
    at_risk_cats = (
        df_d[df_d["status"].isin(["High Risk", "Warning (Late > 1.5x)"])]
        .groupby("product_category_name").size()
        .sort_values(ascending=False).head(12).index.tolist()
    )
    if not at_risk_cats:
        at_risk_cats = df_d["product_category_name"].value_counts().head(12).index.tolist()

    cat_df = df_d[df_d["product_category_name"].isin(at_risk_cats)].copy()

    # MultiIndex ให้ครบทุก combo → ป้องกันชื่อหมวดหาย
    all_statuses = list(STATUS_COLORS.keys())
    idx = pd.MultiIndex.from_product(
        [at_risk_cats, all_statuses],
        names=["product_category_name", "status"],
    )
    cat_total = cat_df.groupby("product_category_name").size().rename("total")
    cat_stat  = (
        cat_df.groupby(["product_category_name", "status"])
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
    y_order = sort_order + [c for c in at_risk_cats if c not in sort_order]

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
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("💡 เรียงจากหมวดที่มี High Risk มากสุด (บนสุด) → น้อยสุด (ล่างสุด)")


def _render_behaviour(grp: pd.DataFrame) -> None:
    """พฤติกรรมกลุ่ม: ราคา ค่าส่ง ส่งช้า รอบซื้อ + หมวดสินค้า + วิธีจ่าย"""
    if grp.empty:
        st.info("ไม่มีข้อมูลในกลุ่มนี้")
        return

    # ── 4 KPI ─────────────────────────────────────────────────
    b1, b2, b3, b4 = st.columns(4)
    avg_price   = grp["price"].mean()          if "price"          in grp.columns else None
    avg_freight = grp["freight_value"].mean()  if "freight_value"  in grp.columns else None
    late_pct    = (grp["delay_days"] > 0).mean() * 100 if "delay_days" in grp.columns else None
    avg_cycle   = grp["cat_median_days"].mean() if "cat_median_days" in grp.columns else None

    b1.metric("🛒 ราคาสินค้าเฉลี่ย",
              f"R$ {avg_price:,.0f}" if avg_price is not None else "N/A")
    b2.metric("🚚 ค่าส่งเฉลี่ย",
              f"R$ {avg_freight:,.0f}" if avg_freight is not None else "N/A")
    b3.metric("⏰ โดนส่งช้า",
              f"{late_pct:.1f}%" if late_pct is not None else "N/A",
              delta_color="inverse")
    b4.metric("🔄 รอบซื้อเฉลี่ย",
              f"{avg_cycle:.0f} วัน" if avg_cycle is not None else "N/A")

    st.markdown("")
    c_cat, c_pay = st.columns(2)

    # ── Top 6 หมวดสินค้า ─────────────────────────────────────
    with c_cat:
        if "product_category_name" in grp.columns:
            top_cat = (
                grp["product_category_name"].value_counts()
                .head(6).reset_index()
            )
            top_cat.columns = ["หมวด", "จำนวน"]
            bar = (
                alt.Chart(top_cat).mark_bar(
                    color="#3498db",
                    cornerRadiusTopRight=4,
                    cornerRadiusBottomRight=4,
                )
                .encode(
                    x=alt.X("จำนวน:Q", title="จำนวนออเดอร์"),
                    y=alt.Y("หมวด:N", sort="-x", title=None),
                    tooltip=["หมวด", alt.Tooltip("จำนวน:Q", format=",")],
                )
                .properties(height=180, title="Top 6 หมวดสินค้า")
            )
            st.altair_chart(bar, use_container_width=True)

    # ── วิธีจ่ายเงิน ──────────────────────────────────────────
    with c_pay:
        if "payment_type" in grp.columns:
            pay_dist = (
                grp["payment_type"].value_counts(normalize=True)
                .mul(100).reset_index()
            )
            pay_dist.columns = ["วิธีจ่าย", "สัดส่วน (%)"]
            pie = (
                alt.Chart(pay_dist).mark_arc(innerRadius=40)
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
                .properties(height=180, title="วิธีชำระเงิน")
            )
            st.altair_chart(pie, use_container_width=True)
