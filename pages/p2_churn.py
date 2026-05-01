# ============================================================
# pages/p2_churn.py — Churn Overview (Redesigned)
# ============================================================
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from utils.helpers import safe_cats, status_display_options, assign_matrix_group, MATRIX_GROUPS

# ── colour palette (ใช้ทั้งไฟล์) ────────────────────────────
STATUS_COLORS = {
    "Active":                  "#2ecc71",
    "Medium Risk":             "#f1c40f",
    "Warning (Late > 1.5x)":   "#e67e22",
    "High Risk":               "#e74c3c",
    "Lost (Late > 3x)":        "#95a5a6",
}
MATRIX_COLORS = {
    "🚨 ด่วน (AI+Rule เห็นตรง)":           "#e74c3c",
    "🔍 Early Warning (AI เห็นก่อน)":       "#2980b9",
    "⚠️ Monitor (Rule เห็น, AI ยังให้โอกาส)": "#e67e22",
    "✅ Active":                             "#2ecc71",
}


def render(df: pd.DataFrame, t, threshold: float = 0.55):
    st.title(t("page_churn_title", default="📊 Churn Overview"))
    st.caption(
        "ทำความเข้าใจภาพรวม → เลือกกลุ่มที่อยากโฟกัส → ไปวางแผนแคมเปญที่หน้าถัดไป"
    )

    # ── ตรวจสอบ columns ที่จำเป็น ────────────────────────────
    required = ["churn_probability", "lateness_score", "status", "payment_value"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ ไม่พบ columns: {missing}")
        return

    # ── เพิ่ม matrix_group ────────────────────────────────────
    df = assign_matrix_group(df.copy(), threshold=threshold)

    # ── Filter bar ───────────────────────────────────────────
    with st.expander("🌪️ กรองข้อมูล", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            sel_cats = st.multiselect(
                "หมวดสินค้า (ว่าง = ทั้งหมด):",
                safe_cats(df), key="p2_cat"
            )
        with col_b:
            display_list, to_internal, to_display = status_display_options(t)
            sel_status_display = st.multiselect(
                "สถานะ (ว่าง = ทั้งหมด):", display_list, key="p2_status"
            )

    df_d = df.copy()
    if sel_cats:
        df_d = df_d[df_d["product_category_name"].isin(sel_cats)]
    if sel_status_display:
        sel_internal = [to_internal[s] for s in sel_status_display]
        df_d = df_d[df_d["status"].isin(sel_internal)]

    total = len(df_d)
    if total == 0:
        st.warning("ไม่มีข้อมูลตาม filter ที่เลือก")
        return

    # ============================================================
    # SECTION 1: KPIs — แยก Rule-based / AI / Context ชัดเจน
    # ============================================================
    st.markdown("---")

    # ── คำนวณ ────────────────────────────────────────────────
    at_risk_mask = df_d["status"].isin(["High Risk", "Warning (Late > 1.5x)"])
    lost_mask    = df_d["status"] == "Lost (Late > 3x)"
    ai_churn_mask = df_d["churn_probability"] >= threshold

    at_risk_n    = at_risk_mask.sum()
    lost_n       = lost_mask.sum()
    ai_churn_n   = ai_churn_mask.sum()
    rev_at_risk  = df_d.loc[at_risk_mask, "payment_value"].sum()
    avg_cycle    = (df_d["cat_median_days"].mean()
                   if "cat_median_days" in df_d.columns else None)

    # ── Row 1: Rule-based ─────────────────────────────────────
    st.caption("📏 **Rule-based** — คิดจาก lateness score เทียบรอบซื้อปกติ")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("🚨 At-Risk (Rule)",
                f"{at_risk_n:,} คน",
                f"{at_risk_n/total*100:.1f}%")
    r1c2.metric("💸 Revenue at Risk",
                f"R$ {rev_at_risk:,.0f}",
                help="ยอดรายได้จากกลุ่ม High Risk + Warning")
    r1c3.metric("⚰️ Lost",
                f"{lost_n:,} คน",
                f"{lost_n/total*100:.1f}%",
                delta_color="inverse",
                help="หายไปนานกว่า 3 เท่าของรอบปกติ — rule ถือว่าสูญไปแล้ว")
    r1c4.metric("🔄 Avg Repurchase Cycle",
                f"{avg_cycle:.0f} วัน" if avg_cycle else "N/A",
                help="ค่าเฉลี่ยรอบซื้อซ้ำ ใช้คำนวณ lateness score")

    st.markdown("")
    # ── Row 2: AI ─────────────────────────────────────────────
    st.caption("🤖 **AI Model** — คิดจาก churn_probability ≥ {:.0f}%".format(threshold*100))
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    early_n  = (df_d["matrix_group"] == MATRIX_GROUPS["early"]).sum()
    urgent_n = (df_d["matrix_group"] == MATRIX_GROUPS["urgent"]).sum()
    r2c1.metric("🤖 AI Predicted Churn",
                f"{ai_churn_n:,} คน",
                f"{ai_churn_n/total*100:.1f}%")
    r2c2.metric("🔍 Early Warning",
                f"{early_n:,} คน",
                f"{early_n/total*100:.1f}%",
                help="AI เห็นก่อน Rule — คนกลุ่มนี้ยังไม่ช้าแต่พฤติกรรมเปลี่ยน")
    r2c3.metric("🚨 ทั้ง AI + Rule เห็น",
                f"{urgent_n:,} คน",
                f"{urgent_n/total*100:.1f}%",
                delta_color="inverse",
                help="กลุ่มที่ต้องทำแคมเปญด่วนที่สุด")

    # Lost → AI predict ว่าอยู่ใน group ไหน
    lost_df = df_d[lost_mask]
    if len(lost_df) > 0:
        lost_ai_churn = (lost_df["churn_probability"] >= threshold).mean()
        r2c4.metric("🤖 Lost: AI predict churn",
                    f"{lost_ai_churn*100:.0f}%",
                    help=(
                        f"จาก {len(lost_df):,} คนที่ Rule ถือว่า Lost แล้ว\n"
                        f"AI ยังให้โอกาสว่า {(1-lost_ai_churn)*100:.0f}% อาจกลับมาได้"
                    ))
    else:
        r2c4.metric("⚰️ Lost group", "ไม่มีข้อมูล")

    # ============================================================
    # SECTION 2: 2×2 Matrix + Category Breakdown
    # ============================================================
    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    # ── 2×2 Agreement Matrix ──────────────────────────────────
    with col_left:
        st.subheader("🔲 AI × Rule-based — เห็นตรงกันไหม?")
        st.caption(
            "กลุ่ม **Early Warning** คือ value ของโมเดล — "
            "จับได้ก่อน rule เห็น ถ้าไม่มี AI จะมองไม่เห็นกลุ่มนี้เลย"
        )

        # คำนวณ 4 กลุ่ม
        ai_on   = df_d["churn_probability"] >= threshold
        rule_on = df_d["lateness_score"] >= 1.5

        groups = {
            "🚨 ด่วน\n(AI+Rule)":             ( ai_on &  rule_on),
            "🔍 Early Warning\n(AI เห็นก่อน)": ( ai_on & ~rule_on),
            "⚠️ Monitor\n(Rule เห็น)":          (~ai_on &  rule_on),
            "✅ Active":                         (~ai_on & ~rule_on),
        }
        cells = []
        for label, mask in groups.items():
            n   = mask.sum()
            pct = n / total * 100
            rev = df_d.loc[mask, "payment_value"].sum() if "payment_value" in df_d.columns else 0
            cells.append({"กลุ่ม": label, "จำนวน (คน)": n,
                          "สัดส่วน (%)": pct, "Revenue (R$)": rev})

        mat_df = pd.DataFrame(cells)

        # แสดงเป็น 2×2 ด้วย columns
        row1 = st.columns(2)
        row2 = st.columns(2)

        display_pairs = list(zip([row1[0], row1[1], row2[0], row2[1]], cells))
        cell_styles = [
            ("🔴 ด่วนที่สุด",    "#fde8e8", "#a32d2d",  "ทำแคมเปญทันที"),
            ("🔵 Early Warning", "#e6f1fb", "#185fa5",  "ส่ง engagement ก่อนสาย"),
            ("🟡 ติดตาม",        "#faeeda", "#854f0b",  "monitor อีก 2 สัปดาห์"),
            ("🟢 ดีอยู่",         "#eaf3de", "#3b6d11",  "รักษาประสบการณ์ดีต่อไป"),
        ]
        for (col, data), (tag, bg, color, action) in zip(display_pairs, cell_styles):
            with col:
                st.markdown(
                    f"""
                    <div style="background:{bg};border-radius:10px;padding:12px;margin:4px 0">
                      <div style="font-size:11px;font-weight:500;color:{color}">{tag}</div>
                      <div style="font-size:22px;font-weight:500;color:{color}">{data['จำนวน (คน)']:,}</div>
                      <div style="font-size:11px;color:{color}">{data['สัดส่วน (%)']:.1f}%</div>
                      <div style="font-size:11px;color:{color};margin-top:4px;opacity:.75">{data['กลุ่ม']}</div>
                      <div style="font-size:11px;color:{color};margin-top:4px;font-weight:500">→ {action}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ── Lost group — AI predict ไว้ใน matrix ไหน ─────────
        if len(lost_df) > 0:
            st.markdown("---")
            st.caption(f"**⚰️ Lost group ({len(lost_df):,} คน)** — Rule ถือว่าสูญไปแล้ว แต่ AI มองว่า:")
            lost_dist = lost_df["matrix_group"].value_counts().reset_index()
            lost_dist.columns = ["matrix_group", "count"]
            lost_dist["pct"] = lost_dist["count"] / len(lost_df) * 100

            lost_bar = alt.Chart(lost_dist).mark_bar(cornerRadiusTopLeft=4,
                                                      cornerRadiusTopRight=4).encode(
                x=alt.X("count:Q", title="จำนวนคน"),
                y=alt.Y("matrix_group:N", sort="-x", title=None),
                color=alt.Color("matrix_group:N",
                    scale=alt.Scale(
                        domain=list(MATRIX_COLORS.keys()),
                        range=list(MATRIX_COLORS.values())
                    ), legend=None),
                tooltip=[
                    alt.Tooltip("matrix_group", title="กลุ่ม"),
                    alt.Tooltip("count", format=",", title="จำนวน"),
                    alt.Tooltip("pct", format=".1f", title="%"),
                ]
            ).properties(height=120, title="Lost customers → AI จัดอยู่กลุ่มไหน?")
            st.altair_chart(lost_bar, use_container_width=True)
            st.caption(
                "💡 ถ้า Lost + AI Early Warning → ยังพอดึงกลับได้ ลองแคมเปญ re-engagement"
            )

    # ── Category Breakdown ────────────────────────────────────
    with col_right:
        st.subheader("📦 สัดส่วนความเสี่ยงต่อหมวดสินค้า")
        st.caption(
            "หมวดไหนมีสีแดง/ส้มเยอะ = ลูกค้าหนีสูง "
            "→ เลือก filter หมวดนั้นก่อนไปหน้า Action Plan"
        )

        if "product_category_name" not in df_d.columns:
            st.info("ไม่มีข้อมูล product_category_name")
        else:
            # Top 12 หมวดที่มี at-risk เยอะสุด
            cat_risk = (
                df_d[df_d["status"].isin(["High Risk", "Warning (Late > 1.5x)"])]
                .groupby("product_category_name")
                .size()
                .sort_values(ascending=False)
                .head(12)
                .index.tolist()
            )
            if not cat_risk:
                cat_risk = df_d["product_category_name"].value_counts().head(12).index.tolist()

            cat_df = df_d[df_d["product_category_name"].isin(cat_risk)].copy()
            cat_stat = (
                cat_df.groupby(["product_category_name", "status"])
                .size()
                .reset_index(name="count")
            )
            cat_total = cat_df.groupby("product_category_name").size().rename("total")
            cat_stat  = cat_stat.merge(cat_total, on="product_category_name")
            cat_stat["pct"] = cat_stat["count"] / cat_stat["total"] * 100

            # เรียงตาม % High Risk สูงสุด
            high_order = (
                cat_stat[cat_stat["status"] == "High Risk"]
                .sort_values("pct", ascending=False)["product_category_name"]
                .tolist()
            )
            remaining = [c for c in cat_risk if c not in high_order]
            sort_order = high_order + remaining

            chart = alt.Chart(cat_stat).mark_bar().encode(
                x=alt.X("pct:Q", stack="normalize",
                        axis=alt.Axis(format="%", title="สัดส่วน")),
                y=alt.Y("product_category_name:N",
                        sort=sort_order, title=None),
                color=alt.Color("status:N",
                    scale=alt.Scale(
                        domain=list(STATUS_COLORS.keys()),
                        range=list(STATUS_COLORS.values())
                    ),
                    legend=alt.Legend(orient="bottom", title=None)),
                order=alt.Order("status:N"),
                tooltip=[
                    alt.Tooltip("product_category_name", title="หมวด"),
                    alt.Tooltip("status", title="สถานะ"),
                    alt.Tooltip("count", format=",", title="จำนวน"),
                    alt.Tooltip("pct", format=".1f", title="%"),
                ]
            ).properties(height=380)

            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "💡 เรียงจากหมวดที่มี High Risk มากสุด (บนสุด) → น้อยสุด (ล่างสุด)"
            )

    # ============================================================
    # SECTION 3: CTA → Action Plan
    # ============================================================
    st.markdown("---")
    urgent_n_final = (df_d["matrix_group"] == MATRIX_GROUPS["urgent"]).sum()
    early_n_final  = (df_d["matrix_group"] == MATRIX_GROUPS["early"]).sum()

    cta_col1, cta_col2 = st.columns([3, 1])
    with cta_col1:
        st.info(
            f"**พร้อมแล้ว? ไปวางแผนแคมเปญที่หน้าถัดไป** 🎯\n\n"
            f"แนะนำเริ่มจาก **กลุ่ม Urgent ({urgent_n_final:,} คน)** ก่อน — "
            f"ทั้ง AI และ Rule เห็นตรงกัน ความเสี่ยงขาดทุนต่ำที่สุด\n\n"
            f"หรือ **Early Warning ({early_n_final:,} คน)** ถ้าอยาก "
            f"ทำ retention ก่อนที่จะสายเกินไป"
        )
    with cta_col2:
        if st.button("🎯 ไปหน้า Action Plan →", use_container_width=True, type="primary"):
            st.session_state["p3_prefilter_matrix"] = MATRIX_GROUPS["urgent"]
            st.session_state["active_page"] = "3. 🎯 Action Plan"
            st.rerun()
