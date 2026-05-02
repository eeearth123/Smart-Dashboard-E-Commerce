# ============================================================
# pages/p3_action.py — Action Plan & ROI Simulator
# ============================================================
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import time

from utils.helpers import safe_cats, assign_matrix_group, MATRIX_GROUPS, MATRIX_GROUP_LIST


def render(df: pd.DataFrame, t, model=None, feature_names=None, threshold: float = 0.55):
    st.title("🎯 Action Plan & Simulator")
    st.caption("จำลองผลกระทบโดยเปลี่ยนฟีเจอร์ → ทำนายซ้ำด้วยโมเดล → วัด Uplift จริง")

    df = assign_matrix_group(df.copy(), threshold=threshold)

    # ============================================================
    # TARGET SELECTOR — รองรับ 2 วิธีเลือกกลุ่ม
    # ============================================================
    with st.expander("🎯 กำหนดกลุ่มเป้าหมาย", expanded=True):
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            # ── วิธีที่ 1: Matrix group (ใหม่) ─────────────────
            # รับค่าจาก p2 ถ้ากด CTA button มา
            prefilter = st.session_state.pop("p3_prefilter_matrix", None)
            default_matrix = [prefilter] if prefilter in MATRIX_GROUP_LIST else []

            sel_matrix = st.multiselect(
                "🔲 กลุ่มจาก AI×Rule Matrix:",
                MATRIX_GROUP_LIST,
                default=default_matrix,
                key="p3_matrix",
                help=(
                    "🚨 ด่วน = ทั้ง AI และ Rule เห็นตรงกัน\n"
                    "🔍 Early Warning = AI จับก่อน Rule ยังไม่เห็น\n"
                    "⚠️ Monitor = Rule เห็นแต่ AI ยังให้โอกาส\n"
                    "✅ Active = ปกติดี"
                )
            )

        with fc2:
            # ── วิธีที่ 2: Status (เดิม) ────────────────────────
            status_opts = [
                "High Risk", "Warning (Late > 1.5x)",
                "Medium Risk", "Lost (Late > 3x)", "Active"
            ]
            sel_status = st.multiselect(
                "📊 หรือเลือกตาม Status:",
                status_opts,
                key="p3_status",
                help="ใช้แทน Matrix ได้ หรือใช้คู่กันเพื่อเจาะกลุ่มให้แม่นขึ้น"
            )

        with fc3:
            sel_cats = st.multiselect(
                "📦 หมวดสินค้า (ว่าง = ทุกหมวด):",
                safe_cats(df),
                key="p3_cat"
            )

    # ── Apply filters ─────────────────────────────────────────
    df_p3 = df.copy()
    if sel_matrix:
        df_p3 = df_p3[df_p3["matrix_group"].isin(sel_matrix)]
    if sel_status:
        df_p3 = df_p3[df_p3["status"].isin(sel_status)]
    if sel_cats:
        df_p3 = df_p3[df_p3["product_category_name"].isin(sel_cats)]

    # ── Filter summary ────────────────────────────────────────
    filter_parts = []
    if sel_matrix:
        filter_parts.append(f"Matrix: {', '.join(sel_matrix[:2])}{'…' if len(sel_matrix)>2 else ''}")
    if sel_status:
        filter_parts.append(f"Status: {', '.join(sel_status[:2])}{'…' if len(sel_status)>2 else ''}")
    if sel_cats:
        filter_parts.append(f"หมวด: {', '.join(sel_cats[:2])}{'…' if len(sel_cats)>2 else ''}")
    filter_msg = " | ".join(filter_parts) if filter_parts else "ภาพรวมทุกกลุ่ม"

    total_pop = len(df_p3)
    avg_ltv   = float(df_p3["payment_value"].mean()) if "payment_value" in df_p3.columns else 150.0

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.info(f"📊 กำลังวิเคราะห์: **{filter_msg}**")
    with c2:
        st.metric("👥 กลุ่มเป้าหมาย", f"{total_pop:,} คน")
    with c3:
        st.metric("💰 LTV เฉลี่ย/คน", f"R$ {avg_ltv:,.0f}")

    # แสดง breakdown ของ matrix_group ที่เลือก
    if total_pop > 0:
        mg_counts = df_p3["matrix_group"].value_counts()
        parts = []
        for grp, cnt in mg_counts.items():
            pct = cnt / total_pop * 100
            parts.append(f"**{grp}**: {cnt:,} คน ({pct:.1f}%)")
        if parts:
            st.caption("  ·  ".join(parts))

    st.markdown("---")

    # ============================================================
    # SIMULATION ENGINE
    # ============================================================
    def run_simulation(target_df, feature_changes: dict, cost_per_head: float,
                       tab_key: str, rec_text: str, strategy_name: str):
        n_target    = len(target_df)
        pct_problem = (n_target / total_pop * 100) if total_pop > 0 else 0

        c_prob, c_sol, c_res = st.columns([1, 1.3, 1])

        with c_prob:
            st.info(f"**📉 ปัญหา:** พบ {n_target:,} คน\n({pct_problem:.1f}% ของกลุ่มนี้)")
            st.progress(min(pct_problem / 100, 1.0))
            st.caption("แถบสีแสดงสัดส่วนคนที่มีปัญหา")
            if not target_df.empty:
                st.markdown("**📋 Feature เฉลี่ย:**")
                for col in list(feature_changes.keys())[:3]:
                    if col in target_df.columns:
                        st.caption(f"• {col}: {target_df[col].mean():.2f}")

            # Matrix breakdown ของ target group
            if "matrix_group" in target_df.columns and len(target_df) > 0:
                st.markdown("**🔲 Matrix group:**")
                mg = target_df["matrix_group"].value_counts(normalize=True)
                for grp, pct in mg.head(3).items():
                    icon = grp.split(" ")[0]
                    st.caption(f"{icon} {pct*100:.0f}%")

        with c_sol:
            st.markdown(f"**🛠️ วิธีแก้ไข: {strategy_name}**")
            st.write(rec_text)
            st.markdown("---")
            cost = st.number_input(
                "งบต่อหัว (R$)", value=float(cost_per_head),
                min_value=0.0, max_value=500.0, step=0.5,
                key=f"cost_{tab_key}"
            )
            break_even_rate = cost / avg_ltv if avg_ltv > 0 else 0
            st.caption(f"📐 จุดคุ้มทุน: ต้องสำเร็จ ≥ **{break_even_rate:.1%}**")

            if model is None or not feature_names:
                max_pot   = 15
                realistic = min(max_pot, 10) if cost >= 15 else min(max_pot, 5)
                st.markdown(f"**🤖 AI Prediction:** `{realistic}%`")
                st.caption("(โมเดลไม่พร้อม → ใช้ค่าประมาณ)")
                lift = st.slider("ปรับค่าคาดการณ์ความสำเร็จ (%)", 1, 100,
                                 realistic, key=f"lift_{tab_key}")
                sim_success_rate = lift / 100
                sim_mode = "manual"
            else:
                sim_mode = "model"
                lift     = None

        with c_res:
            with st.spinner("⚡ โมเดลกำลังจำลอง..."):
                time.sleep(0.3)

                if sim_mode == "model" and not target_df.empty:
                    X_orig    = target_df.reindex(columns=feature_names, fill_value=0).fillna(0)
                    prob_orig = model.predict_proba(X_orig)[:, 1]

                    df_sim = target_df.copy()
                    for col, (op, val) in feature_changes.items():
                        if col in df_sim.columns:
                            if op == "set":          df_sim[col] = val
                            elif op == "multiply":   df_sim[col] = df_sim[col] * val
                            elif op == "clip_upper": df_sim[col] = df_sim[col].clip(upper=val)
                            elif op == "add":        df_sim[col] = df_sim[col] + val

                    if "freight_value" in df_sim.columns and "price" in df_sim.columns:
                        df_sim["freight_ratio"] = (
                            df_sim["freight_value"] /
                            df_sim["price"].replace(0, np.nan)
                        ).fillna(0)

                    X_sim    = df_sim.reindex(columns=feature_names, fill_value=0).fillna(0)
                    prob_sim = model.predict_proba(X_sim)[:, 1]
                    uplift_arr       = prob_orig - prob_sim
                    sim_success_rate = (uplift_arr > 0.08).mean()

                    dist = {
                        "ตอบสนองสูง\n(>15%)":     int((uplift_arr > 0.15).sum()),
                        "ปานกลาง\n(8–15%)":        int(((uplift_arr > 0.08) & (uplift_arr <= 0.15)).sum()),
                        "ต่ำ\n(0–8%)":             int(((uplift_arr > 0) & (uplift_arr <= 0.08)).sum()),
                        "ไม่ตอบสนอง":              int((uplift_arr <= 0).sum()),
                    }
                    dist_df = pd.DataFrame({
                        "กลุ่ม": list(dist.keys()),
                        "จำนวน": list(dist.values())
                    })
                    st.altair_chart(
                        alt.Chart(dist_df).mark_bar().encode(
                            x=alt.X("กลุ่ม", sort=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("จำนวน"),
                            color=alt.Color("กลุ่ม", scale=alt.Scale(
                                domain=list(dist.keys()),
                                range=["#2ecc71","#f1c40f","#e67e22","#95a5a6"]
                            ), legend=None),
                            tooltip=["กลุ่ม","จำนวน"]
                        ).properties(height=160, title="📊 Uplift Distribution"),
                        use_container_width=True
                    )
                else:
                    sim_success_rate = lift / 100 if lift else 0.1

                budget      = n_target * cost
                saved_users = int(n_target * sim_success_rate)
                revenue     = saved_users * avg_ltv
                profit      = revenue - budget
                roi         = (profit / budget * 100) if budget > 0 else 0
                be_final    = cost / avg_ltv if avg_ltv > 0 else 0

                st.markdown("**🚀 ผลลัพธ์**")
                st.metric("🤖 Success Rate (โมเดล)", f"{sim_success_rate:.1%}",
                          delta=f"จุดคุ้มทุน {be_final:.1%}")
                st.metric("👥 ดึงลูกค้าคืน",   f"{saved_users:,} คน")
                st.metric("💸 งบประมาณ",        f"R$ {budget:,.0f}")

                if profit > 0:
                    st.metric("📈 กำไรสุทธิ (ROI)", f"R$ {profit:,.0f}", f"+{roi:.1f}%")
                    st.success("✅ **คุ้มค่าการลงทุน!**")
                else:
                    st.metric("📉 ขาดทุนสุทธิ", f"R$ {profit:,.0f}", f"{roi:.1f}%")
                    gap = be_final - sim_success_rate
                    st.error(
                        f"⚠️ **ขาดทุน!**\n\n"
                        f"ต้องการ Success Rate: **{be_final:.1%}**\n"
                        f"ได้จริง: **{sim_success_rate:.1%}**\n"
                        f"ขาดอีก: **{gap:.1%}**"
                    )
                    max_cost_be = avg_ltv * sim_success_rate
                    st.caption(f"💡 ลดงบต่อหัวเหลือ **R$ {max_cost_be:.0f}** เพื่อเริ่มกำไร")

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚚 1. ส่งฟรี / ลดค่าส่ง",
        "💵 2. ส่วนลดสินค้า",
        "❤️ 3. ง้อลูกค้าส่งช้า",
        "🛍️ 4. ขายพ่วง / Cross-sell"
    ])

    with tab1:
        st.subheader("🚚 กลุ่มค่าส่งแพงเกินรับไหว (Freight Pain)")
        if "freight_ratio" in df_p3.columns:
            target_t1   = df_p3[df_p3["freight_ratio"] > 0.2].copy()
            avg_freight = float(target_t1["freight_value"].mean()) \
                          if (not target_t1.empty and "freight_value" in target_t1.columns) else 15.0
            run_simulation(
                target_df=target_t1,
                feature_changes={"freight_value": ("set", 0), "freight_ratio": ("set", 0)},
                cost_per_head=avg_freight, tab_key="tab1",
                strategy_name="ส่งฟรี (Free Shipping)",
                rec_text=(
                    f"ลูกค้าลังเลเพราะค่าส่งแพง (เฉลี่ย R$ {avg_freight:.0f})\n\n"
                    "👉 **Action:** ตั้ง `freight_value = 0` แล้วให้โมเดลทำนายซ้ำ"
                )
            )
        else:
            st.error("ไม่พบข้อมูล freight_ratio")

    with tab2:
        st.subheader("💵 กลุ่มเสี่ยง Churn (Price Sensitivity)")
        disc_pct = st.radio("เลือก % ส่วนลด:", [10, 20], horizontal=True, key="disc_pct_t2")
        if "price" in df_p3.columns:
            target_t2 = df_p3[df_p3["churn_probability"] > 0.5].copy()
            disc_cost = float(avg_ltv * disc_pct / 100)
            run_simulation(
                target_df=target_t2,
                feature_changes={
                    "price":         ("multiply", 1 - disc_pct/100),
                    "payment_value": ("multiply", 1 - disc_pct/100),
                },
                cost_per_head=disc_cost, tab_key="tab2",
                strategy_name=f"ส่วนลดสินค้า {disc_pct}%",
                rec_text=(
                    f"ลด `price` ลง {disc_pct}% แล้วให้โมเดลทำนายซ้ำ\n\n"
                    f"👉 **Action:** เสนอ Coupon {disc_pct}% เฉพาะลูกค้า churn_prob > 50%"
                )
            )
        else:
            st.error("ไม่พบข้อมูล price")

    with tab3:
        st.subheader("❤️ กลุ่มโดนเท / ของส่งช้า (Delay Recovery)")
        if "delay_days" in df_p3.columns:
            target_t3 = df_p3[df_p3["delay_days"] > 0].copy()
            run_simulation(
                target_df=target_t3,
                feature_changes={
                    "delay_days":            ("set", 0),
                    "delivery_vs_estimated": ("clip_upper", 0),
                },
                cost_per_head=15.0, tab_key="tab3",
                strategy_name="SMS ขอโทษ + คูปองชดเชย",
                rec_text=(
                    "ตั้ง `delay_days = 0` (สมมติว่าปัญหาได้รับการแก้ไข)\n\n"
                    "👉 **Action:** ส่ง SMS ขอโทษทันที + แนบ Coupon ส่วนลดพิเศษ"
                )
            )
        else:
            st.error("ไม่พบข้อมูล delay_days")

    with tab4:
        st.subheader("🛍️ กลุ่มซื้อหมวดเสี่ยง Churn สูง")
        if "cat_churn_risk" in df_p3.columns:
            target_t4 = df_p3[df_p3["cat_churn_risk"] > 0.8].copy()
            run_simulation(
                target_df=target_t4,
                feature_changes={
                    "cat_churn_risk":       ("multiply", 0.6),
                    "payment_installments": ("add", 2),
                },
                cost_per_head=10.0, tab_key="tab4",
                strategy_name="Cross-sell + ผ่อนได้นานขึ้น",
                rec_text=(
                    "ลด `cat_churn_risk` ลง 40% (จาก cross-sell หมวดซื้อซ้ำ)\n\n"
                    "👉 **Action:** ยิงแอดสินค้า Housewares + เพิ่ม installments"
                )
            )
        else:
            st.error("ไม่พบข้อมูล cat_churn_risk")
