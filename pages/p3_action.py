# ============================================================
# pages/p3_action.py — Action Plan & Simulator (v5.0 - 3-Class Calibrated)
# ============================================================
import time
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from i18n import t
from utils.helpers import safe_cats, assign_matrix_group, MATRIX_GROUPS

FILTER_GROUPS = {
    "🟩 Active":        ("status",       "Active"),
    "🟨 Medium Risk":   ("status",       "Medium Risk"),
    "🟧 Warning":       ("status",       "Warning (Late > 1.5x)"),
    "🟥 High Risk":     ("status",       "High Risk"),
    "⬛ Lost":          ("status",       "Lost (Late > 3x)"),
    "🚨 Urgent":        ("matrix_group", MATRIX_GROUPS["urgent"]),
    "🔍 Early Warning": ("matrix_group", MATRIX_GROUPS["early"]),
    "⚠️ Monitor":       ("matrix_group", MATRIX_GROUPS["monitor"]),
}


def render(df: pd.DataFrame, model, feature_names: list) -> None:
    st.title(t("page_action"))
    st.caption(t("p3_caption"))
    st.caption("🏷️ **Model Version:** V5 (Hybrid Calibrated 3-Class)")

    if model is None or not feature_names:
        st.error("❌ โมเดลไม่พร้อม — กรุณาตรวจสอบไฟล์ modelV5.pkl ใน repo")
        st.stop()

    df = assign_matrix_group(df.copy())

    # ── Filter เหมือนหน้า 2 ───────────────────────────────────
    with st.expander(t("p3_target_exp"), expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            sel_groups = st.multiselect(
                "กลุ่ม (ว่าง = ทั้งหมด):",
                list(FILTER_GROUPS.keys()),
                default=["🚨 Urgent", "🟥 High Risk"],
                key="p3_group",
            )
        with f2:
            sel_cats = st.multiselect(t("cat_label"), safe_cats(df), key="p3_cat")

        st.markdown("""
**📖 คำอธิบายกลุ่ม (อิงจาก 3-Class Model):**

| กลุ่ม | เงื่อนไข | ความหมาย |
|---|---|---|
| 🟩 Active | AI ทายว่า Stay (ปกติ) | ลูกค้าปกติ ยังเหนียวแน่น |
| 🟨 Medium Risk | AI ความเสี่ยง Churn > 50% | AI เริ่มเห็นสัญญาณเสี่ยง |
| 🟧 Warning | AI ทายว่า Delay หรือ Late > 1.5x | มีปัญหาเรื่องเวลา เสี่ยงระดับกลาง |
| 🟥 High Risk | AI ทายว่า Churn หรือ ความเสี่ยง > 75% | AI ฟันธงว่าไปแน่ เสี่ยงระดับสูง |
| ⬛ Lost | Late > 3.0x | หายไปนานมากแล้ว rule ถือว่าสูญ |
| 🚨 Urgent | AI ชี้เป้า Churn **และ** Late > 1.5x | ทั้งคู่เห็นตรงกัน — ด่วนที่สุด |
| 🔍 Early Warning | AI ชี้เป้า Churn **แต่** Late ≤ 1.5x | AI เห็นก่อน rule — ยังมีเวลา |
| ⚠️ Monitor | AI ไม่ชี้เป้า Churn **แต่** Late > 1.5x | rule เห็น AI ยังให้โอกาส |
        """)

    df_p3 = df.copy()
    if sel_groups:
        masks = []
        for g in sel_groups:
            col, val = FILTER_GROUPS[g]
            masks.append(df_p3[col] == val)
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        df_p3 = df_p3[combined]
    if sel_cats:
        df_p3 = df_p3[df_p3["product_category_name"].isin(sel_cats)]

    filter_msg = ", ".join(sel_groups[:2]) + ("..." if len(sel_groups) > 2 else "") \
                 if sel_groups else t("p3_all_groups")
    total_pop  = len(df_p3)
    avg_ltv    = float(df_p3["payment_value"].mean()) \
                 if "payment_value" in df_p3.columns and total_pop > 0 else 150.0

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: st.info(t("p3_analyzing", g=filter_msg))
    with c2: st.metric(t("p3_target_pop"), f"{total_pop:,}{t('people_unit')}")
    with c3: st.metric(t("p3_avg_ltv"),    f"R$ {avg_ltv:,.0f}")
    st.markdown("---")

    tab1, tab2 = st.tabs([
        t("p3_tab1"), t("p3_tab2")
    ])

    with tab1:
        st.subheader(t("p3_t1_title"))
        if "freight_ratio" not in df_p3.columns:
            st.error(t("p3_no_freight"))
        else:
            freight_threshold = st.slider(
                "เกณฑ์สัดส่วนค่าส่งต่อราคาสินค้า (0.0 = จำลองทุกคน, 0.2 = ค่าส่ง > 20% ของราคาสินค้า):",
                min_value=0.0, max_value=1.0, value=0.20, step=0.05,
                key="freight_threshold_slider"
            )
            target    = df_p3[df_p3["freight_ratio"] >= freight_threshold].copy()
            avg_fr    = float(target["freight_value"].mean()) \
                        if not target.empty and "freight_value" in target.columns else 15.0
            _run_simulation(
                target,
                {"freight_value": ("set", 0), "freight_ratio": ("set", 0)},
                avg_fr, "tab1", t("p3_t1_strategy"), t("p3_t1_rec", avg=avg_fr),
                total_pop, avg_ltv, model, feature_names,
            )

    with tab2:
        st.subheader(t("p3_t2_title"))
        disc_pct = st.radio(t("p3_t2_disc"), [10, 20], horizontal=True, key="disc_t2")
        if "price" not in df_p3.columns:
            st.error(t("p3_no_price"))
        else:
            target = df_p3[df_p3["churn_probability"] > 0.5].copy()
            _run_simulation(
                target,
                {"price": ("multiply", 1 - disc_pct/100),
                 "payment_value": ("multiply", 1 - disc_pct/100)},
                float(avg_ltv * disc_pct / 100), "tab2",
                t("p3_t2_strategy", d=disc_pct), t("p3_t2_rec", d=disc_pct),
                total_pop, avg_ltv, model, feature_names,
            )




def _apply_changes(df_sim, feature_changes):
    for col, (op, val) in feature_changes.items():
        if col in df_sim.columns:
            if op == "set":          df_sim[col] = val
            elif op == "multiply":   df_sim[col] = df_sim[col] * val
            elif op == "clip_upper": df_sim[col] = df_sim[col].clip(upper=val)
            elif op == "add":        df_sim[col] = df_sim[col] + val
    if "freight_value" in df_sim.columns and "price" in df_sim.columns:
        df_sim["freight_ratio"] = (
            df_sim["freight_value"] / df_sim["price"].replace(0, np.nan)
        ).fillna(0)
    return df_sim


def _run_simulation(target_df, feature_changes, cost_per_head,
                    tab_key, strategy_name, rec_text,
                    total_pop, avg_ltv, model, feature_names):
    n_target    = len(target_df)
    pct_problem = (n_target / total_pop * 100) if total_pop > 0 else 0

    c_prob, c_sol, c_res = st.columns([1, 1.2, 1])

    with c_prob:
        st.info(t("p3_problem", n=f"{n_target:,}", pct=pct_problem))
        st.progress(min(pct_problem / 100, 1.0))
        if not target_df.empty:
            st.markdown(t("p3_feat_avg"))
            for col in list(feature_changes.keys())[:3]:
                if col in target_df.columns:
                    st.caption(f"• {col}: {target_df[col].mean():.2f}")

    with c_sol:
        st.markdown(t("p3_strategy", name=strategy_name))
        st.write(rec_text)
        st.markdown("---")
        cost = st.number_input(
            t("p3_cost_lbl"), value=float(cost_per_head),
            min_value=0.0, max_value=500.0, step=0.5, key=f"cost_{tab_key}",
        )
        be_rate = cost / avg_ltv if avg_ltv > 0 else 0
        st.caption(t("p3_breakeven", r=be_rate))

    with c_res:
        with st.spinner(t("p3_simulating")):
            time.sleep(0.3)

            if target_df.empty:
                st.warning("ไม่มีข้อมูลเป้าหมาย")
                return

            meds   = target_df.reindex(columns=feature_names).median().fillna(0)
            X_orig = target_df.reindex(columns=feature_names).fillna(meds)

            # Support 3-class V5 model
            proba_orig_all = model.predict_proba(X_orig)
            if proba_orig_all.shape[1] == 3:
                prob_orig = proba_orig_all[:, 2] # Class 2 = Churn
            elif proba_orig_all.shape[1] == 2:
                prob_orig = proba_orig_all[:, 1]
            else:
                prob_orig = 1 - proba_orig_all[:, 0]

            df_sim = _apply_changes(target_df.copy(), feature_changes)
            X_sim  = df_sim.reindex(columns=feature_names).fillna(meds)

            proba_sim_all = model.predict_proba(X_sim)
            if proba_sim_all.shape[1] == 3:
                prob_sim = proba_sim_all[:, 2] # Class 2 = Churn
            elif proba_sim_all.shape[1] == 2:
                prob_sim = proba_sim_all[:, 1]
            else:
                prob_sim = 1 - proba_sim_all[:, 0]

            uplift           = prob_orig - prob_sim
            sim_success_rate = float(uplift.mean()) if len(uplift) > 0 else 0.0

            # Uplift distribution chart
            dist = {
                t("p3_resp_high"): int((uplift > 0.15).sum()),
                t("p3_resp_mid"):  int(((uplift > 0.08) & (uplift <= 0.15)).sum()),
                t("p3_resp_low"):  int(((uplift > 0) & (uplift <= 0.08)).sum()),
                t("p3_resp_none"): int((uplift <= 0).sum()),
            }
            dist_df = pd.DataFrame({"Group": list(dist.keys()),
                                    "Count": list(dist.values())})
            st.altair_chart(
                alt.Chart(dist_df).mark_bar().encode(
                    x=alt.X("Group", sort=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Count"),
                    color=alt.Color("Group",
                        scale=alt.Scale(
                            domain=list(dist.keys()),
                            range=["#2ecc71","#f1c40f","#e67e22","#95a5a6"]
                        ), legend=None),
                    tooltip=["Group","Count"],
                ).properties(height=160, title=t("p3_uplift_chart")),
                use_container_width=True,
            )

            # ROI Metrics
            budget      = n_target * cost
            saved_users = int(n_target * sim_success_rate)
            profit      = saved_users * avg_ltv - budget
            roi         = (profit / budget * 100) if budget > 0 else 0

            st.markdown(t("p3_results"))
            st.metric(t("p3_success"), f"{sim_success_rate:.1%}",
                      delta=t("p3_be_delta", r=be_rate))
            st.metric(t("p3_saved"),   f"{saved_users:,}{t('people_unit')}")
            st.metric(t("p3_budget"),  f"R$ {budget:,.0f}")

            if profit > 0:
                st.metric(t("p3_profit"), f"R$ {profit:,.0f}", f"+{roi:.1f}%")
                st.success(t("p3_worthit"))
            else:
                gap = be_rate - sim_success_rate
                st.metric(t("p3_loss"), f"R$ {profit:,.0f}", f"{roi:.1f}%")
                st.error(t("p3_not_worth", be=be_rate, sr=sim_success_rate, gap=gap))
                st.caption(t("p3_reduce_cost", c=avg_ltv * sim_success_rate))

    # ── 🎯 Customer Profile Insights for SAVED CUSTOMERS ONLY ────────
    st.markdown("---")
    st.markdown(f"#### 🎯 ข้อมูลเชิงลึกโปรไฟล์คนที่สามารถดึงกลับมาได้ (Saved Customers Insights: **{saved_users:,} คน**)")

    # Filter target_df to ONLY the saved/recovered customers
    if saved_users > 0 and not target_df.empty:
        target_copy = target_df.copy()
        target_copy["_uplift"] = uplift
        # Select top saved_users responding positively to intervention
        saved_df = target_copy[target_copy["_uplift"] > 0].sort_values("_uplift", ascending=False).head(saved_users)
        if saved_df.empty:
            saved_df = target_copy.sort_values("_uplift", ascending=False).head(saved_users)
    else:
        saved_df = pd.DataFrame()

    if saved_df.empty:
        st.info("💡 ไม่มีกลุ่มคนที่สามารถดึงกลับมาได้จากมาตรการนี้")
        return

    ic1, ic2, ic3, ic4 = st.columns([1, 1.2, 1, 1.1])

    # 1. Donut Chart: Repeat vs First-time Buyers (Saved Group)
    with ic1:
        st.caption("🍩 **สัดส่วนประเภทลูกค้าที่รอด**")
        if "is_first_purchase" in saved_df.columns:
            first_cnt  = int((saved_df["is_first_purchase"] == 1).sum())
            repeat_cnt = int(len(saved_df) - first_cnt)
        else:
            first_cnt  = int(len(saved_df) * 0.3)
            repeat_cnt = len(saved_df) - first_cnt

        donut_df = pd.DataFrame({
            "Category": ["ซื้อซ้ำ (Repeat Buyers)", "ซื้อครั้งแรก (First-time)"],
            "Count": [repeat_cnt, first_cnt]
        })
        donut_chart = alt.Chart(donut_df).mark_arc(innerRadius=45).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Category", type="nominal",
                            scale=alt.Scale(range=["#FF7043", "#3498db"]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=["Category", "Count"]
        ).properties(height=210)
        st.altair_chart(donut_chart, use_container_width=True)

    # 2. Top Product Categories of Saved Customers
    with ic2:
        st.caption("🛍️ **หมวดสินค้าขายดีของกลุ่มที่รอด**")
        if "product_category_name" in saved_df.columns:
            top_saved_cats = (
                saved_df["product_category_name"]
                .value_counts()
                .head(5)
                .reset_index()
            )
            top_saved_cats.columns = ["Category", "Count"]
            cat_chart = alt.Chart(top_saved_cats).mark_bar(color="#2ecc71", cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
                x=alt.X("Count:Q", title="จำนวนคน"),
                y=alt.Y("Category:N", sort="-x", title=None),
                tooltip=["Category", "Count"]
            ).properties(height=210)
            st.altair_chart(cat_chart, use_container_width=True)
        else:
            st.caption("ไม่มีข้อมูลหมวดสินค้า")

    # 3. Days Overdue vs Expected Cycle Details
    with ic3:
        st.caption("⏳ **ระยะเวลาและการเกินรอบ**")
        avg_days = float(saved_df["days_since_purchase"].mean()) if "days_since_purchase" in saved_df.columns else 90.0
        avg_gap  = float(saved_df["avg_purchase_gap"].mean()) if "avg_purchase_gap" in saved_df.columns else 60.0
        overdue  = max(0.0, avg_days - avg_gap)

        st.metric("🗓️ ซื้อล่าสุดเฉลี่ย", f"{avg_days:.0f} วันก่อน")
        st.metric("⏰ ช้ากว่ารอบปกติเฉลี่ย", f"{overdue:.0f} วัน", delta=f"+{overdue:.0f} วันช้าเกินรอบ", delta_color="inverse")

    # 4. Price Range & Past Problem History
    with ic4:
        st.caption("💰 **ช่วงราคา & 🚨 ประวัติปัญหา**")
        price_col = "payment_value" if "payment_value" in saved_df.columns else ("price" if "price" in saved_df.columns else None)
        if price_col:
            p_min = saved_df[price_col].min()
            p_max = saved_df[price_col].max()
            p_avg = saved_df[price_col].mean()
            st.caption(f"💵 **ช่วงราคาสินค้า:** R$ {p_min:,.0f} – {p_max:,.0f}")
            st.caption(f"📊 **ยอดซื้อเฉลี่ย:** R$ {p_avg:,.0f}")

        if "bad_experience_score" in saved_df.columns:
            bad_cnt = (saved_df["bad_experience_score"] > 0).sum()
            bad_pct = bad_cnt / len(saved_df) * 100
            st.caption(f"⚠️ **ประวัติเคยเจอปัญหา:** {bad_pct:.1f}% ({bad_cnt:,} คน)")

        if "delay_days" in saved_df.columns and (saved_df["delay_days"] > 0).sum() > 0:
            avg_delay = saved_df[saved_df["delay_days"] > 0]["delay_days"].mean()
            st.caption(f"🚚 **เคยส่งช้ากว่ากำหนดเฉลี่ย:** {avg_delay:.1f} วัน")
        elif "review_score" in saved_df.columns:
            avg_rev = saved_df["review_score"].mean()
            st.caption(f"⭐ **คะแนนรีวิวเฉลี่ย:** {avg_rev:.1f} / 5.0")

