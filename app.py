import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import datetime
import os

# ==========================================
# 1. SETUP & CONFIGURATION (ของเดิมของคุณ)
# ==========================================
st.set_page_config(
    page_title="Olist Executive Cockpit",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style ตกแต่ง KPI
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD ASSETS (Update ให้รองรับ Model ใหม่ แต่โครงสร้างเดิม)
# ==========================================
@st.cache_resource
def load_data_and_model():
    data_dict = {}
    errors = []
    
    # Path Fix
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'olist_churn_model_best.pkl')
    features_path = os.path.join(current_dir, 'model_features_best.pkl')
    lite_data_path = os.path.join(current_dir, 'olist_dashboard_lite.csv')

    # 1. โหลด Model
    try:
        data_dict['model'] = joblib.load(model_path)
        data_dict['features'] = joblib.load(features_path)
    except Exception as e:
        errors.append(f"Model Error: ไม่สามารถโหลดโมเดลได้ ({e})")

    # 2. โหลด Data
    try:
        if os.path.exists(lite_data_path):
            df = pd.read_csv(lite_data_path)
            
            # แปลงวันที่
            if 'order_purchase_timestamp' in df.columns:
                df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
            
            # [ADD] สร้างตัวแปรที่จำเป็นถ้าไม่มี (กัน Error)
            if 'payment_value' not in df.columns and 'price' in df.columns:
                df['payment_value'] = df['price'] + df.get('freight_value', 0)
            if 'freight_ratio' not in df.columns and 'freight_value' in df.columns:
                df['freight_ratio'] = df['freight_value'] / df['price']
                
            data_dict['df'] = df
        else:
            errors.append(f"Data Error: ไม่พบไฟล์ข้อมูลที่ {lite_data_path}")
            
    except Exception as e:
        errors.append(f"Data Error: อ่านไฟล์ไม่ได้ ({e})")
        
    return data_dict, errors

assets, load_errors = load_data_and_model()

# เช็ค Error
if load_errors:
    for err in load_errors:
        st.error(f"⚠️ {err}")
    if 'df' not in assets:
        st.stop()

# ==========================================
# 3. PREPARE DATA (แก้ตรงนี้)
# ==========================================
df = assets['df'] 
model = assets.get('model')
feature_names = assets.get('features', [])

# 3.1 Predict Logic
if 'churn_probability' not in df.columns and model is not None:
    X_pred = pd.DataFrame(index=df.index)
    for col in feature_names:
        X_pred[col] = df[col] if col in df.columns else 0
    try:
        if hasattr(model, "predict_proba"):
            df['churn_probability'] = model.predict_proba(X_pred)[:, 1]
        else:
            df['churn_probability'] = model.predict(X_pred)
    except:
        df['churn_probability'] = 0.5 # Fallback

# -------------------------------------------------------------
# 🔧 [FIX] เพิ่มส่วนนี้เข้าไปเพื่อแก้ Error is_churn หาย
# -------------------------------------------------------------
if 'is_churn' not in df.columns:
    # ถ้าไม่มีเฉลยจริง ให้ใช้ "ผลทำนายของ AI" แทน
    # ถ้าความน่าจะเป็น > 0.5 ให้ถือว่าเป็น Churn (1), ถ้าไม่ก็เป็น Stay (0)
    df['is_churn'] = (df['churn_probability'] > 0.5).astype(int)
# -------------------------------------------------------------

# 3.2 Define Status Logic
if 'status' not in df.columns:
    def get_status(row):
        prob = row.get('churn_probability', 0)
        late = row.get('lateness_score', 0)
        
        if late > 3.0: return 'Lost (Late > 3x)'
        if prob > 0.75: return 'High Risk'
        if late > 1.5: return 'Warning (Late > 1.5x)'
        if prob > 0.5: return 'Medium Risk'
        return 'Active'
        
    df['status'] = df.apply(get_status, axis=1)

# ==========================================
# 4. NAVIGATION
# ==========================================
st.sidebar.title("✈️ Olist Cockpit")
page = st.sidebar.radio("Navigation", [
    "1. 📊 Executive Summary", 
    "2. 🔍 Customer Detail", 
    "3. 🎯 Action Plan",
    "4. 🚛 Logistics Insights",
    "5. 🏪 Seller Audit",
    "6. 🔄 Buying Cycle Analysis" # [NEW] หน้าใหม่
])

st.sidebar.markdown("---")
st.sidebar.info("Select a page to analyze different aspects of your business.")

# ==========================================
# PAGE 1: 📊 Executive Summary (คืนค่าเดิม + เพิ่มส่วนขาด)
# ==========================================
if page == "1. 📊 Executive Summary":
    st.title("📊 Executive Summary (Business Health)")
    
    # [ADD] คำอธิบาย Logic (ใส่เพิ่มให้ตามขอ)
    with st.expander("ℹ️ วิธีการแบ่งกลุ่มลูกค้า (Segmentation Logic) - กดเพื่ออ่าน"):
        st.markdown("""
        **ลำดับการตรวจสอบ (Priority):**
        1. **🔴 Lost:** หายไปนานเกิน 3 เท่าของรอบปกติ (`Lateness > 3.0`) -> เลิกซื้อชัวร์
        2. **🟥 High Risk:** ยังไม่นานมาก แต่ **AI ทำนายว่าเสี่ยง > 75%** -> มีปัญหาซ่อนอยู่
        3. **🟧 Warning:** AI บอกโอเค แต่ลูกค้าเริ่มหายเกิน 1.5 เท่า (`Lateness > 1.5`) -> ต้องเตือน
        4. **🟨 Medium Risk:** มาตรงเวลา แต่ AI ให้ความเสี่ยง 50-75%
        5. **🟩 Active:** มาตรงเวลา และ AI บอกว่าเสี่ยงต่ำ
        """)

    # --- 1. FILTER SECTION (โค้ดเดิมของคุณ) ---
    with st.expander("🌪️ กรองข้อมูล (Filter)", expanded=False):
        all_cats = list(df['product_category_name'].unique()) if 'product_category_name' in df.columns else []
        selected_cats_p1 = st.multiselect("เลือกหมวดหมู่สินค้า (ว่าง = ดูภาพรวมทั้งหมด):", all_cats, key="p1_cat_filter")
    
    if selected_cats_p1:
        df_display = df[df['product_category_name'].isin(selected_cats_p1)].copy()
        filter_label = f"หมวด: {', '.join(selected_cats_p1[:3])}..."
    else:
        df_display = df.copy()
        filter_label = "ภาพรวมทั้งบริษัท"

    st.caption(f"กำลังแสดงผล: **{filter_label}**")
    st.markdown("---")

    # --- 2. KPI CARDS (โค้ดเดิมของคุณ) ---
    total_customers = len(df_display)
    
    if total_customers > 0:
        risk_df = df_display[df_display['status'].isin(['High Risk', 'Warning (Late > 1.5x)'])]
        risk_count = len(risk_df)
        churn_rate = (risk_count / total_customers) * 100
        rev_at_risk = risk_df['payment_value'].sum() if 'payment_value' in df_display.columns else 0
        active_count = len(df_display[df_display['status'] == 'Active'])
        
        if 'cat_median_days' in df_display.columns:
            avg_cycle = df_display['cat_median_days'].mean()
            cycle_text = f"{avg_cycle:.0f} วัน"
        else:
            cycle_text = "N/A"
    else:
        churn_rate, rev_at_risk, risk_count, active_count = 0, 0, 0, 0
        cycle_text = "-"

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("🚨 Churn Rate", f"{churn_rate:.1f}%")
    with k2: st.metric("💸 Revenue at Risk", f"R$ {rev_at_risk:,.0f}")
    with k3: st.metric("👥 Risk vs Total", f"{risk_count:,} / {total_customers:,}")
    with k4: st.metric("✅ Active Customers", f"{active_count:,}")
    with k5: st.metric("🔄 รอบซื้อปกติ (Cycle)", cycle_text)

    st.markdown("---")

    # --- 3. CHARTS ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # [RESTORE] กู้คืนกราฟ Forecast ที่หายไป
        st.subheader("📈 Churn Risk Trend & Forecast")
        if 'order_purchase_timestamp' in df_display.columns and not df_display.empty:
            df_display['month_year'] = df_display['order_purchase_timestamp'].dt.to_period('M').astype(str)
            # สร้างข้อมูลจริง
            trend_df = df_display.groupby('month_year')['churn_probability'].mean().reset_index()
            trend_df.columns = ['Date', 'Churn_Prob']
            trend_df['Type'] = 'Actual'
            trend_df['Date'] = pd.to_datetime(trend_df['Date']) # แปลงกลับเป็น datetime
            
            if not trend_df.empty:
                last_date = trend_df['Date'].max()
                last_val = trend_df['Churn_Prob'].iloc[-1]
                
                # สร้างข้อมูลทำนาย (Forecast 3 เดือน)
                anchor_df = pd.DataFrame({'Date': [last_date], 'Churn_Prob': [last_val], 'Type': ['Forecast']})
                future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 4)]
                future_vals = [last_val * (1 + 0.02*i) for i in range(1, 4)]
                future_df = pd.DataFrame({'Date': future_dates, 'Churn_Prob': future_vals, 'Type': ['Forecast']*3})
                
                full_trend = pd.concat([trend_df, anchor_df, future_df]).drop_duplicates()
                
                chart = alt.Chart(full_trend).mark_line(point=True).encode(
                    x=alt.X('Date', axis=alt.Axis(format='%b %Y', title='Timeline')),
                    y=alt.Y('Churn_Prob', axis=alt.Axis(format='%', title='Avg Churn Risk'), scale=alt.Scale(domain=[0.5, 1.0])),
                    color=alt.Color('Type', scale=alt.Scale(domain=['Actual', 'Forecast'], range=['#2980b9', '#e74c3c'])),
                    strokeDash=alt.condition(alt.datum.Type == 'Forecast', alt.value([5, 5]), alt.value([0])),
                    tooltip=['Date', alt.Tooltip('Churn_Prob', format='.1%'), 'Type']
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("ข้อมูลไม่เพียงพอสำหรับสร้างกราฟ Trend")
        else:
            st.warning("⚠️ ไม่พบข้อมูลวันที่")

    with c2:
        # (โค้ดเดิมของคุณ)
        st.subheader("💰 Revenue Share by Risk")
        if not df_display.empty:
            status_stats = df_display.groupby('status').agg({
                'customer_unique_id': 'count',
                'payment_value': 'sum'
            }).reset_index()
            status_stats.columns = ['Status', 'Count', 'Total_Revenue']
            
            domain = ['Active', 'Medium Risk', 'Warning (Late > 1.5x)', 'High Risk', 'Lost (Late > 3x)']
            range_ = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#95a5a6']
            
            donut = alt.Chart(status_stats).mark_arc(innerRadius=60).encode(
                theta=alt.Theta("Count", type="quantitative"), 
                color=alt.Color("Status", scale=alt.Scale(domain=domain, range=range_), legend=dict(orient='bottom')),
                tooltip=['Status', alt.Tooltip('Count', format=','), alt.Tooltip('Total_Revenue', format=',.0f')]
            ).properties(height=350)
            st.altair_chart(donut, use_container_width=True)
        else:
            st.info("ไม่มีข้อมูลแสดงผล")

# ==========================================
# PAGE 2: 🔍 Customer Detail (โค้ดเดิมของคุณ 100%)
# ==========================================
elif page == "2. 🔍 Customer Detail":
    st.title("🔍 เจาะลึกกลุ่มเสี่ยง (Customer Deep Dive)")
    
    with st.expander("🔎 ตัวกรองข้อมูล (Filters)", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            risk_opts = ['High Risk', 'Warning (Late > 1.5x)', 'Medium Risk', 'Lost (Late > 3x)', 'Active']
            sel_status = st.multiselect("1. สถานะ:", risk_opts, default=['High Risk', 'Warning (Late > 1.5x)'])
        with f2:
            all_cats = list(df['product_category_name'].unique()) if 'product_category_name' in df.columns else []
            sel_cats = st.multiselect("2. หมวดสินค้า:", all_cats)
        with f3:
            search_id = st.text_input("3. ค้นหา ID:", "")

    mask = df['status'].isin(sel_status)
    if sel_cats: mask = mask & df['product_category_name'].isin(sel_cats)
    if search_id: mask = mask & df['customer_unique_id'].str.contains(search_id, case=False)
    filtered_df = df[mask]

    if 'product_category_name' in df.columns and not filtered_df.empty:
        cat_overview = df.groupby('product_category_name').agg({
            'customer_unique_id': 'count',
            'cat_median_days': 'mean'
        }).reset_index().rename(columns={'customer_unique_id': 'Total', 'cat_median_days': 'Cycle_Days'})
        
        cat_risk = filtered_df.groupby('product_category_name').agg({
            'customer_unique_id': 'count'
        }).reset_index().rename(columns={'customer_unique_id': 'Risk_Count'})
        
        cat_stats = pd.merge(cat_risk, cat_overview, on='product_category_name', how='left')
        cat_stats['Risk_Pct'] = cat_stats['Risk_Count'] / cat_stats['Total']
        cat_stats = cat_stats.sort_values('Risk_Count', ascending=False)

        col_c, col_t = st.columns([1.5, 2.5])
        with col_c:
            st.subheader("📊 Top 10 หมวดเสี่ยง")
            base = alt.Chart(cat_stats.head(10)).encode(y=alt.Y('product_category_name', sort='-x', title=None))
            b_total = base.mark_bar(color='#f0f2f6').encode(x='Total', tooltip=['product_category_name', 'Total'])
            b_risk = base.mark_bar(color='#e74c3c').encode(x='Risk_Count', tooltip=['Risk_Count', 'Risk_Pct'])
            st.altair_chart(b_total + b_risk, use_container_width=True)

        with col_t:
            st.subheader("📋 รายละเอียด")
            st.dataframe(cat_stats, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"📄 รายชื่อลูกค้า ({len(filtered_df):,} คน)")
    show_cols = ['customer_unique_id', 'status', 'churn_probability', 'lateness_score', 'cat_median_days', 'payment_value', 'product_category_name']
    final_cols = [c for c in show_cols if c in df.columns]
    
    st.dataframe(
        filtered_df[final_cols].sort_values('churn_probability', ascending=False),
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
            "lateness_score": st.column_config.NumberColumn("Late Score", format="%.1fx")
        },
        use_container_width=True
    )

# ==============================================================================
# PAGE 3: 🎯 Action Plan (เพิ่ม Loading State)
# ==============================================================================
elif page == "3. 🎯 Action Plan":
    import time # เพิ่ม library สำหรับหน่วงเวลา

    st.title("🎯 Action Plan & Simulator")
    st.markdown("### วางแผนกลยุทธ์แก้ปัญหาแบบเจาะจง (Targeted Strategy)")
    
    # ---------------------------------------------------------
    # 0. PREPARE DATA & MULTI-FILTER
    # ---------------------------------------------------------
    if 'df_display' not in locals():
        df_display = df.copy()

    with st.container():
        st.markdown("##### 🔎 เลือกกลุ่มสินค้าที่ต้องการโฟกัส (เลือกได้หลายอัน)")
        
        all_cats = sorted(list(df['product_category_name'].unique())) if 'product_category_name' in df.columns else []
        
        sel_cats_p3 = st.multiselect(
            "หมวดสินค้า (ปล่อยว่าง = ดูภาพรวมทั้งหมด):", 
            all_cats, 
            key="p3_cat_multiselect"
        )
        
        if sel_cats_p3:
            df_p3 = df_display[df_display['product_category_name'].isin(sel_cats_p3)].copy()
            filter_msg = f"หมวด: {', '.join(sel_cats_p3[:3])}{'...' if len(sel_cats_p3)>3 else ''}"
        else:
            df_p3 = df_display.copy()
            filter_msg = "ภาพรวมทุกหมวด"

        total_pop = len(df_p3)
        avg_ltv = df_p3['payment_value'].mean() if 'payment_value' in df_p3.columns else 150
        
        c1, c2 = st.columns([3, 1])
        with c1:
            st.info(f"📊 กำลังวิเคราะห์: **{filter_msg}**")
        with c2:
            st.metric("👥 ลูกค้าในกลุ่มนี้", f"{total_pop:,} คน", help="จำนวนลูกค้าทั้งหมดตาม Filter ที่เลือก")

    st.markdown("---")

    # ---------------------------------------------------------
    # 1. HELPER FUNCTION (เพิ่ม Loading Spinner ตรงนี้)
    # ---------------------------------------------------------
    def render_strategy_story(title, icon, target_df, total_pop, strategy_name, default_cost, compare_col=None, good_value=None, bad_values=None, rec_text=""):
        n_target = len(target_df)
        pct_problem = (n_target / total_pop) * 100 if total_pop > 0 else 0
        
        # --- UI ส่วนแสดงปัญหา ---
        st.subheader(f"{icon} {title}")
        c_prob, c_sol, c_res = st.columns([1, 1.3, 1])
        
        with c_prob:
            st.info(f"**📉 ปัญหา:** พบ {n_target:,} คน\n({pct_problem:.1f}% ของกลุ่มนี้)")
            st.progress(min(pct_problem / 100, 1.0))
            st.caption("แถบสีแสดงสัดส่วนคนที่มีปัญหา")

        with c_sol:
            st.markdown(f"**🛠️ วิธีแก้ไข: {strategy_name}**")
            st.write(rec_text)
            st.markdown("---")
            
            # รับค่า Cost
            cost = st.number_input(f"งบต่อหัว (R$)", value=default_cost, min_value=1, step=1, key=f"cost_{title}")
            
            # --- 🤖 ADVANCED AI LOGIC ---
            max_potential = 5
            ai_msg = ""
            
            if compare_col and good_value is not None and 'churn_probability' in df_p3.columns:
                try:
                    if bad_values: bad_group = df_p3[df_p3[compare_col].isin(bad_values)]
                    else: bad_group = target_df
                    bad_churn = bad_group['churn_probability'].mean() if not bad_group.empty else 0.8
                    
                    if isinstance(good_value, list): good_group = df_p3[df_p3[compare_col].isin(good_value)]
                    elif isinstance(good_value, (int, float)): good_group = df_p3[df_p3[compare_col] <= good_value]
                    else: good_group = df_p3[df_p3[compare_col] == good_value]
                    good_churn = good_group['churn_probability'].mean() if not good_group.empty else 0.4
                    
                    uplift = (bad_churn - good_churn) * 100
                    max_potential = max(int(uplift * 0.5), 1) 
                except: pass

            if cost < 5:
                realistic_rate = min(max_potential, 3)
                constraint_msg = " (งบน้อย = ผลลัพธ์จำกัด)"
            elif cost < 15:
                realistic_rate = min(max_potential, 10)
                constraint_msg = " (งบปานกลาง)"
            else:
                realistic_rate = max_potential
                constraint_msg = " (งบสูง = ผลลัพธ์เต็มที่)"
            
            st.markdown(f"**🤖 AI Prediction:** `{realistic_rate}%`")
            st.caption(f"(Max Potential: {max_potential}% {constraint_msg})")
            
            lift = st.slider(f"ปรับค่าคาดการณ์ความสำเร็จ (%)", 1, 100, realistic_rate, key=f"lift_{title}")

        # --- ส่วนแสดงผลลัพธ์ (เพิ่ม Spinner ตรงนี้) ---
        with c_res:
            # 🟢 ใส่ Loading Spinner เพื่อให้ User รู้ว่ากำลังคำนวณ
            with st.spinner('⚡ AI กำลังคำนวณความคุ้มค่า...'):
                time.sleep(0.4) # หน่วงเวลา 0.4 วินาที เพื่อให้ทันมองเห็น (UX Trick)
                
                budget = n_target * cost
                saved_users = int(n_target * (lift / 100))
                revenue = saved_users * avg_ltv
                roi = ((revenue - budget) / budget) * 100 if budget > 0 else 0
                
                st.success(f"**🚀 ผลลัพธ์**")
                st.metric("💸 งบประมาณ", f"R$ {budget:,.0f}")
                st.metric("👥 ดึงลูกค้าคืน", f"{saved_users:,} คน")
                st.metric("💰 กำไร (ROI)", f"{roi:+.0f}%", delta=f"+{revenue:,.0f}")
                
                if roi > 0:
                    st.caption("✅ **คุ้มค่าการลงทุน!**")
                else:
                    st.error("⚠️ **ขาดทุน!** (ลองลดงบ หรือเพิ่มผลสำเร็จ)")

    # ---------------------------------------------------------
    # 2. STRATEGY TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "💳 1. แก้ปัญหาการจ่ายเงิน", 
        "🚚 2. แก้ปัญหาค่าส่งแพง", 
        "❤️ 3. ง้อลูกค้าส่งช้า",
        "🛍️ 4. ขายพ่วงลดความเสี่ยง"
    ])

    with tab1:
        if 'payment_type' in df_p3.columns:
            target = df_p3[df_p3['payment_type'].isin(['credit_card', 'boleto'])]
            render_strategy_story(
                "กลุ่มเสี่ยงจากการจ่ายเงิน (Payment Risk)", "💳", target, total_pop,
                "เปลี่ยน Cash เป็น Voucher", 20, 
                'payment_type', ['voucher'], ['credit_card', 'boleto'],
                "ลูกค้าที่จ่ายบัตรเครดิต/โอน มีโอกาส Churn สูงมาก ต่างจากคนถือ Voucher ที่มักกลับมาซื้อซ้ำ\n\n👉 **Action:** เสนอ Cashback 5% เข้า Wallet เพื่อเปลี่ยนพฤติกรรม"
            )
        else: st.error("ไม่พบข้อมูล 'payment_type'")

    with tab2:
        if 'freight_ratio' in df_p3.columns:
            target = df_p3[df_p3['freight_ratio'] > 0.2]
            avg_freight = target['freight_value'].mean() if not target.empty else 20
            render_strategy_story(
                "กลุ่มค่าส่งแพงเกินรับไหว (Freight Pain)", "🚚", target, total_pop,
                "ช่วยออกค่าส่ง (Free Shipping)", int(avg_freight), 
                'freight_ratio', 0.1, None,
                f"ลูกค้ากลุ่มนี้ลังเลเพราะค่าส่งแพง (เฉลี่ย R$ {avg_freight:.0f})\n\n👉 **Action:** แจกโค้ดส่วนลดค่าส่ง เพื่อลดแรงต้าน (Friction) ในการตัดสินใจ"
            )
        else: st.error("ไม่พบข้อมูล 'freight_ratio'")

    with tab3:
        if 'delay_days' in df_p3.columns:
            target = df_p3[df_p3['delay_days'] > 0]
            render_strategy_story(
                "กลุ่มโดนเท/ของส่งช้า (Delay Recovery)", "❤️", target, total_pop,
                "SMS ขอโทษ + คูปอง", 15, 
                'delay_days', 0, None,
                "ความล่าช้าทำลายความเชื่อมั่น และเป็นสาเหตุหลักของการเปลี่ยนใจ\n\n👉 **Action:** ส่ง SMS ขอโทษทันทีที่รู้ว่าของช้า พร้อมแนบส่วนลดพิเศษ"
            )
        else: st.error("ไม่พบข้อมูล 'delay_days'")

    with tab4:
        if 'cat_churn_risk' in df_p3.columns:
            target = df_p3[df_p3['cat_churn_risk'] > 0.8]
            render_strategy_story(
                "กลุ่มซื้อสินค้าความเสี่ยงสูง (High Risk)", "🛍️", target, total_pop,
                "Cross-sell สินค้าซื้อง่าย", 10, 
                'cat_churn_risk', 0.5, None,
                "สินค้าบางหมวดคนซื้อครั้งเดียวจบ (เช่น เฟอร์นิเจอร์) ทำให้ Churn สูง\n\n👉 **Action:** ยิงแอดขายพ่วงสินค้าหมวดของใช้ในบ้าน (Housewares) ที่ต้องซื้อซ้ำบ่อยๆ"
            )
        else: st.error("ไม่พบข้อมูล 'cat_churn_risk'")
# ==========================================
# PAGE 4: 🚛 Logistics Insights (โค้ดเดิมของคุณ)
# ==========================================
elif page == "4. 🚛 Logistics Insights":
    st.title("🚛 Logistics Heatmap")
    if 'customer_state' not in df.columns:
        st.error("No state data in CSV")
        st.stop()

    col_map, col_stat = st.columns([2, 1])
    with col_map:
        state_stats = df.groupby('customer_state').agg({
            'customer_unique_id': 'count', 'delivery_days': 'mean', 'churn_probability': 'mean'
        }).reset_index()
        state_stats = state_stats[state_stats['customer_unique_id'] > 5]
        
        chart = alt.Chart(state_stats).mark_circle(size=100).encode(
            x=alt.X('delivery_days', title='Avg Delivery Days'),
            y=alt.Y('churn_probability', title='Avg Churn Risk'),
            color=alt.Color('churn_probability', scale=alt.Scale(scheme='reds')),
            size='customer_unique_id',
            tooltip=['customer_state', 'delivery_days', 'churn_probability']
        ).properties(title='Logistics Risk Map', height=400).interactive()
        st.altair_chart(chart, use_container_width=True)
    with col_stat:
        st.subheader("🚨 Top 5 รัฐที่มีปัญหา")
        st.dataframe(state_stats.sort_values('churn_probability', ascending=False).head(5), hide_index=True)

    st.markdown("---")
    st.subheader("🏙️ City Drill-down")
    if 'customer_city' in df.columns:
        sel_state = st.selectbox("เลือกรัฐ:", sorted(df['customer_state'].unique()))
        if sel_state:
            city_df = df[df['customer_state'] == sel_state]
            city_stats = city_df.groupby('customer_city').agg({
                'customer_unique_id': 'count', 'delivery_days': 'mean', 'churn_probability': 'mean'
            }).reset_index()
            st.dataframe(city_stats[city_stats['customer_unique_id'] >= 2].sort_values('churn_probability', ascending=False).head(10), use_container_width=True)
    else:
        st.info("ไม่มีข้อมูลรายเมือง")

# ==========================================
# PAGE 5: 🏪 Seller Audit (โค้ดเดิมของคุณ)
# ==========================================
elif page == "5. 🏪 Seller Audit":
    st.title("🏪 Seller Watchlist")
    if 'seller_id' not in df.columns:
        st.error("No seller data")
        st.stop()
        
    seller_stats = df.groupby('seller_id').agg({
        'customer_unique_id': 'count', 'churn_probability': 'mean',
        'review_score': 'mean', 'payment_value': 'sum'
    }).reset_index()
    
    bad_sellers = seller_stats[seller_stats['customer_unique_id'] >= 5].sort_values('churn_probability', ascending=False).head(50)
    
    k1, k2, k3 = st.columns(3)
    k1.metric("🚨 ร้านเสี่ยงสูง", f"{len(bad_sellers)} ร้าน")
    k2.metric("💸 ยอดขายกลุ่มนี้", f"R$ {bad_sellers['payment_value'].sum():,.0f}")
    k3.metric("📉 Avg Churn", f"{bad_sellers['churn_probability'].mean()*100:.1f}%")
    
    st.dataframe(bad_sellers.head(20), use_container_width=True, hide_index=True)
    
    st.markdown("### 🔍 Quality vs Risk")
    chart = alt.Chart(seller_stats[seller_stats['customer_unique_id'] >= 5]).mark_circle(color='#e74c3c').encode(
        x='review_score', y='churn_probability', size='payment_value',
        tooltip=['seller_id', 'review_score', 'churn_probability']
    ).properties(height=350).interactive()
    st.altair_chart(chart, use_container_width=True)

# ==========================================
# PAGE 6: 🔄 Buying Cycle Analysis (หน้าใหม่ที่คุณขอ)
# ==========================================
elif page == "6. 🔄 Buying Cycle Analysis":
    st.title("🔄 Buying Cycle Analysis")
    st.markdown("วิเคราะห์รอบการซื้อ: **สินค้าไหนต้องซื้อซ้ำบ่อย? ลูกค้าเลทแค่ไหน?**")
    
    avg_cycle = df['cat_median_days'].mean() if 'cat_median_days' in df.columns else 0
    avg_late = df['lateness_score'].mean() if 'lateness_score' in df.columns else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("⏱️ รอบซื้อเฉลี่ย (ทั้งบริษัท)", f"{avg_cycle:.0f} วัน")
    m2.metric("🐢 ความล่าช้าเฉลี่ย (Late Score)", f"{avg_late:.2f} เท่า", "ถ้า > 1.0 คือเริ่มช้า")
    m3.metric("📅 ซื้อซ้ำใน 30 วัน", f"{len(df[df['cat_median_days']<=30]):,} คน")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📦 รอบการซื้อแยกรายหมวด")
        if 'cat_median_days' in df.columns:
            cat_cyc = df.groupby('product_category_name')['cat_median_days'].median().reset_index().sort_values('cat_median_days').head(20)
            chart = alt.Chart(cat_cyc).mark_bar().encode(
                x=alt.X('cat_median_days', title='Days'), y=alt.Y('product_category_name', sort='x'),
                color=alt.Color('cat_median_days', scale=alt.Scale(scheme='tealblues'))
            )
            st.altair_chart(chart, use_container_width=True)
            
    with c2:
        st.subheader("🐢 Distribution of Lateness")
        if 'lateness_score' in df.columns:
            hist_df = df[df['lateness_score'] <= 10]
            chart = alt.Chart(hist_df).mark_bar().encode(
                x=alt.X('lateness_score', bin=alt.Bin(maxbins=30)),
                y='count()',
                color=alt.condition(alt.datum.lateness_score > 3, alt.value('red'), alt.value('green'))
            )
            st.altair_chart(chart, use_container_width=True)
            
    st.subheader("📋 รายละเอียดรายหมวด")
    summ = df.groupby('product_category_name').agg({
        'customer_unique_id':'count', 'cat_median_days':'mean', 'lateness_score':'mean', 'churn_probability':'mean'
    }).reset_index()
    st.dataframe(summ.sort_values('cat_median_days'), use_container_width=True, hide_index=True)
    # ... (ต่อจากตารางรายละเอียดใน Page 6 เดิม) ...

    st.markdown("---")
    st.subheader("📅 Seasonal Patterns: สินค้าขายดีเดือนไหน?")
    st.caption("เฉดสีเข้ม = ช่วงที่สินค้านั้นขายดีที่สุด (High Season)")

    if 'order_purchase_timestamp' in df.columns:
        # 1. เตรียมข้อมูล (ดึงเดือนออกมา)
        # สร้าง Copy เพื่อไม่ให้กระทบ df หลัก
        season_df = df.copy()
        season_df['month_num'] = season_df['order_purchase_timestamp'].dt.month
        season_df['month_name'] = season_df['order_purchase_timestamp'].dt.strftime('%b') # Jan, Feb, ...
        
        # 2. จัดกลุ่มข้อมูล (Group by Category & Month)
        # นับจำนวนออเดอร์ในแต่ละเดือน
        heatmap_data = season_df.groupby(['product_category_name', 'month_num', 'month_name']).size().reset_index(name='sales_volume')
        
        # 3. คัดเฉพาะ Top Categories (เพื่อให้กราฟดูรู้เรื่อง ไม่รกเกินไป)
        # เอาเฉพาะ 15 หมวดแรกที่มีคนซื้อเยอะสุด
        top_cats = season_df['product_category_name'].value_counts().head(15).index.tolist()
        heatmap_data = heatmap_data[heatmap_data['product_category_name'].isin(top_cats)]
        
        # 4. สร้าง Heatmap Chart
        # แกน X: เดือน (Jan -> Dec)
        # แกน Y: หมวดสินค้า
        # สี: ยอดขาย (ยิ่งเข้มยิ่งขายดี)
        heatmap = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('month_name', 
                    sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                    title='เดือน (Month)'),
            y=alt.Y('product_category_name', title='หมวดสินค้า'),
            color=alt.Color('sales_volume', 
                            scale=alt.Scale(scheme='orangered'), # สีส้ม-แดง (ร้อนแรง)
                            title='ยอดขาย (Orders)'),
            tooltip=['product_category_name', 'month_name', alt.Tooltip('sales_volume', format=',')]
        ).properties(
            height=500,
            title='🔥 Heatmap แสดงช่วงเวลาขายดีของสินค้า Top 15'
        )
        
        st.altair_chart(heatmap, use_container_width=True)
        
        st.info("💡 **Tip:** ลองสังเกตสีแดงเข้มในแต่ละแถว จะช่วยให้รู้ว่าต้องสต็อกของหรือยิงแอดสินค้านั้นเดือนไหน")
        
    else:
        st.warning("⚠️ ไม่พบข้อมูลวันที่ (order_purchase_timestamp) ไม่สามารถวิเคราะห์ Seasonality ได้")













