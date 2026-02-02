import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import datetime
import os

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Olist Executive Cockpit (AI-Powered)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style ตกแต่ง
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stExpander {
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD ASSETS
# ==========================================
@st.cache_resource
def load_data_and_model():
    data_dict = {}
    errors = []
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'olist_churn_model_best.pkl')
    features_path = os.path.join(current_dir, 'model_features_best.pkl')
    lite_data_path = os.path.join(current_dir, 'olist_dashboard_lite.csv')

    # 1. โหลด Model & Features
    try:
        data_dict['model'] = joblib.load(model_path)
        data_dict['features'] = joblib.load(features_path)
    except Exception as e:
        errors.append(f"Model Error: โหลดโมเดลไม่ได้ ({e})")

    # 2. โหลด Data
    try:
        if os.path.exists(lite_data_path):
            df = pd.read_csv(lite_data_path)
            # แปลงวันที่
            if 'order_purchase_timestamp' in df.columns:
                df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
            data_dict['df'] = df
        else:
            errors.append(f"Data Error: ไม่พบไฟล์ {lite_data_path}")
            
    except Exception as e:
        errors.append(f"Data Error: อ่านไฟล์ไม่ได้ ({e})")
        
    return data_dict, errors

assets, load_errors = load_data_and_model()

if load_errors:
    for err in load_errors: st.error(f"⚠️ {err}")
    if 'df' not in assets: st.stop()

# ==========================================
# 3. PREPARE DATA & LOGIC
# ==========================================
df = assets['df'] 
model = assets.get('model')
feature_names = assets.get('features', [])

# 3.1 สร้างตัวแปรที่จำเป็น (ถ้าไม่มีในไฟล์)
if 'payment_value' not in df.columns:
    df['payment_value'] = df['price'] + df['freight_value']

if 'freight_ratio' not in df.columns:
    df['freight_ratio'] = df['freight_value'] / df['price']

# 3.2 AI Prediction
if 'churn_probability' not in df.columns and model is not None:
    # เตรียมข้อมูลสำหรับทำนาย (เติม 0 ถ้าขาด)
    X_pred = pd.DataFrame(index=df.index)
    for col in feature_names:
        X_pred[col] = df[col] if col in df.columns else 0
        
    try:
        if hasattr(model, "predict_proba"):
            df['churn_probability'] = model.predict_proba(X_pred)[:, 1]
        else:
            df['churn_probability'] = model.predict(X_pred)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        df['churn_probability'] = 0.5 # Fallback

# 3.3 Define Status Logic (สูตรที่คุณต้องการ)
if 'status' not in df.columns:
    def get_status(row):
        prob = row.get('churn_probability', 0)
        late = row.get('lateness_score', 0)
        
        # Priority 1: พฤติกรรมจริง (Lateness)
        if late > 3.0: return 'Lost (Late > 3x)'
        
        # Priority 2: AI ทำนาย (Probability)
        if prob > 0.75: return 'High Risk'
        
        # Priority 3: เริ่มสาย (Warning)
        if late > 1.5: return 'Warning (Late > 1.5x)'
        
        # Priority 4: ก้ำกึ่ง
        if prob > 0.5: return 'Medium Risk'
        
        # Priority 5: ปกติ
        return 'Active'
        
    df['status'] = df.apply(get_status, axis=1)

# ==========================================
# 4. NAVIGATION
# ==========================================
st.sidebar.title("✈️ Olist Cockpit")
page = st.sidebar.radio("เมนูหลัก", [
    "1. 📊 Executive Summary", 
    "2. 🔍 Customer Detail", 
    "3. 🎯 Action Plan",
    "4. 🚛 Logistics Insights",
    "5. 🏪 Seller Audit",
    "6. 🔄 Buying Cycle Analysis"
])

st.sidebar.markdown("---")
st.sidebar.info("Dashboard Version: 2.5 (Final Master)")

# ==========================================
# PAGE 1: 📊 Executive Summary
# ==========================================
if page == "1. 📊 Executive Summary":
    st.title("📊 Executive Summary")
    
    # --- 1. EXPLAINER (Logic) ---
    with st.expander("ℹ️ อ่านวิธีแบ่งกลุ่มลูกค้า (Segmentation Logic)", expanded=False):
        st.markdown("""
        ระบบแบ่งลูกค้าเป็น 5 กลุ่ม โดยดู **พฤติกรรมจริง (Lateness)** ผสมกับ **AI (Prediction)** ตามลำดับนี้:
        
        1. **🔴 Lost (เลิกซื้อชัวร์):** หายไปนานเกิน 3 เท่าของรอบปกติ (`Lateness > 3.0`) ตัดเป็น Lost ทันที
        2. **🟥 High Risk (เสี่ยงสูง):** ยังไม่หายไปนานมาก แต่ **AI ทำนายว่าเสี่ยง > 75%** (อาจเจอของพัง/บริการแย่)
        3. **🟧 Warning (เริ่มล่าช้า):** AI บอกว่ายังโอเค แต่ลูกค้าเริ่มหายเงียบเกิน 1.5 เท่า (`Lateness > 1.5`) ต้องรีบเตือน
        4. **🟨 Medium Risk (ก้ำกึ่ง):** มาตรงเวลา แต่ AI ให้ความเสี่ยง 50-75%
        5. **🟩 Active (ลูกค้าชั้นดี):** มาตรงเวลา และ AI บอกว่าความเสี่ยงต่ำ
        """)
    
    # --- 2. FILTER ---
    with st.expander("🌪️ ตัวกรองข้อมูล (Filter)", expanded=False):
        all_cats = list(df['product_category_name'].unique()) if 'product_category_name' in df.columns else []
        selected_cats = st.multiselect("เลือกหมวดหมู่:", all_cats)
    
    if selected_cats:
        df_show = df[df['product_category_name'].isin(selected_cats)].copy()
    else:
        df_show = df.copy()

    st.markdown("---")

    # --- 3. KPI ---
    total = len(df_show)
    if total > 0:
        risk_df = df_show[df_show['status'].isin(['High Risk', 'Warning (Late > 1.5x)'])]
        churn_rate = (len(risk_df) / total) * 100
        rev_risk = risk_df['payment_value'].sum()
        active = len(df_show[df_show['status'] == 'Active'])
        avg_cycle = df_show['cat_median_days'].mean() if 'cat_median_days' in df_show.columns else 0
    else:
        churn_rate, rev_risk, active, avg_cycle = 0, 0, 0, 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("🚨 Churn Rate", f"{churn_rate:.1f}%")
    with k2: st.metric("💸 Revenue at Risk", f"R$ {rev_risk:,.0f}")
    with k3: st.metric("👥 Risk Users", f"{len(risk_df):,}")
    with k4: st.metric("✅ Active Users", f"{active:,}")
    with k5: st.metric("🔄 Avg Cycle", f"{avg_cycle:.0f} วัน")

    st.markdown("---")

    # --- 4. CHARTS ---
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📈 แนวโน้มความเสี่ยง (Trend)")
        if 'order_purchase_timestamp' in df_show.columns:
            df_show['month'] = df_show['order_purchase_timestamp'].dt.to_period('M').astype(str)
            trend = df_show.groupby('month')['churn_probability'].mean().reset_index()
            chart = alt.Chart(trend).mark_line(point=True).encode(
                x='month', y=alt.Y('churn_probability', title='Avg Risk'),
                tooltip=['month', alt.Tooltip('churn_probability', format='.1%')]
            ).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("ไม่พบข้อมูลวันที่")

    with c2:
        st.subheader("💰 สัดส่วนรายได้ตามความเสี่ยง")
        stats = df_show.groupby('status')['payment_value'].sum().reset_index()
        colors = alt.Scale(domain=['Active', 'Medium Risk', 'Warning (Late > 1.5x)', 'High Risk', 'Lost (Late > 3x)'],
                           range=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#95a5a6'])
        donut = alt.Chart(stats).mark_arc(innerRadius=60).encode(
            theta='payment_value', color=alt.Color('status', scale=colors),
            tooltip=['status', alt.Tooltip('payment_value', format=',.0f')]
        ).properties(height=350)
        st.altair_chart(donut, use_container_width=True)

# ==========================================
# PAGE 2: 🔍 Customer Detail
# ==========================================
elif page == "2. 🔍 Customer Detail":
    st.title("🔍 เจาะลึกรายบุคคล")
    
    # Filter
    c1, c2, c3 = st.columns(3)
    with c1: 
        stats = ['High Risk', 'Warning (Late > 1.5x)', 'Medium Risk', 'Lost (Late > 3x)', 'Active']
        sel_stat = st.multiselect("สถานะ:", stats, default=['High Risk', 'Warning (Late > 1.5x)'])
    with c2:
        all_cats = list(df['product_category_name'].unique()) if 'product_category_name' in df.columns else []
        sel_cat = st.multiselect("หมวดสินค้า:", all_cats)
    with c3:
        uid = st.text_input("ค้นหา ID:", "")

    mask = df['status'].isin(sel_stat)
    if sel_cat: mask = mask & df['product_category_name'].isin(sel_cat)
    if uid: mask = mask & df['customer_unique_id'].str.contains(uid, case=False)
    
    df_filt = df[mask]
    
    st.markdown(f"**พบ {len(df_filt):,} รายการ**")
    cols = ['customer_unique_id', 'status', 'churn_probability', 'lateness_score', 'payment_value', 'product_category_name']
    final_cols = [c for c in cols if c in df.columns]

    st.dataframe(
        df_filt[final_cols].sort_values('churn_probability', ascending=False),
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
            "lateness_score": st.column_config.NumberColumn("Late (x)", format="%.1f เท่า")
        },
        use_container_width=True
    )

# ==========================================
# PAGE 3: 🎯 Action Plan
# ==========================================
elif page == "3. 🎯 Action Plan":
    st.title("🎯 Campaign Simulator")
    
    # Target: High Risk + Warning
    target = df[(df['status'].isin(['High Risk', 'Warning (Late > 1.5x)']))].copy()
    
    if target.empty:
        st.warning("ไม่พบกลุ่มเป้าหมาย")
    else:
        st.info(f"🎯 กลุ่มเป้าหมาย: {len(target):,} คน | ยอดเงินเสี่ยงสูญเสีย: R$ {target['payment_value'].sum():,.0f}")
        
        c1, c2, c3 = st.columns(3)
        with c1: discount = st.slider("📉 ลดค่าส่ง (%)", 0, 100, 0, 10)
        with c2: speed = st.selectbox("🚚 อัปเกรดขนส่ง", ["ปกติ", "ส่งด่วน (-2 วัน)"])
        
        # Calculate Cost
        freight_cost = target['freight_value'].sum() * (discount/100)
        speed_cost = len(target) * 5 if speed == "ส่งด่วน (-2 วัน)" else 0 # สมมติ 5 R$ ต่อคน
        total_cost = freight_cost + speed_cost
        
        st.metric("งบประมาณ (Cost)", f"R$ {total_cost:,.0f}")
        
        # Simulate AI Impact
        sim_df = target.copy()
        
        # 1. ปรับ Feature ตามโปร
        sim_df['freight_value'] = sim_df['freight_value'] * (1 - discount/100)
        sim_df['freight_ratio'] = sim_df['freight_value'] / sim_df['price']
        if speed == "ส่งด่วน (-2 วัน)":
            sim_df['delivery_days'] = sim_df['delivery_days'] - 2
            sim_df['delivery_vs_estimated'] = sim_df['delivery_vs_estimated'] + 2 # ส่งเร็วขึ้น = ดีขึ้น
        
        # 2. Predict ใหม่
        X_sim = pd.DataFrame(index=sim_df.index)
        for col in feature_names:
            X_sim[col] = sim_df[col] if col in sim_df.columns else 0
            
        if hasattr(model, "predict_proba"):
            new_probs = model.predict_proba(X_sim)[:, 1]
        else:
            new_probs = model.predict(X_sim)
            
        # 3. สรุปผล
        saved_count = (new_probs < 0.5).sum()
        saved_val = sim_df[new_probs < 0.5]['payment_value'].sum()
        roi = saved_val - total_cost
        
        r1, r2, r3 = st.columns(3)
        r1.metric("😊 กู้คืนได้", f"{saved_count:,} คน")
        r2.metric("💰 รายได้ที่รักษาได้", f"R$ {saved_val:,.0f}")
        r3.metric("📈 ROI", f"R$ {roi:,.0f}")

# ==========================================
# PAGE 4: 🚛 Logistics
# ==========================================
elif page == "4. 🚛 Logistics Insights":
    st.title("🚛 Logistics Heatmap")
    
    if 'customer_state' not in df.columns:
        st.error("ไม่พบข้อมูล State")
        st.stop()
        
    c1, c2 = st.columns([2, 1])
    with c1:
        stats = df.groupby('customer_state').agg({
            'customer_unique_id':'count', 'delivery_days':'mean', 'churn_probability':'mean'
        }).reset_index()
        stats = stats[stats['customer_unique_id'] > 5]
        
        chart = alt.Chart(stats).mark_circle().encode(
            x='delivery_days', y='churn_probability', 
            color=alt.Color('churn_probability', scale=alt.Scale(scheme='reds')),
            size='customer_unique_id',
            tooltip=['customer_state', 'delivery_days', 'churn_probability']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        
    with c2:
        st.subheader("🚨 Top 5 รัฐที่มีปัญหา")
        st.dataframe(stats.sort_values('churn_probability', ascending=False).head(5), hide_index=True)
        
    if 'customer_city' in df.columns:
        st.subheader("🏙️ City Drill-down")
        state = st.selectbox("เลือกรัฐ:", sorted(df['customer_state'].unique()))
        city_stats = df[df['customer_state']==state].groupby('customer_city').agg({
            'customer_unique_id':'count', 'delivery_days':'mean', 'churn_probability':'mean'
        }).reset_index()
        st.dataframe(city_stats[city_stats['customer_unique_id']>2].sort_values('churn_probability', ascending=False).head(10), use_container_width=True)

# ==========================================
# PAGE 5: 🏪 Seller Audit
# ==========================================
elif page == "5. 🏪 Seller Audit":
    st.title("🏪 Seller Watchlist")
    
    if 'seller_id' not in df.columns:
        st.error("ไม่พบข้อมูล Seller ID")
        st.stop()
        
    s_stats = df.groupby('seller_id').agg({
        'customer_unique_id':'count', 'churn_probability':'mean', 'review_score':'mean', 'payment_value':'sum'
    }).reset_index()
    
    bad = s_stats[s_stats['customer_unique_id'] >= 5].sort_values('churn_probability', ascending=False).head(50)
    
    k1, k2, k3 = st.columns(3)
    k1.metric("🚨 ร้านเสี่ยงสูง", f"{len(bad)}")
    k2.metric("💸 ยอดขายร้านกลุ่มนี้", f"R$ {bad['payment_value'].sum():,.0f}")
    k3.metric("📉 Avg Churn", f"{bad['churn_probability'].mean()*100:.1f}%")
    
    st.dataframe(bad, use_container_width=True, hide_index=True)
    
    chart = alt.Chart(s_stats[s_stats['customer_unique_id']>=5]).mark_circle(color='red').encode(
        x='review_score', y='churn_probability', size='payment_value', tooltip=['seller_id', 'review_score']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# ==========================================
# PAGE 6: 🔄 Buying Cycle (NEW)
# ==========================================
elif page == "6. 🔄 Buying Cycle Analysis":
    st.title("🔄 Buying Cycle Analysis")
    st.markdown("วิเคราะห์รอบการซื้อ: **สินค้าไหนต้องซื้อซ้ำบ่อย? ลูกค้าเลทแค่ไหน?**")
    
    # Overview
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
            hist_df = df[df['lateness_score'] <= 10] # ตัด Outlier
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
