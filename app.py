import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Olist AI Intelligence",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD ASSETS (Model & Data) - แก้ไขใหม่ ห้ามมี UI
# ==========================================
@st.cache_resource
def load_data_and_model():
    """โหลดข้อมูลและโมเดลทั้งหมด (Pure Logic Only)"""
    data_dict = {}
    errors = [] # เก็บ Error ไว้บอกข้างนอก
    
    # 2.1 Load Model & Features
    try:
        data_dict['model'] = joblib.load('olist_churn_model_best.pkl')
        data_dict['features'] = joblib.load('model_features_best.pkl')
    except Exception as e:
        errors.append(f"Model Error: {str(e)}")

    # 2.2 Load Customer Data (Input)
    try:
        # ลองโหลดไฟล์ Lite ก่อน
        try:
            df = pd.read_csv('olist_dashboard_lite.csv')
        except:
            df = pd.read_csv('olist_dashboard_input.csv')
            
        # แปลงวันที่ (ถ้ามี)
        if 'order_purchase_timestamp' in df.columns:
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
            
        data_dict['df'] = df
    except Exception as e:
        errors.append(f"Data Error: {str(e)}")

    # 2.3 Load Category Risk (Optional)
    try:
        data_dict['risk_map'] = pd.read_csv('category_churn_risk.csv')
    except:
        data_dict['risk_map'] = pd.DataFrame() 

    return data_dict, errors

# --- เรียกใช้ฟังก์ชันโหลด แล้วค่อยโชว์ UI ข้างนอก ---
assets, load_errors = load_data_and_model()

# แสดงผลการโหลด (UI Logic)
if load_errors:
    for err in load_errors:
        st.error(f"❌ {err}")
    # ถ้าโหลดข้อมูลหลักไม่ผ่าน ให้หยุด
    if 'df' not in assets or 'model' not in assets:
        st.stop()
else:
    # ถ้าไม่มี Error เลย ค่อยโชว์ Toast (รันครั้งเดียวพอ)
    if 'model_loaded' not in st.session_state:
        st.toast(f"✅ โหลดข้อมูลลูกค้า {len(assets.get('df', [])):,} คน และโมเดลสำเร็จ!", icon="🚀")
        st.session_state['model_loaded'] = True

# ==========================================
# 3. PREDICTION ENGINE (สมอง AI)
# ==========================================
if assets and 'df' in assets and 'model' in assets:
    df = assets['df']
    model = assets['model']
    feature_names = assets['features']

    # (ส่วนนี้เหมือนเดิม แต่เอา st.spinner ออก หรือไว้นอก cache)
    
    # 3.1 เตรียม Features
    X_pred = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X_pred[col] = df[col]
        else:
            X_pred[col] = 0 

    # 3.2 ทำนายผล
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_pred)[:, 1]
        else:
            probs = model.predict(X_pred)
        df['churn_probability'] = probs
    except Exception as e:
        df['churn_probability'] = 0.5 

    # 3.3 Business Logic
    def get_status(row):
        prob = row.get('churn_probability', 0)
        late = row.get('lateness_score', 0)
        if late > 3.0: return 'Lost (Late > 3x)'
        if prob > 0.75: return 'High Risk (AI)'
        if late > 1.5: return 'Warning (Late > 1.5x)'
        if prob > 0.5: return 'Medium Risk'
        return 'Active / Safe'

    df['status'] = df.apply(get_status, axis=1)

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🛍️ Olist Analytics")

# เมนู
page = st.sidebar.radio("เลือกเมนู (Menu)", [
    "1. 📊 ภาพรวมธุรกิจ (Overview)",
    "2. 🔍 เจาะลึกรายคน (Customer List)",
    "3. 📦 สินค้าและหมวดหมู่ (Product Insight)",
    "4. 🎯 ปฏิบัติการกู้คืน (Rescue Mission)"
])

st.sidebar.markdown("---")
if 'df' in assets:
    st.sidebar.info(f"🔢 Total Customers: **{len(df):,}**")
    st.sidebar.caption("✅ System Status: Online")

# ==========================================
# PAGE 1: 📊 Overview
# ==========================================
if page == "1. 📊 ภาพรวมธุรกิจ (Overview)":
    st.title("📊 Business Health Check")
    
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk_count = len(df[df['status'].isin(['High Risk (AI)', 'Warning (Late > 1.5x)'])])
    lost_count = len(df[df['status'] == 'Lost (Late > 3x)'])
    risk_money = df[df['status'].isin(['High Risk (AI)', 'Warning (Late > 1.5x)'])]['payment_value'].sum() if 'payment_value' in df.columns else 0
    
    col1.metric("ลูกค้ากลุ่มเสี่ยง (High + Warn)", f"{high_risk_count:,}", "เป้าหมายหลัก", delta_color="inverse")
    col2.metric("ลูกค้าที่หายไปแล้ว (Lost)", f"{lost_count:,}", "ควรปล่อยผ่าน", delta_color="off")
    col3.metric("มูลค่าที่เสี่ยงสูญเสีย", f"R$ {risk_money:,.0f}", "Money at Risk", delta_color="inverse")
    col4.metric("Lateness Score เฉลี่ย", f"{df['lateness_score'].mean():.2f}x", "Benchmark: 1.0", delta_color="inverse")

    st.markdown("---")
    
    st.subheader("🚦 สัดส่วนลูกค้าตามสถานะความเสี่ยง")
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    chart = alt.Chart(status_counts).mark_arc(innerRadius=60).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Status", type="nominal", 
                        scale=alt.Scale(domain=['Active / Safe', 'Medium Risk', 'Warning (Late > 1.5x)', 'High Risk (AI)', 'Lost (Late > 3x)'],
                                        range=['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#95a5a6'])),
        tooltip=['Status', 'Count']
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

# ==========================================
# PAGE 2: 🔍 Customer List
# ==========================================
elif page == "2. 🔍 เจาะลึกรายคน (Customer List)":
    st.title("🔍 ค้นหาและกรองลูกค้า")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_status = st.multiselect("กรองสถานะ:", df['status'].unique(), default=['High Risk (AI)', 'Warning (Late > 1.5x)'])
    with col_f2:
        search_id = st.text_input("ค้นหา ID:", "")
        
    filtered_df = df[df['status'].isin(filter_status)]
    if search_id:
        filtered_df = filtered_df[filtered_df['customer_unique_id'].str.contains(search_id)]
    
    st.write(f"พบลูกค้าจำนวน: **{len(filtered_df):,}** คน")
    
    show_cols = ['customer_unique_id', 'status', 'churn_probability', 'lateness_score', 'payment_value', 'review_score', 'product_category_name']
    final_cols = [c for c in show_cols if c in df.columns]
    
    st.dataframe(
        filtered_df[final_cols].sort_values(by=['churn_probability', 'lateness_score'], ascending=False),
        use_container_width=True
    )

# ==========================================
# PAGE 3: 📦 Product Insight
# ==========================================
elif page == "3. 📦 สินค้าและหมวดหมู่ (Product Insight)":
    st.title("📦 สินค้าหมวดไหนคนหนีเยอะสุด?")
    
    if 'product_category_name' in df.columns:
        cat_stats = df.groupby('product_category_name').agg({
            'churn_probability': 'mean',
            'lateness_score': 'mean',
            'customer_unique_id': 'count'
        }).reset_index()
        
        cat_stats = cat_stats[cat_stats['customer_unique_id'] > 20].sort_values('churn_probability', ascending=False).head(15)
        
        chart_cat = alt.Chart(cat_stats).mark_bar().encode(
            x=alt.X('churn_probability', title='โอกาส Churn เฉลี่ย'),
            y=alt.Y('product_category_name', sort='-x', title='หมวดสินค้า'),
            color=alt.condition(
                alt.datum.churn_probability > 0.7,
                alt.value('#e74c3c'),
                alt.value('#3498db')
            ),
            tooltip=['product_category_name', 'churn_probability', 'customer_unique_id']
        ).properties(height=500)
        
        st.altair_chart(chart_cat, use_container_width=True)
    else:
        st.error("ไม่พบคอลัมน์ product_category_name")

# ==========================================
# PAGE 4: 🎯 Rescue Mission
# ==========================================
elif page == "4. 🎯 ปฏิบัติการกู้คืน (Rescue Mission)":
    st.title("🎯 รายชื่อลูกค้าเกรด A ที่ต้องดึงกลับมา")
    
    avg_pay = df['payment_value'].mean() if 'payment_value' in df.columns else 0
    rescue_df = df[
        (df['status'] == 'Warning (Late > 1.5x)') & 
        (df['payment_value'] > avg_pay)
    ]
    
    st.success(f"💎 พบลูกค้าศักยภาพสูงที่กำลังจะหลุดมือ: **{len(rescue_df):,} คน**")
    st.dataframe(rescue_df[['customer_unique_id', 'payment_value', 'lateness_score', 'product_category_name']].sort_values('payment_value', ascending=False))
