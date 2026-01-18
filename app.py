import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Olist Real AI Dashboard",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD DATA & MODEL (ของจริง!)
# ==========================================
@st.cache_resource
def load_assets():
    # 1. โหลดโมเดลและฟีเจอร์
    try:
        model = joblib.load('olist_churn_model_best.pkl') # หรือชื่อไฟล์ที่คุณตั้ง
        features = joblib.load('model_features_best.pkl')
    except:
        st.error("⚠️ หาไฟล์โมเดลไม่เจอ! กรุณาเช็คชื่อไฟล์ .pkl")
        return None, None, None, None, None

    # 2. โหลดข้อมูลลูกค้า (Dashboard Input)
    try:
        df = pd.read_csv('olist_dashboard_input.csv')
        # แปลงวันที่
        if 'order_purchase_timestamp' in df.columns:
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    except:
        st.error("⚠️ หาไฟล์ olist_dashboard_input.csv ไม่เจอ")
        return None, None, None, None, None

    # 3. โหลดข้อมูลเสริม
    try:
        risk_map = pd.read_csv('category_churn_risk.csv')
        cycle_map = pd.read_csv('category_cycle_benchmark.csv')
    except:
        risk_map = pd.DataFrame() # กัน error
        cycle_map = pd.DataFrame()

    return model, features, df, risk_map, cycle_map

model, feature_names, df, risk_map, cycle_map = load_assets()

# --- PREDICTION ENGINE ---
if model is not None and df is not None:
    # เตรียมข้อมูลสำหรับทำนาย (เลือกเฉพาะคอลัมน์ที่โมเดลต้องใช้)
    # เติมคอลัมน์ที่ขาดด้วย 0 (เผื่อมีอะไรตกหล่น)
    X_pred = df.reindex(columns=feature_names, fill_value=0)
    
    # ทำนายผล!
    # 0 = Stay, 1 = Churn
    # แต่เราอยากได้ Probability ของการ Churn (คอลัมน์ 1)
    try:
        probs = model.predict_proba(X_pred)[:, 1] 
        df['churn_probability'] = probs
    except:
        # กรณีโมเดลบางตัวไม่มี predict_proba
        preds = model.predict(X_pred)
        df['churn_probability'] = preds.astype(float)

    # --- FINAL LOGIC: ผสม AI + Lateness Score ---
    # ถ้า AI บอกเสี่ยงสูง (Prob > 0.7) OR หายไปนานเกิน (Lateness > 2.0)
    def define_status(row):
        prob = row['churn_probability']
        late = row.get('lateness_score', 0)
        
        if prob > 0.8: return 'High Risk (AI)'
        elif late > 3.0: return 'Lost (Late)'
        elif late > 1.5: return 'Warning (Late)'
        elif prob > 0.5: return 'Medium Risk'
        else: return 'Active'

    df['status'] = df.apply(define_status, axis=1)

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🛍️ Olist AI Analytics")
st.sidebar.caption(f"Total Customers: {len(df):,}")
st.sidebar.markdown("---")
page = st.sidebar.radio("เมนูหลัก", [
    "1. 📊 ภาพรวมธุรกิจ (Overview)",
    "2. 🔍 เจาะลึกรายคน (Customer Risk)",
    "3. 📦 สินค้าเสี่ยง (Product Insight)",
    "4. 🎯 แผนกู้คืนลูกค้า (Action Plan)"
])

if df is None:
    st.warning("กรุณาอัปโหลดไฟล์ข้อมูลก่อนใช้งาน")
    st.stop()

# ==========================================
# PAGE 1: 📊 Executive Summary
# ==========================================
if page == "1. 📊 ภาพรวมธุรกิจ (Overview)":
    st.title("📊 Business Health Check")
    
    # KPI
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    high_risk = len(df[df['status'].str.contains('High|Lost')])
    churn_rate = (high_risk / total_customers) * 100
    avg_lateness = df['lateness_score'].mean()
    
    col1.metric("ลูกค้าทั้งหมด", f"{total_customers:,}")
    col2.metric("กลุ่มเสี่ยงสูง (High Risk)", f"{high_risk:,}", f"{churn_rate:.1f}% ของทั้งหมด", delta_color="inverse")
    col3.metric("คะแนนความล่าช้าเฉลี่ย", f"{avg_lateness:.2f}x", "ยิ่งน้อยยิ่งดี", delta_color="inverse")
    
    # Revenue at Risk (ถ้ามี col payment_value)
    if 'payment_value' in df.columns:
        risk_money = df[df['status'].str.contains('High|Lost')]['payment_value'].sum()
        col4.metric("รายได้ที่เสี่ยงสูญเสีย", f"R$ {risk_money:,.0f}", "Money at Risk")

    st.markdown("---")

    # Chart 1: Distribution of Risk
    st.subheader("🚦 สัดส่วนสถานะลูกค้า (Customer Status)")
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    chart_status = alt.Chart(status_counts).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Status", type="nominal", 
                        scale=alt.Scale(domain=['Active', 'Medium Risk', 'Warning (Late)', 'High Risk (AI)', 'Lost (Late)'],
                                        range=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#34495e'])),
        tooltip=['Status', 'Count']
    )
    st.altair_chart(chart_status, use_container_width=True)

# ==========================================
# PAGE 2: 🔍 Customer Risk Predictor
# ==========================================
elif page == "2. 🔍 เจาะลึกรายคน (Customer Risk)":
    st.title("🔍 ค้นหาลูกค้า & ประเมินความเสี่ยง")
    
    # Search Box
    search_id = st.text_input("ค้นหา Customer ID (หรือปล่อยว่างเพื่อดูทั้งหมด)", "")
    
    # Filter
    filter_status = st.multiselect("กรองสถานะ:", df['status'].unique(), default=['High Risk (AI)', 'Warning (Late)'])
    
    # Apply Filter
    filtered_df = df[df['status'].isin(filter_status)]
    if search_id:
        filtered_df = filtered_df[filtered_df['customer_unique_id'].str.contains(search_id)]
    
    # Show Table
    st.write(f"พบลูกค้าจำนวน: {len(filtered_df):,} คน")
    
    # เลือกคอลัมน์ที่จะโชว์
    show_cols = ['customer_unique_id', 'status', 'churn_probability', 'lateness_score', 'payment_value', 'product_category_name']
    # กรองเอาเฉพาะที่มีอยู่จริง
    final_cols = [c for c in show_cols if c in df.columns]
    
    st.dataframe(
        filtered_df[final_cols].sort_values(by='churn_probability', ascending=False).style.format({
            'churn_probability': '{:.2%}',
            'lateness_score': '{:.2f}',
            'payment_value': '{:,.2f}'
        })
    )

# ==========================================
# PAGE 3: 📦 Product Insight
# ==========================================
elif page == "3. 📦 สินค้าเสี่ยง (Product Insight)":
    st.title("📦 สินค้าไหนเสี่ยง Churn สูงสุด?")
    
    # Group by Category
    cat_risk = df.groupby('product_category_name').agg({
        'churn_probability': 'mean',
        'customer_unique_id': 'count',
        'lateness_score': 'mean'
    }).reset_index()
    
    # Filter only significant categories (> 50 orders)
    cat_risk = cat_risk[cat_risk['customer_unique_id'] > 50].sort_values('churn_probability', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Top 10 หมวดสินค้าที่ลูกค้าหนีเยอะสุด")
        chart_cat = alt.Chart(cat_risk.head(10)).mark_bar().encode(
            x=alt.X('churn_probability', title='Avg Churn Prob'),
            y=alt.Y('product_category_name', sort='-x', title='Category'),
            color=alt.condition(
                alt.datum.churn_probability > 0.8,
                alt.value('red'),
                alt.value('steelblue')
            ),
            tooltip=['product_category_name', 'churn_probability', 'lateness_score']
        )
        st.altair_chart(chart_cat, use_container_width=True)
        
    with col2:
        st.info("💡 **Insight:** สินค้าที่กราฟแดงยาวๆ คือสินค้ากลุ่ม One-time purchase (ซื้อแล้วจบ) หรือสินค้าที่มีปัญหาคุณภาพ")

# ==========================================
# PAGE 4: 🎯 Action Plan
# ==========================================
elif page == "4. 🎯 แผนกู้คืนลูกค้า (Action Plan)":
    st.title("🎯 ใครคือเป้าหมายที่เราต้องช่วย? (The Rescue List)")
    
    st.markdown("""
    เราคัดเลือก **"Golden Segment"** มาให้แล้ว:
    1. **ไม่ใช่คนที่จะหนีแน่นอน (Lost)** -> Lateness Score < 3.0
    2. **แต่เริ่มมีอาการ (Warning)** -> Lateness Score > 1.5
    3. **เป็นลูกค้าชั้นดี (High Value)** -> ยอดซื้อสูงกว่าค่าเฉลี่ย
    """)
    
    # Logic Filter
    avg_spend = df['payment_value'].mean() if 'payment_value' in df.columns else 100
    
    rescue_list = df[
        (df['lateness_score'] > 1.5) & 
        (df['lateness_score'] < 3.0) &
        (df['payment_value'] > avg_spend)
    ]
    
    st.success(f"💎 พบลูกค้า VIP ที่กำลังจะหนีจำนวน: **{len(rescue_list):,} คน** (มูลค่ารวม R$ {rescue_list['payment_value'].sum():,.0f})")
    
    st.write("📋 **รายชื่อสำหรับส่ง SMS/Email Marketing:**")
    st.dataframe(rescue_list[['customer_unique_id', 'product_category_name', 'lateness_score', 'payment_value']])
    
    # ปุ่มดาวน์โหลด
    csv = rescue_list.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Rescue List (.csv)",
        csv,
        "olist_rescue_mission.csv",
        "text/csv",
        key='download-csv'
    )
