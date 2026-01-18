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
# 2. LOAD ASSETS (Model & Data)
# ==========================================
@st.cache_resource
def load_data_and_model():
    """โหลดข้อมูลและโมเดลทั้งหมด พร้อมดักจับ Error"""
    data_dict = {}
    
    # 2.1 Load Model & Features
    try:
        data_dict['model'] = joblib.load('olist_churn_model_best.pkl')
        data_dict['features'] = joblib.load('model_features_best.pkl')
        st.toast("✅ โหลดโมเดลสำเร็จ!", icon="🤖")
    except Exception as e:
        st.error(f"❌ ไม่พบไฟล์โมเดล (olist_churn_model_best.pkl): {e}")
        return None

    # 2.2 Load Customer Data (Input)
    try:
        # ลองโหลดไฟล์ Lite ก่อน ถ้าไม่มีค่อยหา Input ปกติ
        try:
            df = pd.read_csv('olist_dashboard_lite.csv')
        except:
            df = pd.read_csv('olist_dashboard_input.csv')
            
        # แปลงวันที่ (ถ้ามี)
        if 'order_purchase_timestamp' in df.columns:
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
            
        data_dict['df'] = df
        st.toast(f"✅ โหลดข้อมูลลูกค้า {len(df):,} คน", icon="📂")
    except Exception as e:
        st.error(f"❌ ไม่พบไฟล์ข้อมูล (olist_dashboard_input.csv): {e}")
        return None

    # 2.3 Load Category Risk (Optional)
    try:
        data_dict['risk_map'] = pd.read_csv('category_churn_risk.csv')
    except:
        data_dict['risk_map'] = pd.DataFrame() # สร้าง df ว่างๆ กัน error

    return data_dict

# เรียกใช้ฟังก์ชันโหลด
assets = load_data_and_model()

# ==========================================
# 3. PREDICTION ENGINE (สมอง AI)
# ==========================================
if assets and 'df' in assets and 'model' in assets:
    df = assets['df']
    model = assets['model']
    feature_names = assets['features']

    with st.spinner('🤖 AI กำลังวิเคราะห์ข้อมูลลูกค้า...'):
        # 3.1 เตรียม Features ให้ตรงกับตอนเทรนเป๊ะๆ
        # สร้าง DataFrame ว่างๆ ที่มีคอลัมน์ครบตาม feature_names
        X_pred = pd.DataFrame(index=df.index)
        
        for col in feature_names:
            if col in df.columns:
                X_pred[col] = df[col]
            else:
                X_pred[col] = 0 # ถ้าไม่มีใน csv ให้เติม 0 (เช่นพวก dummy variables)

        # 3.2 ทำนายผล (Predict Probability)
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_pred)[:, 1] # เอาความน่าจะเป็นที่จะ Churn (class 1)
            else:
                probs = model.predict(X_pred) # ถ้าไม่มี prob ให้เอาผล 0,1 เลย
            
            df['churn_probability'] = probs
        except Exception as e:
            st.warning(f"⚠️ โมเดลทำนายไม่ได้บางส่วน: {e}")
            df['churn_probability'] = 0.5 # ค่ากลางๆ

        # 3.3 สร้าง Business Logic Status (AI + Lateness)
        def get_status(row):
            prob = row.get('churn_probability', 0)
            late = row.get('lateness_score', 0)
            
            # Logic การจัดกลุ่ม
            if late > 3.0: return 'Lost (Late > 3x)'      # หายไปนานเกินเยียวยา
            if prob > 0.75: return 'High Risk (AI)'       # พฤติกรรมเสี่ยงสูง
            if late > 1.5: return 'Warning (Late > 1.5x)' # เริ่มหายไป (Golden Time ในการตาม)
            if prob > 0.5: return 'Medium Risk'           # กลางๆ
            return 'Active / Safe'                        # ปลอดภัย

        df['status'] = df.apply(get_status, axis=1)

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🛍️ Olist Analytics")

if assets is None or 'df' not in assets:
    st.sidebar.error("🚫 หยุดการทำงาน: ข้อมูลไม่ครบ")
    st.stop()

# เมนู
page = st.sidebar.radio("เลือกเมนู (Menu)", [
    "1. 📊 ภาพรวมธุรกิจ (Overview)",
    "2. 🔍 เจาะลึกรายคน (Customer List)",
    "3. 📦 สินค้าและหมวดหมู่ (Product Insight)",
    "4. 🎯 ปฏิบัติการกู้คืน (Rescue Mission)"
])

st.sidebar.markdown("---")
st.sidebar.info(f"🔢 Total Customers: **{len(df):,}**")
st.sidebar.info(f"📅 Data Status: **Real-Time Prediction**")

# ==========================================
# PAGE 1: 📊 Overview
# ==========================================
if page == "1. 📊 ภาพรวมธุรกิจ (Overview)":
    st.title("📊 Business Health Check")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk_count = len(df[df['status'].isin(['High Risk (AI)', 'Warning (Late > 1.5x)'])])
    lost_count = len(df[df['status'] == 'Lost (Late > 3x)'])
    
    # คำนวณเงินเสี่ยง (ถ้ามีคอลัมน์ payment_value)
    risk_money = df[df['status'].isin(['High Risk (AI)', 'Warning (Late > 1.5x)'])]['payment_value'].sum() if 'payment_value' in df.columns else 0
    
    col1.metric("ลูกค้ากลุ่มเสี่ยง (High + Warn)", f"{high_risk_count:,}", "เป้าหมายหลัก", delta_color="inverse")
    col2.metric("ลูกค้าที่หายไปแล้ว (Lost)", f"{lost_count:,}", "ควรปล่อยผ่าน", delta_color="off")
    col3.metric("มูลค่าที่เสี่ยงสูญเสีย (Revenue at Risk)", f"R$ {risk_money:,.0f}", "ต้องรีบกู้คืน", delta_color="inverse")
    col4.metric("คะแนนความล่าช้าเฉลี่ย (Lateness)", f"{df['lateness_score'].mean():.2f}x", "Benchmark: 1.0", delta_color="inverse")

    st.markdown("---")

    # Chart: Distribution
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
    
    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_status = st.multiselect("กรองสถานะ (Status):", df['status'].unique(), default=['High Risk (AI)', 'Warning (Late > 1.5x)'])
    with col_f2:
        search_id = st.text_input("ค้นหาด้วย ID:", "")
        
    # Apply Filters
    filtered_df = df[df['status'].isin(filter_status)]
    if search_id:
        filtered_df = filtered_df[filtered_df['customer_unique_id'].str.contains(search_id)]
    
    st.write(f"พบลูกค้าจำนวน: **{len(filtered_df):,}** คน")
    
    # Display Table (เลือกคอลัมน์สวยๆ)
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
        # Group Data
        cat_stats = df.groupby('product_category_name').agg({
            'churn_probability': 'mean',
            'lateness_score': 'mean',
            'customer_unique_id': 'count'
        }).reset_index()
        
        # กรองเฉพาะหมวดที่มีคนซื้อเยอะหน่อย (> 20 คน) เพื่อความน่าเชื่อถือ
        cat_stats = cat_stats[cat_stats['customer_unique_id'] > 20].sort_values('churn_probability', ascending=False).head(15)
        
        chart_cat = alt.Chart(cat_stats).mark_bar().encode(
            x=alt.X('churn_probability', title='โอกาส Churn เฉลี่ย (AI Prediction)'),
            y=alt.Y('product_category_name', sort='-x', title='หมวดสินค้า'),
            color=alt.condition(
                alt.datum.churn_probability > 0.7,
                alt.value('#e74c3c'),  # Red for high risk
                alt.value('#3498db')   # Blue for normal
            ),
            tooltip=['product_category_name', 'churn_probability', 'lateness_score', 'customer_unique_id']
        ).properties(height=500)
        
        st.altair_chart(chart_cat, use_container_width=True)
        st.caption("*แสดงเฉพาะหมวดที่มีคำสั่งซื้อมากกว่า 20 รายการ")
    else:
        st.error("ไม่พบคอลัมน์ product_category_name ในไฟล์ข้อมูล")

# ==========================================
# PAGE 4: 🎯 Rescue Mission
# ==========================================
elif page == "4. 🎯 ปฏิบัติการกู้คืน (Rescue Mission)":
    st.title("🎯 รายชื่อลูกค้าเกรด A ที่ต้องดึงกลับมา (Actionable List)")
    
    st.markdown("""
    **เกณฑ์การคัดเลือก (The Sweet Spot):**
    1. 🟡 **เริ่มหาย (Warning):** Lateness Score ระหว่าง 1.5 - 3.0 (ยังไม่สายเกินไป)
    2. 💰 **กระเป๋าหนัก (High Value):** ยอดซื้อสูงกว่าค่าเฉลี่ย
    3. ⭐ **เคยพอใจ (Happy):** รีวิวคะแนนดี (ถ้ามีข้อมูล)
    """)
    
    # Logic Filter
    avg_pay = df['payment_value'].mean() if 'payment_value' in df.columns else 0
    
    rescue_df = df[
        (df['status'] == 'Warning (Late > 1.5x)') & 
        (df['payment_value'] > avg_pay)
    ]
    
    st.success(f"💎 พบลูกค้าศักยภาพสูงที่กำลังจะหลุดมือ: **{len(rescue_df):,} คน**")
    
    # แสดงผล
    st.dataframe(rescue_df[['customer_unique_id', 'payment_value', 'lateness_score', 'product_category_name']].sort_values('payment_value', ascending=False))
    
    # ปุ่ม Download
    csv = rescue_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 ดาวน์โหลดรายชื่อไปยิง Ads/Email (.csv)",
        data=csv,
        file_name='olist_rescue_campaign.csv',
        mime='text/csv',
    )
