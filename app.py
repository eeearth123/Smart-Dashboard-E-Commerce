import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Olist AI Dashboard",
    page_icon="🇧🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. MOCK DATA GENERATOR (จำลองข้อมูล)
# ==========================================
@st.cache_data
def get_mock_data():
    # สร้างข้อมูลจำลอง 500 แถว
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'customer_id': [f'CUST-{i:04d}' for i in range(n)],
        'delivery_days': np.random.normal(12, 4, n), # เฉลี่ยส่ง 12 วัน
        'review_score': np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.1, 0.15, 0.25, 0.4]),
        'monetary': np.random.exponential(150, n),
        'churn_prob': np.random.uniform(0, 1, n),
        # พิกัดจำลอง (แถวๆ บราซิล)
        'lat': np.random.uniform(-23.5, -20.0, n),
        'lon': np.random.uniform(-46.6, -43.0, n),
        'segment': np.random.choice(['Loyal', 'Champion', 'Hibernating', 'At Risk'], n)
    })
    # สร้าง Status จาก churn_prob
    df['status'] = df['churn_prob'].apply(lambda x: 'High Risk' if x > 0.6 else 'Active')
    return df

df_mock = get_mock_data()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🛍️ Olist Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("เลือกเมนู (Menu)", [
    "1. 📊 Executive Summary",
    "2. 🔍 Customer Risk Predictor",
    "3. 👥 Segmentation & Persona",
    "4. 🚚 Logistics & Operations",
    "5. 📦 Product & Category",
    "6. 🎯 Action & Simulation"
])
st.sidebar.markdown("---")
st.sidebar.info("💡 **Demo Mode:** ข้อมูลทั้งหมดถูกจำลองขึ้นเพื่อแสดงผลลัพธ์")

# ==========================================
# PAGE 1: 📊 Executive Summary
# ==========================================
if page == "1. 📊 Executive Summary":
    st.title("📊 Executive Summary")
    st.markdown("ภาพรวมสถานการณ์ธุรกิจและแนวโน้มในอนาคต")

    # --- KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)
    avg_churn = df_mock['churn_prob'].mean() * 100
    risk_count = len(df_mock[df_mock['status'] == 'High Risk'])
    revenue_risk = df_mock[df_mock['status'] == 'High Risk']['monetary'].sum()
    
    col1.metric("Overall Churn Rate", f"{avg_churn:.2f}%", "-1.2%")
    col2.metric("Revenue at Risk", f"R$ {revenue_risk:,.0f}", "High", delta_color="inverse")
    col3.metric("High Risk Customers", f"{risk_count} คน", f"{(risk_count/500)*100:.1f}% ของลูกค้าทั้งหมด")
    col4.metric("Active Customers", f"{500 - risk_count} คน", "+12 คน")

    st.markdown("---")

    # --- Trend & Forecast Chart (Highlight) ---
    st.subheader("📈 Churn Rate Trend & Forecast (AI Prediction)")
    
    # จำลองข้อมูลกราฟ
    dates_past = pd.date_range(start='2018-01-01', periods=6, freq='M')
    churn_past = [12.5, 13.0, 12.8, 13.5, 14.2, 14.5]
    dates_future = pd.date_range(start='2018-07-01', periods=3, freq='M')
    churn_future = [14.8, 15.2, 15.5] # แนวโน้มขึ้น
    
    df_trend = pd.concat([
        pd.DataFrame({'Date': dates_past, 'Rate': churn_past, 'Type': 'Actual'}),
        pd.DataFrame({'Date': dates_future, 'Rate': churn_future, 'Type': 'Forecast'})
    ])
    
    chart_forecast = alt.Chart(df_trend).mark_line(point=True).encode(
        x=alt.X('Date', axis=alt.Axis(format='%b %Y')),
        y=alt.Y('Rate', title='Churn Rate (%)', scale=alt.Scale(domain=[10, 18])),
        color=alt.Color('Type', scale=alt.Scale(domain=['Actual', 'Forecast'], range=['#2ecc71', '#e74c3c'])),
        strokeDash=alt.condition(alt.datum.Type == 'Forecast', alt.value([5, 5]), alt.value([0])),
        tooltip=['Date', 'Rate', 'Type']
    ).properties(height=350)
    
    st.altair_chart(chart_forecast, use_container_width=True)
    st.warning("⚠️ **Alert:** โมเดลพยากรณ์ว่า Churn Rate มีแนวโน้ม **สูงขึ้น** ในอีก 3 เดือนข้างหน้า")

# ==========================================
# PAGE 2: 🔍 Customer Risk Predictor
# ==========================================
elif page == "2. 🔍 Customer Risk Predictor":
    st.title("🔍 Customer Risk Predictor")
    st.markdown("เครื่องมือประเมินความเสี่ยงรายบุคคล (สำหรับทีม CS)")

    col_input, col_res = st.columns([1, 1.5])
    
    with col_input:
        st.subheader("📝 กรอกข้อมูลลูกค้า")
        st.text_input("Customer ID", "CUST-9999")
        days = st.slider("ระยะเวลาจัดส่ง (วัน)", 1, 60, 25)
        score = st.slider("คะแนนรีวิวล่าสุด", 1, 5, 2)
        late = st.number_input("จำนวนครั้งที่ส่งช้า", 0, 10, 2)
        
        predict_btn = st.button("🔮 ประเมินความเสี่ยง", use_container_width=True, type="primary")

    with col_res:
        st.subheader("ผลการประเมิน")
        if predict_btn:
            # Mock Result Logic
            risk_score = 0.85 if (days > 20 or score < 3) else 0.20
            
            if risk_score > 0.5:
                st.error(f"🔴 **HIGH RISK** (โอกาสหนี {risk_score*100:.0f}%)")
                st.progress(risk_score, text="Risk Level")
                st.info("💡 **Action Item:** ลูกค้ารอนานเกินไปและรีวิวแย่ -> **ควรส่งคูปองขอโทษทันที**")
            else:
                st.success(f"🟢 **LOW RISK** (โอกาสหนี {risk_score*100:.0f}%)")
                st.progress(risk_score, text="Risk Level")
        else:
            st.info("👈 กดปุ่มเพื่อทำนายผล")

# ==========================================
# PAGE 3: 👥 Segmentation & Persona
# ==========================================
elif page == "3. 👥 Segmentation & Persona":
    st.title("👥 Customer Segmentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("กลุ่มลูกค้าแบ่งตามพฤติกรรม (RFM)")
        # Bar Chart
        seg_counts = df_mock['segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        
        chart_seg = alt.Chart(seg_counts).mark_bar().encode(
            x='Count',
            y=alt.Y('Segment', sort='-x'),
            color='Segment'
        )
        st.altair_chart(chart_seg, use_container_width=True)
        
    with col2:
        st.subheader("❌ อัตราการ Churn ของแต่ละกลุ่ม")
        # Mock churn rate per segment
        churn_by_seg = pd.DataFrame({
            'Segment': ['At Risk', 'Hibernating', 'Loyal', 'Champion'],
            'Churn Rate': [85, 60, 15, 5]
        })
        chart_rate = alt.Chart(churn_by_seg).mark_bar(color='#ff7f50').encode(
            x='Segment',
            y='Churn Rate'
        )
        st.altair_chart(chart_rate, use_container_width=True)

# ==========================================
# PAGE 4: 🚚 Logistics & Operations
# ==========================================
elif page == "4. 🚚 Logistics & Operations":
    st.title("🚚 Logistics Impact Analysis")
    
    # 1. Correlation
    st.subheader("ยิ่งส่งช้า... ยิ่งหนีจริงไหม?")
    chart_corr = alt.Chart(df_mock).mark_circle(size=60).encode(
        x=alt.X('delivery_days', title='วันรอของ'),
        y=alt.Y('churn_prob', title='โอกาส Churn'),
        color=alt.Color('status', title='สถานะ'),
        tooltip=['delivery_days', 'churn_prob']
    ).interactive()
    st.altair_chart(chart_corr, use_container_width=True)
    
    # 2. Map
    st.subheader("📍 พื้นที่ที่มีปัญหา (High Churn Areas)")
    st.markdown("จุดสีแดงแสดงลูกค้าที่มีความเสี่ยงสูง")
    
    # Filter only high risk for map
    map_data = df_mock[df_mock['status'] == 'High Risk'][['lat', 'lon']]
    st.map(map_data, zoom=4)

# ==========================================
# PAGE 5: 📦 Product & Category
# ==========================================
elif page == "5. 📦 Product & Category":
    st.title("📦 Product Insights")
    
    st.subheader("🏆 หมวดหมู่สินค้าที่คนหนีเยอะที่สุด (Top Churn Categories)")
    
    cat_data = pd.DataFrame({
        'Category': ['Office Furniture', 'Fashion', 'Electronics', 'Toys', 'Books'],
        'Churn Rate (%)': [65, 45, 30, 25, 10]
    })
    
    chart_cat = alt.Chart(cat_data).mark_bar().encode(
        x='Category',
        y='Churn Rate (%)',
        color=alt.condition(
            alt.datum['Churn Rate (%)'] > 50,
            alt.value('red'),  # The positive color
            alt.value('steelblue')  # The negative color
        )
    )
    st.altair_chart(chart_cat, use_container_width=True)
    st.caption("*สินค้ากลุ่มเฟอร์นิเจอร์มีปัญหา Churn สูงสุด อาจเกิดจากการขนส่งเสียหายหรือส่งช้า")

# ==========================================
# PAGE 6: 🎯 Action & Simulation
# ==========================================
elif page == "6. 🎯 Action & Simulation":
    st.title("🎯 Action Plan & Simulation")
    st.markdown("### What-if Analysis: ลองปรับแก้แล้วดูผลลัพธ์")
    
    # Simulation Logic
    st.write("ถ้าเราสามารถลดเวลาจัดส่งเฉลี่ยลงได้...")
    days_reduced = st.slider("ลดเวลาส่งลง (วัน)", 0, 10, 2)
    
    current_churn = 14.5
    predicted_churn = current_churn - (days_reduced * 0.8) # สมมติสูตรคำนวณ
    
    col1, col2 = st.columns(2)
    col1.metric("Current Churn Rate", f"{current_churn}%")
    col2.metric("Predicted Churn Rate", f"{predicted_churn:.2f}%", f"-{current_churn - predicted_churn:.2f}%", delta_color="normal")
    
    st.markdown("---")
    
    st.subheader("📋 Target List for Campaign")
    st.write("รายชื่อลูกค้า Top 50 ที่ควรแจกคูปองเพื่อดึงกลับมา (Export ได้)")
    
    target_list = df_mock[df_mock['status'] == 'High Risk'].sort_values('monetary', ascending=False).head(50)
    st.dataframe(target_list[['customer_id', 'segment', 'monetary', 'churn_prob']])
    
    st.button("📥 Download Excel (Mock)")
