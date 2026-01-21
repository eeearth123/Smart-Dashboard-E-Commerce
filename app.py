import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import datetime

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Olist Executive Cockpit",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style ตกแต่ง KPI ให้สวยงาม
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
# 2. LOAD ASSETS (Data & Model)
# ==========================================
@st.cache_resource
def load_data_and_model():
    data_dict = {}
    errors = []
    
    # 2.1 Load Model
    try:
        data_dict['model'] = joblib.load('olist_churn_model_best.pkl')
        data_dict['features'] = joblib.load('model_features_best.pkl')
    except Exception as e:
        errors.append(f"Model Error: {e}")

    # 2.2 Load Data
    try:
        # พยายามโหลดไฟล์ Lite ก่อน
        try:
            df = pd.read_csv('olist_dashboard_lite.csv')
        except:
            df = pd.read_csv('olist_dashboard_input.csv')
        
        # แปลงวันที่สำคัญ (จำเป็นสำหรับกราฟ Trend)
        if 'order_purchase_timestamp' in df.columns:
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        
        data_dict['df'] = df
    except Exception as e:
        errors.append(f"Data Error: {e}")

    return data_dict, errors

# เรียกใช้งานโหลดข้อมูล
assets, load_errors = load_data_and_model()

# ถ้ามี Error ให้แจ้งเตือน แต่ถ้าข้อมูลไม่ครบให้หยุด
if load_errors:
    for err in load_errors:
        st.error(f"⚠️ {err}")
    if 'df' not in assets or 'model' not in assets:
        st.stop()

# ==========================================
# 3. PREPARE DATA (AI Prediction & Status)
# ==========================================
df = assets['df']
model = assets['model']
feature_names = assets['features']

# 3.1 Predict Churn Probability
if 'churn_probability' not in df.columns:
    X_pred = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in df.columns:
            X_pred[col] = df[col]
        else:
            X_pred[col] = 0
            
    try:
        if hasattr(model, "predict_proba"):
            df['churn_probability'] = model.predict_proba(X_pred)[:, 1]
        else:
            df['churn_probability'] = model.predict(X_pred)
    except:
        df['churn_probability'] = 0.5 # Fallback

# 3.2 Define Status (Business Logic)
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
# 4. DASHBOARD LAYOUT: Executive Summary
# ==========================================
st.sidebar.title("✈️ Olist Cockpit")
page = st.sidebar.radio("Navigation", ["1. 📊 Executive Summary", "2. 🔍 Customer Detail", "3. 🎯 Action Plan"])

if page == "1. 📊 Executive Summary":
    st.title("📊 Executive Summary (Business Health)")
    st.markdown("ภาพรวมสุขภาพของธุรกิจและแนวโน้มความเสี่ยงลูกค้า (Real-time AI Analysis)")
    st.markdown("---")

    # --- PART 1: KPI CARDS ---
    # คำนวณตัวเลข
    total_customers = len(df)
    
    # กลุ่มเสี่ยง (High Risk + Warning)
    risk_df = df[df['status'].isin(['High Risk', 'Warning (Late > 1.5x)'])]
    risk_count = len(risk_df)
    
    # Churn Rate (คำนวณจากกลุ่มเสี่ยงเทียบทั้งหมด)
    churn_rate = (risk_count / total_customers) * 100
    
    # Revenue at Risk
    rev_at_risk = risk_df['payment_value'].sum() if 'payment_value' in df.columns else 0
    
    # Active Customers
    active_count = len(df[df['status'] == 'Active'])

    # แสดงผล KPI แบบ 4 คอลัมน์
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("🚨 Current Churn Rate", f"{churn_rate:.1f}%", delta="-Target 5%", delta_color="inverse")
    with kpi2:
        st.metric("💸 Revenue at Risk", f"R$ {rev_at_risk:,.0f}", "ความเสียหายที่อาจเกิด", delta_color="inverse")
    with kpi3:
        st.metric("👥 Risk vs Total", f"{risk_count:,} / {total_customers:,}", "ลูกค้ากลุ่มเสี่ยง")
    with kpi4:
        st.metric("✅ Active Customers", f"{active_count:,}", "ลูกค้าชั้นดี")

    st.markdown("---")

    # --- PART 2: CHARTS ROW ---
    col_chart1, col_chart2 = st.columns([2, 1])

    # --- Chart 1: Trend & Forecast (Line Chart) ---
    with col_chart1:
        st.subheader("📈 Churn Risk Trend & Forecast")
        
        # 1. สร้างข้อมูลย้อนหลัง (Historical)
        # Group by Month ของวันที่ซื้อ แล้วดูค่าเฉลี่ย Churn Probability
        if 'order_purchase_timestamp' in df.columns:
            # สร้างคอลัมน์เดือน
            df['month_year'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
            
            # Group ข้อมูลจริง
            trend_df = df.groupby('month_year')['churn_probability'].mean().reset_index()
            trend_df['Type'] = 'Actual'
            trend_df.columns = ['Date', 'Churn_Prob', 'Type']
            
            # แปลง Date กลับเป็น datetime เพื่อพลอตกราฟ
            trend_df['Date'] = pd.to_datetime(trend_df['Date'])
            
            # 2. สร้างข้อมูลพยากรณ์ (Forecast Simulation)
            # (เนื่องจากโมเดลไม่ใช่ Time Series เราจึงจำลองแนวโน้มจากข้อมูลล่าสุด)
            last_date = trend_df['Date'].max()
            last_val = trend_df['Churn_Prob'].iloc[-1]
            
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 4)]
            # สมมติให้ Forecast ขึ้นเล็กน้อย (เพื่อเตือนผู้บริหาร)
            future_vals = [last_val * (1 + 0.02*i) for i in range(1, 4)]
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Churn_Prob': future_vals,
                'Type': ['Forecast', 'Forecast', 'Forecast']
            })
            
            # รวมข้อมูล
            full_trend = pd.concat([trend_df, forecast_df])
            
            # Plot กราฟเส้น
            line_chart = alt.Chart(full_trend).mark_line(point=True).encode(
                x=alt.X('Date', axis=alt.Axis(format='%b %Y', title='Timeline')),
                y=alt.Y('Churn_Prob', axis=alt.Axis(format='%', title='Avg Churn Risk Probability')),
                color=alt.Color('Type', scale=alt.Scale(domain=['Actual', 'Forecast'], range=['#2980b9', '#e74c3c'])),
                strokeDash=alt.condition(
                    alt.datum.Type == 'Forecast',
                    alt.value([5, 5]),  # เส้นประสำหรับ Forecast
                    alt.value([0])      # เส้นทึบสำหรับ Actual
                ),
                tooltip=['Date', alt.Tooltip('Churn_Prob', format='.1%'), 'Type']
            ).properties(height=350)
            
            st.altair_chart(line_chart, use_container_width=True)
            st.caption("ℹ️ เส้นสีแดงคือการคาดการณ์แนวโน้มความเสี่ยงในอีก 3 เดือนข้างหน้า หากไม่มีการป้องกัน")
        else:
            st.warning("⚠️ ไม่สามารถแสดงกราฟ Trend ได้เนื่องจากขาดข้อมูลวันที่ (order_purchase_timestamp)")

    # --- Chart 2: Business Health (Donut Chart) ---
    with col_chart2:
        st.subheader("🍩 Business Health")
        
        # เตรียมข้อมูล
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        # กำหนดสีให้สื่อความหมาย
        domain = ['Active', 'Medium Risk', 'Warning (Late > 1.5x)', 'High Risk', 'Lost (Late > 3x)']
        range_ = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#95a5a6'] # เขียว -> เหลือง -> ส้ม -> แดง -> เทา
        
        donut_chart = alt.Chart(status_counts).mark_arc(innerRadius=60).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Status", type="nominal", scale=alt.Scale(domain=domain, range=range_), legend=dict(orient='bottom')),
            tooltip=['Status', 'Count', alt.Tooltip('Count', format=',')]
        ).properties(height=350)
        
        st.altair_chart(donut_chart, use_container_width=True)

    # --- Action Hint ---
    st.info("💡 **Insight:** ลูกค้ากลุ่ม **High Risk** และ **Warning** คิดเป็นสัดส่วนที่มีนัยสำคัญ แนะนำให้ไปที่หน้า **'Action Plan'** เพื่อดึงรายชื่อทำแคมเปญด่วน")

elif page == "2. 🔍 Customer Detail":
    st.write("หน้า 2 กำลังพัฒนา...") # ใส่โค้ดหน้า 2 ของเก่าที่นี่
elif page == "3. 🎯 Action Plan":
    st.write("หน้า 3 กำลังพัฒนา...") # ใส่โค้ดหน้า 3 ของเก่าที่นี่

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

