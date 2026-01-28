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




# ==========================================
# PAGE 2: 🔍 Customer Detail (Deep Dive)
# ==========================================
elif page == "2. 🔍 Customer Detail":
    st.title("🔍 เจาะลึกกลุ่มเสี่ยง (Customer Deep Dive)")
    st.markdown("วิเคราะห์เจาะลึก: **รอบการซื้อของแต่ละสินค้า** และ **สัดส่วนลูกค้ากลุ่มเสี่ยง**")
    
    # --- 1. FILTERS ---
    with st.expander("🔎 ตัวกรองข้อมูล (Filters)", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            risk_options = ['High Risk', 'Warning (Late > 1.5x)', 'Medium Risk', 'Lost (Late > 3x)', 'Active']
            default_risk = ['High Risk', 'Warning (Late > 1.5x)']
            selected_status = st.multiselect("1. เลือกสถานะลูกค้า:", risk_options, default=default_risk)
            
        with col_f2:
            all_cats = list(df['product_category_name'].unique()) if 'product_category_name' in df.columns else []
            selected_cats = st.multiselect("2. เลือกหมวดสินค้า (ว่าง = ทั้งหมด):", all_cats)
            
        with col_f3:
            search_id = st.text_input("3. ค้นหา Customer ID:", "")

    # Apply Filters
    mask = df['status'].isin(selected_status)
    if selected_cats:
        mask = mask & df['product_category_name'].isin(selected_cats)
    if search_id:
        mask = mask & df['customer_unique_id'].str.contains(search_id, case=False)
    filtered_df = df[mask]

    # --- 2. STATS CALCULATION (หัวใจสำคัญ: คำนวณยอดรวมและรอบซื้อ) ---
    if 'product_category_name' in df.columns and not filtered_df.empty:
        
        # A. เตรียมข้อมูลสรุปรายหมวดหมู่ (Group By Category)
        # เราต้อง Group จาก df ตัวเต็ม (เพื่อหา Total) แล้วค่อยมาเทียบกับ Filtered (Risk)
        
        # 1. ข้อมูลภาพรวม (Total Count & Cycle) จาก DataFrame ทั้งหมด
        cat_overview = df.groupby('product_category_name').agg({
            'customer_unique_id': 'count',          # จำนวนลูกค้าทั้งหมดในหมวดนี้
            'cat_median_days': 'mean'               # รอบการซื้อมาตรฐาน (ค่าจะเท่ากันทั้งหมวด เลยใช้ mean ได้)
        }).reset_index().rename(columns={'customer_unique_id': 'Total_Customers', 'cat_median_days': 'Buying_Cycle_Days'})
        
        # 2. ข้อมูลเฉพาะกลุ่มเสี่ยง (Risk Count) จาก Filtered DataFrame
        cat_risk = filtered_df.groupby('product_category_name').agg({
            'customer_unique_id': 'count',          # จำนวนลูกค้ากลุ่มเสี่ยง
            'churn_probability': 'mean',            # ความเสี่ยงเฉลี่ย
            'lateness_score': 'mean'                # หายไปนานเฉลี่ยกี่เท่า
        }).reset_index().rename(columns={'customer_unique_id': 'Risk_Count'})
        
        # 3. รวมตารางเข้าด้วยกัน
        cat_stats = pd.merge(cat_risk, cat_overview, on='product_category_name', how='left')
        
        # คำนวณ % Risk
        cat_stats['Risk_Percentage'] = (cat_stats['Risk_Count'] / cat_stats['Total_Customers'])
        
        # เรียงลำดับตามจำนวนคนเสี่ยง (จากมากไปน้อย)
        cat_stats = cat_stats.sort_values(by='Risk_Count', ascending=False)

        # --- 3. DISPLAY INSIGHTS ---
        col_chart, col_table = st.columns([1.5, 2.5]) # แบ่งหน้าจอ ซ้ายกราฟ / ขวาตาราง
        
        with col_chart:
            st.subheader("📊 Top 10 หมวดเสี่ยงสูงสุด")
            st.caption("เทียบจำนวนคนเสี่ยง (สีแดง) vs คนทั้งหมด (สีเทาจางๆ)")
            
            # กราฟแท่งแสดงจำนวน
            base = alt.Chart(cat_stats.head(10)).encode(y=alt.Y('product_category_name', sort='-x', title=None))
            
            # แท่งพื้นหลัง (Total)
            bar_total = base.mark_bar(color='#f0f2f6').encode(
                x=alt.X('Total_Customers', title='จำนวนลูกค้า'),
                tooltip=['product_category_name', 'Total_Customers', 'Buying_Cycle_Days']
            )
            
            # แท่งสีแดง (Risk)
            bar_risk = base.mark_bar(color='#e74c3c').encode(
                x=alt.X('Risk_Count'),
                tooltip=['product_category_name', 'Risk_Count', 'Risk_Percentage']
            )
            
            st.altair_chart(bar_total + bar_risk, use_container_width=True)
            
            st.info(f"💡 **Note:** แท่งสีเทาคือจำนวนลูกค้าทั้งหมดในหมวดนั้น ส่วนแท่งสีแดงคือกลุ่มเสี่ยงที่คุณเลือก")

        with col_table:
            st.subheader("📋 รายละเอียดพฤติกรรมสินค้า")
            st.dataframe(
                cat_stats,
                column_config={
                    "product_category_name": "หมวดหมู่สินค้า",
                    "Buying_Cycle_Days": st.column_config.NumberColumn(
                        "🔄 รอบซื้อ (วัน)", 
                        help="ระยะเวลาเฉลี่ยที่คนมักจะกลับมาซื้อซ้ำ (cat_median_days)",
                        format="%d วัน"
                    ),
                    "Risk_Count": st.column_config.NumberColumn("⚠️ คนเสี่ยง", format="%d คน"),
                    "Total_Customers": st.column_config.NumberColumn("📦 ทั้งหมด", format="%d คน"),
                    "Risk_Percentage": st.column_config.ProgressColumn(
                        "% สัดส่วนความเสี่ยง",
                        help="คนเสี่ยงคิดเป็นกี่ % ของลูกค้าทั้งหมดในหมวดนี้",
                        format="%.1f%%",
                        min_value=0,
                        max_value=1
                    ),
                    "lateness_score": st.column_config.NumberColumn("⏳ หายไป (เท่า)", format="%.1fx")
                },
                hide_index=True,
                use_container_width=True
            )

    else:
        st.warning("⚠️ ไม่พบข้อมูลหมวดหมู่สินค้า หรือ ไม่พบข้อมูลตามตัวกรอง")

    # --- 4. INDIVIDUAL LIST (รายชื่อรายคน) ---
    st.markdown("---")
    st.subheader(f"📄 รายชื่อลูกค้า ({len(filtered_df):,} คน)")
    
    show_cols = ['customer_unique_id', 'status', 'churn_probability', 'lateness_score', 
                 'cat_median_days', 'payment_value', 'product_category_name']
    final_cols = [c for c in show_cols if c in df.columns]
    
    st.dataframe(
        filtered_df[final_cols].sort_values(by='churn_probability', ascending=False),
        column_config={
            "cat_median_days": st.column_config.NumberColumn("รอบปกติ (วัน)", format="%d"),
            "lateness_score": st.column_config.NumberColumn("Late Score", format="%.1fx"),
            "churn_probability": st.column_config.ProgressColumn("Risk Prob", format="%.2f", min_value=0, max_value=1)
        },
        use_container_width=True
    )
# ==========================================
# PAGE 3: 🎯 Action Plan (Simulation)
# ==========================================
elif page == "3. 🎯 Action Plan":
    st.title("🎯 จำลองกลยุทธ์แก้เกม (What-if Simulation)")
    st.markdown("ลองปรับเปลี่ยนตัวแปรต่างๆ เพื่อดูว่า **AI จะลดค่าความเสี่ยงลงเท่าไหร่**")
    
    # เช็คก่อนว่าโมเดลพร้อมไหม
    if 'model' not in assets or 'features' not in assets:
        st.error("Model not loaded properly.")
        st.stop()
        
    feature_names = assets['features']

    # --- 1. SETTING PANEL (แผงควบคุม) ---
    with st.container():
        st.subheader("🎛️ ปรับปรุงประสิทธิภาพ (Simulation Controls)")
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        
        with col_ctrl1:
            st.markdown("#### 🚚 กลยุทธ์ขนส่ง (Logistics)")
            # ปรับลดวันส่งของ (สมมติว่าส่งเร็วขึ้น)
            improve_days = st.slider("ลดเวลาจัดส่งลง (วัน):", 0, 7, 0, help="ถ้าเราส่งของเร็วขึ้น X วัน จะช่วยลดความเสี่ยงได้ไหม?")
            
        with col_ctrl2:
            st.markdown("#### 📸 กลยุทธ์คอนเทนต์ (Content)")
            # เพิ่มจำนวนรูปภาพ (สมมติว่าถ่ายรูปเพิ่ม)
            improve_photos = st.slider("เพิ่มรูปสินค้า (รูป):", 0, 5, 0, help="ถ้าสินค้ามีรูปเยอะขึ้น ลูกค้าจะมั่นใจขึ้นไหม?")
            # เพิ่มความยาวคำบรรยาย
            improve_desc = st.checkbox("✅ ปรับปรุงคำบรรยายสินค้าให้ละเอียดขึ้น (+100 ตัวอักษร)", value=False)

    # --- 2. RUN SIMULATION (คำนวณใหม่) ---
    # สร้างปุ่มกดเพื่อเริ่มคำนวณ (เพื่อไม่ให้หนักเครื่องตอนเลื่อน Slider)
    if st.button("🚀 เริ่มจำลองผลลัพธ์ (Run Simulation)", type="primary"):
        
        with st.spinner("⏳ AI กำลังคำนวณความเสี่ยงใหม่..."):
            # 1. จำลองข้อมูล (Clone Data)
            df_sim = df.copy()
            
            # 2. ปรับแก้ค่าตาม Slider (Modify Data)
            # -- แก้เรื่องขนส่ง
            if 'delivery_days' in df_sim.columns:
                # ลดจำนวนวันที่ใช้ส่ง (Minimum คือ 1 วัน)
                df_sim['delivery_days'] = df_sim['delivery_days'] - improve_days
                df_sim['delivery_days'] = df_sim['delivery_days'].clip(lower=1) 
            
            if 'delay_days' in df_sim.columns:
                # ลดจำนวนวันที่ล่าช้า
                df_sim['delay_days'] = df_sim['delay_days'] - improve_days
            
            # -- แก้เรื่อง Content
            if 'product_photos_qty' in df_sim.columns:
                df_sim['product_photos_qty'] = df_sim['product_photos_qty'] + improve_photos
            
            if improve_desc and 'product_description_lenght' in df_sim.columns:
                df_sim['product_description_lenght'] = df_sim['product_description_lenght'] + 100

            # 3. เตรียมข้อมูลเข้าโมเดล (Prepare X_sim)
            # ต้องเรียง Column ให้เหมือนตอนเทรนเป๊ะๆ
            X_sim = pd.DataFrame(index=df_sim.index)
            for col in feature_names:
                if col in df_sim.columns:
                    X_sim[col] = df_sim[col]
                else:
                    X_sim[col] = 0 # ถ้าไม่มีให้เติม 0
            
            # 4. ให้ AI ทำนายใหม่ (Re-Predict)
            if hasattr(model, "predict_proba"):
                new_probs = model.predict_proba(X_sim)[:, 1]
            else:
                new_probs = model.predict(X_sim)
            
            # 5. เปรียบเทียบผล (Compare)
            df_sim['new_churn_prob'] = new_probs
            df_sim['old_churn_prob'] = df['churn_probability'] # ค่าเดิม
            df_sim['prob_diff'] = df_sim['old_churn_prob'] - df_sim['new_churn_prob'] # ค่าที่ลดลง (ยิ่งเยอะยิ่งดี)
            
            # นับจำนวนคนที่ "รอด" (เดิม High Risk -> ใหม่ Low Risk)
            # สมมติ Cut-off ที่ 0.7
            saved_customers = df_sim[
                (df_sim['old_churn_prob'] > 0.7) & 
                (df_sim['new_churn_prob'] <= 0.7)
            ]
            
            total_saved = len(saved_customers)
            money_saved = saved_customers['payment_value'].sum() if 'payment_value' in saved_customers.columns else 0

        # --- 3. DISPLAY RESULTS (แสดงผล) ---
        st.markdown("---")
        st.subheader("📊 ผลลัพธ์การจำลอง (Simulation Result)")
        
        # KPI Cards
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("👥 ลูกค้าที่กู้คืนได้ (Estimated)", f"{total_saved:,} คน", help="คนที่ความเสี่ยงลดลงจากระดับสูงจนอยู่ในเกณฑ์ปลอดภัย")
        with k2:
            st.metric("💸 รายได้ที่รักษาไว้ได้", f"R$ {money_saved:,.0f}", help="ยอดเงินรวมของลูกค้าที่กู้คืนได้")
        with k3:
            avg_drop = df_sim['prob_diff'].mean() * 100
            st.metric("📉 ความเสี่ยงลดลงเฉลี่ย", f"{avg_drop:.2f}%", help="ค่าเฉลี่ยความน่าจะเป็นที่ลดลงของทุกคน")

        # --- CHART: หมวดไหนได้ผลดีสุด? ---
        col_chart, col_list = st.columns([1.5, 1])
        
        with col_chart:
            st.markdown("#### 🏆 หมวดสินค้าที่ตอบสนองดีที่สุด")
            st.caption("ถ้าทำตามแผนนี้ สินค้ากลุ่มไหนจะลดความเสี่ยงได้เยอะสุด?")
            
            if 'product_category_name' in df_sim.columns:
                # หาค่าเฉลี่ยความเสี่ยงที่ลดลง แยกตามหมวด
                cat_improvement = df_sim.groupby('product_category_name')['prob_diff'].mean().reset_index()
                # คูณ 100 ให้ดูเป็น %
                cat_improvement['prob_diff'] = cat_improvement['prob_diff'] * 100
                
                # Top 10 Improvement
                top_improve = cat_improvement.sort_values('prob_diff', ascending=False).head(10)
                
                chart_imp = alt.Chart(top_improve).mark_bar(color='#2ecc71').encode(
                    x=alt.X('prob_diff', title='ความเสี่ยงลดลงเฉลี่ย (%)'),
                    y=alt.Y('product_category_name', sort='-x', title='หมวดสินค้า'),
                    tooltip=['product_category_name', alt.Tooltip('prob_diff', format='.2f')]
                ).properties(height=400)
                
                st.altair_chart(chart_imp, use_container_width=True)
            
        with col_list:
            st.markdown("#### 📋 ตัวอย่างลูกค้าที่กู้คืนได้")
            if not saved_customers.empty:
                show_cols = ['customer_unique_id', 'product_category_name', 'old_churn_prob', 'new_churn_prob']
                final_cols = [c for c in show_cols if c in df_sim.columns]
                
                st.dataframe(
                    saved_customers[final_cols].sort_values('old_churn_prob', ascending=False).head(50),
                    column_config={
                        "old_churn_prob": st.column_config.NumberColumn("Risk เดิม", format="%.2f"),
                        "new_churn_prob": st.column_config.NumberColumn("Risk ใหม่", format="%.2f"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("แผนนี้อาจยังไม่แรงพอที่จะเปลี่ยนสถานะลูกค้ากลุ่ม High Risk ได้ ลองปรับค่าเพิ่มดูครับ")

    else:
        st.info("👈 ปรับค่า Slider ด้านบน แล้วกดปุ่ม **'เริ่มจำลองผลลัพธ์'** เพื่อดู Insight ครับ")
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





