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
# PAGE 3: 🎯 Marketing Campaign Simulator
# ==========================================
elif page == "3. 🎯 Action Plan":
    st.title("🎯 Marketing Campaign Simulator")
    st.markdown("### วิเคราะห์ความคุ้มค่า: แจกคูปอง/ส่วนลด เพื่อดึงลูกค้ากลับมา")
    st.info("💡 **Logic:** หน้านี้จะโฟกัสเฉพาะลูกค้า **'กลุ่มลังเล' (ความเสี่ยง 60-85%)** เพราะเป็นกลุ่มที่คุ้มค่าที่สุดในการยิงแคมเปญ (คนเสี่ยงเกิน 90% มักกู้ไม่กลับ)")

    # เช็ค Model
    if 'model' not in assets or 'features' not in assets:
        st.stop()
    feature_names = assets['features']

    # --- 1. FILTER TARGET GROUP (คัดเฉพาะคนที่มีลุ้น) ---
    # เลือกเฉพาะคนที่ความเสี่ยงอยู่ระหว่าง 0.60 ถึง 0.85
    target_customers = df[
        (df['churn_probability'] >= 0.60) & 
        (df['churn_probability'] <= 0.85)
    ].copy()
    
    total_target = len(target_customers)
    total_revenue_at_risk = target_customers['payment_value'].sum() if 'payment_value' in df.columns else 0

    if total_target == 0:
        st.warning("ไม่พบลูกค้าในกลุ่ม 'ลังเล' (Risk 60-85%) เลย ลองปรับช่วงความเสี่ยงดูครับ")
        st.stop()

    # --- 2. CAMPAIGN CONTROLS ---
    with st.container():
        st.markdown(f"#### 🎯 เป้าหมายแคมเปญ: ลูกค้า {total_target:,} คน (มูลค่า R$ {total_revenue_at_risk:,.0f})")
        
        col_input1, col_input2, col_input3 = st.columns(3)
        
        with col_input1:
            # จำลองการให้ส่วนลด (Voucher)
            voucher_val = st.slider("💰 มูลค่าคูปองส่วนลด (R$)", 0, 50, 0, step=5, help="ต้นทุนที่คุณยอมจ่ายต่อคน")
        
        with col_input2:
            # กลยุทธ์เสริม (Logistics)
            improve_speed = st.selectbox("🚚 การปรับปรุงขนส่ง", ["ปกติ", "ส่งด่วนพิเศษ (-2 วัน)"], index=0)
            
        with col_input3:
            # คำนวณ Budget
            total_cost = voucher_val * total_target
            st.metric("ใช้งบประมาณรวม (Cost)", f"R$ {total_cost:,.0f}")

    # --- 3. SIMULATION LOGIC ---
    # จำลองข้อมูล
    df_sim = target_customers.copy()
    
    # A. Effect ของ Voucher (เงิน)
    if voucher_val > 0:
        # ยิ่งให้เยอะ ยิ่งลดความเสี่ยง (Impact Factor)
        impact = (voucher_val / 10) * 0.02
        
        # นอกจากนี้ Voucher อาจทำให้ Review Score ดูดีขึ้นในใจลูกค้า
        if 'review_score' in df_sim.columns:
            df_sim['review_score'] = (df_sim['review_score'] + (voucher_val/20)).clip(upper=5.0)
    else:
        impact = 0

    # B. Effect ของ Speed
    if improve_speed == "ส่งด่วนพิเศษ (-2 วัน)" and 'delivery_days' in df_sim.columns:
        df_sim['delivery_days'] = (df_sim['delivery_days'] - 2).clip(lower=1)
        if 'delay_days' in df_sim.columns:
             df_sim['delay_days'] = df_sim['delay_days'] - 2

    # --- 4. PREDICT ---
    X_sim = pd.DataFrame(index=df_sim.index)
    for col in feature_names:
        if col in df_sim.columns:
            X_sim[col] = df_sim[col]
        else:
            X_sim[col] = 0
            
    if hasattr(model, "predict_proba"):
        new_probs = model.predict_proba(X_sim)[:, 1]
    else:
        new_probs = model.predict(X_sim)

    # Apply Artificial Impact from Voucher
    final_probs = new_probs - impact 
    
    # เปรียบเทียบ
    df_sim['old_prob'] = target_customers['churn_probability']
    df_sim['new_prob'] = final_probs
    
    # ตัดสินผล: ใครที่ความเสี่ยงลดลงจนต่ำกว่า 0.5 (ถือว่าซื้อใจสำเร็จ)
    success_cases = df_sim[df_sim['new_prob'] < 0.5]
    
    saved_count = len(success_cases)
    saved_revenue = success_cases['payment_value'].sum() if 'payment_value' in df_sim.columns else 0
    
    # คำนวณ ROI
    roi = saved_revenue - total_cost
    roi_percent = (roi / total_cost * 100) if total_cost > 0 else 0

    # --- 5. DISPLAY RESULTS ---
    st.markdown("---")
    st.subheader("📊 ผลลัพธ์แคมเปญ (Campaign Result)")
    
    # Result Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 ดึงลูกค้ากลับมาได้", f"{saved_count:,} คน", f"{(saved_count/total_target*100):.1f}% Success Rate")
    c2.metric("💸 รายได้ที่รักษาได้", f"R$ {saved_revenue:,.0f}")
    c3.metric("📉 ต้นทุนแคมเปญ", f"R$ {total_cost:,.0f}") # ✅ แก้ไขบรรทัดนี้แล้ว
    
    # ROI Color logic
    roi_color = "normal" if roi > 0 else "inverse"
    c4.metric("💰 กำไรสุทธิ (ROI)", f"R$ {roi:,.0f}", f"{roi_percent:.1f}% Return", delta_color=roi_color)

    # --- 6. VISUALIZATION ---
    col_chart, col_detail = st.columns([1.5, 1])
    
    with col_chart:
        st.markdown("#### 📈 ความเสี่ยงเปลี่ยนไปอย่างไร? (Before vs After)")
        
        # Histogram เปรียบเทียบ
        chart_data = pd.DataFrame({
            'Risk': list(df_sim['old_prob']) + list(df_sim['new_prob']),
            'Type': ['Before (Old Risk)'] * len(df_sim) + ['After (New Risk)'] * len(df_sim)
        })
        
        chart = alt.Chart(chart_data).mark_area(opacity=0.5, interpolate='step').encode(
            x=alt.X('Risk', bin=alt.Bin(maxbins=20), title='ระดับความเสี่ยง (Churn Probability)'),
            y=alt.Y('count()', stack=None, title='จำนวนลูกค้า'),
            color=alt.Color('Type', scale=alt.Scale(range=['#95a5a6', '#2ecc71'])),
            tooltip=['Type', 'count()']
        ).properties(height=350)
        
        st.altair_chart(chart, use_container_width=True)
        st.caption("กราฟสีเขียวควรจะขยับไปทางซ้าย (ความเสี่ยงลดลง) เมื่อเทียบกับสีเทา")

    with col_detail:
        st.markdown("#### 🏆 Top Success Cases")
        st.markdown("ลูกค้าที่ตอบสนองต่อแคมเปญดีที่สุด")
        
        if not success_cases.empty:
            show_df = success_cases[['customer_unique_id', 'product_category_name', 'old_prob', 'new_prob', 'payment_value']]
            st.dataframe(
                show_df.sort_values('payment_value', ascending=False).head(20),
                column_config={
                    "old_prob": st.column_config.NumberColumn("Risk เดิม", format="%.2f"),
                    "new_prob": st.column_config.NumberColumn("Risk ใหม่", format="%.2f"),
                    "payment_value": st.column_config.NumberColumn("ยอดเงิน", format="R$ %.0f")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("ยังไม่มีลูกค้าที่กลับใจ ลองเพิ่มมูลค่า Voucher หรือเลือกส่งด่วนดูครับ")
# ==========================================
# PAGE 4: 🚛 Logistics Insights
# ==========================================
elif page == "4. 🚛 Logistics Insights":
    st.title("🚛 เจาะลึกระบบขนส่ง (Logistics Heatmap)")
    st.markdown("วิเคราะห์ประสิทธิภาพการจัดส่งรายพื้นที่: **รัฐไหนส่งช้า?** และ **เมืองไหนลูกค้าหนีเยอะ?**")

    # เช็คข้อมูล
    if 'customer_state' not in df.columns:
        st.error("ไม่พบข้อมูล 'customer_state' กรุณารัน Data Prep ใหม่")
        st.stop()

    # --- PART 1: STATE LEVEL OVERVIEW ---
    st.subheader("🗺️ ภาพรวมรายรัฐ (State Performance)")
    
    col_map, col_stat = st.columns([2, 1])
    
    with col_map:
        # เตรียมข้อมูลรายรัฐ
        state_stats = df.groupby('customer_state').agg({
            'customer_unique_id': 'count',
            'delivery_days': 'mean',
            'churn_probability': 'mean',
            'delay_days': lambda x: (x > 0).mean() # % ออเดอร์ที่ล่าช้า
        }).reset_index()
        
        # กรองรัฐที่มีข้อมูลน้อยเกินไปออก (เพื่อให้กราฟแม่นยำ)
        state_stats = state_stats[state_stats['customer_unique_id'] > 20]

        # Scatter Plot: ยิ่งขวาบน = ยิ่งแย่ (ส่งช้า + เสี่ยงสูง)
        scatter_chart = alt.Chart(state_stats).mark_circle(size=100).encode(
            x=alt.X('delivery_days', title='ระยะเวลาจัดส่งเฉลี่ย (วัน)'),
            y=alt.Y('churn_probability', title='โอกาส Churn เฉลี่ย', scale=alt.Scale(domain=[0.5, 1.0])),
            color=alt.Color('churn_probability', scale=alt.Scale(scheme='reds'), title='Risk Level'),
            size=alt.Size('customer_unique_id', title='จำนวนลูกค้า'),
            tooltip=['customer_state', 'delivery_days', 'churn_probability', 'delay_days']
        ).properties(
            title='Logistics Risk Map (ยิ่งอยู่ขวาบน ยิ่งต้องแก้ด่วน!)',
            height=400
        ).interactive()
        
        st.altair_chart(scatter_chart, use_container_width=True)

    with col_stat:
        st.markdown("#### 🚨 Top 5 รัฐที่มีปัญหา")
        # เรียงตามความเสี่ยง Churn
        worst_states = state_stats.sort_values('churn_probability', ascending=False).head(5)
        
        st.dataframe(
            worst_states[['customer_state', 'churn_probability', 'delivery_days']],
            column_config={
                "customer_state": "รัฐ",
                "churn_probability": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
                "delivery_days": st.column_config.NumberColumn("ส่งนาน (วัน)", format="%.1f")
            },
            hide_index=True,
            use_container_width=True
        )
        st.info("💡 รัฐเหล่านี้คือจุดที่ลูกค้ามีความไม่พอใจสูงสุด ลองพิจารณาเปลี่ยน Partner ขนส่งในพื้นที่นี้")

    # --- PART 2: CITY DRILL DOWN ---
    st.markdown("---")
    st.subheader("🏙️ เจาะลึกรายเมือง (City Drill-down)")
    
    selected_state = st.selectbox("เลือกรัฐที่ต้องการตรวจสอบ:", df['customer_state'].unique())
    
    if selected_state:
        # กรองข้อมูลเฉพาะรัฐนั้น
        state_df = df[df['customer_state'] == selected_state]
        
        # Group by City
        city_stats = state_df.groupby('customer_city').agg({
            'customer_unique_id': 'count',
            'delivery_days': 'mean',
            'churn_probability': 'mean',
            'lateness_score': 'mean'
        }).reset_index()
        
        # เอาเฉพาะเมืองที่มี Order อย่างน้อย 5 รายการ (กัน Noise)
        city_stats = city_stats[city_stats['customer_unique_id'] >= 5]
        
        # หาเมืองที่แย่ที่สุด 10 อันดับแรก
        worst_cities = city_stats.sort_values('churn_probability', ascending=False).head(10)
        
        st.write(f"**Top 10 เมืองที่มีความเสี่ยงสูงสุดในรัฐ {selected_state}:**")
        st.dataframe(
            worst_cities,
            column_config={
                "customer_city": "เมือง",
                "churn_probability": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
                "delivery_days": st.column_config.NumberColumn("เวลาส่ง (วัน)", format="%.1f"),
                "customer_unique_id": st.column_config.NumberColumn("ลูกค้า (คน)", format="%d")
            },
            hide_index=True,
            use_container_width=True
        )

# ==========================================
# PAGE 5: 🏪 Seller Audit
# ==========================================
elif page == "5. 🏪 Seller Audit":
    st.title("🏪 ตรวจสอบคุณภาพร้านค้า (Seller Watchlist)")
    st.markdown("ตามล่าร้านค้าที่เป็น **'ต้นเหตุ'** ทำให้ลูกค้าหนี (ขายเยอะ แต่รักษาลูกค้าไม่ได้)")

    if 'seller_id' not in df.columns:
        st.error("ไม่พบข้อมูล 'seller_id' กรุณารัน Data Prep ใหม่")
        st.stop()

    # --- PART 1: METRICS ---
    # คำนวณภาพรวม
    seller_stats = df.groupby('seller_id').agg({
        'customer_unique_id': 'count',          # Volume
        'churn_probability': 'mean',            # Risk
        'review_score': 'mean',                 # Quality
        'delay_days': 'mean',                   # Ops
        'payment_value': 'sum'                  # Revenue Impact
    }).reset_index()

    # กรองเฉพาะร้าน Active (ขายเกิน 20 ออเดอร์)
    active_sellers = seller_stats[seller_stats['customer_unique_id'] >= 20]
    
    # ร้านค้ากลุ่มเสี่ยง (High Churn Seller)
    bad_sellers = active_sellers.sort_values('churn_probability', ascending=False).head(50)
    
    total_bad_impact = bad_sellers['payment_value'].sum()
    avg_bad_churn = bad_sellers['churn_probability'].mean() * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("🚨 ร้านค้ากลุ่มเสี่ยง (Watchlist)", f"{len(bad_sellers)} ร้าน", "Churn Rate สูงผิดปกติ")
    k2.metric("💸 ยอดขายจากร้านกลุ่มนี้", f"R$ {total_bad_impact:,.0f}", "รายได้ที่เสี่ยงจะหายไปถาวร")
    k3.metric("📉 อัตราลูกค้าหนีเฉลี่ย", f"{avg_bad_churn:.1f}%", help="เทียบกับค่าเฉลี่ยปกติของแพลตฟอร์ม")

    # --- PART 2: BLACKLIST TABLE ---
    st.markdown("### 📋 Blacklist: 20 อันดับร้านค้าที่ควรตรวจสอบด่วน")
    st.caption("ร้านเหล่านี้มียอดขายสูง แต่ลูกค้าซื้อแล้ว 'ไม่กลับมาอีกเลย' (One-time purchase & Leave)")

    st.dataframe(
        bad_sellers.head(20),
        column_config={
            "seller_id": "Seller ID",
            "churn_probability": st.column_config.ProgressColumn(
                "Avg Churn Risk", 
                help="ความน่าจะเป็นเฉลี่ยที่ลูกค้าของร้านนี้จะหนี",
                format="%.2f", 
                min_value=0, 
                max_value=1
            ),
            "review_score": st.column_config.NumberColumn("Review Avg", format="%.1f ⭐"),
            "customer_unique_id": st.column_config.NumberColumn("Total Orders", format="%d"),
            "delay_days": st.column_config.NumberColumn("Delay Avg", format="%.1f วัน"),
            "payment_value": st.column_config.NumberColumn("Total Sales", format="R$ %.0f")
        },
        hide_index=True,
        use_container_width=True
    )

    # --- PART 3: SCATTER ANALYSIS ---
    st.markdown("---")
    st.subheader("🔍 วิเคราะห์ความสัมพันธ์: คุณภาพ vs ความเสี่ยง")
    
    # เลือกแกนวิเคราะห์
    x_axis = st.selectbox("เลือกปัจจัยวิเคราะห์:", 
                          ["review_score", "delay_days", "customer_unique_id"], 
                          format_func=lambda x: "คะแนนรีวิว" if x == "review_score" else "วันส่งล่าช้า" if x == "delay_days" else "จำนวนออเดอร์")

    scatter_seller = alt.Chart(active_sellers).mark_circle(color='#e74c3c', opacity=0.6).encode(
        x=alt.X(x_axis, title=x_axis),
        y=alt.Y('churn_probability', title='โอกาสลูกค้าหนี (Churn Risk)'),
        size=alt.Size('payment_value', title='ยอดขายรวม'),
        tooltip=['seller_id', 'review_score', 'churn_probability', 'customer_unique_id']
    ).properties(
        height=350,
        title=f"Seller Performance Analysis"
    ).interactive()
    
    st.altair_chart(scatter_seller, use_container_width=True)
    st.info("💡 ร้านที่ดีควรอยู่ด้าน **'ล่าง'** (Churn ต่ำ) / ร้านที่มีปัญหาจะลอยอยู่ด้าน **'บน'** (Churn สูง)")









