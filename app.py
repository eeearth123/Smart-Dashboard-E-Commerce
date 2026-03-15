import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import datetime
import time

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Olist Executive Cockpit (Real-time)",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
# 2. LOAD ASSETS
# ==========================================

from google.oauth2 import service_account
from google.cloud import bigquery

# --- โหลดข้อมูลจาก BigQuery ---
@st.cache_data(ttl=600)
def load_bq_data():
    try:
        creds_info = st.secrets["connections"]["bigquery"]["service_account_info"]
        scopes = [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/bigquery"
        ]
        credentials = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=scopes
        )
        client = bigquery.Client(
            credentials=credentials,
            project=creds_info["project_id"],
            location="asia-southeast1"
        )
        query = "SELECT * FROM `academic-moon-483615-t2.Dashboard.input`"
        df = client.query(query).to_dataframe()
        return df, None
    except Exception as e:
        return None, str(e)

# --- Feature Engineering (ครบทุกคอลัมน์ที่แต่ละหน้าต้องใช้) ---
def process_features(df):
    df = df.copy()

    # 1. แปลงวันที่
    date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 2. payment_value
    if 'payment_value' not in df.columns:
        if 'price' in df.columns and 'freight_value' in df.columns:
            df['payment_value'] = df['price'] + df['freight_value']
        elif 'price' in df.columns:
            df['payment_value'] = df['price']
        else:
            df['payment_value'] = 0

    # 3. freight_ratio
    if 'freight_ratio' not in df.columns:
        if 'freight_value' in df.columns and 'price' in df.columns:
            df['freight_ratio'] = df['freight_value'] / df['price'].replace(0, np.nan)
            df['freight_ratio'] = df['freight_ratio'].fillna(0)
        else:
            df['freight_ratio'] = 0

    # 4. delivery_days & delay_days (คำนวณจากวันที่จริง ถ้าไม่มีให้สร้างใหม่)
    if 'delivery_days' not in df.columns:
        if 'order_delivered_customer_date' in df.columns and 'order_purchase_timestamp' in df.columns:
            df['delivery_days'] = (
                df['order_delivered_customer_date'] - df['order_purchase_timestamp']
            ).dt.days
        else:
            df['delivery_days'] = np.nan

    if 'delay_days' not in df.columns:
        if 'order_delivered_customer_date' in df.columns and 'order_estimated_delivery_date' in df.columns:
            df['delay_days'] = (
                df['order_delivered_customer_date'] - df['order_estimated_delivery_date']
            ).dt.days
        else:
            df['delay_days'] = 0

    # 5. cat_median_days — รอบการซื้อซ้ำต่อหมวด
    # ⚠️ ไม่ใช้ delivery_days (10 วัน) เพราะ delivery_days ≠ รอบซื้อซ้ำ
    # ใช้ gap ระหว่าง order ของ customer เดียวกัน, fallback = 180 วัน
    if 'cat_median_days' not in df.columns:
        if 'product_category_name' in df.columns and 'order_purchase_timestamp' in df.columns and 'customer_unique_id' in df.columns:
            tmp = df.sort_values(['customer_unique_id', 'product_category_name', 'order_purchase_timestamp'])
            tmp['prev_ts'] = tmp.groupby(['customer_unique_id', 'product_category_name'])['order_purchase_timestamp'].shift(1)
            tmp['order_gap'] = (tmp['order_purchase_timestamp'] - tmp['prev_ts']).dt.days
            valid_gaps = tmp[(tmp['order_gap'] >= 7) & (tmp['order_gap'] <= 730)]
            if len(valid_gaps) > 10:
                cat_med = valid_gaps.groupby('product_category_name')['order_gap'].median().rename('cat_median_days')
                df = df.merge(cat_med, on='product_category_name', how='left')
            else:
                df['cat_median_days'] = 180  # ถ้าลูกค้าส่วนใหญ่ซื้อครั้งเดียว → default 180 วัน
        else:
            df['cat_median_days'] = 180
    df['cat_median_days'] = df['cat_median_days'].fillna(180).clip(lower=7)

    # 6. lateness_score
    # ⚠️ KEY FIX: ใช้ max(order_date) ในข้อมูล ไม่ใช่ today
    # ถ้าใช้ today(2026) กับข้อมูลปี 2018 → days=2800, cat_median=10 → lateness=280 → ทุกคน Lost!
    if 'lateness_score' not in df.columns:
        if 'order_purchase_timestamp' in df.columns:
            ref_date = df['order_purchase_timestamp'].max()  # วันล่าสุดในข้อมูล
            if 'customer_unique_id' in df.columns:
                last_order = df.groupby('customer_unique_id')['order_purchase_timestamp'].transform('max')
                df['days_since_last_order'] = (ref_date - last_order).dt.days
            else:
                df['days_since_last_order'] = (ref_date - df['order_purchase_timestamp']).dt.days
        else:
            df['days_since_last_order'] = 90
        df['lateness_score'] = (df['days_since_last_order'] / df['cat_median_days'].replace(0, 1)).clip(lower=0)

    # 7. review_score (ถ้าไม่มีให้ใส่ค่ากลาง)
    if 'review_score' not in df.columns:
        df['review_score'] = np.nan

    return df

# --- โหลด Model ---
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'olist_churn_model_best.pkl')
    features_path = os.path.join(current_dir, 'model_features_best.pkl')
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features, None
    except Exception as e:
        return None, None, f"Model Error: {e}"

# --- Refresh Button ---
with st.sidebar:
    if st.button('🔄 Refresh Data'):
        st.cache_data.clear()
        st.rerun()

# --- โหลดข้อมูล ---
df_raw, bq_error = load_bq_data()
model, feature_names, model_error = load_models()

if bq_error:
    st.error(f"⚠️ BigQuery Error: {bq_error}")
    st.info("ตรวจสอบว่าได้ตั้งค่า Secrets ใน Streamlit Cloud หรือไฟล์ .streamlit/secrets.toml หรือยัง?")
    st.stop()

if model_error:
    st.warning(f"⚠️ {model_error}")

# ==========================================
# 3. PREPARE DATA & PREDICTION
# ==========================================
df = process_features(df_raw)

# 3.1 Predict churn_probability
if 'churn_probability' not in df.columns and model is not None and feature_names:
    X_pred = pd.DataFrame(index=df.index)
    for col in feature_names:
        X_pred[col] = df[col] if col in df.columns else 0
    try:
        if hasattr(model, "predict_proba"):
            df['churn_probability'] = model.predict_proba(X_pred)[:, 1]
        else:
            df['churn_probability'] = model.predict(X_pred)
    except Exception as e:
        st.warning(f"Prediction fallback: {e}")
        df['churn_probability'] = 0.5
elif 'churn_probability' not in df.columns:
    df['churn_probability'] = 0.5

# 3.2 is_churn
if 'is_churn' not in df.columns:
    df['is_churn'] = (df['churn_probability'] > 0.5).astype(int)

# 3.3 cat_churn_risk — คำนวณใหม่ทุกรอบ ไม่ cache เพราะขึ้นกับข้อมูลล่าสุด
# สูตร: mean(churn_probability) ต่อหมวดสินค้า
# ตัวอย่าง: Electronics & Tech มี churn_prob เฉลี่ย 0.82 → cat_churn_risk = 0.82
# → Page 3 Tab 4 ใช้ค่านี้จับหมวดที่ลูกค้ามีแนวโน้ม churn สูงแบบ structural
if 'product_category_name' in df.columns:
    cat_risk_map     = df.groupby('product_category_name')['churn_probability'].mean()
    df['cat_churn_risk'] = df['product_category_name'].map(cat_risk_map)
else:
    df['cat_churn_risk'] = df['churn_probability']  # fallback

# 3.4 Status
def get_status(row):
    prob = row.get('churn_probability', 0)
    late = row.get('lateness_score', 0)
    if late > 3.0:   return 'Lost (Late > 3x)'
    if prob > 0.75:  return 'High Risk'
    if late > 1.5:   return 'Warning (Late > 1.5x)'
    if prob > 0.5:   return 'Medium Risk'
    return 'Active'

if 'status' not in df.columns:
    df['status'] = df.apply(get_status, axis=1)

# ==========================================
# 4. NAVIGATION
# ==========================================
st.sidebar.title("✈️ Olist Cockpit")
st.sidebar.success(f"✅ โหลดข้อมูลแล้ว ({len(df):,} rows)")
page = st.sidebar.radio("Navigation", [
    "1. 📊 Executive Summary",
    "2. 🔍 Customer Detail",
    "3. 🎯 Action Plan",
    "4. 🚛 Logistics Insights",
    "5. 🏪 Seller Audit",
    "6. 🔄 Buying Cycle Analysis"
])
st.sidebar.markdown("---")
st.sidebar.info("Select a page to analyze different aspects of your business.")

# ==========================================
# PAGE 1: 📊 Executive Summary
# ==========================================
if page == "1. 📊 Executive Summary":
    st.title("📊 Executive Summary (Real-time Cloud)")

    with st.expander("ℹ️ วิธีการแบ่งกลุ่มลูกค้า (Segmentation Logic) - กดเพื่ออ่าน"):
        st.markdown("""
        **ลำดับการตรวจสอบ (Priority):**
        1. **🔴 Lost:** หายไปนานเกิน 3 เท่าของรอบปกติ (`Lateness > 3.0`) -> เลิกซื้อชัวร์
        2. **🟥 High Risk:** ยังไม่นานมาก แต่ **AI ทำนายว่าเสี่ยง > 75%** -> มีปัญหาซ่อนอยู่
        3. **🟧 Warning:** AI บอกโอเค แต่ลูกค้าเริ่มหายเกิน 1.5 เท่า (`Lateness > 1.5`) -> ต้องเตือน
        4. **🟨 Medium Risk:** มาตรงเวลา แต่ AI ให้ความเสี่ยง 50-75%
        5. **🟩 Active:** มาตรงเวลา และ AI บอกว่าเสี่ยงต่ำ
        """)

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

    total_customers = len(df_display)
    if total_customers > 0:
        risk_df    = df_display[df_display['status'].isin(['High Risk', 'Warning (Late > 1.5x)'])]
        risk_count = len(risk_df)
        churn_rate = (risk_count / total_customers) * 100
        rev_at_risk = risk_df['payment_value'].sum() if 'payment_value' in df_display.columns else 0
        active_count = len(df_display[df_display['status'] == 'Active'])
        cycle_text = f"{df_display['cat_median_days'].mean():.0f} วัน" if 'cat_median_days' in df_display.columns else "N/A"
    else:
        churn_rate = rev_at_risk = risk_count = active_count = 0
        cycle_text = "-"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🚨 Churn Rate",         f"{churn_rate:.1f}%")
    k2.metric("💸 Revenue at Risk",    f"R$ {rev_at_risk:,.0f}")
    k3.metric("👥 Risk vs Total",      f"{risk_count:,} / {total_customers:,}")
    k4.metric("✅ Active Customers",   f"{active_count:,}")
    k5.metric("🔄 รอบซื้อปกติ (Cycle)", cycle_text)

    st.markdown("---")
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📈 Churn Risk Trend & Forecast")
        if 'order_purchase_timestamp' in df_display.columns and not df_display.empty:
            df_display['month_year'] = df_display['order_purchase_timestamp'].dt.to_period('M').astype(str)
            trend_df = df_display.groupby('month_year')['churn_probability'].mean().reset_index()
            trend_df.columns = ['Date', 'Churn_Prob']
            trend_df['Type'] = 'Actual'
            trend_df['Date'] = pd.to_datetime(trend_df['Date'])

            if not trend_df.empty:
                last_date = trend_df['Date'].max()
                last_val  = trend_df['Churn_Prob'].iloc[-1]
                anchor_df = pd.DataFrame({'Date': [last_date], 'Churn_Prob': [last_val], 'Type': ['Forecast']})
                future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 4)]
                future_vals  = [last_val * (1 + 0.02*i) for i in range(1, 4)]
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
        st.subheader("💰 Revenue Share by Risk")
        if not df_display.empty:
            status_stats = df_display.groupby('status').agg(
                Count=('customer_unique_id', 'count'),
                Total_Revenue=('payment_value', 'sum')
            ).reset_index()
            domain = ['Active', 'Medium Risk', 'Warning (Late > 1.5x)', 'High Risk', 'Lost (Late > 3x)']
            range_ = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#95a5a6']
            donut = alt.Chart(status_stats).mark_arc(innerRadius=60).encode(
                theta=alt.Theta("Count", type="quantitative"),
                color=alt.Color("status", scale=alt.Scale(domain=domain, range=range_), legend=dict(orient='bottom')),
                tooltip=['status', alt.Tooltip('Count', format=','), alt.Tooltip('Total_Revenue', format=',.0f')]
            ).properties(height=350)
            st.altair_chart(donut, use_container_width=True)

# ==========================================
# PAGE 2: 🔍 Customer Detail
# ==========================================
elif page == "2. 🔍 Customer Detail":
    st.title("🔍 เจาะลึกกลุ่มเสี่ยง (Customer Deep Dive)")

    with st.expander("🔎 ตัวกรองข้อมูล (Filters)", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            risk_opts  = ['High Risk', 'Warning (Late > 1.5x)', 'Medium Risk', 'Lost (Late > 3x)', 'Active']
            sel_status = st.multiselect("1. สถานะ:", risk_opts, default=['High Risk', 'Warning (Late > 1.5x)'])
        with f2:
            all_cats = list(df['product_category_name'].unique()) if 'product_category_name' in df.columns else []
            sel_cats = st.multiselect("2. หมวดสินค้า:", all_cats)
        with f3:
            search_id = st.text_input("3. ค้นหา ID:", "")

    mask = df['status'].isin(sel_status)
    if sel_cats:  mask = mask & df['product_category_name'].isin(sel_cats)
    if search_id: mask = mask & df['customer_unique_id'].str.contains(search_id, case=False, na=False)
    filtered_df = df[mask]

    if 'product_category_name' in df.columns and not filtered_df.empty:
        cat_overview = df.groupby('product_category_name').agg(
            Total=('customer_unique_id', 'count'),
            Cycle_Days=('cat_median_days', 'mean')
        ).reset_index()
        cat_risk = filtered_df.groupby('product_category_name').agg(
            Risk_Count=('customer_unique_id', 'count')
        ).reset_index()
        cat_stats = pd.merge(cat_risk, cat_overview, on='product_category_name', how='left')
        cat_stats['Risk_Pct'] = cat_stats['Risk_Count'] / cat_stats['Total']
        cat_stats = cat_stats.sort_values('Risk_Count', ascending=False)

        col_c, col_t = st.columns([1.5, 2.5])
        with col_c:
            st.subheader("📊 Top 10 หมวดเสี่ยง")
            base    = alt.Chart(cat_stats.head(10)).encode(y=alt.Y('product_category_name', sort='-x', title=None))
            b_total = base.mark_bar(color='#f0f2f6').encode(x='Total', tooltip=['product_category_name', 'Total'])
            b_risk  = base.mark_bar(color='#e74c3c').encode(x='Risk_Count', tooltip=['Risk_Count', 'Risk_Pct'])
            st.altair_chart(b_total + b_risk, use_container_width=True)
        with col_t:
            st.subheader("📋 รายละเอียด")
            st.dataframe(cat_stats, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"📄 รายชื่อลูกค้า ({len(filtered_df):,} คน)")
    show_cols  = ['customer_unique_id', 'status', 'churn_probability', 'lateness_score', 'cat_median_days', 'payment_value', 'product_category_name']
    final_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(
        filtered_df[final_cols].sort_values('churn_probability', ascending=False),
        column_config={
            "churn_probability": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
            "lateness_score": st.column_config.NumberColumn("Late Score", format="%.1fx")
        },
        use_container_width=True
    )

# ==========================================
# PAGE 3: 🎯 Action Plan & Simulator
# ==========================================
elif page == "3. 🎯 Action Plan":
    st.title("🎯 Action Plan & Simulator")
    st.markdown("จำลองการจัดสรรงบประมาณและปรับแต่งแคมเปญเพื่อลดอัตราการสูญเสียลูกค้า (Churn)")

    # กรองข้อมูลหมวดหมู่ (ถ้ามี)
    all_cats = sorted([x for x in df['product_category_name'].unique() if pd.notna(x)]) if 'product_category_name' in df.columns else []
    sel_cats_p3 = st.multiselect("หมวดสินค้า (ปล่อยว่าง = ดูภาพรวมทั้งหมด):", all_cats, key="p3_cat_multiselect")
    
    if sel_cats_p3:
        df_p3 = df[df['product_category_name'].isin(sel_cats_p3)].copy()
        st.caption(f"กำลังวิเคราะห์หมวด: {', '.join(sel_cats_p3[:3])}...")
    else:
        df_p3 = df.copy()
        st.caption("กำลังวิเคราะห์: ภาพรวมทุกหมวด")

    avg_ltv = df_p3['payment_value'].mean() if 'payment_value' in df_p3.columns else 150

    st.markdown("---")
    
    # 🌟 ส่วนที่ 1: รับค่างบประมาณรวม
    st.subheader("💰 1. ระบบจัดสรรงบประมาณอัจฉริยะ (Smart Budget Allocator)")
    total_budget = st.number_input("ระบุงบประมาณการตลาดรวม (Total Budget - R$):", min_value=1000, value=50000, step=5000)
    
    # สร้างกล่องเปล่า (Container) ไว้รอรับตารางสรุปจากด้านล่าง
    alloc_container = st.container()
    
    st.markdown("---")
    
    # 🎯 ส่วนที่ 2: แผงควบคุมแคมเปญ
    st.subheader("🎯 2. แผงควบคุมแคมเปญรายปัญหา (Targeted Campaigns)")
    st.markdown("ปรับแต่งงบต่อหัวและวิธีคำนวณความสำเร็จ ระบบด้านบนจะจัดสรรงบใหม่ให้ทันที!")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚚 1. สายค่าส่งแพง", 
        "🐌 2. สายรอของนาน", 
        "😡 3. สายรีวิวแย่", 
        "💳 4. สายซื้อแพงจ่ายเต็ม"
    ])
    
    campaigns = [] # เก็บข้อมูลแคมเปญเพื่อส่งไปให้ส่วนที่ 1 คำนวณ

    # --- TAB 1: ค่าส่งแพง ---
    with tab1:
        st.markdown("#### 🚚 แคมเปญแจกโค้ดส่งฟรี (Free Shipping Drop)")
        target_df = df_p3[df_p3['freight_ratio'] > 0.2] if 'freight_ratio' in df_p3.columns else pd.DataFrame()
        n_target = len(target_df)
        st.info(f"👥 พบลูกค้าที่จ่ายค่าส่งแพง (เกิน 20% ของราคาสินค้า): **{n_target:,} คน**")
        
        if n_target > 0:
            c1, c2 = st.columns(2)
            with c1: cost_head = st.number_input("งบต่อหัว (R$):", value=20, key="cost_t1")
            with c2: mode = st.radio("วิธีประเมินโอกาสสำเร็จ:", ["🤖 ให้ AI คำนวณ (Simulate)", "✍️ กำหนดเอง (Manual)"], key="mode_t1")
            
            success_rate = 0.0
            if "AI" in mode:
                # จำลองให้ AI ทายใหม่โดยแก้ค่าส่งให้เป็น 0
                if 'churn_probability' in target_df.columns and model is not None:
                    old_churn = target_df['churn_probability'].mean()
                    df_sim = target_df.copy()
                    if 'freight_value' in df_sim.columns: df_sim['freight_value'] = 0
                    if 'freight_ratio' in df_sim.columns: df_sim['freight_ratio'] = 0
                    
                    X_sim = pd.DataFrame(index=df_sim.index)
                    for col in feature_names: X_sim[col] = df_sim[col] if col in df_sim.columns else 0
                    try:
                        new_probs = model.predict_proba(X_sim)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_sim)
                        uplift = max(old_churn - new_probs.mean(), 0.01) # กันติดลบ
                        success_rate = uplift
                    except: success_rate = 0.30
                else: success_rate = 0.30
                st.success(f"🤖 AI จำลองสถานการณ์ (ตั้งค่าส่ง=0): โอกาสสำเร็จ (Uplift) = **{success_rate*100:.1f}%**")
            else:
                success_rate = st.slider("โอกาสสำเร็จ (%)", 1, 100, 30, key="slider_t1") / 100.0
            
            campaigns.append({"name": "🚚 แคมเปญลดค่าส่ง", "people": n_target, "cost_head": cost_head, "success_rate": success_rate, "ltv": avg_ltv})

    # --- TAB 2: ส่งช้า ---
    with tab2:
        st.markdown("#### 🐌 แคมเปญง้อลูกค้าส่งช้า (Delay Recovery)")
        target_df = df_p3[df_p3['delay_days'] > 0] if 'delay_days' in df_p3.columns else pd.DataFrame()
        n_target = len(target_df)
        st.info(f"👥 พบลูกค้าที่ได้รับของช้ากว่ากำหนด: **{n_target:,} คน**")
        
        if n_target > 0:
            c1, c2 = st.columns(2)
            with c1: cost_head = st.number_input("งบต่อหัว (R$):", value=15, key="cost_t2")
            with c2: mode = st.radio("วิธีประเมินโอกาสสำเร็จ:", ["🤖 ให้ AI คำนวณ (Simulate)", "✍️ กำหนดเอง (Manual)"], key="mode_t2")
            
            success_rate = 0.0
            if "AI" in mode:
                if 'churn_probability' in target_df.columns and model is not None:
                    old_churn = target_df['churn_probability'].mean()
                    df_sim = target_df.copy()
                    if 'delay_days' in df_sim.columns: df_sim['delay_days'] = 0 # จำลองว่าไม่ส่งช้าแล้ว
                    
                    X_sim = pd.DataFrame(index=df_sim.index)
                    for col in feature_names: X_sim[col] = df_sim[col] if col in df_sim.columns else 0
                    try:
                        new_probs = model.predict_proba(X_sim)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_sim)
                        success_rate = max(old_churn - new_probs.mean(), 0.01)
                    except: success_rate = 0.35
                else: success_rate = 0.35
                st.success(f"🤖 AI จำลองสถานการณ์ (ตั้ง Delay=0): โอกาสสำเร็จ (Uplift) = **{success_rate*100:.1f}%**")
            else:
                success_rate = st.slider("โอกาสสำเร็จ (%)", 1, 100, 35, key="slider_t2") / 100.0
            
            campaigns.append({"name": "🐌 แคมเปญง้อคนส่งช้า", "people": n_target, "cost_head": cost_head, "success_rate": success_rate, "ltv": avg_ltv})

    # --- TAB 3: รีวิวแย่ ---
    with tab3:
        st.markdown("#### 😡 แคมเปญง้อลูกค้ารีวิวแย่ (Service Recovery)")
        target_df = df_p3[df_p3['review_score'] <= 2] if 'review_score' in df_p3.columns else pd.DataFrame()
        n_target = len(target_df)
        st.info(f"👥 พบลูกค้าที่ให้คะแนนรีวิว 1-2 ดาว: **{n_target:,} คน**")
        
        if n_target > 0:
            c1, c2 = st.columns(2)
            with c1: cost_head = st.number_input("งบต่อหัว (R$):", value=30, key="cost_t3")
            with c2: mode = st.radio("วิธีประเมินโอกาสสำเร็จ:", ["🤖 ให้ AI คำนวณ (Simulate)", "✍️ กำหนดเอง (Manual)"], key="mode_t3")
            
            success_rate = 0.0
            if "AI" in mode:
                if 'churn_probability' in target_df.columns and model is not None:
                    old_churn = target_df['churn_probability'].mean()
                    df_sim = target_df.copy()
                    if 'review_score' in df_sim.columns: df_sim['review_score'] = 5 # จำลองง้อจนได้ 5 ดาว
                    
                    X_sim = pd.DataFrame(index=df_sim.index)
                    for col in feature_names: X_sim[col] = df_sim[col] if col in df_sim.columns else 0
                    try:
                        new_probs = model.predict_proba(X_sim)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_sim)
                        success_rate = max(old_churn - new_probs.mean(), 0.01)
                    except: success_rate = 0.25
                else: success_rate = 0.25
                st.success(f"🤖 AI จำลองสถานการณ์ (ง้อจนได้ 5 ดาว): โอกาสสำเร็จ (Uplift) = **{success_rate*100:.1f}%**")
            else:
                success_rate = st.slider("โอกาสสำเร็จ (%)", 1, 100, 25, key="slider_t3") / 100.0
            
            campaigns.append({"name": "😡 แคมเปญรีวิวแย่", "people": n_target, "cost_head": cost_head, "success_rate": success_rate, "ltv": avg_ltv})

    # --- TAB 4: ซื้อแพงจ่ายเต็ม ---
    with tab4:
        st.markdown("#### 💳 แคมเปญดันยอดผ่อน (Installment Push)")
        if 'price' in df_p3.columns and 'payment_installments' in df_p3.columns:
            target_df = df_p3[(df_p3['price'] > 500) & (df_p3['payment_installments'] == 1)]
        else: target_df = pd.DataFrame()
        n_target = len(target_df)
        st.info(f"👥 พบลูกค้าที่ซื้อของแพง (> R$ 500) แต่จ่ายงวดเดียว: **{n_target:,} คน**")
        
        if n_target > 0:
            c1, c2 = st.columns(2)
            with c1: cost_head = st.number_input("งบต่อหัว (R$):", value=10, key="cost_t4")
            with c2: mode = st.radio("วิธีประเมินโอกาสสำเร็จ:", ["🤖 ให้ AI คำนวณ (Simulate)", "✍️ กำหนดเอง (Manual)"], key="mode_t4")
            
            success_rate = 0.0
            if "AI" in mode:
                if 'churn_probability' in target_df.columns and model is not None:
                    old_churn = target_df['churn_probability'].mean()
                    df_sim = target_df.copy()
                    if 'payment_installments' in df_sim.columns: df_sim['payment_installments'] = 10 # จำลองให้ผ่อน 10 เดือน
                    
                    X_sim = pd.DataFrame(index=df_sim.index)
                    for col in feature_names: X_sim[col] = df_sim[col] if col in df_sim.columns else 0
                    try:
                        new_probs = model.predict_proba(X_sim)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_sim)
                        success_rate = max(old_churn - new_probs.mean(), 0.01)
                    except: success_rate = 0.20
                else: success_rate = 0.20
                st.success(f"🤖 AI จำลองสถานการณ์ (ผ่อน 10 เดือน): โอกาสสำเร็จ (Uplift) = **{success_rate*100:.1f}%**")
            else:
                success_rate = st.slider("โอกาสสำเร็จ (%)", 1, 100, 20, key="slider_t4") / 100.0
            
            campaigns.append({"name": "💳 แคมเปญดันยอดผ่อน", "people": n_target, "cost_head": cost_head, "success_rate": success_rate, "ltv": avg_ltv})

    # ==========================================
    # 🌟 กลับมาแสดงผลใน ส่วนที่ 1 (ประมวลผลจัดสรรงบ)
    # ==========================================
    with alloc_container:
        if campaigns:
            df_camp = pd.DataFrame(campaigns)
            
            # คำนวณต้นทุนและกำไร
            df_camp['Total_Cost'] = df_camp['people'] * df_camp['cost_head']
            df_camp['Expected_Saved'] = df_camp['people'] * df_camp['success_rate']
            df_camp['Expected_Revenue'] = df_camp['Expected_Saved'] * df_camp['ltv']
            
            # ป้องกัน Error หาร 0
            df_camp['ROI_Percent'] = df_camp.apply(
                lambda row: ((row['Expected_Revenue'] - row['Total_Cost']) / row['Total_Cost'] * 100) if row['Total_Cost'] > 0 else 0, 
                axis=1
            )
            
            # เรียงลำดับจาก คุ้มสุดไปน้อยสุด (Greedy Algorithm)
            df_camp = df_camp.sort_values('ROI_Percent', ascending=False).reset_index(drop=True)
            
            allocated_budget = 0
            results = []
            total_expected_revenue = 0
            total_saved_people = 0
            
            # ระบบเทงบลงตะกร้า
            for _, row in df_camp.iterrows():
                budget_left = total_budget - allocated_budget
                if budget_left <= 0:
                    results.append({"แคมเปญ": row['name'], "สถานะ": "❌ งบไม่พอ", "เป้าหมาย (คน)": 0, "งบที่ใช้ (R$)": 0, "กำไรคาดหวัง (R$)": 0, "ROI (%)": row['ROI_Percent']})
                    continue
                    
                if budget_left >= row['Total_Cost']:
                    spend = row['Total_Cost']
                    people_covered = row['people']
                else:
                    spend = budget_left
                    people_covered = int(spend / row['cost_head']) if row['cost_head'] > 0 else 0
                    
                allocated_budget += spend
                revenue_back = (people_covered * row['success_rate']) * row['ltv']
                profit = revenue_back - spend
                
                total_expected_revenue += revenue_back
                total_saved_people += (people_covered * row['success_rate'])
                
                status = "✅ ได้งบเต็ม" if spend == row['Total_Cost'] else "⚠️ ได้งบบางส่วน"
                results.append({
                    "แคมเปญ": row['name'], 
                    "สถานะ": status, 
                    "เป้าหมาย (คน)": people_covered, 
                    "งบที่ใช้ (R$)": spend, 
                    "กำไรคาดหวัง (R$)": profit, 
                    "ROI (%)": row['ROI_Percent']
                })
            
            # แสดง Dashboard สรุปด้านบน
            m1, m2, m3 = st.columns(3)
            m1.metric("💰 งบที่จัดสรรได้", f"R$ {allocated_budget:,.0f}", f"จาก R$ {total_budget:,.0f}")
            m2.metric("👥 คาดว่าจะดึงลูกค้ากลับมาได้", f"{int(total_saved_people):,} คน")
            total_profit = total_expected_revenue - allocated_budget
            m3.metric("✨ กำไรคาดหวังรวมสุทธิ", f"R$ {total_profit:,.0f}", f"+{(total_profit/allocated_budget)*100:.1f}% ROI" if allocated_budget>0 else "0%")
            
            # ตารางสรุปการจัดสรรงบ
            res_df = pd.DataFrame(results)
            st.dataframe(res_df.style.format({
                "เป้าหมาย (คน)": "{:,.0f}",
                "งบที่ใช้ (R$)": "{:,.0f}",
                "กำไรคาดหวัง (R$)": "{:,.0f}",
                "ROI (%)": "{:.1f}%"
            }), use_container_width=True)
            
        else:
            st.warning("ไม่มีข้อมูลลูกค้ากลุ่มเสี่ยงในหมวดหมู่นี้ให้จัดสรรงบประมาณ")

# ==========================================
# PAGE 4: 🚛 Logistics Insights
# ==========================================
elif page == "4. 🚛 Logistics Insights":
    import pydeck as pdk

    st.title("🚛 Logistics Insights")
    st.markdown("วิเคราะห์ปัญหาขนส่งและการเงิน รายรัฐ (Map) และรายเมือง (Table)")

    if 'customer_state' not in df.columns:
        st.error("❌ ไม่พบข้อมูลรัฐ (customer_state)")
        st.stop()

    all_cats = sorted([x for x in df['product_category_name'].unique() if pd.notna(x)]) if 'product_category_name' in df.columns else []
    sel_cats_p4 = st.multiselect("📦 กรองหมวดสินค้า:", all_cats, key="p4_cat_filter")
    df_logistics = df[df['product_category_name'].isin(sel_cats_p4)].copy() if sel_cats_p4 else df.copy()

    brazil_states_coords = {
        'AC': [-9.02,-70.81], 'AL': [-9.57,-36.78], 'AM': [-3.41,-65.85],
        'AP': [0.90,-52.00],  'BA': [-12.58,-41.70], 'CE': [-5.49,-39.32],
        'DF': [-15.79,-47.88],'ES': [-19.18,-40.30], 'GO': [-15.82,-49.84],
        'MA': [-5.19,-45.16], 'MG': [-19.81,-43.95], 'MS': [-20.77,-54.78],
        'MT': [-12.96,-56.92],'PA': [-6.31,-52.46],  'PB': [-7.24,-36.78],
        'PE': [-8.81,-36.95], 'PI': [-7.71,-42.72],  'PR': [-25.25,-52.02],
        'RJ': [-22.90,-43.17],'RN': [-5.40,-36.95],  'RO': [-11.50,-63.58],
        'RR': [2.73,-62.07],  'RS': [-30.03,-51.22], 'SC': [-27.24,-50.21],
        'SE': [-10.57,-37.38],'SP': [-23.55,-46.63], 'TO': [-10.17,-48.33]
    }

    state_metrics = df_logistics.groupby('customer_state').agg(
        payment_value=('payment_value', 'sum'),
        delivery_days=('delivery_days', 'mean'),
        delay_days=('delay_days', lambda x: (x > 0).sum()),
        churn_probability=('churn_probability', 'mean'),
        total_orders=('order_purchase_timestamp', 'count')
    ).reset_index()

    state_metrics['lat'] = state_metrics['customer_state'].map(lambda x: brazil_states_coords.get(x, [0,0])[0])
    state_metrics['lon'] = state_metrics['customer_state'].map(lambda x: brazil_states_coords.get(x, [0,0])[1])

    st.markdown("---")
    col_sel, col_kpi1, col_kpi2, col_kpi3 = st.columns([1.5, 1, 1, 1])
    with col_sel:
        zoom_state = st.selectbox("🔍 โฟกัสรัฐ (Zoom):", ["All (ภาพรวมประเทศ)"] + sorted(state_metrics['customer_state'].unique()))

    if zoom_state != "All (ภาพรวมประเทศ)":
        display_data = state_metrics[state_metrics['customer_state'] == zoom_state]
        if not display_data.empty:
            view_lat, view_lon, view_zoom = display_data['lat'].values[0], display_data['lon'].values[0], 6
        else:
            view_lat, view_lon, view_zoom = -14.2350, -51.9253, 3.5
    else:
        display_data = state_metrics
        view_lat, view_lon, view_zoom = -14.2350, -51.9253, 3.5

    with col_kpi1: st.metric("💰 เงินหมุนเวียน", f"R$ {display_data['payment_value'].sum():,.0f}")
    with col_kpi2: st.metric("🚚 ส่งเฉลี่ย",     f"{display_data['delivery_days'].mean():.1f} วัน")
    with col_kpi3: st.metric("⚠️ ส่งช้า (Late)", f"{display_data['delay_days'].sum():,} ครั้ง", delta_color="inverse")

    c_map, c_state_table = st.columns([2, 1])
    with c_map:
        st.subheader(f"🗺️ แผนที่ ({zoom_state})")
        state_metrics['color'] = state_metrics['churn_probability'].apply(
            lambda x: [231,76,60,200] if x > 0.8 else ([241,196,15,200] if x > 0.5 else [46,204,113,200])
        )
        max_val = state_metrics['payment_value'].max()
        state_metrics['radius'] = (state_metrics['payment_value'] / max_val * 400000) if max_val > 0 else 10000

        layer = pdk.Layer(
            "ScatterplotLayer", state_metrics,
            get_position='[lon, lat]', get_color='color', get_radius='radius',
            pickable=True, opacity=0.8, stroked=True, filled=True,
            radius_min_pixels=5, radius_max_pixels=60,
        )
        tooltip = {
            "html": "<b>รัฐ: {customer_state}</b><br/>💰 ยอดเงิน: R$ {payment_value}<br/>🚚 ส่งเฉลี่ย: {delivery_days} วัน<br/>⚠️ ส่งช้า: {delay_days} ครั้ง<br/>📉 ความเสี่ยง: {churn_probability}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=view_zoom, pitch=20),
            tooltip=tooltip, map_provider='carto', map_style='light'
        )
        st.pydeck_chart(r)

    with c_state_table:
        st.subheader("🚨 รัฐที่มีปัญหา (Top Issues)")
        sort_mode = st.radio("เรียงตาม:", ["ส่งช้าเยอะสุด (Late Count)", "ความเสี่ยงสูงสุด (Risk)"], horizontal=True)
        top_issues = state_metrics.sort_values(
            'delay_days' if "ส่งช้า" in sort_mode else 'churn_probability', ascending=False
        ).head(10)
        st.dataframe(
            top_issues[['customer_state', 'payment_value', 'delivery_days', 'delay_days', 'churn_probability']],
            column_config={
                "customer_state": "รัฐ",
                "payment_value": st.column_config.NumberColumn("เงินหมุนเวียน", format="R$ %.0f"),
                "delivery_days": st.column_config.NumberColumn("ส่ง (วัน)", format="%.1f"),
                "delay_days": st.column_config.NumberColumn("ช้า (ครั้ง)"),
                "churn_probability": st.column_config.ProgressColumn("ความเสี่ยง", format="%.2f", min_value=0, max_value=1)
            },
            hide_index=True, use_container_width=True
        )

    st.markdown("---")
    st.subheader("🏙️ เจาะลึกรายเมือง (City Drill-down)")
    if 'customer_city' in df_logistics.columns:
        city_metrics = df_logistics.groupby(['customer_state', 'customer_city']).agg(
            customer_count=('customer_unique_id', 'count'),
            payment_value=('payment_value', 'sum'),
            delivery_days=('delivery_days', 'mean'),
            delay_days=('delay_days', lambda x: (x > 0).sum()),
            churn_probability=('churn_probability', 'mean')
        ).reset_index()
        city_metrics = city_metrics[city_metrics['customer_count'] >= 2]

        if zoom_state != "All (ภาพรวมประเทศ)":
            city_display = city_metrics[city_metrics['customer_state'] == zoom_state]
            st.info(f"📍 แสดงรายชื่อเมืองในรัฐ: **{zoom_state}**")
        else:
            city_display = city_metrics
            st.info("📍 แสดงรายชื่อเมืองทั่วประเทศ (Top 50 ที่มีปัญหา)")

        st.dataframe(
            city_display.sort_values('delay_days', ascending=False).head(50),
            column_config={
                "customer_state": "รัฐ", "customer_city": "เมือง",
                "customer_count": st.column_config.NumberColumn("จำนวนลูกค้า"),
                "payment_value": st.column_config.NumberColumn("ยอดเงินรวม", format="R$ %.0f"),
                "delivery_days": st.column_config.NumberColumn("ส่งเฉลี่ย (วัน)", format="%.1f"),
                "delay_days": st.column_config.NumberColumn("ส่งช้า (ครั้ง)"),
                "churn_probability": st.column_config.ProgressColumn("ความเสี่ยง (Avg)", format="%.2f", min_value=0, max_value=1)
            },
            hide_index=True, use_container_width=True
        )
    else:
        st.warning("⚠️ ไม่พบข้อมูลเมือง (customer_city)")

# ==========================================
# PAGE 5: 🏪 Seller Audit
# ==========================================
elif page == "5. 🏪 Seller Audit":
    st.title("🏪 Seller Audit & Performance")
    st.markdown("ตรวจสอบประสิทธิภาพและความเสี่ยงรายร้านค้า")

    if 'seller_id' not in df.columns:
        st.error("❌ ไม่พบข้อมูลผู้ขาย (seller_id)")
        st.stop()

    all_cats = sorted([x for x in df['product_category_name'].unique() if pd.notna(x)]) if 'product_category_name' in df.columns else []
    sel_cats_p5 = st.multiselect("📦 กรองหมวดสินค้า:", all_cats, key="p5_cat_filter")
    df_seller_view = df[df['product_category_name'].isin(sel_cats_p5)].copy() if sel_cats_p5 else df.copy()

    agg_dict = {
        'order_purchase_timestamp': 'count',
        'payment_value': 'sum',
        'churn_probability': 'mean',
        'delivery_days': 'mean',
    }
    if 'review_score' in df_seller_view.columns:
        agg_dict['review_score'] = 'mean'

    seller_stats = df_seller_view.groupby('seller_id').agg(agg_dict).reset_index()
    seller_stats = seller_stats.rename(columns={'order_purchase_timestamp': 'total_orders'})
    if 'review_score' not in seller_stats.columns:
        seller_stats['review_score'] = np.nan
    seller_stats = seller_stats[seller_stats['total_orders'] >= 3]

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏪 จำนวนร้านค้า", f"{len(seller_stats):,} ร้าน")
    c2.metric("💸 ยอดขายรวม",    f"R$ {seller_stats['payment_value'].sum():,.0f}")
    c3.metric("⭐ รีวิวเฉลี่ย",  f"{seller_stats['review_score'].mean():.2f}/5.0" if seller_stats['review_score'].notna().any() else "N/A")
    c4.metric("🚚 ส่งเฉลี่ย",   f"{seller_stats['delivery_days'].mean():.1f} วัน")

    st.markdown("---")
    col_sort, col_display = st.columns([1, 3])
    with col_sort:
        st.subheader("⚙️ จัดเรียงข้อมูล (Sort By)")
        sort_mode = st.radio("เลือกเกณฑ์การเรียง:", [
            "🚨 ความเสี่ยงสูงสุด (Highest Risk)",
            "🐢 ส่งของช้าสุด (Slowest Delivery)",
            "⭐ คะแนนต่ำสุด (Lowest Rating)",
            "💸 ยอดขายสูงสุด (Top Revenue)",
            "📦 ขายเยอะสุด (Top Volume)"
        ])
        if "ความเสี่ยง" in sort_mode:
            sorted_df = seller_stats.sort_values('churn_probability', ascending=False)
        elif "ส่งของช้า" in sort_mode:
            sorted_df = seller_stats.sort_values('delivery_days', ascending=False)
        elif "คะแนนต่ำ" in sort_mode:
            sorted_df = seller_stats.sort_values('review_score', ascending=True)
        elif "ยอดขาย" in sort_mode:
            sorted_df = seller_stats.sort_values('payment_value', ascending=False)
        else:
            sorted_df = seller_stats.sort_values('total_orders', ascending=False)

    with col_display:
        st.subheader(f"📋 รายชื่อร้านค้า ({sort_mode})")
        st.dataframe(
            sorted_df,
            column_config={
                "seller_id": "รหัสร้านค้า",
                "total_orders": st.column_config.NumberColumn("จำนวนออเดอร์"),
                "payment_value": st.column_config.NumberColumn("ยอดขายรวม", format="R$ %.0f"),
                "delivery_days": st.column_config.NumberColumn("ส่งเฉลี่ย (วัน)", format="%.1f วัน"),
                "review_score": st.column_config.NumberColumn("รีวิว (ดาว)", format="%.1f ⭐"),
                "churn_probability": st.column_config.ProgressColumn("ความเสี่ยง Churn", format="%.2f", min_value=0, max_value=1)
            },
            hide_index=True, use_container_width=True, height=600
        )

# ==========================================
# PAGE 6: 🔄 Buying Cycle Analysis
# ==========================================
elif page == "6. 🔄 Buying Cycle Analysis":
    st.title("🔄 Buying Cycle Analysis")
    st.markdown("วิเคราะห์รอบการซื้อ: **สินค้าหมวดนี้...ลูกค้ากลับมาซื้อซ้ำเร็วแค่ไหน?**")

    if 'cat_median_days' not in df.columns:
        st.error("❌ ไม่พบข้อมูลรอบการซื้อ (cat_median_days)")
        st.stop()

    all_cats = sorted([x for x in df['product_category_name'].unique() if pd.notna(x)]) if 'product_category_name' in df.columns else []
    sel_cats_p6 = st.multiselect("📦 เลือกหมวดสินค้า (เปรียบเทียบกับภาพรวม):", all_cats, key="p6_cat_filter")

    if sel_cats_p6:
        df_cycle     = df[df['product_category_name'].isin(sel_cats_p6)].copy()
        filter_label = f"หมวด: {', '.join(sel_cats_p6[:3])}{'...' if len(sel_cats_p6)>3 else ''}"
    else:
        df_cycle     = df.copy()
        filter_label = "ภาพรวมทุกหมวด"

    global_avg_cycle = df['cat_median_days'].mean()
    global_avg_late  = df['lateness_score'].mean() if 'lateness_score' in df.columns else 0
    curr_avg_cycle   = df_cycle['cat_median_days'].mean()
    curr_avg_late    = df_cycle['lateness_score'].mean() if 'lateness_score' in df_cycle.columns else 0
    fast_repeaters   = len(df_cycle[df_cycle['cat_median_days'] <= 30])

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"⏱️ รอบซื้อเฉลี่ย ({filter_label})", f"{curr_avg_cycle:.0f} วัน",
              delta=f"{curr_avg_cycle - global_avg_cycle:.0f} วัน (เทียบภาพรวม)", delta_color="inverse")
    m2.metric(f"🐢 ความล่าช้า ({filter_label})", f"{curr_avg_late:.2f} เท่า",
              delta=f"{curr_avg_late - global_avg_late:.2f} (เทียบภาพรวม)", delta_color="inverse")
    m3.metric("📅 ซื้อซ้ำใน 30 วัน", f"{fast_repeaters:,} คน")

    st.markdown("---")
    st.subheader("📊 เปรียบเทียบพฤติกรรมการซื้อซ้ำ (Repurchase Distribution)")
    col_focus, col_bench = st.columns(2)

    with col_focus:
        st.info(f"📍 **{filter_label}** (กลุ่มที่คุณเลือก)")
        hist_focus = alt.Chart(df_cycle).mark_bar().encode(
            x=alt.X('cat_median_days', bin=alt.Bin(maxbins=50), title='ระยะเวลาซื้อซ้ำ (วัน)'),
            y=alt.Y('count()', title='จำนวนลูกค้า'),
            color=alt.value('#3498db'),
            tooltip=['count()', alt.Tooltip('cat_median_days', bin=True, title='ช่วงวัน')]
        ).properties(height=300, title=f"การกระจายตัวของ {filter_label}")
        st.altair_chart(hist_focus, use_container_width=True)

    with col_bench:
        st.warning("🏢 **ภาพรวมทั้งบริษัท** (Benchmark)")
        hist_all = alt.Chart(df).mark_bar().encode(
            x=alt.X('cat_median_days', bin=alt.Bin(maxbins=50), title='ระยะเวลาซื้อซ้ำ (วัน)'),
            y=alt.Y('count()', title='จำนวนลูกค้า'),
            color=alt.value('#95a5a6'),
            tooltip=['count()', alt.Tooltip('cat_median_days', bin=True, title='ช่วงวัน')]
        ).properties(height=300, title="Benchmark: ภาพรวมสินค้าทั้งหมด")
        st.altair_chart(hist_all, use_container_width=True)

    st.markdown("---")
    st.subheader(f"📋 รายละเอียดรายหมวดสินค้า ({filter_label})")
    summ = df_cycle.groupby('product_category_name').agg(
        Total_Customers=('customer_unique_id', 'count'),
        Avg_Cycle_Days=('cat_median_days', 'mean'),
        Avg_Late_Score=('lateness_score', 'mean'),
        Churn_Risk=('churn_probability', 'mean')
    ).reset_index()
    st.dataframe(
        summ.sort_values('Avg_Cycle_Days'),
        column_config={
            "product_category_name": "หมวดสินค้า",
            "Total_Customers": st.column_config.NumberColumn("ลูกค้าทั้งหมด", format="%d คน"),
            "Avg_Cycle_Days": st.column_config.NumberColumn("รอบซื้อเฉลี่ย", format="%.0f วัน"),
            "Avg_Late_Score": st.column_config.NumberColumn("ความล่าช้า", format="%.2f เท่า"),
            "Churn_Risk": st.column_config.ProgressColumn("ความเสี่ยง Churn", format="%.2f", min_value=0, max_value=1)
        },
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    st.subheader("📅 Seasonal Patterns: สินค้าขายดีเดือนไหน?")
    st.caption(f"แสดงข้อมูลเจาะจงของ: **{filter_label}**")

    if 'order_purchase_timestamp' in df_cycle.columns:
        season_df = df_cycle.copy()
        season_df['month_num']  = season_df['order_purchase_timestamp'].dt.month
        season_df['month_name'] = season_df['order_purchase_timestamp'].dt.strftime('%b')
        heatmap_data = season_df.groupby(['product_category_name', 'month_num', 'month_name']).size().reset_index(name='sales_volume')
        top_cats     = season_df['product_category_name'].value_counts().head(15).index.tolist()
        heatmap_data = heatmap_data[heatmap_data['product_category_name'].isin(top_cats)]

        if not heatmap_data.empty:
            heatmap = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X('month_name', sort=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], title='เดือน'),
                y=alt.Y('product_category_name', title='หมวดสินค้า'),
                color=alt.Color('sales_volume', scale=alt.Scale(scheme='orangered'), title='ยอดขาย'),
                tooltip=['product_category_name', 'month_name', alt.Tooltip('sales_volume', format=',')]
            ).properties(height=500)
            st.altair_chart(heatmap, use_container_width=True)
            st.info("💡 **Tip:** สีส้มเข้ม = ช่วง High Season ที่ต้องเตรียมสต็อกสินค้าให้พร้อม")
        else:
            st.info("⚠️ ไม่มีข้อมูลเพียงพอสำหรับสร้าง Heatmap ในหมวดที่เลือก")
    else:
        st.warning("⚠️ ไม่พบข้อมูลวันที่ (order_purchase_timestamp)")
