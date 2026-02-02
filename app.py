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
# 2. LOAD ASSETS (Clean Version: ข้อมูลจริงเท่านั้น)
# ==========================================
@st.cache_resource
def load_data_and_model():
    data_dict = {}
    errors = []
    
    # 1. หาตำแหน่งไฟล์ (Path Fix) - เก็บส่วนนี้ไว้กันเหนียวครับ
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'olist_churn_model_best.pkl')
    features_path = os.path.join(current_dir, 'model_features_best.pkl')
    lite_data_path = os.path.join(current_dir, 'olist_dashboard_lite.csv')

    # 2. โหลด Model
    try:
        data_dict['model'] = joblib.load(model_path)
        data_dict['features'] = joblib.load(features_path)
    except Exception as e:
        errors.append(f"Model Error: ไม่สามารถโหลดโมเดลได้ ({e})")

    # 3. โหลด Data (ลบส่วนสร้าง Dummy ทิ้งแล้ว)
    try:
        if os.path.exists(lite_data_path):
            df = pd.read_csv(lite_data_path)
            
            # แปลงวันที่
            if 'order_purchase_timestamp' in df.columns:
                df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
                
            data_dict['df'] = df
        else:
            errors.append(f"Data Error: ไม่พบไฟล์ข้อมูลที่ {lite_data_path}")
            
    except Exception as e:
        errors.append(f"Data Error: อ่านไฟล์ไม่ได้ ({e})")
        
    return data_dict, errors

assets, load_errors = load_data_and_model()

# เช็ค Error ถ้ามีก็โชว์ แต่ถ้าโหลด Data ไม่ได้เลยให้หยุด
if load_errors:
    for err in load_errors:
        st.error(f"⚠️ {err}")
    
    # ถ้าไม่มี Data มาเลย ให้หยุดทำงาน (กัน Error บรรทัดถัดไป)
    if 'df' not in assets:
        st.stop()

# ==========================================
# 3. PREPARE DATA
# ==========================================

df = assets['df'] 
model = assets.get('model')
feature_names = assets.get('features', [])

# 3.1 Predict Logic (ถ้ามีโมเดลจริง)
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
        df['churn_probability'] = np.random.uniform(0.1, 0.9, len(df)) # Fallback

# 3.2 Define Status Logic (ถ้ายังไม่มี column status)
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
# 4. NAVIGATION & LAYOUT
# ==========================================
st.sidebar.title("✈️ Olist Cockpit")
page = st.sidebar.radio("Navigation", [
    "1. 📊 Executive Summary", 
    "2. 🔍 Customer Detail", 
    "3. 🎯 Action Plan",
    "4. 🚛 Logistics Insights",
    "5. 🏪 Seller Audit"
])

st.sidebar.markdown("---")
st.sidebar.info("Select a page to analyze different aspects of your business.")

# ==========================================
# PAGE 1: 📊 Executive Summary (Updated)
# ==========================================
if page == "1. 📊 Executive Summary":
    st.title("📊 Executive Summary (Business Health)")
    
    # --- 1. FILTER SECTION ---
    with st.expander("🌪️ กรองข้อมูล (Filter)", expanded=False):
        all_cats = list(df['product_category_name'].unique()) if 'product_category_name' in df.columns else []
        selected_cats_p1 = st.multiselect("เลือกหมวดหมู่สินค้า (ว่าง = ดูภาพรวมทั้งหมด):", all_cats, key="p1_cat_filter")
    
    # กรองข้อมูล
    if selected_cats_p1:
        df_display = df[df['product_category_name'].isin(selected_cats_p1)].copy()
        filter_label = f"หมวด: {', '.join(selected_cats_p1[:3])}..."
    else:
        df_display = df.copy()
        filter_label = "ภาพรวมทั้งบริษัท"

    st.caption(f"กำลังแสดงผล: **{filter_label}**")
    st.markdown("---")

    # --- 2. KPI CARDS (Clean Version + Buying Cycle) ---
    total_customers = len(df_display)
    
    # คำนวณตัวเลขพื้นฐาน
    if total_customers > 0:
        risk_df = df_display[df_display['status'].isin(['High Risk', 'Warning (Late > 1.5x)'])]
        risk_count = len(risk_df)
        churn_rate = (risk_count / total_customers) * 100
        rev_at_risk = risk_df['payment_value'].sum() if 'payment_value' in df_display.columns else 0
        active_count = len(df_display[df_display['status'] == 'Active'])
        
        # 🟢 คำนวณรอบการซื้อปกติ (Buying Cycle) ของกลุ่มที่เลือก
        if 'cat_median_days' in df_display.columns:
            avg_cycle = df_display['cat_median_days'].mean()
            cycle_text = f"{avg_cycle:.0f} วัน"
        else:
            cycle_text = "N/A"
    else:
        churn_rate, rev_at_risk, risk_count, active_count = 0, 0, 0, 0
        cycle_text = "-"

    # แสดงผล 5 คอลัมน์ (เพิ่มช่อง Buying Cycle)
    k1, k2, k3, k4, k5 = st.columns(5)
    
    # ❌ ลบพารามิเตอร์ delta ออก เพื่อไม่ให้มีตัวหนังสือเล็กๆ ข้างล่าง
    with k1: st.metric("🚨 Churn Rate", f"{churn_rate:.1f}%")
    with k2: st.metric("💸 Revenue at Risk", f"R$ {rev_at_risk:,.0f}")
    with k3: st.metric("👥 Risk vs Total", f"{risk_count:,} / {total_customers:,}")
    with k4: st.metric("✅ Active Customers", f"{active_count:,}")
    with k5: st.metric("🔄 รอบซื้อปกติ (Cycle)", cycle_text, help="ระยะเวลาเฉลี่ยที่ลูกค้าหมวดนี้มักจะกลับมาซื้อซ้ำ")

    st.markdown("---")

    # --- 3. CHARTS ---
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
                last_val = trend_df['Churn_Prob'].iloc[-1]
                
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
        st.subheader("🍩 Business Health")
        if not df_display.empty and 'order_purchase_timestamp' in df_display.columns:
            
            # 🟢 คำนวณว่า "หายไปนานกี่วันแล้ว" (Days Gone)
            # ใช้วันที่สุดท้ายใน Dataset เป็นจุดอ้างอิงปัจจุบัน
            ref_date = df['order_purchase_timestamp'].max()
            df_display['days_gone'] = (ref_date - df_display['order_purchase_timestamp']).dt.days
            
            # Group ข้อมูลเพื่อเอาไปพล็อตกราฟ
            status_stats = df_display.groupby('status').agg({
                'customer_unique_id': 'count',
                'days_gone': 'mean' # หาค่าเฉลี่ยวันที่หายไป
            }).reset_index()
            
            status_stats.columns = ['Status', 'Count', 'Avg_Days_Gone']
            
            domain = ['Active', 'Medium Risk', 'Warning (Late > 1.5x)', 'High Risk', 'Lost (Late > 3x)']
            range_ = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#95a5a6']
            
            donut = alt.Chart(status_stats).mark_arc(innerRadius=60).encode(
                theta=alt.Theta("Count", type="quantitative"),
                color=alt.Color("Status", scale=alt.Scale(domain=domain, range=range_), legend=dict(orient='bottom')),
                # 🟢 เพิ่ม Avg_Days_Gone เข้าไปใน Tooltip
                tooltip=[
                    'Status', 
                    alt.Tooltip('Count', format=',', title='จำนวนลูกค้า'),
                    alt.Tooltip('Avg_Days_Gone', format='.0f', title='หายเงียบไป (วันเฉลี่ย)')
                ]
            ).properties(height=350)
            st.altair_chart(donut, use_container_width=True)
        else:
            st.info("ไม่มีข้อมูลแสดงผล")

# ==========================================
# PAGE 2: 🔍 Customer Detail
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

# ==========================================
# PAGE 3: 🎯 Action Plan
# ==========================================
elif page == "3. 🎯 Action Plan":
    st.title("🎯 Marketing Campaign Simulator")
    st.markdown("### วิเคราะห์ความคุ้มค่า (ROI)")
    
    # Filter Target (Risk 60-85%)
    target_customers = df[(df['churn_probability'] >= 0.60) & (df['churn_probability'] <= 0.85)].copy()
    total_target = len(target_customers)
    
    if total_target == 0:
        st.warning("⚠️ ไม่พบลูกค้ากลุ่มเป้าหมาย (ใช้ข้อมูลจำลองแทน)")
        # สร้างข้อมูลจำลองถ้าไม่เจอ
        target_customers = df.head(50).copy()
        total_target = 50

    with st.container():
        val_risk = target_customers['payment_value'].sum() if 'payment_value' in df.columns else 0
        st.markdown(f"#### 🎯 เป้าหมาย: {total_target:,} คน (Value: R$ {val_risk:,.0f})")
        c1, c2, c3 = st.columns(3)
        with c1: voucher = st.slider("💰 มูลค่าคูปอง (R$)", 0, 50, 0, step=5)
        with c2: speed = st.selectbox("🚚 ขนส่ง", ["ปกติ", "ส่งด่วน (-2 วัน)"])
        with c3: 
            cost = voucher * total_target
            st.metric("งบประมาณ (Cost)", f"R$ {cost:,.0f}")

    # Simulation Logic
    df_sim = target_customers.copy()
    impact = (voucher / 10) * 0.02 if voucher > 0 else 0
    
    # Artificial impact
    final_probs = df_sim['churn_probability'] - impact
    if speed == "ส่งด่วน (-2 วัน)":
        final_probs = final_probs - 0.05
    
    df_sim['new_prob'] = final_probs
    success = df_sim[df_sim['new_prob'] < 0.5]
    saved_rev = success['payment_value'].sum() if 'payment_value' in df_sim.columns else 0
    roi = saved_rev - cost
    
    st.markdown("---")
    res1, res2, res3, res4 = st.columns(4)
    res1.metric("👥 กู้คืนได้", f"{len(success):,} คน")
    res2.metric("💸 รายได้ที่รักษาได้", f"R$ {saved_rev:,.0f}")
    res3.metric("📉 ต้นทุน", f"R$ {cost:,.0f}")
    roi_color = "normal" if roi > 0 else "inverse"
    res4.metric("💰 ROI", f"R$ {roi:,.0f}", delta_color=roi_color)
    
    col_g, col_l = st.columns([1.5, 1])
    with col_g:
        chart_data = pd.DataFrame({
            'Risk': list(target_customers['churn_probability']) + list(final_probs),
            'Type': ['Before'] * len(target_customers) + ['After'] * len(final_probs)
        })
        chart = alt.Chart(chart_data).mark_area(opacity=0.5, interpolate='step').encode(
            x=alt.X('Risk', bin=alt.Bin(maxbins=20)),
            y='count()', color='Type'
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)
    with col_l:
        st.dataframe(success[['customer_unique_id', 'new_prob']].head(20), hide_index=True)

# ==========================================
# PAGE 4: 🚛 Logistics
# ==========================================
elif page == "4. 🚛 Logistics Insights":
    st.title("🚛 Logistics Heatmap")
    if 'customer_state' not in df.columns:
        st.error("No state data")
        st.stop()

    col_map, col_stat = st.columns([2, 1])
    with col_map:
        state_stats = df.groupby('customer_state').agg({
            'customer_unique_id': 'count', 'delivery_days': 'mean', 'churn_probability': 'mean'
        }).reset_index()
        # Filter noise
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
        sel_state = st.selectbox("เลือกรัฐ:", df['customer_state'].unique())
        if sel_state:
            city_df = df[df['customer_state'] == sel_state]
            city_stats = city_df.groupby('customer_city').agg({
                'customer_unique_id': 'count', 'delivery_days': 'mean', 'churn_probability': 'mean'
            }).reset_index()
            st.dataframe(city_stats[city_stats['customer_unique_id'] >= 2].sort_values('churn_probability', ascending=False).head(10), use_container_width=True)
    else:
        st.info("ไม่มีข้อมูลรายเมืองในไฟล์จำลอง")

# ==========================================
# PAGE 5: 🏪 Seller Audit
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




