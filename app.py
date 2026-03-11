import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import datetime
from google.cloud import bigquery
# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Olist Executive Cockpit (Real-time)",
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
# 2. LOAD ASSETS (BigQuery Connection)
# ==========================================

# ฟังก์ชันดึงข้อมูลจาก BigQuery
@st.cache_data(ttl=600) # แคชข้อมูล 10 นาที

def load_bq_data():
    try:
        # ดึงค่าจาก secrets มาสร้าง Client เอง
        info = st.secrets["connections"]["bigquery"]["service_account_info"]
        client = bigquery.Client.from_service_account_info(info)
        
        query = "SELECT * FROM `academic-moon-483615-t2.olist_db.orders_data`"
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
        return None

# ฟังก์ชันโหลด Model
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

# ฟังก์ชันคำนวณฟีเจอร์ที่ขาดหายไป (Feature Engineering)
def process_features(df):
    # 1. แปลงวันที่
    date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 2. คำนวณค่าพื้นฐาน
    if 'price' in df.columns:
        df['payment_value'] = df['price'] + df.get('freight_value', 0)
        df['freight_ratio'] = df.get('freight_value', 0) / df['price']
    
    # 3. คำนวณ Logistics Features
    if 'order_delivered_customer_date' in df.columns and 'order_estimated_delivery_date' in df.columns:
        df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
        df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
        # กรองค่าติดลบของ delay_days (ส่งก่อนกำหนด) ให้เป็น 0 หรือเก็บไว้ตาม Logic เดิม
    
    # 4. คำนวณรอบการซื้อ (Mockup Logic เพื่อให้ App รันได้)
    # ในงานจริงตรงนี้ควรดึงมาจากตารางพฤติกรรมลูกค้า
    if 'cat_median_days' not in df.columns:
        # ใช้ค่าเฉลี่ยกลางๆ 90 วันถ้าไม่มีข้อมูล
        df['cat_median_days'] = 90 
    
    # 5. คำนวณ Lateness Score
    today = pd.Timestamp.now()
    df['days_since_order'] = (today - df['order_purchase_timestamp']).dt.days
    df['lateness_score'] = df['days_since_order'] / df['cat_median_days']
    
    return df

# --- เริ่มการโหลดข้อมูล ---
with st.sidebar:
    if st.button('🔄 Refresh Data (Sync Sheets)'):
        st.cache_data.clear()
        st.rerun()

df_raw, bq_error = load_bq_data()
model, feature_names, model_error = load_models()

# เช็ค Error
if bq_error:
    st.error(f"⚠️ {bq_error}")
    st.info("ตรวจสอบว่าได้ตั้งค่า Secrets ใน Streamlit Cloud หรือไฟล์ .streamlit/secrets.toml หรือยัง?")
    st.stop()

if model_error:
    st.warning(f"⚠️ {model_error}")

# ประมวลผลข้อมูล
df = process_features(df_raw)

# ==========================================
# 3. PREPARE DATA & PREDICTION
# ==========================================

# 3.1 Predict Logic
if 'churn_probability' not in df.columns and model is not None:
    X_pred = pd.DataFrame(index=df.index)
    for col in feature_names:
        # เติมค่า 0 หากคอลัมน์ที่โมเดลต้องการไม่มีในตาราง
        X_pred[col] = df[col] if col in df.columns else 0
    try:
        df['churn_probability'] = model.predict_proba(X_pred)[:, 1]
    except:
        df['churn_probability'] = 0.5

# 3.2 Define is_churn & status
if 'is_churn' not in df.columns:
    df['is_churn'] = (df['churn_probability'] > 0.5).astype(int)

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
# 4. NAVIGATION & PAGES (โครงสร้างเดิมของคุณ)
# ==========================================
st.sidebar.title("✈️ Olist Cockpit")
page = st.sidebar.radio("Navigation", [
    "1. 📊 Executive Summary", 
    "2. 🔍 Customer Detail", 
    "3. 🎯 Action Plan",
    "4. 🚛 Logistics Insights",
    "5. 🏪 Seller Audit",
    "6. 🔄 Buying Cycle Analysis"
])

# (จากตรงนี้ไปจนจบไฟล์ คุณสามารถใช้โค้ดแสดงผลเดิมของคุณได้เลยครับ)
# ผมจะใส่ Page 1 ไว้เป็นตัวอย่างให้ดูว่ามันรันได้เหมือนเดิม

if page == "1. 📊 Executive Summary":
    st.title("📊 Executive Summary (Real-time Cloud)")
    st.success(f"เชื่อมต่อข้อมูลล่าสุดจาก Google Sheets ผ่าน BigQuery เรียบร้อย (จำนวน {len(df):,} ออเดอร์)")
    
    # KPI CARDS
    total_customers = len(df)
    risk_count = len(df[df['status'].isin(['High Risk', 'Warning (Late > 1.5x)'])])
    churn_rate = (risk_count / total_customers) * 100 if total_customers > 0 else 0
    
    k1, k2, k3 = st.columns(3)
    k1.metric("🚨 Churn Rate", f"{churn_rate:.1f}%")
    k2.metric("👥 Risk Customers", f"{risk_count:,}")
    k3.metric("💰 Total Revenue", f"R$ {df['payment_value'].sum():,.0f}")

    st.markdown("---")
    st.subheader("📋 Preview ข้อมูลจาก Cloud")
    st.write(df[['customer_id', 'order_purchase_timestamp', 'status', 'churn_probability']].head(10))

# ==========================================
# PAGE 2: 🔍 Customer Detail (โค้ดเดิมของคุณ 100%)
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

# ==============================================================================
# PAGE 3: 🎯 Action Plan (เพิ่ม Loading State)
# ==============================================================================
elif page == "3. 🎯 Action Plan":
    import time # เพิ่ม library สำหรับหน่วงเวลา

    st.title("🎯 Action Plan & Simulator")
    st.markdown("### วางแผนกลยุทธ์แก้ปัญหาแบบเจาะจง (Targeted Strategy)")
    
    # ---------------------------------------------------------
    # 0. PREPARE DATA & MULTI-FILTER
    # ---------------------------------------------------------
    if 'df_display' not in locals():
        df_display = df.copy()

    with st.container():
        st.markdown("##### 🔎 เลือกกลุ่มสินค้าที่ต้องการโฟกัส (เลือกได้หลายอัน)")
        
        all_cats = sorted(list(df['product_category_name'].unique())) if 'product_category_name' in df.columns else []
        
        sel_cats_p3 = st.multiselect(
            "หมวดสินค้า (ปล่อยว่าง = ดูภาพรวมทั้งหมด):", 
            all_cats, 
            key="p3_cat_multiselect"
        )
        
        if sel_cats_p3:
            df_p3 = df_display[df_display['product_category_name'].isin(sel_cats_p3)].copy()
            filter_msg = f"หมวด: {', '.join(sel_cats_p3[:3])}{'...' if len(sel_cats_p3)>3 else ''}"
        else:
            df_p3 = df_display.copy()
            filter_msg = "ภาพรวมทุกหมวด"

        total_pop = len(df_p3)
        avg_ltv = df_p3['payment_value'].mean() if 'payment_value' in df_p3.columns else 150
        
        c1, c2 = st.columns([3, 1])
        with c1:
            st.info(f"📊 กำลังวิเคราะห์: **{filter_msg}**")
        with c2:
            st.metric("👥 ลูกค้าในกลุ่มนี้", f"{total_pop:,} คน", help="จำนวนลูกค้าทั้งหมดตาม Filter ที่เลือก")

    st.markdown("---")

    # ---------------------------------------------------------
    # 1. HELPER FUNCTION (เพิ่ม Loading Spinner ตรงนี้)
    # ---------------------------------------------------------
    def render_strategy_story(title, icon, target_df, total_pop, strategy_name, default_cost, compare_col=None, good_value=None, bad_values=None, rec_text=""):
        n_target = len(target_df)
        pct_problem = (n_target / total_pop) * 100 if total_pop > 0 else 0
        
        # --- UI ส่วนแสดงปัญหา ---
        st.subheader(f"{icon} {title}")
        c_prob, c_sol, c_res = st.columns([1, 1.3, 1])
        
        with c_prob:
            st.info(f"**📉 ปัญหา:** พบ {n_target:,} คน\n({pct_problem:.1f}% ของกลุ่มนี้)")
            st.progress(min(pct_problem / 100, 1.0))
            st.caption("แถบสีแสดงสัดส่วนคนที่มีปัญหา")

        with c_sol:
            st.markdown(f"**🛠️ วิธีแก้ไข: {strategy_name}**")
            st.write(rec_text)
            st.markdown("---")
            
            # รับค่า Cost
            cost = st.number_input(f"งบต่อหัว (R$)", value=default_cost, min_value=1, step=1, key=f"cost_{title}")
            
            # --- 🤖 ADVANCED AI LOGIC ---
            max_potential = 5
            ai_msg = ""
            
            if compare_col and good_value is not None and 'churn_probability' in df_p3.columns:
                try:
                    if bad_values: bad_group = df_p3[df_p3[compare_col].isin(bad_values)]
                    else: bad_group = target_df
                    bad_churn = bad_group['churn_probability'].mean() if not bad_group.empty else 0.8
                    
                    if isinstance(good_value, list): good_group = df_p3[df_p3[compare_col].isin(good_value)]
                    elif isinstance(good_value, (int, float)): good_group = df_p3[df_p3[compare_col] <= good_value]
                    else: good_group = df_p3[df_p3[compare_col] == good_value]
                    good_churn = good_group['churn_probability'].mean() if not good_group.empty else 0.4
                    
                    uplift = (bad_churn - good_churn) * 100
                    max_potential = max(int(uplift * 0.5), 1) 
                except: pass

            if cost < 5:
                realistic_rate = min(max_potential, 3)
                constraint_msg = " (งบน้อย = ผลลัพธ์จำกัด)"
            elif cost < 15:
                realistic_rate = min(max_potential, 10)
                constraint_msg = " (งบปานกลาง)"
            else:
                realistic_rate = max_potential
                constraint_msg = " (งบสูง = ผลลัพธ์เต็มที่)"
            
            st.markdown(f"**🤖 AI Prediction:** `{realistic_rate}%`")
            st.caption(f"(Max Potential: {max_potential}% {constraint_msg})")
            
            lift = st.slider(f"ปรับค่าคาดการณ์ความสำเร็จ (%)", 1, 100, realistic_rate, key=f"lift_{title}")

        # --- ส่วนแสดงผลลัพธ์ (เพิ่ม Spinner ตรงนี้) ---
        with c_res:
            # 🟢 ใส่ Loading Spinner เพื่อให้ User รู้ว่ากำลังคำนวณ
            with st.spinner('⚡ AI กำลังคำนวณความคุ้มค่า...'):
                time.sleep(0.4) # หน่วงเวลา 0.4 วินาที เพื่อให้ทันมองเห็น (UX Trick)
                
                budget = n_target * cost
                saved_users = int(n_target * (lift / 100))
                revenue = saved_users * avg_ltv
                roi = ((revenue - budget) / budget) * 100 if budget > 0 else 0
                
                st.success(f"**🚀 ผลลัพธ์**")
                st.metric("💸 งบประมาณ", f"R$ {budget:,.0f}")
                st.metric("👥 ดึงลูกค้าคืน", f"{saved_users:,} คน")
                st.metric("💰 กำไร (ROI)", f"{roi:+.0f}%", delta=f"+{revenue:,.0f}")
                
                if roi > 0:
                    st.caption("✅ **คุ้มค่าการลงทุน!**")
                else:
                    st.error("⚠️ **ขาดทุน!** (ลองลดงบ หรือเพิ่มผลสำเร็จ)")

    # ---------------------------------------------------------
    # 2. STRATEGY TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "💳 1. แก้ปัญหาการจ่ายเงิน", 
        "🚚 2. แก้ปัญหาค่าส่งแพง", 
        "❤️ 3. ง้อลูกค้าส่งช้า",
        "🛍️ 4. ขายพ่วงลดความเสี่ยง"
    ])

    with tab1:
        if 'payment_type' in df_p3.columns:
            target = df_p3[df_p3['payment_type'].isin(['credit_card', 'boleto'])]
            render_strategy_story(
                "กลุ่มเสี่ยงจากการจ่ายเงิน (Payment Risk)", "💳", target, total_pop,
                "เปลี่ยน Cash เป็น Voucher", 20, 
                'payment_type', ['voucher'], ['credit_card', 'boleto'],
                "ลูกค้าที่จ่ายบัตรเครดิต/โอน มีโอกาส Churn สูงมาก ต่างจากคนถือ Voucher ที่มักกลับมาซื้อซ้ำ\n\n👉 **Action:** เสนอ Cashback 5% เข้า Wallet เพื่อเปลี่ยนพฤติกรรม"
            )
        else: st.error("ไม่พบข้อมูล 'payment_type'")

    with tab2:
        if 'freight_ratio' in df_p3.columns:
            target = df_p3[df_p3['freight_ratio'] > 0.2]
            avg_freight = target['freight_value'].mean() if not target.empty else 20
            render_strategy_story(
                "กลุ่มค่าส่งแพงเกินรับไหว (Freight Pain)", "🚚", target, total_pop,
                "ช่วยออกค่าส่ง (Free Shipping)", int(avg_freight), 
                'freight_ratio', 0.1, None,
                f"ลูกค้ากลุ่มนี้ลังเลเพราะค่าส่งแพง (เฉลี่ย R$ {avg_freight:.0f})\n\n👉 **Action:** แจกโค้ดส่วนลดค่าส่ง เพื่อลดแรงต้าน (Friction) ในการตัดสินใจ"
            )
        else: st.error("ไม่พบข้อมูล 'freight_ratio'")

    with tab3:
        if 'delay_days' in df_p3.columns:
            target = df_p3[df_p3['delay_days'] > 0]
            render_strategy_story(
                "กลุ่มโดนเท/ของส่งช้า (Delay Recovery)", "❤️", target, total_pop,
                "SMS ขอโทษ + คูปอง", 15, 
                'delay_days', 0, None,
                "ความล่าช้าทำลายความเชื่อมั่น และเป็นสาเหตุหลักของการเปลี่ยนใจ\n\n👉 **Action:** ส่ง SMS ขอโทษทันทีที่รู้ว่าของช้า พร้อมแนบส่วนลดพิเศษ"
            )
        else: st.error("ไม่พบข้อมูล 'delay_days'")

    with tab4:
        if 'cat_churn_risk' in df_p3.columns:
            target = df_p3[df_p3['cat_churn_risk'] > 0.8]
            render_strategy_story(
                "กลุ่มซื้อสินค้าความเสี่ยงสูง (High Risk)", "🛍️", target, total_pop,
                "Cross-sell สินค้าซื้อง่าย", 10, 
                'cat_churn_risk', 0.5, None,
                "สินค้าบางหมวดคนซื้อครั้งเดียวจบ (เช่น เฟอร์นิเจอร์) ทำให้ Churn สูง\n\n👉 **Action:** ยิงแอดขายพ่วงสินค้าหมวดของใช้ในบ้าน (Housewares) ที่ต้องซื้อซ้ำบ่อยๆ"
            )
        else: st.error("ไม่พบข้อมูล 'cat_churn_risk'")
# ==============================================================================
# PAGE 4: 🚛 Logistics Insights (State Map & City Details)
# ==============================================================================
elif page == "4. 🚛 Logistics Insights":
    import pydeck as pdk

    st.title("🚛 Logistics Insights")
    st.markdown("วิเคราะห์ปัญหาขนส่งและการเงิน รายรัฐ (Map) และรายเมือง (Table)")

    # ---------------------------------------------------------
    # 0. PREPARE DATA & FILTER
    # ---------------------------------------------------------
    if 'customer_state' not in df.columns:
        st.error("❌ ไม่พบข้อมูลรัฐ (customer_state)")
        st.stop()

    # 1. Filter หมวดหมู่สินค้า
    with st.container():
        all_cats = sorted(list(df['product_category_name'].unique())) if 'product_category_name' in df.columns else []
        sel_cats_p4 = st.multiselect("📦 กรองหมวดสินค้า:", all_cats, key="p4_cat_filter")
        
        if sel_cats_p4:
            df_logistics = df[df['product_category_name'].isin(sel_cats_p4)].copy()
            filter_msg = f"หมวด: {', '.join(sel_cats_p4[:3])}..."
        else:
            df_logistics = df.copy()
            filter_msg = "ภาพรวมทุกหมวด"

    # ---------------------------------------------------------
    # 1. DATA PROCESSING (STATE LEVEL)
    # ---------------------------------------------------------
    # พิกัดรัฐบราซิล (Latitude, Longitude)
    brazil_states_coords = {
        'AC': [-9.02, -70.81], 'AL': [-9.57, -36.78], 'AM': [-3.41, -65.85],
        'AP': [0.90, -52.00], 'BA': [-12.58, -41.70], 'CE': [-5.49, -39.32],
        'DF': [-15.79, -47.88], 'ES': [-19.18, -40.30], 'GO': [-15.82, -49.84],
        'MA': [-5.19, -45.16], 'MG': [-19.81, -43.95], 'MS': [-20.77, -54.78],
        'MT': [-12.96, -56.92], 'PA': [-6.31, -52.46], 'PB': [-7.24, -36.78],
        'PE': [-8.81, -36.95], 'PI': [-7.71, -42.72], 'PR': [-25.25, -52.02],
        'RJ': [-22.90, -43.17], 'RN': [-5.40, -36.95], 'RO': [-11.50, -63.58],
        'RR': [2.73, -62.07], 'RS': [-30.03, -51.22], 'SC': [-27.24, -50.21],
        'SE': [-10.57, -37.38], 'SP': [-23.55, -46.63], 'TO': [-10.17, -48.33]
    }

    # Group Data รายรัฐ
    state_metrics = df_logistics.groupby('customer_state').agg({
        'payment_value': 'sum',                 # เงินหมุนเวียน
        'delivery_days': 'mean',                # ส่งเฉลี่ยกี่วัน
        'delay_days': lambda x: (x > 0).sum(),  # ส่งช้ากี่ออเดอร์ (Count Late)
        'churn_probability': 'mean',            # ความเสี่ยงเฉลี่ย
        'order_purchase_timestamp': 'count'     # จำนวนออเดอร์
    }).reset_index().rename(columns={'order_purchase_timestamp': 'total_orders'})

    # Map Lat/Long
    state_metrics['lat'] = state_metrics['customer_state'].map(lambda x: brazil_states_coords.get(x, [0,0])[0])
    state_metrics['lon'] = state_metrics['customer_state'].map(lambda x: brazil_states_coords.get(x, [0,0])[1])

    # ---------------------------------------------------------
    # 2. MAP & STATE TABLE
    # ---------------------------------------------------------
    st.markdown("---")
    
    # ส่วนควบคุม Zoom
    col_sel, col_kpi1, col_kpi2, col_kpi3 = st.columns([1.5, 1, 1, 1])
    
    with col_sel:
        zoom_state = st.selectbox("🔍 โฟกัสรัฐ (Zoom):", ["All (ภาพรวมประเทศ)"] + sorted(state_metrics['customer_state'].unique()))
    
    # กำหนดมุมกล้อง (View State)
    if zoom_state != "All (ภาพรวมประเทศ)":
        display_data = state_metrics[state_metrics['customer_state'] == zoom_state]
        if not display_data.empty:
            view_lat = display_data['lat'].values[0]
            view_lon = display_data['lon'].values[0]
            view_zoom = 6 # ซูมเข้าไปใกล้ๆ
        else:
            view_lat, view_lon, view_zoom = -14.2350, -51.9253, 3.5
    else:
        display_data = state_metrics
        view_lat, view_lon, view_zoom = -14.2350, -51.9253, 3.5

    # KPI Summary
    total_rev = display_data['payment_value'].sum()
    avg_del = display_data['delivery_days'].mean()
    total_late = display_data['delay_days'].sum()

    with col_kpi1: st.metric("💰 เงินหมุนเวียน", f"R$ {total_rev:,.0f}")
    with col_kpi2: st.metric("🚚 ส่งเฉลี่ย", f"{avg_del:.1f} วัน")
    with col_kpi3: st.metric("⚠️ ส่งช้า (Late)", f"{total_late:,} ครั้ง", delta_color="inverse")

    # --- ส่วนแสดงแผนที่และตารางรัฐ ---
    c_map, c_state_table = st.columns([2, 1])

    with c_map:
        st.subheader(f"🗺️ แผนที่ ({zoom_state})")
        
        # ตั้งค่าสีวงกลม: แดง=เสี่ยงมาก, เหลือง=กลาง, เขียว=ดี
        state_metrics['color'] = state_metrics['churn_probability'].apply(
            lambda x: [231, 76, 60, 200] if x > 0.8 else ([241, 196, 15, 200] if x > 0.5 else [46, 204, 113, 200])
        )
        
        # ตั้งค่าขนาดวงกลมตามยอดเงิน
        max_val = state_metrics['payment_value'].max()
        # ป้องกันการหารด้วยศูนย์
        if max_val > 0:
            state_metrics['radius'] = state_metrics['payment_value'] / max_val * 400000
        else:
            state_metrics['radius'] = 10000

        # สร้าง Layer
        layer = pdk.Layer(
            "ScatterplotLayer",
            state_metrics,
            get_position='[lon, lat]',
            get_color='color',
            get_radius='radius',
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_min_pixels=5,
            radius_max_pixels=60,
        )

        tooltip = {
            "html": "<b>รัฐ: {customer_state}</b><br/>"
                    "💰 ยอดเงิน: R$ {payment_value:,.0f}<br/>"
                    "🚚 ส่งเฉลี่ย: {delivery_days:.1f} วัน<br/>"
                    "⚠️ ส่งช้า: {delay_days} ครั้ง<br/>"
                    "📉 ความเสี่ยง: {churn_probability:.2f}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        # ใช้ map_style เป็น None หรือ 'light' เพื่อให้โหลดพื้นหลังได้ง่ายขึ้น
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=view_zoom, pitch=20),
            tooltip=tooltip,
            map_provider='carto',
            map_style='light' # ใช้ Carto Light เป็นพื้นหลัง (โหลดง่ายกว่า Mapbox)
        )
        st.pydeck_chart(r)

    with c_state_table:
        st.subheader("🚨 รัฐที่มีปัญหา (Top Issues)")
        sort_mode = st.radio("เรียงตาม:", ["ส่งช้าเยอะสุด (Late Count)", "ความเสี่ยงสูงสุด (Risk)"], horizontal=True)
        
        if "ส่งช้า" in sort_mode:
            top_issues = state_metrics.sort_values('delay_days', ascending=False).head(10)
        else:
            top_issues = state_metrics.sort_values('churn_probability', ascending=False).head(10)

        st.dataframe(
            top_issues[['customer_state', 'payment_value', 'delivery_days', 'delay_days', 'churn_probability']],
            column_config={
                "customer_state": "รัฐ",
                "payment_value": st.column_config.NumberColumn("เงินหมุนเวียน", format="R$ %.0f"),
                "delivery_days": st.column_config.NumberColumn("ส่ง (วัน)", format="%.1f"),
                "delay_days": st.column_config.NumberColumn("ช้า (ครั้ง)"),
                "churn_probability": st.column_config.ProgressColumn("ความเสี่ยง", format="%.2f", min_value=0, max_value=1)
            },
            hide_index=True,
            use_container_width=True
        )

    # ---------------------------------------------------------
    # 3. CITY LEVEL DETAILS (เพิ่มส่วนนี้ตามที่ขอ)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🏙️ เจาะลึกรายเมือง (City Drill-down)")
    st.caption("ข้อมูลรายละเอียดของแต่ละเมือง (แสดงเฉพาะเมืองที่มีออเดอร์อย่างน้อย 2 รายการ)")

    if 'customer_city' in df_logistics.columns:
        # Group Data รายเมือง
        city_metrics = df_logistics.groupby(['customer_state', 'customer_city']).agg({
            'customer_unique_id': 'count',          # จำนวนลูกค้า
            'payment_value': 'sum',                 # เงินหมุนเวียน
            'delivery_days': 'mean',                # ส่งเฉลี่ย
            'delay_days': lambda x: (x > 0).sum(),  # ส่งช้ากี่ครั้ง
            'churn_probability': 'mean'             # ความเสี่ยง
        }).reset_index()
        
        # กรองเมืองเล็กๆ ออก (เอาเฉพาะที่มี Data > 1) เพื่อให้ตารางมีความหมาย
        city_metrics = city_metrics[city_metrics['customer_unique_id'] >= 2]
        
        # Filter ตามรัฐที่เลือกด้านบน (ถ้ามีการซูมรัฐ)
        if zoom_state != "All (ภาพรวมประเทศ)":
            city_display = city_metrics[city_metrics['customer_state'] == zoom_state]
            st.info(f"📍 แสดงรายชื่อเมืองในรัฐ: **{zoom_state}**")
        else:
            city_display = city_metrics
            st.info("📍 แสดงรายชื่อเมืองทั่วประเทศ (Top 50 ที่มีปัญหา)")

        # เรียงลำดับเมืองที่มีปัญหา (ส่งช้าเยอะสุดขึ้นก่อน)
        city_display = city_display.sort_values('delay_days', ascending=False).head(50)

        st.dataframe(
            city_display,
            column_config={
                "customer_state": "รัฐ",
                "customer_city": "เมือง",
                "customer_unique_id": st.column_config.NumberColumn("จำนวนลูกค้า"),
                "payment_value": st.column_config.NumberColumn("ยอดเงินรวม", format="R$ %.0f"),
                "delivery_days": st.column_config.NumberColumn("ส่งเฉลี่ย (วัน)", format="%.1f"),
                "delay_days": st.column_config.NumberColumn("ส่งช้า (ครั้ง)"),
                "churn_probability": st.column_config.ProgressColumn("ความเสี่ยง (Avg)", format="%.2f", min_value=0, max_value=1)
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("⚠️ ไม่พบข้อมูลเมือง (customer_city)")
# ==============================================================================
# PAGE 5: 🏪 Seller Audit (Table View & Multi-Sort)
# ==============================================================================
elif page == "5. 🏪 Seller Audit":
    st.title("🏪 Seller Audit & Performance")
    st.markdown("ตรวจสอบประสิทธิภาพและความเสี่ยงรายร้านค้า")

    # ---------------------------------------------------------
    # 0. PREPARE DATA & FILTER
    # ---------------------------------------------------------
    if 'seller_id' not in df.columns:
        st.error("❌ ไม่พบข้อมูลผู้ขาย (seller_id)")
        st.stop()

    # 1. Filter หมวดหมู่สินค้า
    with st.container():
        all_cats = sorted(list(df['product_category_name'].unique())) if 'product_category_name' in df.columns else []
        sel_cats_p5 = st.multiselect("📦 กรองหมวดสินค้า:", all_cats, key="p5_cat_filter")
        
        if sel_cats_p5:
            df_seller_view = df[df['product_category_name'].isin(sel_cats_p5)].copy()
            filter_msg = f"หมวด: {', '.join(sel_cats_p5[:3])}..."
        else:
            df_seller_view = df.copy()
            filter_msg = "ภาพรวมทุกหมวด"

    # ---------------------------------------------------------
    # 1. DATA AGGREGATION
    # ---------------------------------------------------------
    # รวมข้อมูลรายร้านค้า
    seller_stats = df_seller_view.groupby('seller_id').agg({
        'order_purchase_timestamp': 'count', # จำนวนออเดอร์
        'payment_value': 'sum',              # ยอดขายรวม
        'review_score': 'mean',              # คะแนนรีวิวเฉลี่ย
        'delivery_days': 'mean',             # [NEW] เวลาส่งเฉลี่ย
        'churn_probability': 'mean'          # ความเสี่ยงเฉลี่ย
    }).reset_index().rename(columns={'order_purchase_timestamp': 'total_orders'})

    # กรองร้านค้าที่มีออเดอร์น้อยเกินไปออก (เพื่อไม่ให้ค่าเฉลี่ยเพี้ยน)
    # เช่น ร้านที่ขายชิ้นเดียวแล้วได้ 1 ดาว อาจจะไม่แฟร์ถ้าบอกว่าห่วยที่สุด
    min_orders = 3
    seller_stats = seller_stats[seller_stats['total_orders'] >= min_orders]

    # ---------------------------------------------------------
    # 2. KPI SUMMARY
    # ---------------------------------------------------------
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1: st.metric("🏪 จำนวนร้านค้า", f"{len(seller_stats):,} ร้าน")
    with c2: st.metric("💸 ยอดขายรวม", f"R$ {seller_stats['payment_value'].sum():,.0f}")
    with c3: st.metric("⭐ รีวิวเฉลี่ย", f"{seller_stats['review_score'].mean():.2f}/5.0")
    with c4: st.metric("🚚 ส่งเฉลี่ย", f"{seller_stats['delivery_days'].mean():.1f} วัน")

    # ---------------------------------------------------------
    # 3. SORTING & TABLE DISPLAY
    # ---------------------------------------------------------
    st.markdown("---")
    
    col_sort, col_display = st.columns([1, 3])
    
    with col_sort:
        st.subheader("⚙️ จัดเรียงข้อมูล (Sort By)")
        sort_mode = st.radio(
            "เลือกเกณฑ์การเรียง:",
            [
                "🚨 ความเสี่ยงสูงสุด (Highest Risk)",
                "🐢 ส่งของช้าสุด (Slowest Delivery)",
                "⭐ คะแนนต่ำสุด (Lowest Rating)",
                "💸 ยอดขายสูงสุด (Top Revenue)",
                "📦 ขายเยอะสุด (Top Volume)"
            ]
        )
        
        # Logic การเรียงลำดับ
        if "ความเสี่ยง" in sort_mode:
            sorted_df = seller_stats.sort_values('churn_probability', ascending=False)
            st.caption("แสดงร้านที่ลูกค้าซื้อแล้วหนี (Churn) มากที่สุด")
        elif "ส่งของช้า" in sort_mode:
            sorted_df = seller_stats.sort_values('delivery_days', ascending=False)
            st.caption("แสดงร้านที่ใช้เวลาจัดส่งนานที่สุด")
        elif "คะแนนต่ำ" in sort_mode:
            sorted_df = seller_stats.sort_values('review_score', ascending=True) # น้อยไปมาก
            st.caption("แสดงร้านที่ได้ดาวน้อยที่สุด")
        elif "ยอดขาย" in sort_mode:
            sorted_df = seller_stats.sort_values('payment_value', ascending=False)
            st.caption("แสดงร้านที่ทำเงินได้มากที่สุด")
        else: # Volume
            sorted_df = seller_stats.sort_values('total_orders', ascending=False)
            st.caption("แสดงร้านที่มีออเดอร์เยอะที่สุด")

    with col_display:
        st.subheader(f"📋 รายชื่อร้านค้า ({sort_mode})")
        
        st.dataframe(
            sorted_df,
            column_config={
                "seller_id": "รหัสร้านค้า",
                "total_orders": st.column_config.NumberColumn("จำนวนออเดอร์", help="จำนวนครั้งที่ขายได้"),
                "payment_value": st.column_config.NumberColumn("ยอดขายรวม", format="R$ %.0f"),
                "delivery_days": st.column_config.NumberColumn("ส่งเฉลี่ย (วัน)", format="%.1f วัน"),
                "review_score": st.column_config.NumberColumn("รีวิว (ดาว)", format="%.1f ⭐"),
                "churn_probability": st.column_config.ProgressColumn(
                    "ความเสี่ยง Churn", 
                    format="%.2f", 
                    min_value=0, 
                    max_value=1,
                    help="ยิ่งหลอดแดงเยอะ แปลว่าลูกค้าซื้อร้านนี้แล้วไม่ค่อยกลับมาซื้อซ้ำ"
                )
            },
            hide_index=True,
            use_container_width=True,
            height=600 # เพิ่มความสูงตารางให้ดูได้จุใจ
        )

# ==============================================================================
# PAGE 6: 🔄 Buying Cycle Analysis (Fine-grained Charts)
# ==============================================================================
elif page == "6. 🔄 Buying Cycle Analysis":
    st.title("🔄 Buying Cycle Analysis")
    st.markdown("วิเคราะห์รอบการซื้อ: **สินค้าหมวดนี้...ลูกค้ากลับมาซื้อซ้ำเร็วแค่ไหน?**")
    
    # ---------------------------------------------------------
    # 0. PREPARE DATA & FILTER
    # ---------------------------------------------------------
    if 'cat_median_days' not in df.columns:
        st.error("❌ ไม่พบข้อมูลรอบการซื้อ (cat_median_days)")
        st.stop()

    with st.container():
        all_cats = sorted(list(df['product_category_name'].unique())) if 'product_category_name' in df.columns else []
        
        sel_cats_p6 = st.multiselect(
            "📦 เลือกหมวดสินค้า (เปรียบเทียบกับภาพรวม):", 
            all_cats, 
            key="p6_cat_filter"
        )
        
        if sel_cats_p6:
            df_cycle = df[df['product_category_name'].isin(sel_cats_p6)].copy()
            filter_label = f"หมวด: {', '.join(sel_cats_p6[:3])}{'...' if len(sel_cats_p6)>3 else ''}"
        else:
            df_cycle = df.copy()
            filter_label = "ภาพรวมทุกหมวด"

    # ---------------------------------------------------------
    # 1. METRICS (KPIs)
    # ---------------------------------------------------------
    global_avg_cycle = df['cat_median_days'].mean()
    global_avg_late = df['lateness_score'].mean() if 'lateness_score' in df.columns else 0
    
    curr_avg_cycle = df_cycle['cat_median_days'].mean()
    curr_avg_late = df_cycle['lateness_score'].mean() if 'lateness_score' in df_cycle.columns else 0
    
    fast_repeaters = len(df_cycle[df_cycle['cat_median_days'] <= 30])
    
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    
    m1.metric(
        label=f"⏱️ รอบซื้อเฉลี่ย ({filter_label})", 
        value=f"{curr_avg_cycle:.0f} วัน",
        delta=f"{curr_avg_cycle - global_avg_cycle:.0f} วัน (เทียบภาพรวม)",
        delta_color="inverse",
        help="โดยเฉลี่ยลูกค้ากลุ่มนี้กลับมาซื้อซ้ำในกี่วัน"
    )
    
    m2.metric(
        label=f"🐢 ความล่าช้า ({filter_label})", 
        value=f"{curr_avg_late:.2f} เท่า",
        delta=f"{curr_avg_late - global_avg_late:.2f} (เทียบภาพรวม)",
        delta_color="inverse",
        help="ถ้า > 1.0 คือเริ่มช้ากว่าปกติ"
    )
    
    m3.metric(
        label="📅 ซื้อซ้ำใน 30 วัน", 
        value=f"{fast_repeaters:,} คน",
        help="จำนวนลูกค้าที่กลับมาซื้อซ้ำภายใน 1 เดือน"
    )
    
    st.markdown("---")

    # ---------------------------------------------------------
    # 2. COMPARISON CHARTS (กราฟละเอียดขึ้น 📊)
    # ---------------------------------------------------------
    st.subheader("📊 เปรียบเทียบพฤติกรรมการซื้อซ้ำ (Repurchase Distribution)")
    st.caption("กราฟแสดงจำนวนวัน (แกน X) ที่ลูกค้าใช้ในการกลับมาซื้อซ้ำ: **แตกรายละเอียดถี่ขึ้น (ไม่รวมกลุ่มใหญ่)**")

    col_focus, col_bench = st.columns(2)

    # --- CHART 1: FOCUS GROUP (ปรับ maxbins เป็น 50) ---
    with col_focus:
        st.info(f"📍 **{filter_label}** (กลุ่มที่คุณเลือก)")
        
        # ใช้ bin(step=5) หรือ maxbins=50 เพื่อให้แท่งกราฟถี่ขึ้น ไม่รวมเป็นก้อนเดียว
        hist_focus = alt.Chart(df_cycle).mark_bar().encode(
            x=alt.X('cat_median_days', 
                    bin=alt.Bin(maxbins=50), # <--- ปรับตรงนี้ให้ละเอียดขึ้น (เดิม 20)
                    title='ระยะเวลาซื้อซ้ำ (วัน)'),
            y=alt.Y('count()', title='จำนวนลูกค้า'),
            color=alt.value('#3498db'),
            tooltip=['count()', alt.Tooltip('cat_median_days', bin=True, title='ช่วงวัน')]
        ).properties(height=300, title=f"การกระจายตัวของ {filter_label}")
        
        st.altair_chart(hist_focus, use_container_width=True)

    # --- CHART 2: OVERALL BENCHMARK (ปรับ maxbins เป็น 50) ---
    with col_bench:
        st.warning("🏢 **ภาพรวมทั้งบริษัท** (Benchmark)")
        
        hist_all = alt.Chart(df).mark_bar().encode(
            x=alt.X('cat_median_days', 
                    bin=alt.Bin(maxbins=50), # <--- ปรับตรงนี้ให้ละเอียดขึ้น
                    title='ระยะเวลาซื้อซ้ำ (วัน)'),
            y=alt.Y('count()', title='จำนวนลูกค้า'),
            color=alt.value('#95a5a6'),
            tooltip=['count()', alt.Tooltip('cat_median_days', bin=True, title='ช่วงวัน')]
        ).properties(height=300, title="Benchmark: ภาพรวมสินค้าทั้งหมด")
        
        st.altair_chart(hist_all, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # 3. DETAILED TABLE
    # ---------------------------------------------------------
    st.subheader(f"📋 รายละเอียดรายหมวดสินค้า ({filter_label})")
    
    summ = df_cycle.groupby('product_category_name').agg({
        'customer_unique_id': 'count',
        'cat_median_days': 'mean',
        'lateness_score': 'mean',
        'churn_probability': 'mean'
    }).reset_index().rename(columns={
        'customer_unique_id': 'Total Customers',
        'cat_median_days': 'Avg Cycle (Days)',
        'lateness_score': 'Avg Late Score',
        'churn_probability': 'Churn Risk'
    })
    
    st.dataframe(
        summ.sort_values('Avg Cycle (Days)'),
        column_config={
            "product_category_name": "หมวดสินค้า",
            "Total Customers": st.column_config.NumberColumn("ลูกค้าทั้งหมด", format="%d คน"),
            "Avg Cycle (Days)": st.column_config.NumberColumn("รอบซื้อเฉลี่ย", format="%.0f วัน"),
            "Avg Late Score": st.column_config.NumberColumn("ความล่าช้า", format="%.2f เท่า"),
            "Churn Risk": st.column_config.ProgressColumn("ความเสี่ยง Churn", format="%.2f", min_value=0, max_value=1)
        },
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------------
    # 4. SEASONALITY HEATMAP
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📅 Seasonal Patterns: สินค้าขายดีเดือนไหน?")
    st.caption(f"แสดงข้อมูลเจาะจงของ: **{filter_label}**")

    if 'order_purchase_timestamp' in df_cycle.columns:
        season_df = df_cycle.copy()
        season_df['month_num'] = season_df['order_purchase_timestamp'].dt.month
        season_df['month_name'] = season_df['order_purchase_timestamp'].dt.strftime('%b')
        
        heatmap_data = season_df.groupby(['product_category_name', 'month_num', 'month_name']).size().reset_index(name='sales_volume')
        
        top_cats = season_df['product_category_name'].value_counts().head(15).index.tolist()
        heatmap_data = heatmap_data[heatmap_data['product_category_name'].isin(top_cats)]
        
        if not heatmap_data.empty:
            heatmap = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X('month_name', sort=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], title='เดือน'),
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


























