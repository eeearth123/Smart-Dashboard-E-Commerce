import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP
# ==========================================
st.set_page_config(
    page_title="Olist Executive Cockpit",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .metric-card { background-color:#f0f2f6; border-radius:10px;
                   padding:15px; box-shadow:2px 2px 5px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD FROM BIGQUERY
# ==========================================
from google.oauth2 import service_account
from google.cloud import bigquery

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
            creds_info, scopes=scopes)
        client = bigquery.Client(
            credentials=credentials,
            project=creds_info["project_id"],
            location="asia-southeast1"
        )
        df = client.query(
            "SELECT * FROM `academic-moon-483615-t2.Dashboard.input`"
        ).to_dataframe()
        return df, None
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def process_features(df_raw):
    df = df_raw.copy()

    # ── 3.1 แปลงวันที่ ────────────────────────────────────────────────────
    for col in ['order_purchase_timestamp',
                'order_delivered_customer_date',
                'order_estimated_delivery_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # ── 3.2 เรียงข้อมูลตาม customer + เวลา ──────────────────────────────
    if 'order_purchase_timestamp' in df.columns:
        df = df.sort_values(['customer_unique_id', 'order_purchase_timestamp']
                            ).reset_index(drop=True)

    # ── 3.3 Logistics Features ────────────────────────────────────────────
    if 'order_delivered_customer_date' in df.columns and \
       'order_purchase_timestamp' in df.columns:
        df['delivery_days'] = (
            df['order_delivered_customer_date'] -
            df['order_purchase_timestamp']
        ).dt.days.clip(lower=0)
    else:
        df['delivery_days'] = np.nan

    if 'order_estimated_delivery_date' in df.columns and \
       'order_purchase_timestamp' in df.columns:
        df['estimated_days'] = (
            df['order_estimated_delivery_date'] -
            df['order_purchase_timestamp']
        ).dt.days
    else:
        df['estimated_days'] = np.nan

    # delivery_vs_estimated
    df['delivery_vs_estimated'] = df['estimated_days'] - df['delivery_days']

    # ── 3.4 Price & Freight ───────────────────────────────────────────────
    if 'freight_value' in df.columns and 'price' in df.columns:
        df['freight_ratio'] = np.where(
            df['price'] > 0,
            df['freight_value'] / df['price'], 0
        )
        df['payment_value'] = df['price'] + df['freight_value']
    else:
        df['freight_ratio']  = 0
        df['payment_value']  = df.get('price', 0)

    # ── 3.5 Payment Features ─────────────────────────────────────────────
    if 'payment_sequential' in df.columns:
        df['uses_multiple_payments'] = (
            df['payment_sequential'].fillna(1) > 1
        ).astype(int)
    else:
        df['uses_multiple_payments'] = 0

    if 'payment_type' in df.columns:
        df['uses_voucher'] = (
            df['payment_type'].fillna('') == 'voucher'
        ).astype(int)
    else:
        df['uses_voucher'] = 0

    # ── 3.6 Review Score → Binary ─────────────────────────────────────────
    if 'review_score' in df.columns:
        df['review_score']  = pd.to_numeric(df['review_score'], errors='coerce')
        df['is_low_score']  = (df['review_score'].fillna(3) <= 2).astype(int)
        df['is_high_score'] = (df['review_score'].fillna(3) == 5).astype(int)
    else:
        df['review_score']  = 3.0
        df['is_low_score']  = 0
        df['is_high_score'] = 0

    # ── 3.7 Purchase Count & Repeat Buyer ─────────────────────────────────
    df['purchase_count'] = (
        df.groupby('customer_unique_id').cumcount() + 1
    )
    df['is_first_purchase'] = (df['purchase_count'] == 1).astype(int)
    df['is_repeat_buyer']   = (df['purchase_count'] >= 2).astype(int)

    # ── 3.8 Gap Features ─────────────────────────────────────────────────
    if 'order_purchase_timestamp' in df.columns:
        df['prev_purchase_date'] = df.groupby('customer_unique_id')[
            'order_purchase_timestamp'].shift(1)
        df['days_since_last_purchase'] = (
            df['order_purchase_timestamp'] - df['prev_purchase_date']
        ).dt.days

        median_gap = df.loc[df['is_repeat_buyer'] == 1,
                            'days_since_last_purchase'].median()
        if pd.isna(median_gap): median_gap = 90.0

        df['avg_purchase_gap'] = (
            df.groupby('customer_unique_id')['days_since_last_purchase']
            .transform('mean')
        )
        global_avg = df['avg_purchase_gap'].median()
        df['avg_purchase_gap'] = df['avg_purchase_gap'].fillna(global_avg)

        df['gap_vs_avg'] = df['avg_purchase_gap'] - df['days_since_last_purchase']
        df['days_since_last_purchase'] = df['days_since_last_purchase'].fillna(median_gap)
        df['gap_vs_avg']               = df['gap_vs_avg'].fillna(0)

        # gap_real / gap_vs_avg_real — เฉพาะ Repeat Buyers
        df['gap_real']        = np.where(df['is_repeat_buyer'] == 1,
                                         df['days_since_last_purchase'], 0)
        df['gap_vs_avg_real'] = np.where(df['is_repeat_buyer'] == 1,
                                         df['gap_vs_avg'], 0)
    else:
        for c in ['days_since_last_purchase','avg_purchase_gap',
                  'gap_vs_avg','gap_real','gap_vs_avg_real']:
            df[c] = 0

    # ── 3.9 Category Churn Risk ──────────────────────────────────────────
    if 'cat_churn_risk' not in df.columns:
        df['cat_churn_risk'] = 0.80

    # ── 3.10 Lateness Score ──────────────────────────────────────────────
    if 'order_purchase_timestamp' in df.columns:
        ref_date = df['order_purchase_timestamp'].max()
        last_order = df.groupby('customer_unique_id')[
            'order_purchase_timestamp'].transform('max')
        df['days_since_purchase'] = (ref_date - last_order).dt.days

        tmp = df.sort_values(['customer_unique_id','product_category_name',
                              'order_purchase_timestamp'])
        tmp['prev_ts'] = tmp.groupby(
            ['customer_unique_id','product_category_name']
        )['order_purchase_timestamp'].shift(1)
        tmp['order_gap'] = (
            tmp['order_purchase_timestamp'] - tmp['prev_ts']
        ).dt.days
        valid_gaps = tmp[(tmp['order_gap'] >= 7) & (tmp['order_gap'] <= 730)]
        if len(valid_gaps) > 10:
            cat_med = valid_gaps.groupby('product_category_name')[
                'order_gap'].median().rename('cat_median_days')
            df = df.merge(cat_med, on='product_category_name', how='left')
        else:
            df['cat_median_days'] = 180
        df['cat_median_days'] = df['cat_median_days'].fillna(180).clip(lower=7)
        df['lateness_score']  = (
            df['days_since_purchase'] / df['cat_median_days']
        ).clip(lower=0)
    else:
        df['days_since_purchase'] = 90
        df['cat_median_days']     = 180
        df['lateness_score']      = 0.5

    # ── 3.11 delay_days ──────────────────────────────────────────────────
    if 'order_delivered_customer_date' in df.columns and \
       'order_estimated_delivery_date' in df.columns:
        df['delay_days'] = (
            df['order_delivered_customer_date'] -
            df['order_estimated_delivery_date']
        ).dt.days.fillna(0)
    else:
        df['delay_days'] = 0

    return df

# ==========================================
# 4. LOAD MODEL (แก้ไขชื่อไฟล์)
# ==========================================
@st.cache_resource
def load_models():
    d = os.path.dirname(os.path.abspath(__file__))
    try:
        model    = joblib.load(os.path.join(d, 'olist_churn_model_final (1).pkl'))
        features = joblib.load(os.path.join(d, 'model_features_final (1).pkl'))
        return model, features, None
    except Exception as e:
        return None, None, str(e)

# ==========================================
# 5. PREDICT
# ==========================================
def predict_churn(df, model, feature_names, threshold):
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        X[col] = df[col] if col in df.columns else 0
    X = X.fillna(X.median())

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X).astype(float)

    pred = (proba >= threshold).astype(int)
    return proba, pred

# ==========================================
# 6. SIDEBAR & REFRESH
# ==========================================
with st.sidebar:
    if st.button('🔄 Refresh Data'):
        st.cache_data.clear()
        st.rerun()

df_raw, bq_error = load_bq_data()
model, feature_names, model_error = load_models()
best_threshold = 0.55  # เปลี่ยนเป็น 0.55

if bq_error:
    st.error(f"⚠️ BigQuery Error: {bq_error}")
    st.stop()
if model_error:
    st.warning(f"⚠️ Model: {model_error}")

# ==========================================
# 7. PROCESS & PREDICT
# ==========================================
df = process_features(df_raw)

if model is not None and feature_names:
    proba, pred = predict_churn(df, model, feature_names, best_threshold)
    df['churn_probability'] = proba
    df['churn_prediction']  = pred

    if 'product_category_name' in df.columns:
        cat_risk_map     = df.groupby('product_category_name')['churn_probability'].mean()
        df['cat_churn_risk'] = df['product_category_name'].map(cat_risk_map)
else:
    df['churn_probability'] = 0.5
    df['churn_prediction']  = 1

df['is_churn'] = df['churn_prediction']

# ==========================================
# 8. STATUS CLASSIFICATION (แก้ไข Medium Risk เป็น 0.40-0.75)
# ==========================================
def get_status(row):
    prob = row.get('churn_probability', 0)
    late = row.get('lateness_score', 0)
    if late > 3.0:  return 'Lost (Late > 3x)'
    if prob > 0.75: return 'High Risk'
    if late > 1.5:  return 'Warning (Late > 1.5x)'
    if prob >= 0.40:  return 'Medium Risk'  # เปลี่ยนจาก 0.50 เป็น 0.40
    return 'Active'

df['status'] = df.apply(get_status, axis=1)

# ==========================================
# 9. NAVIGATION
# ==========================================
st.sidebar.title("✈️ Olist Cockpit")
st.sidebar.success(f"✅ โหลดข้อมูลแล้ว ({len(df):,} rows)")
st.sidebar.info(f"🎯 Model Threshold: {best_threshold:.2f}")
st.sidebar.markdown("""
**📊 Business Rules:**
- 🔴 Lost: Late > 3.0
- 🟥 High Risk: AI > 75%
- 🟧 Warning: Late > 1.5
- 🟨 Medium Risk: AI 40-75%
- 🟩 Active: AI < 40%
""")
page = st.sidebar.radio("Navigation", [
    "1. 💰 Business Overview",
    "2. 📊 Churn Overview",
    "3. 🎯 Action Plan",
    "4. 🚛 Logistics Insights",
    "5. 🏪 Seller Audit",
    "6. 🔄 Buying Cycle Analysis",
    "7. 🔍 Customer Detail",
])
st.sidebar.markdown("---")

# ==========================================
# PAGE 1: Business Overview
# ==========================================
if page == "1. 💰 Business Overview":
    st.title("💰 Business Overview")
    st.caption("ภาพรวมรายได้และสุขภาพธุรกิจ — วิเคราะห์จากข้อมูลย้อนหลัง")

    with st.expander("🌪️ กรองข้อมูล", expanded=False):
        all_cats = sorted(df['product_category_name'].dropna().unique()) \
                   if 'product_category_name' in df.columns else []
        sel_cats = st.multiselect("หมวดสินค้า (ว่าง = ทั้งหมด):", all_cats, key="p1_cat")

    df_d = df[df['product_category_name'].isin(sel_cats)].copy() if sel_cats else df.copy()
    st.markdown("---")

    total_rev   = df_d['payment_value'].sum() if 'payment_value' in df_d.columns else 0
    avg_order   = df_d['payment_value'].mean() if 'payment_value' in df_d.columns else 0
    n_customers = df_d['customer_unique_id'].nunique() if 'customer_unique_id' in df_d.columns else 0
    clv         = avg_order * df_d.groupby('customer_unique_id').size().mean() \
                  if 'customer_unique_id' in df_d.columns and n_customers > 0 else avg_order

    mom_growth = None
    if 'order_purchase_timestamp' in df_d.columns and not df_d.empty:
        df_d['_month'] = df_d['order_purchase_timestamp'].dt.to_period('M')
        all_months = pd.period_range(start=df_d['_month'].min(), end=df_d['_month'].max(), freq='M')
        monthly_rev_series = df_d.groupby('_month')['payment_value'].sum().reindex(all_months, fill_value=0)

        if len(monthly_rev_series) >= 3:
            last_complete_m = monthly_rev_series.iloc[-2]
            prev_m          = monthly_rev_series.iloc[-3]
            if prev_m > 0:
                mom_growth = ((last_complete_m - prev_m) / prev_m) * 100
        elif len(monthly_rev_series) == 2:
            last_m = monthly_rev_series.iloc[-1]
            first_m = monthly_rev_series.iloc[-2]
            if first_m > 0:
                mom_growth = ((last_m - first_m) / first_m) * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Total Revenue", f"R$ {total_rev:,.0f}")
    if mom_growth is not None:
        k2.metric("📈 MoM Growth", f"{mom_growth:+.1f}%", delta=f"{mom_growth:+.1f}%")
    else:
        k2.metric("📈 MoM Growth", "N/A")
    k3.metric("🛒 Avg Order Value", f"R$ {avg_order:,.0f}")
    k4.metric("👤 CLV (Estimated)", f"R$ {clv:,.0f}")
    st.markdown("---")

    st.subheader("📈 Monthly Revenue Trend")
    if 'order_purchase_timestamp' in df_d.columns and not df_d.empty:
        rev_trend = df_d.set_index('order_purchase_timestamp')['payment_value'].resample('MS').sum().fillna(0).reset_index()
        rev_trend.columns = ['Month', 'Revenue']
        rev_trend['Growth'] = rev_trend['Revenue'].pct_change().replace([np.inf, -np.inf], np.nan) * 100
        plot_df = rev_trend.iloc[:-1] if len(rev_trend) > 1 else rev_trend

        base = alt.Chart(plot_df).encode(
            x=alt.X('Month:T', axis=alt.Axis(format='%b %Y', title='', labelAngle=-45))
        )

        bars = base.mark_bar(color='#1E88E5', opacity=0.7).encode(
            y=alt.Y('Revenue:Q', title='Revenue (R$)', axis=alt.Axis(grid=False)),
            tooltip=[alt.Tooltip('Month:T', format='%B %Y'), alt.Tooltip('Revenue:Q', format=',.0f')]
        )

        line = base.mark_line(color='#E53935', strokeWidth=3, point=alt.OverlayMarkDef(color='#E53935')).encode(
            y=alt.Y('Growth:Q', title='Growth Rate (%)', axis=alt.Axis(titleColor='#E53935', orient='right')),
            tooltip=[alt.Tooltip('Month:T', format='%B %Y'), alt.Tooltip('Growth:Q', format='.1f', title='Growth %')]
        )

        chart = alt.layer(bars, line).resolve_scale(y='independent').properties(height=350)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("ไม่มีข้อมูลเพียงพอสำหรับการแสดง Trend")

# ==========================================
# PAGE 2: Churn Overview
# ==========================================
elif page == "2. 📊 Churn Overview":
    st.title("📊 Churn Overview")

    with st.expander("ℹ️ วิธีแบ่งกลุ่มลูกค้า (อัปเดตใหม่)", expanded=True):
        st.markdown("""
| สถานะ | เงื่อนไข |
|---|---|
| 🔴 Lost | Lateness > 3.0 |
| 🟥 High Risk | AI Predict > 75% |
| 🟧 Warning | Lateness > 1.5 |
| 🟨 **Medium Risk** | **AI Predict 40–75%** (ขยายแล้ว!) |
| 🟩 Active | AI < 40% + มาตามรอบปกติ |
        """)

    with st.expander("🌪️ กรองข้อมูล", expanded=False):
        all_cats = sorted(df['product_category_name'].dropna().unique()) \
                   if 'product_category_name' in df.columns else []
        sel_cats = st.multiselect("หมวดสินค้า (ว่าง = ทั้งหมด):", all_cats, key="p2_cat")

    df_d = df[df['product_category_name'].isin(sel_cats)].copy() if sel_cats else df.copy()
    st.caption(f"กำลังแสดง: **{'ทั้งหมด' if not sel_cats else ', '.join(sel_cats[:3])}**")
    st.markdown("---")

    total   = len(df_d)
    risk_df = df_d[df_d['status'].isin(['High Risk','Warning (Late > 1.5x)'])]
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("🚨 At-Risk",        f"{len(risk_df)/total*100:.1f}%" if total else "0%")
    k2.metric("🤖 AI Predicted",   f"{(df_d['churn_probability'] >= best_threshold).mean()*100:.1f}%")
    k3.metric("💸 Revenue at Risk", f"R$ {risk_df['payment_value'].sum():,.0f}")
    k4.metric("👥 Risk / Total",    f"{len(risk_df):,} / {total:,}")
    k5.metric("🔄 Avg Cycle",       f"{df_d['cat_median_days'].mean():.0f} วัน"
                                    if 'cat_median_days' in df_d.columns else "N/A")
    st.markdown("---")

    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("📈 Churn Risk Trend")
        if 'order_purchase_timestamp' in df_d.columns and not df_d.empty:
            df_d['month_year'] = df_d['order_purchase_timestamp'].dt.to_period('M')
            trend_data = []
            for name, grp in df_d.groupby('month_year'):
                t = len(grp)
                if t == 0: continue
                rule = len(grp[grp['status'].isin(['High Risk','Warning (Late > 1.5x)'])])
                ai   = (grp['churn_probability'] >= best_threshold).sum()
                trend_data.append({'Date': str(name),
                                   'Rule-based Risk (%)': rule/t*100,
                                   'AI Predicted Churn (%)': ai/t*100})
            tdf2 = pd.DataFrame(trend_data)
            if len(tdf2) > 1:
                tdf2 = tdf2.iloc[:-1]
                tdf2['Date'] = pd.to_datetime(tdf2['Date'])
                melted = tdf2.melt('Date', var_name='Type', value_name='Rate (%)')
                chart = alt.Chart(melted).mark_line(point=True).encode(
                    x=alt.X('Date', axis=alt.Axis(format='%b %Y', title='Timeline')),
                    y=alt.Y('Rate (%)', title='Churn Rate (%)'),
                    color=alt.Color('Type', scale=alt.Scale(
                        domain=['Rule-based Risk (%)','AI Predicted Churn (%)'],
                        range=['#e67e22','#8e44ad']),
                        legend=alt.Legend(orient='bottom')),
                    tooltip=['Date','Type',alt.Tooltip('Rate (%)', format='.1f')]
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("ข้อมูลไม่เพียงพอสำหรับ Trend")

    with c2:
        st.subheader("💰 Revenue by Risk")
        if not df_d.empty:
            stats = df_d.groupby('status').agg(
                Count=('customer_unique_id','count'),
                Revenue=('payment_value','sum')
            ).reset_index()
            domain = ['Active','Medium Risk','Warning (Late > 1.5x)','High Risk','Lost (Late > 3x)']
            range_ = ['#2ecc71','#f1c40f','#e67e22','#e74c3c','#95a5a6']
            donut = alt.Chart(stats).mark_arc(innerRadius=60).encode(
                theta=alt.Theta('Count', type='quantitative'),
                color=alt.Color('status', scale=alt.Scale(domain=domain, range=range_),
                                legend=dict(orient='bottom')),
                tooltip=['status', alt.Tooltip('Count',format=','),
                         alt.Tooltip('Revenue',format=',.0f')]
            ).properties(height=350)
            st.altair_chart(donut, use_container_width=True)

# ==========================================
# PAGE 4: Logistics Insights
# ==========================================
elif page == "4. 🚛 Logistics Insights":
    st.title("🚛 Logistics & Delivery Insights")
    st.caption("วิเคราะห์ประสิทธิภาพการจัดส่ง และผลกระทบต่อ Churn")
    
    # ── 4.1 Filters ──────────────────────────────────────────────────────
    with st.expander("🔍 กรองข้อมูล", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        
        with f1:
            all_states = sorted(df['customer_state'].dropna().unique()) \
                        if 'customer_state' in df.columns else []
            sel_states = st.multiselect("รัฐ/จังหวัด:", all_states)
        
        with f2:
            all_cats = sorted(df['product_category_name'].dropna().unique()) \
                      if 'product_category_name' in df.columns else []
            sel_cats = st.multiselect("หมวดสินค้า:", all_cats)
        
        with f3:
            risk_filter = st.multiselect(
                "กลุ่มความเสี่ยง:",
                options=['High Risk', 'Warning (Late > 1.5x)', 'Medium Risk', 
                         'Lost (Late > 3x)', 'Active'],
                default=['High Risk', 'Warning (Late > 1.5x)', 'Lost (Late > 3x)']
            )
        
        with f4:
            delay_threshold = st.slider(
                "ความล่าช้า (วัน):",
                min_value=0, max_value=30,
                value=(0, 10)
            )
    
    # ── 4.2 Filter Data ──────────────────────────────────────────────────
    df_log = df.copy()
    
    if sel_states:
        df_log = df_log[df_log['customer_state'].isin(sel_states)]
    
    if sel_cats:
        df_log = df_log[df_log['product_category_name'].isin(sel_cats)]
    
    if risk_filter:
        df_log = df_log[df_log['status'].isin(risk_filter)]
    
    if 'delay_days' in df_log.columns:
        df_log = df_log[
            (df_log['delay_days'] >= delay_threshold[0]) &
            (df_log['delay_days'] <= delay_threshold[1])
        ]
    
    # ── 4.3 Key Metrics ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 ภาพรวม Logistics Performance")
    
    # คำนวณ metrics
    total_orders = len(df_log)
    avg_delivery_days = df_log['delivery_days'].mean() if 'delivery_days' in df_log.columns else 0
    median_delivery_days = df_log['delivery_days'].median() if 'delivery_days' in df_log.columns else 0
    
    if 'delay_days' in df_log.columns:
        late_orders = len(df_log[df_log['delay_days'] > 0])
        on_time_rate = ((df_log['delay_days'] <= 0).sum() / total_orders * 100) if total_orders > 0 else 0
        avg_delay = df_log[df_log['delay_days'] > 0]['delay_days'].mean()
    else:
        late_orders = 0
        on_time_rate = 100
        avg_delay = 0
    
    if 'delivery_vs_estimated' in df_log.columns:
        early_deliveries = len(df_log[df_log['delivery_vs_estimated'] > 0])
        late_deliveries = len(df_log[df_log['delivery_vs_estimated'] < 0])
    else:
        early_deliveries = 0
        late_deliveries = 0
    
    avg_freight = df_log['freight_value'].mean() if 'freight_value' in df_log.columns else 0
    avg_freight_ratio = df_log['freight_ratio'].mean() if 'freight_ratio' in df_log.columns else 0
    
    # แสดง metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    
    m1.metric("📦 ออเดอร์ทั้งหมด", f"{total_orders:,}")
    m2.metric("⏱️ ส่งเฉลี่ย", f"{avg_delivery_days:.1f} วัน", 
              f"Median: {median_delivery_days:.0f} วัน")
    m3.metric("✅ ส่งตรงเวลา", f"{on_time_rate:.1f}%",
              "เป้าหมาย: 90%+" if on_time_rate < 90 else "ดีมาก! 🎉")
    m4.metric("🐌 ล่าช้าเฉลี่ย", f"{avg_delay:.1f} วัน" if late_orders > 0 else "N/A",
              f"{late_orders:,} ออเดอร์" if late_orders > 0 else "ไม่มี")
    m5.metric("🚚 ค่าส่งเฉลี่ย", f"R$ {avg_freight:.2f}",
              f"{avg_freight_ratio:.1%} ของราคา")
    
    # ── 4.4 Delivery Performance Trend ───────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Delivery Performance Trend")
    
    if 'order_purchase_timestamp' in df_log.columns and not df_log.empty:
        # จัดกลุ่มตามเดือน
        df_log['_month'] = df_log['order_purchase_timestamp'].dt.to_period('M')
        
        monthly_stats = df_log.groupby('_month').agg({
            'delivery_days': 'mean',
            'delay_days': lambda x: (x > 0).sum() if 'delay_days' in df_log.columns else 0,
            'customer_unique_id': 'count'
        }).reset_index()
        
        monthly_stats.columns = ['Month', 'Avg Delivery Days', 'Late Count', 'Total Orders']
        monthly_stats['On-Time Rate'] = (1 - monthly_stats['Late Count'] / monthly_stats['Total Orders']) * 100
        monthly_stats['Date'] = monthly_stats['Month'].astype(str)
        
        if len(monthly_stats) > 1:
            # สร้างกราฟ
            base = alt.Chart(monthly_stats).encode(
                x=alt.X('Date:T', axis=alt.Axis(format='%b %Y', labelAngle=-45))
            )
            
            # กราฟเส้น: Delivery Days
            line_delivery = base.mark_line(color='#2E86AB', strokeWidth=3, point=True).encode(
                y=alt.Y('Avg Delivery Days:Q', title='ระยะเวลาจัดส่ง (วัน)', axis=alt.Axis(titleColor='#2E86AB')),
                tooltip=['Date', 'Avg Delivery Days', 'Total Orders']
            )
            
            # กราฟแท่ง: On-Time Rate
            bar_ontime = base.mark_bar(color='#A23B72', opacity=0.7).encode(
                y=alt.Y('On-Time Rate:Q', title='On-Time Rate (%)', axis=alt.Axis(titleColor='#A23B72', orient='right')),
                tooltip=['Date', 'On-Time Rate']
            )
            
            # รวมกราฟ
            chart = alt.layer(line_delivery, bar_ontime).resolve_scale(y='independent').properties(
                height=350,
                title='Delivery Performance Over Time'
            )
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("📊 มีข้อมูลไม่เพียงพอสำหรับแสดง Trend")
    
    # ── 4.5 Delivery Time Distribution ───────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Delivery Time Distribution")
    
    if 'delivery_days' in df_log.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram: Delivery Days
            hist_data = df_log['delivery_days'].dropna()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(hist_data, bins=30, color='#2E86AB', edgecolor='white', alpha=0.8)
            ax.axvline(median_delivery_days, color='red', linestyle='--', linewidth=2, 
                      label=f'Median: {median_delivery_days:.0f} วัน')
            ax.axvline(avg_delivery_days, color='orange', linestyle='--', linewidth=2,
                      label=f'Mean: {avg_delivery_days:.1f} วัน')
            ax.set_xlabel('ระยะเวลาจัดส่ง (วัน)')
            ax.set_ylabel('จำนวนออเดอร์')
            ax.set_title('Distribution of Delivery Days')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
        
        with col2:
            # Box Plot: Delivery Days by State
            if 'customer_state' in df_log.columns:
                state_delivery = df_log.groupby('customer_state')['delivery_days'].median().reset_index()
                state_delivery = state_delivery.sort_values('delivery_days', ascending=False).head(15)
                
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.barh(state_delivery['customer_state'], state_delivery['delivery_days'], 
                        color='#A23B72', edgecolor='white')
                ax2.set_xlabel('Median Delivery Days')
                ax2.set_title('Top 15 States - Slowest Delivery')
                ax2.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig2)
    
    # ── 4.6 Delay Analysis ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🐌 Delay Analysis")
    
    if 'delay_days' in df_log.columns:
        delay_stats = df_log[df_log['delay_days'] > 0]['delay_days']
        
        if len(delay_stats) > 0:
            d1, d2, d3, d4 = st.columns(4)
            
            d1.metric("📉 ออเดอร์ที่ล่าช้า", f"{len(delay_stats):,}",
                     f"{len(delay_stats)/total_orders*100:.1f}% ของทั้งหมด")
            d2.metric("⏳ ล่าช้าเฉลี่ย", f"{delay_stats.mean():.1f} วัน")
            d3.metric("🔴 ล่าช้าสูงสุด", f"{delay_stats.max():.0f} วัน")
            d4.metric("📊 Median Delay", f"{delay_stats.median():.1f} วัน")
            
            # Delay Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(delay_stats, bins=20, color='#E74C3C', edgecolor='white', alpha=0.8)
            ax.axvline(delay_stats.mean(), color='black', linestyle='--', linewidth=2,
                      label=f'Mean: {delay_stats.mean():.1f} วัน')
            ax.set_xlabel('ความล่าช้า (วัน)')
            ax.set_ylabel('จำนวนออเดอร์')
            ax.set_title('Delay Distribution (Late Orders Only)')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
            
            # Delay Impact on Churn
            st.markdown("#### 🔗 ความสัมพันธ์ระหว่าง Delay กับ Churn")
            
            df_delay_analysis = df_log.copy()
            df_delay_analysis['is_late'] = (df_delay_analysis['delay_days'] > 0).astype(int)
            
            if 'is_churn' in df_delay_analysis.columns:
                late_churn_rate = df_delay_analysis[df_delay_analysis['is_late'] == 1]['is_churn'].mean()
                ontime_churn_rate = df_delay_analysis[df_delay_analysis['is_late'] == 0]['is_churn'].mean()
                
                impact_col1, impact_col2 = st.columns(2)
                
                impact_col1.metric(
                    "🔴 Churn Rate (ล่าช้า)",
                    f"{late_churn_rate:.1%}",
                    delta=f"+{((late_churn_rate/ontime_churn_rate)-1)*100:.1f}% vs ตรงเวลา"
                )
                
                impact_col2.metric(
                    "🟢 Churn Rate (ตรงเวลา)",
                    f"{ontime_churn_rate:.1%}",
                    delta="Baseline"
                )
                
                if late_churn_rate > ontime_churn_rate:
                    st.warning(f"""
                    ⚠️ **ออเดอร์ที่ล่าช้ามี Churn Rate สูงกว่า {late_churn_rate/ontime_churn_rate:.1f} เท่า!**
                    
                    **แนะนำ:**
                    • ปรับปรุง logistics ในรัฐที่มีปัญหา
                    • แจ้งลูกค้าล่วงหน้าหากมีการล่าช้า
                    • ให้ชดเชย (คูปอง/ส่วนลด) สำหรับออเดอร์ที่ล่าช้า
                    """)
    
    # ── 4.7 Geographic Analysis ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🗺️ Geographic Logistics Performance")
    
    if 'customer_state' in df_log.columns:
        # Brazil coordinates
        brazil_coords = {
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
        
        # Aggregate by state
        state_stats = df_log.groupby('customer_state').agg({
            'delivery_days': 'mean',
            'delay_days': lambda x: (x > 0).sum() if 'delay_days' in df_log.columns else 0,
            'freight_value': 'mean',
            'customer_unique_id': 'count',
            'churn_probability': 'mean'
        }).reset_index()
        
        state_stats.columns = ['State', 'Avg Delivery Days', 'Late Count', 'Avg Freight', 'Orders', 'Churn Risk']
        state_stats['On-Time Rate'] = 1 - (state_stats['Late Count'] / state_stats['Orders'])
        
        # Add coordinates
        state_stats['lat'] = state_stats['State'].map(lambda x: brazil_coords.get(x, [0, 0])[0])
        state_stats['lon'] = state_stats['State'].map(lambda x: brazil_coords.get(x, [0, 0])[1])
        
        # Map visualization
        import pydeck as pdk
        
        st.markdown("#### 📍 Logistics Performance by State")
        
        view_state = pdk.ViewState(
            latitude=-14.24,
            longitude=-51.93,
            zoom=3.5,
            pitch=20
        )
        
        # Layer: Delivery performance
        layer = pdk.Layer(
            "ScatterplotLayer",
            state_stats,
            get_position='[lon, lat]',
            get_color=[241, 196, 15, 200],
            get_radius=200000,
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_min_pixels=5,
            radius_max_pixels=50
        )
        
        tooltip = {
            "html": "<b>{State}</b><br/>"
                    "📦 Orders: {Orders}<br/>"
                    "⏱️ Avg Delivery: {Avg Delivery Days:.1f} days<br/>"
                    "🐌 Late: {Late Count}<br/>"
                    "⚠️ Churn Risk: {Churn Risk:.1%}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
        
        map_chart = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_provider='carto',
            map_style='light'
        )
        
        st.pydeck_chart(map_chart)
        
        # State ranking table
        st.markdown("#### 📊 State Performance Ranking")
        
        tab1, tab2 = st.tabs(["🐌 ส่งช้าที่สุด", "⚠️ Churn Risk สูงสุด"])
        
        with tab1:
            slow_states = state_stats.nlargest(10, 'Avg Delivery Days')
            st.dataframe(
                slow_states[['State', 'Avg Delivery Days', 'Late Count', 'Orders', 'On-Time Rate']],
                column_config={
                    "Avg Delivery Days": st.column_config.NumberColumn("Avg Days", format="%.1f"),
                    "On-Time Rate": st.column_config.ProgressColumn(format="%.1%")
                },
                hide_index=True,
                use_container_width=True
            )
        
        with tab2:
            risk_states = state_stats.nlargest(10, 'Churn Risk')
            st.dataframe(
                risk_states[['State', 'Churn Risk', 'Avg Delivery Days', 'Late Count', 'Orders']],
                column_config={
                    "Churn Risk": st.column_config.ProgressColumn(format="%.1%"),
                    "Avg Delivery Days": st.column_config.NumberColumn("Avg Days", format="%.1f")
                },
                hide_index=True,
                use_container_width=True
            )
    
    # ── 4.8 Freight Analysis ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💰 Freight Cost Analysis")
    
    if 'freight_value' in df_log.columns and 'freight_ratio' in df_log.columns:
        f1, f2, f3 = st.columns(3)
        
        f1.metric("💵 ค่าส่งเฉลี่ย", f"R$ {avg_freight:.2f}")
        f2.metric("📊 Freight Ratio", f"{avg_freight_ratio:.1%}",
                 "สูง!" if avg_freight_ratio > 0.2 else "ปกติ")
        f3.metric("💸 Total Freight Cost", 
                 f"R$ {df_log['freight_value'].sum():,.0f}")
        
        # Freight vs Churn correlation
        if 'churn_probability' in df_log.columns:
            high_freight_churn = df_log[df_log['freight_ratio'] > 0.2]['churn_probability'].mean()
            low_freight_churn = df_log[df_log['freight_ratio'] <= 0.2]['churn_probability'].mean()
            
            st.markdown("#### 🔗 Freight Ratio vs Churn Probability")
            
            corr_col1, corr_col2 = st.columns(2)
            
            corr_col1.metric(
                "🔴 High Freight (>20%)",
                f"Churn Prob: {high_freight_churn:.1%}",
                delta=f"+{((high_freight_churn/low_freight_churn)-1)*100:.1f}% vs ต่ำ"
            )
            
            corr_col2.metric(
                "🟢 Low Freight (≤20%)",
                f"Churn Prob: {low_freight_churn:.1%}",
                delta="Baseline"
            )
            
            if high_freight_churn > low_freight_churn:
                st.info(f"""
                💡 **Insight:** ลูกค้าที่มี Freight Ratio สูง (>20%) มีแนวโน้ม Churn สูงกว่า 
                {high_freight_churn/low_freight_churn:.1f} เท่า
                
                **แนะนำ:**
                • ให้ส่วนลดค่าส่งสำหรับลูกค้ากลุ่มนี้
                • ปรับปรุง logistics เพื่อลดค่าส่ง
                • เสนอ Free Shipping เมื่อซื้อครบ
                """)
        
        # Freight distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df_log['freight_ratio'].dropna(), bins=30, color='#27AE60', edgecolor='white', alpha=0.8)
        ax.axvline(0.2, color='red', linestyle='--', linewidth=2, label='Threshold: 20%')
        ax.set_xlabel('Freight Ratio')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Freight Ratio')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
    
    # ── 4.9 Actionable Insights ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Actionable Insights & Recommendations")
    
    insights = []
    
    # Insight 1: Late delivery impact
    if 'delay_days' in df_log.columns and 'is_churn' in df_log.columns:
        late_churn = df_log[df_log['delay_days'] > 0]['is_churn'].mean()
        ontime_churn = df_log[df_log['delay_days'] <= 0]['is_churn'].mean()
        
        if late_churn > ontime_churn * 1.2:
            insights.append({
                "priority": "🔴 High",
                "issue": "ออเดอร์ล่าช้ามี Churn Rate สูง",
                "impact": f"Churn สูงขึ้น {late_churn/ontime_churn:.1f} เท่า",
                "action": "• ปรับปรุง logistics<br>• แจ้งลูกค้าล่วงหน้า<br>• ให้ชดเชย"
            })
    
    # Insight 2: High freight ratio
    if 'freight_ratio' in df_log.columns and 'churn_probability' in df_log.columns:
        high_freight = df_log[df_log['freight_ratio'] > 0.2]['churn_probability'].mean()
        low_freight = df_log[df_log['freight_ratio'] <= 0.2]['churn_probability'].mean()
        
        if high_freight > low_freight * 1.1:
            insights.append({
                "priority": "🟠 Medium",
                "issue": "ค่าส่งสูงสัมพันธ์กับ Churn",
                "impact": f"Churn Prob. สูงขึ้น {high_freight/low_freight:.1f} เท่า",
                "action": "• ลดค่าส่ง<br>• Free Shipping threshold<br>• เจรจากับ carrier"
            })
    
    # Insight 3: Slow states
    if 'customer_state' in df_log.columns and 'delivery_days' in df_log.columns:
        slow_states = df_log.groupby('customer_state')['delivery_days'].median()
        if len(slow_states) > 0:
            slowest = slow_states.idxmax()
            slowest_days = slow_states.max()
            
            if slowest_days > 15:
                insights.append({
                    "priority": "🟡 Low",
                    "issue": f"รัฐ {slowest} ส่งช้ามาก",
                    "impact": f"เฉลี่ย {slowest_days:.0f} วัน",
                    "action": "• หา carrier ใหม่<br>• เพิ่ม warehouse<br>• แจ้งลูกค้า"
                })
    
    # Display insights
    if insights:
        for i, insight in enumerate(insights, 1):
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 3, 2, 3])
                
                col1.markdown(f"**{insight['priority']}**")
                col2.markdown(f"**{insight['issue']}**")
                col3.markdown(f"📊 {insight['impact']}")
                col4.markdown(f"🔧 {insight['action']}")
                
                if i < len(insights):
                    st.markdown("---")
    else:
        st.success("✅ Logistics performance ดีมาก! ไม่พบปัญหาใหญ่")
    
    # ── 4.10 Export Report ───────────────────────────────────────────────
    st.markdown("---")
    if st.button("📥 Export Logistics Report (CSV)"):
        export_data = df_log[[
            'customer_unique_id', 'order_id', 'delivery_days', 'delay_days',
            'freight_value', 'freight_ratio', 'churn_probability', 'status'
        ]].copy()
        
        csv = export_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ ดาวน์โหลด Logistics Report",
            data=csv,
            file_name="logistics_analysis.csv",
            mime='text/csv'
        )
# ==========================================
# PAGE 4: Logistics
# ==========================================
elif page == "4. 🚛 Logistics Insights":
    import pydeck as pdk
    st.title("🚛 Logistics Insights")

    if 'customer_state' not in df.columns:
        st.error("❌ ไม่พบ customer_state"); st.stop()

    c1,c2 = st.columns(2)
    with c1:
        all_cats = sorted(df['product_category_name'].dropna().unique()) \
                   if 'product_category_name' in df.columns else []
        sel_c = st.multiselect("📦 หมวดสินค้า:", all_cats, key="p4_cat")
    with c2:
        sel_s = st.multiselect("👥 สถานะ:",
            ['High Risk','Warning (Late > 1.5x)','Medium Risk','Lost (Late > 3x)','Active'],
            key="p4_st")

    df_log = df.copy()
    if sel_c: df_log = df_log[df_log['product_category_name'].isin(sel_c)]
    if sel_s: df_log = df_log[df_log['status'].isin(sel_s)]

    brazil = {
        'AC':[-9.02,-70.81],'AL':[-9.57,-36.78],'AM':[-3.41,-65.85],
        'AP':[0.90,-52.00], 'BA':[-12.58,-41.70],'CE':[-5.49,-39.32],
        'DF':[-15.79,-47.88],'ES':[-19.18,-40.30],'GO':[-15.82,-49.84],
        'MA':[-5.19,-45.16],'MG':[-19.81,-43.95],'MS':[-20.77,-54.78],
        'MT':[-12.96,-56.92],'PA':[-6.31,-52.46],'PB':[-7.24,-36.78],
        'PE':[-8.81,-36.95],'PI':[-7.71,-42.72],'PR':[-25.25,-52.02],
        'RJ':[-22.90,-43.17],'RN':[-5.40,-36.95],'RO':[-11.50,-63.58],
        'RR':[2.73,-62.07], 'RS':[-30.03,-51.22],'SC':[-27.24,-50.21],
        'SE':[-10.57,-37.38],'SP':[-23.55,-46.63],'TO':[-10.17,-48.33]
    }

    sm = df_log.groupby('customer_state').agg(
        payment_value=('payment_value','sum'),
        delivery_days=('delivery_days','mean'),
        delay_count=('delay_days', lambda x: (x>0).sum()),
        churn_probability=('churn_probability','mean'),
        total_orders=('order_purchase_timestamp','count')
    ).reset_index()
    sm['lat'] = sm['customer_state'].map(lambda x: brazil.get(x,[0,0])[0])
    sm['lon'] = sm['customer_state'].map(lambda x: brazil.get(x,[0,0])[1])

    st.markdown("---")
    cs,k1,k2,k3 = st.columns([1.5,1,1,1])
    with cs:
        zoom = st.selectbox("🔍 โฟกัสรัฐ:",
                            ["All"]+sorted(sm['customer_state'].unique()))
    disp = sm if zoom=="All" else sm[sm['customer_state']==zoom]
    view_lat = disp['lat'].mean() if zoom!="All" else -14.24
    view_lon = disp['lon'].mean() if zoom!="All" else -51.93
    view_z   = 6 if zoom!="All" else 3.5
    k1.metric("💰 ยอดเงิน",  f"R$ {disp['payment_value'].sum():,.0f}")
    k2.metric("🚚 ส่งเฉลี่ย",f"{disp['delivery_days'].mean():.1f} วัน")
    k3.metric("⚠️ ส่งช้า",   f"{disp['delay_count'].sum():,} ครั้ง")

    cm_,ct_ = st.columns([2,1])
    with cm_:
        sm['color']  = sm['churn_probability'].apply(
            lambda x: [231,76,60,200] if x>0.8 else
                      ([241,196,15,200] if x>0.5 else [46,204,113,200]))
        mx = sm['payment_value'].max()
        sm['radius'] = (sm['payment_value']/mx*400000) if mx>0 else 10000
        layer = pdk.Layer("ScatterplotLayer", sm,
            get_position='[lon,lat]', get_color='color', get_radius='radius',
            pickable=True, opacity=0.8, stroked=True, filled=True,
            radius_min_pixels=5, radius_max_pixels=60)
        tooltip = {"html":"<b>{customer_state}</b><br/>💰 R$ {payment_value}<br/>"
                          "🚚 {delivery_days} วัน<br/>⚠️ {delay_count} ครั้ง",
                   "style":{"backgroundColor":"steelblue","color":"white"}}
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=view_lat,longitude=view_lon,
                                             zoom=view_z,pitch=20),
            tooltip=tooltip, map_provider='carto', map_style='light'))
    with ct_:
        st.subheader("🚨 Top Issues")
        sort_m = st.radio("เรียงตาม:", ["ส่งช้า","ความเสี่ยง"], horizontal=True)
        top_i  = sm.sort_values('delay_count' if "ช้า" in sort_m else 'churn_probability',
                                ascending=False).head(10)
        st.dataframe(top_i[['customer_state','payment_value','delivery_days',
                             'delay_count','churn_probability']],
            column_config={
                "payment_value": st.column_config.NumberColumn("เงิน", format="R$%.0f"),
                "delivery_days": st.column_config.NumberColumn("ส่ง(วัน)", format="%.1f"),
                "churn_probability": st.column_config.ProgressColumn("Risk", format="%.2f",min_value=0,max_value=1)
            }, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("🏙️ เจาะลึกรายเมือง")
    if 'customer_city' in df_log.columns:
        city_m = df_log.groupby(['customer_state','customer_city']).agg(
            n=('customer_unique_id','count'),
            revenue=('payment_value','sum'),
            del_days=('delivery_days','mean'),
            late=('delay_days', lambda x: (x>0).sum()),
            risk=('churn_probability','mean')
        ).reset_index()
        city_m = city_m[city_m['n'] >= 2]
        disp_c = city_m[city_m['customer_state']==zoom] if zoom!="All" else city_m
        st.dataframe(disp_c.sort_values('late',ascending=False).head(50),
            column_config={
                "n": st.column_config.NumberColumn("ลูกค้า"),
                "revenue": st.column_config.NumberColumn("ยอดเงิน", format="R$%.0f"),
                "del_days": st.column_config.NumberColumn("ส่ง(วัน)", format="%.1f"),
                "late": st.column_config.NumberColumn("ส่งช้า"),
                "risk": st.column_config.ProgressColumn("Risk", format="%.2f",min_value=0,max_value=1)
            }, hide_index=True, use_container_width=True)

# ==========================================
# PAGE 5: Seller Audit
# ==========================================
elif page == "5. 🏪 Seller Audit":
    st.title("🏪 Seller Audit")

    if 'seller_id' not in df.columns:
        st.error("❌ ไม่พบ seller_id"); st.stop()

    c1,c2 = st.columns(2)
    with c1:
        all_cats = sorted(df['product_category_name'].dropna().unique()) \
                   if 'product_category_name' in df.columns else []
        sel_c = st.multiselect("📦 หมวดสินค้า:", all_cats, key="p5c")
    with c2:
        sel_s = st.multiselect("👥 สถานะ:",
            ['High Risk','Warning (Late > 1.5x)','Medium Risk','Lost (Late > 3x)','Active'], key="p5s")

    dfs = df.copy()
    if sel_c: dfs = dfs[dfs['product_category_name'].isin(sel_c)]
    if sel_s: dfs = dfs[dfs['status'].isin(sel_s)]

    agg = {'order_purchase_timestamp':'count','payment_value':'sum',
           'churn_probability':'mean','delivery_days':'mean'}
    if 'review_score' in dfs.columns: agg['review_score']='mean'
    ss = dfs.groupby('seller_id').agg(agg).reset_index()
    ss = ss.rename(columns={'order_purchase_timestamp':'orders'})
    ss = ss[ss['orders'] >= 3]

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("🏪 ร้านค้า",     f"{len(ss):,}")
    k2.metric("💸 ยอดขายรวม",   f"R$ {ss['payment_value'].sum():,.0f}")
    k3.metric("⭐ รีวิวเฉลี่ย", f"{ss['review_score'].mean():.2f}" if 'review_score' in ss.columns else "N/A")
    k4.metric("🚚 ส่งเฉลี่ย",   f"{ss['delivery_days'].mean():.1f} วัน")

    st.markdown("---")
    cs_,cd_ = st.columns([1,3])
    with cs_:
        sort_m = st.radio("เรียงตาม:", [
            "🚨 ความเสี่ยง","🐢 ส่งช้า","⭐ คะแนนต่ำ","💸 ยอดขาย","📦 ปริมาณ"])
    with cd_:
        if "ความเสี่ยง" in sort_m:  sdf = ss.sort_values('churn_probability',ascending=False)
        elif "ส่งช้า" in sort_m:    sdf = ss.sort_values('delivery_days',ascending=False)
        elif "คะแนนต่ำ" in sort_m:  sdf = ss.sort_values('review_score',ascending=True)
        elif "ยอดขาย" in sort_m:   sdf = ss.sort_values('payment_value',ascending=False)
        else:                       sdf = ss.sort_values('orders',ascending=False)
        st.dataframe(sdf, column_config={
            "orders": st.column_config.NumberColumn("Orders"),
            "payment_value": st.column_config.NumberColumn("Revenue", format="R$%.0f"),
            "delivery_days": st.column_config.NumberColumn("ส่ง(วัน)", format="%.1f"),
            "review_score": st.column_config.NumberColumn("Review", format="%.1f⭐"),
            "churn_probability": st.column_config.ProgressColumn("Risk", format="%.2f",min_value=0,max_value=1)
        }, hide_index=True, use_container_width=True, height=600)

# ==========================================
# PAGE 6: Buying Cycle
# ==========================================
elif page == "6. 🔄 Buying Cycle Analysis":
    st.title("🔄 Buying Cycle Analysis")

    all_cats = sorted(df['product_category_name'].dropna().unique()) \
               if 'product_category_name' in df.columns else []
    sel_c  = st.multiselect("📦 หมวดสินค้า:", all_cats, key="p6c")
    df_cy  = df[df['product_category_name'].isin(sel_c)].copy() if sel_c else df.copy()
    label  = ', '.join(sel_c[:3]) if sel_c else "ทุกหมวด"

    g_avg  = df['cat_median_days'].mean()
    c_avg  = df_cy['cat_median_days'].mean()
    c_late = df_cy['lateness_score'].mean() if 'lateness_score' in df_cy.columns else 0
    fast   = (df_cy['cat_median_days'] <= 30).sum()

    m1,m2,m3 = st.columns(3)
    m1.metric(f"⏱️ รอบซื้อเฉลี่ย", f"{c_avg:.0f} วัน",
              f"{c_avg-g_avg:+.0f} วัน vs ภาพรวม", delta_color="inverse")
    m2.metric("🐢 ความล่าช้า", f"{c_late:.2f}x")
    m3.metric("📅 ซื้อซ้ำใน 30 วัน", f"{fast:,} คน")

    st.markdown("---")
    st.subheader("📈 Buying Cycle Trend")
    if 'order_purchase_timestamp' in df_cy.columns:
        tmp2 = df_cy.sort_values(['customer_unique_id','order_purchase_timestamp'])
        tmp2['prev_t'] = tmp2.groupby('customer_unique_id')['order_purchase_timestamp'].shift(1)
        tmp2['gap']    = (tmp2['order_purchase_timestamp']-tmp2['prev_t']).dt.days
        rep = tmp2[tmp2['gap'].notna() & (tmp2['gap']>0)].copy()
        if not rep.empty:
            rep['month_year'] = rep['order_purchase_timestamp'].dt.to_period('M')
            tgap = rep.groupby('month_year')['gap'].mean().reset_index()
            if len(tgap) > 1:
                tgap = tgap.iloc[:-1]
                tgap['Date'] = pd.to_datetime(tgap['month_year'].astype(str))
                lc = alt.Chart(tgap).mark_line(point=True, strokeWidth=3).encode(
                    x=alt.X('Date', axis=alt.Axis(format='%b %Y')),
                    y=alt.Y('gap', title='ระยะเวลาซื้อซ้ำเฉลี่ย (วัน)',
                            scale=alt.Scale(zero=False)),
                    color=alt.value('#e67e22'),
                    tooltip=['Date', alt.Tooltip('gap', format='.1f', title='วัน')]
                ).properties(height=350)
                st.altair_chart(lc, use_container_width=True)
            else:
                st.info("ข้อมูลไม่เพียงพอสำหรับ Trend")
        else:
            st.info("ไม่พบลูกค้าที่ซื้อซ้ำในหมวดนี้")

    st.markdown("---")
    st.subheader("📋 รายละเอียดรายหมวด")
    summ = df_cy.groupby('product_category_name').agg(
        Customers=('customer_unique_id','count'),
        Cycle_Days=('cat_median_days','mean'),
        Late_Score=('lateness_score','mean'),
        Churn_Risk=('churn_probability','mean')
    ).reset_index().sort_values('Cycle_Days')
    st.dataframe(summ, column_config={
        "Customers": st.column_config.NumberColumn("ลูกค้า", format="%d คน"),
        "Cycle_Days": st.column_config.NumberColumn("รอบซื้อ", format="%.0f วัน"),
        "Late_Score": st.column_config.NumberColumn("ความล่าช้า", format="%.2fx"),
        "Churn_Risk": st.column_config.ProgressColumn("Risk",format="%.2f",min_value=0,max_value=1)
    }, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Seasonal Heatmap")
    if 'order_purchase_timestamp' in df_cy.columns:
        sea = df_cy.copy()
        sea['month_num']  = sea['order_purchase_timestamp'].dt.month
        sea['month_name'] = sea['order_purchase_timestamp'].dt.strftime('%b')
        hm = sea.groupby(['product_category_name','month_num','month_name']
                         ).size().reset_index(name='vol')
        top_c = sea['product_category_name'].value_counts().head(15).index.tolist()
        hm    = hm[hm['product_category_name'].isin(top_c)]
        if not hm.empty:
            chart = alt.Chart(hm).mark_rect().encode(
                x=alt.X('month_name', sort=['Jan','Feb','Mar','Apr','May','Jun',
                                            'Jul','Aug','Sep','Oct','Nov','Dec'],
                        title='เดือน'),
                y=alt.Y('product_category_name', title='หมวด'),
                color=alt.Color('vol', scale=alt.Scale(scheme='orangered'), title='ยอดขาย'),
                tooltip=['product_category_name','month_name',
                         alt.Tooltip('vol', format=',')]
            ).properties(height=500)
            st.altair_chart(chart, use_container_width=True)
            st.info("💡 สีส้มเข้ม = High Season → เตรียมสต็อกล่วงหน้า")

# ==========================================
# PAGE 7: Customer Detail (ย้ายมาหน้าสุดท้าย)
# ==========================================
elif page == "7. 🔍 Customer Detail":
    st.title("🔍 Customer Deep Dive")

    with st.expander("🔎 Filters", expanded=True):
        f1,f2,f3 = st.columns(3)
        with f1:
            sel_status = st.multiselect("สถานะ:",
                ['High Risk','Warning (Late > 1.5x)','Medium Risk','Lost (Late > 3x)','Active'],
                default=['High Risk','Warning (Late > 1.5x)'])
        with f2:
            all_cats = sorted(df['product_category_name'].dropna().unique()) \
                       if 'product_category_name' in df.columns else []
            sel_cats = st.multiselect("หมวดสินค้า:", all_cats)
        with f3:
            search_id = st.text_input("ค้นหา Customer ID:", "")

    mask = df['status'].isin(sel_status)
    if sel_cats:   mask = mask & df['product_category_name'].isin(sel_cats)
    if search_id:  mask = mask & df['customer_unique_id'].str.contains(
                        search_id, case=False, na=False)
    filtered = df[mask]

    if 'product_category_name' in df.columns and not filtered.empty:
        cat_ov   = df.groupby('product_category_name').agg(
            Total=('customer_unique_id','count'),
            Cycle=('cat_median_days','mean')).reset_index()
        cat_risk = filtered.groupby('product_category_name').agg(
            Risk=('customer_unique_id','count')).reset_index()
        cat_s    = cat_risk.merge(cat_ov, on='product_category_name', how='left')
        cat_s['Risk_Pct'] = cat_s['Risk'] / cat_s['Total']
        cat_s = cat_s.sort_values('Risk', ascending=False)

        cc, ct = st.columns([1.5, 2.5])
        with cc:
            st.subheader("📊 Top 10 หมวดเสี่ยง")
            base   = alt.Chart(cat_s.head(10)).encode(
                y=alt.Y('product_category_name', sort='-x', title=None))
            b_tot  = base.mark_bar(color='#f0f2f6').encode(x='Total')
            b_risk = base.mark_bar(color='#e74c3c').encode(x='Risk')
            st.altair_chart(b_tot+b_risk, use_container_width=True)
        with ct:
            st.subheader("📋 รายละเอียด")
            st.dataframe(cat_s, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"📄 รายชื่อลูกค้า ({len(filtered):,} คน)")
    show = [c for c in ['customer_unique_id','status','churn_probability',
                        'lateness_score','cat_median_days','payment_value',
                        'product_category_name'] if c in df.columns]
    st.dataframe(
        filtered[show].sort_values('churn_probability', ascending=False),
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "Risk", format="%.2f", min_value=0, max_value=1),
            "lateness_score": st.column_config.NumberColumn("Late", format="%.1fx")
        }, use_container_width=True)
