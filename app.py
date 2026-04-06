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
#    สร้างทุก Feature จาก columns ที่มีใน BQ
#    Available: customer_unique_id, customer_id,
#    order_purchase_timestamp, order_delivered_customer_date,
#    order_estimated_delivery_date, product_category_name,
#    price, freight_value, payment_type, payment_installments,
#    payment_sequential, review_score, has_comment,
#    seller_id, customer_city, customer_state
# ==========================================
def process_features(df_raw):
    df = df_raw.copy()

    # ── 3.1 แปลงวันที่ ────────────────────────────────────────────────────
    for col in ['order_purchase_timestamp',
                'order_delivered_customer_date',
                'order_estimated_delivery_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # ── 3.2 เรียงข้อมูลตาม customer + เวลา (สำคัญมากสำหรับ Shift) ──────
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

    # delivery_vs_estimated (บวก = ส่งเร็วกว่า, ลบ = ส่งช้ากว่า)
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
    # cumcount → purchase_count (นับจาก 1)
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

        # Median ของ Repeat Buyers เท่านั้น (ไม่ Impute ด้วย one-time buyer)
        median_gap = df.loc[df['is_repeat_buyer'] == 1,
                            'days_since_last_purchase'].median()
        if pd.isna(median_gap): median_gap = 90.0

        # avg_purchase_gap ต่อ customer
        df['avg_purchase_gap'] = (
            df.groupby('customer_unique_id')['days_since_last_purchase']
            .transform('mean')
        )
        global_avg = df['avg_purchase_gap'].median()
        df['avg_purchase_gap'] = df['avg_purchase_gap'].fillna(global_avg)

        # gap_vs_avg (บวก = มาเร็วกว่าปกติ)
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
    # คำนวณชั่วคราวจากค่าเฉลี่ย churn_probability (จะ update หลัง predict)
    if 'cat_churn_risk' not in df.columns:
        df['cat_churn_risk'] = 0.80   # default ก่อน predict

    # ── 3.10 Lateness Score (ใช้ Reference Date = max ในข้อมูล) ─────────
    if 'order_purchase_timestamp' in df.columns:
        ref_date = df['order_purchase_timestamp'].max()
        last_order = df.groupby('customer_unique_id')[
            'order_purchase_timestamp'].transform('max')
        df['days_since_purchase'] = (ref_date - last_order).dt.days

        # cat_median_days จาก Order Gap ภายในกลุ่ม Category
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

    # ── 3.11 delay_days (สำหรับแสดงผลใน Dashboard เท่านั้น) ────────────
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
# 4. LOAD MODEL
# ==========================================
@st.cache_resource
def load_models():
    d = os.path.dirname(os.path.abspath(__file__))
    try:
        model    = joblib.load(os.path.join(d, 'olist_churn_model_v4.pkl'))
        features = joblib.load(os.path.join(d, 'model_features_v4.pkl'))
        return model, features, None
    except Exception as e:
        return None, None, str(e)

# ==========================================
# 5. PREDICT
# ==========================================
def predict_churn(df, model, feature_names, threshold):
    """สร้าง X จาก feature_names แล้ว predict"""
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
best_threshold = 0.6

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

    # อัปเดต cat_churn_risk หลัง predict แล้ว
    if 'product_category_name' in df.columns:
        cat_risk_map     = df.groupby('product_category_name')['churn_probability'].mean()
        df['cat_churn_risk'] = df['product_category_name'].map(cat_risk_map)
else:
    df['churn_probability'] = 0.5
    df['churn_prediction']  = 1

# is_churn (label สำหรับ Display)
df['is_churn'] = df['churn_prediction']

# Status (Rule-based + AI ผสม)
def get_status(row):
    prob = row.get('churn_probability', 0)
    late = row.get('lateness_score', 0)
    if late > 3.0:  return 'Lost (Late > 3x)'
    if prob > 0.75: return 'High Risk'
    if late > 1.5:  return 'Warning (Late > 1.5x)'
    if prob > 0.5:  return 'Medium Risk'
    return 'Active'

df['status'] = df.apply(get_status, axis=1)

# ==========================================
# 8. NAVIGATION
# ==========================================
st.sidebar.title("✈️ Olist Cockpit")
st.sidebar.success(f"✅ โหลดข้อมูลแล้ว ({len(df):,} rows)")
st.sidebar.info(f"🎯 Model Threshold: {best_threshold:.2f}")
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
# PAGE 1: Business Overview (UPDATED VERSION)
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

    # ── Section 1: KPI หลัก ───────────────────────────────────────────────
    total_rev   = df_d['payment_value'].sum() if 'payment_value' in df_d.columns else 0
    avg_order   = df_d['payment_value'].mean() if 'payment_value' in df_d.columns else 0
    n_customers = df_d['customer_unique_id'].nunique() if 'customer_unique_id' in df_d.columns else 0
    clv         = avg_order * df_d.groupby('customer_unique_id').size().mean() \
                  if 'customer_unique_id' in df_d.columns and n_customers > 0 else avg_order

    # คำนวณ MoM Growth แบบ Robust
    mom_growth = None
    if 'order_purchase_timestamp' in df_d.columns and not df_d.empty:
        df_d['_month'] = df_d['order_purchase_timestamp'].dt.to_period('M')
        all_months = pd.period_range(start=df_d['_month'].min(), end=df_d['_month'].max(), freq='M')
        monthly_rev_series = df_d.groupby('_month')['payment_value'].sum().reindex(all_months, fill_value=0)

        if len(monthly_rev_series) >= 3:
            last_complete_m = monthly_rev_series.iloc[-2]  # เดือนที่จบไปแล้ว
            prev_m          = monthly_rev_series.iloc[-3]  # เดือนก่อนหน้า
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

    # ── Section 2: Monthly Revenue Trend ──────────────────────────────────
    st.subheader("📈 Monthly Revenue Trend")
    if 'order_purchase_timestamp' in df_d.columns and not df_d.empty:
        # เตรียมข้อมูลรายเดือนแบบ Resample เพื่อให้แกนเวลาต่อเนื่อง
        rev_trend = df_d.set_index('order_purchase_timestamp')['payment_value'].resample('MS').sum().fillna(0).reset_index()
        rev_trend.columns = ['Month', 'Revenue']
        
        # คำนวณ Growth % สำหรับทำเส้น Line Chart
        rev_trend['Growth'] = rev_trend['Revenue'].pct_change().replace([np.inf, -np.inf], np.nan) * 100
        
        # ตัดเดือนสุดท้ายออกถ้าเป็นข้อมูลที่ยังไม่จบเดือน (ป้องกันกราฟดิ่งลงเหว)
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

    st.markdown("---")

    # ── Section 3: Top / Bottom Category ─────────────────────────────────
    st.subheader("🏆 Top 5 / Bottom 5 Category by Revenue")
    if 'product_category_name' in df_d.columns:
        cat_rev = df_d.groupby('product_category_name').agg(
            Revenue=('payment_value', 'sum'),
            Orders=('customer_unique_id', 'count'),
            Churn_Risk=('churn_probability', 'mean')
        ).reset_index().sort_values('Revenue', ascending=False)

        top5 = cat_rev.head(5).copy()
        bot5 = cat_rev.tail(5).copy()

        ct1, ct2 = st.columns(2)
        with ct1:
            st.markdown("**🟢 Top 5 — ทำเงินสูงสุด**")
            top5_chart = alt.Chart(top5).mark_bar(color='#43A047').encode(
                x=alt.X('Revenue:Q', title='Revenue (R$)'),
                y=alt.Y('product_category_name:N', sort='-x', title=None),
                tooltip=['product_category_name', alt.Tooltip('Revenue', format=',.0f')]
            ).properties(height=220)
            st.altair_chart(top5_chart, use_container_width=True)
            st.dataframe(top5.style.format({'Revenue': 'R${:,.0f}', 'Churn_Risk': '{:.1%}'}), hide_index=True)

        with ct2:
            st.markdown("**🔴 Bottom 5 — ทำเงินน้อยสุด**")
            bot5_chart = alt.Chart(bot5).mark_bar(color='#E53935').encode(
                x=alt.X('Revenue:Q', title='Revenue (R$)'),
                y=alt.Y('product_category_name:N', sort='-x', title=None),
                tooltip=['product_category_name', alt.Tooltip('Revenue', format=',.0f')]
            ).properties(height=220)
            st.altair_chart(bot5_chart, use_container_width=True)
            st.dataframe(bot5.style.format({'Revenue': 'R${:,.0f}', 'Churn_Risk': '{:.1%}'}), hide_index=True)

    st.info("💡 Tip: เน้นจัดการหมวดที่ Revenue สูงแต่ Churn Risk สูงก่อนเพื่อรักษา Market Share")

# ==========================================
# PAGE 2: Churn Overview (เดิมคือ Page 1)
# ==========================================
elif page == "2. 📊 Churn Overview":
    st.title("📊 Churn Overview")

    with st.expander("ℹ️ วิธีแบ่งกลุ่มลูกค้า"):
        st.markdown("""
| สถานะ | เงื่อนไข |
|---|---|
| 🔴 Lost | Lateness > 3.0 (หายนานเกิน 3 เท่าของรอบปกติ) |
| 🟥 High Risk | AI Predict > 75% |
| 🟧 Warning | Lateness > 1.5 |
| 🟨 Medium Risk | AI Predict 50–75% |
| 🟩 Active | มาตามรอบปกติ + AI เสี่ยงต่ำ |
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
# PAGE 3: Action Plan
# ==========================================
elif page == "3. 🎯 Action Plan":
    st.title("🎯 Action Plan & Simulator")

    all_cats = sorted(df['product_category_name'].dropna().unique()) \
               if 'product_category_name' in df.columns else []
    sel_cats = st.multiselect("หมวดสินค้า (ว่าง = ทั้งหมด):", all_cats, key="p3_cat")
    df_p3    = df[df['product_category_name'].isin(sel_cats)].copy() if sel_cats else df.copy()
    avg_ltv  = df_p3['payment_value'].mean() if 'payment_value' in df_p3.columns else 150

    st.markdown("---")
    st.subheader("💰 Smart Budget Allocator")
    total_budget  = st.number_input("งบประมาณรวม (R$):", min_value=1000, value=50000, step=5000)
    alloc_box     = st.container()

    st.markdown("---")
    st.subheader("🎯 แผงควบคุมแคมเปญ")
    tab1,tab2,tab3,tab4 = st.tabs([
        "🚚 ค่าส่งแพง","🐌 ส่งช้า","😡 รีวิวแย่","💳 ซื้อแพงจ่ายเต็ม"])
    campaigns = []

    def sim_campaign(df_sim_in, col_override, val, model, feature_names, threshold):
        """จำลองแก้ค่า Feature โดยการหักลบส่วนลด (Discount) แล้ว Predict ใหม่"""
        old_prob = df_sim_in['churn_probability'].mean()
        if model is None: return 0.30
    
        sim = df_sim_in.copy()
    
        if col_override in sim.columns:
        # --- กรณีค่าส่ง: ลบออกตามงบต่อหัวที่ตั้งไว้ ---
            if col_override == 'freight_value':
                # ค่าส่งใหม่ = ค่าส่งเดิม - งบส่วนลด (val)
                sim['freight_value'] = sim['freight_value'] - val
            # ห้ามค่าส่งติดลบ (ต่ำสุดคือ 0)
            sim['freight_value'] = sim['freight_value'].clip(lower=0)
            
            # สำคัญมาก: ต้องคำนวณ freight_ratio ใหม่ด้วย เพราะโมเดลใช้ฟีเจอร์นี้
                if 'price' in sim.columns:
                    sim['freight_ratio'] = sim['freight_value'] / sim['price']
        
        # --- กรณีรีวิว: ยังคงใช้การเซตเป็น 5 ดาว (หรือตามที่ระบุ) ---
        elif col_override == 'review_score':
            sim['review_score'] = val
            sim['is_low_score'] = (sim['review_score'].fillna(3) <= 2).astype(int)
            sim['is_high_score'] = (sim['review_score'].fillna(3) == 5).astype(int)
        
        # --- กรณีอื่นๆ: เช่น งวดผ่อน ให้เซตเป็นค่าใหม่ตามปกติ ---
        else:
            sim[col_override] = val

    # ทำนายผลด้วย AI บนข้อมูลที่จำลองส่วนลดแล้ว
    new_proba, _ = predict_churn(sim, model, feature_names, threshold)
    
    # คืนค่า Uplift (ส่วนต่างความน่าจะเป็นที่ลดลง)
    return max(old_prob - new_proba.mean(), 0.01)

    # TAB 1: ค่าส่งแพง
    # TAB 1: ค่าส่งแพง
    with tab1:
        st.markdown("#### 🚚 แคมเปญส่งฟรี")
        # ... (โค้ดส่วนอื่นคงเดิม) ...
        if len(tgt) > 0:
            c1,c2 = st.columns(2)
            with c1: cost_h = st.number_input("งบต่อหัว (R$):", value=20, key="c1")
            with c2: mode = st.radio("ประเมินโอกาส:", ["🤖 AI","✍️ Manual"], key="m1")
            
            if "AI" in mode:
                # แก้ไขตรงนี้: จาก 0 เป็น cost_h
                sr = sim_campaign(tgt, 'freight_value', cost_h, model, feature_names, best_threshold)
                st.success(f"🤖 Uplift ≈ **{sr*100:.1f}%**")
            else:
                sr = st.slider("โอกาส (%):", 1,100,30,key="s1")/100
            
            # (ส่วนที่เหลือคงเดิม)

    # TAB 2: ส่งช้า
    with tab2:
        st.markdown("#### 🐌 แคมเปญง้อส่งช้า")
        tgt = df_p3[df_p3['delay_days'] > 0] if 'delay_days' in df_p3.columns else pd.DataFrame()
        st.info(f"👥 ลูกค้าที่ได้รับของช้ากว่ากำหนด: **{len(tgt):,} คน**")
        if len(tgt) > 0:
            c1,c2 = st.columns(2)
            with c1: cost_h = st.number_input("งบต่อหัว (R$):", value=15, key="c2")
            with c2: mode   = st.radio("ประเมินโอกาส:", ["🤖 AI","✍️ Manual"], key="m2")
            if "AI" in mode:
                sr = sim_campaign(tgt,'delivery_vs_estimated',5, model,feature_names,best_threshold)
                st.success(f"🤖 Uplift ≈ **{sr*100:.1f}%**")
            else:
                sr = st.slider("โอกาส (%):", 1,100,35,key="s2")/100
            campaigns.append({"name":"🐌 ง้อส่งช้า","people":len(tgt),
                               "cost_head":cost_h,"success_rate":sr,"ltv":avg_ltv})

    # TAB 3: รีวิวแย่
    with tab3:
        st.markdown("#### 😡 แคมเปญง้อรีวิวแย่")
        tgt = df_p3[df_p3['review_score'] <= 2] if 'review_score' in df_p3.columns else pd.DataFrame()
        st.info(f"👥 ลูกค้าให้คะแนน 1-2 ดาว: **{len(tgt):,} คน**")
        if len(tgt) > 0:
            c1,c2 = st.columns(2)
            with c1: cost_h = st.number_input("งบต่อหัว (R$):", value=30, key="c3")
            with c2: mode   = st.radio("ประเมินโอกาส:", ["🤖 AI","✍️ Manual"], key="m3")
            if "AI" in mode:
                sr = sim_campaign(tgt,'review_score',5, model,feature_names,best_threshold)
                st.success(f"🤖 Uplift ≈ **{sr*100:.1f}%**")
            else:
                sr = st.slider("โอกาส (%):", 1,100,25,key="s3")/100
            campaigns.append({"name":"😡 ง้อรีวิวแย่","people":len(tgt),
                               "cost_head":cost_h,"success_rate":sr,"ltv":avg_ltv})

    # TAB 4: ซื้อแพงจ่ายเต็ม
    with tab4:
        st.markdown("#### 💳 แคมเปญดันยอดผ่อน")
        if 'price' in df_p3.columns and 'payment_installments' in df_p3.columns:
            tgt = df_p3[(df_p3['price'] > 500) & (df_p3['payment_installments'] == 1)]
        else:
            tgt = pd.DataFrame()
        st.info(f"👥 ลูกค้าซื้อแพง (> R$500) จ่ายงวดเดียว: **{len(tgt):,} คน**")
        if len(tgt) > 0:
            c1,c2 = st.columns(2)
            with c1: cost_h = st.number_input("งบต่อหัว (R$):", value=10, key="c4")
            with c2: mode   = st.radio("ประเมินโอกาส:", ["🤖 AI","✍️ Manual"], key="m4")
            if "AI" in mode:
                sr = sim_campaign(tgt,'payment_installments',10, model,feature_names,best_threshold)
                st.success(f"🤖 Uplift ≈ **{sr*100:.1f}%**")
            else:
                sr = st.slider("โอกาส (%):", 1,100,20,key="s4")/100
            campaigns.append({"name":"💳 ดันยอดผ่อน","people":len(tgt),
                               "cost_head":cost_h,"success_rate":sr,"ltv":avg_ltv})

    # Allocator
    with alloc_box:
        if campaigns:
            dc = pd.DataFrame(campaigns)
            dc['Total_Cost']      = dc['people'] * dc['cost_head']
            dc['Expected_Revenue']= dc['people'] * dc['success_rate'] * dc['ltv']
            dc['ROI_Percent']     = ((dc['Expected_Revenue'] - dc['Total_Cost'])
                                     / dc['Total_Cost'].replace(0,1) * 100)
            dc = dc.sort_values('ROI_Percent', ascending=False).reset_index(drop=True)

            alloc, total_rev, total_saved = 0, 0, 0
            rows = []
            for _, r in dc.iterrows():
                left = total_budget - alloc
                if left <= 0:
                    rows.append({"แคมเปญ":r['name'],"สถานะ":"❌ งบหมด",
                                 "คน":0,"งบ(R$)":0,"กำไร(R$)":0,"ROI%":r['ROI_Percent']})
                    continue
                spend   = r['Total_Cost'] if left >= r['Total_Cost'] else left
                covered = int(spend / r['cost_head']) if r['cost_head'] > 0 else 0
                alloc  += spend
                rev     = covered * r['success_rate'] * r['ltv']
                total_rev   += rev
                total_saved += covered * r['success_rate']
                status  = "✅ เต็ม" if spend == r['Total_Cost'] else "⚠️ บางส่วน"
                rows.append({"แคมเปญ":r['name'],"สถานะ":status,
                             "คน":covered,"งบ(R$)":spend,
                             "กำไร(R$)":rev-spend,"ROI%":r['ROI_Percent']})

            m1,m2,m3 = st.columns(3)
            m1.metric("💰 งบที่จัดสรร",  f"R$ {alloc:,.0f}",   f"จาก R$ {total_budget:,.0f}")
            m2.metric("👥 คาดดึงกลับ",   f"{int(total_saved):,} คน")
            profit = total_rev - alloc
            m3.metric("✨ กำไรคาดหวัง",  f"R$ {profit:,.0f}",
                      f"+{profit/alloc*100:.1f}% ROI" if alloc > 0 else "0%")
            st.dataframe(pd.DataFrame(rows).style.format({
                "คน":"{:,.0f}","งบ(R$)":"{:,.0f}",
                "กำไร(R$)":"{:,.0f}","ROI%":"{:.1f}%"}),
                use_container_width=True)

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
